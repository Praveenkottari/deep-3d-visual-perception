#!/usr/bin/env python3
import os, sys, rclpy, cv2, torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2

from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import draw_predictions
from heads.SFA3D.sfa.data_process.kitti_data_utils import get_filtered_lidar
from heads.SFA3D.sfa.data_process.kitti_bev_utils import makeBEVMap
import heads.SFA3D.sfa.config.kitti_config as cnf
from heads.SFA3D.sfa.utils.evaluation_utils import convert_det_to_real_values
from heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from heads.SFA3D.sfa.data_process.kitti_data_utils import Calibration


from pkgs.fusion_utils import draw_velo_on_rgbimage
from pkgs.cam_to_cam import cam_transformation
from pkgs.lid_to_cam import lid_transformation

bridge = CvBridge()

CAM_CALIB_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_cam_to_cam.txt"
LID_CALIB_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_velo_to_cam.txt"

class Detect3DOverlayBEVNode(Node):
    def __init__(self):
        super().__init__('detect3d_overlay_bev_node')

        # Load model
        self.cfg = parse_demo_configs()
        self.cfg.no_cuda = True
        self.cfg.device = 'cpu'
        self.model = create_model(self.cfg)
        self.model.load_state_dict(torch.load(self.cfg.pretrained_path, map_location='cpu'))
        self.model.eval()
        self.get_logger().info(f"Loaded weights: {self.cfg.pretrained_path}")



        # Calibration
        P2, R0_ext, T_ref0_ref2 = cam_transformation(CAM_CALIB_FILE)
        T_velo_ref0 = lid_transformation(LID_CALIB_FILE)
        self.T_velo_cam2 = P2 @ R0_ext @ T_ref0_ref2 @ T_velo_ref0
        
        # Get calibration matrices for projection
        self.calib = Calibration(self.cfg.calib_path)


        # Publishers
        self.pub_rgb_overlay = self.create_publisher(Image, '/detect3d/rgb_overlay', 5)
        self.pub_bev_raw = self.create_publisher(Image, '/detect3d/bev_image', 5)
        self.pub_bev_det = self.create_publisher(Image, '/detect3d/bev_detections', 5)
        self.pub_rgb_boxes = self.create_publisher(Image, '/detect3d/rgb_boxes', 5)


        # Subscribers
        img_sub = Subscriber(self, Image, '/kitti/image/color/left')
        pcl_sub = Subscriber(self, PointCloud2, '/kitti/point_cloud')
        self.sync = ApproximateTimeSynchronizer([img_sub, pcl_sub], 30, 0.1)
        self.sync.registerCallback(self.cb)

    def cloud_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        raw = np.frombuffer(msg.data, dtype=np.float32)
        pts = raw.reshape(-1, int(msg.point_step / 4))[:, :4]
        return pts.copy()


    
    def cb(self, img_msg, pcl_msg):
        self.get_logger().info("Received image and pointcloud")

        rgb = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        pts_lidar = self.cloud_to_numpy(pcl_msg)

        rgb_overlay = draw_velo_on_rgbimage(
            pts_lidar.T, self.T_velo_cam2, rgb.copy(),
            remove_plane=False, draw_lidar=True
        )
        self.pub_rgb_overlay.publish(bridge.cv2_to_imgmsg(rgb_overlay, 'bgr8', img_msg.header))

 
        # Preprocess LiDAR
        front_lidar = get_filtered_lidar(pts_lidar, cnf.boundary)
        back_lidar = get_filtered_lidar(pts_lidar, cnf.boundary_back)

        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)

        front_bevmap = torch.from_numpy(front_bevmap).float()
        back_bevmap = torch.from_numpy(back_bevmap).float()

        print("Front tensor shape:", front_bevmap.shape)
        print("Back tensor shape:", back_bevmap.shape)
        # Inference
        with torch.no_grad():
            front_detections, front_bevmap, _ = do_detect(self.cfg, self.model, front_bevmap, is_front=True)
            back_detections, back_bevmap, _ = do_detect(self.cfg, self.model, back_bevmap, is_front=False)

        # Convert tensors to images
        front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))

        # Draw predictions BEFORE rotating
        front_bevmap = draw_predictions(front_bevmap, front_detections, self.cfg.num_classes)
        back_bevmap = draw_predictions(back_bevmap, back_detections, self.cfg.num_classes)

        # Rotate for display (same as main.py)

        front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)

        # Concatenate: BACK on left, FRONT on right
        full_bev_det = np.concatenate((back_bevmap, front_bevmap), axis=1)

        self.pub_bev_det.publish(bridge.cv2_to_imgmsg(full_bev_det, 'bgr8', img_msg.header))


        # Convert detections to real-world dimensions
        if front_detections is not None and len(front_detections) > 0:
            front_real = convert_det_to_real_values(front_detections)
            if isinstance(front_real, torch.Tensor):
                front_real = front_real.cpu().numpy()

            if front_real.size > 0:
                front_cam = front_real.copy()
                front_cam[:, 1:] = lidar_to_camera_box(
                    front_cam[:, 1:], self.calib.V2C, self.calib.R0, self.calib.P2)

                rgb_with_boxes = show_rgb_image_with_boxes(rgb.copy(), front_cam, self.calib)

                self.pub_rgb_boxes.publish(bridge.cv2_to_imgmsg(rgb_with_boxes, 'bgr8', img_msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = Detect3DOverlayBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
