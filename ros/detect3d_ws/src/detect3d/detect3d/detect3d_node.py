#!/usr/bin/env python3
import os, sys, rclpy, cv2, torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import draw_predictions
from heads.SFA3D.sfa.data_process.kitti_data_utils import get_filtered_lidar
from heads.SFA3D.sfa.data_process.kitti_bev_utils import makeBEVMap
import heads.SFA3D.sfa.config.kitti_config as cnf

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

        # Publishers
        self.pub_rgb_overlay = self.create_publisher(Image, '/detect3d/rgb_overlay', 5)
        self.pub_bev_raw = self.create_publisher(Image, '/detect3d/bev_image', 5)
        self.pub_bev_det = self.create_publisher(Image, '/detect3d/bev_detections', 5)

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

        front_tensor = torch.from_numpy(front_bevmap).float()
        back_tensor = torch.from_numpy(back_bevmap).float()

        print("Front tensor shape:", front_tensor.shape)
        print("Back tensor shape:", back_tensor.shape)
        # Inference
        with torch.no_grad():
            front_detections, _, _ = do_detect(self.cfg, self.model, front_tensor, is_front=True)
            back_detections, _, _ = do_detect(self.cfg, self.model, back_tensor, is_front=False)

        # Convert tensors to images
        front_img = (front_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        back_img = (back_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        front_img = cv2.resize(front_img, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        back_img = cv2.resize(back_img, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))

        # Draw predictions BEFORE rotating
        front_img_det = draw_predictions(front_img.copy(), front_detections, self.cfg.num_classes)
        back_img_det = draw_predictions(back_img.copy(), back_detections, self.cfg.num_classes)

        # Rotate for display (same as main.py)
        front_img = cv2.rotate(front_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back_img = cv2.rotate(back_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        front_img_det = cv2.rotate(front_img_det, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back_img_det = cv2.rotate(back_img_det, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Concatenate: BACK on left, FRONT on right
        full_bev_raw = np.concatenate((back_img, front_img), axis=1)
        full_bev_det = np.concatenate((back_img_det, front_img_det), axis=1)

        self.pub_bev_raw.publish(bridge.cv2_to_imgmsg(full_bev_raw, 'bgr8', img_msg.header))
        self.pub_bev_det.publish(bridge.cv2_to_imgmsg(full_bev_det, 'bgr8', img_msg.header))

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
