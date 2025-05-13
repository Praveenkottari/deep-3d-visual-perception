#!/usr/bin/env python3
import os, sys, struct, rclpy, cv2, torch
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from .heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect
from .heads.SFA3D.sfa.models.model_utils import create_model
from .heads.SFA3D.sfa.utils.evaluation_utils import convert_det_to_real_values, draw_predictions
from .heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from .heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from .heads.SFA3D.sfa.config import kitti_config as cnf
from .pkgs.fusion_utils import draw_velo_on_rgbimage, annotate_depths_3d
from .pkgs.cam_to_cam import cam_transformation
from .pkgs.lid_to_cam import lid_transformation

bridge = CvBridge()

CAM_CALIB_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_cam_to_cam.txt"
LID_CALIB_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_velo_to_cam.txt"
WEIGHT_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/weights/fpn_resnet_18/fpn_resnet_18_epoch_300.pth"


def cloud_to_numpy(msg: PointCloud2) -> np.ndarray:
    raw = np.frombuffer(msg.data, dtype=np.float32)
    pts = raw.reshape(-1, int(msg.point_step / 4))[:, :4]
    return pts.copy()


def makeBEVmap(points, boundary, is_front=True):
    if is_front:
        mask = (points[:, 0] >= boundary["minX"]) & (points[:, 0] <= boundary["maxX"])
    else:
        mask = (points[:, 0] <= -boundary["minX"]) & (points[:, 0] >= -boundary["maxX"])
        points[:, 0] *= -1
        points[:, 1] *= -1

    points = points[mask]
    height, width = cnf.BEV_HEIGHT, cnf.BEV_WIDTH
    height_map = np.zeros((height, width))
    intensity_map = np.zeros((height, width))
    density_map = np.zeros((height, width))

    if points.shape[0] == 0:
        return np.zeros((height, width, 3), dtype=np.float32)

    discretization = (boundary["maxX"] - boundary["minX"]) / height
    x_img = np.floor((points[:, 0] - boundary["minX"]) / discretization).astype(np.int32)
    y_img = np.floor((points[:, 1] - boundary["minY"]) / discretization).astype(np.int32)
    x_img = np.clip(x_img, 0, height - 1)
    y_img = np.clip(y_img, 0, width - 1)

    points[:, 2] = np.clip(points[:, 2], boundary["minZ"], boundary["maxZ"])
    norm_heights = (points[:, 2] - boundary["minZ"]) / (boundary["maxZ"] - boundary["minZ"])

    for i in range(len(x_img)):
        xi, yi = x_img[i], y_img[i]
        height_map[xi, yi] = norm_heights[i]
        intensity_map[xi, yi] = points[i, 3]

    count_map = np.zeros((height, width))
    for xi, yi in zip(x_img, y_img):
        count_map[xi, yi] += 1
    density_map = np.log(count_map + 1) / np.log(64)
    density_map = np.clip(density_map, 0, 1)

    bev_map = np.stack([height_map, intensity_map, density_map], axis=2).astype(np.float32)
    return bev_map


class Detect3DNode(Node):
    def __init__(self):
        super().__init__('detect3d_node')

        cfg = parse_demo_configs()
        cfg.pretrained_path = WEIGHT_FILE
        cfg.no_cuda = True
        cfg.device = torch.device('cpu')
        self.model = create_model(cfg)
        self.model.load_state_dict(torch.load(cfg.pretrained_path, map_location='cpu'))
        self.model.eval()
        self.cfg = cfg
        self.get_logger().info(f"Loaded weights: {cfg.pretrained_path}")

        P2, R0_ext, T_ref0_ref2 = cam_transformation(CAM_CALIB_FILE)
        T_velo_ref0 = lid_transformation(LID_CALIB_FILE)
        self.T_velo_cam2 = P2 @ R0_ext @ T_ref0_ref2 @ T_velo_ref0
        self.V2C = (R0_ext @ T_velo_ref0)[:3, :]
        self.R0 = R0_ext[:3, :3]
        self.P2 = P2

        self.pub_rgb_overlay = self.create_publisher(Image, '/detect3d/rgb_overlay', 5)
        self.pub_rgb_boxes = self.create_publisher(Image, '/detect3d/rgb_boxes', 5)

        img_sub = Subscriber(self, Image, '/kitti/image/color/left')
        pcl_sub = Subscriber(self, PointCloud2, '/kitti/point_cloud')
        self.sync = ApproximateTimeSynchronizer([img_sub, pcl_sub], 30, 0.1)
        self.sync.registerCallback(self.cb)

    def cb(self, img_msg, pcl_msg):
        self.get_logger().info("Received image and pointcloud")
        rgb = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        pts = cloud_to_numpy(pcl_msg)

        rgb_overlay = draw_velo_on_rgbimage(pts.T, self.T_velo_cam2, rgb.copy(), remove_plane=False, draw_lidar=True)
        self.pub_rgb_overlay.publish(bridge.cv2_to_imgmsg(rgb_overlay, 'bgr8', img_msg.header))

        self.get_logger().info("Running detection model on computed BEV tensors...")
        front_np = makeBEVmap(pts.copy(), cnf.boundary, is_front=True)
        back_np = makeBEVmap(pts.copy(), cnf.boundary, is_front=False)
        front_tensor = torch.from_numpy(front_np).permute(2, 0, 1).unsqueeze(0).to(self.cfg.device)
        back_tensor = torch.from_numpy(back_np).permute(2, 0, 1).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            det_f, _, _ = do_detect(self.cfg, self.model, front_tensor[0], True)
            det_b, _, _ = do_detect(self.cfg, self.model, back_tensor[0], False)

        det_list = []
        for d in (det_f, det_b):
            if isinstance(d, dict):
                for arr in d.values():
                    if isinstance(arr, torch.Tensor):
                        arr = arr.cpu().numpy()
                    if arr.shape[1] == 8:
                        det_list.append(arr)
            elif isinstance(d, np.ndarray) and d.shape[1] == 8:
                det_list.append(d)

        if not det_list:
            self.get_logger().info("No valid detection arrays found.")
            return

        all_det = np.vstack(det_list)
        self.get_logger().info(f"Stacked detection shape: {all_det.shape}")
        if all_det.shape[0] == 0:
            self.get_logger().info("No detections to process. Skipping frame.")
            return

        real = convert_det_to_real_values(all_det)
        cam = real.copy()
        cam[:, 1:] = lidar_to_camera_box(cam[:, 1:], self.V2C, self.R0, self.P2)
        rgb_boxes = show_rgb_image_with_boxes(rgb.copy(), cam, self)
        rgb_boxes, _ = annotate_depths_3d(rgb_boxes, real, self, True)

        self.pub_rgb_boxes.publish(bridge.cv2_to_imgmsg(rgb_boxes, 'bgr8', img_msg.header))


def main(args=None):
    rclpy.init(args=args)
    node = Detect3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()