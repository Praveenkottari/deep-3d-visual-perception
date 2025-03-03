#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print("Starting script")
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import os
import sys

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import sfa3d.config.kitti_config as cnf
from sfa3d.data_process.demo_dataset import Demo_KittiDataset
from sfa3d.models.model_utils import create_model
from sfa3d.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from sfa3d.data_process.transformation import lidar_to_camera_box
from sfa3d.utils.visualization_utils import show_rgb_image_with_boxes
from sfa3d.data_process.kitti_data_utils import Calibration
from sfa3d.utils.demo_utils import parse_demo_configs, do_detect

class SFA3DNode(Node):
    def __init__(self):
        super().__init__('3dbox_node')

        # ROS 2 initialization
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/kitti/image/color/left', self.image_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/kitti/point_cloud', self.pointcloud_callback, 10)
        self.image_pub = self.create_publisher(Image, '/sfa3d/output_image', 10)

        # SFA3D Configuration and Model Setup
        self.configs = parse_demo_configs()
        self.configs.pretrained_path = "/home/airl010/1_Thesis/deep-3d-visual-perception/weights/fpn_resnet_18/fpn_resnet_18_epoch_300.pth"
        self.configs.calib_path = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibation/calib.txt"
        self.get_logger().info(f"Loading model from: {self.configs.pretrained_path}")
        self.get_logger().info(f"Using calibration file: {self.configs.calib_path}")
        self.model = create_model(self.configs)
        
        # Load pretrained weights
        if not os.path.isfile(self.configs.pretrained_path):
            self.get_logger().error(f"Pretrained model file not found: {self.configs.pretrained_path}")
            raise FileNotFoundError(f"No file at {self.configs.pretrained_path}")
        self.model.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cpu'))
        self.configs.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.configs.device)
        self.model.eval()

        # Calibration
        try:
            self.calib = Calibration(self.configs.calib_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {str(e)}")
            raise

        # Boundary for BEV map
        self.boundary = {"minX": 0, "maxX": 50, "minY": -25, "maxY": 25, "minZ": -2, "maxZ": 3}

        # State variables
        self.latest_image = None
        self.latest_pointcloud = None

        # Periodic status check
        self.timer = self.create_timer(5.0, self.status_check)

        self.get_logger().info("3DBox ROS 2 Node initialized successfully")
        self.get_logger().info(f"Device: {self.configs.device}")
        self.get_logger().info("Subscribed to /kitti/image/color/left and /kitti/point_cloud")
        self.get_logger().info("Publishing to /sfa3d/output_image")

    def status_check(self):
        """Periodic check to ensure node is alive and report status."""
        self.get_logger().info("Node status check: Running")
        self.get_logger().info(f"Latest image received: {self.latest_image is not None}")
        self.get_logger().info(f"Latest point cloud received: {self.latest_pointcloud is not None}")

    def image_callback(self, msg):
        """Callback for image topic."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info(f"Received image with shape: {self.latest_image.shape}")
            self.process_data()
        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {str(e)}")

    def pointcloud_callback(self, msg):
        """Callback for point cloud topic."""
        try:
            pc_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
            self.latest_pointcloud = pc_data[:, :4]  # [x, y, z, intensity]
            self.get_logger().info(f"Received point cloud with {self.latest_pointcloud.shape[0]} points")
            self.get_logger().debug(f"Point cloud sample: {self.latest_pointcloud[:5]}")
            self.process_data()
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud_callback: {str(e)}")

    def process_data(self):
        """Process image and point cloud when both are available."""
        if self.latest_image is None or self.latest_pointcloud is None:
            self.get_logger().warning("Missing data: Image or PointCloud not received yet")
            return

        self.get_logger().info("Starting data processing...")
        try:
            front_bevmap, back_bevmap = self.pointcloud_to_bev(self.latest_pointcloud)

            # Perform inference
            with torch.no_grad():
                front_detections, front_bevmap, fps = do_detect(self.configs, self.model, front_bevmap, is_front=True)
                back_detections, back_bevmap, _ = do_detect(self.configs, self.model, back_bevmap, is_front=False)

            self.get_logger().info(f"Inference completed. FPS: {fps:.2f}")

            # Draw predictions on BEV maps
            front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            front_bevmap = draw_predictions(front_bevmap, front_detections, self.configs.num_classes)
            front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

            back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            back_bevmap = draw_predictions(back_bevmap, back_detections, self.configs.num_classes)
            back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)

            full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)

            # Overlay 3D boxes on image
            img_bgr = self.latest_image.copy()
            kitti_dets = convert_det_to_real_values(front_detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], self.calib.V2C, self.calib.R0, self.calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, self.calib)
            img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))

            # Combine image and BEV
            out_img = np.concatenate((img_bgr, full_bev), axis=0)
            self.get_logger().info(f"Output image shape: {out_img.shape}")

            # Publish output image
            out_msg = self.bridge.cv2_to_imgmsg(out_img, encoding='bgr8')
            self.image_pub.publish(out_msg)
            self.get_logger().info("Published output image to /sfa3d/output_image")
        except Exception as e:
            self.get_logger().error(f"Error in process_data: {str(e)}")

    def pointcloud_to_bev(self, pointcloud):
        """Convert point cloud to BEV maps."""
        try:
            mask = (
                (pointcloud[:, 0] >= self.boundary["minX"]) & (pointcloud[:, 0] <= self.boundary["maxX"]) &
                (pointcloud[:, 1] >= self.boundary["minY"]) & (pointcloud[:, 1] <= self.boundary["maxY"]) &
                (pointcloud[:, 2] >= self.boundary["minZ"]) & (pointcloud[:, 2] <= self.boundary["maxZ"])
            )
            filtered_pc = pointcloud[mask]
            self.get_logger().info(f"Filtered point cloud: {filtered_pc.shape[0]} points")

            mid_x = (self.boundary["maxX"] + self.boundary["minX"]) / 2
            front_mask = filtered_pc[:, 0] >= mid_x
            back_mask = filtered_pc[:, 0] < mid_x

            front_pc = filtered_pc[front_mask]
            back_pc = filtered_pc[back_mask]
            self.get_logger().info(f"Front points: {front_pc.shape[0]}, Back points: {back_pc.shape[0]}")

            front_rgb_map = self.makeBEVMap(front_pc, self.boundary) if len(front_pc) > 0 else np.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
            back_rgb_map = self.makeBEVMap(back_pc, self.boundary) if len(back_pc) > 0 else np.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH))

            front_bevmap = torch.tensor(front_rgb_map).float().unsqueeze(0).to(self.configs.device)
            back_bevmap = torch.tensor(back_rgb_map).float().unsqueeze(0).to(self.configs.device)
            self.get_logger().info(f"BEV maps created: Front {front_bevmap.shape}, Back {back_bevmap.shape}")

            return front_bevmap, back_bevmap
        except Exception as e:
            self.get_logger().error(f"Error in pointcloud_to_bev: {str(e)}")
            raise

    def makeBEVMap(self, PointCloud_, boundary):
        try:
            Height = cnf.BEV_HEIGHT + 1
            Width = cnf.BEV_WIDTH + 1

            PointCloud = np.copy(PointCloud_)
            PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
            PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

            sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
            PointCloud = PointCloud[sorted_indices]
            _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
            PointCloud_top = PointCloud[unique_indices]

            heightMap = np.zeros((Height, Width))
            intensityMap = np.zeros((Height, Width))
            densityMap = np.zeros((Height, Width))

            max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
            heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height
            normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
            intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
            densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

            RGB_Map = np.zeros((3, Height - 1, Width - 1))
            RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
            RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]

            return RGB_Map
        except Exception as e:
            self.get_logger().error(f"Error in makeBEVMap: {str(e)}")
            raise

def main(args=None):
    rclpy.init(args=args)
    try:
        node = SFA3DNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()