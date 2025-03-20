#!/usr/bin/env python3
"""
ROS2 Node with:
 - Ground-plane removal (RANSAC) for LiDAR
 - Bird's-Eye-View (BEV) top-down image from LiDAR
 - Publishes an "object-only" point cloud (non-ground)

Topics IN:
   /kitti/image/color/left  (Image)
   /kitti/point_cloud       (PointCloud2)

Topics OUT:
   /kitti_detection/detected_image  (Image)         - bounding boxes + depth
   /kitti_detection/lidar_overlay   (Image)         - LiDAR points on camera
   /kitti_detection/detections      (Detection2DArray)
   /kitti_detection/bev_image       (Image)         - top-down (BEV) from LiDAR
   /kitti_detection/object_points   (PointCloud2)   - LiDAR minus ground plane
"""

import rclpy
from rclpy.node import Node

# ROS messages
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import numpy as np
import cv2
import struct
import math

from ultralytics import YOLO
import torch

from .pkg.utils import *

import os
import sys
sys.path.append('/home/airl010/1_Thesis/deep-3d-visual-perception/ros/3d_detection_ws/src/vehicle_detection/vehicle_detection')
from sfa.data_process.demo_dataset import Demo_KittiDataset
from sfa.models.model_utils import create_model
from sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from sfa.data_process.transformation import lidar_to_camera_box
from sfa.utils.demo_utils import parse_demo_configs, do_detect
import sfa.config.kitti_config as cnf
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from geometry_msgs.msg import Pose, Point, Quaternion


# ------------------------------
# The Node
# ------------------------------
class KittiCalibratedNode(Node):
    def __init__(self):
        super().__init__('kitti_calibrated_node')
        self.bridge = CvBridge()

        # (1) YOLO model – reduce resolution for speed
        self.model = YOLO("/home/airl010/1_Thesis/deep-3d-visual-perception/weights/yolov8/yolov8n.pt")
        # Move to GPU if available
        if torch.cuda.is_available():
            self.get_logger().info("Using GPU for YOLO inference...")
            self.model.to('cuda')
        # Also reduce default img size
        self.model.overrides['imgsz'] = 320

        self.model.conf = 0.5
        self.model.iou = 0.5
        # Example: detect persons, cars, etc.
        # Adjust to your desired classes
        self.model.classes = [0,1,2,3,5,7]  # person,bicycle,car,motorcycle,bus,truck



        # SFA3D model
        self.sfa_configs = parse_demo_configs()
        self.sfa_configs.pretrained_path = "/home/airl010/1_Thesis/deep-3d-visual-perception/weights/fpn_resnet_18/fpn_resnet_18_epoch_300.pth"  # Confirm path
        self.sfa_model = create_model(self.sfa_configs)
        assert os.path.isfile(self.sfa_configs.pretrained_path), f"No file at {self.sfa_configs.pretrained_path}"
        self.sfa_model.load_state_dict(torch.load(self.sfa_configs.pretrained_path, map_location='cpu'))
        self.sfa_configs.device = torch.device('cuda:0' if torch.cuda.is_available() and self.sfa_configs.gpu_idx != -1 else 'cpu')
        self.sfa_model = self.sfa_model.to(self.sfa_configs.device)
        self.sfa_model.eval()
        self.get_logger().info(f"SFA3D model loaded on {self.sfa_configs.device}")


        # calibration file paths – set them to your actual paths
        self.calib_cam_to_cam  = '/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_cam_to_cam.txt'
        self.calib_velo_to_cam = '/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_velo_to_cam.txt'
        self.load_calibrations()

        # (2) Publishers
        self.pub_det_image    = self.create_publisher(Image, 'kitti_detection/detected_image', 10)
        self.pub_lidar_overlay= self.create_publisher(Image, 'kitti_detection/lidar_overlay', 10)
        self.pub_detections   = self.create_publisher(Detection2DArray, 'kitti_detection/detections', 10)
        self.pub_detections_3d = self.create_publisher(Detection3DArray, 'kitti_detection/detections_3d', 10)
        self.pub_bev_image    = self.create_publisher(Image, 'kitti_detection/bev_image', 10)
        self.pub_obj_points   = self.create_publisher(PointCloud2, 'kitti_detection/object_points', 10)

        # (3) Subscribers
        self.sub_image      = self.create_subscription(Image,       '/kitti/image/color/left', self.cb_image,      10)
        self.sub_pointcloud = self.create_subscription(PointCloud2, '/kitti/point_cloud',       self.cb_pointcloud, 10)

        # Buffers
        self.latest_img_msg = None
        self.latest_pc_msg  = None

        self.get_logger().info("KittiCalibratedNode started (Camera+LiDAR only).")

    def load_calibrations(self):
        """
        Loads your KITTI calibration info needed:
         T_velo_cam2, T_cam2_velo
        """
        # read lines from cam_to_cam
        with open(self.calib_cam_to_cam, 'r') as f:
            lines = f.readlines()

        # line[25] => P_rect2_cam2
        self.P_rect2_cam2 = np.array([float(x) for x in lines[25].strip().split(' ')[1:]]).reshape((3,4))

        # line[24] => R_ref0_rect2
        R_ref0_rect2 = np.array([float(x) for x in lines[24].strip().split(' ')[1:]]).reshape((3,3))
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, [0,0,0], axis=0)
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, [0,0,0,1], axis=1)

        # line[21],[22] => R_2, t_2
        R_2 = np.array([float(x) for x in lines[21].strip().split(' ')[1:]]).reshape((3,3))
        t_2 = np.array([float(x) for x in lines[22].strip().split(' ')[1:]]).reshape((3,1))
        T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, [0,0,0,1], axis=0)

        # read velo->cam
        T_velo_ref0 = get_rigid_transformation(self.calib_velo_to_cam)

        # T_velo_cam2_3x4 = P_rect2_cam2 * R_ref0_rect2 * T_ref0_ref2 * T_velo_ref0
        T_velo_cam2_3x4 = self.P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0
        self.T_velo_cam2_4x4 = np.insert(T_velo_cam2_3x4, 3, [0,0,0,1], axis=0)
        self.T_cam2_velo_4x4 = np.linalg.inv(self.T_velo_cam2_4x4)

        self.get_logger().info("Calibration loaded successfully.")

    def cb_image(self, msg: Image):
        self.latest_img_msg = msg
        # We only run if we have both image+pointcloud
        if self.latest_pc_msg is not None:
            self.run_pipeline()

    def cb_pointcloud(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def run_pipeline(self):
        """
        1) Convert ROS image -> CV
        2) YOLO detection
        3) PointCloud -> Nx3 => ground removal => project => uvz
        4) bounding box approximate depth
        5) publish detection image
        6) LiDAR overlay image
        7) publish object-only point cloud
        8) BEV image
        """
        # 1) Convert image
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')

        # 2) YOLO detection
        # We can do: results = self.model.predict(cv_img, imgsz=320, conf=0.5, iou=0.5)
        results = self.model(cv_img)

        desired_classes = self.model.classes
        conf_thr = 0.5
        bboxes = []
        for box in results[0].boxes.data.cpu().numpy():
            conf, cls_ = box[4], int(box[5])
            if conf >= conf_thr and cls_ in desired_classes:
                bboxes.append(box[:6])
        bboxes_np = np.array(bboxes) if len(bboxes) > 0 else np.zeros((0,6))

        detect_img = cv_img.copy()
        for bb in bboxes_np:
            x1,y1,x2,y2, cf, cc = bb
            label = f"{self.model.names[int(cc)]} {cf:.2f}"
            cv2.rectangle(detect_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(detect_img, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 3) PointCloud -> Nx3 => ground removal
        pc_xyz = self.pointcloud2_to_xyz(self.latest_pc_msg)  # Nx3
        obj_xyz = remove_ground_plane(pc_xyz, threshold=0.2, max_trials=2000)

        # project object LiDAR points onto image
        uvz = self.project_lidar_to_camera(obj_xyz, cv_img.shape)

        # 4) bounding box depth
        bboxes_out = self.attach_depth(detect_img, bboxes_np, uvz)

        # 5) publish detection image
        msg_det = self.bridge.cv2_to_imgmsg(detect_img, encoding='bgr8')
        msg_det.header = self.latest_img_msg.header
        self.pub_det_image.publish(msg_det)

        # 6) LiDAR overlay (using object points for clarity)
        overlay_img = self.draw_lidar_points(cv_img.copy(), uvz)
        msg_ol = self.bridge.cv2_to_imgmsg(overlay_img, encoding='bgr8')
        msg_ol.header = self.latest_img_msg.header
        self.pub_lidar_overlay.publish(msg_ol)

        # Publish detection array
        d2d_array = Detection2DArray()
        d2d_array.header = self.latest_img_msg.header
        for row in bboxes_out:
            x1,y1,x2,y2, cf, cc, uu, vv, zz = row
            d2 = Detection2D()
            d2.header = d2d_array.header
            d2.bbox.center.x = float((x1+x2)/2.)
            d2.bbox.center.y = float((y1+y2)/2.)
            d2.bbox.size_x   = float(x2 - x1)
            d2.bbox.size_y   = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.id = str(int(cc))
            hyp.score = float(cf)
            # We could store depth in pose if needed
            # hyp.pose.pose.position.z = float(zz)
            d2.results.append(hyp)
            d2d_array.detections.append(d2)
        self.pub_detections.publish(d2d_array)

        # 7) publish the object-only point cloud
        obj_pc_msg = create_pointcloud2(obj_xyz)
        obj_pc_msg.header = self.latest_pc_msg.header
        self.pub_obj_points.publish(obj_pc_msg)

        # 8) build BEV image from entire LiDAR (or obj_xyz, your choice)
        bev_img = create_bev_image(pc_xyz)
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = self.latest_img_msg.header
        self.pub_bev_image.publish(bev_msg)

    # ----------------------------------------------------
    # Helper: pointcloud2 => Nx3
    # ----------------------------------------------------
    def pointcloud2_to_xyz(self, pc_msg: PointCloud2):
        data = pc_msg.data
        step = pc_msg.point_step
        n_points = len(data) // step
        xyz = []
        for i in range(n_points):
            offset = i*step
            x = struct.unpack_from('f', data, offset+0)[0]
            y = struct.unpack_from('f', data, offset+4)[0]
            z = struct.unpack_from('f', data, offset+8)[0]
            xyz.append([x,y,z])
        return np.array(xyz, dtype=np.float32)

    # ----------------------------------------------------
    # Helper: project Nx3 => Nx3 (u,v,z) in camera image
    # ----------------------------------------------------
    def project_lidar_to_camera(self, xyz_points, img_shape):
        """
        Nx3 => Nx4 => T_velo_cam2_4x4 => keep z>0 => (u,v) in bounds
        """
        if xyz_points.shape[0] == 0:
            return np.zeros((0,3))

        ones = np.ones((xyz_points.shape[0],1), dtype=np.float32)
        xyz1 = np.hstack((xyz_points, ones))  # Nx4
        proj = self.T_velo_cam2_4x4 @ xyz1.T  # 4xN

        # remove negative depth
        valid = proj[2,:] > 0
        proj = proj[:, valid]

        # scale
        proj[0,:] /= proj[2,:]
        proj[1,:] /= proj[2,:]

        h, w, _ = img_shape
        u_ = proj[0,:]
        v_ = proj[1,:]
        z_ = proj[2,:]
        in_bounds = (u_>=0)&(u_<w)&(v_>=0)&(v_<h)
        proj = proj[:, in_bounds]
        uvz = proj[:3,:].T  # Nx3
        return uvz

    # ----------------------------------------------------
    # Helper: attach depth to bounding boxes
    # ----------------------------------------------------
    def attach_depth(self, image, bboxes, uvz):
        """
        bboxes Nx6 => [x1,y1,x2,y2,conf,class]
        uvz Nx3 => [u,v,z] from LiDAR projection
        => Nx9 => + [u,v,z], also draws text on image
        """
        if bboxes.shape[0] == 0:
            return np.zeros((0,9))
        out = np.zeros((bboxes.shape[0], 9))
        out[:,:6] = bboxes

        uvals = uvz[:,0]
        vvals = uvz[:,1]
        zvals = uvz[:,2]

        for i, bb in enumerate(bboxes):
            x1,y1,x2,y2, cf, cc = bb
            cx = (x1 + x2)/2
            cy = (y1 + y2)/2
            # find the LiDAR point closest to the center of the bounding box
            dist = (uvals - cx)**2 + (vvals - cy)**2
            idx = np.argmin(dist)
            depth = zvals[idx]
            out[i,6] = uvals[idx]
            out[i,7] = vvals[idx]
            out[i,8] = depth

            cv2.putText(
                image,
                f"{depth:.2f} m",
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,0,0), 2
            )
        return out

    # ----------------------------------------------------
    # Helper: draw LiDAR points
    # ----------------------------------------------------
    def draw_lidar_points(self, image, uvz):
        for i in range(uvz.shape[0]):
            u_ = int(round(uvz[i,0]))
            v_ = int(round(uvz[i,1]))
            cv2.circle(image, (u_, v_), 2, (0,0,255), -1)
        return image


def main(args=None):
    rclpy.init(args=args)
    node = KittiCalibratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
