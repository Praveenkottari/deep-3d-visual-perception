#!/usr/bin/env python3
"""
ROS2 Node for KITTI-like detection with fully applied calibration
(using user-provided calibration files for camera->camera, velo->cam, imu->velo).
It reproduces the logic in "main.py" with:
 - Subscribing to /kitti/image/color/left
 - Subscribing to /kitti/point_cloud
 - Subscribing to /kitti/imu, /kitti/nav_sat_fix
 - Loading & applying calibration transforms
 - Running YOLOv8 for detection
 - Projecting LiDAR -> camera for depth
 - Building a Folium map with object GPS markers
 - Publishing images & detection arrays, saving an HTML map

Topics Published:
  1) /kitti_detection/detected_image
  2) /kitti_detection/lidar_overlay
  3) /kitti_detection/detections
  4) (Optional) scenario top-down view, if desired.

Additionally, saves drive_map.html with the IMU-based trajectory & detections.
"""

import rclpy
from rclpy.node import Node

# ROS msgs
from sensor_msgs.msg import Image, PointCloud2, Imu
from sensor_msgs.msg import NavSatFix
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import numpy as np
import cv2
import torch
import struct
import math
import folium
import pymap3d as pm
from PIL import Image as PILImage
from ultralytics import YOLO
from sklearn import linear_model

# ------------------------------
# Utility Functions (merged from kitti_utils.py, kitti_detection_utils.py)
# ------------------------------

def get_rigid_transformation(calib_path):
    """Reads a KITTI-style [R, T] calibration text file => 4×4 transform."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    # lines[1] => R
    # lines[2] => T
    R = np.array([float(x) for x in lines[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in lines[2].strip().split(' ')[1:]]).reshape((3,1))

    T_4x4 = np.vstack((np.hstack((R, t)), np.array([0,0,0,1])))
    return T_4x4

def decompose_projection_matrix(P):
    """Decompose 3x4 projection matrix P into K, R, T (with T scaled)."""
    K, R, T, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    T = T/T[3]
    return K, R, T

def transform_uvz(uvz, T_4x4):
    """Convert Nx3 => Nx3 in some other frame by interpreting (u,v,z)->(X,Y,Z)."""
    # (u,v,z) => homogeneous => (u*z, v*z, z, 1)
    uvzw = np.column_stack((
        uvz[:,0]*uvz[:,2],  # X = u*z
        uvz[:,1]*uvz[:,2],  # Y = v*z
        uvz[:,2],
        np.ones(len(uvz))
    ))
    transformed = T_4x4 @ uvzw.T  # shape (4,N)
    return transformed[:3,:].T

def imu2geodetic(x, y, z, lat0, lon0, alt0, heading0):
    """
    Convert (x,y,z) in IMU frame to lat/lon using heading offset.
    Logic from your original main.py script.
    """
    rng = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x)) + np.degrees(heading0)
    el = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z)) + 90.0
    lla = pm.aer2geodetic(az, el, rng, lat0, lon0, alt0)
    return np.column_stack([lla[0], lla[1], lla[2]])

# For simple color mapping of LiDAR points
def color_map_depth(z):
    """A naive color mapping by z or use a colormap. We'll just do red for demonstration."""
    return (0, 0, 255)  # (B,G,R)

# ------------------------------
# The Node
# ------------------------------
class KittiCalibratedNode(Node):
    def __init__(self):
        super().__init__('kitti_calibrated_node')

        self.bridge = CvBridge()

        # --- Declare (or get) parameters for calibration paths ---
        self.declare_parameter('calib_cam_to_cam', '/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_calib/calib_cam_to_cam.txt')
        self.declare_parameter('calib_velo_to_cam', '/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_calib/calib_velo_to_cam.txt')
        self.declare_parameter('calib_imu_to_velo', '/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_calib/calib_imu_to_velo.txt')
        self.declare_parameter('save_map_html', 'drive_map.html')

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")
        self.model.conf = 0.8
        self.model.iou = 0.5
        self.model.classes = [0,1,2,3,5,7]  # person,bicycle,car,motorcycle,bus,truck

        # Setup Subscribers
        self.sub_img = self.create_subscription(Image,       '/kitti/image/color/left', self.cb_image,      10)
        self.sub_pc  = self.create_subscription(PointCloud2, '/kitti/point_cloud',       self.cb_pointcloud, 10)
        self.sub_imu = self.create_subscription(Imu,         '/kitti/imu',               self.cb_imu,        10)
        self.sub_gps = self.create_subscription(NavSatFix,   '/kitti/nav_sat_fix',       self.cb_gps,        10)

        # Setup Publishers
        self.pub_det_img  = self.create_publisher(Image, 'kitti_detection/detected_image', 10)
        self.pub_lidar_ol = self.create_publisher(Image, 'kitti_detection/lidar_overlay',   10)
        self.pub_detections= self.create_publisher(Detection2DArray, 'kitti_detection/detections', 10)
        # Optionally, we might also publish a scenario top-down. If you want, define another publisher.

        # Buffers
        self.latest_img_msg = None
        self.latest_pc_msg  = None
        self.latest_imu_msg = None
        self.latest_gps_msg = None

        # For Folium map creation
        self.have_initialized_map = False
        self.drive_map = None

        # Container to store the loaded calibrations
        self.P_rect2_cam2   = None
        self.T_velo_cam2_4x4= None
        self.T_cam2_velo_4x4= None
        self.T_imu_cam2_4x4 = None
        self.T_cam2_imu_4x4 = None

        self.load_calibrations()  # read them on start

        self.get_logger().info("KittiCalibratedNode initialized (with calibration).")

    def load_calibrations(self):
        """
        Same logic as your main.py, but reading path from parameters.
        T_velo_cam2 = P_rect2_cam2 * R_rect * T_ref0_ref2 * T_velo_ref0
        """
        calib_cam_to_cam  = self.get_parameter('calib_cam_to_cam').get_parameter_value().string_value
        calib_velo_to_cam = self.get_parameter('calib_velo_to_cam').get_parameter_value().string_value
        calib_imu_to_velo = self.get_parameter('calib_imu_to_velo').get_parameter_value().string_value

        # read lines from cam_to_cam
        with open(calib_cam_to_cam, 'r') as f:
            lines = f.readlines()

        # Typically:
        # line[25] => P_rect2_cam2
        self.P_rect2_cam2 = np.array([float(x) for x in lines[25].strip().split(' ')[1:]]).reshape((3,4))

        # line[24] => R_rect
        R_ref0_rect2 = np.array([float(x) for x in lines[24].strip().split(' ')[1:]]).reshape((3,3))
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, [0,0,0], axis=0)
        R_ref0_rect2 = np.insert(R_ref0_rect2, 3, [0,0,0,1], axis=1)

        # line[21], [22] => R_2, t_2
        R_2 = np.array([float(x) for x in lines[21].strip().split(' ')[1:]]).reshape((3,3))
        t_2 = np.array([float(x) for x in lines[22].strip().split(' ')[1:]]).reshape((3,1))
        T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, [0,0,0,1], axis=0)

        # read velo->cam
        T_velo_ref0 = get_rigid_transformation(calib_velo_to_cam)
        # read imu->velo
        T_imu_velo  = get_rigid_transformation(calib_imu_to_velo)

        # compute T_velo_cam2 (3x4):
        T_velo_cam2_3x4 = self.P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0
        # expand to 4x4:
        self.T_velo_cam2_4x4 = np.insert(T_velo_cam2_3x4, 3, [0,0,0,1], axis=0)
        self.T_cam2_velo_4x4 = np.linalg.inv(self.T_velo_cam2_4x4)

        # T_imu_cam2 => T_velo_cam2_3x4 @ T_imu_velo => shape(3x4), expand => 4x4
        T_imu_cam2_3x4 = T_velo_cam2_3x4 @ T_imu_velo
        self.T_imu_cam2_4x4 = np.insert(T_imu_cam2_3x4, 3, [0,0,0,1], axis=0)
        self.T_cam2_imu_4x4 = np.linalg.inv(self.T_imu_cam2_4x4)

        self.get_logger().info("Calibration loaded successfully.")

    def cb_image(self, msg: Image):
        self.latest_img_msg = msg
        if self.latest_pc_msg is not None:
            self.process_data()

    def cb_pointcloud(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def cb_imu(self, msg: Imu):
        """Stores latest IMU (pitch/roll/yaw, etc.). We'll read heading from orientation if needed."""
        self.latest_imu_msg = msg

    def cb_gps(self, msg: NavSatFix):
        self.latest_gps_msg = msg

    def process_data(self):
        """
        1) Convert the camera image => OpenCV
        2) YOLOv8 detection
        3) Convert pointcloud => Nx4 => project => (u,v,z)
        4) Associate each detection with LiDAR depth
        5) Publish detection image, LiDAR overlay
        6) Transform object centers => IMU frame => lat/lon
        7) Update Folium map => save to disk
        """
        # 1) Convert image
        cv_image = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')

        # 2) YOLO detection
        detections = self.model(cv_image)
        desired_classes = [0,1,2,3,5,7]
        conf_thresh = 0.5
        bboxes = []
        for box in detections[0].boxes.data.cpu().numpy():
            conf, cls_ = box[4], int(box[5])
            if conf >= conf_thresh and cls_ in desired_classes:
                # [x1, y1, x2, y2, conf, cls]
                bboxes.append(box[:6])
        bboxes_np = np.array(bboxes) if len(bboxes) > 0 else np.zeros((0,6))

        # We'll keep a copy to draw bounding boxes
        det_image = cv_image.copy()
        for bb in bboxes_np:
            x1,y1,x2,y2, cf, cc = bb
            label = f"{self.model.names[int(cc)]} {cf:.2f}"
            cv2.rectangle(det_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(det_image, label, (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 3) Convert pointcloud => Nx4 => project => (u,v,z)
        pc_array = self.pointcloud2_to_nx4(self.latest_pc_msg)  # Nx4 homogeneous
        uvz = self.project_lidar_to_camera(pc_array, cv_image.shape)

        # 4) For each bounding box => find nearest LiDAR point
        bboxes_out = self.attach_lidar_depth(det_image, bboxes_np, uvz)

        # 5a) Publish detection image
        msg_det = self.bridge.cv2_to_imgmsg(det_image, encoding='bgr8')
        msg_det.header = self.latest_img_msg.header
        self.pub_det_img.publish(msg_det)

        # 5b) Make LiDAR overlay
        lidar_overlay = self.draw_lidar_points(cv_image.copy(), uvz)
        msg_ol = self.bridge.cv2_to_imgmsg(lidar_overlay, encoding='bgr8')
        msg_ol.header = self.latest_img_msg.header
        self.pub_lidar_ol.publish(msg_ol)

        # 5c) Publish detection array
        d2d_msg = Detection2DArray()
        d2d_msg.header = self.latest_img_msg.header
        for row in bboxes_out:
            x1,y1,x2,y2, cf, cc, uu,vv,zz = row
            d2 = Detection2D()
            d2.header = self.latest_img_msg.header
            d2.bbox.center.x = float((x1+x2)/2.)
            d2.bbox.center.y = float((y1+y2)/2.)
            d2.bbox.size_x   = float(x2 - x1)
            d2.bbox.size_y   = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.id = str(int(cc))
            hyp.score = float(cf)
            # Optionally store depth in pose
            # hyp.pose.pose.position.z = float(zz)
            d2.results.append(hyp)
            d2d_msg.detections.append(d2)
        self.pub_detections.publish(d2d_msg)

        # 6) Convert detected centers => IMU frame => lat/lon => update Folium
        # Only if we have IMU & GPS
        if (self.latest_imu_msg is not None) and (self.latest_gps_msg is not None):
            # get camera->IMU transform
            # bboxes_out => Nx9 => last 3 => uvz in camera. 
            # transform uvz => IMU => get lat/lon => add to map
            uvz_centers = bboxes_out[:,-3:]  # Nx3
            xyz_imu = transform_uvz(uvz_centers, self.T_cam2_imu_4x4)

            lat0 = self.latest_gps_msg.latitude
            lon0 = self.latest_gps_msg.longitude
            alt0 = self.latest_gps_msg.altitude
            # heading from IMU's orientation? Let's assume yaw
            # For simplicity, let's say heading0=0 if no orientation->heading conversion
            heading0 = self.yaw_from_imu(self.latest_imu_msg)

            lla = imu2geodetic(xyz_imu[:,0], xyz_imu[:,1], xyz_imu[:,2],
                               lat0, lon0, alt0, heading0)

            # init folium if needed
            if not self.have_initialized_map:
                self.drive_map = folium.Map(
                    location=(lat0, lon0),
                    zoom_start=17
                )
                # draw the car's current position in red
                folium.CircleMarker(location=(lat0, lon0),
                                    radius=2,
                                    weight=5,
                                    color='red').add_to(self.drive_map)
                self.have_initialized_map = True

            # add detection markers
            for pos in lla:
                folium.CircleMarker(location=(pos[0], pos[1]),
                                    radius=2,
                                    weight=5,
                                    color='green').add_to(self.drive_map)

            # we also place the ego vehicle again in red
            folium.CircleMarker(location=(lat0, lon0),
                                radius=2,
                                weight=5,
                                color='red').add_to(self.drive_map)

            # 7) Save Folium map
            map_path = self.get_parameter('save_map_html').get_parameter_value().string_value
            self.drive_map.save(map_path)
            self.get_logger().info(f"Folium map updated at {map_path}.")

    # ------------------------------------------
    # Helper: parse PointCloud2 => Nx4 homogeneous
    # ------------------------------------------
    def pointcloud2_to_nx4(self, pc_msg: PointCloud2):
        data = pc_msg.data
        step = pc_msg.point_step
        n_points = len(data)//step
        pts = []
        for i in range(n_points):
            offset = i*step
            x = struct.unpack_from('f', data, offset+0)[0]
            y = struct.unpack_from('f', data, offset+4)[0]
            z = struct.unpack_from('f', data, offset+8)[0]
            pts.append([x, y, z, 1.0])
        return np.array(pts, dtype=np.float32)

    # ------------------------------------------
    # Helper: project LiDAR->camera => Nx3 (u,v,z)
    # ------------------------------------------
    def project_lidar_to_camera(self, pc_homog, img_shape):
        """
        pc_homog: Nx4
        T_velo_cam2_4x4: (4x4)
        => multiply => shape(4xN)
        => keep only z>0, keep 0 <= (u,v) < (w,h)
        => return Nx3 [u,v,z]
        """
        proj = self.T_velo_cam2_4x4 @ pc_homog.T  # shape(4,N)
        # remove negative depth
        valid_depth = proj[2,:] > 0
        proj = proj[:, valid_depth]

        # (u,v) => divide by z
        proj[0,:] /= proj[2,:]
        proj[1,:] /= proj[2,:]

        # out-of-bounds
        h, w, _ = img_shape
        u_ = proj[0,:]
        v_ = proj[1,:]
        z_ = proj[2,:]

        in_bounds = (u_ >= 0) & (u_ < w) & (v_ >= 0) & (v_ < h)
        proj = proj[:, in_bounds]

        uvz = proj[:3,:].T  # Nx3
        return uvz

    # ------------------------------------------
    # Helper: attach depth to YOLO bboxes
    # ------------------------------------------
    def attach_lidar_depth(self, image, bboxes, uvz):
        """
        bboxes => Nx6 => [x1,y1,x2,y2, conf, cls]
        uvz => Mx3 => [u,v,z]
        => For each bounding box, find nearest LiDAR point => store [u,v,z]
           => Nx9 => [x1,y1,x2,y2,conf,cls,u,v,z]
        Also draw depth on 'image' near the bounding box center.
        """
        if bboxes.shape[0] == 0:
            return np.zeros((0,9))

        out = np.zeros((bboxes.shape[0], 9))
        out[:,:6] = bboxes
        uvals = uvz[:,0]
        vvals = uvz[:,1]
        zvals = uvz[:,2]

        for i, bb in enumerate(bboxes):
            x1,y1,x2,y2, cf, cls_ = bb
            cx = (x1 + x2)/2.
            cy = (y1 + y2)/2.

            # naive nearest neighbor
            dist = (uvals - cx)**2 + (vvals - cy)**2
            idx = np.argmin(dist)
            out[i,6] = uvals[idx]
            out[i,7] = vvals[idx]
            out[i,8] = zvals[idx]

            depth_str = f"{zvals[idx]:.2f} m"
            cv2.putText(
                image,
                depth_str,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,0,0), 2
            )
        return out

    # ------------------------------------------
    # Helper: draw LiDAR points on image
    # ------------------------------------------
    def draw_lidar_points(self, image, uvz):
        for i in range(uvz.shape[0]):
            u_ = int(round(uvz[i,0]))
            v_ = int(round(uvz[i,1]))
            # color by z or fix color
            color = color_map_depth(uvz[i,2])
            cv2.circle(image, (u_, v_), 2, color, -1)
        return image

    # ------------------------------------------
    # Helper: parse yaw from IMU
    # ------------------------------------------
    def yaw_from_imu(self, imu_msg: Imu):
        """
        If orientation is in quaternion form, convert to euler => yaw
        """
        # quaternion
        qx = imu_msg.orientation.x
        qy = imu_msg.orientation.y
        qz = imu_msg.orientation.z
        qw = imu_msg.orientation.w
        # euler
        # yaw (z-axis rotation)
        # Following standard formula:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw


def main(args=None):
    rclpy.init(args=args)
    node = KittiCalibratedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
