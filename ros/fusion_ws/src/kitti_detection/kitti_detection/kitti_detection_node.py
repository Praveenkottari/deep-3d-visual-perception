#!/usr/bin/env python3
"""
Updated ROS2 Node with:
 - YOLOv8 (reduced resolution for faster speed)
 - Ground-plane removal (RANSAC) for LiDAR
 - Bird's-Eye-View (BEV) top-down image
 - Publishes an "object-only" point cloud (non-ground)

Topics IN:
   /kitti/image/color/left
   /kitti/point_cloud
   /kitti/imu
   /kitti/nav_sat_fix

Topics OUT:
   /kitti_detection/detected_image     (Image)  - bounding boxes + depth
   /kitti_detection/lidar_overlay      (Image)  - LiDAR points on camera
   /kitti_detection/detections         (Detection2DArray)
   /kitti_detection/bev_image          (Image)  - top-down view
   /kitti_detection/object_points      (PointCloud2) - LiDAR minus ground plane
"""
import rclpy
from rclpy.node import Node

# ROS messages
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix, PointField
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import numpy as np
import cv2
import struct
import math
import folium
import pymap3d as pm
from PIL import Image as PILImage

from ultralytics import YOLO
import torch
from sklearn import linear_model

# for output point cloud
from sensor_msgs.msg import PointCloud2, PointField

# ---------------
# Utility functions
# ---------------

def get_rigid_transformation(calib_path):
    """Reads a KITTI-style [R, T] calibration => 4×4 transform."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    R = np.array([float(x) for x in lines[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in lines[2].strip().split(' ')[1:]]).reshape((3,1))
    T_4x4 = np.vstack((np.hstack((R, t)), np.array([0,0,0,1])))
    return T_4x4

def transform_uvz(uvz, T_4x4):
    """Convert Nx3 => Nx3 in target frame by interpreting (u,v,z)->(X,Y,Z)."""
    uvzw = np.column_stack((
        uvz[:,0]*uvz[:,2],
        uvz[:,1]*uvz[:,2],
        uvz[:,2],
        np.ones(len(uvz))
    ))
    xyz_h = T_4x4 @ uvzw.T
    return xyz_h[:3,:].T

def imu2geodetic(x, y, z, lat0, lon0, alt0, heading0):
    rng = np.sqrt(x**2 + y**2 + z**2)
    az = np.degrees(np.arctan2(y, x)) + np.degrees(heading0)
    el = np.degrees(np.arctan2(np.sqrt(x**2 + y**2), z)) + 90.0
    lla = pm.aer2geodetic(az, el, rng, lat0, lon0, alt0)
    return np.column_stack([lla[0], lla[1], lla[2]])

def remove_ground_plane(xyz_points, threshold=0.1, max_trials=5000):
    """
    RANSAC-based ground removal
    Input: Nx3 (x,y,z)
    Output: Nx3 (just the points not on ground)
    """
    if xyz_points.shape[0] < 10:
        return xyz_points
    ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression(),
        residual_threshold=threshold,
        max_trials=max_trials
    )
    X = xyz_points[:, :2]  # (x, y)
    y = xyz_points[:, 2]   # z
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_  # ground
    # We want the outliers (non-ground)
    obj_points = xyz_points[~inlier_mask]
    return obj_points

def create_pointcloud2(points_xyz):
    """
    Convert Nx3 array -> PointCloud2 (ROS) with fields x,y,z float32
    """
    msg = PointCloud2()
    msg.header.frame_id = "base_link"  # or camera_link, etc.
    msg.height = 1
    msg.width  = points_xyz.shape[0]
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 12  # 3 x float32
    msg.row_step     = msg.point_step * points_xyz.shape[0]
    msg.is_dense     = True

    # flatten Nx3 into bytes
    data = []
    for p in points_xyz:
        data.append(struct.pack('fff', p[0], p[1], p[2]))
    msg.data = b"".join(data)
    return msg

def draw_scenario(canvas, imu_xyz, canvas_height, scale=12):
    """
    Simple top-down “Bird’s-Eye-View”:
    - Ego is green rectangle at bottom center
    - Objects in red
    - scale adjusts how quickly objects move away from ego in the top-down
    - We interpret (x=forward, y=left)
    """
    ego_center = (int(canvas.shape[1]/2), int(canvas_height*0.9))
    # Draw ego
    ex1 = ego_center[0]-5
    ey1 = ego_center[1]-10
    ex2 = ego_center[0]+5
    ey2 = ego_center[1]+10
    cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (0,255,0), -1)

    # Draw objects
    for xyz in imu_xyz:
        # xyz in IMU frame => x forward, y left
        # convert to “canvas coords”
        # x => negative Y in image, y => X in image
        dx = -int(scale*xyz[1])
        dy = -int(scale*xyz[0])
        obj_cx = ego_center[0] + dx
        obj_cy = ego_center[1] + dy
        cv2.rectangle(canvas,
                      (obj_cx-4, obj_cy-4),
                      (obj_cx+4, obj_cy+4),
                      (0,0,255), -1)
    return canvas

# ------------------------------
# The Node
# ------------------------------
class KittiCalibratedNode(Node):
    def __init__(self):
        super().__init__('kitti_calibrated_node')

        self.bridge = CvBridge()

        # (1) YOLO model – reduce resolution for speed
        self.model = YOLO("yolov8n.pt")
        # Move to GPU if available:
        if torch.cuda.is_available():
            self.get_logger().info("Using GPU for YOLO inference...")
            self.model.to('cuda')
        # Also reduce default img size, or pass in .predict(..., imgsz=320):
        self.model.overrides['imgsz'] = 320

        self.model.conf = 0.8
        self.model.iou = 0.5
        self.model.classes = [0,1,2,3,5,7]  # person,bicycle,car,motorcycle,bus,truck

        # calibration file paths – set them to your actual paths
        self.calib_cam_to_cam  = '/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_calib/calib_cam_to_cam.txt'
        self.calib_velo_to_cam = '/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_calib/calib_velo_to_cam.txt'
        self.calib_imu_to_velo = '/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_calib/calib_imu_to_velo.txt'
        self.load_calibrations()

        # (2) Publishers
        self.pub_det_image    = self.create_publisher(Image, 'kitti_detection/detected_image', 10)
        self.pub_lidar_overlay= self.create_publisher(Image, 'kitti_detection/lidar_overlay', 10)
        self.pub_detections   = self.create_publisher(Detection2DArray, 'kitti_detection/detections', 10)
        self.pub_bev_image    = self.create_publisher(Image, 'kitti_detection/bev_image', 10)
        self.pub_obj_points   = self.create_publisher(PointCloud2, 'kitti_detection/object_points', 10)

        # (3) Subscribers
        self.sub_image      = self.create_subscription(Image,       '/kitti/image/color/left', self.cb_image,      10)
        self.sub_pointcloud = self.create_subscription(PointCloud2, '/kitti/point_cloud',       self.cb_pointcloud, 10)
        self.sub_imu        = self.create_subscription(Imu,         '/kitti/imu',               self.cb_imu,        10)
        self.sub_gps        = self.create_subscription(NavSatFix,   '/kitti/nav_sat_fix',       self.cb_gps,        10)

        # Buffers
        self.latest_img_msg = None
        self.latest_pc_msg  = None
        self.latest_imu_msg = None
        self.latest_gps_msg = None

        # For Folium map
        self.drive_map = None
        self.map_initialized = False

        self.get_logger().info("KittiCalibratedNode with BEV + ground removal started.")

    def load_calibrations(self):
        """
        Loads your KITTI calibration info:
         T_velo_cam2, T_cam2_velo, T_imu_cam2, ...
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

        # read velo->cam, imu->velo
        T_velo_ref0 = get_rigid_transformation(self.calib_velo_to_cam)
        T_imu_velo  = get_rigid_transformation(self.calib_imu_to_velo)

        # T_velo_cam2 => P_rect2_cam2 * R_ref0_rect2 * T_ref0_ref2 * T_velo_ref0
        T_velo_cam2_3x4 = self.P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0
        self.T_velo_cam2_4x4 = np.insert(T_velo_cam2_3x4, 3, [0,0,0,1], axis=0)
        self.T_cam2_velo_4x4 = np.linalg.inv(self.T_velo_cam2_4x4)

        # T_imu_cam2 => T_velo_cam2_3x4 @ T_imu_velo
        T_imu_cam2_3x4 = T_velo_cam2_3x4 @ T_imu_velo
        self.T_imu_cam2_4x4 = np.insert(T_imu_cam2_3x4, 3, [0,0,0,1], axis=0)
        self.T_cam2_imu_4x4 = np.linalg.inv(self.T_imu_cam2_4x4)

        self.get_logger().info("Calibration loaded successfully.")

    def cb_image(self, msg: Image):
        self.latest_img_msg = msg
        if self.latest_pc_msg is not None:
            self.run_pipeline()

    def cb_pointcloud(self, msg: PointCloud2):
        self.latest_pc_msg = msg

    def cb_imu(self, msg: Imu):
        self.latest_imu_msg = msg

    def cb_gps(self, msg: NavSatFix):
        self.latest_gps_msg = msg

    def run_pipeline(self):
        """
        1) Convert ROS image -> CV
        2) YOLO detection (with possibly reduced resolution)
        3) pointcloud -> Nx4 -> ground removal -> project -> uvz
        4) bounding box depth
        5) publish images + detection array
        6) build BEV image
        7) optionally place detections on Folium map => save
        """
        # 1) Convert image
        cv_img = self.bridge.imgmsg_to_cv2(self.latest_img_msg, desired_encoding='bgr8')

        # 2) YOLO detection (faster if we do .predict(img, imgsz=320, device='cuda' if available,...)
        # we can do: results = self.model.predict(cv_img, imgsz=320)
        results = self.model(cv_img)

        desired_classes = [0,1,2,3,5,7]
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

        # 3) Convert pointcloud => Nx3, remove ground
        pc_xyz = self.pointcloud2_to_xyz(self.latest_pc_msg)  # Nx3
        obj_xyz = remove_ground_plane(pc_xyz, threshold=0.2, max_trials=2000)

        # Publish object-only point cloud
        obj_pc_msg = create_pointcloud2(obj_xyz)
        obj_pc_msg.header = self.latest_pc_msg.header
        self.pub_obj_points.publish(obj_pc_msg)

        # project entire LiDAR or just object points onto image?
        # We'll project obj_xyz for a cleaner overlay
        uvz = self.project_lidar_to_camera(obj_xyz, cv_img.shape)

        # 4) bounding box depth
        bboxes_out = self.attach_depth(detect_img, bboxes_np, uvz)

        # 5) Publish detection image
        msg_det = self.bridge.cv2_to_imgmsg(detect_img, encoding='bgr8')
        msg_det.header = self.latest_img_msg.header
        self.pub_det_image.publish(msg_det)

        # LiDAR overlay
        overlay_img = self.draw_lidar_points(cv_img.copy(), uvz)
        msg_ol = self.bridge.cv2_to_imgmsg(overlay_img, encoding='bgr8')
        msg_ol.header = self.latest_img_msg.header
        self.pub_lidar_overlay.publish(msg_ol)

        # detection array
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
            # store depth in pose if needed
            # hyp.pose.pose.position.z = float(zz)
            d2.results.append(hyp)
            d2d_array.detections.append(d2)
        self.pub_detections.publish(d2d_array)

        # 6) Build a top-down BEV image
        #    transform detection centers from camera->IMU => draw scenario
        #    We'll do Nx3 => (x,y,z) in IMU frame => draw
        bev_img = np.zeros((600,600,3), dtype=np.uint8)  # black canvas
        if self.latest_imu_msg and self.latest_gps_msg:
            # transform detection centers => IMU
            uvz_centers = bboxes_out[:,-3:]  # Nx3
            xyz_imu = transform_uvz(uvz_centers, self.T_cam2_imu_4x4)
            bev_img = draw_scenario(bev_img, xyz_imu, bev_img.shape[0], scale=2)

        # publish
        bev_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding='bgr8')
        bev_msg.header = self.latest_img_msg.header
        self.pub_bev_image.publish(bev_msg)

        # 7) Add to Folium map if you want
        if (self.latest_imu_msg is not None) and (self.latest_gps_msg is not None):
            lat0 = self.latest_gps_msg.latitude
            lon0 = self.latest_gps_msg.longitude
            alt0 = self.latest_gps_msg.altitude
            heading0 = self.imu_yaw(self.latest_imu_msg)

            # object centers => IMU => lat/lon
            lla = imu2geodetic(xyz_imu[:,0], xyz_imu[:,1], xyz_imu[:,2],
                               lat0, lon0, alt0, heading0)
            if not self.map_initialized:
                self.drive_map = folium.Map(location=(lat0, lon0), zoom_start=17)
                folium.CircleMarker(location=(lat0, lon0),
                                    radius=2, weight=5, color='red').add_to(self.drive_map)
                self.map_initialized = True

            for pos in lla:
                folium.CircleMarker(location=(pos[0], pos[1]),
                                    radius=2, weight=5, color='green').add_to(self.drive_map)

            # also the ego again
            folium.CircleMarker(location=(lat0, lon0),
                                radius=2, weight=5, color='red').add_to(self.drive_map)
            


            # save map
            self.drive_map.save("drive_map.html")
            self.get_logger().info("Folium map updated at drive_map.html.")

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
    # Helper: project Nx3 => Nx3 (u,v,z)
    # ----------------------------------------------------
    def project_lidar_to_camera(self, xyz_points, img_shape):
        """
        Nx3 => Nx4 => T_velo_cam2_4x4 => => keep z>0 => (u,v) in bounds
        """
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
        uvz Nx3 => [u,v,z]
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
            dist = (uvals - cx)**2 + (vvals - cy)**2
            idx = np.argmin(dist)
            out[i,6] = uvals[idx]
            out[i,7] = vvals[idx]
            out[i,8] = zvals[idx]
            cv2.putText(
                image,
                f"{zvals[idx]:.2f} m",
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
            # color
            color = (0,0,255)
            cv2.circle(image, (u_, v_), 2, color, -1)
        return image

    # ----------------------------------------------------
    # Helper: get yaw from IMU (quaternion -> euler)
    # ----------------------------------------------------
    def imu_yaw(self, imu_msg: Imu):
        qx = imu_msg.orientation.x
        qy = imu_msg.orientation.y
        qz = imu_msg.orientation.z
        qw = imu_msg.orientation.w
        siny_cosp = 2.0*(qw*qz + qx*qy)
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
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
