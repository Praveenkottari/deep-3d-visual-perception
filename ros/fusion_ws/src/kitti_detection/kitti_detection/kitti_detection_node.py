#!/usr/bin/env python3
"""
ROS2 Node with:
 - YOLOv8 (reduced resolution for faster speed)
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
from sklearn import linear_model

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

    data = []
    for p in points_xyz:
        data.append(struct.pack('fff', p[0], p[1], p[2]))
    msg.data = b"".join(data)
    return msg

def create_bev_image(lidar_xyz, size=(600,600), scale=10.0, max_range=30.0):
    """
    Creates a simple BEV (bird's-eye view) image from LiDAR points.
    - Assumes x is forward, y is left, z is up in LiDAR frame.
    - scale: pixels per meter
    - max_range: we only draw points within [-max_range, max_range] in X and Y
    - size: output image size (H, W)
    The origin of the image is placed at bottom-center of the image.
    """
    H, W = size
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    # Filter points by range
    mask = (
        (lidar_xyz[:,0] > -5) & (lidar_xyz[:,0] < max_range) &
        (lidar_xyz[:,1] > -max_range) & (lidar_xyz[:,1] < max_range)
    )
    points = lidar_xyz[mask]

    # Center in the image
    # Let's define x forward => up in image, y left => left in image.
    # So the origin is at (W/2, H-1).
    origin = (W // 2, H - 1)

    for (x, y, z) in points:
        # Convert x,y in [m] to pixel coords
        px = int(origin[0] + (-y) * scale)  # -y so left is left
        py = int(origin[1] - x * scale)     # x forward => up in image
        if (px < 0 or px >= W or py < 0 or py >= H):
            continue

        # Color by height z
        # Let's clamp z to [-2..2] just for color mapping
        z_clamped = max(-2.0, min(2.0, z))
        # scale it to 0..255
        intensity = int((z_clamped + 2.0) / 4.0 * 255)
        bev[py, px] = (intensity, intensity, intensity)

    return bev


# ------------------------------
# The Node
# ------------------------------
class KittiCalibratedNode(Node):
    def __init__(self):
        super().__init__('kitti_calibrated_node')

        self.bridge = CvBridge()

        # (1) YOLO model – reduce resolution for speed
        self.model = YOLO("yolov8n.pt")
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

        # calibration file paths – set them to your actual paths
        self.calib_cam_to_cam  = '/path/to/calib_cam_to_cam.txt'
        self.calib_velo_to_cam = '/path/to/calib_velo_to_cam.txt'
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
