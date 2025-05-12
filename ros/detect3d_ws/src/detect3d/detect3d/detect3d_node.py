#!/usr/bin/env python3
"""
detect3d_node  –  LiDAR + RGB 3D detection (SFA3D) for ROS 2 Foxy

IN  :
  /kitti/image/color/left   sensor_msgs/Image
  /kitti/point_cloud        sensor_msgs/PointCloud2
OUT :
  /detect3d/rgb_overlay     sensor_msgs/Image        – RGB + LiDAR dots
  /detect3d/rgb_boxes       sensor_msgs/Image        – RGB + 3D boxes + depth
  /detect3d/bev_boxes       sensor_msgs/Image        – BEV (front/back) + boxes
  /detect3d/boxes           visualization_msgs/MarkerArray – 3D cubes
"""
import os, sys, time, warnings
from collections import deque
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np, cv2, torch, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import transformations as tf_tf

PKG = os.path.dirname(os.path.realpath(__file__))
sys.path.extend([PKG, os.path.join(PKG, 'heads')])

from heads.SFA3D.sfa.utils.demo_utils import do_detect, parse_demo_configs
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import convert_det_to_real_values, draw_predictions
from heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from BEV.bev import bev_from_pcl
from pkgs.fusion_utils import annotate_depths_3d, draw_velo_on_rgbimage
from pkgs.cam_to_cam import cam_transformation
from pkgs.lid_to_cam import lid_transformation

bridge = CvBridge()

WEIGHT_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/weights/fpn_resnet_18/fpn_resnet_18_epoch_300.pth"
CAM_CALIB_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_cam_to_cam.txt"
LID_CALIB_FILE = "/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_velo_to_cam.txt"

def cloud_to_numpy(pc: PointCloud2) -> np.ndarray:
    raw = np.frombuffer(pc.data, dtype=np.float32)
    pts = raw.reshape(-1, int(pc.point_step/4))[:, :4]
    return pts.copy()

def bev_tensors(points: np.ndarray, device):
    front_np, back_np = bev_from_pcl(points)
    front = torch.from_numpy(front_np).permute(2,0,1).to(device)
    back = torch.from_numpy(back_np ).permute(2,0,1).to(device)
    return front, back, front_np, back_np

class Detect3D(Node):
    def __init__(self):
        super().__init__('detect3d_node')

        cfg = parse_demo_configs()
        cfg.conf_thresh = 0.2
        cfg.peak_thresh = 0.2
        cfg.pretrained_path = WEIGHT_FILE
        cfg.no_cuda, cfg.device = True, torch.device('cpu')
        self.model = create_model(cfg)
        self.model.load_state_dict(torch.load(WEIGHT_FILE, map_location='cpu'))
        self.model.to(cfg.device).eval()
        self.cfg = cfg
        self.get_logger().info(f"Loaded weights: {WEIGHT_FILE}")

        P2, R0_ext, T_ref0_ref2 = cam_transformation(CAM_CALIB_FILE)
        T_velo_ref0 = lid_transformation(LID_CALIB_FILE)
        self.T_velo_cam2 = P2 @ R0_ext @ T_ref0_ref2 @ T_velo_ref0
        self.V2C = (R0_ext @ T_velo_ref0)[:3,:];  self.R0 = R0_ext[:3,:3];  self.P2 = P2

        self.pub_rgb_overlay = self.create_publisher(Image, '/detect3d/rgb_overlay', 5)
        self.pub_rgb_boxes   = self.create_publisher(Image, '/detect3d/rgb_boxes',   5)
        self.pub_bev_boxes   = self.create_publisher(Image, '/detect3d/bev_boxes',   5)
        self.pub_boxes       = self.create_publisher(MarkerArray, '/detect3d/boxes', 5)

        img_sub = Subscriber(self, Image,       '/kitti/image/color/left')
        pcl_sub = Subscriber(self, PointCloud2, '/kitti/point_cloud')
        self.sync = ApproximateTimeSynchronizer([img_sub, pcl_sub], 50, 0.5)
        self.sync.registerCallback(self.cb)

        self.prev_t, self.fps = time.time(), deque(maxlen=30)

    def cb(self, img_msg, pcl_msg):
        self.get_logger().info("Received image and pointcloud")
        rgb = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        pts = cloud_to_numpy(pcl_msg)
        t_front, t_back, np_front, np_back = bev_tensors(pts, self.cfg.device)
        self.get_logger().info(f"BEV front max: {np.max(np_front):.3f}, back max: {np.max(np_back):.3f}")

        with torch.no_grad():
            det_f, _, _ = do_detect(self.cfg, self.model, t_front, True)
            self.get_logger().info(f"Sample det_f: {det_f}")

            det_b, _, _ = do_detect(self.cfg, self.model, t_back, False)
            self.get_logger().info(f"Det_front: {len(det_f) if det_f else 0}, Det_back: {len(det_b) if det_b else 0}")

            det_list = []
            for d in (det_f, det_b):
                if isinstance(d, dict):
                    for arr in d.values():
                        if arr is not None and arr.shape[0] > 0:
                            det_list.append(arr)

            if not det_list:
                self.get_logger().info("No valid detection arrays found.")
                return

            real = convert_det_to_real_values(np.vstack(det_list))
            self.get_logger().info(f"Valid detections: {len(real)}")


        rgb_overlay = draw_velo_on_rgbimage(pts.T, self.T_velo_cam2, rgb.copy(), False, True)
        cam = real.copy(); cam[:,1:] = lidar_to_camera_box(cam[:,1:], self.V2C, self.R0, self.P2)
        rgb_boxes = show_rgb_image_with_boxes(rgb.copy(), cam, self)
        rgb_boxes, _ = annotate_depths_3d(rgb_boxes, real, self, True)

        bev_f = (np_front*255).astype(np.uint8)
        bev_b = (np_back *255).astype(np.uint8)
        bev   = np.concatenate((cv2.rotate(bev_b, cv2.ROTATE_90_CLOCKWISE),
                                cv2.rotate(bev_f, cv2.ROTATE_90_COUNTERCLOCKWISE)), axis=1)
        bev   = draw_predictions(bev, det_f, 3)
        bev   = draw_predictions(bev, det_b, 3)

        hdr = img_msg.header
        for img, pub in zip(
            [rgb_overlay, rgb_boxes, cv2.cvtColor(bev, cv2.COLOR_RGB2BGR)],
            [self.pub_rgb_overlay, self.pub_rgb_boxes, self.pub_bev_boxes]):
            msg = bridge.cv2_to_imgmsg(img, 'bgr8')
            msg.header = hdr
            pub.publish(msg)

        self.pub_boxes.publish(self.mk_markers(real, hdr))

        now = time.time(); self.fps.append(1/(now-self.prev_t)); self.prev_t = now
        if len(self.fps) == self.fps.maxlen:
            self.get_logger().info(f"FPS {sum(self.fps)/len(self.fps):.1f}")

    def mk_markers(self, arr, header): 
        ma = MarkerArray()
        for i, (_,x,y,z,h,w,l,yaw) in enumerate(arr):
            m = Marker(header=header, id=i, type=Marker.CUBE, action=Marker.ADD,
                       scale=[l,w,h], color=[1,0,0,0.4])
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z+h/2
            q = tf_tf.quaternion_from_euler(0,0,-yaw)
            (m.pose.orientation.x,m.pose.orientation.y,
             m.pose.orientation.z,m.pose.orientation.w) = q
            ma.markers.append(m)
        return ma


def main(args=None):
    rclpy.init(args=args); node = Detect3D()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node(); rclpy.shutdown()
