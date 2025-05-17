# detect3d_v2/detect3d_node.py

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2
import torch
from collections import deque
from datetime import datetime

# === SFA3D + fusion imports === #
from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import heads.SFA3D.sfa.config.kitti_config as cnf
from heads.SFA3D.sfa.data_process.kitti_data_utils import Calibration
from heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from pkgs.fusion_utils import draw_velo_on_rgbimage, annotate_depths_3d
from BEV.bev import makeBEVMap
from pkgs.fusion_utils import *

class Detect3DFusionNode(Node):
    def __init__(self):
        super().__init__('detect3d_v2node')

        self.bridge = CvBridge()
        self.configs = parse_demo_configs()
        self.calib = Calibration(self.configs.calib_path)
        self.prev_t = datetime.now().timestamp()
        self.fps_window = deque(maxlen=30)

        # Create transformation matrix
        V2C_4x4 = np.eye(4)
        V2C_4x4[:3, :] = self.calib.V2C
        R0_4x4 = np.eye(4)
        R0_4x4[:3, :3] = self.calib.R0
        self.T_velo_cam2 = self.calib.P2 @ (R0_4x4 @ V2C_4x4)

        # Model loading
        self.model3d = create_model(self.configs)
        assert os.path.exists(self.configs.pretrained_path)
        self.model3d.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cpu'))
        self.model3d = self.model3d.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model3d.eval()

        self.CLASS_NAME_BY_ID = {v: k for k, v in cnf.CLASS_NAME_TO_ID.items() if v >= 0}

        # ROS Subscriptions
        self.image_sub = Subscriber(self, Image, "/camera/image_color")
        self.lidar_sub = Subscriber(self, PointCloud2, "/lidar/points")
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], 10, 0.1)
        self.ts.registerCallback(self.callback)

    def callback(self, image_msg, pc_msg):
        img_rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))

        points = np.array(list(pc2.read_points(pc_msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)))
        lidar_xyz = points.T

        # draw projected LiDAR on image
        img_bgr = draw_velo_on_rgbimage(lidar_xyz, self.T_velo_cam2, img_bgr, remove_plane=False, draw_lidar=False)

        # Generate BEV maps
        front_bevmap = makeBEVMap(points, projection="front")
        back_bevmap = makeBEVMap(points, projection="back")

        with torch.no_grad():
            front_dets, front_bevmap, _ = do_detect(self.configs, self.model3d, front_bevmap, is_front=True)
            back_dets, back_bevmap, _ = do_detect(self.configs, self.model3d, back_bevmap, is_front=False)

            # Process BEV
            front_bevmap = draw_predictions(cv2.resize((front_bevmap.permute(1, 2, 0).numpy()*255).astype(np.uint8),
                                                      (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)),
                                            front_dets, self.configs.num_classes)
            back_bevmap = draw_predictions(cv2.resize((back_bevmap.permute(1, 2, 0).numpy()*255).astype(np.uint8),
                                                     (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)),
                                           back_dets, self.configs.num_classes)

            front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE)
            back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)
            full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)

            if front_dets is not None and len(front_dets) > 0:
                front_real = convert_det_to_real_values(front_dets)
                if isinstance(front_real, torch.Tensor):
                    front_real = front_real.cpu().numpy()

                front_cam = front_real.copy()
                front_cam[:, 1:] = lidar_to_camera_box(front_cam[:, 1:], self.calib.V2C, self.calib.R0, self.calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, front_cam, self.calib)

                for det in front_real:
                    cls_id = int(det[0])
                    x, y, z, h = det[1:5]
                    top_z = z + h
                    class_name = self.CLASS_NAME_BY_ID.get(cls_id, f"Class_{cls_id}")
                    box_center = np.array([[x], [y], [top_z], [1.0]])
                    cam_coords = self.calib.V2C @ box_center
                    rect_coords = self.calib.R0 @ cam_coords
                    img_point = self.calib.P2 @ np.vstack((rect_coords, [1.0]))
                    u = int(img_point[0][0] / img_point[2][0])
                    v = int(img_point[1][0] / img_point[2][0])
                    cv2.putText(img_bgr, class_name, (u, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                img_bgr, _ = annotate_depths_3d(img_bgr, front_real, self.calib, use_euclidean=True, draw=True)

            out_img = np.concatenate((img_bgr, full_bev), axis=0)

            # Depth map
            lidar_hom = np.vstack((lidar_xyz[:3, :], np.ones((1, lidar_xyz.shape[1]))))
            depth_map = create_depth_map(lidar_hom, self.T_velo_cam2, image_shape=(375, cnf.BEV_WIDTH * 2))
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
            depth_colored[depth_map == 0] = (255, 255, 255)

            # FPS
            now_t = datetime.now().timestamp()
            dt = now_t - self.prev_t
            fps = 1.0 / dt if dt else 0.0
            self.fps_window.append(fps)
            smooth_fps = sum(self.fps_window) / len(self.fps_window)
            self.prev_t = now_t
            cv2.putText(out_img, f"Speed: {smooth_fps:5.1f} FPS", (900, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            # Display windows
            cv2.imshow("3D Detection", out_img)
            cv2.imshow("Full BEV", full_bev)
            cv2.imshow("Depth Map", depth_colored)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = Detect3DFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()