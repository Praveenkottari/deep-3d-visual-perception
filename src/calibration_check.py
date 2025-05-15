import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import torch
import numpy as np
import time
from datetime import datetime
from collections import deque
import csv

# detection module
from heads.SFA3D.sfa.data_process.demo_dataset import Demo_KittiDataset
from heads.SFA3D.sfa.models.model_utils import create_model
import heads.SFA3D.sfa.config.kitti_config as cnf
from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs
from heads.SFA3D.sfa.data_process.kitti_data_utils import Calibration

# fusion modules
from pkgs.kitti_utils import *
from pkgs.kitti_detection_utils import *
from pkgs.utils import *
from pkgs.fusion_utils import *
from BEV.bev import *

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DEBUG = True


def main(): 
    configs = parse_demo_configs()
    
    if not hasattr(configs, 'detect_logs'):
        configs.detect_logs = './logs'
    if not hasattr(configs, 'results_dir'):
        configs.results_dir = './results'


    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_drive_0027_sync/"
    calib = Calibration(configs.calib_path)

    # Create 4x4 V2C from 3x4
    V2C_4x4 = np.eye(4)
    V2C_4x4[:3, :] = calib.V2C  # calib.V2C is 3x4

    # Create 4x4 R0 from 3x3
    R0_4x4 = np.eye(4)
    R0_4x4[:3, :3] = calib.R0  # calib.R0 is 3x3

    # Compose full 4x4 transformation
    T_velo_to_rect = R0_4x4 @ V2C_4x4  # Now 4x4

    # Final projection: P2 (3x4) × T_velo_to_rect (4x4)
    T_velo_cam2 = calib.P2 @ T_velo_to_rect  # (3x4) = (3x4) × (4x4)

    ## Model
    model3d = create_model(configs)
    print('\n\n' + '*' * 60 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model3d.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))
    
    configs.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model3d = model3d.to(device=configs.device)
    model3d.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)

    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)
            lidar_xyz = metadatas['lidarData'][:, :4]  # drop reflectance
            lidar_xyz = lidar_xyz.T

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))  

            # lidar projection using computed calibration
            img_bgr = draw_velo_on_rgbimage(lidar_xyz, T_velo_cam2, img_bgr, remove_plane=False, draw_lidar=True)

            # Step 1: make lidar points homogeneous (4, N)
            lidar_hom = np.vstack((lidar_xyz[:3, :], np.ones((1, lidar_xyz.shape[1]))))  # (4, N)

            # Step 2: generate depth map
            depth_map = create_depth_map(lidar_hom, T_velo_cam2, image_shape=(375, cnf.BEV_WIDTH * 2))  # same size as img_bgr

            # Step 3: visualize or save
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)

            # Apply color map
            depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)

            # Replace black (0 depth) pixels with a specific color (e.g., dark gray or red)
            mask = (depth_map == 0)
            depth_colored[mask] = (255, 255, 255)  # You can try (0, 0, 255) for red

            cv2.imshow("Depth Map", depth_colored)
                        

            if not DEBUG:
                if out_cap is None:
                    out_cap_h, out_cap_w = img_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out_path = os.path.join(configs.results_dir, f'{timestamp}_3d_detection.avi')
                    print('Create video at {}'.format(out_path))
                    out_cap = cv2.VideoWriter(out_path, fourcc, 15, (out_cap_w, out_cap_h))
                out_cap.write(img_bgr)

            cv2.imshow("3D detection", img_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
