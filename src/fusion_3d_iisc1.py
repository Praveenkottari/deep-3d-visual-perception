
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

#detection module
from heads.DETECTOR.sfa.data_process.demo_dataset import Demo_KittiDataset
from heads.DETECTOR.sfa.models.model_utils import create_model
from heads.DETECTOR.sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import heads.DETECTOR.sfa.config.kitti_config as cnf
from heads.DETECTOR.sfa.data_process.transformation import lidar_to_camera_box
from heads.DETECTOR.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from heads.DETECTOR.sfa.data_process.kitti_data_utils import Calibration
from heads.DETECTOR.sfa.utils.demo_utils import parse_demo_configs, do_detect

# fusion modules
from pkgs.kitti_utils import *
from pkgs.kitti_detection_utils import *
from pkgs.utils import *
from pkgs.cam_to_cam import cam_transformation
from pkgs.lid_to_cam import lid_transformation

from pkgs.fusion_utils import *
from BEV.bev import *

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


DEBUG = True


## main loop
def main(): 
    configs = parse_demo_configs()
   
        # Ensure default paths exist if not set in config
    if not hasattr(configs, 'detect_logs'):
        configs.detect_logs = './logs'
    if not hasattr(configs, 'results_dir'):
        configs.results_dir = './results'
    CLASS_NAME_BY_ID = {v: k for k, v in cnf.CLASS_NAME_TO_ID.items() if v >= 0}

    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/IISc_drive"
    calib = Calibration(configs.calib_path)

    # # Create 4x4 V2C from 3x4
    # V2C_4x4 = np.eye(4)
    # V2C_4x4[:3, :] = calib.V2C  # calib.V2C is 3x4
    # # Create 4x4 R0 from 3x3
    # R0_4x4 = np.eye(4)
    # R0_4x4[:3, :3] = calib.R0  # calib.R0 is 3x3
    # # Compose full 4x4 transformation
    # T_velo_to_rect = R0_4x4 @ V2C_4x4  # Now 4x4

    # # Final projection: projection matrix X Lidar to rectified camera matrix
    # T_velo_cam2 = calib.P2 @ T_velo_to_rect  # (3x4) = (3x4) × (4x4)



        # KITTI Intrinsics (P2)
    P2 = np.array([
        [707.0493, 0, 604.0814, 45.75831],
        [0, 707.0493, 180.5066, -0.3454157],
        [0, 0, 1.0, 0.004981016]
    ])

    # Identity Rectification
    R0 = np.eye(4)

    # Extrinsic (your Tr_velo_to_cam → padded to 4x4)
    V2C = np.array([
        [0.0569924, -0.997872,  0.0316729, -0.266746],
        [0.0267669, -0.0301858, -0.999186,  0.059941],
        [0.998016,   0.0577937,  0.0249896,  0.247585],
        [0, 0, 0, 1]
    ])



# # Identity Rectification
    # R0 = np.eye(4)
    # P2 = np.array([
    #     [1895.0136, 0.0, 734.5304, 0.0],
    #     [0.0, 1888.1559, 510.8486, 0.0],
    #     [0.0, 0.0, 1.0, 0.0]
    # ])

    # V2C = np.array([
    #     [0.05673, -0.99837, 0.00627, -0.18829],
    #     [0.01552, -0.00539, -0.99987, 0.12501],
    #     [0.99827, 0.05682, 0.01519, 0.17587],
    #     [0, 0, 0, 1]
    # ])

    # # Compose final transformation
    # T_velo_to_rect = R0 @ V2C
    # T_velo_cam2 = P2 @ T_velo_to_rect



    # Compose final transformation
    T_velo_to_rect = R0 @ V2C
    T_velo_cam2 = P2 @ T_velo_to_rect

    np.set_printoptions(precision=6, suppress=True)
    print("T_velo_cam2 = \n", T_velo_cam2)

    ## Model
    model3d = create_model(configs)
    print('\n\n' + '*' * 60 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model3d.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))
    #Assign process to the CPU
    # configs.device = torch.device('cpu' if configs.no_cuda or configs.gpu_idx == -1 else 'cuda:{}'.format(configs.gpu_idx))
    configs.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model3d = model3d.to(device=configs.device)
    model3d.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)
    fps_window = deque(maxlen=30) 
    prev_t = time.time()


    if DEBUG == False:

        log_dir = os.path.join(configs.detect_logs, timestamp + "_logs")
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "all_detections.csv")
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'class', 'x', 'y', 'z', 'h', 'w', 'l', 'yaw'])  # header


    ## Looping thorugh all the samples in the dataset
    with torch.no_grad():
        sample_idx = 10
        
        metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)

        lidar_xyz = metadatas['lidarData'][:, :4]          # drop reflectance
        lidar_xyz = lidar_xyz.T

        ## RGB raw Image from the dataset
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))  

        #lidar projection on rgb with ground plan removal option
        img_bgr = draw_velo_on_rgbimage(lidar_xyz,T_velo_cam2, img_bgr,remove_plane=False,draw_lidar = True)
        cv2.imshow("overlay",img_bgr)
        cv2.waitKey(0)

        # Front and back detection in the lidar space
        front_detections, front_bevmap, _= do_detect(configs, model3d, front_bevmap, is_front=True)
        back_detections, back_bevmap, _ = do_detect(configs, model3d, back_bevmap, is_front=False) 

        front_bevmap_nobox = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        front_bevmap_nobox = cv2.resize(front_bevmap_nobox, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) 
        front_bevmap_nobox = cv2.rotate(front_bevmap_nobox, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back_bevmap_nobox = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        back_bevmap_nobox = cv2.resize(back_bevmap_nobox, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT)) 
        back_bevmap_nobox = cv2.rotate(back_bevmap_nobox, cv2.ROTATE_90_CLOCKWISE)  

        full_nobox = np.concatenate((back_bevmap_nobox, front_bevmap_nobox), axis=1)
        cv2.imshow("bev_nobox",full_nobox)
        cv2.waitKey(1)
        

        # Draw prediction on front top view lidar image
        front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        front_bevmap = draw_predictions(front_bevmap, front_detections, configs.num_classes)

        
        # Draw prediction back topview of lidar image
        back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        back_bevmap = draw_predictions(back_bevmap, back_detections, configs.num_classes)

        

        # Rotate the front_bevmap
        front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE) 
        # Rotate the back_bevmap
        back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)
        # merge front and back bevmap to get full top lidar view with detection and boudning box
        full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)

        
        # skip early if nothing detected
        if front_detections is not None and len(front_detections) > 0:
            # Convert to metric Velodyne frame
            front_real = convert_det_to_real_values(front_detections)

            # Convert tensor to numpy array before use
            if isinstance(front_real, torch.Tensor):
                front_real = front_real.cpu().numpy()

            if front_real.size > 0:
                # Logging (only if DEBUG is off)
                if DEBUG == False:                      
                    for det in front_real:
                        cls_id = int(det[0])
                        x, y, z, h, w, l, yaw = det[1:]
                        csv_writer.writerow([sample_idx, cls_id, x, y, z, h, w, l, yaw])

            # # draw 3-D wireframes (need camera-frame corners)
            # front_cam = front_real.copy()
            # front_cam[:, 1:] = lidar_to_camera_box(front_cam[:, 1:], calib.V2C, calib.R0, calib.P2)
            print(f"[INFO] front_real shape: {front_real.shape}")


            # Ensure proper shape
            if front_real.ndim == 1:
                front_real = np.expand_dims(front_real, axis=0)

            front_cam = front_real.copy()
            front_cam[:, 1:] = lidar_to_camera_box(front_cam[:, 1:], calib.V2C, calib.R0, calib.P2)



            img_bgr = show_rgb_image_with_boxes(img_bgr, front_cam, calib)
            
            # Draw class labels on image
            for det in front_real:
                cls_id = int(det[0])
                x, y, z, h = det[1:5]
                top_z = z + h   # KITTI boxes grow downward, so top = center - height/2
                class_name = CLASS_NAME_BY_ID.get(cls_id, f"Class_{cls_id}")

                # Project box center to 2D
                box_center_velo = np.array([[x], [y], [top_z], [1.0]])  # 4x1
                cam_coords = calib.V2C @ box_center_velo           # 3x1
                rect_coords = calib.R0 @ cam_coords                # 3x1
                img_point = calib.P2 @ np.vstack((rect_coords, [1.0]))  # 3x4 * 4x1

                u = int(img_point[0][0] / img_point[2][0])
                v = int(img_point[1][0] / img_point[2][0])

                # Draw label just above box center
                cv2.putText(img_bgr,class_name,(u, v - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2,cv2.LINE_AA)


            # depth text at each box centre
            img_bgr, _ = annotate_depths_3d(img_bgr,front_real,calib,use_euclidean=True,draw=True)
        # cv2.imshow("detect", img_bgr)


        out_img = np.concatenate((img_bgr, full_bev), axis=0)
        full_bev = cv2.rotate(full_bev, cv2.ROTATE_90_COUNTERCLOCKWISE)


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


        # ── FPS calculation ─────────────────────────────
        now_t  = time.time()    
        # if you run on GPU, force CUDA to finish first so the timing is accurate
        if configs.device.type == "cuda":
            torch.cuda.synchronize()

        dt    = now_t - prev_t          # seconds taken for this frame
        fps   = 1.0 / dt if dt else 0.0
        fps_window.append(fps)
        smooth_fps = sum(fps_window) / len(fps_window)
        prev_t = now_t      
        # ── annotate fps #
        # cv2.putText(out_img,f"Speed: {smooth_fps:5.1f} FPS",(900, 400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255, 255, 255),2,cv2.LINE_AA)

        #--------*************************************----------------------#
        ### Create the video writer
        if DEBUG == False:
            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(configs.results_dir, f'{timestamp}_3d_detction.avi')
                print('Create video at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 15, (out_cap_w, out_cap_h))
            ### Write the output frame to the video
            out_cap.write(out_img)
        
        #--------*************************************----------------------#

        ###### DISPLAY REAL TIME
        cv2.imshow("3D detection", out_img)
        # cv2.imshow('full bev with detection',full_bev)
        # cv2.imshow("Depth Map", depth_colored)


    if out_cap:
        out_cap.release()
        cv2.destroyAllWindows()
        csv_file.close()


if __name__ == '__main__':
    main()
