
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
from heads.SFA3D.sfa.data_process.demo_dataset import Demo_KittiDataset
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import heads.SFA3D.sfa.config.kitti_config as cnf
from heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from heads.SFA3D.sfa.data_process.kitti_data_utils import Calibration
from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect

# fusion modules
from pkgs.kitti_utils import *
from pkgs.kitti_detection_utils import *
from pkgs.utils import *
from pkgs.cam_to_cam import cam_transformation
from pkgs.lid_to_cam import lid_transformation

from pkgs.fusion_utils import *
from BEV.bev import *

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────────────────────────────────────#
### Calibration matrix calculation
cam_calib_file = '/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_cam_to_cam.txt'
lid_calib_file = '/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_velo_to_cam.txt'

P_rect2_cam2,R_ref0_rect2,T_ref0_ref2 = cam_transformation(cam_calib_file)
T_velo_ref0 = lid_transformation(lid_calib_file)

# transform from LiDAR to camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0       

# homogeneous transform from camera to LiDAR (shape: 4x4)
# T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 

### This is the calibration matrix that above code outputs
# T_velo_cam2 = np.array([
#     [ 607.48      , -718.54      ,  -10.188    ,  -95.573   ],
#     [ 180.03      ,    5.8992    , -720.15     ,  -93.457   ],
#     [   0.99997   ,    0.00048595,   -0.0072069,   -0.28464 ]
# ], dtype=np.float32)   # (3, 4)
# ──────────────────────────────────────────────────────────────────────────────#

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

    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_drive_0027_sync/"
    calib = Calibration(configs.calib_path)

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
        for sample_idx in range(len(demo_dataset)):
            
            metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)
            lidar_xyz = metadatas['lidarData'][:, :4]          # drop reflectance
            lidar_xyz = lidar_xyz.T

            ## RGB raw Image from the dataset
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))  
            

            #lidar projection on rgb with ground plan removal option
            img_bgr = draw_velo_on_rgbimage(lidar_xyz,T_velo_cam2, img_bgr,remove_plane=False,draw_lidar = False)

            # Front and back detection in the lidar space
            front_detections, front_bevmap, _= do_detect(configs, model3d, front_bevmap, is_front=True)
            back_detections, back_bevmap, _ = do_detect(configs, model3d, back_bevmap, is_front=False)    
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

                # draw 3-D wireframes (need camera-frame corners)
                front_cam = front_real.copy()
                front_cam[:, 1:] = lidar_to_camera_box(
                    front_cam[:, 1:], calib.V2C, calib.R0, calib.P2)
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


            out_img = np.concatenate((img_bgr, full_bev), axis=0)
            full_bev = cv2.rotate(full_bev, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
            cv2.putText(out_img,f"Speed: {smooth_fps:5.1f} FPS",(900, 400),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255, 255, 255),2,cv2.LINE_AA)

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

            # DISPLAY REAL TIME
            cv2.imshow("3D detection", out_img)
            # cv2.imshow('full bev with detection',full_bev)


            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    vis.destroy_window()


if __name__ == '__main__':
    main()
