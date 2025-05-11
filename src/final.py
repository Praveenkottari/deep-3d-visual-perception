
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import torch
import numpy as np
import time
from collections import deque

#detection module
from heads.SFA3D.sfa.data_process.demo_dataset import Demo_KittiDataset
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import heads.SFA3D.sfa.config.kitti_config as cnf
from heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes
from heads.SFA3D.sfa.data_process.kitti_data_utils import Calibration
from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect
from heads.detection_head import *

# fusion modules
from pkgs.kitti_utils import *
from pkgs.kitti_detection_utils import *
from pkgs.utils import *
from pkgs.cam_to_cam import cam_transformation
from pkgs.lid_to_cam import lid_transformation

from pkgs.fusion_utils import *
from BEV.bev import *

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


## main loop
def main(): 
    configs = parse_demo_configs()
    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_drive_0027_sync/"
    calib = Calibration(configs.calib_path)

    ## Model
    model3d = create_model(configs)
    print('\n\n' + '*' * 60 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model3d.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))
    #Assign process to the CPU
    configs.device = torch.device('cpu' if configs.no_cuda or configs.gpu_idx == -1 else 'cuda:{}'.format(configs.gpu_idx))
    model3d = model3d.to(device=configs.device)
    model3d.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)


    prev_t = time.time()      # wall‑clock of previous frame
    fps_window = deque(maxlen=30) # rolling‑mean smoother (≈½ s @ 60 FPS)

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
            front_detections, front_bevmap, _ = do_detect(configs, model3d, front_bevmap, is_front=True)
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
            cv2.imshow("full_bev",full_bev)   
        
            # skip early if nothing detected
            if front_detections is not None and len(front_detections) > 0:
                # convert network output to metric Velodyne frame
                front_real = convert_det_to_real_values(front_detections)
                if isinstance(front_real, torch.Tensor):
                    front_real = front_real.cpu().numpy()

                if front_real.size > 0:
                    # draw 3-D wireframes (need camera-frame corners)
                    front_cam = front_real.copy()
                    front_cam[:, 1:] = lidar_to_camera_box(
                        front_cam[:, 1:], calib.V2C, calib.R0, calib.P2)

                    img_bgr = show_rgb_image_with_boxes(img_bgr, front_cam, calib)

                    # depth text at each box centre
                    img_bgr, _ = annotate_depths_3d(
                        img_bgr,
                        front_real,      
                        calib,
                        use_euclidean=True,draw=True)

            out_img = np.concatenate((img_bgr, full_bev), axis=0)
            # ── FPS calculation ─────────────────────────────
            now_t  = time.time()    
            # if you run on GPU, force CUDA to finish first so the timing is accurate
            if configs.device.type == "cuda":
                torch.cuda.synchronize()

            dt    = now_t - prev_t          # seconds taken for this frame
            prev_t = now_t
            fps   = 1.0 / dt if dt else 0.0

            fps_window.append(fps)
            smooth_fps = sum(fps_window) / len(fps_window)

            # ── annotate ────────────────────────────────────
            cv2.putText(
                out_img,
                f"Speed: {smooth_fps:5.1f} FPS",
                (900, 400),                       # (x, y) top‑left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,                            # font scale
                (255, 255, 255),                  # BGR colour (light yellow)
                2,                              # thickness
                cv2.LINE_AA                     # anti‑aliased
            )
            # Create the video writer
            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(configs.results_dir, '{}3d_detction.avi'.format(configs.foldername))
                print('Create video at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 15, (out_cap_w, out_cap_h))
            ### Write the output frame to the video
            out_cap.write(out_img)

            # DISPLAY REAL TIME
            cv2.imshow("3D detection Demo", out_img)
            key = cv2.waitKey(1) & 0xFF
            # 'q' to stop
            if key == ord('q'):
                break
    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
