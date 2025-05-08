
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
import torch
import numpy as np

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





### Calibration matrix calculation
cam_calib_file = '/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_cam_to_cam.txt'
P_rect2_cam2,R_ref0_rect2,T_ref0_ref2 = cam_transformation(cam_calib_file)

lid_calib_file = '/home/airl010/1_Thesis/deep-3d-visual-perception/calibration/calib_velo_to_cam.txt'
T_velo_ref0 = lid_transformation(lid_calib_file)

# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 

# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 

### This is the calibration matrix that above code outputs
# T_velo_cam2 = np.array([
#     [ 607.48      , -718.54      ,  -10.188    ,  -95.573   ],
#     [ 180.03      ,    5.8992    , -720.15     ,  -93.457   ],
#     [   0.99997   ,    0.00048595,   -0.0072069,   -0.28464 ]
# ], dtype=np.float32)   # (3, 4)

# ──────────────────────────────────────────────────────────────────────────────#




def main(): 
    configs = parse_demo_configs()
    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_drive_0047_sync/"
    calib = Calibration(configs.calib_path)

    model3d = create_model(configs)
    print('\n\n' + '*' * 60 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model3d.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    #Assign process to the CPU
    configs.device = torch.device('cpu' if configs.no_cuda or configs.gpu_idx == -1 else 'cuda:{}'.format(configs.gpu_idx))
    # configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    model3d = model3d.to(device=configs.device)
    model3d.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)


    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            
            metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)
            lidar_xyz = metadatas['lidarData'][:, :4]          # drop reflectance
            lidar_xyz = lidar_xyz.T


              #RGB raw Image from the dataset
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))  
            

            # get detections and object centers in uvz
            velo_uvz = lidar_points(img_bgr, lidar_xyz, T_velo_cam2,remove_plane=True)

            # canvas_out = draw_scenario(velo_uvz,T_cam2_velo,line_draw=True)
            # cv2.imshow("dist_canvas",canvas_out)

            #lidar projection on rgb
            lidar_proj_image = draw_velo_on_image(velo_uvz, img_bgr,draw_lidar = False)



            # print(metadatas['lidarData'])
            front_detections, front_bevmap, _ = do_detect(configs, model3d, front_bevmap, is_front=True)
            back_detections, back_bevmap, _ = do_detect(configs, model3d, back_bevmap, is_front=False)

            

            # # after you already have front_detections & back_detections
            # front_real = convert_det_to_real_values(front_detections)
            # back_real  = convert_det_to_real_values(back_detections)
            # dets_velo  = np.vstack([front_real, back_real])          # (N,10)

            # img_bgr, dets_with_depth = annotate_depths_3d(
            #         img_bgr, dets_velo, calib,
            #         use_euclidean=True,         # or False for forward range only
            #         draw=True)


            # Draw prediction in the top view lidar image
            front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            front_bevmap = draw_predictions(front_bevmap, front_detections, configs.num_classes)

            # Draw prediction in the topview of lidar image
            back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            back_bevmap = draw_predictions(back_bevmap, back_detections, configs.num_classes)

            # Rotate the front_bevmap
            front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE) 
            # cv2.imshow("fron_bev",front_bevmap)           
            # Rotate the back_bevmap
            back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)
            # cv2.imshow("back_bev",back_bevmap)           
            # merge front and back bevmap to get full top lidar view with detection and boudning box
            full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)
            # cv2.imshow("full_bev",full_bev)   
          

        
            kitti_dets = convert_det_to_real_values(front_detections)
            # ─── FRONT-VIEW ONLY: draw 3-D boxes + depth labels ─────────────
            # 1. skip early if nothing detected
            if front_detections is not None and len(front_detections) > 0:

                # 2. convert network output to metric Velodyne frame
                front_real = convert_det_to_real_values(front_detections)
                if isinstance(front_real, torch.Tensor):
                    front_real = front_real.cpu().numpy()

                # ---- optional confidence filter ---------------------------
                # keep only rows whose score (> last column) exceeds 0.35
                # front_real = front_real[front_real[:, -1] > 0.35]
                # -----------------------------------------------------------

                if front_real.size > 0:
                    # 3-a.  draw 3-D wireframes (need camera-frame corners)
                    front_cam = front_real.copy()
                    front_cam[:, 1:] = lidar_to_camera_box(
                        front_cam[:, 1:], calib.V2C, calib.R0, calib.P2)

                    img_bgr = show_rgb_image_with_boxes(img_bgr, front_cam, calib)

                    # 3-b.  add depth text at each box centre
                    img_bgr, front_with_depth = annotate_depths_3d(
                        img_bgr,
                        front_real,      # still in Velodyne coords
                        calib,
                        use_euclidean=True,   # forward-range? set False if you prefer
                        draw=True)
            # ────────────────────────────────────────────────────────────────


            # kitti_dets = convert_det_to_real_values(front_detections)

            # if len(kitti_dets) > 0:
            #     kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            #     img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)



            out_img = np.concatenate((img_bgr, full_bev), axis=0)
            # cv2.putText(out_img, 'Speed: {:.1f} FPS'.format(fps), org=(900, 400), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,  color = (255, 255, 255), thickness = 2)

            # Create the video writer if not already created
            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(configs.results_dir, '{}demo.avi'.format(configs.foldername))
                print('Create video writer at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 15, (out_cap_w, out_cap_h))
            ###Write the output frame to the video
            out_cap.write(out_img)

            # DISPLAY IN REAL TIME
            cv2.imshow("3D box Demo", out_img)
            key = cv2.waitKey(1) & 0xFF
            # If you want to stop early by pressing 'q'
            if key == ord('q'):
                break

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()