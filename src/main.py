import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

from pkgs.kitti_utils import *
from pkgs.kitti_detection_utils import *
from pkgs.utils import *

from models.detection_head import *
from calibration.cam_to_cam import *
from calibration.lid_to_cam import *
from calibration.imu_to_lid import *

from BEV.bev import *

from sklearn import linear_model
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from models.detection_head import *

import folium



###################################################################################

#dataset 
DATA_PATH = r'./../dataset/2011_10_03_drive_0047_sync'

#image path
image_paths = sorted(glob(os.path.join(DATA_PATH, 'image_02/data/*.png')))
#lidar path
lid_paths = sorted(glob(os.path.join(DATA_PATH, 'velodyne_points/data/*.bin')))
#GPS/IMU path
imu_paths = sorted(glob(os.path.join(DATA_PATH, r'oxts/data**/*.txt')))


print(f"Number of left images: {len(image_paths)}")
print(f"Number of LiDAR point clouds: {len(lid_paths)}")
print(f"Number of GPS/IMU frames: {len(imu_paths)}")

cam_calib_file = './../dataset/2011_10_03_calib/calib_cam_to_cam.txt'
P_rect2_cam2,R_ref0_rect2,T_ref0_ref2 = cam_transformation(cam_calib_file)

lid_calib_file = './../dataset/2011_10_03_calib/calib_velo_to_cam.txt'
T_velo_ref0 = lid_transformation(lid_calib_file)

imu_calib_file = './../dataset/2011_10_03_calib/calib_imu_to_velo.txt'
T_imu_velo = imu_transformation(imu_calib_file)

# transform from velo (LiDAR) to left color camera (shape 3x4)
T_velo_cam2 = P_rect2_cam2 @ R_ref0_rect2 @ T_ref0_ref2 @ T_velo_ref0 
# homogeneous transform from left color camera to velo (LiDAR) (shape: 4x4)
T_cam2_velo = np.linalg.inv(np.insert(T_velo_cam2, 3, values=[0,0,0,1], axis=0)) 
# transform from IMU to left color camera (shape 3x4)
T_imu_cam2 = T_velo_cam2 @ T_imu_velo
# homogeneous transform from left color camera to IMU (shape: 4x4)
T_cam2_imu = np.linalg.inv(np.insert(T_imu_cam2, 3, values=[0,0,0,1], axis=0)) 

####################################################################################
# detection model

model = detection_model(weights,classes)

####################################################################################





#################################################################################
def main(video=True):
    index = 8

    image_original = cv2.cvtColor(cv2.imread(image_paths[index]), cv2.COLOR_BGR2RGB)

    left_image = image_original.copy()
    bin_path = lid_paths[index]
    #oxts_frame = get_oxts(imu_paths[index])


    # get detections and object centers in uvz
    bboxes, velo_uvz = get_detection_coordinates(left_image, bin_path, model,T_velo_cam2, remove_plane=False)
    Image.fromarray(left_image).show()

    # draw LiDAR points on a blank image or a copy of left_image
    lidar_proj_image = np.zeros_like(left_image)  # black background
    lidar_proj_image = draw_velo_on_image(velo_uvz, lidar_proj_image)

    ## #Draw bounding boxes onto the LiDAR-projected image
    # lidar_proj_image_with_bboxes = draw_bboxes_on_lidar_image(lidar_proj_image.copy(), bboxes)
    # Image.fromarray(lidar_proj_image_with_bboxes).show()

    # #lidar points in the frame
    # velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(left_image))
    # Image.fromarray(velo_image).show()

    uvz = bboxes[:, -3:]
    #lidar co ordinate for detected obejcts 
    canvas_out = draw_scenario(uvz,T_cam2_velo)
    Image.fromarray(canvas_out).show()

    #lidar on image
    velo_on_image = draw_velo_on_image(velo_uvz, image_original)
    Image.fromarray(velo_on_image).show()
    
    if video: 
        #imgae to video
        result_video,cam2_fps,h,w = input_to_video(model,DATA_PATH,image_paths,lid_paths,T_cam2_velo,T_velo_cam2)
        out = cv2.VideoWriter('./result/out2.avi',
                        cv2.VideoWriter_fourcc(*'DIVX'), 
                        cam2_fps, 
                        (w,h))
    
        for i in range(len(result_video)):
            out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))
        out.release()

###################################################################################################################
if __name__ == "__main__":
    main(video=True)