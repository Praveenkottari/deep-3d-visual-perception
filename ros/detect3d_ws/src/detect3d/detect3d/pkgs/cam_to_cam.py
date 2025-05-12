import os
from glob import glob
import numpy as np

def cam_transformation(filepath):
    with open(filepath,'r') as f:
        calib = f.readlines()

    # get projection matrices (rectified left camera --> left camera (u,v,z))
    P_rect2_cam2 = np.array([float(x) for x in calib[25].strip().split(' ')[1:]]).reshape((3,4))

    # get rectified rotation matrices (left camera --> rectified left camera)
    R_ref0_rect2 = np.array([float(x) for x in calib[24].strip().split(' ')[1:]]).reshape((3, 3,))

    # add (0,0,0) translation and convert to homogeneous coordinates
    R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0], axis=0)
    R_ref0_rect2 = np.insert(R_ref0_rect2, 3, values=[0,0,0,1], axis=1)


    # get rigid transformation from Camera 0 (ref) to Camera 2
    R_2 = np.array([float(x) for x in calib[21].strip().split(' ')[1:]]).reshape((3,3))
    t_2 = np.array([float(x) for x in calib[22].strip().split(' ')[1:]]).reshape((3,1))

    # get cam0 to cam2 rigid body transformation in homogeneous coordinates
    T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)
    
    return P_rect2_cam2, R_ref0_rect2, T_ref0_ref2



if __name__ == '__main__':
    pass