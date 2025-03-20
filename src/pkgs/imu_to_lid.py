from pkgs.kitti_detection_utils import *

def imu_transformation(filepath):

    T_imu_velo = get_rigid_transformation(filepath)
    return T_imu_velo

if __name__ == '__main__':
    imu_transformation()