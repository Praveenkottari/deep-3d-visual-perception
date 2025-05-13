from .kitti_detection_utils import *

def lid_transformation(filepath):

    T_velo_ref0 = get_rigid_transformation(filepath)
    return T_velo_ref0

if __name__ == '__main__':
    lid_transformation()