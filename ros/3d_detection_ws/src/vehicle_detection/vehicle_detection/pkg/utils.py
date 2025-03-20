
from sensor_msgs.msg import Image, PointCloud2, PointField
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from sklearn import linear_model
import struct


import numpy as np



# ---------------
# Utility functions
# ---------------

def get_rigid_transformation(calib_path):
    """Reads a KITTI-style [R, T] calibration => 4×4 transform."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    R = np.array([float(x) for x in lines[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in lines[2].strip().split(' ')[1:]]).reshape((3,1))
    T_4x4 = np.vstack((np.hstack((R, t)), np.array([0,0,0,1])))
    return T_4x4

def transform_uvz(uvz, T_4x4):
    """Convert Nx3 => Nx3 in target frame by interpreting (u,v,z)->(X,Y,Z)."""
    uvzw = np.column_stack((
        uvz[:,0]*uvz[:,2],
        uvz[:,1]*uvz[:,2],
        uvz[:,2],
        np.ones(len(uvz))
    ))
    xyz_h = T_4x4 @ uvzw.T
    return xyz_h[:3,:].T

def remove_ground_plane(xyz_points, threshold=0.1, max_trials=5000):
    """
    RANSAC-based ground removal
    Input: Nx3 (x,y,z)
    Output: Nx3 (just the points not on ground)
    """
    if xyz_points.shape[0] < 10:
        return xyz_points
    ransac = linear_model.RANSACRegressor(
        linear_model.LinearRegression(),
        residual_threshold=threshold,
        max_trials=max_trials
    )
    X = xyz_points[:, :2]  # (x, y)
    y = xyz_points[:, 2]   # z
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_  # ground
    # We want the outliers (non-ground)
    obj_points = xyz_points[~inlier_mask]
    return obj_points

def create_pointcloud2(points_xyz):
    """
    Convert Nx3 array -> PointCloud2 (ROS) with fields x,y,z float32
    """
    msg = PointCloud2()
    msg.header.frame_id = "base_link"  # or camera_link, etc.
    msg.height = 1
    msg.width  = points_xyz.shape[0]
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step   = 12  # 3 x float32
    msg.row_step     = msg.point_step * points_xyz.shape[0]
    msg.is_dense     = True

    data = []
    for p in points_xyz:
        data.append(struct.pack('fff', p[0], p[1], p[2]))
    msg.data = b"".join(data)
    return msg

def create_bev_image(lidar_xyz, size=(600,600), scale=10.0, max_range=30.0):
    """
    Creates a simple BEV (bird's-eye view) image from LiDAR points.
    - Assumes x is forward, y is left, z is up in LiDAR frame.
    - scale: pixels per meter
    - max_range: we only draw points within [-max_range, max_range] in X and Y
    - size: output image size (H, W)
    The origin of the image is placed at bottom-center of the image.
    """
    H, W = size
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    # Filter points by range
    mask = (
        (lidar_xyz[:,0] > -5) & (lidar_xyz[:,0] < max_range) &
        (lidar_xyz[:,1] > -max_range) & (lidar_xyz[:,1] < max_range)
    )
    points = lidar_xyz[mask]

    # Center in the image
    # Let's define x forward => up in image, y left => left in image.
    # So the origin is at (W/2, H-1).
    origin = (W // 2, H - 1)

    for (x, y, z) in points:
        # Convert x,y in [m] to pixel coords
        px = int(origin[0] + (-y) * scale)  # -y so left is left
        py = int(origin[1] - x * scale)     # x forward => up in image
        if (px < 0 or px >= W or py < 0 or py >= H):
            continue

        # Color by height z
        # Let's clamp z to [-2..2] just for color mapping
        z_clamped = max(-2.0, min(2.0, z))
        # scale it to 0..255
        intensity = int((z_clamped + 2.0) / 4.0 * 255)
        bev[py, px] = (intensity, intensity, intensity)

    return bev
