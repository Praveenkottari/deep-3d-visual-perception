import numpy as np
import cv2
from sklearn import linear_model

from .kitti_utils import xyzw2camera


def velo_to_image(pts_velo, calib):
    """
    pts_velo : (N,3)   xyz in Velodyne frame
    returns   : (N,2)   pixel coordinates in the RGB image
    """
    pts_velo_h = np.c_[pts_velo, np.ones(pts_velo.shape[0])]      # (N,4)
    pts_cam    = calib.V2C @ pts_velo_h.T                         # (3,N)
    pts_rect   = calib.R0 @ pts_cam                               # (3,N)
    pts_img_h  = calib.P2 @ np.vstack([pts_rect, np.ones((1,pts_rect.shape[1]))])
    pts_img_h[:2] /= pts_img_h[2:]
    return pts_img_h[:2].T                                        # (N,2)
# ──────────────────────────────────────────────────────────────────────────────#


def _get_box_corners_velo(cx, cy, cz, h, w, l, yaw):
    """
    Returns the 8 corners of a 3‑D box in the Velodyne frame.
    """
    # box corners in the object’s local frame
    x_c =  l / 2.0;  x_n = -x_c
    y_c =  w / 2.0;  y_n = -y_c
    z_b = 0.0        # KITTI boxes have cz at the bottom face in Velodyne ↑
    z_t = -h         # top face (negative z in Velodyne)

    corners = np.array([
        [x_c, y_c, z_b], [x_c, y_n, z_b], [x_n, y_n, z_b], [x_n, y_c, z_b],
        [x_c, y_c, z_t], [x_c, y_n, z_t], [x_n, y_n, z_t], [x_n, y_c, z_t]
    ]).T  # (3, 8)

    # rotate about Z (up) then translate
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[ c, -s, 0.],
                  [ s,  c, 0.],
                  [ 0., 0., 1.]], dtype=np.float32)

    corners = (R @ corners).T + np.array([cx, cy, cz], dtype=np.float32)
    return corners  # (8, 3)
# ──────────────────────────────────────────────────────────────────────────────#


def annotate_depths_3d(image, dets_velo, calib,
                       use_euclidean=True, draw=True):

    """
        Annotate an RGB frame with the distance to each detected object and
        return a copy of the detections array with that distance appended.

        The distance is measured from the ego LiDAR origin to the **nearest
        corner** of every 3‑D bounding box (rather than to its geometric
        centre).  Optionally, the value is overlaid as text on the image.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), dtype=np.uint8
            BGR image (OpenCV format) that will be modified *in‑place* if
            ``draw`` is True.
        dets_velo : np.ndarray, shape (N, 8+)  
            Detections in the Velodyne coordinate frame. Each row is  
            ``[cls, cx, cy, cz, h, w, l, yaw, (score, ...)]``.
        calib : kitti_data_utils.Calibration
            KITTI calibration object providing V2C, R0, and P matrices used
            to project Velodyne points into the camera image.
        use_euclidean : bool, default True
            • **True** distance = √(x² + y² + z²) of the nearest corner  
            • **False** distance = smallest positive *x* (forward range)
            among the eight corners; corners behind the sensor are ignored.
        draw : bool, default True
            If True, draws a “〈distance〉m” label just above each box centre.

        Returns
        -------
        image_out : np.ndarray
            Reference to the same ``image`` array (text is drawn if requested).
        dets_out : np.ndarray, shape (N, dets_velo.shape[1] + 1)
            Copy of ``dets_velo`` with an extra column containing the computed
            distance for every detection.

        Notes
        -----
        • KITTIs Velodyne frame has +Z pointing up; the bottom face of each
        box lies on the z = 0 plane.  
        • For forward range mode (``use_euclidean=False``) the function
        discards corners with x<0 before taking the minimum, ensuring
        that objects behind the ego vehicle are not considered.
        """
    N        = dets_velo.shape[0]
    out      = np.zeros((N, dets_velo.shape[1] + 1))
    out[:, :dets_velo.shape[1]] = dets_velo

    # Pre‑compute pixel coords of all *centres* once (for placing the text)
    centres_velo = dets_velo[:, 1:4]            # (N,3)
    centres_img  = velo_to_image(centres_velo, calib)  # (N,2)

    for i, det in enumerate(dets_velo):
        _, cx, cy, cz, h, w, l, yaw = det[:8]

        # ----- nearest‑point distance -------------------------------------------------
        corners = _get_box_corners_velo(cx, cy, cz, h, w, l, yaw)  # (8,3)
        if use_euclidean:
            dists = np.linalg.norm(corners, axis=1)                # (8,)
        else:  # “forward‑range” ⇒ use x‑component only
            dists = corners[:, 0]                                  # (8,)
            dists[dists < 0] = np.inf                              # ignore behind‑ego points
        depth = float(dists.min())
        # ------------------------------------------------------------------------------
        out[i, -1] = depth

        if draw:
            u, v = map(int, np.round(centres_img[i]))
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.putText(image,
                            f"{depth:.1f} m",
                            (u, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)
    return image, out
# ──────────────────────────────────────────────────────────────────────────────#

def project_lid2uvz(lidar_xyz, T_uvz_velo, image, remove_plane=False):
    ''' Projects LiDAR point cloud onto the image coordinate frame (u, v, z)
        '''
    if remove_plane:
        # lidar_xyz = lidar_xyz.T
        # lidar_xyz = lidar_xyz[:,0:3]
        lidar_xyz  =lidar_xyz[:3,:].T
        lidar_xyz = np.delete(lidar_xyz, np.where(lidar_xyz[3, :] < 0), axis=1)

        ransac = linear_model.RANSACRegressor(
                                linear_model.LinearRegression(),
                                residual_threshold=0.1,
                                max_trials=5000
                                )

        X = lidar_xyz[:, :2]
        y = lidar_xyz[:, -1]
        ransac.fit(X, y)
        
        # remove outlier points (i.e. remove ground plane)
        mask = ransac.inlier_mask_
        xyz = lidar_xyz[~mask]

        lidar_xyz = np.insert(xyz, 3, 1, axis=1).T 
    else:
        lidar_xyz =lidar_xyz

    # project velo (x, z, y, w) onto camera (u, v, z) coordinates
    velo_uvz = xyzw2camera(lidar_xyz, T_uvz_velo, image, remove_outliers=True)
    return velo_uvz

##──────────────────────────────────────────────────────────────────────────#

def lidar_points(img_rgb, lidar_xyz, T_velo_cam2,remove_plane):

    # Project LiDAR points to camera space
    velo_uvz = project_lid2uvz(lidar_xyz, T_velo_cam2, img_rgb, remove_plane=remove_plane)
    return velo_uvz

# ──────────────────────────────────────────────────────────────────────────────#

# plotting functions (place these in KITTI plot utils
from matplotlib import cm

# get color map function
rainbow_r = cm.get_cmap('rainbow_r', lut=100)
get_color = lambda z : [255*val for val in rainbow_r(int(z.round()))[:3]]

def draw_velo_on_rgbimage(lidar_xyz,T_velo_cam2,image, remove_plane=True,draw_lidar = True,color_map=get_color):
    
    velo_uvz = project_lid2uvz(lidar_xyz,T_velo_cam2, image=image, remove_plane=remove_plane)
    if draw_lidar:
        # unpack LiDAR points
        u, v, z = velo_uvz

        # draw LiDAR point cloud on blank image
        for i in range(len(u)):
            cv2.circle(image, (int(u[i]), int(v[i])), 1, 
                    color_map(z[i]), -1)
        return image
    else:       
        return image
