
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np

from heads.SFA3D.sfa.data_process.demo_dataset import Demo_KittiDataset
from heads.SFA3D.sfa.models.model_utils import create_model
from heads.SFA3D.sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
import heads.SFA3D.sfa.config.kitti_config as cnf
from heads.SFA3D.sfa.data_process.transformation import lidar_to_camera_box
from heads.SFA3D.sfa.utils.visualization_utils import show_rgb_image_with_boxes

from heads.SFA3D.sfa.data_process.kitti_data_utils import Calibration
from heads.SFA3D.sfa.utils.demo_utils import parse_demo_configs, do_detect, write_credit

# detection model 
from pkgs.kitti_utils import *
from pkgs.kitti_detection_utils import *
from pkgs.utils import *

from heads.detection_head import *
from BEV.bev import *

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
            • **True**  – distance = √(x² + y² + z²) of the nearest corner  
            • **False** – distance = smallest positive *x* (forward range)
            among the eight corners; corners behind the sensor are ignored.
        draw : bool, default True
            If True, draws a “〈distance〉 m” label just above each box centre.

        Returns
        -------
        image_out : np.ndarray
            Reference to the same ``image`` array (text is drawn if requested).
        dets_out : np.ndarray, shape (N, dets_velo.shape[1] + 1)
            Copy of ``dets_velo`` with an extra column containing the computed
            distance for every detection.

        Notes
        -----
        • KITTI’s Velodyne frame has +Z pointing up; the bottom face of each
        box lies on the z = 0 plane.  
        • For forward‑range mode (``use_euclidean=False``) the function
        discards corners with x < 0 before taking the minimum, ensuring
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

T_velo_cam2 = np.array([
    [ 607.48      , -718.54      ,  -10.188    ,  -95.573   ],
    [ 180.03      ,    5.8992    , -720.15     ,  -93.457   ],
    [   0.99997   ,    0.00048595,   -0.0072069,   -0.28464 ]
], dtype=np.float32)   # (3, 4)

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

# ──────────────────────────────────────────────────────────────────────────────#

def lidar_points(img_rgb, lidar_xyz, T_velo_cam2,remove_plane):

    # Project LiDAR points to camera space
    velo_uvz = project_lid2uvz(lidar_xyz, T_velo_cam2, img_rgb, remove_plane=remove_plane)
    return velo_uvz

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