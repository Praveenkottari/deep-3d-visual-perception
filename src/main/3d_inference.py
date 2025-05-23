
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

if __name__ == '__main__':
    configs = parse_demo_configs()

    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_drive_0027_sync/"

    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_path))

    # Updated line
    configs.device = torch.device('cpu' if configs.no_cuda or configs.gpu_idx == -1 else 'cuda:{}'.format(configs.gpu_idx))
    # configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)
    model.eval()

    out_cap = None
    demo_dataset = Demo_KittiDataset(configs)

    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)
            front_detections, front_bevmap, fps = do_detect(configs, model, front_bevmap, is_front=True)
            back_detections, back_bevmap, _ = do_detect(configs, model, back_bevmap, is_front=False)

            # Draw prediction in the image
            front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            front_bevmap = draw_predictions(front_bevmap, front_detections, configs.num_classes)
            # Rotate the front_bevmap
            front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Draw prediction in the image
            back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            back_bevmap = draw_predictions(back_bevmap, back_detections, configs.num_classes)
            # Rotate the back_bevmap
            back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)

            # merge front and back bevmap
            full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)

            img_path = metadatas['img_path'][0]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            calib = Calibration(configs.calib_path)
            kitti_dets = convert_det_to_real_values(front_detections)
            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
                img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
            img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))

            out_img = np.concatenate((img_bgr, full_bev), axis=0)
            write_credit(out_img, (50, 410), text_author='', org_fps=(900, 410), fps=fps)

            # Create the video writer if not already created
            if out_cap is None:
                out_cap_h, out_cap_w = out_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out_path = os.path.join(configs.results_dir, '{}lid_cam_off_road.avi'.format(configs.foldername))
                print('Create video writer at {}'.format(out_path))
                out_cap = cv2.VideoWriter(out_path, fourcc, 15, (out_cap_w, out_cap_h))

            # Write the output frame to the video
            out_cap.write(out_img)

            # DISPLAY IN REAL TIME
            # --------------------
            cv2.imshow("SFA3D Demo", out_img)
            key = cv2.waitKey(1) & 0xFF
            # If you want to stop early by pressing 'q'
            if key == ord('q'):
                break

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()
