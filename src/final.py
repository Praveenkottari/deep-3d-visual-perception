
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
from heads.detection_head import *


                   # metres

def draw_depth_labels(img_bgr: np.ndarray,
                      boxes_2d: np.ndarray,
                      depths: np.ndarray,
                      color=(0, 255, 255)) -> np.ndarray:
    for (x1, y1, x2, y2), d in zip(boxes_2d, depths):
        txt = "--" if np.isinf(d) else f"{d:4.1f} m"
        cv2.putText(img_bgr, txt,
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img_bgr




def main():
    
    configs = parse_demo_configs()

    configs.dataset_dir = "/home/airl010/1_Thesis/visionNav/fusion/dataset/2011_10_03_drive_0027_sync/"

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


    draw_boxes =False
    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            
            metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)
            lidar_xyz = metadatas['lidarData'][:, :3]          # drop reflectance

                        # kitti_dets produced exactly as before


            # print(metadatas['lidarData'])
            front_detections, front_bevmap, fps = do_detect(configs, model3d, front_bevmap, is_front=True)
            back_detections, back_bevmap, _ = do_detect(configs, model3d, back_bevmap, is_front=False)

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

            #RGB raw Image from the dataset
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            # cv2.imshow("img_bgr", img_bgr)






            # detections = model(img_bgr)
            # # Filter detections based on confidence and class indices
            # desired_classes = [0, 1, 2, 3, 5, 7]  # Only person, bicycle, car, motorcycle, bus, truck
            # confidence_threshold = 0.5
            # filtered_boxes = []
            # for box in detections[0].boxes.data.cpu().numpy():  # [x1, y1, x2, y2, confidence, class]
            #     confidence, cls = box[4], int(box[5])
            #     if confidence >= confidence_threshold and cls in desired_classes:
            #         filtered_boxes.append(box)
            
            # filtered_boxes = np.array(filtered_boxes)

            
            # # Draw boxes on the image
            # if draw_boxes:
            #     if len(filtered_boxes) > 0:
            #         for box in filtered_boxes:
            #             x1, y1, x2, y2, conf, cls = box
            #             label = f"{model.names[int(cls)]} {conf:.2f}"
            #             # Draw rectangle and label on the image
            #             # image = cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            #             image = cv2.putText(img_bgr, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
            #     else:
            #         print("No detections met the criteria.")
            




            calib = Calibration(configs.calib_path)
            kitti_dets = convert_det_to_real_values(front_detections)

            if len(kitti_dets) > 0:
                kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))
            
        


            out_img = np.concatenate((img_bgr, full_bev), axis=0)
            # cv2.putText(out_img, 'Speed: {:.1f} FPS'.format(fps), org=(900, 400), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1,  color = (255, 255, 255), thickness = 2)

            # # Create the video writer if not already created
            # if out_cap is None:
            #     out_cap_h, out_cap_w = out_img.shape[:2]
            #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #     out_path = os.path.join(configs.results_dir, '{}lid_cam_off_road.avi'.format(configs.foldername))
            #     print('Create video writer at {}'.format(out_path))
            #     out_cap = cv2.VideoWriter(out_path, fourcc, 15, (out_cap_w, out_cap_h))
            # Write the output frame to the video
            #out_cap.write(out_img)

            # DISPLAY IN REAL TIME
            cv2.imshow("Demo", out_img)
            key = cv2.waitKey(1) & 0xFF
            # If you want to stop early by pressing 'q'
            if key == ord('q'):
                break

    if out_cap:
        out_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()