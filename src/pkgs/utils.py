import numpy as np
import cv2
from PIL import Image
from pkgs.kitti_utils import *
from models.detection_head import *
from BEV.bev import *
from pkgs.kitti_detection_utils import *
def get_uvz_centers(image, velo_uvz, bboxes, draw=True):
    ''' Inputs:
          image  
          velo_uvz 
          bboxes 
          draw 
        Outputs:
          bboxes_out
        '''

    # unpack LiDAR camera coordinates
    u, v, z = velo_uvz

    # get new output
    bboxes_out = np.zeros((bboxes.shape[0], bboxes.shape[1] + 3))
    bboxes_out[:, :bboxes.shape[1]] = bboxes

    # iterate through all detected bounding boxes
    for i, bbox in enumerate(bboxes):
        pt1 = np.round(bbox[0:2]).astype(int)
        pt2 = np.round(bbox[2:4]).astype(int)

        # get center location of the object on the image
        obj_x_center = (pt1[1] + pt2[1]) / 2
        obj_y_center = (pt1[0] + pt2[0]) / 2

        # now get the closest LiDAR points to the center
        center_delta = np.abs(np.array((v, u)) 
                              - np.array([[obj_x_center, obj_y_center]]).T)
        
        # choose coordinate pair with the smallest L2 norm
        min_loc = np.argmin(np.linalg.norm(center_delta, axis=0))

        # get LiDAR location in image/camera space
        velo_depth = z[min_loc]; # LiDAR depth in camera space
        uvz_location = np.array([u[min_loc], v[min_loc], velo_depth])
        
        # add velo projections (u, v, z) to bboxes_out
        bboxes_out[i, -3:] = uvz_location

        # draw depth on image at center of each bounding box
        # This is depth as perceived by the camera
        if draw:
            object_center = (np.round(obj_y_center).astype(int), 
                             np.round(obj_x_center).astype(int))
            cv2.putText(image, 
                        '{0:.2f} m'.format(velo_depth), 
                        object_center, # top left
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, # font scale
                        (255, 0, 0), 2, cv2.LINE_AA)    
            
    return bboxes_out
    

def draw_bboxes_on_lidar_image(
    lidar_image, 
    bboxes, 
    color=(0, 255, 0),    # (B, G, R) => red bounding boxes
    thickness=2
):
    
    for bbox in bboxes:
        # Unpack the bounding box fields 
        # (Adjust indices if your format is slightly different)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
      
        
        # Draw bounding box
        cv2.rectangle(
            lidar_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=color,
            thickness=thickness
        )
    return lidar_image

### video function from series of images

get_total_seconds = lambda hms: hms[0]*60*60 + hms[1]*60 + hms[2]


def timestamps2seconds(timestamp_path):
    ''' Reads in timestamp path and returns total seconds (does not account for day rollover '''
    timestamps = pd.read_csv(timestamp_path, 
                             header=None) \
                             .squeeze(axis=1).astype(object) \
                                          .apply(lambda x: x.split(' ')[1]) 
    
    # Get Hours, Minutes, and Seconds
    hours = timestamps.apply(lambda x: x.split(':')[0]).astype(np.float64)
    minutes = timestamps.apply(lambda x: x.split(':')[1]).astype(np.float64)
    seconds = timestamps.apply(lambda x: x.split(':')[2]).astype(np.float64)

    hms_vals = np.vstack((hours, minutes, seconds)).T
    
    total_seconds = np.array(list(map(get_total_seconds, hms_vals)))
    
    return total_seconds


result_video = []

def input_to_video(model,DATA_PATH,left_image_paths,lid_paths,T_cam2_velo,T_velo_cam2):
    cam2_total_seconds = timestamps2seconds(os.path.join(DATA_PATH, r'image_02/timestamps.txt'))
    cam2_fps = 1/np.median(np.diff(cam2_total_seconds))

    for index in range(len(left_image_paths)-1):
        left_image = cv2.cvtColor(cv2.imread(left_image_paths[index]), cv2.COLOR_BGR2RGB)
        bin_path = lid_paths[index]
        # oxts_frame = get_oxts(oxts_paths[index])

        # get detections and object centers in uvz
        bboxes, velo_uvz = get_detection_coordinates(left_image, bin_path,model,T_velo_cam2, remove_plane=False)

        # get transformed coordinates
        uvz = bboxes[:, -3:]


        # draw velo on blank image
        velo_image = draw_velo_on_image(velo_uvz, np.zeros_like(left_image))

        # stack frames
        stacked = np.vstack((left_image, velo_image))
        stacked_h, stacked_w, _ = stacked.shape
        if stacked_h != canvas_height:
            # scale width proportionally
            new_width = int((stacked_w / stacked_h) * canvas_height)
            stacked = cv2.resize(stacked, (new_width, canvas_height))


        # draw top down scenario on canvas
        canvas = draw_scenario(uvz,T_cam2_velo)

        # place everything in a single frame
        frame = np.hstack((stacked, 
                        255*np.ones((canvas_height, 1, 3), dtype=np.uint8),
                        canvas))
        vid_show = Image.fromarray(frame)
        vid_show.show()

        # add to result video
        result_video.append(frame)

        h, w, _ = frame.shape


    return result_video,cam2_fps,h,w