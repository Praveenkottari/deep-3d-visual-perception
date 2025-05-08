import cv2
import numpy as np
from PIL import Image
import math

from pkgs.kitti_utils import *

canvas_height = 620
canvas_width = 376

# Get consistent center for ego vehicle
ego_center = (canvas_width // 2, int(canvas_height * 0.95))

# Get rectangle coordinates for ego vehicle
ego_x1 = ego_center[0] - 5
ego_y1 = ego_center[1] - 10
ego_x2 = ego_center[0] + 5
ego_y2 = ego_center[1] + 10

def draw_scenario(uvz, T_cam2_velo, sf=1,line_draw=True):
    """
    uvz: N x 3 array of [u, v, z], where:
         (u, v) are image coordinates, 
          z is the depth (in meters) from the camera to the object.
    T_cam2_velo: 4x4 transformation matrix to transform [u, v, z] (in camera coords)
                 into [x, y, z] in LiDAR coordinates (if needed).
    sf: scaling factor for how many pixels each meter should be
    """

    # Transform uvz to LiDAR coordinates if needed
    velo_xyz = transform_uvz(uvz, T_cam2_velo)  # shape: N x 3

    # Create a blank canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Draw ego vehicle (blue rectangle + black center)
    cv2.rectangle(canvas, (ego_x1, ego_y1), (ego_x2, ego_y2), (255, 0, 0), -1)
    cv2.circle(canvas, ((ego_x1 + ego_x2) // 2, (ego_y1 + ego_y2) // 2),
               3, (0, 0, 0), -1)

    # Draw detected objects
    for i, (vel) in enumerate(velo_xyz):
        # vel: [x, y, z] in LiDAR coordinates (after transformation)
        # depth from uvz (camera coordinates) is uvz[i, 2]
        depth_value = uvz[i, 2]  # in meters (z in camera frame)

        # Convert LiDAR coords (x, y) to top-down canvas
        # val[0] = x, val[1] = y in LiDAR frame
        obj_center = (
            ego_center[0] - sf * int(round(vel[1])),
            ego_center[1] - sf * int(round(vel[0]))
        )

        # Draw object (rectangle + small circle)
        obj_x1 = obj_center[0] - 5
        obj_y1 = obj_center[1] - 10
        obj_x2 = obj_center[0] + 5
        obj_y2 = obj_center[1] + 10

        cv2.rectangle(canvas, (obj_x1, obj_y1), (obj_x2, obj_y2), (0, 255, 0), -1)
        cv2.circle(canvas, obj_center, 3, (0, 0, 0), -1)

        if line_draw:
            # 1) Draw a thin line from ego_center to object center
            cv2.line(canvas, ego_center, obj_center, (0, 255, 255), 1)

            # 2) Place the depth annotation near the object (e.g., just above it)
            #    Using the z-value from uvz as the "distance" in meters
            text_x = obj_center[0]
            text_y = obj_y1 - 5  # slightly above the object rectangle

            cv2.putText(
                canvas,
                f"{depth_value:.1f}m",  # e.g., "10.3m"
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,               # font scale
                (0, 255, 255),     # color: yellow-ish
                1,                 # thickness
                cv2.LINE_AA
            )

    return canvas

if __name__ == "__main__":
    #  uvz array: each row is [u, v, z]
    # Suppose we have 4 objects with known pixel coords & depths
    uvz_sample = np.array([
        [389.1, 235.81, 11.076],
        [764.76, 239.46, 10.354],
        [290.0, 211.96, 17.51],
        [694.49, 207.36, 21.393]
    ], dtype=np.float32)

    # Example identity transform for testing (no real transformation)
    T_cam2_velo_test = np.eye(4)

    # Draw scenario
    canvas_out = draw_scenario(uvz_sample, T_cam2_velo_test)

    # Display the result
    Image.fromarray(canvas_out).show()
