#!/usr/bin/env python
import rospy
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import message_filters

# Set save directories
save_dir = "/home/airl010/1_Thesis/rosbag/data/synch"
image_dir = os.path.join(save_dir, "images")
lidar_dir = os.path.join(save_dir, "lidar")

os.makedirs(image_dir, exist_ok=True)
os.makedirs(lidar_dir, exist_ok=True)

bridge = CvBridge()
frame_id = 1  # Start from 0000000001

def callback(image_msg, lidar_msg):
    global frame_id

    # Convert image
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        image_path = os.path.join(image_dir, f"{frame_id:010d}.png")
        cv2.imwrite(image_path, cv_image)
    except Exception as e:
        rospy.logerr(f"Image error: {e}")
        return

    # Convert point cloud to KITTI .bin format (x, y, z, intensity)
    try:
        points = []
        for p in pc2.read_points(lidar_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            points.append([p[0], p[1], p[2], p[3]])
        points = np.array(points, dtype=np.float32)
        lidar_path = os.path.join(lidar_dir, f"{frame_id:010d}.bin")
        points.tofile(lidar_path)
    except Exception as e:
        rospy.logerr(f"LiDAR error: {e}")
        return

    frame_id += 1

def main():
    rospy.init_node('save_synced_frames', anonymous=True)

    image_sub = message_filters.Subscriber("/camera/image_color", Image)
    lidar_sub = message_filters.Subscriber("/velodyne_points", PointCloud2)

    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, lidar_sub],
        queue_size=20,
        slop=0.05,  # Max time difference between image and lidar
        allow_headerless=False
    )
    ats.registerCallback(callback)

    rospy.spin()

if __name__ == "__main__":
    main()
