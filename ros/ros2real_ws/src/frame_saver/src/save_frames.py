#!/usr/bin/env python
import os
import rospy
import cv2
import struct
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

save_dir = "/home/airl010/1_Thesis/rosbag/data/unsync"
image_dir = os.path.join(save_dir, "images")
lidar_dir = os.path.join(save_dir, "lidar")

os.makedirs(image_dir, exist_ok=True)
os.makedirs(lidar_dir, exist_ok=True)

bridge = CvBridge()
frame_id = 0

def image_callback(msg):
    global frame_id
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        filename = os.path.join(image_dir, f"{frame_id:010d}.png")
        cv2.imwrite(filename, cv_image)
    except Exception as e:
        rospy.logerr(f"Image conversion error: {e}")

def lidar_callback(msg):
    global frame_id
    points = []
    for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
        points.append([p[0], p[1], p[2], p[3]])
    points = np.array(points, dtype=np.float32)
    filename = os.path.join(lidar_dir, f"{frame_id:010d}.bin")
    points.tofile(filename)
    frame_id += 1

def main():
    rospy.init_node("extract_lidar_and_image_saver", anonymous=True)
    rospy.Subscriber("/camera/image_color", Image, image_callback, queue_size=1)
    rospy.Subscriber("/velodyne_points", PointCloud2, lidar_callback, queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    main()
