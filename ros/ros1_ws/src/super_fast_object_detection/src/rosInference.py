#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import timeit
import warnings
import sys
import os
import torch
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)

# SFA3D imports
sys.path.append('./')
from sfa.models.model_utils import create_model
from sfa.utils.evaluation_utils import draw_predictions, convert_det_to_real_values
from sfa.data_process.transformation import lidar_to_camera_box
from sfa.utils.visualization_utils import merge_rgb_to_bev, show_rgb_image_with_boxes
from sfa.data_process.kitti_data_utils import Calibration
from sfa.utils.demo_utils import parse_demo_configs, do_detect, download_and_unzip, write_credit
from sfa.data_process.kitti_bev_utils import makeBEVMap
import sfa.config.kitti_config as cnf
from sfa.data_process.kitti_data_utils import get_filtered_lidar

# Class mapping
ID_TO_CLASS_NAME = {
    0: 'pedestrian',
    1: 'car',
    2: 'cyclist',
    -3: 'truck',
    -99: 'tram',
    -1: 'unknown'
}

# Quaternion conversion
def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

# Scan callback
def on_scan(scan):
    start = timeit.default_timer()
    rospy.loginfo("Got scan")

    gen = [np.array([p[0], p[1], p[2], p[3]/100.0]) for p in pc2.read_points(scan, field_names=("x", "y", "z", "intensity"), skip_nans=True)]
    gen_numpy = np.array(gen, dtype=np.float32)

    front_lidar = get_filtered_lidar(gen_numpy, cnf.boundary)
    bev_map = makeBEVMap(front_lidar, cnf.boundary)
    bev_map = torch.from_numpy(bev_map)

    with torch.no_grad():
        detections, bev_map, fps = do_detect(configs, model, bev_map, is_front=True)

    print("FPS:", fps)

    # DetectedObjectArray msg
    objects_msg = DetectedObjectArray()
    objects_msg.header.stamp = rospy.Time.now()
    objects_msg.header.frame_id = scan.header.frame_id

    # MarkerArray for RViz
    marker_array = MarkerArray()

    obj_id = 0
    for j in range(configs.num_classes):
        class_name = ID_TO_CLASS_NAME[j]

        for det in detections[j]:
            _score, _x, _y, _z, _h, _w, _l, _yaw = det
            yaw = -_yaw
            x = _y / cnf.BEV_HEIGHT * cnf.bound_size_x + cnf.boundary['minX']
            y = _x / cnf.BEV_WIDTH * cnf.bound_size_y + cnf.boundary['minY']
            z = _z + cnf.boundary['minZ']
            w = _w / cnf.BEV_WIDTH * cnf.bound_size_y
            l = _l / cnf.BEV_HEIGHT * cnf.bound_size_x

            # Fill DetectedObject
            obj = DetectedObject()
            obj.header.stamp = rospy.Time.now()
            obj.header.frame_id = scan.header.frame_id
            obj.space_frame = scan.header.frame_id
            obj.label = class_name
            obj.score = _score
            obj.pose_reliable = True
            obj.pose.position.x = x
            obj.pose.position.y = y
            obj.pose.position.z = z
            qx, qy, qz, qw = euler_to_quaternion(yaw, 0, 0)
            obj.pose.orientation.x = qx
            obj.pose.orientation.y = qy
            obj.pose.orientation.z = qz
            obj.pose.orientation.w = qw
            obj.dimensions.x = l
            obj.dimensions.y = w
            obj.dimensions.z = _h
            objects_msg.objects.append(obj)

            # Fill Marker for RViz
            marker = Marker()
            marker.header = obj.header
            marker.ns = "detected"
            marker.id = obj_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = obj.pose
            marker.scale.x = obj.dimensions.x
            marker.scale.y = obj.dimensions.y
            marker.scale.z = obj.dimensions.z
            marker.color.a = 0.6
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
            obj_id += 1

            # Add optional text label
            text_marker = Marker()
            text_marker.header = obj.header
            text_marker.ns = "labels"
            text_marker.id = obj_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = obj.pose
            text_marker.pose.position.z += obj.dimensions.z + 0.5
            text_marker.text = obj.label
            text_marker.scale.z = 0.6
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            marker_array.markers.append(text_marker)
            obj_id += 1

    # Publish
    if objects_msg.objects:
        pub.publish(objects_msg)
        marker_pub.publish(marker_array)

    stop = timeit.default_timer()
    print('Callback Time: {:.2f}s'.format(stop - start))

# Main
if __name__ == '__main__':
    print("Started Node")
    rospy.init_node('SuperFastObjectDetection', anonymous=True)

    # Setup model
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('super_fast_object_detection')
    configs = parse_demo_configs()
    configs.pretrained_path = os.path.join(package_path, 'fpn_resnet_18', 'fpn_resnet_18_epoch_300.pth')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

    model = create_model(configs)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cuda:0'))
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(configs.device)
    model.eval()

    # Publishers
    pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=10)
    marker_pub = rospy.Publisher('detected_objects_marker', MarkerArray, queue_size=10)

    # Subscriber
    rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, on_scan)
    rospy.spin()
