from setuptools import setup
from setuptools import find_packages

package_name = 'kitti_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','pytest'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='ROS2 node for KITTI detection with YOLOv8 and LiDAR overlay',
    license='MIT',
    entry_points={
        'console_scripts': [
            'kitti_detection_node = kitti_detection.kitti_detection_node:main',
        ],
    },
)
