from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'detect3d'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=['detect3d', 'detect3d.*']),
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools', 'numpy', 'opencv-python', 'torch', 'transforms3d'
    ],
    zip_safe=True,
    maintainer='praveen',
    maintainer_email='praveen@example.com',
    description='ROS2 node for 3D object detection',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detect3d_node = detect3d.detect3d_node:main',
        ],
    },
)
