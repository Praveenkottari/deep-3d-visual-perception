from setuptools import setup

package_name = 'vehicle_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='airl010',
    maintainer_email='kottaripraveen3004@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            "vehicle_detection_node=vehicle_detection.vehicle_detection_node:main",
        ],
    },
)
