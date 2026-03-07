import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skully',
    maintainer_email='boliyadeep2003@gmail.com',
    description='Perception package for AutoIQ - Object and lane detection',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'perception_node = perception.nodes.perception_node:main',
            'semantic_segmentation_node = perception.nodes.semantic_segmentation_node:main',
            'semantic_pointcloud_node = perception.nodes.semantic_pointcloud_node:main',
            'lane_detection_node = perception.nodes.lane_detection_node:main',
            'save_camera_snapshot = perception.tools.save_camera_snapshot:main',
        ],
    },
)
