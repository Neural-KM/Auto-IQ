import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'visualization'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skully',
    maintainer_email='boliyadeep2003@gmail.com',
    description='Visualization package for Auto-IQ',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'visualization_node = visualization.visualization_node:main',
            'semantic_colorizer_node = visualization.semantic_colorizer_node:main',
            'live_2d_viz_node = visualization.live_2d_viz_node:main',
            'carla_viz_node = visualization.carla_viz_node:main',
            'lidar_camera_overlay_node = visualization.lidar_camera_overlay_node:main',
        ],
    },
)
