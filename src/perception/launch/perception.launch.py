"""
Perception Launch File

All default parameters are defined in config/perception_params.yaml (single source of truth).
Override via command line: ros2 launch perception perception.launch.py params_file:=/path/to/custom.yaml
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package share directory for config file
    pkg_dir = get_package_share_directory('perception')
    default_params_file = os.path.join(pkg_dir, 'config', 'perception_params.yaml')
    
    return LaunchDescription([
        # Single argument to override the entire config file if needed
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to perception parameters YAML file'
        ),
        
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'use_sim_time': True}
            ]
        ),
        
        # ==================== Semantic PointCloud Node ====================
        Node(
            package='perception',
            executable='semantic_pointcloud_node',
            name='semantic_pointcloud_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'lidar_topic': '/carla/ego_vehicle/sensor/lidar/top/point_cloud',
                'semantic_topic': '/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
                'camera_info_topic': '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
                'output_topic': '/perception/semantic_pointcloud',
                'max_range': 100.0,
                'camera_frame': 'ego_vehicle/sensor/camera/rgb/front',
                'lidar_frame': 'ego_vehicle/sensor/lidar/top',
            }]
        ),
    ])
