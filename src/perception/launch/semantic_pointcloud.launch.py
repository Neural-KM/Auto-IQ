"""Semantic PointCloud Node Launch File"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ==================== Launch Arguments ====================
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/carla/ego_vehicle/sensor/lidar/top/point_cloud',
            description='LiDAR point cloud topic'
        ),
        DeclareLaunchArgument(
            'semantic_topic',
            default_value='/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
            description='Semantic segmentation image topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
            description='Camera info topic for intrinsics'
        ),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/perception/semantic_pointcloud',
            description='Output colored point cloud topic'
        ),
        DeclareLaunchArgument(
            'max_range',
            default_value='100.0',
            description='Maximum range for points (meters)'
        ),
        
        # ==================== Semantic PointCloud Node ====================
        Node(
            package='perception',
            executable='semantic_pointcloud_node',
            name='semantic_pointcloud_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'lidar_topic': LaunchConfiguration('lidar_topic'),
                'semantic_topic': LaunchConfiguration('semantic_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
                'max_range': LaunchConfiguration('max_range'),
                'camera_frame': 'ego_vehicle/sensor/camera/rgb/front',
                'lidar_frame': 'ego_vehicle/sensor/lidar/top',
            }]
        ),
    ])
