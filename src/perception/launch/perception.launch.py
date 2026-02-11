"""
Perception Launch File
=======================

Launches the full perception stack:
  1. YOLO-based object detection (perception_node)
  2. Semantic Segmentation (SegFormer or CARLA Ground Truth)
  3. Road Extraction + Semantic Fusion (lane marking injection)
  4. Semantic PointCloud projection

Usage:
    ros2 launch perception perception.launch.py segmentation_source:=segformer
    ros2 launch perception perception.launch.py segmentation_source:=ground_truth
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
        # ==================== Arguments ====================
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params_file,
            description='Path to perception parameters YAML file'
        ),
        DeclareLaunchArgument(
            'segmentation_source', default_value='segformer',
            description="'segformer' (camera-based) or 'ground_truth' (CARLA sim)"
        ),
        DeclareLaunchArgument(
            'model_name', default_value='b5',
            description='SegFormer model variant: b0-b5'
        ),
        DeclareLaunchArgument(
            'device', default_value='cuda',
            description='SegFormer inference device: cuda or cpu'
        ),

        # ==================== Node 1: YOLO Object Detection ====================
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

        # ==================== Node 2: Semantic Segmentation ====================
        # Publishes: /perception/semantic_map (mono8, class IDs)
        #            /perception/semantic_map/colored (rgb8)
        # Note: SegFormer does NOT produce road_line (class 24)
        Node(
            package='perception',
            executable='semantic_segmentation_node',
            name='semantic_segmentation_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'segmentation_source': LaunchConfiguration('segmentation_source'),
                'model_name': LaunchConfiguration('model_name'),
                'device': LaunchConfiguration('device'),
                'publish_colored': True,
                'rgb_topic': '/carla/ego_vehicle/sensor/camera/rgb/front/image',
                'ground_truth_topic': '/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
                'output_topic': '/perception/semantic_map',
            }],
        ),

        # ==================== Node 3: Road Extraction + Semantic Fusion ====================
        # Takes semantic map (mono8) + RGB camera image.
        # Detects lane markings via adaptive thresholding within the road mask,
        # then writes class 24 (road_line) into the semantic map.
        #
        # Subscribes: /perception/semantic_map (mono8) + RGB camera
        # Publishes:  /perception/semantic_map/fused (mono8) — ready for mapping
        #             /perception/semantic_map/fused/colored (rgb8)
        #             /perception/road_mask, /perception/lane_markings (debug)
        #             /perception/lane_markings/ipm (optional IPM viz)
        Node(
            package='perception',
            executable='road_extraction_node',
            name='road_extraction_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'semantic_topic': '/perception/semantic_map',
                'rgb_topic': '/carla/ego_vehicle/sensor/camera/rgb/front/image',
                'output_topic': '/perception/semantic_map/fused',
                'publish_colored': True,
                'enable_ipm': True,
                'publish_overlay': True,
                'publish_ipm': True,
                'fov': 110.0,
                'pitch': 0.0,
                'camera_height': 1.2,
                'ipm_x_min': 3.0,
                'ipm_x_max': 30.0,
                'ipm_y_min': -10.0,
                'ipm_y_max': 10.0,
                'pixels_per_meter': 30.0,
            }],
        ),

        # ==================== Node 4: Semantic PointCloud ====================
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
