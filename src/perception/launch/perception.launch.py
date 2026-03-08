"""
Perception Launch File
=======================

Launches the full perception stack:
  1. YOLO-based object detection + ground truth obstacles (perception_node)
  2. Semantic Segmentation (SegFormer or CARLA Ground Truth)
  3. Lane Detection (polynomial fits on semantic segmentation)
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
            'model_name', default_value='b2',
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

        # ==================== Node 2: Semantic Segmentation + Road Lane Extraction =============
        # Publishes: /perception/semantic_map (mono8, class IDs)
        #            /perception/semantic_map/colored (rgb8)
        #            /perception/semantic_map/road_lane_extraction (mono8, with lane markings)
        #            /perception/semantic_map/road_lane_extraction/colored (rgb8)
        #            /perception/road_mask, /perception/lane_markings (debug)
        # Note: SegFormer does NOT natively produce road_line (class 24);
        #       the integrated road lane extraction step detects and injects them.
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
            }],
        ),

        # ==================== Node 3: Lane Detection (polynomial fits) ====================
        # Subscribes to the semantic segmentation output and fits lane polynomials.
        # Publishes: /perception/lane_detection  (LaneDetection msg)
        #            /perception/lanes           (MarkerArray for RViz)
        #            /perception/debug/lane_bev  (debug BEV image)
        Node(
            package='perception',
            executable='lane_detection_node',
            name='lane_detection_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'semantic_topic': '/perception/semantic_map/road_lane_extraction',
                'camera_info_topic': '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
                'enable_debug_viz': False,
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
