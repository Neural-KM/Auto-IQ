"""
Perception Launch File
=======================

Launches the perception stack for pixel-level labeling:
  1. YOLO-based object detection + ground truth obstacles (perception_node)
     -- gated by `enable_perception_node` (default true). The node imports
        `carla` and reads the ground-truth obstacles from the simulator,
        so it must be disabled for real-hardware bringups.
  2. Semantic Segmentation (SegFormer or CARLA Ground Truth)
  3. Semantic PointCloud projection (gated by `enable_semantic_pointcloud`,
     default true). Needs LiDAR + a synced camera, so disable it on
     hardware bringups that run without a LiDAR.

Note: Lane polynomial fitting is handled by the mapping package's lane_fitting_node.

Usage:
    # CARLA simulation (default):
    ros2 launch perception perception.launch.py segmentation_source:=ground_truth

    # Real hardware, ZED X One front camera, segformer:
    ros2 launch perception perception.launch.py \
        segmentation_source:=segformer \
        rgb_topic:=/zed_1/zed_node/rgb/color/image \
        use_sim_time:=false \
        enable_perception_node:=false \
        enable_semantic_pointcloud:=false
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
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
            'use_sim_time', default_value='true', choices=['true', 'false'],
            description='Drive nodes off /clock instead of wall time. Set '
                        'false for real-hardware bringups.'
        ),
        DeclareLaunchArgument(
            'carla_host', default_value='localhost',
            description='CARLA simulator host'
        ),
        DeclareLaunchArgument(
            'carla_port', default_value='2000',
            description='CARLA simulator port'
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

        # ----- Topic + frame overrides for the segmentation node -----
        DeclareLaunchArgument(
            'rgb_topic',
            default_value='/carla/ego_vehicle/sensor/camera/rgb/front/image',
            description='RGB image topic that segformer subscribes to. For '
                        'the hardware rig set to e.g. '
                        '/zed_1/zed_node/rgb/color/image'),
        DeclareLaunchArgument(
            'ground_truth_topic',
            default_value='/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
            description='Ground-truth semantic image topic (CARLA only).'),

        # ----- Gates for sim-only / lidar-only nodes -----
        DeclareLaunchArgument(
            'enable_perception_node', default_value='true',
            choices=['true', 'false'],
            description='Run the YOLO + CARLA-ground-truth perception_node. '
                        'Disable on real hardware (it imports `carla`).'),
        DeclareLaunchArgument(
            'enable_semantic_pointcloud', default_value='true',
            choices=['true', 'false'],
            description='Run the LiDAR-image fusion semantic_pointcloud_node. '
                        'Disable on hardware bringups without a LiDAR.'),

        # ==================== Node 1: YOLO Object Detection ====================
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_perception_node')),
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'carla_host': LaunchConfiguration('carla_host'),
                    'carla_port': LaunchConfiguration('carla_port'),
                }
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
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'segmentation_source': LaunchConfiguration('segmentation_source'),
                'model_name': LaunchConfiguration('model_name'),
                'device': LaunchConfiguration('device'),
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'ground_truth_topic': LaunchConfiguration('ground_truth_topic'),
            }],
        ),

        # ==================== Node 3: Semantic PointCloud ====================
        Node(
            package='perception',
            executable='semantic_pointcloud_node',
            name='semantic_pointcloud_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_semantic_pointcloud')),
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
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
