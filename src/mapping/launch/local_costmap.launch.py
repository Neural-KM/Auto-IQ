"""
Local Costmap Launch File
=========================

Launches the inverse_perspective_mapper_v2 node which provides:
  - 2D occupancy grid from semantic segmentation (reverse projection IPM)
  - Road boundary extraction from local lane data mask
  - Lane marking polynomial fits from road_line pixels
  - Structured local lane data with lane centerlines

All local road estimation is integrated into the single IPM node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ==================== Arguments ====================
        DeclareLaunchArgument(
            'semantic_source', default_value='road_lane_extraction',
            description="'road_lane_extraction' (SegFormer + lane detection) or 'ground_truth' (CARLA sim)"
        ),
        DeclareLaunchArgument(
            'grid_size', default_value='80.0',
            description='Grid size in meters (square grid)'
        ),
        DeclareLaunchArgument(
            'resolution', default_value='0.4',
            description='Grid resolution in meters per cell'
        ),
        DeclareLaunchArgument(
            'mapper_executable',
            default_value='inverse_perspective_mapper_node_v2',
            description='Local costmap node executable to launch'
        ),

        # ==================== IPM Node (with integrated lane estimation) ====================
        # Publishes:
        #   /mapping/ipm_costmap (OccupancyGrid)
        #   /mapping/boundary/left, /mapping/boundary/right (Path)
        #   /mapping/local_lane_data (LocalLaneData with lane structure)
        #   /mapping/lane_detection (LaneDetection with polynomial fits)
        #   /mapping/lanes (MarkerArray for RViz)
        Node(
            package='mapping',
            executable=LaunchConfiguration('mapper_executable'),
            name='local_costmap_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'semantic_source': LaunchConfiguration('semantic_source'),
                'grid_size': LaunchConfiguration('grid_size'),
                'resolution': LaunchConfiguration('resolution'),
            }],
        ),
    ])
