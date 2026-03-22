"""
Mapping Launch File
===================

Launches the mapping nodes:

1. inverse_perspective_mapper_v2 - Provides:
   - 2D occupancy grid from semantic segmentation (reverse projection IPM)
   - Road boundary extraction from local lane data mask
   - Structured local lane data with lane centerlines

2. lane_fitting_node - Provides:
   - Multi-lane polynomial fitting from road_line pixels
   - RViz lane markers for visualization
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
            'enable_lane_debug_viz', default_value='true',
            description='Enable lane fitting debug BEV image publishing'
        ),

        # ==================== IPM Node (with local lane estimation) ====================
        # Publishes:
        #   /mapping/ipm_costmap (OccupancyGrid)
        #   /mapping/boundary/left, /mapping/boundary/right (Path)
        #   /mapping/local_lane_data (LocalLaneData with lane structure)
        Node(
            package='mapping',
            executable='inverse_perspective_mapper_node_v2',
            name='inverse_perspective_mapper_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'semantic_source': LaunchConfiguration('semantic_source'),
                'grid_size': LaunchConfiguration('grid_size'),
                'resolution': LaunchConfiguration('resolution'),
            }],
        ),

        # ==================== Lane Fitting Node ====================
        # Publishes:
        #   /mapping/lane_detection (LaneDetection with polynomial fits)
        #   /mapping/lanes (MarkerArray for RViz)
        #   /mapping/debug/lane_bev (Image, debug visualization)
        Node(
            package='mapping',
            executable='lane_fitting_node',
            name='lane_fitting_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'semantic_source': LaunchConfiguration('semantic_source'),
                'enable_debug_viz': LaunchConfiguration('enable_lane_debug_viz'),
            }],
        ),
    ])
