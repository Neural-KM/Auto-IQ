"""
Mapping Launch File
===================

Launches the mapping nodes:

1. inverse_perspective_mapper_v2 - Provides:
   - 2D occupancy grid from semantic segmentation (reverse projection IPM)

2. lane_fitting_node - Provides:
   - Multi-lane polynomial fitting from road_line pixels
   - Boundary-based local lane estimation from occupancy grid
   - RViz lane markers for visualization
   - EstimatedLanes with estimated lane structure

All sensor topics + frames are launch arguments so the same nodes can run
against either the CARLA bridge (defaults) or the on-vehicle ZED X One
(`hardware.launch.py` overrides them).
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression


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

        # ----- Frames + topics (override for hardware) -----
        DeclareLaunchArgument(
            'use_sim_time', default_value='true', choices=['true', 'false'],
            description='Drive nodes off /clock instead of wall time. Set '
                        'false for real-hardware bringups.'),
        DeclareLaunchArgument(
            'base_frame', default_value='ego_vehicle',
            description='Robot base frame. CARLA bridge: ego_vehicle. '
                        'Hardware (sensors_bringup URDF): base_link.'),
        DeclareLaunchArgument(
            'camera_frame', default_value='ego_vehicle/sensor/camera/rgb/front',
            description='Camera optical frame (must match what publishes '
                        'the camera_info topic).'),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
            description='CameraInfo topic for the front camera.'),
        DeclareLaunchArgument(
            'ground_truth_semantic_topic',
            default_value='/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
            description='Semantic image topic when semantic_source=ground_truth.'),

        # ==================== IPM Node (Occupancy Grid) ====================
        # Publishes:
        #   /mapping/ipm_costmap (OccupancyGrid)
        #   /mapping/debug/drop_skirt (Image)
        Node(
            package='mapping',
            executable='inverse_perspective_mapper_node_v2',
            name='inverse_perspective_mapper_node',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'semantic_source': LaunchConfiguration('semantic_source'),
                'grid_size': LaunchConfiguration('grid_size'),
                'resolution': LaunchConfiguration('resolution'),
                'base_frame': LaunchConfiguration('base_frame'),
                'camera_frame': LaunchConfiguration('camera_frame'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'ground_truth_semantic_topic': LaunchConfiguration('ground_truth_semantic_topic'),
            }],
        ),

        # ==================== Lane Fitting Node ====================
        # Publishes:
        #   /mapping/lane_detection (LaneDetection with polynomial fits)
        #   /mapping/lanes (MarkerArray for RViz)
        #   /mapping/debug/lane_bev (Image, debug visualization)
        #   /mapping/estimated_lanes (EstimatedLanes with lane structure)
        #   /mapping/boundary_markers (MarkerArray for drivable boundary viz)
        #   /mapping/debug/boundary (Image)
        Node(
            package='mapping',
            executable='lane_fitting_node',
            name='lane_fitting_node',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'semantic_source': LaunchConfiguration('semantic_source'),
                'enable_debug_viz': PythonExpression(["'", LaunchConfiguration('enable_lane_debug_viz'), "'.lower() == 'true'"]),
                'base_frame': LaunchConfiguration('base_frame'),
                'camera_frame': LaunchConfiguration('camera_frame'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'ground_truth_semantic_topic': LaunchConfiguration('ground_truth_semantic_topic'),
            }],
        ),
    ])
