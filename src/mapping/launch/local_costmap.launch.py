"""
Local Costmap Launch File
=========================

Launches the Rolling Window Binary Bayes Filter costmap node with configurable parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ==================== Arguments ====================
        # DeclareLaunchArgument(
        #     'pointcloud_topic',
        #     # default_value='/perception/semantic_pointcloud',
        #     default_value='/carla/ego_vehicle/ground_truth/lidar/top_semantic/point_cloud',
        #     description='Input labeled point cloud topic (e.g., semantic LiDAR or camera-projected cloud)'
        # ),
        # DeclareLaunchArgument(
        #     'odom_topic',
        #     default_value='/carla/ego_vehicle/ground_truth/odometry',
        #     description='Input odometry topic'
        # ),
        # DeclareLaunchArgument(
        #     'costmap_topic',
        #     default_value='/mapping/local_costmap',
        #     description='Output occupancy grid topic'
        # ),
        # DeclareLaunchArgument(
        #     'odom_frame',
        #     default_value='map',
        #     description='World-fixed reference frame (map frame in CARLA)'
        # ),
        DeclareLaunchArgument(
            'grid_size',
            default_value='80.0',
            description='Grid size in meters (square grid)'
        ),
        DeclareLaunchArgument(
            'resolution',
            default_value='0.4',
            description='Grid resolution in meters per cell'
        ),
        # DeclareLaunchArgument(
        #     'l_hit',
        #     default_value='2.0',
        #     description='Log-odds value added for occupied cells'
        # ),
        # DeclareLaunchArgument(
        #     'l_miss',
        #     default_value='0.7',
        #     description='Log-odds value subtracted for free cells'
        # ),
        # DeclareLaunchArgument(
        #     'enable_raycasting',
        #     default_value='true',
        #     description='Enable Bresenham raycasting for free space'
        # ),
        DeclareLaunchArgument(
            'mapper_executable',
            default_value='inverse_perspective_mapper_node',
            description='Local costmap node executable to launch'
        ),
        # ==================== Node ====================
        Node(
            package='mapping',
            executable=LaunchConfiguration('mapper_executable'),
            name='local_costmap_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,  # Use simulation time from /clock
                'grid_size': LaunchConfiguration('grid_size'),
                'resolution': LaunchConfiguration('resolution'),
            }]
            # parameters=[{
            #     'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            #     'odom_topic': LaunchConfiguration('odom_topic'),
            #     'costmap_topic': LaunchConfiguration('costmap_topic'),
            #     'odom_frame': LaunchConfiguration('odom_frame'),
            #     'base_link_frame': 'ego_vehicle',

            #     'min_height': -0.3,
            #     'max_height': 2.5,
            #     'l_hit': LaunchConfiguration('l_hit'),
            #     'l_miss': LaunchConfiguration('l_miss'),
            #     'l_min': -5.0,
            #     'l_max': 5.0,
            #     'l_occ_threshold': 0.0,
            #     'max_ray_range': 30.0,
            #     'enable_raycasting': LaunchConfiguration('enable_raycasting'),
            # }]
        ),
    ])
