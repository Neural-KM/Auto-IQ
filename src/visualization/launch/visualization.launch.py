"""Visualization Node Launch File"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ==================== Launch Arguments ====================

        # CARLA connection
        DeclareLaunchArgument('carla_host', default_value='localhost'),
        DeclareLaunchArgument('carla_port', default_value='2000'),

        # Node enable flags
        DeclareLaunchArgument('enable_semantic_colorizer', default_value='true'),
        DeclareLaunchArgument('enable_2d_viz', default_value='false'),
        DeclareLaunchArgument('enable_carla_viz', default_value='true'),
        DeclareLaunchArgument('enable_local_lane_data_viz', default_value='true'),
        # Comma-separated list of objects to draw in CARLA debug view
        # Options: trajectory, global_route, predicted_traj, ground_truth_obstacles, detected_obstacles, speed, sensors
        # Example: "trajectory,ground_truth_obstacles,speed"
        DeclareLaunchArgument('carla_debug_draw', default_value='speed, sensors'),

        # ==================== Node A: Semantic Colorizer ====================
        Node(
            package='visualization',
            executable='semantic_colorizer_node',
            name='semantic_colorizer_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_semantic_colorizer')),
            parameters=[{'use_sim_time': True}]
        ),

        # ==================== Node B: Live 2D Visualizer ====================
        Node(
            package='visualization',
            executable='live_2d_viz_node',
            name='live_2d_viz_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_2d_viz')),
            parameters=[{'use_sim_time': True}]
        ),

        # ==================== Node C: CARLA Debug Drawing ====================
        Node(
            package='visualization',
            executable='carla_viz_node',
            name='carla_viz_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_carla_viz')),
            parameters=[{
                'use_sim_time': True,
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_debug_draw': LaunchConfiguration('carla_debug_draw'),
            }]
        ),
        # ==================== Node D: Local Lane Data Visualizer ====================
        Node(
            package='visualization',
            executable='local_lane_data_viz_node',
            name='local_lane_data_viz_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_local_lane_data_viz')),
            parameters=[{
                'use_sim_time': True,
            }]
        ),
    ])
