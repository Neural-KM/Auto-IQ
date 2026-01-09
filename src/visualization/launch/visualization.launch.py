"""Visualization Node Launch File"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('carla_host', default_value='localhost'),
        DeclareLaunchArgument('carla_port', default_value='2000'),
        DeclareLaunchArgument('carla_timeout', default_value='10.0'),
        DeclareLaunchArgument('enable_2d_viz', default_value='false'),
        
        # Comma-separated list of objects to draw in CARLA debug view
        # Options: trajectory, global_route, predicted, ground_truth_obstacles, detected_obstacles, speed
        # Example: "trajectory,ground_truth_obstacles,speed"
        # Empty string disables CARLA debug drawing entirely
        DeclareLaunchArgument('carla_debug_draw', default_value='speed'),

        Node(
            package='visualization',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
            parameters=[{
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
                'enable_2d_viz': LaunchConfiguration('enable_2d_viz'),
                'carla_debug_draw': LaunchConfiguration('carla_debug_draw'),
            }]
        ),
    ])
