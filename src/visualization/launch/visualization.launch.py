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
        DeclareLaunchArgument('enable_carla_viz', default_value='true'),
        DeclareLaunchArgument('enable_2d_viz', default_value='false'),
        DeclareLaunchArgument('draw_ground_truth_obstacles', default_value='true'),
        DeclareLaunchArgument('draw_detected_obstacles', default_value='false'),
        
        Node(
            package='visualization',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
            parameters=[{
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
                'enable_carla_viz': LaunchConfiguration('enable_carla_viz'),
                'enable_2d_viz': LaunchConfiguration('enable_2d_viz'),
                'draw_ground_truth_obstacles': LaunchConfiguration('draw_ground_truth_obstacles'),
                'draw_detected_obstacles': LaunchConfiguration('draw_detected_obstacles'),
            }]
        ),
    ])
