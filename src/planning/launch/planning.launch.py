"""Planning Node Launch File"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('carla_host', default_value='localhost'),
        DeclareLaunchArgument('carla_port', default_value='2000'),
        DeclareLaunchArgument('carla_timeout', default_value='10.0'),
        DeclareLaunchArgument('target_speed', default_value='40.0'),
        DeclareLaunchArgument('use_ground_truth', default_value='true'),
        
        Node(
            package='planning',
            executable='planning_node',
            name='planning_node',
            output='screen',
            parameters=[{
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
                'target_speed': LaunchConfiguration('target_speed'),
                'use_ground_truth': LaunchConfiguration('use_ground_truth'),
            }]
        ),
    ])
