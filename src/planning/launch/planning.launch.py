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
        DeclareLaunchArgument('target_speed', default_value='30.0'),
        DeclareLaunchArgument('use_ground_truth', default_value='true'),
        DeclareLaunchArgument('planner_executable', default_value='planning_sampling_based_node'),
        DeclareLaunchArgument('costmap_topic', default_value='/mapping/ipm_costmap'),
        Node(
            package='planning',
            executable=LaunchConfiguration('planner_executable'),
            name='PlannerNode',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
                'target_speed': LaunchConfiguration('target_speed'),
                'use_ground_truth': LaunchConfiguration('use_ground_truth'),
                'costmap_topic': LaunchConfiguration('costmap_topic'),
            }]
        ),
    ])
