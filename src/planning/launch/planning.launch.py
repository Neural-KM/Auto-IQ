"""Planning Node Launch File

Wraps the CARLA-side global planners. The lane-follower-only path used by
the real-hardware bringup (`autoiq_bridge/launch/hardware.launch.py`)
instantiates ``lane_follower_node`` directly so we don't end up forcing
parameters this generic launcher knows nothing about onto whichever
``planner_executable`` is selected.
"""

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
        DeclareLaunchArgument(
            'use_sim_time', default_value='true', choices=['true', 'false'],
            description='Drive nodes off /clock instead of wall time. Set '
                        'false for real-hardware bringups.'),

        Node(
            package='planning',
            executable=LaunchConfiguration('planner_executable'),
            name='PlannerNode',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
                'target_speed': LaunchConfiguration('target_speed'),
                'use_ground_truth': LaunchConfiguration('use_ground_truth'),
                'costmap_topic': LaunchConfiguration('costmap_topic'),
            }]
        ),
    ])
