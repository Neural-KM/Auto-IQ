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
        DeclareLaunchArgument(
            'semantic_source', default_value='fused',
            description="'fused' (SegFormer + lane detection) or 'ground_truth' (CARLA sim)"
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

        # ==================== Node ====================
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
