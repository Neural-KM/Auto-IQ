"""
CARLA Bridge Launch File
========================

Launches the CARLA bridge node with configurable parameters.

Usage:
    ros2 launch autoiq_bridge carla_bridge.launch.py
    ros2 launch autoiq_bridge carla_bridge.launch.py carla_host:=192.168.1.100 carla_port:=2000
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('autoiq_bridge')
    
    # Default config file path
    default_objects_file = os.path.join(pkg_share, 'config', 'objects.json')
    
    # Declare launch arguments
    host_arg = DeclareLaunchArgument(
        'carla_host',
        default_value='localhost',
        description='CARLA server hostname'
    )

    port_arg = DeclareLaunchArgument(
        'carla_port',
        default_value='2000',
        description='CARLA server port'
    )
    
    timeout_arg = DeclareLaunchArgument(
        'timeout',
        default_value='10.0',
        description='CARLA connection timeout in seconds'
    )
    
    sync_mode_arg = DeclareLaunchArgument(
        'synchronous_mode',
        default_value='true',
        description='Enable synchronous mode'
    )
    
    delta_seconds_arg = DeclareLaunchArgument(
        'fixed_delta_seconds',
        default_value='0.05',
        description='Fixed time step for synchronous mode (20 Hz)'
    )
    
    objects_file_arg = DeclareLaunchArgument(
        'objects_file',
        default_value=default_objects_file,
        description='Path to objects.json config file'
    )
    
    spawn_point_arg = DeclareLaunchArgument(
        'spawn_point_index',
        default_value='50',
        description='Spawn point index on the map'
    )
    
    # CARLA Bridge Node
    carla_bridge_node = Node(
        package='autoiq_bridge',
        executable='carla_bridge_node',
        name='carla_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'carla_host': LaunchConfiguration('carla_host'),
            'carla_port': LaunchConfiguration('carla_port'),
            'timeout': LaunchConfiguration('timeout'),
            'synchronous_mode': LaunchConfiguration('synchronous_mode'),
            'fixed_delta_seconds': LaunchConfiguration('fixed_delta_seconds'),
            'objects_file': LaunchConfiguration('objects_file'),
            'spawn_point_index': LaunchConfiguration('spawn_point_index'),
        }]
    )
    
    return LaunchDescription([
        host_arg,
        port_arg,
        timeout_arg,
        sync_mode_arg,
        delta_seconds_arg,
        objects_file_arg,
        spawn_point_arg,
        carla_bridge_node,
    ])
