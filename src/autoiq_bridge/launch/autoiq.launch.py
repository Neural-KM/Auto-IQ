"""
Auto-IQ Complete Stack Launch File

Launches bridge, perception, planning, control, visualization, and RViz.
Each component can be enabled/disabled via launch arguments.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    autoiq_bridge_pkg = get_package_share_directory('autoiq_bridge')
    perception_pkg = get_package_share_directory('perception')
    planning_pkg = get_package_share_directory('planning')
    control_pkg = get_package_share_directory('control')
    visualization_pkg = get_package_share_directory('visualization')
    
    # Config files
    rviz_config = os.path.join(autoiq_bridge_pkg, 'config', 'autoiq.rviz')
    
    # ==================== Arguments ====================
    args = [
        # Node enables
        DeclareLaunchArgument('enable_bridge', default_value='true'),
        DeclareLaunchArgument('enable_perception', default_value='true'),
        DeclareLaunchArgument('enable_planning', default_value='true'),
        DeclareLaunchArgument('enable_control', default_value='true'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),
        DeclareLaunchArgument('enable_rviz', default_value='true'),
        
        # Shared config
        DeclareLaunchArgument('host', default_value='localhost'),
        DeclareLaunchArgument('port', default_value='2000'),
        DeclareLaunchArgument('timeout', default_value='10.0'),
        DeclareLaunchArgument('use_ground_truth', default_value='true'),
        DeclareLaunchArgument('controller', default_value='mpc'),
        DeclareLaunchArgument('target_speed', default_value='30.0'),
        DeclareLaunchArgument('spawn_point_index', default_value='50'),
    ]
    
    # Common launch arguments to pass
    common_args = {
        'carla_host': LaunchConfiguration('host'),
        'carla_port': LaunchConfiguration('port'),
        'carla_timeout': LaunchConfiguration('timeout'),
    }
    
    # ==================== Include Launch Files ====================
    
    # Bridge
    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(autoiq_bridge_pkg, 'launch', 'carla_bridge.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_bridge')),
        launch_arguments={
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'timeout': LaunchConfiguration('timeout'),
            'spawn_point_index': LaunchConfiguration('spawn_point_index'),
        }.items()
    )
    
    # Perception
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_pkg, 'launch', 'perception.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_perception')),
    )
    
    # Planning
    planning_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(planning_pkg, 'launch', 'planning.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_planning')),
        launch_arguments={
            **common_args,
            'target_speed': LaunchConfiguration('target_speed'),
            'use_ground_truth': LaunchConfiguration('use_ground_truth'),
        }.items()
    )
    
    # Control
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg, 'launch', 'control.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_control')),
        launch_arguments={
            **common_args,
            'controller': LaunchConfiguration('controller'),
        }.items()
    )
    
    # Visualization
    visualization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(visualization_pkg, 'launch', 'visualization.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_visualization')),
        launch_arguments={
            **common_args,
            'draw_ground_truth_obstacles': LaunchConfiguration('use_ground_truth'),
        }.items()
    )
    
    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else []
    )
    
    return LaunchDescription(
        args + [
            bridge_launch,
            perception_launch,
            planning_launch,
            control_launch,
            visualization_launch,
            rviz_node,
        ]
    )
