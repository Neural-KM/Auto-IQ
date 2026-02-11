"""
Auto-IQ Complete Stack Launch File

Launches bridge, perception, planning, control, visualization, and RViz.
Each component can be enabled/disabled via launch arguments.
Each package uses its own default parameters from its own launch file.
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
    mapping_pkg = get_package_share_directory('mapping')
    
    # Config files
    rviz_config = os.path.join(autoiq_bridge_pkg, 'config', 'autoiq.rviz')
    
    # ==================== Arguments (enable/disable only) ====================
    args = [
        DeclareLaunchArgument('enable_bridge', default_value='true'),
        DeclareLaunchArgument('enable_perception', default_value='true'),
        DeclareLaunchArgument('segmentation_source', default_value='segformer',
                              description="'segformer' (camera-based) or 'ground_truth' (CARLA sim)"),
        DeclareLaunchArgument('semantic_source', default_value='fused',
                              description="Mapping input: 'fused' (perception pipeline) or 'ground_truth' (CARLA sim)"),
        DeclareLaunchArgument('enable_planning', default_value='true'),
        DeclareLaunchArgument('enable_control', default_value='true'),
        DeclareLaunchArgument('enable_visualization', default_value='true'),
        DeclareLaunchArgument('enable_rviz', default_value='true'),
        DeclareLaunchArgument('enable_mapping', default_value='true'),
    ]
    
    # ==================== Include Launch Files ====================
    
    # Bridge
    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(autoiq_bridge_pkg, 'launch', 'carla_bridge.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_bridge')),
    )
    
    # Perception
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_pkg, 'launch', 'perception.launch.py')
        ),
        launch_arguments={'segmentation_source': LaunchConfiguration('segmentation_source')}.items(),
        condition=IfCondition(LaunchConfiguration('enable_perception')),
    )
    
    # Planning
    planning_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(planning_pkg, 'launch', 'planning.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_planning')),
    )
    
    # Control
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg, 'launch', 'control.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_control')),
    )
    
    # Visualization
    visualization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(visualization_pkg, 'launch', 'visualization.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_visualization')),
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

    # Mapping
    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mapping_pkg, 'launch', 'local_costmap.launch.py')
        ),
        launch_arguments={'semantic_source': LaunchConfiguration('semantic_source')}.items(),
        condition=IfCondition(LaunchConfiguration('enable_mapping')),
    )
    
    return LaunchDescription(
        args + [
            bridge_launch,
            perception_launch,
            planning_launch,
            control_launch,
            visualization_launch,
            rviz_node,
            mapping_launch
        ]
    )
