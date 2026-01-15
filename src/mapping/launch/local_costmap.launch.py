"""
Local Costmap Launch File
=========================

Launches the local costmap generator node with configurable parameters.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        # ==================== Arguments ====================
        DeclareLaunchArgument(
            'pointcloud_topic',
            default_value='/perception/semantic_pointcloud',
            description='Input labeled point cloud topic'
        ),
        DeclareLaunchArgument(
            'costmap_topic',
            default_value='/mapping/local_costmap',
            description='Output occupancy grid topic'
        ),
        DeclareLaunchArgument(
            'grid_width',
            default_value='40.0',
            description='Grid width in meters (lateral)'
        ),
        DeclareLaunchArgument(
            'grid_length',
            default_value='40.0',
            description='Grid length in meters (longitudinal)'
        ),
        DeclareLaunchArgument(
            'resolution',
            default_value='0.2',
            description='Grid resolution in meters per cell'
        ),
        
        # ==================== Node ====================
        Node(
            package='mapping',
            executable='local_costmap_node',
            name='local_costmap_node',
            output='screen',
            parameters=[{
                'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
                'costmap_topic': LaunchConfiguration('costmap_topic'),
                'grid_width': LaunchConfiguration('grid_width'),
                'grid_length': LaunchConfiguration('grid_length'),
                'resolution': LaunchConfiguration('resolution'),
                'frame_id': 'ego_vehicle',
                'origin_x_offset': 0.5,
                'min_height': -0.5,
                'max_height': 3.0,
            }]
        ),
    ])
