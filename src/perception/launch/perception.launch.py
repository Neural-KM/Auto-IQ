"""
Perception Launch File
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='yolov8n.pt'),
        DeclareLaunchArgument('conf_threshold', default_value='0.5'),
        DeclareLaunchArgument('enable_object_detection', default_value='true'),
        DeclareLaunchArgument('enable_lane_detection', default_value='true'),
        DeclareLaunchArgument('enable_debug_viz', default_value='false'),
        DeclareLaunchArgument('camera_pitch', default_value='-15.0'),
        DeclareLaunchArgument('camera_height', default_value='1.8'),
        
        Node(
            package='perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'conf_threshold': LaunchConfiguration('conf_threshold'),
                'enable_object_detection': LaunchConfiguration('enable_object_detection'),
                'enable_lane_detection': LaunchConfiguration('enable_lane_detection'),
                'enable_debug_viz': LaunchConfiguration('enable_debug_viz'),
                'camera.pitch': LaunchConfiguration('camera_pitch'),
                'camera.height': LaunchConfiguration('camera_height'),
                'hood_mask.enabled': True,
                'hood_mask.top_y': 0.85,
                'hood_mask.top_left_x': 0.25,
                'hood_mask.top_right_x': 0.75,
                'hood_mask.bottom_y': 1.0,
                'hood_mask.bottom_left_x': 0.08,
                'hood_mask.bottom_right_x': 0.92,
            }]
        ),
    ])
