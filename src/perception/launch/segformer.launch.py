"""SegFormer Semantic Segmentation Node Launch File"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ==================== Launch Arguments ====================
        DeclareLaunchArgument(
            'model_name',
            default_value='nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
            description='HuggingFace model name. Options: b0, b1, b2, b3, b4, b5 variants'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda',
            description='Device for inference: cuda or cpu'
        ),
        DeclareLaunchArgument(
            'publish_colored',
            default_value='true',
            description='Publish colored visualization of segmentation'
        ),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/carla/ego_vehicle/sensor/camera/rgb/front/image',
            description='Input RGB image topic'
        ),
        
        # ==================== SegFormer Node ====================
        Node(
            package='perception',
            executable='segformer_node',
            name='segformer_node',
            output='screen',
            parameters=[{
                'model_name': LaunchConfiguration('model_name'),
                'device': LaunchConfiguration('device'),
                'publish_colored': LaunchConfiguration('publish_colored'),
                'input_topic': LaunchConfiguration('input_topic'),
                'resize_output': True,
                'use_sim_time': True,  # Use simulation time from /clock
            }],
        ),
    ])
