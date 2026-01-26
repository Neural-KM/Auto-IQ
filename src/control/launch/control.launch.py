"""Control Node Launch File"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('carla_host', default_value='localhost'),
        DeclareLaunchArgument('carla_port', default_value='2000'),
        DeclareLaunchArgument('carla_timeout', default_value='10.0'),
        DeclareLaunchArgument('controller', default_value='pure_pursuit',
                              description="'mpc' or 'pure_pursuit'"),
        
        # MPC Node
        Node(
            package='control',
            executable='mpc_node',
            name='mpc_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('controller'), "' == 'mpc'"])
            ),
            parameters=[{
                'use_sim_time': True,
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
            }]
        ),
        
        # Pure Pursuit Node
        Node(
            package='control',
            executable='pure_pursuit_node',
            name='pure_pursuit_node',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('controller'), "' == 'pure_pursuit'"])
            ),
            parameters=[{
                'use_sim_time': True,
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
            }]
        ),
    ])
