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
        DeclareLaunchArgument(
            'use_sim_time', default_value='true', choices=['true', 'false'],
            description='Drive nodes off /clock instead of wall time. '
                        'Set false for real-hardware bringups.'),
        DeclareLaunchArgument(
            'local_only', default_value='false', choices=['true', 'false'],
            description='Run pure_pursuit without odometry (vehicle assumed '
                        'at origin of trajectory frame). Use on real '
                        'hardware where there is no map/localisation.'),
        DeclareLaunchArgument(
            'control_topic', default_value='/carla/ego_vehicle/control',
            description='VehicleControl topic the controller publishes to. '
                        'Both cmd_to_can_node and the CARLA bridge default '
                        'to /carla/ego_vehicle/control, so the same topic '
                        'works for sim and hardware.'),
        DeclareLaunchArgument(
            'trajectory_topic', default_value='/planning/trajectory',
            description='Trajectory topic the controller subscribes to.'),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/carla/ego_vehicle/ground_truth/odometry',
            description='Odometry topic (ignored when local_only=true).'),

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
                'use_sim_time': LaunchConfiguration('use_sim_time'),
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
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'carla_host': LaunchConfiguration('carla_host'),
                'carla_port': LaunchConfiguration('carla_port'),
                'carla_timeout': LaunchConfiguration('carla_timeout'),
                'local_only': LaunchConfiguration('local_only'),
                'control_topic': LaunchConfiguration('control_topic'),
                'trajectory_topic': LaunchConfiguration('trajectory_topic'),
                'odom_topic': LaunchConfiguration('odom_topic'),
            }]
        ),
    ])
