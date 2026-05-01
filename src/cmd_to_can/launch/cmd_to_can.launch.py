"""Launch file for the cmd_to_can bridge node.

Example::

    ros2 launch cmd_to_can cmd_to_can.launch.py \
        can_channel:=can0 \
        target_velocity:=0.0 \
        autonomous_mode:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _typed(name: str, value_type):
    """Resolve a launch argument and force it into the requested Python type.

    ``LaunchConfiguration`` always evaluates to a string at launch time, but
    the node declares typed parameters. ``ParameterValue(..., value_type=...)``
    is the canonical way to bridge that gap without writing a YAML file.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "control_topic",
            default_value="/carla/ego_vehicle/control",
            description="Topic to subscribe to for VehicleControl commands.",
        ),
        DeclareLaunchArgument(
            "publish_rate_hz",
            default_value="20.0",
            description="Periodic CAN send rate (mode/angle/velocity frames).",
        ),
        DeclareLaunchArgument(
            "can_interface",
            default_value="socketcan",
            description="python-can interface name (e.g. socketcan, pcan).",
        ),
        DeclareLaunchArgument(
            "can_channel",
            default_value="can0",
            description="CAN channel name (e.g. can0, PCAN_USBBUS1).",
        ),
        DeclareLaunchArgument(
            "can_bitrate",
            default_value="250000",
            description="CAN bus bitrate in bits/s.",
        ),
        DeclareLaunchArgument(
            "target_velocity",
            default_value="0.0",
            description=(
                "Hard-coded velocity setpoint sent on CAN ID 0x104, in the "
                "same scaled engineering units the firmware's VEL_FB_SCALE "
                "produces (typically m/s). 0 keeps the throttle parked. "
                "Bypasses VehicleControl.throttle/brake."
            ),
        ),
        DeclareLaunchArgument(
            "autonomous_mode",
            default_value="false",
            description="If true, request the Teensy autonomous relay HIGH.",
        ),
        DeclareLaunchArgument(
            "steer_mode",
            default_value="1",
            description="0=DIRECT (PWM torque), 1=ANGLE_PID (firmware angle PID).",
        ),
        DeclareLaunchArgument(
            "throttle_mode",
            default_value="1",
            description="0=DIRECT (host throttle), 1=VELOCITY_PID (firmware vel PID).",
        ),
        DeclareLaunchArgument(
            "steer_sign",
            default_value="-1.0",
            description=(
                "Sign applied when mapping ROS steer (left positive) to "
                "firmware angle (right positive). Default -1.0."
            ),
        ),
        DeclareLaunchArgument(
            "max_steer_angle_deg",
            default_value="15.0",
            description=(
                "Maximum |steer angle| in degrees sent to the firmware. "
                "Hard-clamped by the firmware at 15 deg regardless."
            ),
        ),
        DeclareLaunchArgument(
            "blacklist_file",
            default_value="",
            description=(
                "Path to a used-CAN-ID blacklist (compute_side_tui format). "
                "Empty = use the file shipped in the package's share/config."
            ),
        ),
        DeclareLaunchArgument(
            "disable_blacklist",
            default_value="false",
            description="If true, skip blacklist enforcement entirely.",
        ),

        Node(
            package="cmd_to_can",
            executable="cmd_to_can_node",
            name="cmd_to_can_node",
            output="screen",
            parameters=[{
                "control_topic": _typed("control_topic", str),
                "publish_rate_hz": _typed("publish_rate_hz", float),
                "can_interface": _typed("can_interface", str),
                "can_channel": _typed("can_channel", str),
                "can_bitrate": _typed("can_bitrate", int),
                "target_velocity": _typed("target_velocity", float),
                "autonomous_mode": _typed("autonomous_mode", bool),
                "steer_mode": _typed("steer_mode", int),
                "throttle_mode": _typed("throttle_mode", int),
                "steer_sign": _typed("steer_sign", float),
                "max_steer_angle_deg": _typed("max_steer_angle_deg", float),
                "blacklist_file": _typed("blacklist_file", str),
                "disable_blacklist": _typed("disable_blacklist", bool),
            }],
        ),
    ])
