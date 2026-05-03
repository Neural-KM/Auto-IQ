"""
Auto-IQ Hardware Launch File
============================

End-to-end autonomy stack for the real golf-cart rig (no CARLA, no
LIDAR, front camera only). Lives in `autoiq_bridge` because that's the
package name the rest of the docs use as the "stack entry point", even
though there's no actual sim-bridge running here.

Pipeline
--------
  sensors_bringup (1x ZED X One Mono GS, base_link TF)
        │
        ▼  /zed_1/zed_node/rgb/color/image
  perception (segformer + road-lane extraction; no YOLO, no LiDAR)
        │
        ▼  /perception/semantic_map/road_lane_extraction
  mapping (IPM occupancy grid + lane fitting in base_link)
        │
        ▼  /mapping/estimated_lanes
  planning/lane_follower_node  (local_only=true)
        │
        ▼  /planning/trajectory (base_link), /target_velocity (Float32)
  control/pure_pursuit_node    (local_only=true)
        │
        ▼  /carla/ego_vehicle/control  (VehicleControl)
  cmd_to_can_node              (autonomous_mode starts FALSE)
        │
        ▼  CAN frames to Teensy (steer angle + velocity setpoint)

`use_sim_time` is forced to `false` everywhere (real hardware → wall
time). The CARLA-only nodes (perception_node, semantic_pointcloud_node,
planning_sampling_based_node, carla_viz_node, the bridge itself) are
NOT included.

Usage
-----
    ros2 launch autoiq_bridge hardware.launch.py
    ros2 launch autoiq_bridge hardware.launch.py serials:=300257165 rviz:=false
    ros2 launch autoiq_bridge hardware.launch.py autonomous_mode:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Topics + frames the ZED X One front camera publishes through the upstream
# zed-ros2-wrapper. The wrapper namespaces everything under the camera name
# (zed_1) and the node name (zed_node), so a swap of zed_1 → zed_2 only
# requires changing `front_camera` below.
# ---------------------------------------------------------------------------
DEFAULT_FRONT_CAMERA = 'zed_1'
DEFAULT_FRONT_SERIAL = '300257165'  # matches CAMERAS table in zed_cameras.launch.py


def _front_topic(camera_name: str, subtopic: str) -> str:
    """Compose `/<camera>/zed_node/<subtopic>` for the upstream wrapper."""
    return f'/{camera_name}/zed_node/{subtopic}'


def _typed(name: str, value_type):
    """Coerce a string-valued LaunchConfiguration into the requested type
    so it lines up with each node's declared parameter type. The launch
    system would otherwise pass everything as a string and the node-side
    parameter declaration would reject it.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    autoiq_bridge_pkg = get_package_share_directory('autoiq_bridge')
    perception_pkg = get_package_share_directory('perception')
    mapping_pkg = get_package_share_directory('mapping')
    control_pkg = get_package_share_directory('control')
    sensors_bringup_pkg = get_package_share_directory('sensors_bringup')
    cmd_to_can_pkg = get_package_share_directory('cmd_to_can')

    rviz_config = os.path.join(autoiq_bridge_pkg, 'config', 'autoiq.rviz')

    # --------------------- Arguments ---------------------
    args = [
        DeclareLaunchArgument(
            'front_camera', default_value=DEFAULT_FRONT_CAMERA,
            description='ZED camera namespace to use as the front-facing '
                        'perception source (matches sensors_drivers/launch/'
                        'zed_cameras.launch.py CAMERAS table).'),
        DeclareLaunchArgument(
            'serials', default_value=DEFAULT_FRONT_SERIAL,
            description='Comma-separated serial numbers passed straight '
                        'through to sensors_bringup. Default: only the '
                        'front camera is brought up.'),
        DeclareLaunchArgument(
            'lens_label', default_value='wide',
            choices=['wide', 'medium', 'narrow', 'unknown'],
            description='Informational tag for the ZED X One Mono GS M12 '
                        'lens variant. Intrinsics still come from the '
                        'per-serial factory calibration in EEPROM.'),
        DeclareLaunchArgument(
            'base_frame', default_value='base_link',
            description='Vehicle root frame published by the sensor_rig URDF.'),
        DeclareLaunchArgument(
            'rviz', default_value='true', choices=['true', 'false'],
            description='Start the rig RViz2 view.'),
        DeclareLaunchArgument(
            'enable_sensors', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument(
            'enable_perception', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument(
            'enable_mapping', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument(
            'enable_planning', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument(
            'enable_control', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument(
            'enable_can', default_value='true', choices=['true', 'false'],
            description='Run cmd_to_can_node. Set false when running on a '
                        'desk without the Teensy attached.'),
        DeclareLaunchArgument(
            'enable_rviz_config', default_value='true', choices=['true', 'false'],
            description='Use the autoiq RViz config (else launch RViz2 with '
                        'no preset).'),
        # CAN-side knobs forwarded to cmd_to_can_node.
        DeclareLaunchArgument(
            'can_channel', default_value='can0',
            description='SocketCAN channel name (e.g. can0).'),
        DeclareLaunchArgument(
            'autonomous_mode', default_value='false', choices=['true', 'false'],
            description='Engage the Teensy autonomous relay at startup. '
                        'Defaults FALSE so a fresh launch never silently '
                        'energises the actuator. Flip via `ros2 param set` '
                        'once you have eyes on the bus.'),
        # SegFormer knobs.
        DeclareLaunchArgument(
            'segformer_model', default_value='b2',
            description='SegFormer variant: b0 (fastest) ... b5 (best).'),
        DeclareLaunchArgument(
            'segformer_device', default_value='cuda',
            description='SegFormer device: cuda (Jetson GPU) or cpu.'),
    ]

    # --------------------- Derived topic strings ---------------------
    # We resolve these via PythonLaunchDescriptionSource arguments so the
    # leaf launches see the fully-qualified topic; using a Substitution
    # would also work but stringly-typing this once at the top makes the
    # data flow easier to read.
    front_camera = LaunchConfiguration('front_camera')

    # The upstream ZedCameraOne component (see
    # zed_components/src/zed_camera_one/src/zed_camera_one_component_video.cpp)
    # advertises the rectified colour image via image_transport at:
    #   <ns>/<node>/rgb/color/rect/image
    #   <ns>/<node>/rgb/color/rect/camera_info
    # which is what the perception + mapping stack should consume so the
    # ZED SDK applies the per-serial wide-lens calibration before the
    # frame ever leaves the camera.
    rgb_topic = [front_camera, '/zed_node/rgb/color/rect/image']
    camera_info_topic = [front_camera, '/zed_node/rgb/color/rect/camera_info']
    camera_optical_frame = [front_camera, '_camera_optical_frame']

    # --------------------- Sensors (URDF + ZED + monitor; no LIDAR) ---------------------
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sensors_bringup_pkg, 'launch', 'all_sensors.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_sensors')),
        launch_arguments={
            'cameras': 'true',
            'lidar': 'false',
            'rviz': 'false',          # we run a single shared RViz below
            'monitor': 'true',
            'publish_urdf': 'true',
            'use_sim_time': 'false',
            'serials': LaunchConfiguration('serials'),
            'lens_label': LaunchConfiguration('lens_label'),
        }.items(),
    )

    # --------------------- Perception (segformer only; no carla, no lidar) ---------------------
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_pkg, 'launch', 'perception.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_perception')),
        launch_arguments={
            'use_sim_time': 'false',
            'segmentation_source': 'segformer',
            'model_name': LaunchConfiguration('segformer_model'),
            'device': LaunchConfiguration('segformer_device'),
            'rgb_topic': rgb_topic,
            'enable_perception_node': 'false',         # imports carla; not on hardware
            'enable_semantic_pointcloud': 'false',     # needs LiDAR
        }.items(),
    )

    # --------------------- Mapping (base_link, ZED intrinsics) ---------------------
    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mapping_pkg, 'launch', 'mapping.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_mapping')),
        launch_arguments={
            'use_sim_time': 'false',
            'semantic_source': 'road_lane_extraction',
            'base_frame': LaunchConfiguration('base_frame'),
            'camera_frame': camera_optical_frame,
            'camera_info_topic': camera_info_topic,
            'enable_lane_debug_viz': 'true',
        }.items(),
    )

    # --------------------- Planning (lane_follower in local_only mode) ---------------------
    # Run lane_follower_node directly rather than going through
    # planning.launch.py, because the other planners take CARLA-only
    # parameters that would have to be conditionally suppressed.
    lane_follower_node = Node(
        condition=IfCondition(LaunchConfiguration('enable_planning')),
        package='planning',
        executable='lane_follower_node',
        name='lane_follower_node',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'local_only': True,
            'base_frame': _typed('base_frame', str),
            'estimated_lanes_topic': '/mapping/estimated_lanes',
            'trajectory_topic': '/planning/trajectory',
            'midline_topic': '/planning/lane_midline',
            'target_velocity_topic': '/target_velocity',
        }],
    )

    # --------------------- Control (pure_pursuit in local_only mode) ---------------------
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg, 'launch', 'control.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_control')),
        launch_arguments={
            'use_sim_time': 'false',
            'controller': 'pure_pursuit',
            'local_only': 'true',
            'control_topic': '/carla/ego_vehicle/control',
            'trajectory_topic': '/planning/trajectory',
        }.items(),
    )

    # --------------------- CAN bridge ---------------------
    cmd_to_can_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(cmd_to_can_pkg, 'launch', 'cmd_to_can.launch.py')
        ),
        condition=IfCondition(LaunchConfiguration('enable_can')),
        launch_arguments={
            'control_topic': '/carla/ego_vehicle/control',
            'target_velocity_topic': '/target_velocity',
            'can_channel': LaunchConfiguration('can_channel'),
            'autonomous_mode': LaunchConfiguration('autonomous_mode'),
        }.items(),
    )

    # --------------------- RViz (single shared window) ---------------------
    rviz_node = Node(
        condition=IfCondition(LaunchConfiguration('rviz')),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        parameters=[{'use_sim_time': False}],
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
    )

    return LaunchDescription(args + [
        sensors_launch,
        perception_launch,
        mapping_launch,
        lane_follower_node,
        control_launch,
        cmd_to_can_launch,
        rviz_node,
    ])
