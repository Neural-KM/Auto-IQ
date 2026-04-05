"""
Auto-IQ Common Library
======================

Shared types and converters for the Auto-IQ stack.

This package provides:
- Pure Python dataclasses for internal logic (types.py)
- Conversion functions between Python types and ROS messages (converters.py)
- Camera parameter utilities (camera_params.py)
"""

from autoiq_common.types import (
    VehicleState,
    TrajectoryPoint,
    Obstacle,
    Trajectory,
    ObstacleArray,
)

from autoiq_common.converters import (
    vehicle_state_from_odometry,
    vehicle_state_to_pose_stamped,
    trajectory_point_to_ros,
    trajectory_point_from_ros,
    obstacle_from_ros,
    obstacle_to_ros,
    trajectory_to_ros,
    trajectory_from_ros,
    obstacle_array_from_ros,
    obstacle_array_to_ros,
)

from autoiq_common.utils import (
    quaternion_to_yaw,
    yaw_to_quaternion,
)

from autoiq_common.camera_params import (
    CameraParams,
    CameraParamsProvider,
)

__all__ = [
    # Types
    'VehicleState',
    'TrajectoryPoint',
    'Obstacle',
    'Trajectory',
    'ObstacleArray',
    # Converters
    'vehicle_state_from_odometry',
    'vehicle_state_to_pose_stamped',
    'trajectory_point_to_ros',
    'trajectory_point_from_ros',
    'obstacle_from_ros',
    'obstacle_to_ros',
    'trajectory_to_ros',
    'trajectory_from_ros',
    'obstacle_array_from_ros',
    'obstacle_array_to_ros',
    'quaternion_to_yaw',
    'yaw_to_quaternion',
    # Camera params
    'CameraParams',
    'CameraParamsProvider',
]
