"""
ROS Message Converters for Auto-IQ
==================================

Conversion functions between pure Python types (types.py) and ROS messages.

Convention:
- xxx_to_ros(py_obj) -> RosMsg
- xxx_from_ros(ros_msg) -> PyObj
"""

import math
from typing import List

# Import pure Python types
from autoiq_common.types import (
    VehicleState,
    TrajectoryPoint,
    Obstacle,
    Trajectory,
    ObstacleArray,
)

# Import ROS message types
from autoiq_msgs.msg import (
    TrajectoryPoint as TrajectoryPointMsg,
    Trajectory as TrajectoryMsg,
    Obstacle as ObstacleMsg,
    ObstacleArray as ObstacleArrayMsg,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Vector3, Quaternion, PoseStamped
from std_msgs.msg import Header

# Import utility functions
from autoiq_common.utils import quaternion_to_yaw, yaw_to_quaternion


# ==================== VehicleState Conversions ====================

def vehicle_state_from_odometry(msg: Odometry) -> VehicleState:
    """
    Convert nav_msgs/Odometry to VehicleState.
    
    Args:
        msg: Odometry message
        
    Returns:
        VehicleState dataclass
    """
    pos = msg.pose.pose.position
    quat = msg.pose.pose.orientation
    vel = msg.twist.twist.linear
    
    yaw = quaternion_to_yaw(quat)
    speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
    
    return VehicleState(
        x=pos.x,
        y=pos.y,
        yaw=yaw,
        speed=speed,
    )


def vehicle_state_to_pose_stamped(state: VehicleState, frame_id: str = "map") -> PoseStamped:
    """
    Convert VehicleState to geometry_msgs/PoseStamped.
    
    Args:
        state: VehicleState dataclass
        frame_id: Coordinate frame
        
    Returns:
        PoseStamped message
    """
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = state.x
    pose.pose.position.y = state.y
    pose.pose.position.z = 0.0
    pose.pose.orientation = yaw_to_quaternion(state.yaw)
    return pose


# ==================== TrajectoryPoint Conversions ====================

def trajectory_point_to_ros(point: TrajectoryPoint) -> TrajectoryPointMsg:
    """
    Convert TrajectoryPoint dataclass to ROS message.
    
    Args:
        point: TrajectoryPoint dataclass
        
    Returns:
        autoiq_msgs/TrajectoryPoint message
    """
    msg = TrajectoryPointMsg()
    msg.x = point.x
    msg.y = point.y
    msg.yaw = point.yaw
    msg.speed = point.speed
    msg.curvature = point.curvature
    msg.time_from_start = point.time_from_start
    return msg


def trajectory_point_from_ros(msg: TrajectoryPointMsg) -> TrajectoryPoint:
    """
    Convert ROS message to TrajectoryPoint dataclass.
    
    Args:
        msg: autoiq_msgs/TrajectoryPoint message
        
    Returns:
        TrajectoryPoint dataclass
    """
    return TrajectoryPoint(
        x=msg.x,
        y=msg.y,
        yaw=msg.yaw,
        s=0.0,  # Not in ROS message, set to default
        d=0.0,  # Not in ROS message, set to default
        speed=msg.speed,
        curvature=msg.curvature,
        time_from_start=msg.time_from_start,
    )


# ==================== Trajectory Conversions ====================

def trajectory_to_ros(traj: Trajectory) -> TrajectoryMsg:
    """
    Convert Trajectory dataclass to ROS message.
    
    Args:
        traj: Trajectory dataclass
        
    Returns:
        autoiq_msgs/Trajectory message
    """
    msg = TrajectoryMsg()
    msg.header.frame_id = traj.frame_id
    msg.header.stamp.sec = traj.header_stamp_sec
    msg.header.stamp.nanosec = traj.header_stamp_nanosec
    msg.points = [trajectory_point_to_ros(p) for p in traj.points]
    return msg


def trajectory_from_ros(msg: TrajectoryMsg) -> Trajectory:
    """
    Convert ROS message to Trajectory dataclass.
    
    Args:
        msg: autoiq_msgs/Trajectory message
        
    Returns:
        Trajectory dataclass
    """
    return Trajectory(
        points=[trajectory_point_from_ros(p) for p in msg.points],
        header_stamp_sec=msg.header.stamp.sec,
        header_stamp_nanosec=msg.header.stamp.nanosec,
        frame_id=msg.header.frame_id,
    )


# ==================== Obstacle Conversions ====================

def obstacle_from_ros(msg: ObstacleMsg) -> Obstacle:
    """
    Convert ROS message to Obstacle dataclass.
    
    Args:
        msg: autoiq_msgs/Obstacle message
        
    Returns:
        Obstacle dataclass
    """
    yaw = quaternion_to_yaw(msg.pose.orientation)
    
    return Obstacle(
        id=msg.id,
        type=msg.type,
        x=msg.pose.position.x,
        y=msg.pose.position.y,
        yaw=yaw,
        vx=msg.velocity.x,
        vy=msg.velocity.y,
        length=msg.size.x,
        width=msg.size.y,
        height=msg.size.z,
        confidence=msg.confidence,
    )


def obstacle_to_ros(obs: Obstacle) -> ObstacleMsg:
    """
    Convert Obstacle dataclass to ROS message.
    
    Args:
        obs: Obstacle dataclass
        
    Returns:
        autoiq_msgs/Obstacle message
    """
    msg = ObstacleMsg()
    msg.id = obs.id
    msg.type = obs.type
    msg.pose.position.x = obs.x
    msg.pose.position.y = obs.y
    msg.pose.position.z = obs.height / 2.0  # Center height
    msg.pose.orientation = yaw_to_quaternion(obs.yaw)
    msg.velocity.x = obs.vx
    msg.velocity.y = obs.vy
    msg.velocity.z = 0.0
    msg.size.x = obs.length
    msg.size.y = obs.width
    msg.size.z = obs.height
    msg.confidence = obs.confidence
    return msg


# ==================== ObstacleArray Conversions ====================

def obstacle_array_from_ros(msg: ObstacleArrayMsg) -> ObstacleArray:
    """
    Convert ROS message to ObstacleArray dataclass.
    
    Args:
        msg: autoiq_msgs/ObstacleArray message
        
    Returns:
        ObstacleArray dataclass
    """
    return ObstacleArray(
        obstacles=[obstacle_from_ros(o) for o in msg.obstacles],
        header_stamp_sec=msg.header.stamp.sec,
        header_stamp_nanosec=msg.header.stamp.nanosec,
        frame_id=msg.header.frame_id,
    )


def obstacle_array_to_ros(arr: ObstacleArray) -> ObstacleArrayMsg:
    """
    Convert ObstacleArray dataclass to ROS message.
    
    Args:
        arr: ObstacleArray dataclass
        
    Returns:
        autoiq_msgs/ObstacleArray message
    """
    msg = ObstacleArrayMsg()
    msg.header.frame_id = arr.frame_id
    msg.header.stamp.sec = arr.header_stamp_sec
    msg.header.stamp.nanosec = arr.header_stamp_nanosec
    msg.obstacles = [obstacle_to_ros(o) for o in arr.obstacles]
    return msg
