"""
Utility Functions for Auto-IQ
=============================

Common utility functions for coordinate transformations and conversions.
"""

import math
from geometry_msgs.msg import Quaternion


def quaternion_to_yaw(q) -> float:
    """
    Convert quaternion to yaw angle in radians.
    
    Args:
        q: Quaternion (ROS message or any object with x, y, z, w attributes)
        
    Returns:
        Yaw angle in radians
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """
    Convert yaw angle in radians to quaternion.
    
    Args:
        yaw: Yaw angle in radians
        
    Returns:
        Quaternion ROS message with only z rotation
    """
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q
