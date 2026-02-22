"""
CARLA to ROS2 Message Converters
================================

Utility functions to convert CARLA data types to standard ROS2 messages.
"""

import numpy as np
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from nav_msgs.msg import Odometry
import math

from cv_bridge import CvBridge

_cv_bridge = CvBridge()


def carla_timestamp_to_ros(carla_timestamp: float) -> Time:
    """Convert CARLA timestamp (seconds) to ROS2 Time message."""
    sec = int(carla_timestamp)
    nanosec = int((carla_timestamp - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)


def create_header(frame_id: str, timestamp: float) -> Header:
    """Create a ROS2 Header with frame_id and timestamp."""
    header = Header()
    header.frame_id = frame_id
    header.stamp = carla_timestamp_to_ros(timestamp)
    return header


def carla_image_to_ros(carla_image, frame_id: str, encoding: str = 'rgb8') -> Image:
    """
    Convert CARLA image to ROS2 Image message.
    
    Args:
        carla_image: CARLA image object or numpy array
        frame_id: TF frame ID for the camera
        encoding: Image encoding ('rgb8', 'bgra8', 'mono16', '32FC1', etc.)
        
    Returns:
        sensor_msgs/Image message
    """
    # Handle both raw CARLA image and numpy array
    if hasattr(carla_image, 'raw_data'):
        # Raw CARLA image - convert based on type
        array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
        array = array.reshape((carla_image.height, carla_image.width, 4))
        # CARLA gives BGRA, convert to RGB
        if encoding == 'rgb8':
            array = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        timestamp = carla_image.timestamp
        height = carla_image.height
        width = carla_image.width
    else:
        # Already numpy array
        array = carla_image
        timestamp = 0.0  # Will need to be set externally
        height, width = array.shape[:2]
    
    # Cast to appropriate dtype for the encoding
    dtype_map = {
        'rgb8': np.uint8,
        'bgra8': np.uint8,
        '32FC1': np.float32,
        'mono8': np.uint8,
        'mono16': np.uint16,
    }
    if encoding in dtype_map:
        array = array.astype(dtype_map[encoding])
    
    msg = _cv_bridge.cv2_to_imgmsg(np.ascontiguousarray(array), encoding=encoding)
    msg.header = create_header(frame_id, timestamp)
    
    return msg


def carla_depth_to_ros(depth_array: np.ndarray, frame_id: str, timestamp: float) -> Image:
    """
    Convert CARLA depth image to ROS2 Image message.
    
    Args:
        depth_array: Depth values in meters (float32)
        frame_id: TF frame ID
        timestamp: CARLA timestamp
        
    Returns:
        sensor_msgs/Image with 32FC1 encoding
    """
    msg = _cv_bridge.cv2_to_imgmsg(depth_array.astype(np.float32), encoding='32FC1')
    msg.header = create_header(frame_id, timestamp)
    return msg


def carla_semantic_to_ros(semantic_array: np.ndarray, frame_id: str, timestamp: float) -> Image:
    """
    Convert CARLA semantic segmentation to ROS2 Image message.
    
    Args:
        semantic_array: Semantic labels (uint8)
        frame_id: TF frame ID
        timestamp: CARLA timestamp
        
    Returns:
        sensor_msgs/Image with mono8 encoding
    """
    msg = _cv_bridge.cv2_to_imgmsg(semantic_array.astype(np.uint8), encoding='mono8')
    msg.header = create_header(frame_id, timestamp)
    return msg


def carla_lidar_to_ros(points: np.ndarray, frame_id: str, timestamp: float) -> PointCloud2:
    """
    Convert CARLA LiDAR points to ROS2 PointCloud2 message with RGB coloring by height.
    
    Args:
        points: Nx4 array of (x, y, z, intensity)
        frame_id: TF frame ID
        timestamp: CARLA timestamp
        
    Returns:
        sensor_msgs/PointCloud2 message with x, y, z, intensity, and rgb fields
    """
    msg = PointCloud2()
    msg.header = create_header(frame_id, timestamp)
    
    num_points = len(points)
    
    # Compute RGB color based on height (Z) for nice visualization
    z_values = points[:, 2]
    z_min, z_max = -2.0, 1.0
    z_normalized = np.clip((z_values - z_min) / (z_max - z_min), 0.0, 1.0)
    
    # Red-Yellow-Green gradient: red (low) -> yellow (mid) -> green (high)
    r = np.clip(2.0 - 2.0 * z_normalized, 0, 1)
    g = np.clip(2.0 * z_normalized, 0, 1)
    b = np.zeros_like(z_normalized)
    
    # Pack RGB into uint32 (RViz expects this format for 'rgb' field)
    r_uint = (r * 255).astype(np.uint8)
    g_uint = (g * 255).astype(np.uint8)
    b_uint = (b * 255).astype(np.uint8)
    rgb_packed = (r_uint.astype(np.uint32) << 16) | (g_uint.astype(np.uint32) << 8) | b_uint.astype(np.uint32)
    rgb_float = rgb_packed.view(np.float32)
    
    # Define fields: x, y, z, intensity, rgb
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 20
    
    # Build cloud data array
    cloud_data = np.zeros((num_points, 5), dtype=np.float32)
    cloud_data[:, 0] = points[:, 0]
    cloud_data[:, 1] = points[:, 1]
    cloud_data[:, 2] = points[:, 2]
    cloud_data[:, 3] = points[:, 3]
    cloud_data[:, 4] = rgb_float
    
    cloud_data = np.ascontiguousarray(cloud_data)
    
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.height = 1
    msg.width = num_points
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    msg.data = cloud_data.tobytes()
    
    return msg


def carla_semantic_lidar_to_ros(points: np.ndarray, labels: np.ndarray, frame_id: str, timestamp: float) -> PointCloud2:
    """
    Convert CARLA Semantic LiDAR detections to ROS2 PointCloud2.

    Args:
        points: Nx3 array of XYZ coordinates.
        labels: N array of semantic class IDs (uint32).
        frame_id: TF frame ID.
        timestamp: CARLA timestamp.

    Returns:
        PointCloud2 message with fields x, y, z, label.
    """
    msg = PointCloud2()
    msg.header = create_header(frame_id, timestamp)

    num_points = len(points)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='label', offset=12, datatype=PointField.UINT32, count=1),
    ]
    point_step = 16

    cloud_data = np.zeros(num_points, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('label', '<u4')])
    cloud_data['x'] = points[:, 0]
    cloud_data['y'] = points[:, 1]
    cloud_data['z'] = points[:, 2]
    cloud_data['label'] = labels.astype(np.uint32)

    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.height = 1
    msg.width = num_points
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    msg.data = cloud_data.tobytes()

    return msg


def create_camera_info(width: int, height: int, fov: float, frame_id: str, timestamp: float) -> CameraInfo:
    """
    Create CameraInfo message from camera parameters.
    
    Args:
        width: Image width
        height: Image height
        fov: Horizontal field of view in degrees
        frame_id: TF frame ID
        timestamp: CARLA timestamp
        
    Returns:
        sensor_msgs/CameraInfo message
    """
    msg = CameraInfo()
    msg.header = create_header(frame_id, timestamp)
    msg.width = width
    msg.height = height
    
    # Compute focal length from FOV
    # fov = 2 * atan(width / (2 * fx))
    # fx = width / (2 * tan(fov/2))
    fov_rad = math.radians(fov)
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx  # Square pixels
    cx = width / 2.0
    cy = height / 2.0
    
    # Intrinsic camera matrix (K)
    msg.k = [fx, 0.0, cx,
             0.0, fy, cy,
             0.0, 0.0, 1.0]
    
    # Rectification matrix (identity for non-stereo)
    msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    
    # Projection matrix
    msg.p = [fx, 0.0, cx, 0.0,
             0.0, fy, cy, 0.0,
             0.0, 0.0, 1.0, 0.0]
    
    msg.distortion_model = 'plumb_bob'
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
    
    return msg


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """
    Convert Euler angles (radians) to Quaternion.
    
    Args:
        roll: Rotation around X axis
        pitch: Rotation around Y axis  
        yaw: Rotation around Z axis
        
    Returns:
        geometry_msgs/Quaternion
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def carla_transform_to_ros(carla_transform, frame_id: str, child_frame_id: str, 
                            timestamp: float) -> TransformStamped:
    """
    Convert CARLA Transform to ROS2 TransformStamped.
    
    Args:
        carla_transform: CARLA Transform object
        frame_id: Parent frame
        child_frame_id: Child frame
        timestamp: CARLA timestamp
        
    Returns:
        geometry_msgs/TransformStamped
    """
    tf = TransformStamped()
    tf.header = create_header(frame_id, timestamp)
    tf.child_frame_id = child_frame_id
    
    # CARLA uses left-handed coordinate system, ROS uses right-handed
    # CARLA: X=forward, Y=right, Z=up
    # ROS: X=forward, Y=left, Z=up
    tf.transform.translation.x = carla_transform.location.x
    tf.transform.translation.y = -carla_transform.location.y  # Flip Y
    tf.transform.translation.z = carla_transform.location.z
    
    # Convert rotation (CARLA uses degrees, ROS uses radians)
    roll = math.radians(carla_transform.rotation.roll)
    pitch = math.radians(carla_transform.rotation.pitch)
    yaw = math.radians(-carla_transform.rotation.yaw)  # Flip yaw for coordinate system
    
    tf.transform.rotation = euler_to_quaternion(roll, pitch, yaw)
    
    return tf


def carla_vehicle_to_odometry(vehicle, frame_id: str = 'map', 
                               child_frame_id: str = 'ego_vehicle') -> Odometry:
    """
    Convert CARLA vehicle state to ROS2 Odometry message.
    
    Args:
        vehicle: CARLA vehicle actor
        frame_id: Parent frame (usually 'map')
        child_frame_id: Vehicle frame
        
    Returns:
        nav_msgs/Odometry message
    """
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    angular_velocity = vehicle.get_angular_velocity()
    
    msg = Odometry()
    msg.header = create_header(frame_id, 0.0)  # Timestamp will be set by caller
    msg.child_frame_id = child_frame_id
    
    # Position (with coordinate system conversion)
    msg.pose.pose.position.x = transform.location.x
    msg.pose.pose.position.y = -transform.location.y  # Flip Y
    msg.pose.pose.position.z = transform.location.z
    
    # Orientation
    roll = math.radians(transform.rotation.roll)
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(-transform.rotation.yaw)  # Flip yaw
    msg.pose.pose.orientation = euler_to_quaternion(roll, pitch, yaw)
    
    # Linear velocity (in vehicle frame would need rotation, keeping in world frame)
    msg.twist.twist.linear.x = velocity.x
    msg.twist.twist.linear.y = -velocity.y  # Flip Y
    msg.twist.twist.linear.z = velocity.z
    
    # Angular velocity
    msg.twist.twist.angular.x = math.radians(angular_velocity.x)
    msg.twist.twist.angular.y = math.radians(-angular_velocity.y)
    msg.twist.twist.angular.z = math.radians(-angular_velocity.z)
    
    return msg
