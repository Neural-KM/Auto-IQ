#!/usr/bin/env python3
"""
Semantic LiDAR Labeling Node
=============================

ROS2 node that projects semantic segmentation labels onto LiDAR point clouds.
Uses ground truth semantic camera data to label each 3D point with its semantic class.

Pipeline:
1. Receive LiDAR point cloud and semantic segmentation image
2. Transform LiDAR points to camera frame
3. Project 3D points to 2D image coordinates
4. Look up semantic label at each projected pixel
5. Publish colored point cloud for visualization

Subscribes:
- /carla/ego_vehicle/sensor/lidar/top/point_cloud (sensor_msgs/PointCloud2) - LiDAR data
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (sensor_msgs/Image) - Semantic segmentation
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info (sensor_msgs/CameraInfo) - Camera intrinsics

Publishes:
- /perception/semantic_pointcloud (sensor_msgs/PointCloud2) - Labeled colored point cloud

Note: Only points visible in the camera's field of view will be labeled.
Points behind the camera or outside the image bounds are discarded.
"""

import math
import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header

import tf2_ros
from tf2_ros import TransformException

from perception.data_classes.semantic_labels import SEMANTIC_CLASSES


class SemanticPointCloudNode(Node):
    """
    ROS2 Node for labeling LiDAR points with semantic segmentation data.
    
    Projects LiDAR points onto the semantic segmentation image and assigns
    colors based on the semantic class at each projected pixel location.
    """
    
    def __init__(self):
        super().__init__('semantic_pointcloud_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('lidar_topic', '/carla/ego_vehicle/sensor/lidar/top/point_cloud')
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('camera_info_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info')
        self.declare_parameter('output_topic', '/perception/semantic_pointcloud')
        self.declare_parameter('camera_frame', 'ego_vehicle/sensor/camera/rgb/front')
        self.declare_parameter('lidar_frame', 'ego_vehicle/sensor/lidar/top')
        self.declare_parameter('max_range', 100.0)  # Max range in meters
        
        lidar_topic = self.get_parameter('lidar_topic').value
        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.max_range = self.get_parameter('max_range').value
        
        # ==================== State ====================
        self.camera_info_received = False
        self.camera_tf_received = False
        self.lidar_tf_received = False
        self.initialized = False
        
        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.img_width = None
        self.img_height = None
        
        # Camera pose (from TF)
        self.camera_x = None
        self.camera_y = None
        self.camera_z = None
        
        # LiDAR pose (from TF)
        self.lidar_x = None
        self.lidar_y = None
        self.lidar_z = None
        
        # Latest sensor data
        self.latest_semantic = None
        self.latest_lidar = None
        
        # ==================== TF2 ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # ==================== Subscribers ====================
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic,
            self.camera_info_callback, sensor_qos)
        
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic,
            self.semantic_callback, sensor_qos)
        
        self.lidar_sub = self.create_subscription(
            PointCloud2, lidar_topic,
            self.lidar_callback, sensor_qos)
        
        # ==================== Publishers ====================
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, output_topic, 10)
        
        # ==================== Initialization Timer ====================
        self.init_timer = self.create_timer(0.5, self._check_initialization)
        
        self.get_logger().info('Semantic PointCloud Node started')
        self.get_logger().info(f'  LiDAR: {lidar_topic}')
        self.get_logger().info(f'  Semantic: {semantic_topic}')
        self.get_logger().info(f'  Output: {output_topic}')
    
    # ==================== Initialization ====================
    
    def _check_initialization(self):
        """Check if all required data is available."""
        if self.initialized:
            return
        
        # Try to get TF transforms
        if not self.camera_tf_received:
            self._try_get_camera_tf()
        
        if not self.lidar_tf_received:
            self._try_get_lidar_tf()
        
        # Check if ready
        if self.camera_info_received and self.camera_tf_received and self.lidar_tf_received:
            self.initialized = True
            self.init_timer.cancel()
            self.get_logger().info('Semantic PointCloud Node fully initialized!')
    
    def _try_get_camera_tf(self) -> bool:
        """Try to get camera transform from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle', self.camera_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            
            self.camera_x = transform.transform.translation.x
            self.camera_y = transform.transform.translation.y
            self.camera_z = transform.transform.translation.z
            
            self.camera_tf_received = True
            self.get_logger().info(
                f'Camera TF: x={self.camera_x:.2f}, y={self.camera_y:.2f}, z={self.camera_z:.2f}')
            return True
        except TransformException:
            return False
    
    def _try_get_lidar_tf(self) -> bool:
        """Try to get LiDAR transform from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle', self.lidar_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
            
            self.lidar_x = transform.transform.translation.x
            self.lidar_y = transform.transform.translation.y
            self.lidar_z = transform.transform.translation.z
            
            self.lidar_tf_received = True
            self.get_logger().info(
                f'LiDAR TF: x={self.lidar_x:.2f}, y={self.lidar_y:.2f}, z={self.lidar_z:.2f}')
            return True
        except TransformException:
            return False
    
    # ==================== Callbacks ====================
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics."""
        if self.camera_info_received:
            return
        
        self.img_width = msg.width
        self.img_height = msg.height
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        
        self.camera_info_received = True
        self.get_logger().info(
            f'CameraInfo: {self.img_width}x{self.img_height}, fx={self.fx:.1f}')
    
    def semantic_callback(self, msg: Image):
        """Store latest semantic segmentation image."""
        # Convert to numpy array
        if msg.encoding == 'bgra8':
            # CARLA semantic camera uses BGRA8 where R channel contains class ID
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            # Class ID is in the R channel (index 2 for BGRA)
            self.latest_semantic = img[:, :, 2].copy()
        elif msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            # Assume class ID is in R channel
            self.latest_semantic = img[:, :, 0].copy()
        elif msg.encoding == 'mono8':
            self.latest_semantic = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width).copy()
        else:
            self.get_logger().warn(f'Unsupported semantic encoding: {msg.encoding}')
            return
        
        # Process if we have both semantic and lidar
        if self.initialized and self.latest_lidar is not None:
            self._process_and_publish()
    
    def lidar_callback(self, msg: PointCloud2):
        """Store latest LiDAR point cloud."""
        # Parse point cloud
        self.latest_lidar = self._parse_pointcloud2(msg)
        self.latest_lidar_header = msg.header
        
        # Process if we have both semantic and lidar
        if self.initialized and self.latest_semantic is not None:
            self._process_and_publish()
    
    # ==================== Processing ====================
    
    def _process_and_publish(self):
        """Project LiDAR points, label with semantic classes, and publish."""
        if self.latest_lidar is None or self.latest_semantic is None:
            return
        
        lidar_points = self.latest_lidar
        semantic_img = self.latest_semantic
        
        if len(lidar_points) == 0:
            return
        
        # Take only x, y, z
        points_xyz = lidar_points[:, :3] if lidar_points.shape[1] >= 3 else lidar_points
        
        # 1. Transform LiDAR points to Vehicle Frame
        lidar_offset = np.array([self.lidar_x, self.lidar_y, self.lidar_z])
        points_veh = points_xyz + lidar_offset
        
        # 2. Transform to Camera Frame
        camera_offset = np.array([self.camera_x, self.camera_y, self.camera_z])
        pts_rel = points_veh - camera_offset
        
        # Rotate: Vehicle (X-forward, Y-left, Z-up) -> Camera (X-right, Y-down, Z-forward)
        points_cam = np.zeros_like(pts_rel)
        points_cam[:, 0] = -pts_rel[:, 1]  # cam_x = -veh_y
        points_cam[:, 1] = -pts_rel[:, 2]  # cam_y = -veh_z
        points_cam[:, 2] = pts_rel[:, 0]   # cam_z = veh_x
        
        # 3. Filter points in front of camera
        front_mask = points_cam[:, 2] > 0.5  # At least 0.5m in front
        points_cam_front = points_cam[front_mask]
        points_veh_front = points_veh[front_mask]
        
        if len(points_cam_front) == 0:
            return
        
        # 4. Project to image plane
        z = points_cam_front[:, 2]
        u = (points_cam_front[:, 0] * self.fx / z) + self.cx
        v = (points_cam_front[:, 1] * self.fy / z) + self.cy
        
        # 5. Filter points within image bounds
        in_image_mask = (
            (u >= 0) & (u < self.img_width) &
            (v >= 0) & (v < self.img_height)
        )
        
        u_valid = u[in_image_mask].astype(np.int32)
        v_valid = v[in_image_mask].astype(np.int32)
        points_veh_valid = points_veh_front[in_image_mask]
        
        if len(u_valid) == 0:
            return
        
        # 6. Look up semantic class at each pixel
        class_ids = semantic_img[v_valid, u_valid]
        
        # 7. Create and publish labeled point cloud (XYZ + label)
        labeled_cloud = self._create_labeled_pointcloud2(
            points_veh_valid, class_ids, self.latest_lidar_header.stamp)
        
        self.pointcloud_pub.publish(labeled_cloud)
    
    def _parse_pointcloud2(self, msg: PointCloud2) -> np.ndarray:
        """Parse PointCloud2 message to numpy array."""
        # Get field offsets
        field_names = [f.name for f in msg.fields]
        
        # Standard field offsets for XYZI point cloud
        x_offset = 0
        y_offset = 4
        z_offset = 8
        
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        # Parse points
        point_step = msg.point_step
        num_points = msg.width * msg.height
        
        points = np.zeros((num_points, 3), dtype=np.float32)
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        for i in range(num_points):
            offset = i * point_step
            points[i, 0] = struct.unpack('f', data[offset + x_offset:offset + x_offset + 4])[0]
            points[i, 1] = struct.unpack('f', data[offset + y_offset:offset + y_offset + 4])[0]
            points[i, 2] = struct.unpack('f', data[offset + z_offset:offset + z_offset + 4])[0]
        
        # Filter invalid points (NaN, Inf, or at origin)
        valid_mask = (
            np.isfinite(points).all(axis=1) &
            (np.abs(points).sum(axis=1) > 0.1)
        )
        
        return points[valid_mask]
    
    def _create_labeled_pointcloud2(
        self, 
        points: np.ndarray, 
        labels: np.ndarray,
        stamp
    ) -> PointCloud2:
        """
        Create a PointCloud2 message with XYZL (label) fields.
        
        Args:
            points: Nx3 array of XYZ coordinates
            labels: N array of semantic class IDs (uint8)
            stamp: Timestamp for the message
            
        Returns:
            PointCloud2 message with x, y, z, label fields
        """
        n_points = len(points)
        
        # Create header
        header = Header()
        header.stamp = stamp
        header.frame_id = 'ego_vehicle'
        
        # Define fields: x, y, z, label
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # Pack data: x, y, z, label
        point_step = 16  # 3 floats + 1 uint32 = 16 bytes
        data = bytearray(n_points * point_step)
        
        for i in range(n_points):
            offset = i * point_step
            struct.pack_into('fffI', data, offset,
                             points[i, 0], points[i, 1], points[i, 2],
                             int(labels[i]))
        
        # Create message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = n_points
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.row_step = point_step * n_points
        msg.data = bytes(data)
        msg.is_dense = True
        
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SemanticPointCloudNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
