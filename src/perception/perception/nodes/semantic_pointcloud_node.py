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
5. Publish labeled point cloud (XYZ + label)

Subscribes:
- /carla/ego_vehicle/sensor/lidar/top/point_cloud (sensor_msgs/PointCloud2) - LiDAR data
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (sensor_msgs/Image) - Semantic segmentation
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info (sensor_msgs/CameraInfo) - Camera intrinsics

Publishes:
- /perception/semantic_pointcloud (sensor_msgs/PointCloud2) - Labeled point cloud (XYZ + label)

Note: Only points visible in the camera's field of view will be labeled.
Points behind the camera or outside the image bounds are discarded.
"""

import struct
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header

from autoiq_common import CameraParams, CameraParamsProvider

import tf2_ros
from tf2_ros import TransformException

from cv_bridge import CvBridge

# CARLA Semantic Class IDs
SEMANTIC_ROAD = 1
SEMANTIC_VEHICLE = 14
SEMANTIC_VEHICLES = {14, 15, 16, 17, 18, 19}  # vehicle, truck, bus, train, motorcycle, bicycle

# Import for debug visualization
import cv2



class SemanticPointCloudNode(Node):
    """
    ROS2 Node for labeling LiDAR points with semantic segmentation data.
    
    Projects LiDAR points onto the semantic segmentation image and assigns
    labels based on the semantic class at each projected pixel location.
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
        self.declare_parameter('sample_radius', 2)  # Pixel radius for neighborhood sampling (0 = single pixel)
        self.declare_parameter('publish_debug_image', True)  # Publish debug visualization
        
        # Ego hood mask parameters (in vehicle frame: X-forward, Y-left, Z-up)
        # Points within this thin ground-level cuboid are reclassified as road
        # This handles the case where LiDAR shoots through the hood and hits the ground,
        # but the camera sees the hood, causing ground points to be labeled as "vehicle"
        
        self.declare_parameter('hood_x_min', 3)    # Start at vehicle center
        self.declare_parameter('hood_x_max', 5)    # Hood extends ~2.5m forward
        self.declare_parameter('hood_y_min', -2.2)   # Right edge
        self.declare_parameter('hood_y_max', 2.2)    # Left edge
        self.declare_parameter('hood_z_min', -0.1)   # Well below ground to catch slopes
        self.declare_parameter('hood_z_max', 0.1)    # Just above ground (keeps real obstacles on hood)
        
        lidar_topic = self.get_parameter('lidar_topic').value
        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.max_range = self.get_parameter('max_range').value
        self.sample_radius = self.get_parameter('sample_radius').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value
        
        # Hood mask bounds (3D cuboid in vehicle frame)
        self.hood_x_min = self.get_parameter('hood_x_min').value
        self.hood_x_max = self.get_parameter('hood_x_max').value
        self.hood_y_min = self.get_parameter('hood_y_min').value
        self.hood_y_max = self.get_parameter('hood_y_max').value
        self.hood_z_min = self.get_parameter('hood_z_min').value
        self.hood_z_max = self.get_parameter('hood_z_max').value
        
        # ==================== State ====================
        self.lidar_tf_received = False
        self.initialized = False

        # LiDAR pose (from TF)
        self.lidar_x = None
        self.lidar_y = None
        self.lidar_z = None
        
        # Latest sensor data
        self.latest_semantic = None
        self.latest_lidar = None
        
        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()

        # ==================== TF2 (for LiDAR only - camera handled by CameraParamsProvider) ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ==================== Camera Parameters (intrinsics + extrinsics) ====================
        self.camera = CameraParamsProvider(
            node=self,
            camera_info_topic=camera_info_topic,
            base_frame='ego_vehicle',
            camera_frame=self.camera_frame,
            on_ready=self._on_camera_ready,
            qos=sensor_qos,
        )

        # ==================== Subscribers ====================
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic,
            self.semantic_callback, sensor_qos)

        self.lidar_sub = self.create_subscription(
            PointCloud2, lidar_topic,
            self.lidar_callback, sensor_qos)
        
        # ==================== Publishers ====================
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, output_topic, 10)

        # Debug visualization publisher
        self.debug_image_pub = self.create_publisher(
            Image, '/perception/semantic_pointcloud_debug', sensor_qos)
        
        # ==================== Initialization Timer ====================
        self.init_timer = self.create_timer(0.5, self._check_initialization)
        
        self.get_logger().info('Semantic PointCloud Node started')
        self.get_logger().info(f'  LiDAR: {lidar_topic}')
        self.get_logger().info(f'  Semantic: {semantic_topic}')
        self.get_logger().info(f'  Output: {output_topic}')
    
    # ==================== Initialization ====================

    def _on_camera_ready(self, params: CameraParams):
        """Called when camera intrinsics and extrinsics are available."""
        self.get_logger().info(
            f'Camera ready: {params.width}x{params.height}, '
            f'fx={params.fx:.1f}, pos=({params.x_m:.2f}, {params.y_m:.2f}, {params.height_m:.2f})')
        self._try_complete_initialization()

    def _check_initialization(self):
        """Check if all required data is available."""
        if self.initialized:
            return

        # Try to get camera extrinsics (if intrinsics already received)
        if self.camera.intrinsics_received and not self.camera.extrinsics_received:
            self.camera.try_get_extrinsics()

        if not self.lidar_tf_received:
            self._try_get_lidar_tf()

        self._try_complete_initialization()

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

    def _try_complete_initialization(self):
        """Complete initialization once all data is available."""
        if self.initialized:
            return

        if not self.camera.ready:
            return
        if not self.lidar_tf_received:
            return

        self.initialized = True
        self.init_timer.cancel()
        self.get_logger().info('Semantic PointCloud Node fully initialized!')

    # ==================== Callbacks ====================

    def semantic_callback(self, msg: Image):
        """Store latest semantic segmentation image."""
        # Convert to numpy array via cv_bridge
        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return
        
        # Extract class IDs: multi-channel images store class ID in R channel
        if img.ndim == 3:
            self.latest_semantic = (img[:, :, 2] if img.shape[2] == 4 else img[:, :, 0]).copy()
        else:
            self.latest_semantic = img.copy()
        
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
        params = self.camera.params

        if len(lidar_points) == 0:
            return

        # Take only x, y, z
        points_xyz = lidar_points[:, :3] if lidar_points.shape[1] >= 3 else lidar_points

        # 1. Transform LiDAR points to Vehicle Frame
        lidar_offset = np.array([self.lidar_x, self.lidar_y, self.lidar_z])
        points_veh = points_xyz + lidar_offset

        # 2. Transform to Camera Frame
        camera_offset = np.array([params.x_m, params.y_m, params.height_m])
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

        # 4. Project to image plane (keep float for sub-pixel accuracy)
        z = points_cam_front[:, 2]
        u_float = (points_cam_front[:, 0] * params.fx / z) + params.cx
        v_float = (points_cam_front[:, 1] * params.fy / z) + params.cy

        # 5. Filter points within image bounds (with margin for neighborhood sampling)
        # Add +1 to margin to account for rounding (e.g., 1277.6 rounds to 1278)
        margin = self.sample_radius + 1
        in_image_mask = (
            (u_float >= margin) & (u_float < params.width - margin) &
            (v_float >= margin) & (v_float < params.height - margin)
        )
        
        u_valid_float = u_float[in_image_mask]
        v_valid_float = v_float[in_image_mask]
        points_veh_valid = points_veh_front[in_image_mask]
        
        if len(u_valid_float) == 0:
            return
        
        # 6. Look up semantic class using neighborhood sampling
        # This fixes boundary issues where single-pixel lookup can pick wrong class
        class_ids = self._sample_semantic_labels(
            semantic_img, u_valid_float, v_valid_float, self.sample_radius)
        
        # 7. Apply ego hood mask - reclassify points in hood region as road
        # This prevents ground points under the hood (misclassified due to camera
        # seeing hood while LiDAR shoots through it) from appearing as obstacles
        hood_mask = (
            (points_veh_valid[:, 0] >= self.hood_x_min) &
            (points_veh_valid[:, 0] <= self.hood_x_max) &
            (points_veh_valid[:, 1] >= self.hood_y_min) &
            (points_veh_valid[:, 1] <= self.hood_y_max) &
            (points_veh_valid[:, 2] >= self.hood_z_min) &
            (points_veh_valid[:, 2] <= self.hood_z_max)
        )
        class_ids[hood_mask] = SEMANTIC_ROAD
        
        # 8. Publish debug visualization if enabled
        if self.publish_debug_image:
            self._publish_debug_image(
                semantic_img, u_valid_float, v_valid_float, class_ids, points_veh_valid)
        
        # 9. Create and publish labeled point cloud (XYZ + label)
        labeled_cloud = self._create_labeled_pointcloud2(
            points_veh_valid, class_ids, self.latest_lidar_header.stamp)
        self.pointcloud_pub.publish(labeled_cloud)
    
    def _sample_semantic_labels(
        self,
        semantic_img: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        radius: int
    ) -> np.ndarray:
        """
        Sample semantic labels using neighborhood voting.
        
        Instead of taking a single pixel, sample a small neighborhood and use
        majority voting. This handles boundary cases where a point projects
        near the edge between two semantic classes.
        
        Args:
            semantic_img: HxW semantic segmentation image (class IDs)
            u: N array of x pixel coordinates (float)
            v: N array of y pixel coordinates (float)
            radius: Pixel radius for neighborhood (0 = single pixel)
            
        Returns:
            N array of semantic class IDs
        """
        n_points = len(u)
        
        # Round to nearest pixel for center
        u_center = np.round(u).astype(np.int32)
        v_center = np.round(v).astype(np.int32)
        
        if radius == 0:
            # Simple single-pixel lookup
            return semantic_img[v_center, u_center]
        
        # Create sampling offsets for the neighborhood
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append((dy, dx))
        n_samples = len(offsets)
        
        # Vectorized: gather all neighborhood samples at once
        # Shape: (n_points, n_samples)
        all_labels = np.zeros((n_points, n_samples), dtype=np.uint8)
        for idx, (dy, dx) in enumerate(offsets):
            all_labels[:, idx] = semantic_img[v_center + dy, u_center + dx]
        
        # Vectorized majority vote using bincount
        class_ids = np.zeros(n_points, dtype=np.uint8)
        for i in range(n_points):
            class_ids[i] = np.bincount(all_labels[i]).argmax()
        
        return class_ids
    
    def _publish_debug_image(
        self,
        semantic_img: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        class_ids: np.ndarray,
        points_veh: np.ndarray
    ):
        """
        Publish a debug image showing projected LiDAR points on semantic image.
        
        Only points classified as vehicle are drawn (for debugging misclassification).
        Points are colored by distance: red (near) -> green (far).
        """
        # Create RGB visualization from semantic image
        from perception.data_classes.semantic_labels import SEMANTIC_COLOR_LUT
        debug_img = SEMANTIC_COLOR_LUT[semantic_img].copy()
        
        # Draw only vehicle-classified points
        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)
        
        # Filter to only vehicle classes
        vehicle_mask = np.isin(class_ids, list(SEMANTIC_VEHICLES))
        vehicle_indices = np.where(vehicle_mask)[0]
        
        if len(vehicle_indices) > 0:
            # Compute distances for vehicle points
            distances = np.sqrt(
                points_veh[vehicle_indices, 0]**2 + 
                points_veh[vehicle_indices, 1]**2
            )
            
            # Normalize distance to [0, 1] for color gradient
            # Near (0-5m) = red, Far (>30m) = green
            min_dist, max_dist = 5.0, 30.0
            norm_dist = np.clip((distances - min_dist) / (max_dist - min_dist), 0, 1)
            
            for idx, i in enumerate(vehicle_indices):
                # Interpolate color: red (255,0,0) -> green (0,255,0)
                t = norm_dist[idx]
                r = int(255 * (1 - t))
                g = int(255 * t)
                color = (r, g, 0)  # RGB
                
                # Draw point with white outline
                cv2.circle(debug_img, (u_int[i], v_int[i]), 4, (255, 255, 255), -1)
                cv2.circle(debug_img, (u_int[i], v_int[i]), 3, color, -1)
        
        # Publish as ROS Image
        msg = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        
        self.debug_image_pub.publish(msg)
    
    def _parse_pointcloud2(self, msg: PointCloud2) -> np.ndarray:
        """Parse PointCloud2 message to numpy array."""
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
