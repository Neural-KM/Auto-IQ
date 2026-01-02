#!/usr/bin/env python3
"""
Perception Node
===============

ROS2 node for perception (Object Detection + Lane Detection).
Uses the Perception class directly (same as original PerceptionManager).

Initialization Requirements (must all be received before processing starts):
1. TF transform: ego_vehicle -> camera_frame
2. TF transform: ego_vehicle -> lidar_frame (only for frustum filtering method)
3. CameraInfo message (intrinsics)

Publishes:
- /perception/obstacles (MarkerArray with 3D bounding boxes)
- /perception/lanes (MarkerArray with lane lines)
- /perception/debug/* (debug images)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from builtin_interfaces.msg import Time

import tf2_ros
from tf2_ros import TransformException

from perception.perception import Perception
from perception.lane_detector import LaneDetector
from perception.lane import LaneDetection
from perception.obstacle import Obstacle


class PerceptionNode(Node):
    """
    ROS2 Perception Node.
    
    Uses the Perception class directly (same as original PerceptionManager).
    
    Initialization is gated - no processing occurs until:
    - Camera TF transform is received
    - LiDAR TF transform is received (if using frustum filtering)
    - CameraInfo is received
    """
    
    def __init__(self):
        super().__init__('perception_node')
        
        # ==================== Parameters ====================
        # All parameters MUST be provided via YAML config file
        self.use_ground_truth = self.declare_parameter('use_ground_truth').value
        self.model_path = self.declare_parameter('model_path').value
        self.conf_threshold = self.declare_parameter('conf_threshold').value
        self.obstacle_radius = self.declare_parameter('obstacle_radius').value
        self.enable_object_detection = self.declare_parameter('enable_object_detection').value
        self.enable_lane_detection = self.declare_parameter('enable_lane_detection').value
        self.enable_debug_viz = self.declare_parameter('enable_debug_viz').value
        self.perception_method = self.declare_parameter('perception_method').value
        
        # TF frame names
        self.camera_frame = self.declare_parameter('camera_frame').value
        self.lidar_frame = self.declare_parameter('lidar_frame').value

        # Hood mask parameters
        self.hood_mask_config = {
            'enabled': self.declare_parameter('hood_mask.enabled').value,
            'top_y': self.declare_parameter('hood_mask.top_y').value,
            'top_left_x': self.declare_parameter('hood_mask.top_left_x').value,
            'top_right_x': self.declare_parameter('hood_mask.top_right_x').value,
            'bottom_y': self.declare_parameter('hood_mask.bottom_y').value,
            'bottom_left_x': self.declare_parameter('hood_mask.bottom_left_x').value,
            'bottom_right_x': self.declare_parameter('hood_mask.bottom_right_x').value,
        }
        
        # ==================== Initialization State ====================
        # All must be True before processing starts
        self.camera_info_received = False
        self.camera_tf_received = False
        self.lidar_tf_received = False
        self.initialized = False
        
        # Camera intrinsics (from CameraInfo - standard ROS2 approach)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.img_width = None
        self.img_height = None
        self.fov_deg = None
        
        # Camera pose (from TF - standard ROS2 approach)
        self.camera_x = None
        self.camera_y = None
        self.camera_height = None
        self.camera_pitch = None
        
        # LiDAR pose (from TF)
        self.lidar_x = None
        self.lidar_y = None
        self.lidar_z = None
        
        # ==================== TF2 ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== Sensor Data Buffers ====================
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_lidar = None
        
        # ==================== Perception Components (initialized later) ====================
        self.perception = None
        self.lane_detector = None
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # ==================== Subscribers ====================
        # CameraInfo (standard ROS2 for intrinsics)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
            self.camera_info_callback, sensor_qos)
        
        # Semantic camera for lane detection
        if self.enable_lane_detection:
            self.semantic_sub = self.create_subscription(
                Image, '/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
                self.semantic_callback, sensor_qos)
        
        if not self.use_ground_truth:
            # Real perception mode
            self.get_logger().info('Using Real Perception mode (YOLO + Depth/LiDAR)')
            self.rgb_sub = self.create_subscription(
                Image, '/carla/ego_vehicle/sensor/camera/rgb/front/image',
                self.rgb_callback, sensor_qos)
            
            self.depth_sub = self.create_subscription(
                Image, '/carla/ego_vehicle/sensor/camera/depth/front/image',
                self.depth_callback, sensor_qos)
            
            if self.perception_method == Perception.METHOD_FRUSTUM_FILTERING:
                self.lidar_sub = self.create_subscription(
                    PointCloud2, '/carla/ego_vehicle/sensor/lidar/top/point_cloud',
                    self.lidar_callback, sensor_qos)
        
        # ==================== Publishers ====================
        self.obstacles_pub = self.create_publisher(MarkerArray, '/perception/obstacles', 10)
        self.lanes_pub = self.create_publisher(MarkerArray, '/perception/lanes', 10)
        
        if self.enable_debug_viz:
            self.debug_rgb_pub = self.create_publisher(Image, '/perception/debug/rgb', 10)
            self.debug_bev_pub = self.create_publisher(Image, '/perception/debug/bev', 10)
        
        # ==================== Initialization Timer ====================
        # Periodically try to complete initialization
        self.init_timer = self.create_timer(0.5, self._try_initialize)
        
        self.get_logger().info(f'Perception Node created (ground_truth={self.use_ground_truth})')
        self.get_logger().info('Waiting for TF transforms and CameraInfo before starting...')
    
    # ==================== Initialization ====================
    
    def _try_initialize(self):
        """Try to complete initialization by checking for required data."""
        if self.initialized:
            return
        
        # Try to get camera TF
        if not self.camera_tf_received:
            self._try_get_camera_tf()
        
        # Try to get LiDAR TF (only needed for frustum filtering)
        if not self.lidar_tf_received:
            if self.perception_method == Perception.METHOD_FRUSTUM_FILTERING or self.use_ground_truth:
                self._try_get_lidar_tf()
            else:
                # Not needed for depth map method in real perception
                self.lidar_tf_received = True
        
        # Check if ready to initialize
        if self._check_ready():
            self._complete_initialization()
        else:
            self.get_logger().info('Waiting for initialization data...')
    
    def _try_get_camera_tf(self) -> bool:
        """Try to get camera transform from TF."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle', self.camera_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            
            self.camera_x = transform.transform.translation.x
            self.camera_y = transform.transform.translation.y
            self.camera_height = transform.transform.translation.z
            
            # Extract pitch from quaternion
            q = transform.transform.rotation
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            self.camera_pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
            
            self.camera_tf_received = True
            self.get_logger().info(
                f'Camera TF received: x={self.camera_x:.2f}, y={self.camera_y:.2f}, '
                f'z={self.camera_height:.2f}m, pitch={self.camera_pitch:.1f}°')
            return True
        except TransformException as e:
            self.get_logger().warn(f'Camera TF lookup failed: {e}')
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
                f'LiDAR TF received: x={self.lidar_x:.2f}, y={self.lidar_y:.2f}, z={self.lidar_z:.2f}m')
            return True
        except TransformException:
            return False
    
    def _check_ready(self) -> bool:
        """Check if all required initialization data has been received."""
        if not self.camera_info_received:
            self.get_logger().info('CameraInfo not yet received')
            return False
        if not self.camera_tf_received:
            self.get_logger().info('Camera TF not yet received')
            return False
        if not self.lidar_tf_received:
            self.get_logger().info('LiDAR TF not yet received')
            return False
        return True
    
    def _complete_initialization(self):
        """Complete initialization once all data is available."""
        self.get_logger().info('All initialization data received, completing setup...')
        
        # Initialize Perception class (for real perception mode)
        if self.enable_object_detection and not self.use_ground_truth:
            self.perception = Perception(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                method=self.perception_method,
                hood_mask_config=self.hood_mask_config
            )
            self.perception.set_intrinsics(self.fx, self.fy, self.cx, self.cy)
            self.perception.set_camera_position(
                x=self.camera_x, y=self.camera_y, z=self.camera_height,
                pitch=self.camera_pitch
            )
            if self.perception_method == Perception.METHOD_FRUSTUM_FILTERING:
                self.perception.set_lidar_position(
                    x=self.lidar_x, y=self.lidar_y, z=self.lidar_z
                )
            self.get_logger().info(f'Perception initialized: {self.model_path}, method={self.perception_method}')
        
        # Initialize Lane Detector
        if self.enable_lane_detection:
            self.lane_detector = LaneDetector(
                image_width=self.img_width,
                image_height=self.img_height,
                camera_fov=self.fov_deg,
                camera_pitch=self.camera_pitch,
                camera_height=self.camera_height
            )
            self.get_logger().info(
                f'Lane detector initialized: {self.img_width}x{self.img_height}, FOV={self.fov_deg:.1f}°')
        
        # Cancel init timer
        self.init_timer.cancel()
        
        self.initialized = True
        self.get_logger().info('Perception Node fully initialized and ready!')
    
    # ==================== CameraInfo Callback ====================
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics from CameraInfo (standard ROS2 approach)."""
        if self.camera_info_received:
            return
        
        self.img_width = msg.width
        self.img_height = msg.height
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        
        # Compute FOV from intrinsics
        if self.fx > 0:
            fov_rad = 2 * np.arctan(self.img_width / (2 * self.fx))
            self.fov_deg = np.degrees(fov_rad)
        else:
            self.fov_deg = 90.0
        
        self.camera_info_received = True
        self.get_logger().info(
            f'CameraInfo received: {self.img_width}x{self.img_height}, '
            f'fx={self.fx:.1f}, fy={self.fy:.1f}, FOV={self.fov_deg:.1f}°')
    
    # ==================== Sensor Callbacks ====================
    
    def rgb_callback(self, msg: Image):
        """Process RGB image for object detection."""
        self.get_logger().info('RGB image received for object detection', throttle_duration_sec=1.0)
        if not self.initialized or self.perception is None:
            self.get_logger().warn(f'Init {self.initialized}, perception {self.perception}')
            return
        
        # Convert ROS Image to numpy
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1]
        else:
            self.get_logger().warn(f'Unsupported RGB encoding: {msg.encoding}')
            return
        
        self.latest_rgb = img
        
        # Run detection
        obstacles = []
        if self.perception_method == Perception.METHOD_DEPTH_MAP:
            if self.latest_depth is not None:
                obstacles = self.perception.detect_obstacles(
                    rgb_image=img,
                    depth_image=self.latest_depth,
                    radius=self.obstacle_radius
                )
        else:
            if self.latest_lidar is not None:
                obstacles = self.perception.detect_obstacles(
                    rgb_image=img,
                    lidar_points=self.latest_lidar,
                    radius=self.obstacle_radius
                )
        if obstacles:
            self._publish_obstacle_markers(obstacles)
        
        if self.enable_debug_viz:
            annotated = self.perception.get_annotated_image()
            if annotated is not None:
                self._publish_debug_rgb(msg, annotated)
    
    def depth_callback(self, msg: Image):
        """Store latest depth image."""
        if not self.initialized:
            return
        
        if msg.encoding == '32FC1':
            self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
    
    def lidar_callback(self, msg: PointCloud2):
        """Store latest LiDAR points."""
        self.get_logger().info('LiDAR point cloud received for frustum filtering', throttle_duration_sec=1.0)
        if not self.initialized:
            return
        
        import struct
        
        point_step = msg.point_step
        data = msg.data
        num_points = len(data) // point_step
        
        if num_points == 0:
            return
        
        points = np.zeros((num_points, 3), dtype=np.float32)
        for i in range(num_points):
            offset = i * point_step
            points[i, 0] = struct.unpack('f', data[offset:offset+4])[0]
            points[i, 1] = struct.unpack('f', data[offset+4:offset+8])[0]
            points[i, 2] = struct.unpack('f', data[offset+8:offset+12])[0]
        
        self.latest_lidar = points
    
    def semantic_callback(self, msg: Image):
        """Process semantic image for lane detection."""
        if not self.initialized or self.lane_detector is None:
            return
        
        if msg.encoding == 'mono8':
            semantic_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        else:
            self.get_logger().warn(f'Unsupported semantic encoding: {msg.encoding}')
            return
        
        result = self.lane_detector.detect(semantic_img)
        self._publish_lane_markers(result)
        
        if self.enable_debug_viz:
            self._publish_debug_bev()
    
    # ==================== Publishers ====================
    
    def _publish_obstacle_markers(self, obstacles: list):
        """Publish 3D bounding box markers for obstacles."""
        marker_array = MarkerArray()
        
        # Use timestamp 0 to tell RViz to use the latest available TF
        # This avoids "extrapolation into future" errors when clocks differ slightly
        timestamp = Time(sec=0, nanosec=0)
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'ego_vehicle'
        clear_marker.header.stamp = timestamp
        clear_marker.ns = 'obstacles'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = 'ego_vehicle'
            marker.header.stamp = timestamp
            marker.ns = 'obstacles'
            marker.id = i + 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = obs.z
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(obs.yaw / 2)
            marker.pose.orientation.w = math.cos(obs.yaw / 2)
            
            marker.scale = Vector3(x=obs.length, y=obs.width, z=obs.height)
            
            if obs.obstacle_type == 'vehicle':
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)
            else:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)
            
            marker.lifetime.nanosec = 200000000
            marker_array.markers.append(marker)
            
            # Text label
            text_marker = Marker()
            text_marker.header.frame_id = 'ego_vehicle'
            text_marker.header.stamp = timestamp
            text_marker.ns = 'obstacle_labels'
            text_marker.id = i + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = obs.x
            text_marker.pose.position.y = obs.y
            text_marker.pose.position.z = obs.z + obs.height / 2 + 0.5
            
            distance = (obs.x**2 + obs.y**2)**0.5
            text_marker.text = f"{obs.obstacle_type}: {distance:.1f}m"
            text_marker.scale.z = 0.5
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.lifetime.nanosec = 200000000
            marker_array.markers.append(text_marker)
        
        self.obstacles_pub.publish(marker_array)
    
    def _publish_lane_markers(self, result: LaneDetection):
        """Publish lane markers."""
        marker_array = MarkerArray()
        
        # Use timestamp 0 to tell RViz to use the latest available TF
        timestamp = Time(sec=0, nanosec=0)
        
        clear_marker = Marker()
        clear_marker.header.frame_id = 'ego_vehicle'
        clear_marker.header.stamp = timestamp
        clear_marker.ns = 'lanes'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        if not result.valid:
            self.lanes_pub.publish(marker_array)
            return
        
        bev_params = result.bev_params
        
        if result.left_fit is not None:
            left_marker = self._create_lane_marker(
                result.left_fit, bev_params, 1, ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8), timestamp)
            marker_array.markers.append(left_marker)
        
        if result.right_fit is not None:
            right_marker = self._create_lane_marker(
                result.right_fit, bev_params, 2, ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8), timestamp)
            marker_array.markers.append(right_marker)
        
        self.lanes_pub.publish(marker_array)
    
    def _create_lane_marker(self, fit, bev_params, marker_id, color, timestamp):
        """Create a lane line marker."""
        marker = Marker()
        marker.header.frame_id = 'ego_vehicle'
        marker.header.stamp = timestamp
        marker.ns = 'lanes'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.color = color
        
        for y_bev in np.linspace(0, bev_params['height'] - 1, 20):
            x_bev = np.polyval(fit, y_bev)
            x_world = bev_params['y_range'][1] - (y_bev / bev_params['height']) * (bev_params['y_range'][1] - bev_params['y_range'][0])
            y_world = bev_params['x_range'][0] + (x_bev / bev_params['width']) * (bev_params['x_range'][1] - bev_params['x_range'][0])
            pt = Point(x=float(x_world), y=float(-y_world), z=0.0)
            marker.points.append(pt)
        
        return marker
    
    # ==================== Debug Visualization ====================
    
    def _publish_debug_rgb(self, msg: Image, annotated: np.ndarray):
        """Publish debug RGB image with bboxes."""
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = annotated.shape[0]
        out_msg.width = annotated.shape[1]
        out_msg.encoding = 'rgb8'
        out_msg.step = annotated.shape[1] * 3
        out_msg.data = annotated.tobytes()
        self.debug_rgb_pub.publish(out_msg)
    
    def _publish_debug_bev(self):
        """Publish debug BEV image."""
        if self.lane_detector is None:
            return
        debug_images = self.lane_detector.get_debug_images()
        lane_fit_img = debug_images.get('lane_fit')
        if lane_fit_img is not None:
            out_msg = Image()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.header.frame_id = 'ego_vehicle'
            out_msg.height = lane_fit_img.shape[0]
            out_msg.width = lane_fit_img.shape[1]
            out_msg.encoding = 'bgr8'
            out_msg.step = lane_fit_img.shape[1] * 3
            out_msg.data = lane_fit_img.tobytes()
            self.debug_bev_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
