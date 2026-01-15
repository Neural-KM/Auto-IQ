#!/usr/bin/env python3
"""
Perception Node
===============

ROS2 node for perception (Object Detection + Lane Detection).
Uses the Perception class directly (same as original PerceptionManager).

Initialization Requirements (must all be received before processing starts):
1. TF transform: ego_vehicle -> camera_frame
2. TF transform: ego_vehicle -> lidar_frame
3. CameraInfo message (intrinsics)
4. CARLA connection (for ground truth obstacles)

Publishes:
- /perception/obstacles (ObstacleArray) - detected obstacles from YOLO + LiDAR
- /perception/ground_truth/obstacles (ObstacleArray) - ground truth from CARLA
- /perception/lanes (MarkerArray) - lane lines (visualization-only, polynomial fits)
- /perception/debug/* (debug images)

Note: Lane detection publishes MarkerArray directly because lane polynomial fits
require coordinate transformation that's tightly coupled to the detection logic.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import struct

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time

from autoiq_msgs.msg import ObstacleArray

import tf2_ros
from tf2_ros import TransformException

from perception.core.perception import Perception
from perception.core.lane_detector import LaneDetector
from perception.data_classes.lane import LaneDetection
from perception.core.ground_truth_perception import GroundTruthPerception

import carla


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
        self.model_path = self.declare_parameter('model_path', 'yolov8n.pt').value
        self.conf_threshold = self.declare_parameter('conf_threshold', 0.5).value
        self.obstacle_radius = self.declare_parameter('obstacle_radius', 50.0).value
        self.enable_object_detection = self.declare_parameter('enable_object_detection', True).value
        self.enable_lane_detection = self.declare_parameter('enable_lane_detection', False).value
        self.enable_debug_viz = self.declare_parameter('enable_debug_viz', False).value
        self.perception_method = self.declare_parameter('perception_method', 0).value
        
        # CARLA connection parameters (for ground truth)
        self.declare_parameter('carla_host', 'localhost')
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('carla_timeout', 10.0)
        self.declare_parameter('ground_truth_rate', 10.0)  # Hz for ground truth publishing
        
        carla_host = self.get_parameter('carla_host').value
        carla_port = self.get_parameter('carla_port').value
        carla_timeout = self.get_parameter('carla_timeout').value
        ground_truth_rate = self.get_parameter('ground_truth_rate').value
        
        # TF frame names
        self.camera_frame = self.declare_parameter('camera_frame', 'ego_vehicle/sensor/camera/rgb/front').value
        self.lidar_frame = self.declare_parameter('lidar_frame', 'ego_vehicle/sensor/lidar/top').value

        # Hood mask parameters (normalized 0-1 values)
        self.hood_mask_config = {
            'enabled': self.declare_parameter('hood_mask.enabled', True).value,
            'top_y': self.declare_parameter('hood_mask.top_y', 0.85).value,
            'top_left_x': self.declare_parameter('hood_mask.top_left_x', 0.25).value,
            'top_right_x': self.declare_parameter('hood_mask.top_right_x', 0.75).value,
            'bottom_y': self.declare_parameter('hood_mask.bottom_y', 1.0).value,
            'bottom_left_x': self.declare_parameter('hood_mask.bottom_left_x', 0.08).value,
            'bottom_right_x': self.declare_parameter('hood_mask.bottom_right_x', 0.92).value,
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
        
        # ==================== CARLA Connection (for ground truth) ====================
        self.carla_world = None
        self.ground_truth_perception = None
        try:
            carla_client = carla.Client(carla_host, carla_port)
            carla_client.set_timeout(carla_timeout)
            self.carla_world = carla_client.get_world()
            self.get_logger().info('Connected to CARLA for ground truth obstacles')
        except Exception as e:
            self.get_logger().warn(f'Ground truth obstacles will not be available. ERROR:{e}')

        
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
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
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
        
        # Real perception mode (YOLO + LiDAR)
        if self.enable_object_detection:
            self.get_logger().info('Using Real Perception mode (YOLO + LiDAR)')
            self.rgb_sub = self.create_subscription(
                Image, '/carla/ego_vehicle/sensor/camera/rgb/front/image',
                self.rgb_callback, sensor_qos)
            
            # LiDAR is required for method 0 (frustum filtering)
            self.lidar_sub = self.create_subscription(
                PointCloud2, '/carla/ego_vehicle/sensor/lidar/top/point_cloud',
                self.lidar_callback, sensor_qos)
        
        # ==================== Publishers ====================
        # Ground truth obstacles from CARLA (ObstacleArray for planning)
        self.ground_truth_pub = self.create_publisher(
            ObstacleArray, '/perception/ground_truth/obstacles', 10)
        
        # Detected obstacles from YOLO + LiDAR (ObstacleArray for planning)
        self.obstacles_pub = self.create_publisher(ObstacleArray, '/perception/obstacles', 10)
        self.lanes_pub = self.create_publisher(MarkerArray, '/perception/lanes', 10)
        
        if self.enable_debug_viz:
            self.debug_rgb_pub = self.create_publisher(Image, '/perception/debug/rgb', 10)
            self.debug_bev_pub = self.create_publisher(Image, '/perception/debug/bev', 10)
        
        # ==================== Ground Truth Timer ====================
        # Always publish ground truth at a fixed rate
        self.gt_timer = self.create_timer(1.0 / ground_truth_rate, self._publish_ground_truth_obstacles)
        
        # ==================== Initialization Timer ====================
        # Periodically try to complete initialization
        self.init_timer = self.create_timer(0.5, self._try_initialize)
    
    # ==================== Initialization ====================
    
    def _try_initialize(self):
        """Try to complete initialization by checking for required data."""
        if self.initialized:
            return
        
        # Try to get camera TF
        if not self.camera_tf_received:
            self._try_get_camera_tf()
        
        # Try to get LiDAR TF (always needed for method 0 or ground truth)
        if not self.lidar_tf_received:
            self._try_get_lidar_tf()
        
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
            self.get_logger().warn('CameraInfo not yet received')
            return False
        if not self.camera_tf_received:
            self.get_logger().warn('Camera TF not yet received')
            return False
        if not self.lidar_tf_received:
            self.get_logger().warn('LiDAR TF not yet received')
            return False
        return True
    
    def _complete_initialization(self):
        """Complete initialization once all data is available."""
        self.get_logger().info('All initialization data received, completing setup...')
        
        # Initialize Perception class for object detection
        if self.enable_object_detection:
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
            # Always set LiDAR position for method 0
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
        self.get_logger().info('RGB image received', once=True)
        if not self.initialized or self.perception is None:
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
        
        # Run detection (method 0 always uses LiDAR)
        obstacles = []
        if self.latest_lidar is not None:
            obstacles = self.perception.detect_obstacles(
                rgb_image=img,
                lidar_points=self.latest_lidar,
                radius=self.obstacle_radius
            )
        self._publish_obstacles(obstacles)
        
        if self.enable_debug_viz:
            annotated = self.perception.get_annotated_image()
            if annotated is not None:
                self._publish_debug_rgb(msg, annotated)
    
    def lidar_callback(self, msg: PointCloud2):
        """Store latest LiDAR points."""
        self.get_logger().info('LiDAR point cloud received', once=True)
        if not self.initialized:
            return
        
        
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
    
    def _publish_obstacles(self, obstacles: list):
        """Publish obstacles as ObstacleArray message."""
        obstacle_array = ObstacleArray()
        obstacle_array.header.frame_id = 'ego_vehicle'
        obstacle_array.header.stamp = self.get_clock().now().to_msg()
        
        for obs in obstacles:
            # Obstacles are already Obstacle messages, just update header
            obs.header = obstacle_array.header
            obstacle_array.obstacles.append(obs)
        
        self.obstacles_pub.publish(obstacle_array)
    
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
    
    # ==================== Ground Truth Detection ====================
    
    def _init_ground_truth_perception(self) -> bool:
        """Initialize GroundTruthPerception by finding ego vehicle. Returns True if successful."""
        if self.carla_world is None:
            self.get_logger().warn('No CARLA connection, cannot initialize GroundTruthPerception')
            return False
        
        # Look for vehicle with role_name 'ego_vehicle' or 'hero'
        actors = self.carla_world.get_actors()
        vehicles = actors.filter('vehicle.*')
        
        for vehicle in vehicles:
            if vehicle.attributes.get('role_name') == 'hero':
                self.ground_truth_perception = GroundTruthPerception(vehicle, self.carla_world)
                self.get_logger().info('GroundTruthPerception initialized')
                return True
        
        return False
    
    def _publish_ground_truth_obstacles(self):
        """Publish ground truth obstacles as ObstacleArray."""
        self.get_logger().info('Publishing ground truth obstacles', once=True)
        if self.carla_world is None:
            self.get_logger().warn('No CARLA connection, cannot publish ground truth obstacles')
            return
        
        # Initialize GroundTruthPerception if not done yet
        if self.ground_truth_perception is None:
            if not self._init_ground_truth_perception():
                self.get_logger().warn('Ego vehicle not found in CARLA, cannot publish ground truth obstacles', throttle_duration_sec=5.0)
                return
        
        # Detect obstacles using GroundTruthPerception class
        obstacles = self.ground_truth_perception.detect_obstacles()
        
        # Create ObstacleArray message
        obstacle_array = self._create_obstacle_array(obstacles)
        self.ground_truth_pub.publish(obstacle_array)
    
    def _create_obstacle_array(self, obstacles: list) -> ObstacleArray:
        """Create an ObstacleArray message from obstacle list."""
        obstacle_array = ObstacleArray()
        obstacle_array.header.frame_id = 'ego_vehicle'
        obstacle_array.header.stamp = self.get_clock().now().to_msg()
        
        if not obstacles:
            return obstacle_array
        
        for obs in obstacles:
            # Obstacles are already Obstacle messages, just update header
            obs.header = obstacle_array.header
            obstacle_array.obstacles.append(obs)
        
        return obstacle_array


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
