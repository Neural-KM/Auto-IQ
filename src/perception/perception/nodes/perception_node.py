#!/usr/bin/env python3
"""
Perception Node
===============

ROS2 node for perception (Object Detection).
Uses the Perception class directly (same as original PerceptionManager).

Initialization Requirements (must all be received before processing starts):
1. Camera intrinsics (CameraInfo) + extrinsics (TF) via CameraParamsProvider
2. TF transform: ego_vehicle -> lidar_frame
3. CARLA connection (for ground truth obstacles)

Publishes:
- /perception/obstacles (ObstacleArray) - detected obstacles from YOLO + LiDAR
- /perception/ground_truth/obstacles (ObstacleArray) - ground truth from CARLA
- /perception/debug/* (debug images)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import struct


from sensor_msgs.msg import Image, PointCloud2

from autoiq_msgs.msg import ObstacleArray
from autoiq_common import CameraParams, CameraParamsProvider

from cv_bridge import CvBridge
import tf2_ros
from tf2_ros import TransformException

from perception.core.perception import Perception
from perception.core.ground_truth_perception import GroundTruthPerception

import carla


class PerceptionNode(Node):
    """
    ROS2 Perception Node.

    Uses the Perception class directly (same as original PerceptionManager).

    Initialization is gated - no processing occurs until:
    - Camera params are ready (intrinsics + extrinsics via CameraParamsProvider)
    - LiDAR TF transform is received (if using frustum filtering)
    """
    
    def __init__(self):
        super().__init__('perception_node')
        
        # ==================== Parameters ====================
        self.model_path = self.declare_parameter('model_path', 'yolov8n.pt').value
        self.conf_threshold = self.declare_parameter('conf_threshold', 0.5).value
        self.obstacle_radius = self.declare_parameter('obstacle_radius', 50.0).value
        self.enable_object_detection = self.declare_parameter('enable_object_detection', True).value
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
        self.lidar_tf_received = False
        self.initialized = False

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

        
        # ==================== TF2 (for LiDAR only - camera handled by CameraParamsProvider) ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # ==================== Sensor Data Buffers ====================
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_lidar = None
        
        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()
        
        # ==================== Perception Components (initialized later) ====================
        self.perception = None
        
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
            camera_info_topic='/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
            base_frame='ego_vehicle',
            camera_frame=self.camera_frame,
            on_ready=self._on_camera_ready,
            qos=sensor_qos,
        )

        # ==================== Subscribers ====================
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
        
        if self.enable_debug_viz:
            self.debug_rgb_pub = self.create_publisher(Image, '/perception/debug/rgb', sensor_qos)
        
        # ==================== Ground Truth Timer ====================
        # Always publish ground truth at a fixed rate
        self.gt_timer = self.create_timer(1.0 / ground_truth_rate, self._publish_ground_truth_obstacles)
        
        # ==================== Initialization Timer ====================
        # Periodically try to complete initialization
        self.init_timer = self.create_timer(0.5, self._try_initialize)
    
    # ==================== Initialization ====================

    def _on_camera_ready(self, params: CameraParams):
        """Called when camera intrinsics and extrinsics are available."""
        self.get_logger().info(
            f'Camera ready: {params.width}x{params.height}, '
            f'FOV={params.fov_deg:.1f}deg, height={params.height_m:.2f}m')
        # Try to complete full initialization
        self._try_complete_initialization()

    def _try_initialize(self):
        """Try to complete initialization by checking for required data."""
        if self.initialized:
            return

        # Try to get camera extrinsics
        if not self.camera.ready:
            self.camera.try_get_extrinsics()

        # Try to get LiDAR TF (always needed for method 0 or ground truth)
        if not self.lidar_tf_received:
            self._try_get_lidar_tf()

        # Check if ready to initialize
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
                f'LiDAR TF received: x={self.lidar_x:.2f}, y={self.lidar_y:.2f}, z={self.lidar_z:.2f}m')
            return True
        except TransformException:
            return False

    def _try_complete_initialization(self):
        """Complete initialization once all data is available."""
        if self.initialized:
            return

        if not self.camera.ready:
            self.get_logger().info('Waiting for camera params...', throttle_duration_sec=2.0)
            return
        if not self.lidar_tf_received:
            self.get_logger().info('Waiting for LiDAR TF...', throttle_duration_sec=2.0)
            return

        self.get_logger().info('All initialization data received, completing setup...')
        params = self.camera.params

        # Initialize Perception class for object detection
        if self.enable_object_detection:
            self.perception = Perception(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                method=self.perception_method,
                hood_mask_config=self.hood_mask_config
            )
            self.perception.set_intrinsics(params.fx, params.fy, params.cx, params.cy)
            self.perception.set_camera_position(
                x=params.x_m, y=params.y_m, z=params.height_m,
                pitch=params.pitch_deg
            )
            # Always set LiDAR position for method 0
            self.perception.set_lidar_position(
                x=self.lidar_x, y=self.lidar_y, z=self.lidar_z
            )
            self.get_logger().info(f'Perception initialized: {self.model_path}, method={self.perception_method}')

        # Cancel init timer
        self.init_timer.cancel()

        self.initialized = True
        self.get_logger().info('Perception Node fully initialized and ready!')

    # ==================== Sensor Callbacks ====================
    
    def rgb_callback(self, msg: Image):
        """Process RGB image for object detection."""
        self.get_logger().info('RGB image received', once=True)
        if not self.initialized or self.perception is None:
            return
        
        # Convert ROS Image to numpy via cv_bridge
        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
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
    
    # ==================== Debug Visualization ====================
    
    def _publish_debug_rgb(self, msg: Image, annotated: np.ndarray):
        """Publish debug RGB image with bboxes."""
        out_msg = self.cv_bridge.cv2_to_imgmsg(annotated, encoding='rgb8')
        out_msg.header = msg.header
        self.debug_rgb_pub.publish(out_msg)
    
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
