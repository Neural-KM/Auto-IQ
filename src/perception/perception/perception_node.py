#!/usr/bin/env python3
"""
Perception Node
===============

ROS2 node for perception (Object Detection + Lane Detection).
Subscribes to camera topics and publishes detection results.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from perception.object_detector import ObjectDetector
from perception.lane_detector import LaneDetector
from perception.lane import LaneDetection


class PerceptionNode(Node):
    """
    ROS2 Perception Node.
    
    Subscribes to:
        - /carla/ego_vehicle/rgb_front/image_raw
        - /carla/ego_vehicle/semantic_front/image_raw
        - /carla/ego_vehicle/rgb_front/camera_info
        
    Publishes:
        - /perception/detections (MarkerArray)
        - /perception/lanes (MarkerArray)
        - /perception/debug/rgb (Image)
        - /perception/debug/bev (Image)
    """
    
    def __init__(self):
        super().__init__('perception_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('enable_object_detection', True)
        self.declare_parameter('enable_lane_detection', True)
        self.declare_parameter('enable_debug_viz', False)
        
        # Hood mask parameters
        self.declare_parameter('hood_mask.enabled', True)
        self.declare_parameter('hood_mask.top_y', 0.85)
        self.declare_parameter('hood_mask.top_left_x', 0.25)
        self.declare_parameter('hood_mask.top_right_x', 0.75)
        self.declare_parameter('hood_mask.bottom_y', 1.0)
        self.declare_parameter('hood_mask.bottom_left_x', 0.08)
        self.declare_parameter('hood_mask.bottom_right_x', 0.92)
        
        # Camera parameters for lane detection
        self.declare_parameter('camera.pitch', -15.0)
        self.declare_parameter('camera.height', 1.8)
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.enable_object_detection = self.get_parameter('enable_object_detection').value
        self.enable_lane_detection = self.get_parameter('enable_lane_detection').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value
        self.camera_pitch = self.get_parameter('camera.pitch').value
        self.camera_height = self.get_parameter('camera.height').value
        
        # Build hood mask config
        hood_mask_config = {
            'enabled': self.get_parameter('hood_mask.enabled').value,
            'top_y': self.get_parameter('hood_mask.top_y').value,
            'top_left_x': self.get_parameter('hood_mask.top_left_x').value,
            'top_right_x': self.get_parameter('hood_mask.top_right_x').value,
            'bottom_y': self.get_parameter('hood_mask.bottom_y').value,
            'bottom_left_x': self.get_parameter('hood_mask.bottom_left_x').value,
            'bottom_right_x': self.get_parameter('hood_mask.bottom_right_x').value,
        }
        
        # Initialize Object Detector
        self.object_detector = None
        if self.enable_object_detection:
            self.object_detector = ObjectDetector(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                hood_mask_config=hood_mask_config
            )
            self.get_logger().info(f'Object detector initialized: {self.model_path}')
        
        # Lane detector (initialized when camera_info received)
        self.lane_detector = None
        self.camera_info_received = False
        
        # QoS profile
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/carla/ego_vehicle/sensor/camera/rgb/front/image',
            self.rgb_callback, sensor_qos)
        
        self.semantic_sub = self.create_subscription(
            Image, '/carla/ego_vehicle/ground_truth/camera/semantic/front/image',
            self.semantic_callback, sensor_qos)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info',
            self.camera_info_callback, sensor_qos)
        
        # Publishers
        self.detections_pub = self.create_publisher(MarkerArray, '/perception/detections', 10)
        self.lanes_pub = self.create_publisher(MarkerArray, '/perception/lanes', 10)
        
        if self.enable_debug_viz:
            self.debug_rgb_pub = self.create_publisher(Image, '/perception/debug/rgb', 10)
            self.debug_bev_pub = self.create_publisher(Image, '/perception/debug/bev', 10)
        
        self.get_logger().info('Perception Node initialized')
    
    def camera_info_callback(self, msg: CameraInfo):
        """Initialize lane detector with camera info."""
        if self.camera_info_received or not self.enable_lane_detection:
            return
        
        width = msg.width
        height = msg.height
        fx = msg.k[0]
        
        if fx > 0:
            fov_rad = 2 * np.arctan(width / (2 * fx))
            fov_deg = np.degrees(fov_rad)
        else:
            fov_deg = 90.0
        
        self.lane_detector = LaneDetector(
            image_width=width,
            image_height=height,
            camera_fov=fov_deg,
            camera_pitch=self.camera_pitch,
            camera_height=self.camera_height
        )
        
        self.camera_info_received = True
        self.get_logger().info(f'Lane detector initialized: {width}x{height}, FOV={fov_deg:.1f}°')
    
    def rgb_callback(self, msg: Image):
        """Process RGB image for object detection."""
        if not self.enable_object_detection or self.object_detector is None:
            return
        
        # Convert ROS Image to numpy
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img = img[:, :, ::-1]
        else:
            self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
            return
        
        # Run detection
        detections = self.object_detector.detect(img)
        
        # Publish markers
        marker_array = MarkerArray()
        
        clear_marker = Marker()
        clear_marker.header.frame_id = 'ego_vehicle'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = 'detections'
        clear_marker.id = -1
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            marker = Marker()
            marker.header.frame_id = 'ego_vehicle/front'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detections'
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            marker.pose.position.x = 5.0
            marker.pose.position.y = -(cx - msg.width/2) * 0.01
            marker.pose.position.z = -(cy - msg.height/2) * 0.01
            marker.scale.z = 0.5
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            marker.text = f"{det['class']}: {det['score']:.2f}"
            marker.lifetime.nanosec = 200000000
            marker_array.markers.append(marker)
        
        self.detections_pub.publish(marker_array)
        
        if self.enable_debug_viz:
            self._publish_debug_rgb(msg, img, detections)
    
    def semantic_callback(self, msg: Image):
        """Process semantic image for lane detection."""
        if not self.enable_lane_detection or self.lane_detector is None:
            return
        
        if msg.encoding == 'mono8':
            semantic_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        else:
            self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
            return
        
        result = self.lane_detector.detect(semantic_img)
        self._publish_lane_markers(result)
        
        if self.enable_debug_viz:
            self._publish_debug_bev()
    
    def _publish_lane_markers(self, result: LaneDetection):
        """Publish lane markers."""
        marker_array = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        
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
            left_marker = Marker()
            left_marker.header.frame_id = 'ego_vehicle'
            left_marker.header.stamp = timestamp
            left_marker.ns = 'lanes'
            left_marker.id = 1
            left_marker.type = Marker.LINE_STRIP
            left_marker.action = Marker.ADD
            left_marker.scale.x = 0.2
            left_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            
            for y_bev in np.linspace(0, bev_params['height'] - 1, 20):
                x_bev = np.polyval(result.left_fit, y_bev)
                x_world = bev_params['y_range'][1] - (y_bev / bev_params['height']) * (bev_params['y_range'][1] - bev_params['y_range'][0])
                y_world = bev_params['x_range'][0] + (x_bev / bev_params['width']) * (bev_params['x_range'][1] - bev_params['x_range'][0])
                pt = Point(x=float(x_world), y=float(-y_world), z=0.0)
                left_marker.points.append(pt)
            marker_array.markers.append(left_marker)
        
        if result.right_fit is not None:
            right_marker = Marker()
            right_marker.header.frame_id = 'ego_vehicle'
            right_marker.header.stamp = timestamp
            right_marker.ns = 'lanes'
            right_marker.id = 2
            right_marker.type = Marker.LINE_STRIP
            right_marker.action = Marker.ADD
            right_marker.scale.x = 0.2
            right_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            
            for y_bev in np.linspace(0, bev_params['height'] - 1, 20):
                x_bev = np.polyval(result.right_fit, y_bev)
                x_world = bev_params['y_range'][1] - (y_bev / bev_params['height']) * (bev_params['y_range'][1] - bev_params['y_range'][0])
                y_world = bev_params['x_range'][0] + (x_bev / bev_params['width']) * (bev_params['x_range'][1] - bev_params['x_range'][0])
                pt = Point(x=float(x_world), y=float(-y_world), z=0.0)
                right_marker.points.append(pt)
            marker_array.markers.append(right_marker)
        
        self.lanes_pub.publish(marker_array)
    
    def _publish_debug_rgb(self, msg: Image, img: np.ndarray, detections: list):
        """Publish debug RGB image with bboxes."""
        import cv2
        debug_img = img.copy()
        for det in detections:
            bbox = [int(x) for x in det['bbox']]
            color = (255, 0, 0) if det['class'] == 'vehicle' else (0, 255, 0)
            cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(debug_img, f"{det['class']}: {det['score']:.2f}", (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = debug_img.shape[0]
        out_msg.width = debug_img.shape[1]
        out_msg.encoding = 'rgb8'
        out_msg.step = debug_img.shape[1] * 3
        out_msg.data = debug_img.tobytes()
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
