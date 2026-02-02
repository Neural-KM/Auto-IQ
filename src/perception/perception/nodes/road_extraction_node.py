#!/usr/bin/env python3
"""
Road Extraction Node
=====================

ROS2 node that subscribes to colored semantic segmentation visualizations and extracts
lane markings using the RoadExtractor core module.

This node is agnostic to the segmentation source - it processes whatever
semantic map is published to the input topic (from SegFormer or ground truth).
The colored semantic image is converted to grayscale internally to extract the road mask.

Parameters:
- semantic_topic: Input colored semantic topic (default: /perception/semantic_map/colored)
- rgb_topic: Input RGB camera topic (for lane detection visualization)
- road_grayscale_id: Grayscale intensity value for road pixels (default: 90)
- enable_ipm: Use Inverse Perspective Mapping for lane detection (default: True)
- fov: Camera horizontal field of view in degrees (default: 110.0)
- pitch: Camera pitch angle in degrees (default: 0.0)
- camera_height: Camera height above ground in meters (default: 1.2)

Subscribes:
- /perception/semantic_map/colored (sensor_msgs/Image) - Colored semantic visualization (rgb8)
- /carla/ego_vehicle/sensor/camera/rgb/front/image (sensor_msgs/Image) - RGB camera

Publishes:
- /perception/road_mask (sensor_msgs/Image) - Binary road mask (mono8)
- /perception/lane_markings (sensor_msgs/Image) - Lane markings mask (mono8)
- /perception/lane_markings/ipm (sensor_msgs/Image) - IPM view with lane markings (rgb8)
- /perception/lane_markings/overlay (sensor_msgs/Image) - Lane overlay on original image (rgb8)
"""

import numpy as np
import cv2
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

from perception.core.road_extraction import RoadExtractor, CONFIG
from perception.data_classes.semantic_labels import SEMANTIC_BY_NAME


class RoadExtractionNode(Node):
    """
    ROS2 Node for road and lane marking extraction from semantic maps.
    
    Subscribes to semantic segmentation maps and RGB images, extracts road regions
    and lane markings, and publishes the results.
    """
    
    def __init__(self):
        super().__init__('road_extraction_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('semantic_topic', '/perception/semantic_map/colored')
        self.declare_parameter('rgb_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/image')
        self.declare_parameter('road_grayscale_id', 90)  # Grayscale intensity for road (from colored->gray)
        self.declare_parameter('enable_ipm', True)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('publish_ipm', True)
        
        # Camera parameters for IPM
        self.declare_parameter('fov', 110.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('camera_height', 1.2)
        
        # IPM range parameters
        self.declare_parameter('ipm_x_min', 3.0)
        self.declare_parameter('ipm_x_max', 30.0)
        self.declare_parameter('ipm_y_min', -10.0)
        self.declare_parameter('ipm_y_max', 10.0)
        self.declare_parameter('pixels_per_meter', 30.0)
        
        semantic_topic = self.get_parameter('semantic_topic').value
        rgb_topic = self.get_parameter('rgb_topic').value
        road_grayscale_id = self.get_parameter('road_grayscale_id').value
        self.enable_ipm = self.get_parameter('enable_ipm').value
        self.publish_overlay = self.get_parameter('publish_overlay').value
        self.publish_ipm = self.get_parameter('publish_ipm').value
        
        # Camera params
        self.fov = self.get_parameter('fov').value
        self.pitch = self.get_parameter('pitch').value
        self.camera_height = self.get_parameter('camera_height').value
        
        # IPM params
        self.ipm_x_range = (
            self.get_parameter('ipm_x_min').value,
            self.get_parameter('ipm_x_max').value
        )
        self.ipm_y_range = (
            self.get_parameter('ipm_y_min').value,
            self.get_parameter('ipm_y_max').value
        )
        self.pixels_per_meter = self.get_parameter('pixels_per_meter').value
        
        # ==================== Initialize RoadExtractor ====================
        # RoadExtractor expects grayscale intensity from colored semantic image
        self.road_extractor = RoadExtractor(road_grayscale_id=road_grayscale_id)
        self.get_logger().info(f'RoadExtractor initialized with road_grayscale_id={road_grayscale_id}')
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Message Sync Storage ====================
        self._latest_rgb: Optional[Image] = None
        self._latest_rgb_np: Optional[np.ndarray] = None
        self._latest_semantic: Optional[Image] = None
        
        # ==================== Subscribers ====================
        self.semantic_sub = self.create_subscription(
            Image,
            semantic_topic,
            self.semantic_callback,
            sensor_qos
        )
        self.get_logger().info(f'Subscribed to semantic: {semantic_topic}')
        
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            sensor_qos
        )
        self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')
        
        # ==================== Publishers ====================
        self.road_mask_pub = self.create_publisher(Image, '/perception/road_mask', 10)
        self.lane_mask_pub = self.create_publisher(Image, '/perception/lane_markings', 10)
        
        if self.publish_ipm:
            self.ipm_pub = self.create_publisher(Image, '/perception/lane_markings/ipm', 10)
        
        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(Image, '/perception/lane_markings/overlay', 10)
        
        # ==================== Stats ====================
        self._frame_count = 0
        self._total_time = 0.0
        
        self.get_logger().info('Road Extraction Node initialized!')
    
    def rgb_callback(self, msg: Image):
        """
        Cache the latest RGB image for lane detection.
        
        Args:
            msg: ROS2 Image message
        """
        self._latest_rgb = msg
        self._latest_rgb_np = self._ros_image_to_numpy_rgb(msg)
    
    def semantic_callback(self, msg: Image):
        """
        Process semantic segmentation map to extract road mask and lane markings.
        
        Args:
            msg: ROS2 Image message (rgb8 colored semantic visualization)
        """
        start_time = time.perf_counter()
        
        # Parse colored semantic map and convert to grayscale
        if msg.encoding == 'rgb8':
            semantic_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            # Convert to grayscale for RoadExtractor
            semantic_map = cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2GRAY)
        elif msg.encoding == 'bgr8':
            semantic_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            semantic_map = cv2.cvtColor(semantic_bgr, cv2.COLOR_BGR2GRAY)
        elif msg.encoding == 'mono8':
            # Already grayscale
            semantic_map = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width)
        else:
            self.get_logger().error(f'Unsupported semantic encoding: {msg.encoding}')
            return
        
        # Check if we have a matching RGB image
        if self._latest_rgb_np is None:
            self.get_logger().warn('No RGB image available yet, skipping lane detection')
            return
        
        # Ensure size match
        if self._latest_rgb_np.shape[:2] != semantic_map.shape:
            self.get_logger().warn(
                f'Size mismatch: RGB {self._latest_rgb_np.shape[:2]} vs Semantic {semantic_map.shape}'
            )
            return
        
        # ==================== Extract Road Mask ====================
        road_mask = self.road_extractor.extract_road_mask(semantic_map)
        
        # Publish road mask
        road_msg = self._numpy_to_ros_image(road_mask, msg.header, 'mono8')
        self.road_mask_pub.publish(road_msg)
        
        # Check if road is present
        road_pixels = np.count_nonzero(road_mask)
        if road_pixels == 0:
            self.get_logger().debug('No road pixels detected in current frame')
            return
        
        # ==================== Extract Lane Markings ====================
        if self.enable_ipm:
            # Use IPM-based lane detection (more robust for BEV)
            try:
                road_only, ipm_image, ipm_lane_mask, ipm_road_mask, ipm_size = \
                    self.road_extractor.extract_lanes_ipm(
                        self._latest_rgb_np,
                        road_mask,
                        fov=self.fov,
                        pitch=self.pitch,
                        camera_height=self.camera_height,
                        ipm_x_range=self.ipm_x_range,
                        ipm_y_range=self.ipm_y_range,
                        pixels_per_meter=self.pixels_per_meter
                    )
                
                # Publish lane mask (IPM space)
                lane_msg = self._numpy_to_ros_image(ipm_lane_mask, msg.header, 'mono8')
                self.lane_mask_pub.publish(lane_msg)
                
                # Publish IPM visualization
                if self.publish_ipm:
                    # Create IPM visualization with lane overlay
                    ipm_viz = ipm_image.copy()
                    # Highlight lanes in green
                    ipm_viz[ipm_lane_mask > 0] = [0, 255, 0]
                    ipm_msg = self._numpy_to_ros_image(ipm_viz, msg.header, 'rgb8')
                    self.ipm_pub.publish(ipm_msg)
                
            except Exception as e:
                self.get_logger().error(f'IPM lane extraction failed: {e}')
                # Fallback to perspective lane detection
                lane_mask = self.road_extractor.extract_lanes(self._latest_rgb_np, road_mask)
                lane_msg = self._numpy_to_ros_image(lane_mask, msg.header, 'mono8')
                self.lane_mask_pub.publish(lane_msg)
        else:
            # Simple perspective lane detection
            lane_mask = self.road_extractor.extract_lanes(self._latest_rgb_np, road_mask)
            lane_msg = self._numpy_to_ros_image(lane_mask, msg.header, 'mono8')
            self.lane_mask_pub.publish(lane_msg)
        
        # ==================== Publish Overlay Visualization ====================
        if self.publish_overlay:
            overlay = self._latest_rgb_np.copy()
            # Road in purple (semi-transparent)
            road_color = np.array([128, 64, 128], dtype=np.uint8)
            overlay[road_mask > 0] = (
                overlay[road_mask > 0] * 0.6 + road_color * 0.4
            ).astype(np.uint8)
            
            # Lane markings in green
            if self.enable_ipm:
                # For IPM, we need to project back or just show road overlay
                pass
            else:
                overlay[lane_mask > 0] = [0, 255, 0]
            
            overlay_msg = self._numpy_to_ros_image(overlay, msg.header, 'rgb8')
            self.overlay_pub.publish(overlay_msg)
        
        # ==================== Stats ====================
        elapsed = (time.perf_counter() - start_time) * 1000
        self._frame_count += 1
        self._total_time += elapsed
        
        if self._frame_count % 30 == 0:
            avg_time = self._total_time / self._frame_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'Lane extraction: {elapsed:.1f}ms (avg: {avg_time:.1f}ms, ~{fps:.1f} FPS), '
                f'road={road_pixels} px'
            )
    
    def _ros_image_to_numpy_rgb(self, msg: Image) -> Optional[np.ndarray]:
        """
        Convert ROS2 Image message to RGB numpy array.
        
        Args:
            msg: ROS2 Image message
            
        Returns:
            RGB numpy array (H, W, 3) or None if conversion failed
        """
        if msg.encoding == 'bgra8':
            image_bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            return cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
        elif msg.encoding == 'bgr8':
            image_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif msg.encoding == 'rgb8':
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3).copy()
        else:
            self.get_logger().error(f'Unsupported RGB encoding: {msg.encoding}')
            return None
    
    def _numpy_to_ros_image(self, array: np.ndarray, header, encoding: str) -> Image:
        """
        Convert numpy array to ROS2 Image message.
        
        Args:
            array: Numpy array (H, W) or (H, W, 3)
            header: ROS2 Header
            encoding: Image encoding ('mono8', 'rgb8', etc.)
            
        Returns:
            ROS2 Image message
        """
        msg = Image()
        msg.header = header
        msg.height = array.shape[0]
        msg.width = array.shape[1]
        msg.encoding = encoding
        
        if encoding == 'mono8':
            msg.step = msg.width
        elif encoding in ['rgb8', 'bgr8']:
            msg.step = msg.width * 3
        else:
            msg.step = msg.width * array.shape[2] if len(array.shape) > 2 else msg.width
        
        msg.data = array.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = RoadExtractionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
