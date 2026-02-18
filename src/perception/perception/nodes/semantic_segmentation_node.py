#!/usr/bin/env python3
"""
Semantic Segmentation Node
===========================

ROS2 node that publishes semantic segmentation maps from either:
- SegFormer model inference (learned segmentation)
- CARLA ground truth semantic camera (perfect labels)

The output is published to a unified topic, allowing downstream nodes
(like road_extraction_node) to be agnostic to the source.

Parameters:
- segmentation_source: 'segformer' or 'ground_truth' (default: 'ground_truth')
- model_name: SegFormer model variant (b0-b5) when using segformer
- device: 'cuda' or 'cpu' for SegFormer inference
- rgb_topic: Input RGB camera topic (for SegFormer mode)
- ground_truth_topic: Input ground truth semantic topic (for ground_truth mode)

Subscribes:
- /carla/ego_vehicle/sensor/camera/rgb/front/image (for SegFormer mode)
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (for ground_truth mode)

Publishes:
- /perception/semantic_map (sensor_msgs/Image) - Semantic segmentation mask (mono8, class IDs)
- /perception/semantic_map/colored (sensor_msgs/Image) - Colored visualization (rgb8)
"""

import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

from perception.data_classes.semantic_labels import colorize_mask


class SemanticSegmentationNode(Node):
    """
    ROS2 Node for unified semantic segmentation publishing.
    
    Supports switching between SegFormer inference and CARLA ground truth
    via the 'segmentation_source' parameter.
    """
    
    def __init__(self):
        super().__init__('semantic_segmentation_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('segmentation_source', 'ground_truth')  # 'segformer' or 'ground_truth'
        self.declare_parameter('model_name', 'b2')  # SegFormer model variant
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('publish_colored', True)
        self.declare_parameter('rgb_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/image')
        self.declare_parameter('ground_truth_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('output_topic', '/perception/semantic_map')
        
        self.segmentation_source = self.get_parameter('segmentation_source').value
        model_name = self.get_parameter('model_name').value
        device = self.get_parameter('device').value
        self.publish_colored = self.get_parameter('publish_colored').value
        rgb_topic = self.get_parameter('rgb_topic').value
        ground_truth_topic = self.get_parameter('ground_truth_topic').value
        output_topic = self.get_parameter('output_topic').value
        
        self.get_logger().info(f'Segmentation source: {self.segmentation_source}')
        
        # ==================== Initialize SegFormer (if needed) ====================
        self.segformer = None
        if self.segmentation_source == 'segformer':
            self.get_logger().info(f'Loading SegFormer model: {model_name}')
            try:
                from perception.core.segformer import SegFormer
                self.segformer = SegFormer(model_name=model_name, device=device)
                self.get_logger().info(f'SegFormer loaded on {self.segformer.device}')
            except Exception as e:
                self.get_logger().error(f'Failed to load SegFormer: {e}')
                raise RuntimeError(f'SegFormer loading failed: {e}')
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        if self.segmentation_source == 'segformer':
            self.image_sub = self.create_subscription(
                Image,
                rgb_topic,
                self.rgb_callback,
                sensor_qos
            )
            self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')
        elif self.segmentation_source == 'ground_truth':
            self.image_sub = self.create_subscription(
                Image,
                ground_truth_topic,
                self.ground_truth_callback,
                sensor_qos
            )
            self.get_logger().info(f'Subscribed to Ground Truth: {ground_truth_topic}')
        else:
            self.get_logger().error(f'Invalid segmentation_source: {self.segmentation_source}. Options are: segformer, ground_truth')
            raise ValueError(f'Invalid segmentation_source: {self.segmentation_source}')
        
        # ==================== Publishers ====================
        self.semantic_pub = self.create_publisher(Image, output_topic, 10)
        
        if self.publish_colored:
            self.colored_pub = self.create_publisher(Image, f'{output_topic}/colored', 10)
        
        # ==================== Stats ====================
        self._frame_count = 0
        self._total_time = 0.0
        
        self.get_logger().info(f'Semantic Segmentation Node initialized! Publishing to: {output_topic}')
    
    def rgb_callback(self, msg: Image):
        """
        Process RGB image through SegFormer and publish semantic map.
        
        Args:
            msg: ROS2 Image message (expected: bgra8 from CARLA)
        """
        start_time = time.perf_counter()
        
        # Convert to RGB numpy array
        image_np = self._ros_image_to_numpy_rgb(msg)
        if image_np is None:
            return
        
        # Run SegFormer inference
        segmentation = self.segformer.predict(image_np)
        
        # SegFormer outputs Cityscapes classes (0-18), convert to CARLA classes
        # Cityscapes road=0, CARLA road=1, so we add 1
        segmentation_carla = (segmentation + 1).astype(np.uint8)
        
        # Publish semantic map
        self._publish_semantic(segmentation_carla, msg.header)
        
        # Stats
        self._update_stats(start_time, 'SegFormer')
    
    def ground_truth_callback(self, msg: Image):
        """
        Process CARLA ground truth semantic camera and publish semantic map.
        
        CARLA semantic camera outputs BGRA8 where the R channel (index 2) 
        contains the class ID.
        
        Args:
            msg: ROS2 Image message (bgra8 from CARLA semantic camera)
        """
        start_time = time.perf_counter()
        
        # Extract semantic mask from CARLA semantic camera
        # Class ID is in the R channel (index 2 for BGRA)
        if msg.encoding == 'bgra8':
            image_bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            # R channel contains class ID
            segmentation = image_bgra[:, :, 2].copy()
        elif msg.encoding == 'mono8':
            # Already a grayscale class ID image
            segmentation = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width)
        else:
            self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
            return
        
        # Publish semantic map
        self._publish_semantic(segmentation, msg.header)
        
        # Stats
        self._update_stats(start_time, 'Ground Truth')
    
    def _publish_semantic(self, segmentation: np.ndarray, header):
        """
        Publish semantic segmentation map and optional colored visualization.
        
        Args:
            segmentation: Semantic mask (H, W) with class IDs
            header: ROS2 Header
        """
        # Publish mono8 semantic map
        seg_msg = Image()
        seg_msg.header = header
        seg_msg.height = segmentation.shape[0]
        seg_msg.width = segmentation.shape[1]
        seg_msg.encoding = 'mono8'
        seg_msg.step = segmentation.shape[1]
        seg_msg.data = segmentation.tobytes()
        self.semantic_pub.publish(seg_msg)
        
        # Publish colored visualization
        if self.publish_colored:
            colored = colorize_mask(segmentation)
            colored_msg = Image()
            colored_msg.header = header
            colored_msg.height = colored.shape[0]
            colored_msg.width = colored.shape[1]
            colored_msg.encoding = 'rgb8'
            colored_msg.step = colored.shape[1] * 3
            colored_msg.data = colored.tobytes()
            self.colored_pub.publish(colored_msg)
    
    def _ros_image_to_numpy_rgb(self, msg: Image) -> np.ndarray:
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
                msg.height, msg.width, 3)
        else:
            self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
            return None
    
    def _update_stats(self, start_time: float, source: str):
        """Update and log processing statistics."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self._frame_count += 1
        self._total_time += elapsed
        
        if self._frame_count % 30 == 0:
            avg_time = self._total_time / self._frame_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'[{source}] {elapsed:.1f}ms (avg: {avg_time:.1f}ms, ~{fps:.1f} FPS)'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
