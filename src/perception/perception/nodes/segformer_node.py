#!/usr/bin/env python3
"""
SegFormer Semantic Segmentation Node
=====================================

ROS2 node for semantic segmentation using HuggingFace's SegFormer model
trained on the Cityscapes dataset.

Subscribes:
- /carla/ego_vehicle/sensor/camera/rgb/front/image (sensor_msgs/Image) - RGB camera

Publishes:
- /perception/segmentation/image (sensor_msgs/Image) - Segmentation mask (mono8)
- /perception/segmentation/colored (sensor_msgs/Image) - Colored visualization (rgb8)
"""

import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

from perception.core.segformer import SegFormer


class SegFormerNode(Node):
    """
    ROS2 Node for SegFormer semantic segmentation.
    
    Uses nvidia/segformer-b0-finetuned-cityscapes-1024-1024 by default,
    which provides a good balance between speed and accuracy.
    """
    
    def __init__(self):
        super().__init__('segformer_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('model_name', 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
        self.declare_parameter('device', 'cuda')  # 'cuda' or 'cpu'
        self.declare_parameter('publish_colored', True)
        self.declare_parameter('resize_output', True)  # Resize output to match input size
        self.declare_parameter('input_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/image')
        
        model_name = self.get_parameter('model_name').value
        device_param = self.get_parameter('device').value
        self.publish_colored = self.get_parameter('publish_colored').value
        self.resize_output = self.get_parameter('resize_output').value
        input_topic = self.get_parameter('input_topic').value
        
        # ==================== Initialize Core SegFormer ====================
        self.get_logger().info(f'Loading SegFormer model: {model_name}')
        
        try:
            self.segformer = SegFormer(model_name=model_name, device=device_param)
            self.get_logger().info(f'SegFormer model loaded on {self.segformer.device}')
        except Exception as e:
            self.get_logger().error(f'Failed to load SegFormer model: {e}')
            raise RuntimeError(f'Model loading failed: {e}')
        
        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # ==================== Subscribers ====================
        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            sensor_qos
        )
        self.get_logger().info(f'Subscribed to: {input_topic}')
        
        # ==================== Publishers ====================
        self.seg_pub = self.create_publisher(
            Image, '/perception/segmentation/image', 10)
        
        if self.publish_colored:
            self.colored_pub = self.create_publisher(
                Image, '/perception/segmentation/colored', 10)
        
        # ==================== Stats ====================
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        self.get_logger().info('SegFormer Node initialized!')
    
    def image_callback(self, msg: Image):
        """
        Process incoming RGB image through SegFormer.
        
        Args:
            msg: ROS2 Image message (expected: bgra8 from CARLA)
        """
        start_time = time.perf_counter()
        
        # Log encoding on first few messages to verify
        if self._inference_count < 3:
            self.get_logger().info(f'Received image: {msg.width}x{msg.height}, encoding={msg.encoding}')
        
        # Convert CARLA bgra8 image to RGB numpy array
        # CARLA cameras publish bgra8 encoding
        if msg.encoding == 'bgra8':
            image_bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            image_np = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
        elif msg.encoding == 'bgr8':
            image_bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        elif msg.encoding == 'rgb8':
            image_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
        else:
            self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
            return
        
        original_height, original_width = image_np.shape[:2]
        
        # Run inference using core SegFormer
        segmentation_mask = self.segformer.predict(image_np)
        
        # Resize output to match input if requested
        if self.resize_output and segmentation_mask.shape != (original_height, original_width):
            segmentation_mask = cv2.resize(
                segmentation_mask, 
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Publish segmentation mask
        seg_msg = self._numpy_to_ros_image(
            segmentation_mask.astype(np.uint8), 
            msg.header, 
            encoding='mono8'
        )
        self.seg_pub.publish(seg_msg)
        
        # Publish colored visualization
        if self.publish_colored:
            colored = self.segformer.colorize(segmentation_mask)
            colored_msg = self._numpy_to_ros_image(
                colored, 
                msg.header, 
                encoding='rgb8'
            )
            self.colored_pub.publish(colored_msg)
        
        # Update stats
        inference_time = (time.perf_counter() - start_time) * 1000
        self._inference_count += 1
        self._total_inference_time += inference_time
        
        if self._inference_count % 30 == 0:
            avg_time = self._total_inference_time / self._inference_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'SegFormer: {inference_time:.1f}ms (avg: {avg_time:.1f}ms, ~{fps:.1f} FPS)'
            )
    
    def _numpy_to_ros_image(self, array: np.ndarray, header, encoding: str) -> Image:
        """
        Convert numpy array to ROS2 Image message.
        
        Args:
            array: Numpy array (H, W) or (H, W, 3)
            header: ROS2 Header to use
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
    node = SegFormerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Allow clean shutdown on Ctrl+C; resources are released in the finally block.
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
