#!/usr/bin/env python3
"""
Semantic Colorizer Node
=======================

ROS2 node for colorizing semantic segmentation images and labeled point clouds.

Subscribes to:
- Semantic segmentation image (mono8/rgb8/bgra8)
- Labeled point cloud (XYZ + label)

Publishes:
- Colored semantic image (rgb8)
- Colored point cloud (XYZRGB)
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2, PointField

from perception.data_classes.semantic_labels import colorize_mask, SEMANTIC_COLOR_LUT


class SemanticColorizerNode(Node):
    """
    ROS2 Node for colorizing semantic data.
    
    Colorizes semantic segmentation images and labeled point clouds
    using a predefined color palette.
    """
    
    def __init__(self):
        super().__init__('semantic_colorizer_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('semantic_output_topic', '/visualization/semantic_image')
        self.declare_parameter('labeled_pointcloud_topic', '/carla/ego_vehicle/ground_truth/lidar/top_semantic/point_cloud')
        self.declare_parameter('colored_pointcloud_topic', '/visualization/semantic_pointcloud')
        
        semantic_topic = self.get_parameter('semantic_topic').value
        semantic_output_topic = self.get_parameter('semantic_output_topic').value
        labeled_pointcloud_topic = self.get_parameter('labeled_pointcloud_topic').value
        colored_pointcloud_topic = self.get_parameter('colored_pointcloud_topic').value
        
        # ==================== Publishers ====================
        self.semantic_image_pub = self.create_publisher(
            Image, semantic_output_topic, 10)
        self.colored_pointcloud_pub = self.create_publisher(
            PointCloud2, colored_pointcloud_topic, 10)
        
        # ==================== Subscribers ====================
        # QoS for sensor data (BEST_EFFORT to match publisher)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic,
            self.semantic_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to semantic image: {semantic_topic}')
        
        self.labeled_pointcloud_sub = self.create_subscription(
            PointCloud2, labeled_pointcloud_topic,
            self.labeled_pointcloud_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to labeled pointcloud: {labeled_pointcloud_topic}')
        
        self.get_logger().info('Semantic Colorizer Node initialized')
    
    # ==================== Callbacks ====================
    
    def semantic_callback(self, msg: Image):
        """Receive semantic segmentation image and republish colored version."""
        # Parse semantic class IDs from image
        if msg.encoding == 'bgra8':
            # CARLA semantic camera: class ID in R channel (index 2 for BGRA)
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 4)
            class_ids = img[:, :, 2]
        elif msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            class_ids = img[:, :, 0]
        elif msg.encoding == 'mono8':
            class_ids = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width)
        else:
            return
        
        # Colorize using semantic labels palette
        colored = colorize_mask(class_ids)
        
        # Publish colored image
        out_msg = Image()
        out_msg.header = msg.header
        out_msg.height = msg.height
        out_msg.width = msg.width
        out_msg.encoding = 'rgb8'
        out_msg.step = msg.width * 3
        out_msg.data = colored.tobytes()
        
        self.semantic_image_pub.publish(out_msg)
    
    def labeled_pointcloud_callback(self, msg: PointCloud2):
        """Vectorized conversion of labeled pointcloud to colored pointcloud (XYZRGB)."""
        # 1. Convert ROS msg to Numpy (view casting is instant)
        # Assumes fields are float32, float32, float32, uint32 aligned (standard CARLA format)
        dtype_list = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.uint32),
        ]
        
        cloud_arr = np.frombuffer(msg.data, dtype=dtype_list)
        
        # 2. Vectorized color lookup (no for-loops)
        labels = cloud_arr['label']
        safe_labels = np.clip(labels, 0, 255).astype(int)
        colors = SEMANTIC_COLOR_LUT[safe_labels]  # (N, 3) array
        
        # 3. Pack RGB into float32 for PointField
        # RViz expects RGB as single float32 where bytes are 0x00RRGGBB
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_uint32 = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint32.view(np.float32)
        
        # 4. Create output structured array
        out_dtype = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32),
        ]
        out_arr = np.empty(len(cloud_arr), dtype=out_dtype)
        out_arr['x'] = cloud_arr['x']
        out_arr['y'] = cloud_arr['y']
        out_arr['z'] = cloud_arr['z']
        out_arr['rgb'] = rgb_float
        
        # 5. Publish
        out_msg = PointCloud2()
        out_msg.header = msg.header
        out_msg.height = 1
        out_msg.width = len(cloud_arr)
        out_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out_msg.is_bigendian = False
        out_msg.point_step = 16
        out_msg.row_step = 16 * len(cloud_arr)
        out_msg.is_dense = True
        out_msg.data = out_arr.tobytes()
        
        self.colored_pointcloud_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticColorizerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
