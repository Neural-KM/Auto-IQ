#!/usr/bin/env python3
"""
Semantic Segmentation Node
===========================

ROS2 node that publishes semantic segmentation maps from either:
- SegFormer model inference (learned segmentation)
- CARLA ground truth semantic camera (perfect labels)

Includes integrated **road extraction / semantic fusion**: detects lane markings
via adaptive thresholding within the road region and injects class 24
(road_line) into the semantic map.  The fused map is published on a separate
topic so that the mapping/costmap node receives lane-aware semantics.

Parameters:
- segmentation_source: 'segformer' or 'ground_truth' (default: 'ground_truth')
- model_name: SegFormer model variant (b0-b5) when using segformer
- device: 'cuda' or 'cpu' for SegFormer inference
- rgb_topic: Input RGB camera topic
- ground_truth_topic: Input ground truth semantic topic (for ground_truth mode)
- output_topic: Base semantic map topic (default: /perception/semantic_map)
- fused_topic: Fused semantic map with lane markings (default: /perception/semantic_map/fused)
- enable_fusion: Enable lane-marking fusion (default: True)
- publish_colored: Publish colored visualization (default: True)

Subscribes:
- /carla/ego_vehicle/sensor/camera/rgb/front/image (RGB camera — always)
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (ground_truth mode)

Publishes:
- /perception/semantic_map           (mono8) — raw semantic segmentation
- /perception/semantic_map/colored    (rgb8)  — colored visualization
- /perception/semantic_map/fused      (mono8) — fused with lane markings
- /perception/semantic_map/fused/colored (rgb8) — colored fused visualization
- /perception/road_mask               (mono8) — binary road mask  [debug]
- /perception/lane_markings           (mono8) — lane markings mask [debug]
"""

import numpy as np
import cv2
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

from perception.core.road_extraction import RoadExtractor
from perception.data_classes.semantic_labels import (
    SEMANTIC_BY_NAME,
    colorize_mask,
)


class SemanticSegmentationNode(Node):
    """
    ROS2 Node for unified semantic segmentation + lane-marking fusion.

    Supports switching between SegFormer inference and CARLA ground truth
    via the 'segmentation_source' parameter.  When *enable_fusion* is True
    (default), detected lane markings are injected into the semantic map as
    class 24 (road_line) and published on a separate fused topic.
    """

    def __init__(self):
        super().__init__('semantic_segmentation_node')

        # ==================== Parameters ====================
        self.declare_parameter('segmentation_source', 'ground_truth')
        self.declare_parameter('model_name', 'b2')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('publish_colored', True)
        self.declare_parameter('rgb_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/image')
        self.declare_parameter('ground_truth_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('output_topic', '/perception/semantic_map')

        # Road-extraction / fusion parameters
        self.declare_parameter('enable_fusion', True)
        self.declare_parameter('fused_topic', '/perception/semantic_map/fused')

        self.segmentation_source = self.get_parameter('segmentation_source').value
        model_name = self.get_parameter('model_name').value
        device = self.get_parameter('device').value
        self.publish_colored = self.get_parameter('publish_colored').value
        rgb_topic = self.get_parameter('rgb_topic').value
        ground_truth_topic = self.get_parameter('ground_truth_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.enable_fusion = self.get_parameter('enable_fusion').value
        fused_topic = self.get_parameter('fused_topic').value

        self.get_logger().info(f'Segmentation source: {self.segmentation_source}')
        self.get_logger().info(f'Lane-marking fusion: {"enabled" if self.enable_fusion else "disabled"}')

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

        # ==================== Initialize RoadExtractor (if fusion enabled) ====================
        self.road_extractor: Optional[RoadExtractor] = None
        if self.enable_fusion:
            self.road_extractor = RoadExtractor(road_grayscale_id=0)
            self.get_logger().info('RoadExtractor initialized for lane-marking fusion')

        # Semantic class IDs used by fusion
        self._road_id = SEMANTIC_BY_NAME.get('road', 1)
        self._road_line_id = SEMANTIC_BY_NAME.get('road_line', 24)

        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================== Cached RGB for fusion ====================
        self._latest_rgb: Optional[np.ndarray] = None

        # ==================== Subscribers ====================
        if self.segmentation_source == 'segformer':
            # In segformer mode, the RGB callback runs inference AND caches RGB
            self.image_sub = self.create_subscription(
                Image, rgb_topic, self.rgb_callback, sensor_qos)
            self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')
        elif self.segmentation_source == 'ground_truth':
            self.image_sub = self.create_subscription(
                Image, ground_truth_topic, self.ground_truth_callback, sensor_qos)
            self.get_logger().info(f'Subscribed to Ground Truth: {ground_truth_topic}')

            # Also subscribe to RGB for lane-marking fusion
            if self.enable_fusion:
                self.rgb_sub = self.create_subscription(
                    Image, rgb_topic, self._rgb_cache_callback, sensor_qos)
                self.get_logger().info(f'Subscribed to RGB (for fusion): {rgb_topic}')
        else:
            self.get_logger().error(
                f'Invalid segmentation_source: {self.segmentation_source}. '
                f'Options are: segformer, ground_truth')
            raise ValueError(f'Invalid segmentation_source: {self.segmentation_source}')

        # ==================== Publishers ====================
        # Raw semantic map
        self.semantic_pub = self.create_publisher(Image, output_topic, 10)
        if self.publish_colored:
            self.colored_pub = self.create_publisher(Image, f'{output_topic}/colored', 10)

        # Fused semantic map (with lane markings injected)
        if self.enable_fusion:
            self.fused_pub = self.create_publisher(Image, fused_topic, 10)
            if self.publish_colored:
                self.fused_colored_pub = self.create_publisher(
                    Image, f'{fused_topic}/colored', 10)
            # Debug publishers
            self.road_mask_pub = self.create_publisher(Image, '/perception/road_mask', 10)
            self.lane_mask_pub = self.create_publisher(Image, '/perception/lane_markings', 10)

        # ==================== Stats ====================
        self._frame_count = 0
        self._total_time = 0.0

        self.get_logger().info(
            f'Semantic Segmentation Node initialized! '
            f'Publishing to: {output_topic}'
            + (f' | fused to: {fused_topic}' if self.enable_fusion else '')
        )
    
    # ------------------------------------------------------------------ #
    #  RGB cache callback (ground_truth mode only)                        #
    # ------------------------------------------------------------------ #

    def _rgb_cache_callback(self, msg: Image):
        """Cache latest RGB image for lane-marking fusion (ground_truth mode)."""
        self._latest_rgb = self._ros_image_to_numpy_rgb(msg)

    # ------------------------------------------------------------------ #
    #  SegFormer callback                                                 #
    # ------------------------------------------------------------------ #

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

        # Cache for fusion
        self._latest_rgb = image_np

        # Run SegFormer inference
        segmentation = self.segformer.predict(image_np)

        # SegFormer outputs Cityscapes classes (0-18), convert to CARLA classes
        # Cityscapes road=0, CARLA road=1, so we add 1
        segmentation_carla = (segmentation + 1).astype(np.uint8)

        # Publish raw semantic map
        self._publish_semantic(segmentation_carla, msg.header)

        # Fuse lane markings and publish fused map
        self._fuse_and_publish(segmentation_carla, msg.header)

        # Stats
        self._update_stats(start_time, 'SegFormer')
    
    # ------------------------------------------------------------------ #
    #  Ground-truth callback                                              #
    # ------------------------------------------------------------------ #

    def ground_truth_callback(self, msg: Image):
        """
        Process CARLA ground truth semantic camera and publish semantic map.

        The bridge extracts the class-ID channel and publishes mono8.

        Args:
            msg: ROS2 Image message (mono8 from bridge)
        """
        start_time = time.perf_counter()

        # Bridge already extracts the semantic label channel as mono8
        segmentation = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width)

        # Publish raw semantic map
        self._publish_semantic(segmentation, msg.header)

        # Fuse lane markings and publish fused map
        self._fuse_and_publish(segmentation, msg.header)

        # Stats
        self._update_stats(start_time, 'Ground Truth')
    
    # ------------------------------------------------------------------ #
    #  Fusion logic                                                       #
    # ------------------------------------------------------------------ #

    def _fuse_and_publish(self, segmentation: np.ndarray, header):
        """
        Detect lane markings, inject them into the semantic map, and publish
        the fused result.  Gracefully does nothing when:
        - fusion is disabled
        - no RGB frame is available yet
        - size mismatch between RGB and semantic map
        - no road pixels are visible
        - no lane markings are detected
        """
        if not self.enable_fusion:
            return

        rgb = self._latest_rgb

        if rgb is None:
            # No RGB yet — publish unfused semantic map as fused
            self._publish_image(segmentation, header, 'mono8', self.fused_pub)
            if self.publish_colored:
                self._publish_image(
                    colorize_mask(segmentation), header, 'rgb8',
                    self.fused_colored_pub)
            return

        if rgb.shape[:2] != segmentation.shape[:2]:
            self.get_logger().warn(
                f'Size mismatch: RGB {rgb.shape[:2]} vs Semantic '
                f'{segmentation.shape[:2]}. Publishing unfused map.')
            self._publish_image(segmentation, header, 'mono8', self.fused_pub)
            if self.publish_colored:
                self._publish_image(
                    colorize_mask(segmentation), header, 'rgb8',
                    self.fused_colored_pub)
            return

        # --- Run fusion (safe even when no markings are found) ---
        try:
            fused_map = segmentation.copy()  # avoid mutating the original
            fused_map, lane_mask = self.road_extractor.fuse_lane_markings(
                fused_map, rgb,
                road_class_id=self._road_id,
                road_line_class_id=self._road_line_id,
            )
        except Exception as e:
            self.get_logger().error(f'Lane fusion failed: {e}')
            # Publish the unfused map so downstream never loses data
            self._publish_image(segmentation, header, 'mono8', self.fused_pub)
            if self.publish_colored:
                self._publish_image(
                    colorize_mask(segmentation), header, 'rgb8',
                    self.fused_colored_pub)
            return

        # --- Publish fused semantic map ---
        self._publish_image(fused_map, header, 'mono8', self.fused_pub)
        if self.publish_colored:
            self._publish_image(
                colorize_mask(fused_map), header, 'rgb8',
                self.fused_colored_pub)

        # --- Debug: road mask & lane mask ---
        road_mask = np.where(
            (fused_map == self._road_id) | (fused_map == self._road_line_id),
            255, 0,
        ).astype(np.uint8)
        self._publish_image(road_mask, header, 'mono8', self.road_mask_pub)
        self._publish_image(lane_mask, header, 'mono8', self.lane_mask_pub)

    # ------------------------------------------------------------------ #
    #  Publishing helpers                                                 #
    # ------------------------------------------------------------------ #

    def _publish_image(self, array: np.ndarray, header, encoding: str, publisher):
        """Publish a numpy array as a ROS2 Image on *publisher*."""
        msg = Image()
        msg.header = header
        msg.height = array.shape[0]
        msg.width = array.shape[1]
        msg.encoding = encoding
        if encoding == 'mono8':
            msg.step = msg.width
        elif encoding in ('rgb8', 'bgr8'):
            msg.step = msg.width * 3
        else:
            msg.step = msg.width * (array.shape[2] if array.ndim > 2 else 1)
        msg.data = array.tobytes()
        publisher.publish(msg)

    def _publish_semantic(self, segmentation: np.ndarray, header):
        """
        Publish raw semantic segmentation map and optional colored visualization.

        Args:
            segmentation: Semantic mask (H, W) with class IDs
            header: ROS2 Header
        """
        self._publish_image(segmentation, header, 'mono8', self.semantic_pub)

        if self.publish_colored:
            colored = colorize_mask(segmentation)
            self._publish_image(colored, header, 'rgb8', self.colored_pub)
    
    def _ros_image_to_numpy_rgb(self, msg: Image) -> np.ndarray:
        """
        Convert ROS2 Image message to RGB numpy array.

        The bridge converts CARLA's BGRA to rgb8 before publishing.

        Args:
            msg: ROS2 Image message (rgb8 from bridge)

        Returns:
            RGB numpy array (H, W, 3)
        """
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
    
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
