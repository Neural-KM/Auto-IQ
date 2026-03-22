#!/usr/bin/env python3
"""
Semantic Segmentation Node
===========================

ROS2 node that publishes semantic segmentation maps from either:
- SegFormer model inference (learned segmentation)
- CARLA ground truth semantic camera (perfect labels)

Flow:
- **ground_truth mode**: CARLA ground truth already contains lane markings
  (class 24 road_line).  The map is published as-is to both the raw and
  road_lane_extraction topics.
- **segformer mode**: SegFormer output does NOT contain lane markings.
  When *enable_road_lane_extraction* is True, the LaneExtractor detects lane
  markings via adaptive thresholding on the RGB image and injects class 24
  (road_line) into the semantic map.  The extracted map is published on a
  separate topic so that downstream nodes receive lane-aware semantics.

Note: Lane polynomial fitting is handled by the mapping package's lane_fitting_node.

Parameters:
- segmentation_source: 'segformer' or 'ground_truth' (default: 'ground_truth')
- model_name: SegFormer model variant (b0-b5), set in perception.launch.py
- device: 'cuda' or 'cpu' for SegFormer inference
- rgb_topic: Input RGB camera topic
- ground_truth_topic: Input ground truth semantic topic (for ground_truth mode)
- output_topic: Base semantic map topic (default: /perception/semantic_map)
- road_lane_extraction_topic: Road lane extraction map with lane markings
- enable_road_lane_extraction: Enable lane-marking extraction via LaneExtractor

Subscribes:
- /carla/ego_vehicle/sensor/camera/rgb/front/image (RGB camera - segformer mode)
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (ground_truth mode)

Publishes:
- /perception/semantic_map           (mono8) - raw semantic segmentation
- /perception/semantic_map/colored    (rgb8)  - colored visualization
- /perception/semantic_map/road_lane_extraction      (mono8) - with lane markings
- /perception/semantic_map/road_lane_extraction/colored (rgb8) - colored road lane extraction
- /perception/road_mask               (mono8) - binary road mask  [debug]
- /perception/lane_markings           (mono8) - lane markings mask [debug]
"""

import numpy as np
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image

from perception.core.lane_extraction import LaneExtractor
from perception.data_classes.semantic_labels import (
    SEMANTIC_BY_NAME,
    colorize_mask,
)

from cv_bridge import CvBridge


class SemanticSegmentationNode(Node):
    """ROS2 Node for semantic segmentation with optional road lane extraction.

    See module docstring for the ground_truth vs segformer flow.
    """

    def __init__(self):
        super().__init__('semantic_segmentation_node')

        # ==================== Parameters ====================
        self.declare_parameter('segmentation_source', 'ground_truth')
        self.declare_parameter('model_name', 'b0')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('publish_colored', True)
        self.declare_parameter('rgb_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/image')
        self.declare_parameter('ground_truth_topic', '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')
        self.declare_parameter('output_topic', '/perception/semantic_map')

        # Road-extraction / road lane extraction parameters
        self.declare_parameter('enable_road_lane_extraction', True)
        self.declare_parameter('road_lane_extraction_topic', '/perception/semantic_map/road_lane_extraction')

        self.segmentation_source = self.get_parameter('segmentation_source').value
        model_name = self.get_parameter('model_name').value
        device = self.get_parameter('device').value
        self.publish_colored = self.get_parameter('publish_colored').value
        rgb_topic = self.get_parameter('rgb_topic').value
        ground_truth_topic = self.get_parameter('ground_truth_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.enable_road_lane_extraction = self.get_parameter('enable_road_lane_extraction').value
        road_lane_extraction_topic = self.get_parameter('road_lane_extraction_topic').value

        self.get_logger().info(f'Segmentation source: {self.segmentation_source}')
        self.get_logger().info(f'Road lane extraction: {"enabled" if self.enable_road_lane_extraction else "disabled"}')

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

        # ==================== Initialize LaneExtractor (segformer + road lane extraction only) ==
        # Ground truth already contains lane markings - no extraction needed.
        self.lane_extractor: Optional[LaneExtractor] = None
        if self.segmentation_source == 'segformer' and self.enable_road_lane_extraction:
            self.lane_extractor = LaneExtractor(road_grayscale_id=0)
            self.get_logger().info('LaneExtractor initialized for road lane extraction (segformer mode)')

        # Semantic class IDs used by road lane extraction
        self._road_id = SEMANTIC_BY_NAME.get('road', 1)
        self._road_line_id = SEMANTIC_BY_NAME.get('road_line', 24)

        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================== Cached RGB for road lane extraction ====================
        self._latest_rgb: Optional[np.ndarray] = None

        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()

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
            # NOTE: No RGB subscription needed - ground truth already has lane markings.
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

        # Road lane extraction semantic map (with lane markings injected)
        if self.enable_road_lane_extraction:
            self.road_lane_extraction_pub = self.create_publisher(Image, road_lane_extraction_topic, 10)
            if self.publish_colored:
                self.road_lane_extraction_colored_pub = self.create_publisher(
                    Image, f'{road_lane_extraction_topic}/colored', 10)
            # Debug publishers
            self.road_mask_pub = self.create_publisher(Image, '/perception/road_mask', 10)
            self.lane_mask_pub = self.create_publisher(Image, '/perception/lane_markings', 10)

        # ==================== State ====================
        self._frame_count = 0
        self._total_time = 0.0
        self._last_lane_mask: Optional[np.ndarray] = None

        self.get_logger().info(
            f'Semantic Segmentation Node initialized! '
            f'Publishing to: {output_topic}'
            + (f' | road_lane_extraction to: {road_lane_extraction_topic}' if self.enable_road_lane_extraction else '')
        )

    # ------------------------------------------------------------------ #
    #  SegFormer callback                                                 #
    # ------------------------------------------------------------------ #

    def rgb_callback(self, msg: Image):
        """Run SegFormer on an RGB frame and publish semantic maps."""
        start_time = time.perf_counter()

        image_np = self._ros_image_to_numpy_rgb(msg)
        if image_np is None:
            return

        self._latest_rgb = image_np
        segmentation = self.segformer.predict(image_np)

        # Cityscapes classes (0-18) -> CARLA classes (+1)
        segmentation_carla = (segmentation + 1).astype(np.uint8)

        self._process_and_publish(segmentation_carla, msg.header)
        self._update_stats(start_time, 'SegFormer')

    # ------------------------------------------------------------------ #
    #  Ground-truth callback                                              #
    # ------------------------------------------------------------------ #

    def ground_truth_callback(self, msg: Image):
        """Publish CARLA ground truth semantic map (already has lane markings)."""
        start_time = time.perf_counter()

        try:
            segmentation = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        self._process_and_publish(segmentation, msg.header)
        self._update_stats(start_time, 'Ground Truth')

    # ------------------------------------------------------------------ #
    #  Unified publish pipeline                                           #
    # ------------------------------------------------------------------ #

    def _process_and_publish(self, segmentation: np.ndarray, header, publish_raw: bool = True):
        """Publish raw semantic map and road lane extraction map."""
        if publish_raw:
            self._publish_image(segmentation, header, 'mono8', self.semantic_pub)

        if self.enable_road_lane_extraction:
            extracted_map, extraction_ok = self._extract_road_lane_markings(segmentation.copy())

            # Publish road lane extraction mono8
            self._publish_image(extracted_map, header, 'mono8', self.road_lane_extraction_pub)

            # Debug publishers
            if extraction_ok:
                road_mask = ((extracted_map == self._road_id) | (extracted_map == self._road_line_id)).astype(np.uint8) * 255
                self._publish_image(road_mask, header, 'mono8', self.road_mask_pub)
                if self._last_lane_mask is not None:
                    self._publish_image(self._last_lane_mask, header, 'mono8',
                                        self.lane_mask_pub)

            # Publish colored visualizations
            if self.publish_colored:
                if publish_raw:
                    self._publish_image(colorize_mask(segmentation), header, 'rgb8',
                                        self.colored_pub)
                self._publish_image(colorize_mask(extracted_map), header, 'rgb8',
                                    self.road_lane_extraction_colored_pub)
        else:
            if self.publish_colored and publish_raw:
                self._publish_image(colorize_mask(segmentation), header, 'rgb8',
                                    self.colored_pub)

    def _extract_road_lane_markings(self, segmentation: np.ndarray):
        """Extract road lane markings into segmentation via LaneExtractor.

        Returns:
            (extracted_map, True) on success, (segmentation, False) on skip/failure.
        """
        rgb = self._latest_rgb

        if rgb is None:
            return segmentation, False

        if rgb.shape[:2] != segmentation.shape[:2]:
            self.get_logger().warn(
                f'Size mismatch: RGB {rgb.shape[:2]} vs Semantic '
                f'{segmentation.shape[:2]}. Publishing unextracted map.')
            return segmentation, False

        try:
            extracted_map, lane_mask = self.lane_extractor.extract_road_lane_markings(
                segmentation, rgb,
                road_class_id=self._road_id,
                road_line_class_id=self._road_line_id,
            )
            self._last_lane_mask = lane_mask
            return extracted_map, True
        except Exception as e:
            self.get_logger().error(f'Road lane extraction failed: {e}')
            return segmentation, False

    # ------------------------------------------------------------------ #
    #  Publishing helpers                                                 #
    # ------------------------------------------------------------------ #

    def _publish_image(self, array: np.ndarray, header, encoding: str, publisher):
        """Publish a numpy array as a ROS2 Image on *publisher*."""
        msg = self.cv_bridge.cv2_to_imgmsg(array, encoding=encoding)
        msg.header = header
        publisher.publish(msg)

    def _ros_image_to_numpy_rgb(self, msg: Image) -> np.ndarray:
        """Convert ROS2 Image (rgb8) to numpy array (H, W, 3)."""
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _update_stats(self, start_time: float, source: str):
        """Log average FPS every 30 frames."""
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
