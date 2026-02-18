#!/usr/bin/env python3
"""
Road Extraction Node
=====================

ROS2 node that performs two jobs:

1. **Semantic Fusion** — Takes a mono8 semantic map (e.g. from SegFormer) plus
   an RGB camera image, detects lane markings via adaptive thresholding within
   the road region, and injects class 24 (road_line) into the map.
   This fused map is a drop-in replacement for the CARLA ground-truth semantic
   sensor and is consumed directly by the mapping/costmap node.

2. **Lane Visualization** — Optionally produces IPM (bird's-eye-view) lane
   overlays, road masks, and other debug images.

Parameters:
- semantic_topic:  Input semantic topic (default: /perception/semantic_map)
- rgb_topic:       Input RGB camera topic
- output_topic:    Fused semantic map output (default: /perception/semantic_map/fused)
- publish_colored: Publish an rgb8 colored visualization of the fused map
- enable_ipm:      Enable IPM-based lane visualization outputs
- fov / pitch / camera_height: Camera parameters for IPM

Subscribes:
- /perception/semantic_map (sensor_msgs/Image, mono8) — from semantic_segmentation_node
- /carla/ego_vehicle/sensor/camera/rgb/front/image (sensor_msgs/Image) — RGB camera

Publishes:
- /perception/semantic_map/fused          (sensor_msgs/Image, mono8) — fused semantic map
- /perception/semantic_map/fused/colored  (sensor_msgs/Image, rgb8)  — colored visualization
- /perception/road_mask                   (sensor_msgs/Image, mono8) — binary road mask
- /perception/lane_markings               (sensor_msgs/Image, mono8) — lane markings mask
- /perception/lane_markings/ipm           (sensor_msgs/Image, rgb8)  — IPM lane overlay  [optional]
- /perception/lane_markings/overlay       (sensor_msgs/Image, rgb8)  — lane overlay       [optional]
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


class RoadExtractionNode(Node):
    """
    ROS2 Node for road/lane extraction **and** semantic fusion.

    Subscribes to a mono8 semantic segmentation map and RGB images.
    Detects lane markings, injects road_line (class 24) into the semantic map,
    and publishes the fused result for downstream mapping.
    """

    def __init__(self):
        super().__init__('road_extraction_node')

        # ==================== Parameters ====================
        self.declare_parameter('semantic_topic', '/perception/semantic_map')
        self.declare_parameter('rgb_topic', '/carla/ego_vehicle/sensor/camera/rgb/front/image')
        self.declare_parameter('output_topic', '/perception/semantic_map/fused')
        self.declare_parameter('publish_colored', True)
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
        self.output_topic = self.get_parameter('output_topic').value
        self.publish_colored = self.get_parameter('publish_colored').value
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
            self.get_parameter('ipm_x_max').value,
        )
        self.ipm_y_range = (
            self.get_parameter('ipm_y_min').value,
            self.get_parameter('ipm_y_max').value,
        )
        self.pixels_per_meter = self.get_parameter('pixels_per_meter').value

        # ==================== Initialize RoadExtractor ====================
        # road_grayscale_id is only used by the legacy colored-semantic path;
        # fuse_lane_markings() works directly with class IDs.
        self.road_extractor = RoadExtractor(road_grayscale_id=0)
        self.get_logger().info('RoadExtractor initialized (fusion mode)')

        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================== Message Sync Storage ====================
        self._latest_rgb: Optional[np.ndarray] = None

        # ==================== Subscribers ====================
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic, self.semantic_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to semantic: {semantic_topic}')

        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')

        # ==================== Publishers ====================
        # Primary outputs — consumed by mapping node
        self.fused_pub = self.create_publisher(Image, self.output_topic, 10)
        if self.publish_colored:
            self.colored_pub = self.create_publisher(
                Image, f'{self.output_topic}/colored', 10)

        # Debug / visualization outputs
        self.road_mask_pub = self.create_publisher(Image, '/perception/road_mask', 10)
        self.lane_mask_pub = self.create_publisher(Image, '/perception/lane_markings', 10)

        if self.publish_ipm:
            self.ipm_pub = self.create_publisher(Image, '/perception/lane_markings/ipm', 10)

        if self.publish_overlay:
            self.overlay_pub = self.create_publisher(Image, '/perception/lane_markings/overlay', 10)

        # ==================== Stats ====================
        self._frame_count = 0
        self._total_time = 0.0

        self.get_logger().info(
            f'Road Extraction Node initialized!  '
            f'[{semantic_topic}] + [{rgb_topic}] → [{self.output_topic}]'
        )

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def rgb_callback(self, msg: Image):
        """Cache the latest RGB image for lane detection."""
        self._latest_rgb = self._ros_image_to_numpy_rgb(msg)

    def semantic_callback(self, msg: Image):
        """
        Main processing callback — triggered on every new semantic frame.

        Pipeline:
            1. Parse mono8 semantic map (class IDs)
            2. Fuse lane markings (road_extraction.fuse_lane_markings)
            3. Publish fused semantic map + colored visualization
            4. Publish debug outputs (road_mask, lane_mask, IPM, overlay)
        """
        start_time = time.perf_counter()

        # --- 1. Parse semantic map ---
        if msg.encoding != 'mono8':
            self.get_logger().error(
                f'Expected mono8 semantic map, got {msg.encoding}')
            return

        sem = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width).copy()  # writable copy

        # --- 2. Need RGB to detect lanes ---
        if self._latest_rgb is None:
            # No RGB yet — pass through original semantic map unchanged
            self._publish_fused(sem, msg.header)
            return

        rgb = self._latest_rgb
        if rgb.shape[:2] != sem.shape:
            self.get_logger().warn(
                f'Size mismatch: RGB {rgb.shape[:2]} vs Semantic {sem.shape}. '
                f'Passing through unfused.')
            self._publish_fused(sem, msg.header)
            return

        # --- 3. Fuse lane markings into semantic map ---
        fused_map, lane_mask = self.road_extractor.fuse_lane_markings(sem, rgb)

        # --- 4. Publish fused semantic map ---
        self._publish_fused(fused_map, msg.header)

        # --- 5. Publish debug outputs ---
        # Road mask (from class 1 — note: fuse already modified the map, so
        # build mask from original road pixels + injected road_line pixels)
        road_id = SEMANTIC_BY_NAME['road']
        road_line_id = SEMANTIC_BY_NAME['road_line']
        road_mask = np.where(
            (fused_map == road_id) | (fused_map == road_line_id), 255, 0
        ).astype(np.uint8)

        road_msg = self._numpy_to_ros_image(road_mask, msg.header, 'mono8')
        self.road_mask_pub.publish(road_msg)

        lane_msg = self._numpy_to_ros_image(lane_mask, msg.header, 'mono8')
        self.lane_mask_pub.publish(lane_msg)

        # IPM visualization (optional)
        if self.publish_ipm and np.count_nonzero(lane_mask) > 0:
            try:
                _, ipm_image, ipm_lane_mask, _, _ = \
                    self.road_extractor.extract_lanes_ipm(
                        rgb, road_mask,
                        fov=self.fov,
                        pitch=self.pitch,
                        camera_height=self.camera_height,
                        ipm_x_range=self.ipm_x_range,
                        ipm_y_range=self.ipm_y_range,
                        pixels_per_meter=self.pixels_per_meter,
                    )
                ipm_viz = ipm_image.copy()
                ipm_viz[ipm_lane_mask > 0] = [0, 255, 0]
                ipm_msg = self._numpy_to_ros_image(ipm_viz, msg.header, 'rgb8')
                self.ipm_pub.publish(ipm_msg)
            except Exception as e:
                self.get_logger().error(f'IPM lane viz failed: {e}')

        # Overlay visualization (optional)
        if self.publish_overlay:
            overlay = rgb.copy()
            road_color = np.array([128, 64, 128], dtype=np.uint8)
            road_only_mask = (fused_map == road_id).astype(np.uint8) * 255
            overlay[road_only_mask > 0] = (
                overlay[road_only_mask > 0] * 0.6 + road_color * 0.4
            ).astype(np.uint8)
            overlay[lane_mask > 0] = [0, 255, 0]
            overlay_msg = self._numpy_to_ros_image(overlay, msg.header, 'rgb8')
            self.overlay_pub.publish(overlay_msg)

        # --- Stats ---
        elapsed = (time.perf_counter() - start_time) * 1000
        self._frame_count += 1
        self._total_time += elapsed

        if self._frame_count % 30 == 0:
            avg_time = self._total_time / self._frame_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            lane_px = np.count_nonzero(lane_mask)
            self.get_logger().info(
                f'Fusion+Lane: {elapsed:.1f}ms (avg: {avg_time:.1f}ms, ~{fps:.1f} FPS), '
                f'lane_px={lane_px}'
            )

    # ------------------------------------------------------------------ #
    #  Publishing helpers                                                 #
    # ------------------------------------------------------------------ #

    def _publish_fused(self, sem: np.ndarray, header):
        """Publish fused mono8 semantic map and optional colored visualization."""
        fused_msg = self._numpy_to_ros_image(sem, header, 'mono8')
        self.fused_pub.publish(fused_msg)

        if self.publish_colored:
            colored = colorize_mask(sem)
            col_msg = self._numpy_to_ros_image(colored, header, 'rgb8')
            self.colored_pub.publish(col_msg)

    # ------------------------------------------------------------------ #
    #  Conversion helpers                                                 #
    # ------------------------------------------------------------------ #

    def _ros_image_to_numpy_rgb(self, msg: Image) -> Optional[np.ndarray]:
        """Convert ROS2 Image message to RGB numpy array."""
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
        """Convert numpy array to ROS2 Image message."""
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
        node.get_logger().info("Shutting down RoadExtractionNode due to KeyboardInterrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
