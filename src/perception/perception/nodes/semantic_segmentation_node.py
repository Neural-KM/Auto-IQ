#!/usr/bin/env python3
"""
Semantic Segmentation Node
===========================

ROS2 node that publishes semantic segmentation maps from either:
- SegFormer model inference (learned segmentation)
- CARLA ground truth semantic camera (perfect labels)

Also performs lane polynomial fitting on the semantic map and publishes
lane detection results (LaneDetection msg + RViz markers).

Flow:
- **ground_truth mode**: CARLA ground truth already contains lane markings
  (class 24 road_line).  The map is published as-is to both the raw and
  road_lane_extraction topics — no LaneExtractor processing is needed.
- **segformer mode**: SegFormer output does NOT contain lane markings.
  When *enable_road_lane_extraction* is True, the LaneExtractor detects lane
  markings via adaptive thresholding on the RGB image and injects class 24
  (road_line) into the semantic map.  The extracted map is published on a
  separate topic so that the mapping/costmap node receives lane-aware semantics.

After semantic processing, a LaneDetector instance runs BEV polynomial fitting
on the semantic map and publishes lane geometry for path planning.

Parameters:
- segmentation_source: 'segformer' or 'ground_truth' (default: 'ground_truth')
- model_name: SegFormer model variant (b0-b5), set in perception.launch.py
- device: 'cuda' or 'cpu' for SegFormer inference
- rgb_topic: Input RGB camera topic
- ground_truth_topic: Input ground truth semantic topic (for ground_truth mode)
- output_topic: Base semantic map topic (default: /perception/semantic_map)
- road_lane_extraction_topic: Road lane extraction map with lane markings (default: /perception/semantic_map/road_lane_extraction)
- enable_road_lane_extraction: Enable lane-marking extraction via LaneExtractor (segformer mode only, default: True)
- publish_colored: Publish colored visualization (default: True)
- camera_info_topic: Camera info for intrinsics (needed for lane fitting)
- camera_frame: TF frame name for the camera sensor (needed for lane fitting)
- enable_debug_viz: Publish debug BEV images for lane fitting

Subscribes:
- /carla/ego_vehicle/sensor/camera/rgb/front/image (RGB camera — segformer mode)
- /carla/ego_vehicle/ground_truth/camera/semantic/front/image (ground_truth mode)
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info (camera intrinsics)

Publishes:
- /perception/semantic_map           (mono8) — raw semantic segmentation
- /perception/semantic_map/colored    (rgb8)  — colored visualization
- /perception/semantic_map/road_lane_extraction      (mono8) — extracted with lane markings
- /perception/semantic_map/road_lane_extraction/colored (rgb8) — colored road lane extraction visualization
- /perception/road_mask               (mono8) — binary road mask  [debug]
- /perception/lane_markings           (mono8) — lane markings mask [debug]
- /perception/lane_detection          (LaneDetection) — polynomial fits + metrics
- /perception/lanes                   (MarkerArray) — RViz lane line visualization
- /perception/debug/lane_bev          (Image, bgr8) — BEV debug visualization

"""

import numpy as np
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import tf2_ros

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time

from autoiq_msgs.msg import LaneDetection as LaneDetectionMsg

from perception.core.lane_extraction import LaneExtractor
from perception.core.lane_fitting import LaneDetector
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

        # Lane detection (polynomial fitting) parameters
        self.declare_parameter(
            'camera_info_topic',
            '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info')
        self.declare_parameter(
            'camera_frame',
            'ego_vehicle/sensor/camera/rgb/front')
        self.declare_parameter('enable_debug_viz', False)

        self.segmentation_source = self.get_parameter('segmentation_source').value
        model_name = self.get_parameter('model_name').value
        device = self.get_parameter('device').value
        self.publish_colored = self.get_parameter('publish_colored').value
        rgb_topic = self.get_parameter('rgb_topic').value
        ground_truth_topic = self.get_parameter('ground_truth_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.enable_road_lane_extraction = self.get_parameter('enable_road_lane_extraction').value
        road_lane_extraction_topic = self.get_parameter('road_lane_extraction_topic').value

        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value

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
        # Ground truth already contains lane markings — no extraction needed.
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

        # ==================== TF2 (for camera extrinsics) ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================== Lane Detector Init Gate ====================
        # LaneDetector is created once CameraInfo + camera TF are available.
        self.lane_detector: Optional[LaneDetector] = None
        self._lane_detector_initialized = False
        self._camera_info_received = False
        self._camera_tf_received = False
        self._cam_width: Optional[int] = None
        self._cam_height: Optional[int] = None
        self._cam_fov_deg: Optional[float] = None
        self._cam_pitch: Optional[float] = None
        self._cam_z: Optional[float] = None

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
            # NOTE: No RGB subscription needed — ground truth already has lane markings.
        else:
            self.get_logger().error(
                f'Invalid segmentation_source: {self.segmentation_source}. '
                f'Options are: segformer, ground_truth')
            raise ValueError(f'Invalid segmentation_source: {self.segmentation_source}')

        # CameraInfo subscription (needed for lane polynomial fitting)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic,
            self._camera_info_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to CameraInfo: {camera_info_topic}')

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

        # Lane detection publishers (polynomial fits + RViz markers)
        self.lane_detection_pub = self.create_publisher(
            LaneDetectionMsg, '/perception/lane_detection', 10)
        self.lane_markers_pub = self.create_publisher(
            MarkerArray, '/perception/lanes', 10)
        if self.enable_debug_viz:
            self.debug_bev_pub = self.create_publisher(
                Image, '/perception/debug/lane_bev', 10)

        # ==================== State ====================
        self._frame_count = 0
        self._total_time = 0.0
        self._last_lane_mask: Optional[np.ndarray] = None

        # Timer to periodically attempt lane detector initialization
        self._init_timer = self.create_timer(0.5, self._try_init_lane_detector)

        self.get_logger().info(
            f'Semantic Segmentation Node initialized! '
            f'Publishing to: {output_topic}'
            + (f' | road_lane_extraction to: {road_lane_extraction_topic}' if self.enable_road_lane_extraction else '')
        )
    
    # ------------------------------------------------------------------ #
    #  Lane Detector Initialization                                       #
    # ------------------------------------------------------------------ #

    def _try_init_lane_detector(self):
        """Periodically attempt to initialize the LaneDetector.

        Gated on having both CameraInfo (intrinsics) and camera TF
        (extrinsics: height + pitch).
        """
        if self._lane_detector_initialized:
            return

        if not self._camera_tf_received:
            self._try_get_camera_tf()

        if self._camera_info_received and self._camera_tf_received:
            self.lane_detector = LaneDetector(
                image_width=self._cam_width,
                image_height=self._cam_height,
            )
            self.lane_detector.set_camera_params(
                fov=self._cam_fov_deg,
                pitch=self._cam_pitch,
                camera_height=self._cam_z,
            )
            self._lane_detector_initialized = True
            self._init_timer.cancel()
            self.get_logger().info(
                f'LaneDetector initialized: '
                f'{self._cam_width}x{self._cam_height}, '
                f'FOV={self._cam_fov_deg:.1f}°, '
                f'pitch={self._cam_pitch:.1f}°, '
                f'height={self._cam_z:.2f}m')
        else:
            missing = []
            if not self._camera_info_received:
                missing.append('CameraInfo')
            if not self._camera_tf_received:
                missing.append('Camera TF')
            self.get_logger().info(
                f'LaneDetector waiting for: {", ".join(missing)}...',
                throttle_duration_sec=3.0)

    def _try_get_camera_tf(self):
        """Look up camera height and pitch from the TF tree."""
        transform = self.tf_buffer.lookup_transform(
            'ego_vehicle', self.camera_frame, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=2.0))

        self._cam_z = transform.transform.translation.z
        if self._cam_z is None or self._cam_z <= 0.0:
            raise RuntimeError(
                f'Invalid camera height from TF: {self._cam_z}')

        # Extract pitch from quaternion
        q = transform.transform.rotation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self._cam_pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))

        self._camera_tf_received = True
        self.get_logger().info(
            f'Camera TF received: '
            f'height={self._cam_z:.2f}m, pitch={self._cam_pitch:.1f}°')

    def _camera_info_callback(self, msg: CameraInfo):
        """Extract image dimensions and FOV from the first CameraInfo msg."""
        if self._camera_info_received:
            return

        self._cam_width = msg.width
        self._cam_height = msg.height
        fx = msg.k[0]
        if fx <= 0:
            self.get_logger().error(
                f'Invalid focal length fx={fx} in CameraInfo. '
                f'Cannot compute FOV — check camera calibration.')
            return

        self._cam_fov_deg = np.degrees(2 * np.arctan(self._cam_width / (2 * fx)))
        self._camera_info_received = True
        self.get_logger().info(
            f'CameraInfo: {self._cam_width}x{self._cam_height}, '
            f'FOV={self._cam_fov_deg:.1f}°')

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

        # Cityscapes classes (0-18) → CARLA classes (+1)
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
        """Publish raw semantic map, road lane extraction, and run lane polynomial fitting."""
        if publish_raw:
            self._publish_image(segmentation, header, 'mono8', self.semantic_pub)

        # Determine the map that contains lane markings  (used for polynomial fitting)
        lane_map = segmentation  # ground_truth already has road_line class

        if self.enable_road_lane_extraction:
            extracted_map, extraction_ok = self._extract_road(segmentation)

            # Publish road lane extraction mono8
            self._publish_image(extracted_map, header, 'mono8', self.road_lane_extraction_pub)

            # Debug publishers
            if extraction_ok:
                lane_map = extracted_map
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

        # ---- Lane polynomial fitting (runs when LaneDetector is ready) ----
        if self._lane_detector_initialized:
            self._run_lane_detection(lane_map, header)

    def _extract_road(self, segmentation: np.ndarray):
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
    #  Lane Detection (polynomial fitting)                                #
    # ------------------------------------------------------------------ #

    def _run_lane_detection(self, semantic_img: np.ndarray, header):
        """Run BEV + sliding window + polynomial fit and publish results."""
        result = self.lane_detector.detect(semantic_img)

        self._publish_lane_detection(result, header)
        self._publish_lane_markers(result)

        if self.enable_debug_viz:
            self._publish_debug_bev(header)

    def _publish_lane_detection(self, result, header):
        """Publish a LaneDetection message with polynomial coefficients."""
        det_msg = LaneDetectionMsg()
        det_msg.header.stamp = header.stamp
        det_msg.header.frame_id = 'ego_vehicle'
        det_msg.valid = result.valid
        det_msg.confidence = float(result.confidence)

        if result.left_fit is not None:
            det_msg.left_poly_coeffs = result.left_fit.tolist()
        if result.right_fit is not None:
            det_msg.right_poly_coeffs = result.right_fit.tolist()

        det_msg.lane_center = result.lane_center
        det_msg.left_lane_x = result.left_lane if result.left_lane is not None else -1
        det_msg.right_lane_x = result.right_lane if result.right_lane is not None else -1
        det_msg.lane_width_px = result.lane_width if result.lane_width is not None else -1

        bev = result.bev_params
        det_msg.bev_width = bev['width']
        det_msg.bev_height = bev['height']
        lateral_range = bev['lateral_range']
        forward_range = bev['forward_range']
        det_msg.bev_x_min = float(lateral_range[0])
        det_msg.bev_x_max = float(lateral_range[1])
        det_msg.bev_y_min = float(forward_range[0])
        det_msg.bev_y_max = float(forward_range[1])

        if result.lane_width_bev is not None:
            meters_per_px = (lateral_range[1] - lateral_range[0]) / bev['width']
            det_msg.lane_width_m = float(result.lane_width_bev * meters_per_px)
        else:
            det_msg.lane_width_m = -1.0

        self.lane_detection_pub.publish(det_msg)

    def _publish_lane_markers(self, result):
        """Publish lane polynomial fits as RViz MarkerArray."""
        marker_array = MarkerArray()
        timestamp = Time(sec=0, nanosec=0)

        # Clear previous markers
        clear = Marker()
        clear.header.frame_id = 'ego_vehicle'
        clear.header.stamp = timestamp
        clear.ns = 'lanes'
        clear.id = 0
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        if not result.valid:
            self.lane_markers_pub.publish(marker_array)
            return

        bev = result.bev_params

        if result.left_fit is not None:
            marker_array.markers.append(
                self._make_lane_marker(
                    result.left_fit, bev, marker_id=1,
                    color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),
                    stamp=timestamp))

        if result.right_fit is not None:
            marker_array.markers.append(
                self._make_lane_marker(
                    result.right_fit, bev, marker_id=2,
                    color=ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8),
                    stamp=timestamp))

        # Center line
        if result.left_fit is not None and result.right_fit is not None:
            center_fit = (result.left_fit + result.right_fit) / 2.0
            marker_array.markers.append(
                self._make_lane_marker(
                    center_fit, bev, marker_id=3,
                    color=ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.6),
                    stamp=timestamp))

        self.lane_markers_pub.publish(marker_array)

    @staticmethod
    def _make_lane_marker(fit, bev_params, marker_id, color, stamp):
        """Create a LINE_STRIP marker from polynomial coefficients in BEV space."""
        marker = Marker()
        marker.header.frame_id = 'ego_vehicle'
        marker.header.stamp = stamp
        marker.ns = 'lanes'
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.color = color

        bw = bev_params['width']
        bh = bev_params['height']
        lateral_range = bev_params['lateral_range']
        forward_range = bev_params['forward_range']

        for y_bev in np.linspace(0, bh - 1, 30):
            x_bev = np.polyval(fit, y_bev)
            x_world = (forward_range[1]
                       - (y_bev / bh) * (forward_range[1] - forward_range[0]))
            y_world = (lateral_range[0]
                       + (x_bev / bw) * (lateral_range[1] - lateral_range[0]))
            pt = Point(x=float(x_world), y=float(-y_world), z=0.0)
            marker.points.append(pt)

        return marker

    def _publish_debug_bev(self, header):
        """Publish the BEV lane-fit debug image."""
        debug = self.lane_detector.get_debug_images()
        lane_fit_img = debug.get('lane_fit') if debug is not None else None
        if lane_fit_img is None:
            return

        out = self.cv_bridge.cv2_to_imgmsg(lane_fit_img, encoding='bgr8')
        out.header = header
        self.debug_bev_pub.publish(out)

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
