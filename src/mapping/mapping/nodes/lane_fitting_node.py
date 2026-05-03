#!/usr/bin/env python3
"""
Lane Fitting Node
=================

ROS2 node for lane detection from semantic segmentation images.

Lane Fitting Pipeline (from BEV):
---------------------------------
1. Receives semantic segmentation image (mono8)
2. LaneDetector transforms to BEV using IPM (Inverse Perspective Mapping)
3. Histogram peak detection finds lane base positions
4. Sliding windows track lane pixels from vehicle to horizon
5. 2nd-order polynomials fitted to each detected lane
6. Results published as LaneDetection msg and RViz markers

Boundary-Based Lane Estimation Pipeline:
----------------------------------------
1. BoundaryExtractor creates BEV from semantic image (road + road_line)
2. Sliding window extracts left/right road boundaries
3. LocalLaneEstimator divides road into lanes using a priori knowledge
4. Results published as EstimatedLanes msg and boundary markers

Debug Visualization:
--------------------
Set enable_debug_viz:=true to enable the BEV debug image.
The '/mapping/debug/lane_bev' topic shows:
- Yellow pixels: Detected road_line pixels in BEV
- Colored filled rectangles: Sliding window search regions (one color per lane)
- Colored polylines: Fitted polynomial curves
- White line: Vehicle centerline

Parameters:
-----------
- semantic_source: 'road_lane_extraction' (perception) or 'ground_truth' (CARLA GT)
- camera_info_topic: Camera info for intrinsics
- camera_frame: TF frame name for the camera sensor
- enable_debug_viz: Publish debug BEV images (default: false)
- lane_config_file: Path to YAML file with lane structure parameters

Subscribes:
-----------
- Semantic segmentation (mono8) - topic depends on semantic_source parameter
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info (CameraInfo)

Publishes:
----------
- /mapping/lane_detection (LaneDetection) - polynomial fits + metrics
- /mapping/lanes (MarkerArray) - RViz lane line visualization
- /mapping/debug/lane_bev (Image, bgr8) - BEV debug visualization (if enabled)
- /mapping/estimated_lanes (EstimatedLanes) - estimated lanes from boundaries
- /mapping/boundary_markers (MarkerArray) - drivable boundary visualization
- /mapping/debug/boundary (Image) - boundary extraction debug

Changelog:
----------
2026-04-01: Refactored boundary extraction:
    - Added BoundaryExtractor class (IPM + sliding window extraction)
    - LocalLaneEstimator now receives boundaries directly
    - Cleaner separation of concerns
2026-03-25: Fixed curve handling:
    - Increased max_quadratic from 0.002 to 0.015 (allows tighter curves)
    - Semi-transparent filled windows in debug visualization
"""

import numpy as np
from pathlib import Path as FilePath
from typing import Optional, Dict, Any

import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time

from autoiq_msgs.msg import LaneDetection as LaneDetectionMsg
from autoiq_msgs.msg import EstimatedLanes, EstimatedLane

from autoiq_common import CameraParams, CameraParamsProvider

from mapping.core.lane_fitting import LaneDetector
from mapping.core.local_lane_estimator import LocalLaneEstimator
from mapping.core.boundary_extractor import BoundaryExtractor

from cv_bridge import CvBridge


# ============================================================================
# Lane Fitting Node
# ============================================================================


class LaneFittingNode(Node):
    """ROS2 Node for lane polynomial fitting and lane estimation from boundaries."""

    def __init__(self):
        super().__init__('lane_fitting_node')

        # ==================== Parameters ====================
        self.declare_parameter('semantic_source', 'road_lane_extraction')  # 'road_lane_extraction' or 'ground_truth'
        self.declare_parameter(
            'camera_info_topic',
            '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info')
        self.declare_parameter(
            'camera_frame',
            'ego_vehicle/sensor/camera/rgb/front')
        self.declare_parameter('enable_debug_viz', False)
        self.declare_parameter('base_frame', 'ego_vehicle')

        # Lane config file path (relative to mapping package share directory)
        self.declare_parameter('lane_config_file', 'config/local_lane_data.yaml')

        # Allow the ground-truth topic to be overridden so the same node
        # can run against either the CARLA bridge or a custom rosbag /
        # alternative semantic source. The road_lane_extraction topic
        # stays fixed because it's an internal contract with perception.
        self.declare_parameter(
            'ground_truth_semantic_topic',
            '/carla/ego_vehicle/ground_truth/camera/semantic/front/image')

        # Resolve semantic source → only the image topic changes
        semantic_source = self.get_parameter('semantic_source').value
        if semantic_source == 'ground_truth':
            semantic_topic = self.get_parameter('ground_truth_semantic_topic').value
        else:
            semantic_topic = '/perception/semantic_map/road_lane_extraction'

        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value
        self.base_frame = self.get_parameter('base_frame').value

        # ==================== Load Lane Config from YAML ====================
        lane_config = self._load_lane_config()
        self.get_logger().info(f'Loaded lane config: lane_count={lane_config["lane_count"]}, '
                               f'total_road_width={lane_config["total_road_width"]}m')

        self.get_logger().info(f'Semantic source: {semantic_source}')
        self.get_logger().info(f'Subscribing to semantic map: {semantic_topic}')

        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()

        # ==================== Camera Params Provider ====================
        self.camera = CameraParamsProvider(
            node=self,
            camera_info_topic=camera_info_topic,
            base_frame=self.base_frame,
            camera_frame=self.camera_frame,
            on_ready=self._on_camera_ready,
            qos=sensor_qos,
        )

        # ==================== Lane Detector Init Gate ====================
        # LaneDetector is created once CameraInfo + camera TF are available.
        self.lane_detector: Optional[LaneDetector] = None
        self._lane_detector_initialized = False

        # ==================== Boundary Extractor ====================
        # Extracts road boundaries from semantic image using IPM + sliding window
        self.boundary_extractor = BoundaryExtractor(
            bev_x_range=tuple(lane_config['bev_x_range']),
            bev_y_range=tuple(lane_config['bev_y_range']),
            pixels_per_meter=lane_config['pixels_per_meter'],
        )

        # ==================== Lane Estimator (from boundaries) ====================
        # Uses config from YAML file - estimates lanes from road boundaries
        self.lane_estimator = LocalLaneEstimator(
            lane_count=lane_config['lane_count'],
            total_road_width=lane_config['total_road_width'],
            clearance_from_wall=lane_config['clearance_from_wall'],
            clearance_from_divider=lane_config['clearance_from_divider'],
            min_road_width=lane_config['min_road_width'],
            max_road_width=lane_config['max_road_width'],
        )

        # ==================== Subscribers ====================
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic, self._semantic_callback, sensor_qos)

        # ==================== Publishers ====================
        # Lane fitting publishers
        self.lane_detection_pub = self.create_publisher(
            LaneDetectionMsg, '/mapping/lane_detection', 10)
        self.lane_markers_pub = self.create_publisher(
            MarkerArray, '/mapping/lanes', 10)
        if self.enable_debug_viz:
            self.debug_bev_pub = self.create_publisher(
                Image, '/mapping/debug/lane_bev', sensor_qos)

        # Lane estimation publishers (boundary-based estimation)
        self.estimated_lanes_pub = self.create_publisher(
            EstimatedLanes, '/mapping/estimated_lanes', 10)
        self.boundary_markers_pub = self.create_publisher(
            MarkerArray, '/mapping/boundary_markers', 10)
        self.boundary_debug_pub = self.create_publisher(
            Image, '/mapping/debug/boundary', sensor_qos)

        self.get_logger().info('Lane Fitting Node initialized!')

    # ------------------------------------------------------------------ #
    #  YAML Config Loading                                                #
    # ------------------------------------------------------------------ #

    def _load_lane_config(self) -> Dict[str, Any]:
        """Load lane structure configuration from YAML file."""
        config_file = self.get_parameter('lane_config_file').value
        config_path = FilePath(get_package_share_directory('mapping')) / config_file

        if not config_path.exists():
            self.get_logger().error(f'Lane config not found: {config_path}')
            raise FileNotFoundError(f'Lane config not found: {config_path}')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.get_logger().info(f'Loaded lane config from: {config_path}')
        return config

    # ------------------------------------------------------------------ #
    #  Camera Ready Callback                                              #
    # ------------------------------------------------------------------ #

    def _on_camera_ready(self, params: CameraParams):
        """Called when camera intrinsics + extrinsics are available."""
        # Initialize LaneDetector
        self.lane_detector = LaneDetector(
            image_width=params.width,
            image_height=params.height,
        )
        self.lane_detector.set_camera_params(
            fov=params.fov_deg,
            pitch=params.pitch_deg,
            camera_height=params.height_m,
        )

        # Initialize BoundaryExtractor with camera params
        self.boundary_extractor.set_camera_params(
            fov=params.fov_deg,
            pitch=params.pitch_deg,
            camera_height=params.height_m,
        )

        self._lane_detector_initialized = True
        self.get_logger().info(
            f'LaneDetector initialized: '
            f'{params.width}x{params.height}, '
            f'FOV={params.fov_deg:.1f}deg, '
            f'pitch={params.pitch_deg:.1f}deg, '
            f'height={params.height_m:.2f}m')

    # ------------------------------------------------------------------ #
    #  Semantic Map Callback                                              #
    # ------------------------------------------------------------------ #

    def _semantic_callback(self, msg: Image):
        """Process semantic map and run both lane fitting and lane estimation."""
        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        # Wait for camera params before processing
        if not self.camera.ready:
            self.camera.try_get_extrinsics()
            return

        # Run lane fitting (polynomial detection from road_line markings)
        if self._lane_detector_initialized:
            self._run_lane_fitting(semantic_img, msg.header)

        # Run lane estimation from boundaries (uses shared IPM)
        self._estimate_lanes_from_boundaries(
            semantic_img=semantic_img,
            stamp=msg.header.stamp,
        )

    # ------------------------------------------------------------------ #
    #  Lane Detection (polynomial fitting)                                #
    # ------------------------------------------------------------------ #

    def _run_lane_fitting(self, semantic_img: np.ndarray, header):
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
        det_msg.header.frame_id = self.base_frame
        det_msg.valid = result.valid
        det_msg.confidence = float(result.confidence)

        # Use all_fits from multi-lane detection
        lane_fits = []
        if result.all_fits:
            lane_fits = [np.asarray(f, dtype=np.float64) for f in result.all_fits]
        elif result.left_fit is not None or result.right_fit is not None:
            # Fallback for backwards compatibility
            if result.left_fit is not None:
                lane_fits.append(np.asarray(result.left_fit, dtype=np.float64))
            if result.right_fit is not None:
                lane_fits.append(np.asarray(result.right_fit, dtype=np.float64))

        # Sort by near-ego lateral position for deterministic ordering
        if lane_fits and result.bev_params:
            y_eval = result.bev_params['height'] - 10
            lane_fits.sort(key=lambda coeffs: float(np.polyval(coeffs, y_eval)))

        det_msg.lane_count = len(lane_fits)
        det_msg.poly_coeffs_flat = []
        for coeffs in lane_fits:
            det_msg.poly_coeffs_flat.extend(coeffs.tolist())

        bev = result.bev_params
        det_msg.bev_width = bev['width']
        det_msg.bev_height = bev['height']
        lateral_range = bev['lateral_range']
        forward_range = bev['forward_range']
        det_msg.bev_x_min = float(lateral_range[0])
        det_msg.bev_x_max = float(lateral_range[1])
        det_msg.bev_y_min = float(forward_range[0])
        det_msg.bev_y_max = float(forward_range[1])

        self.lane_detection_pub.publish(det_msg)

    def _publish_lane_markers(self, result):
        """Publish lane polynomial fits as RViz MarkerArray."""
        marker_array = MarkerArray()
        timestamp = Time(sec=0, nanosec=0)

        # Clear previous markers
        clear = Marker()
        clear.header.frame_id = self.base_frame
        clear.header.stamp = timestamp
        clear.ns = 'lanes'
        clear.id = 0
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        if not result.valid:
            self.lane_markers_pub.publish(marker_array)
            return

        bev = result.bev_params

        # Use all_fits from multi-lane detection
        lane_fits = []
        if result.all_fits:
            lane_fits = [np.asarray(f, dtype=np.float64) for f in result.all_fits]
        elif result.left_fit is not None or result.right_fit is not None:
            # Fallback for backwards compatibility
            if result.left_fit is not None:
                lane_fits.append(np.asarray(result.left_fit, dtype=np.float64))
            if result.right_fit is not None:
                lane_fits.append(np.asarray(result.right_fit, dtype=np.float64))

        # Sort by near-ego lateral position
        if lane_fits:
            y_eval = bev['height'] - 10
            lane_fits.sort(key=lambda coeffs: float(np.polyval(coeffs, y_eval)))

        palette = [
            ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.8),
            ColorRGBA(r=0.2, g=1.0, b=0.2, a=0.8),
            ColorRGBA(r=0.2, g=0.4, b=1.0, a=0.8),
            ColorRGBA(r=1.0, g=0.8, b=0.2, a=0.8),
            ColorRGBA(r=0.8, g=0.2, b=0.8, a=0.8),
            ColorRGBA(r=0.2, g=0.8, b=0.8, a=0.8),
        ]

        for idx, coeffs in enumerate(lane_fits):
            marker_array.markers.append(
                self._make_lane_marker(
                    coeffs,
                    bev,
                    marker_id=1 + idx,
                    color=palette[idx % len(palette)],
                    stamp=timestamp,
                    frame_id=self.base_frame))

        self.lane_markers_pub.publish(marker_array)

    @staticmethod
    def _make_lane_marker(fit, bev_params, marker_id, color, stamp, frame_id):
        """Create a LINE_STRIP marker from polynomial coefficients in BEV space."""
        marker = Marker()
        marker.header.frame_id = frame_id
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
    #  Lane Estimation from Boundaries                                    #
    # ------------------------------------------------------------------ #

    def _estimate_lanes_from_boundaries(
        self,
        semantic_img: np.ndarray,
        stamp,
    ):
        """Estimate lanes from road boundaries.

        Pipeline:
        1. BoundaryExtractor extracts left/right boundaries from semantic image
        2. LocalLaneEstimator divides road into lanes using a priori knowledge
        """
        # 1. Extract boundaries using BoundaryExtractor
        boundary_result = self.boundary_extractor.extract(semantic_img)

        # 2. Publish boundaries as MarkerArray
        self._publish_boundary_markers(
            boundary_result.left_boundary,
            boundary_result.right_boundary,
            stamp,
        )

        # 3. Publish boundary debug image
        if boundary_result.debug_image is not None:
            debug_msg = self.cv_bridge.cv2_to_imgmsg(
                boundary_result.debug_image, encoding='bgr8')
            debug_msg.header.stamp = stamp
            debug_msg.header.frame_id = self.base_frame
            self.boundary_debug_pub.publish(debug_msg)

        # 4. Estimate lanes from boundaries
        lane_result = self.lane_estimator.estimate_from_boundaries(
            left_boundary=boundary_result.left_boundary,
            right_boundary=boundary_result.right_boundary,
        )

        # 5. Publish estimated lanes message
        self._publish_estimated_lanes(lane_result, stamp)

    def _publish_estimated_lanes(self, result, stamp):
        """Publish EstimatedLanes message with structured lane data."""
        msg = EstimatedLanes()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.valid = result.valid
        msg.status_code = result.status_code
        msg.road_width_avg = result.road_width_avg
        msg.lane_count = result.lane_count

        # Convert numpy arrays to geometry_msgs/Point[]
        for x, y in result.left_boundary:
            msg.left_boundary.append(Point(x=float(x), y=float(y), z=0.0))
        for x, y in result.right_boundary:
            msg.right_boundary.append(Point(x=float(x), y=float(y), z=0.0))

        # Convert lane results
        for lane_result in result.lanes:
            lane_msg = EstimatedLane()
            lane_msg.lane_index = lane_result.lane_index
            lane_msg.width_avg = lane_result.width_avg
            lane_msg.is_valid = lane_result.is_valid
            lane_msg.is_partial = lane_result.is_partial

            for x, y in lane_result.left_boundary:
                lane_msg.left_boundary.append(Point(x=float(x), y=float(y), z=0.0))
            for x, y in lane_result.right_boundary:
                lane_msg.right_boundary.append(Point(x=float(x), y=float(y), z=0.0))
            for x, y in lane_result.centerline:
                lane_msg.centerline.append(Point(x=float(x), y=float(y), z=0.0))
            msg.lanes.append(lane_msg)

        self.get_logger().info(
            f'Publishing EstimatedLanes with {len(msg.lanes)} lanes',
            throttle_duration_sec=5.0)
        self.estimated_lanes_pub.publish(msg)

    def _publish_boundary_markers(
        self,
        left_pts: Optional[np.ndarray],
        right_pts: Optional[np.ndarray],
        stamp,
    ):
        """Publish drivable area boundaries as MarkerArray (RED color)."""
        marker_array = MarkerArray()

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.stamp = stamp
        clear_marker.header.frame_id = self.base_frame
        clear_marker.ns = 'drivable_boundaries'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # Left drivable boundary (RED)
        if left_pts is not None and len(left_pts) > 1:
            left_marker = Marker()
            left_marker.header.stamp = stamp
            left_marker.header.frame_id = self.base_frame
            left_marker.ns = 'drivable_boundaries'
            left_marker.id = 1
            left_marker.type = Marker.LINE_STRIP
            left_marker.action = Marker.ADD
            left_marker.scale.x = 0.15  # Line width
            left_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # RED
            left_marker.pose.orientation.w = 1.0

            for x, y in left_pts:
                left_marker.points.append(Point(x=float(x), y=float(y), z=0.1))

            marker_array.markers.append(left_marker)

        # Right drivable boundary (RED)
        if right_pts is not None and len(right_pts) > 1:
            right_marker = Marker()
            right_marker.header.stamp = stamp
            right_marker.header.frame_id = self.base_frame
            right_marker.ns = 'drivable_boundaries'
            right_marker.id = 2
            right_marker.type = Marker.LINE_STRIP
            right_marker.action = Marker.ADD
            right_marker.scale.x = 0.15  # Line width
            right_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # RED
            right_marker.pose.orientation.w = 1.0

            for x, y in right_pts:
                right_marker.points.append(Point(x=float(x), y=float(y), z=0.1))

            marker_array.markers.append(right_marker)

        self.boundary_markers_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFittingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
