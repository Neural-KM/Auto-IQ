#!/usr/bin/env python3
"""
Lane Fitting Node
=================

ROS2 node that performs:
1. Multi-lane polynomial fitting on semantic segmentation maps
2. Local lane data estimation from occupancy grid boundaries

Subscribes to semantic maps from the perception package and occupancy grid
from IPM node, publishes lane detection results and local lane data.

Parameters:
- semantic_source: 'road_lane_extraction' (perception) or 'ground_truth' (CARLA GT)
- camera_info_topic: Camera info for intrinsics
- camera_frame: TF frame name for the camera sensor
- enable_debug_viz: Publish debug BEV images
- grid_size: Size of occupancy grid in meters
- resolution: Occupancy grid resolution in meters/cell
- lane_count: Expected number of lanes for boundary-based estimation
- total_road_width: Expected road width for boundary estimation

Subscribes:
- Semantic segmentation (mono8) - topic depends on semantic_source parameter
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info (CameraInfo)
- /mapping/ipm_costmap (OccupancyGrid) - for boundary-based lane estimation

Publishes:
- /mapping/lane_detection (LaneDetection) - polynomial fits + metrics
- /mapping/lanes (MarkerArray) - RViz lane line visualization
- /mapping/debug/lane_bev (Image, bgr8) - BEV debug visualization
- /mapping/local_lane_data (LocalLaneData) - estimated lanes from boundaries
- /mapping/boundary/left (Path) - left road boundary
- /mapping/boundary/right (Path) - right road boundary
- /mapping/boundary_debug_image (Image) - boundary extraction debug
"""

import numpy as np
from typing import Optional, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import tf2_ros

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PoseStamped
from builtin_interfaces.msg import Time

from autoiq_msgs.msg import LaneDetection as LaneDetectionMsg
from autoiq_msgs.msg import LocalLaneData, Lane

from mapping.core.lane_fitting import LaneDetector
from mapping.core.local_lane_estimator import LocalLaneEstimator, LocalRoadEstimate

from cv_bridge import CvBridge


class LaneFittingNode(Node):
    """ROS2 Node for lane polynomial fitting and local lane estimation."""

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

        # Local lane estimation parameters (from occupancy grid)
        self.declare_parameter('grid_size', 40.0)    # meters (must match IPM node)
        self.declare_parameter('resolution', 0.4)    # meters per cell
        self.declare_parameter('lane_count', 4)
        self.declare_parameter('total_road_width', 22.0)
        self.declare_parameter('clearance_from_wall', 0.3)
        self.declare_parameter('clearance_from_divider', 0.3)
        self.declare_parameter('min_road_width', 10.0)
        self.declare_parameter('max_road_width', 30.0)

        # Resolve semantic source → only the image topic changes
        semantic_source = self.get_parameter('semantic_source').value
        if semantic_source == 'ground_truth':
            semantic_topic = '/carla/ego_vehicle/ground_truth/camera/semantic/front/image'
        else:
            semantic_topic = '/perception/semantic_map/road_lane_extraction'

        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value
        self.base_frame = self.get_parameter('base_frame').value

        # Grid parameters for local lane estimation
        grid_size = self.get_parameter('grid_size').value
        resolution = self.get_parameter('resolution').value
        self.grid_cells = int(grid_size / resolution)
        self.origin_x = -grid_size / 2.0
        self.origin_y = -grid_size / 2.0
        self.resolution = resolution

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

        # ==================== TF2 (for camera extrinsics) ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==================== Camera Params (packaged as dict) ====================
        self._camera_params: Dict[str, Any] = {
            'width': None,
            'height': None,
            'fov_deg': None,
            'pitch_deg': None,
            'height_m': None,
        }
        self._camera_info_received = False
        self._camera_tf_received = False

        # ==================== Lane Detector Init Gate ====================
        # LaneDetector is created once CameraInfo + camera TF are available.
        self.lane_detector: Optional[LaneDetector] = None
        self._lane_detector_initialized = False

        # ==================== Local Lane Estimator ====================
        self.lane_estimator = LocalLaneEstimator(
            grid_cells=self.grid_cells,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.resolution,
            lane_count=self.get_parameter('lane_count').value,
            total_road_width=self.get_parameter('total_road_width').value,
            clearance_from_wall=self.get_parameter('clearance_from_wall').value,
            clearance_from_divider=self.get_parameter('clearance_from_divider').value,
            min_road_width=self.get_parameter('min_road_width').value,
            max_road_width=self.get_parameter('max_road_width').value,
        )
        self._latest_semantic_img: Optional[np.ndarray] = None
        self._latest_semantic_stamp = None

        # ==================== Subscribers ====================
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic, self._semantic_callback, sensor_qos)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic,
            self._camera_info_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to CameraInfo: {camera_info_topic}')

        # Subscribe to occupancy grid for boundary-based lane estimation
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/mapping/ipm_costmap',
            self._costmap_callback, sensor_qos)
        self.get_logger().info('Subscribed to occupancy grid: /mapping/ipm_costmap')

        # ==================== Publishers ====================
        # Lane fitting publishers
        self.lane_detection_pub = self.create_publisher(
            LaneDetectionMsg, '/mapping/lane_detection', 10)
        self.lane_markers_pub = self.create_publisher(
            MarkerArray, '/mapping/lanes', 10)
        if self.enable_debug_viz:
            self.debug_bev_pub = self.create_publisher(
                Image, '/mapping/debug/lane_bev', 10)

        # Local lane data publishers (boundary-based estimation)
        self.local_lane_data_pub = self.create_publisher(
            LocalLaneData, '/mapping/local_lane_data', 10)
        self.left_boundary_pub = self.create_publisher(
            Path, '/mapping/boundary/left', 10)
        self.right_boundary_pub = self.create_publisher(
            Path, '/mapping/boundary/right', 10)
        self.boundary_debug_pub = self.create_publisher(
            Image, '/mapping/boundary_debug_image', 10)

        # Timer to periodically attempt lane detector initialization
        self._init_timer = self.create_timer(0.5, self._try_init_lane_detector)

        self.get_logger().info('Lane Fitting Node initialized!')

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
            cam = self._camera_params
            self.lane_detector = LaneDetector(
                image_width=cam['width'],
                image_height=cam['height'],
            )
            self.lane_detector.set_camera_params(
                fov=cam['fov_deg'],
                pitch=cam['pitch_deg'],
                camera_height=cam['height_m'],
            )
            self._lane_detector_initialized = True
            self._init_timer.cancel()
            self.get_logger().info(
                f'LaneDetector initialized: '
                f'{cam["width"]}x{cam["height"]}, '
                f'FOV={cam["fov_deg"]:.1f}deg, '
                f'pitch={cam["pitch_deg"]:.1f}deg, '
                f'height={cam["height_m"]:.2f}m')
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
        try:
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle', self.camera_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except tf2_ros.LookupException:
            return
        except tf2_ros.ExtrapolationException:
            return

        cam_z = transform.transform.translation.z
        if cam_z is None or cam_z <= 0.0:
            self.get_logger().warn(
                f'Invalid camera height from TF: {cam_z}')
            return

        # Extract pitch from quaternion
        q = transform.transform.rotation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        cam_pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))

        self._camera_params['height_m'] = cam_z
        self._camera_params['pitch_deg'] = cam_pitch
        self._camera_tf_received = True
        self.get_logger().info(
            f'Camera TF received: '
            f'height={cam_z:.2f}m, pitch={cam_pitch:.1f}deg')

    def _camera_info_callback(self, msg: CameraInfo):
        """Extract image dimensions and FOV from the first CameraInfo msg."""
        if self._camera_info_received:
            return

        fx = msg.k[0]
        if fx <= 0:
            self.get_logger().error(
                f'Invalid focal length fx={fx} in CameraInfo. '
                f'Cannot compute FOV.')
            return

        fov_deg = np.degrees(2 * np.arctan(msg.width / (2 * fx)))

        self._camera_params['width'] = msg.width
        self._camera_params['height'] = msg.height
        self._camera_params['fov_deg'] = fov_deg
        self._camera_info_received = True
        self.get_logger().info(
            f'CameraInfo: {msg.width}x{msg.height}, '
            f'FOV={fov_deg:.1f}deg')

    # ------------------------------------------------------------------ #
    #  Semantic Map Callback                                              #
    # ------------------------------------------------------------------ #

    def _semantic_callback(self, msg: Image):
        """Process semantic map and run lane fitting."""
        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        # Store for use in costmap callback (local lane estimation needs semantic for lane fits)
        self._latest_semantic_img = semantic_img
        self._latest_semantic_stamp = msg.header.stamp

        # Only run lane fitting if detector is initialized
        if self._lane_detector_initialized:
            self._run_lane_fitting(semantic_img, msg.header)

    # ------------------------------------------------------------------ #
    #  Costmap Callback (Local Lane Estimation)                           #
    # ------------------------------------------------------------------ #

    def _costmap_callback(self, msg: OccupancyGrid):
        """Process occupancy grid for boundary-based lane estimation."""
        # Parse occupancy grid
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

        # Run local lane estimation (separate from lane fitting)
        self._run_local_lane_estimation(grid, msg.header.stamp)

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
        det_msg.header.frame_id = 'ego_vehicle'
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
        det_msg.lane_confidences = []
        det_msg.lane_pixel_counts = []
        for coeffs in lane_fits:
            det_msg.poly_coeffs_flat.extend(coeffs.tolist())
            det_msg.lane_confidences.append(float(result.confidence))
            det_msg.lane_pixel_counts.append(0)

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
    #  Local Lane Estimation (from occupancy grid boundaries)            #
    # ------------------------------------------------------------------ #

    def _run_local_lane_estimation(self, grid: np.ndarray, stamp):
        """Run boundary-based lane estimation and publish results.

        This is separate from lane fitting - it estimates lanes from
        road boundaries using a priori knowledge about lane structure.
        """
        # Get camera params for lane marking fitting (if available)
        cam = self._camera_params
        camera_fov = cam['fov_deg']
        camera_pitch = cam['pitch_deg']
        camera_height = cam['height_m']

        # Run unified estimator
        road_estimate = self.lane_estimator.estimate(
            grid=grid,
            semantic_img=self._latest_semantic_img,
            camera_fov=camera_fov,
            camera_pitch=camera_pitch,
            camera_height=camera_height,
        )

        # Publish all road structure data
        self._publish_road_estimate(road_estimate, stamp)

    def _publish_road_estimate(self, estimate: LocalRoadEstimate, stamp):
        """Publish all components of the road estimate."""
        # Publish boundaries as Path messages
        if estimate.left_boundary is not None:
            self._publish_boundary_path(estimate.left_boundary, stamp, self.left_boundary_pub)
        if estimate.right_boundary is not None:
            self._publish_boundary_path(estimate.right_boundary, stamp, self.right_boundary_pub)

        # Publish boundary debug image
        if 'boundary_overlay' in estimate.debug_images:
            debug_msg = self.cv_bridge.cv2_to_imgmsg(
                estimate.debug_images['boundary_overlay'], encoding='bgr8')
            debug_msg.header.stamp = stamp
            debug_msg.header.frame_id = self.base_frame
            self.boundary_debug_pub.publish(debug_msg)

        # Publish local lane data
        if estimate.local_lane_data is not None:
            self._publish_local_lane_data(estimate.local_lane_data, stamp)

    def _publish_local_lane_data(self, result, stamp):
        """Publish LocalLaneData message with structured lane data."""
        msg = LocalLaneData()
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
            lane_msg = Lane()
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
            f'Publishing LocalLaneData with {len(msg.lanes)} lanes',
            throttle_duration_sec=5.0)
        self.local_lane_data_pub.publish(msg)

    def _publish_boundary_path(self, points: np.ndarray, stamp, publisher):
        """Publish an Nx2 array of (x, y) base_link points as a Path."""
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.base_frame

        for x, y in points:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        publisher.publish(path_msg)


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
