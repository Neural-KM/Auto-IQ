#!/usr/bin/env python3
"""
Lane Detection Node
====================

ROS2 node that performs polynomial lane fitting on the semantic segmentation
output from the semantic_segmentation_node.

Subscribes to the processed semantic map (which already includes lane markings
via ground truth or RoadExtractor-injected road_line class), applies a bird's
eye view transform, runs a sliding-window search, and fits 2nd-order
polynomials to the left and right lane boundaries.

Publishes:
- /perception/lane_detection  (LaneDetection msg)  — polynomial fits + metrics
- /perception/lanes           (MarkerArray)          — RViz lane line visualization
- /perception/debug/lane_bev  (Image, bgr8)          — BEV debug visualization

Subscribes:
- /perception/semantic_map/road_lane_extraction  (mono8) — semantic map with lane markings
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info  — camera intrinsics

Parameters:
- semantic_topic: Input semantic map topic
- camera_info_topic: Camera info for intrinsics
- camera_height: Camera height above ground in meters (required)
- camera_pitch: Camera pitch angle in degrees (required)
- enable_debug_viz: Publish debug BEV images
"""

import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time

from autoiq_msgs.msg import LaneDetection as LaneDetectionMsg

from cv_bridge import CvBridge

from perception.core.lane_detector import LaneDetector, CONFIG


class LaneDetectionNode(Node):
    """
    ROS2 node for lane detection via polynomial fitting on semantic segmentation.

    Initialization is gated on receiving CameraInfo so that the
    LaneDetector BEV transform is configured with correct image dimensions
    and FOV. Camera height and pitch must be provided as parameters.
    """

    def __init__(self):
        super().__init__('lane_detection_node')

        # ==================== Parameters ====================
        self.declare_parameter(
            'semantic_topic',
            '/perception/semantic_map/road_lane_extraction')
        self.declare_parameter(
            'camera_info_topic',
            '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info')
        self.declare_parameter('camera_height')
        self.declare_parameter('camera_pitch')
        self.declare_parameter('enable_debug_viz', False)

        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value

        # Camera extrinsics — must be provided via launch file or command line.
        self.camera_height = self.get_parameter('camera_height').value
        self.camera_pitch = self.get_parameter('camera_pitch').value
        if self.camera_height is None or self.camera_pitch is None:
            raise RuntimeError(
                'camera_height and camera_pitch parameters are required. '
                'Provide them via launch file or ros2 run --ros-args -p camera_height:=<val> -p camera_pitch:=<val>')

        # ==================== Initialization Gate ====================
        self.initialized = False
        self.camera_info_received = False

        # Camera intrinsics (filled from CameraInfo)
        self.img_width = None
        self.img_height = None
        self.fov_deg = None

        # ==================== Performance Stats ====================
        self._frame_count = 0
        self._total_time = 0.0

        # ==================== CV Bridge ====================
        self.cv_bridge = CvBridge()

        # ==================== Lane Detector (created after init) ====================
        self.lane_detector: LaneDetector = None

        # ==================== QoS ====================
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ==================== Subscribers ====================
        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic,
            self.camera_info_callback, sensor_qos)

        self.semantic_sub = self.create_subscription(
            Image, semantic_topic,
            self.semantic_callback, sensor_qos)

        self.get_logger().info(f'Subscribed to semantic: {semantic_topic}')
        self.get_logger().info(f'Subscribed to camera_info: {camera_info_topic}')

        # ==================== Publishers ====================
        self.lane_detection_pub = self.create_publisher(
            LaneDetectionMsg, '/perception/lane_detection', 10)

        self.lane_markers_pub = self.create_publisher(
            MarkerArray, '/perception/lanes', 10)

        if self.enable_debug_viz:
            self.debug_bev_pub = self.create_publisher(
                Image, '/perception/debug/lane_bev', 10)

        # ==================== Init Timer ====================
        self.init_timer = self.create_timer(0.5, self._try_initialize)

        self.get_logger().info(
            f'Lane Detection Node created — waiting for CameraInfo... '
            f'(camera_height={self.camera_height}m, camera_pitch={self.camera_pitch}°)')

    # ------------------------------------------------------------------ #
    #  Initialization                                                     #
    # ------------------------------------------------------------------ #

    def _try_initialize(self):
        if self.initialized:
            return

        if self.camera_info_received:
            self._complete_initialization()
        else:
            self.get_logger().info(
                'Waiting for CameraInfo...',
                throttle_duration_sec=3.0)

    def _complete_initialization(self):
        self.lane_detector = LaneDetector(
            image_width=self.img_width,
            image_height=self.img_height,
        )
        self.lane_detector.set_camera_params(
            fov=self.fov_deg,
            pitch=self.camera_pitch,
            camera_height=self.camera_height,
        )
        self.init_timer.cancel()
        self.initialized = True
        self.get_logger().info(
            f'Lane Detection Node initialized: '
            f'{self.img_width}x{self.img_height}, '
            f'FOV={self.fov_deg:.1f}°, '
            f'pitch={self.camera_pitch:.1f}°, '
            f'height={self.camera_height:.2f}m')

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        self.img_width = msg.width
        self.img_height = msg.height
        fx = msg.k[0]
        if fx <= 0:
            self.get_logger().error(
                f'Invalid focal length fx={fx} in CameraInfo. '
                f'Cannot compute FOV — check camera calibration.')
            return

        self.fov_deg = np.degrees(2 * np.arctan(self.img_width / (2 * fx)))
        self.camera_info_received = True
        self.get_logger().info(
            f'CameraInfo: {self.img_width}x{self.img_height}, '
            f'FOV={self.fov_deg:.1f}°')

    def semantic_callback(self, msg: Image):
        if not self.initialized:
            return

        start_time = time.perf_counter()

        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        # Run lane detection (BEV + sliding window + polynomial fit)
        result = self.lane_detector.detect(semantic_img)

        # Publish LaneDetection message
        self._publish_lane_detection(result, msg.header)

        # Publish RViz markers
        self._publish_lane_markers(result)

        # Publish debug BEV
        if self.enable_debug_viz:
            self._publish_debug_bev(msg.header)

        self._update_stats(start_time)

    # ------------------------------------------------------------------ #
    #  Performance Stats                                                  #
    # ------------------------------------------------------------------ #

    def _update_stats(self, start_time: float):
        """Log average FPS every 30 frames."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self._frame_count += 1
        self._total_time += elapsed

        if self._frame_count % 30 == 0:
            avg_time = self._total_time / self._frame_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'[LaneDetection] {elapsed:.1f}ms '
                f'(avg: {avg_time:.1f}ms, ~{fps:.1f} FPS)')

    # ------------------------------------------------------------------ #
    #  Publishers                                                         #
    # ------------------------------------------------------------------ #

    def _publish_lane_detection(self, result, header):
        """Publish a LaneDetection message with polynomial coefficients."""
        det_msg = LaneDetectionMsg()
        # Use the original timestamp but explicitly set the frame to ego_vehicle,
        # since all lane polynomials and BEV parameters are expressed in that frame.
        det_msg.header.stamp = header.stamp
        det_msg.header.frame_id = 'ego_vehicle'
        det_msg.valid = result.valid
        det_msg.confidence = float(result.confidence)

        # Polynomial coefficients
        if result.left_fit is not None:
            det_msg.left_poly_coeffs = result.left_fit.tolist()
        if result.right_fit is not None:
            det_msg.right_poly_coeffs = result.right_fit.tolist()

        # Center polynomial (average of left and right)
        if result.left_fit is not None and result.right_fit is not None:
            center_fit = (result.left_fit + result.right_fit) / 2.0
            det_msg.center_poly_coeffs = center_fit.tolist()

        # Image-space lane positions
        det_msg.lane_center = result.lane_center
        det_msg.left_lane_x = result.left_lane if result.left_lane is not None else -1
        det_msg.right_lane_x = result.right_lane if result.right_lane is not None else -1
        det_msg.lane_width_px = result.lane_width if result.lane_width is not None else -1

        # BEV parameters (always present after detect())
        bev = result.bev_params
        det_msg.bev_width = bev['width']
        det_msg.bev_height = bev['height']
        lateral_range = bev['lateral_range']
        forward_range = bev['forward_range']
        det_msg.bev_x_min = float(lateral_range[0])
        det_msg.bev_x_max = float(lateral_range[1])
        det_msg.bev_y_min = float(forward_range[0])
        det_msg.bev_y_max = float(forward_range[1])

        # Real-world lane width
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

    def _make_lane_marker(self, fit, bev_params, marker_id, color, stamp):
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
            # BEV row 0 = far, row bh-1 = near
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
            # No debug BEV image available (e.g., before first frame or on lane-fit failure)
            return

        out = self.cv_bridge.cv2_to_imgmsg(lane_fit_img, encoding='bgr8')
        out.header = header
        self.debug_bev_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
