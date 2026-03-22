#!/usr/bin/env python3
"""
Lane Fitting Node
=================

ROS2 node that performs polynomial lane fitting on semantic segmentation maps.
Subscribes to semantic maps from the perception package and publishes lane
detection results (LaneDetection msg + RViz markers).

This node is responsible for fitting polynomial curves to lane markings,
while the perception package handles pixel-level semantic labeling.

Parameters:
- semantic_topic: Input semantic map topic (default: /perception/semantic_map/road_lane_extraction)
- camera_info_topic: Camera info for intrinsics
- camera_frame: TF frame name for the camera sensor
- enable_debug_viz: Publish debug BEV images

Subscribes:
- /perception/semantic_map/road_lane_extraction (mono8) - semantic segmentation with lane markings
- /carla/ego_vehicle/sensor/camera/rgb/front/camera_info (CameraInfo) - camera intrinsics

Publishes:
- /mapping/lane_detection (LaneDetection) - polynomial fits + metrics
- /mapping/lanes (MarkerArray) - RViz lane line visualization
- /mapping/debug/lane_bev (Image, bgr8) - BEV debug visualization
"""

import numpy as np
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

from mapping.core.lane_fitting import LaneDetector

from cv_bridge import CvBridge


class LaneFittingNode(Node):
    """ROS2 Node for lane polynomial fitting on semantic segmentation maps."""

    def __init__(self):
        super().__init__('lane_fitting_node')

        # ==================== Parameters ====================
        self.declare_parameter(
            'semantic_topic',
            '/perception/semantic_map/road_lane_extraction')
        self.declare_parameter(
            'camera_info_topic',
            '/carla/ego_vehicle/sensor/camera/rgb/front/camera_info')
        self.declare_parameter(
            'camera_frame',
            'ego_vehicle/sensor/camera/rgb/front')
        self.declare_parameter('enable_debug_viz', False)

        semantic_topic = self.get_parameter('semantic_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.enable_debug_viz = self.get_parameter('enable_debug_viz').value

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
        self.semantic_sub = self.create_subscription(
            Image, semantic_topic, self._semantic_callback, sensor_qos)

        self.camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic,
            self._camera_info_callback, sensor_qos)
        self.get_logger().info(f'Subscribed to CameraInfo: {camera_info_topic}')

        # ==================== Publishers ====================
        self.lane_detection_pub = self.create_publisher(
            LaneDetectionMsg, '/mapping/lane_detection', 10)
        self.lane_markers_pub = self.create_publisher(
            MarkerArray, '/mapping/lanes', 10)
        if self.enable_debug_viz:
            self.debug_bev_pub = self.create_publisher(
                Image, '/mapping/debug/lane_bev', 10)

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
                f'FOV={self._cam_fov_deg:.1f}deg, '
                f'pitch={self._cam_pitch:.1f}deg, '
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
        try:
            transform = self.tf_buffer.lookup_transform(
                'ego_vehicle', self.camera_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except tf2_ros.LookupException:
            return
        except tf2_ros.ExtrapolationException:
            return

        self._cam_z = transform.transform.translation.z
        if self._cam_z is None or self._cam_z <= 0.0:
            self.get_logger().warn(
                f'Invalid camera height from TF: {self._cam_z}')
            return

        # Extract pitch from quaternion
        q = transform.transform.rotation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self._cam_pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))

        self._camera_tf_received = True
        self.get_logger().info(
            f'Camera TF received: '
            f'height={self._cam_z:.2f}m, pitch={self._cam_pitch:.1f}deg')

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
                f'Cannot compute FOV.')
            return

        self._cam_fov_deg = np.degrees(2 * np.arctan(self._cam_width / (2 * fx)))
        self._camera_info_received = True
        self.get_logger().info(
            f'CameraInfo: {self._cam_width}x{self._cam_height}, '
            f'FOV={self._cam_fov_deg:.1f}deg')

    # ------------------------------------------------------------------ #
    #  Semantic Map Callback                                              #
    # ------------------------------------------------------------------ #

    def _semantic_callback(self, msg: Image):
        """Process semantic map and run lane fitting."""
        if not self._lane_detector_initialized:
            return

        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}')
            return

        self._run_lane_fitting(semantic_img, msg.header)

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
