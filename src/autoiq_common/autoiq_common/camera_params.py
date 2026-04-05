"""
Camera Parameters Provider
==========================

Unified camera parameter acquisition from CameraInfo + TF.

Eliminates boilerplate code duplicated across nodes that need camera
intrinsics (from CameraInfo) and extrinsics (from TF).

Usage:
    from autoiq_common.camera_params import CameraParams, CameraParamsProvider

    class MyNode(Node):
        def __init__(self):
            super().__init__('my_node')

            self.camera = CameraParamsProvider(
                node=self,
                camera_info_topic='/camera/camera_info',
                base_frame='ego_vehicle',
                camera_frame='ego_vehicle/sensor/camera/rgb/front',
                on_ready=self._on_camera_ready,
            )

        def _on_camera_ready(self, params: CameraParams):
            self.get_logger().info(f'Camera ready: {params.width}x{params.height}')
            # Initialize components that need camera params

        def process_image(self, img):
            if not self.camera.ready:
                return
            params = self.camera.params
            # Use params.fov_deg, params.height_m, etc.
"""

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import tf2_ros

from sensor_msgs.msg import CameraInfo


@dataclass
class CameraParams:
    """Camera intrinsics and extrinsics."""
    # Image dimensions
    width: int
    height: int

    # Intrinsics
    fx: float
    fy: float
    cx: float
    cy: float

    # Derived from intrinsics
    fov_deg: float

    # Extrinsics (from TF)
    x_m: float           # Camera X position relative to base frame
    y_m: float           # Camera Y position relative to base frame
    height_m: float      # Camera height above ground (Z)
    pitch_deg: float     # Camera pitch angle (positive = looking down)


class CameraParamsProvider:
    """
    Unified camera parameter acquisition from CameraInfo + TF.

    Subscribes to CameraInfo for intrinsics and looks up TF for extrinsics.
    Calls on_ready callback when both are available.

    Args:
        node: The ROS2 node (for subscriptions, TF, logging)
        camera_info_topic: Topic for CameraInfo messages
        base_frame: Base/vehicle frame name
        camera_frame: Camera TF frame name
        on_ready: Callback when params become available (called once)
        qos: QoS profile for CameraInfo subscription (default: sensor QoS)
    """

    def __init__(
        self,
        node: Node,
        camera_info_topic: str,
        base_frame: str,
        camera_frame: str,
        on_ready: Optional[Callable[[CameraParams], None]] = None,
        qos: Optional[QoSProfile] = None,
    ):
        self._node = node
        self._base_frame = base_frame
        self._camera_frame = camera_frame
        self._on_ready = on_ready

        # State
        self._params: Optional[CameraParams] = None
        self._intrinsics_received = False
        self._extrinsics_received = False
        self._ready_called = False

        # Partial data (before complete)
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._fx: Optional[float] = None
        self._fy: Optional[float] = None
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None
        self._fov_deg: Optional[float] = None
        self._x_m: Optional[float] = None
        self._y_m: Optional[float] = None
        self._height_m: Optional[float] = None
        self._pitch_deg: Optional[float] = None

        # TF
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        # QoS
        if qos is None:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

        # Subscribe to CameraInfo
        self._camera_info_sub = node.create_subscription(
            CameraInfo, camera_info_topic,
            self._camera_info_callback, qos)

        node.get_logger().info(f'CameraParamsProvider: waiting for {camera_info_topic} + TF')

    @property
    def ready(self) -> bool:
        """True when both intrinsics and extrinsics are available."""
        return self._params is not None

    @property
    def params(self) -> Optional[CameraParams]:
        """Camera parameters (None if not ready)."""
        return self._params

    def try_get_extrinsics(self) -> bool:
        """
        Attempt to look up camera extrinsics from TF.

        Returns True if successful, False if TF not available yet.
        Call this periodically or before processing if extrinsics not received.
        """
        if self._extrinsics_received:
            return True

        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return False

        # Extract position
        cam_x = transform.transform.translation.x
        cam_y = transform.transform.translation.y
        cam_z = transform.transform.translation.z
        if cam_z is None or cam_z <= 0.0:
            self._node.get_logger().warn(
                f'CameraParamsProvider: invalid camera height from TF: {cam_z}')
            return False

        # Extract pitch from quaternion (rotation around X axis)
        q = transform.transform.rotation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        cam_pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))

        self._x_m = cam_x
        self._y_m = cam_y
        self._height_m = cam_z
        self._pitch_deg = cam_pitch
        self._extrinsics_received = True

        self._node.get_logger().info(
            f'CameraParamsProvider: TF received - x={cam_x:.2f}m, y={cam_y:.2f}m, '
            f'height={cam_z:.2f}m, pitch={cam_pitch:.1f}deg')

        self._try_finalize()
        return True

    def _camera_info_callback(self, msg: CameraInfo):
        """Handle CameraInfo message (called once, then ignored)."""
        if self._intrinsics_received:
            return

        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        if fx <= 0 or fy <= 0:
            self._node.get_logger().error(
                f'CameraParamsProvider: invalid focal length fx={fx}, fy={fy}')
            return

        # Compute horizontal FOV from focal length and image width
        fov_deg = np.degrees(2 * np.arctan(msg.width / (2 * fx)))

        self._width = msg.width
        self._height = msg.height
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._fov_deg = fov_deg
        self._intrinsics_received = True

        self._node.get_logger().info(
            f'CameraParamsProvider: CameraInfo received - '
            f'{msg.width}x{msg.height}, FOV={fov_deg:.1f}deg')

        # Try to get TF now that we have intrinsics
        self.try_get_extrinsics()
        self._try_finalize()

    def _try_finalize(self):
        """Finalize params and call on_ready if both intrinsics and extrinsics available."""
        if self._ready_called:
            return

        if not self._intrinsics_received or not self._extrinsics_received:
            return

        self._params = CameraParams(
            width=self._width,
            height=self._height,
            fx=self._fx,
            fy=self._fy,
            cx=self._cx,
            cy=self._cy,
            fov_deg=self._fov_deg,
            x_m=self._x_m,
            y_m=self._y_m,
            height_m=self._height_m,
            pitch_deg=self._pitch_deg,
        )

        self._ready_called = True

        if self._on_ready is not None:
            self._on_ready(self._params)
