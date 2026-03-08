#!/usr/bin/env python3
"""
LiDAR-Camera Overlay Node
===========================

Projects AT128 LiDAR points onto the front RGB camera image to visualize
exactly where the LiDAR beams hit in camera view. Points are colored by
distance (near=red, far=blue).

This gives an intuitive understanding of the LiDAR's vertical and horizontal
FOV relative to what the camera sees.

Published on: /visualization/lidar_camera_overlay

Usage:
    ros2 run visualization lidar_camera_overlay_node
"""

import math
import os
import json
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


class LidarCameraOverlayNode(Node):
    """Projects LiDAR points onto camera image for FOV visualization."""

    def __init__(self):
        super().__init__('lidar_camera_overlay_node')

        # ---- Load config from objects.json ----
        cam_cfg, lidar_cfg = self._load_sensor_configs()

        cam_x = cam_cfg['x']
        cam_y = cam_cfg['y']
        cam_z = cam_cfg['z']
        fov = cam_cfg['fov']
        self.img_w = cam_cfg['width']
        self.img_h = cam_cfg['height']

        lidar_x = lidar_cfg['x']
        lidar_y = lidar_cfg['y']
        lidar_z = lidar_cfg['z']
        self.upper_fov = lidar_cfg.get('upper_fov', 12.9)
        self.lower_fov = lidar_cfg.get('lower_fov', -12.5)

        # Visualization
        self.declare_parameter('point_size', 2)
        self.declare_parameter('max_range', float(lidar_cfg.get('range', 200.0)))

        self.point_size = self.get_parameter('point_size').value
        self.max_range = self.get_parameter('max_range').value

        # ---- Compute camera intrinsics ----
        fov_rad = math.radians(fov)
        self.fx = self.img_w / (2.0 * math.tan(fov_rad / 2.0))
        self.fy = self.fx  # square pixels
        self.cx = self.img_w / 2.0
        self.cy = self.img_h / 2.0

        # ---- Compute LiDAR -> Camera translation ----
        # Both sensors have zero rotation, so only translation matters.
        # In CARLA frame: X=forward, Y=right, Z=up
        # The bridge flips Y for ROS, but both sensors have y=0, so no effect.
        # Translation from lidar origin to camera origin (in vehicle frame):
        self.t_cam_lidar = np.array([
            cam_x - lidar_x,   # forward offset
            cam_y - lidar_y,   # lateral offset
            cam_z - lidar_z,   # vertical offset
        ], dtype=np.float64)

        self.get_logger().info(
            f'LiDAR->Camera translation: dx={self.t_cam_lidar[0]:.2f}, '
            f'dy={self.t_cam_lidar[1]:.2f}, dz={self.t_cam_lidar[2]:.2f}'
        )
        self.get_logger().info(
            f'Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, '
            f'cx={self.cx:.1f}, cy={self.cy:.1f}'
        )

        # ---- Build JET colormap LUT ----
        lut_indices = np.arange(256, dtype=np.uint8).reshape(1, -1)
        self.jet_lut = cv2.applyColorMap(lut_indices, cv2.COLORMAP_JET)[0]  # (256, 3)

        self.cv_bridge = CvBridge()

        # ---- Latest data storage ----
        self.latest_image = None
        self.latest_cloud = None

        # ---- QoS ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---- Subscribers ----
        self.image_sub = self.create_subscription(
            Image,
            '/carla/ego_vehicle/sensor/camera/rgb/front/image',
            self.image_callback,
            sensor_qos
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/carla/ego_vehicle/sensor/lidar/at128_front/point_cloud',
            self.cloud_callback,
            sensor_qos
        )

        # ---- Publisher ----
        self.overlay_pub = self.create_publisher(
            Image, '/visualization/lidar_camera_overlay', 10)

        # ---- Performance stats ----
        self._frame_count = 0
        self._total_time = 0.0

        self.get_logger().info('LiDAR-Camera overlay node started')

    def image_callback(self, msg: Image):
        """Store latest camera image and trigger overlay."""
        self.latest_image = msg
        self._publish_overlay()

    def cloud_callback(self, msg: PointCloud2):
        """Store latest point cloud."""
        self.latest_cloud = msg

    def _publish_overlay(self):
        """Project LiDAR points onto image and publish."""
        if self.latest_image is None or self.latest_cloud is None:
            return

        start_time = time.perf_counter()

        # ---- Decode camera image ----
        cv_img = self.cv_bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')

        # ---- Parse point cloud ----
        points = self._parse_pointcloud2(self.latest_cloud)
        if points is None or len(points) == 0:
            # Publish image without overlay
            out_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            out_msg.header = self.latest_image.header
            self.overlay_pub.publish(out_msg)
            return

        # Points are in lidar frame (ROS: X=forward, Y=left, Z=up)
        # Bridge already flipped Y, so Y is left in ROS convention.
        # But for camera projection we need to undo this:
        # Camera projection uses CARLA-like convention where Y=right
        x_lidar = points[:, 0]
        y_lidar = points[:, 1]  # already flipped to left by bridge
        z_lidar = points[:, 2]

        # ---- Transform to camera frame ----
        # Both sensors face forward with no rotation difference,
        # so just apply translation
        x_cam = x_lidar + self.t_cam_lidar[0]
        y_cam = y_lidar + self.t_cam_lidar[1]
        z_cam = z_lidar + self.t_cam_lidar[2]

        # ---- Only keep points in front of camera (x > 0) ----
        front_mask = x_cam > 0.5
        x_cam = x_cam[front_mask]
        y_cam = y_cam[front_mask]
        z_cam = z_cam[front_mask]

        if len(x_cam) == 0:
            out_msg = self.cv_bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            out_msg.header = self.latest_image.header
            self.overlay_pub.publish(out_msg)
            return

        # ---- Project to image plane ----
        # Camera optical frame: Z=forward, X=right, Y=down
        # From ROS vehicle frame: z_opt = x_cam, x_opt = -y_cam, y_opt = -z_cam
        z_opt = x_cam   # forward
        x_opt = -y_cam   # right (ROS Y is left, so negate)
        y_opt = -z_cam   # down  (ROS Z is up, so negate)

        # Pinhole projection
        u = (self.fx * x_opt / z_opt + self.cx).astype(np.int32)
        v = (self.fy * y_opt / z_opt + self.cy).astype(np.int32)

        # ---- Filter to image bounds ----
        valid = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        u = u[valid]
        v = v[valid]
        dist = np.sqrt(x_cam[valid]**2 + y_cam[valid]**2 + z_cam[valid]**2)

        # ---- Color by distance ----
        dist_norm = np.clip(dist / self.max_range, 0.0, 1.0)
        dist_u8 = (dist_norm * 255).astype(np.uint8)
        colors = self.jet_lut[dist_u8]  # (N, 3) BGR

        # ---- Draw points on image (vectorized for 20 Hz) ----
        overlay = cv_img.copy()
        ps = self.point_size
        if ps <= 1:
            # Single-pixel: direct numpy indexing (fastest)
            overlay[v, u] = colors
        else:
            # Small radius: use offsets for a filled square (much faster than cv2.circle loop)
            offsets = np.arange(-ps, ps + 1)
            dy, dx = np.meshgrid(offsets, offsets, indexing='ij')
            mask = dx**2 + dy**2 <= ps**2
            dy = dy[mask]
            dx = dx[mask]
            # Expand all points by offsets
            v_exp = (v[:, None] + dy[None, :]).ravel()
            u_exp = (u[:, None] + dx[None, :]).ravel()
            c_exp = np.repeat(colors, len(dy), axis=0)
            # Clip to bounds
            valid_exp = (v_exp >= 0) & (v_exp < self.img_h) & (u_exp >= 0) & (u_exp < self.img_w)
            overlay[v_exp[valid_exp], u_exp[valid_exp]] = c_exp[valid_exp]

        # ---- Draw FOV boundary lines ----
        for fov_deg, label, color in [(self.upper_fov, f'+{self.upper_fov} upper', (0, 120, 255)),
                                       (self.lower_fov, f'{self.lower_fov} lower', (255, 120, 0))]:
            fov_rad = math.radians(fov_deg)
            bx = 50.0 * math.cos(fov_rad)
            bz = 50.0 * math.sin(fov_rad)
            bx_cam = bx + self.t_cam_lidar[0]
            bz_cam = bz + self.t_cam_lidar[2]
            if bx_cam > 0:
                v_line = int(self.fy * (-bz_cam) / bx_cam + self.cy)
                if 0 <= v_line < self.img_h:
                    cv2.line(overlay, (0, v_line), (self.img_w, v_line), color, 1)
                    cv2.putText(overlay, label, (self.img_w - 160, v_line - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        n_pts = len(u)
        cv2.putText(overlay, f'AT128 | {n_pts} pts',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---- Publish ----
        out_msg = self.cv_bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        out_msg.header = self.latest_image.header
        self.overlay_pub.publish(out_msg)
        self._update_stats(start_time)

    def _update_stats(self, start_time: float):
        """Log average FPS every 30 frames."""
        elapsed = (time.perf_counter() - start_time) * 1000
        self._frame_count += 1
        self._total_time += elapsed

        if self._frame_count % 30 == 0:
            avg_time = self._total_time / self._frame_count
            fps = 1000.0 / avg_time if avg_time > 0 else 0
            self.get_logger().info(
                f'[Overlay] {elapsed:.1f}ms (avg: {avg_time:.1f}ms, ~{fps:.1f} FPS)'
            )

    def _load_sensor_configs(self):
        """Load camera and AT128 lidar configs from objects.json."""
        pkg_share = get_package_share_directory('autoiq_bridge')
        objects_path = os.path.join(pkg_share, 'config', 'objects.json')

        with open(objects_path, 'r') as f:
            data = json.load(f)

        cam_cfg = None
        lidar_cfg = None

        for obj in data['objects']:
            if obj['id'] != 'ego_vehicle':
                continue
            for sensor in obj['sensors']:
                sp = sensor['spawn_point']
                if sensor['type'] == 'sensor.camera.rgb' and sensor['id'] == 'front':
                    cam_cfg = {
                        'x': float(sp['x']),
                        'y': float(sp['y']),
                        'z': float(sp['z']),
                        'fov': float(sensor['fov']),
                        'width': int(sensor['image_size_x']),
                        'height': int(sensor['image_size_y']),
                    }
                elif sensor['id'] == 'at128_front':
                    lidar_cfg = {
                        'x': float(sp['x']),
                        'y': float(sp['y']),
                        'z': float(sp['z']),
                        'range': float(sensor['range']),
                        'upper_fov': float(sensor['upper_fov']),
                        'lower_fov': float(sensor['lower_fov']),
                        'horizontal_fov': float(sensor['horizontal_fov']),
                    }

        assert cam_cfg is not None, 'Front RGB camera (id=front) not found in objects.json'
        assert lidar_cfg is not None, 'AT128 lidar (id=at128_front) not found in objects.json'

        self.get_logger().info(f'Loaded sensor configs from {objects_path}')
        return cam_cfg, lidar_cfg

    def _parse_pointcloud2(self, msg: PointCloud2) -> np.ndarray:
        """Parse PointCloud2 into Nx4 float32 array (x, y, z, intensity)."""
        field_map = {f.name: f.offset for f in msg.fields}
        if 'x' not in field_map:
            return None

        num_points = msg.width * msg.height
        if num_points == 0:
            return None

        point_step = msg.point_step
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(num_points, point_step)

        points = np.zeros((num_points, 4), dtype=np.float32)
        for col, name in enumerate(['x', 'y', 'z', 'intensity']):
            if name in field_map:
                off = field_map[name]
                points[:, col] = raw[:, off:off+4].view(np.float32).flatten()

        return points


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
