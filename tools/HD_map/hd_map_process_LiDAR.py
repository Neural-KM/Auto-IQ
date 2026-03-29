#!/usr/bin/env python3
"""
HD Map Processor (LiDAR) - Offline Rosbag Processing
====================================================

Processes recorded rosbags to generate HD maps using LiDAR-based 2D-to-3D projection.

Pipeline Steps:
    A. Extract trajectory from odometry (IMPLEMENTED)
    B. Extract semantic masks (road/lanes) from ground truth camera (IMPLEMENTED)
    C. Project 2D lane pixels to 3D using hybrid LiDAR + ground plane (IMPLEMENTED)
       - Primary: LiDAR depth lookup (projects LiDAR → camera, finds nearest depth)
       - Fallback: Local ground plane fit (fits plane to nearby LiDAR ground points)
    D. Transform to global frame using trajectory (IMPLEMENTED)
    E. Cluster and fit splines (IMPLEMENTED)
       - DBSCAN clustering to separate individual lane markings
       - B-spline fitting for smooth lane geometry
       - Exports lane centerlines as JSON

Usage:
    # Full processing (Steps A-E)
    python tools/hd_map_process_LiDAR.py --bag Rosbags/hdmap_2026_03_21-05_52_15

    # With visualizations
    python tools/hd_map_process_LiDAR.py --bag Rosbags/hdmap_2026_03_21-05_52_15 --visualize

    # Process every 5th frame (faster)
    python tools/hd_map_process_LiDAR.py --bag Rosbags/hdmap_2026_03_21-05_52_15 --visualize --sample-rate 5

Output:
    Creates <bag_name>_hdmap/ directory with:
    - trajectory.json       : Vehicle poses over time
    - hd_map_lanes.json     : Fitted lane splines (Step E output)
    - trajectory.png        : Trajectory visualization (if --visualize)
    - semantic_summary.png  : Lane/road statistics (if --visualize)
    - semantic_frame_*.png  : Sample frame visualizations (if --visualize)
    - lane_points_3d.png    : 3D lane points BEV in vehicle frame (if --visualize)
    - global_lane_map.png   : Global lane map with trajectory (if --visualize)
    - lane_splines.png      : Fitted lane splines visualization (if --visualize)
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Pose:
    """Vehicle pose at a specific timestamp."""
    timestamp_ns: int  # nanoseconds
    x: float
    y: float
    z: float
    roll: float   # radians
    pitch: float  # radians
    yaw: float    # radians
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9

    @property 
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix from quaternion."""
        qx, qy, qz, qw = self.qx, self.qy, self.qz, self.qw
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

    def transform_point(self, local_point: np.ndarray) -> np.ndarray:
        """Transform a point from local (vehicle) frame to global frame."""
        return self.rotation_matrix @ local_point + self.position


@dataclass
class Trajectory:
    """Collection of poses representing the vehicle trajectory."""
    poses: List[Pose] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.poses)

    def add_pose(self, pose: Pose):
        self.poses.append(pose)

    def get_pose_at_time(self, timestamp_ns: int, tolerance_ns: int = 50_000_000) -> Optional[Pose]:
        """Get the closest pose to a given timestamp (within tolerance)."""
        if not self.poses:
            return None
        timestamps = np.array([p.timestamp_ns for p in self.poses])
        idx = np.searchsorted(timestamps, timestamp_ns)
        candidates = []
        if idx > 0:
            candidates.append((abs(timestamps[idx-1] - timestamp_ns), idx-1))
        if idx < len(timestamps):
            candidates.append((abs(timestamps[idx] - timestamp_ns), idx))
        if not candidates:
            return None
        best_diff, best_idx = min(candidates)
        return self.poses[best_idx] if best_diff <= tolerance_ns else None

    def total_distance(self) -> float:
        if len(self.poses) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.poses)):
            total += np.linalg.norm(self.poses[i].position - self.poses[i-1].position)
        return total

    def duration_sec(self) -> float:
        if len(self.poses) < 2:
            return 0.0
        return (self.poses[-1].timestamp_ns - self.poses[0].timestamp_ns) / 1e9

    def to_numpy(self) -> Dict[str, np.ndarray]:
        if not self.poses:
            return {}
        return {
            'timestamps': np.array([p.timestamp_ns for p in self.poses]),
            'x': np.array([p.x for p in self.poses]),
            'y': np.array([p.y for p in self.poses]),
            'z': np.array([p.z for p in self.poses]),
            'yaw': np.array([p.yaw for p in self.poses]),
        }

    def to_json(self) -> List[dict]:
        return [
            {'timestamp_ns': p.timestamp_ns, 'x': p.x, 'y': p.y, 'z': p.z,
             'roll': p.roll, 'pitch': p.pitch, 'yaw': p.yaw,
             'qx': p.qx, 'qy': p.qy, 'qz': p.qz, 'qw': p.qw}
            for p in self.poses
        ]


@dataclass
class SemanticFrame:
    """Semantic segmentation data for a single frame."""
    timestamp_ns: int
    road_mask: np.ndarray          # Binary mask of road pixels (H x W)
    lane_pixels: np.ndarray        # Nx2 array of (row, col) lane marking pixel coordinates

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9

    @property
    def num_lane_pixels(self) -> int:
        return len(self.lane_pixels) if self.lane_pixels is not None else 0

    @property
    def road_coverage(self) -> float:
        """Percentage of image that is road."""
        if self.road_mask is None:
            return 0.0
        return np.sum(self.road_mask) / self.road_mask.size * 100


@dataclass
class LanePoints3D:
    """3D lane marking points for a single frame (in camera/vehicle frame)."""
    timestamp_ns: int
    points: np.ndarray  # Nx3 array of (x, y, z) in vehicle frame (x=forward, y=left, z=up)

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9

    @property
    def num_points(self) -> int:
        return len(self.points) if self.points is not None else 0


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int) -> 'CameraIntrinsics':
        """Create intrinsics from horizontal FOV."""
        fov_rad = np.radians(fov_deg)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx  # Square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


@dataclass
class LidarFrame:
    """LiDAR point cloud for a single frame."""
    timestamp_ns: int
    points: np.ndarray  # Nx4 array of (x, y, z, intensity) in LiDAR frame

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9

    @property
    def num_points(self) -> int:
        return len(self.points) if self.points is not None else 0

    def get_xyz(self) -> np.ndarray:
        """Return Nx3 array of (x, y, z)."""
        return self.points[:, :3] if self.points is not None else np.empty((0, 3))


@dataclass
class GlobalLaneMap:
    """Accumulated lane marking points in global frame."""
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    # Nx3 array of (x, y, z) in global/map frame

    def add_points(self, new_points: np.ndarray):
        """Add new points to the global map."""
        if new_points is not None and len(new_points) > 0:
            self.points = np.vstack([self.points, new_points.astype(np.float32)])

    @property
    def num_points(self) -> int:
        return len(self.points)

    def get_bounds(self) -> dict:
        """Get bounding box of all points."""
        if self.num_points == 0:
            return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
        return {
            'x_min': float(np.min(self.points[:, 0])),
            'x_max': float(np.max(self.points[:, 0])),
            'y_min': float(np.min(self.points[:, 1])),
            'y_max': float(np.max(self.points[:, 1])),
        }


@dataclass
class LaneSpline:
    """Fitted spline representing a single lane marking."""
    lane_id: int
    points: np.ndarray           # Nx3 raw clustered points
    spline_points: np.ndarray    # Mx3 interpolated spline points
    length: float                # Total arc length of the spline (meters)
    num_raw_points: int          # Number of raw points in cluster

    def to_json(self) -> dict:
        """Export lane spline to JSON-serializable dict."""
        return {
            'lane_id': self.lane_id,
            'length_m': self.length,
            'num_raw_points': self.num_raw_points,
            'spline_points': self.spline_points.tolist()
        }


@dataclass
class HDMapLanes:
    """Collection of fitted lane splines representing the HD map."""
    lanes: List[LaneSpline] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.lanes)

    def add_lane(self, lane: LaneSpline):
        self.lanes.append(lane)

    def total_length(self) -> float:
        """Total length of all lanes in meters."""
        return sum(lane.length for lane in self.lanes)

    def get_bounds(self) -> dict:
        """Get bounding box of all lane splines."""
        if not self.lanes:
            return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
        all_pts = np.vstack([lane.spline_points for lane in self.lanes])
        return {
            'x_min': float(np.min(all_pts[:, 0])),
            'x_max': float(np.max(all_pts[:, 0])),
            'y_min': float(np.min(all_pts[:, 1])),
            'y_max': float(np.max(all_pts[:, 1])),
        }

    def to_json(self) -> List[dict]:
        return [lane.to_json() for lane in self.lanes]


# =============================================================================
# Utility Functions
# =============================================================================

def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# =============================================================================
# HD Map Processor
# =============================================================================

class HDMapProcessor:
    """
    Processes recorded rosbags to generate HD maps.

    Steps:
        A. Extract trajectory from odometry
        B. Extract semantic masks (road/lanes) from ground truth camera
        C. Project 2D lane pixels to 3D using IPM (ground plane assumption)
        D. Transform to global frame using trajectory
        E. Cluster and fit splines
    """

    ODOMETRY_TOPIC = '/carla/ego_vehicle/ground_truth/odometry'
    GNSS_TOPIC = '/carla/ego_vehicle/sensor/gnss/gnss/fix'
    SEMANTIC_TOPIC = '/carla/ego_vehicle/ground_truth/camera/semantic/front/image'  # mono8 encoding
    LIDAR_TOPIC = '/carla/ego_vehicle/sensor/lidar/at128_front/point_cloud'

    CLASS_ROAD = 1
    CLASS_LANE_MARKING = 24

    # Camera parameters from objects.json (semantic camera at same position as RGB front)
    CAMERA_X = 0.2   # Forward offset from vehicle origin (m)
    CAMERA_Y = 0.0   # Lateral offset (m)
    CAMERA_Z = 1.5   # Height above ground (m)
    CAMERA_FOV = 90.0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    # AT128 LiDAR parameters from objects.json
    LIDAR_X = 2.0    # Forward offset from vehicle origin (m)
    LIDAR_Y = 0.0    # Lateral offset (m)
    LIDAR_Z = 1.0    # Height above ground (m)

    def __init__(self, bag_path: str):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {bag_path}")
        self.trajectory = Trajectory()
        self.gnss_data: List[dict] = []
        self.semantic_frames: List[SemanticFrame] = []  # Step B data
        self.lidar_frames: List[LidarFrame] = []        # LiDAR data for Step C
        self.lane_points_3d: List[LanePoints3D] = []    # Step C data
        self.global_lane_map: GlobalLaneMap = GlobalLaneMap()  # Step D data
        self.hd_map_lanes: HDMapLanes = HDMapLanes()    # Step E data
        self._msg_type_cache: Dict[str, type] = {}

        # Initialize camera intrinsics
        self.camera_intrinsics = CameraIntrinsics.from_fov(
            self.CAMERA_FOV, self.CAMERA_WIDTH, self.CAMERA_HEIGHT
        )

        # LiDAR to camera translation (both sensors face forward, no rotation)
        self.t_lidar_to_cam = np.array([
            self.CAMERA_X - self.LIDAR_X,  # forward offset
            self.CAMERA_Y - self.LIDAR_Y,  # lateral offset
            self.CAMERA_Z - self.LIDAR_Z,  # vertical offset
        ], dtype=np.float32)

    def _get_message_type(self, type_name: str):
        """Get message class from type name string."""
        if type_name not in self._msg_type_cache:
            from rosidl_runtime_py.utilities import get_message
            self._msg_type_cache[type_name] = get_message(type_name)
        return self._msg_type_cache[type_name]

    def _create_bag_reader(self):
        """Create a rosbag reader."""
        import rosbag2_py
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(uri=str(self.bag_path), storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
        return reader

    def extract_trajectory(self) -> Trajectory:
        """
        Step A: Extract odometry/trajectory from the rosbag.

        Reads ground truth odometry and builds a time-indexed trajectory.
        """
        import rosbag2_py
        from rclpy.serialization import deserialize_message

        print(f"\n{'='*60}")
        print(" Step A: Extracting Trajectory from Odometry")
        print(f"{'='*60}")

        reader = self._create_bag_reader()
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        if self.ODOMETRY_TOPIC not in topic_types:
            raise ValueError(f"Odometry topic not found: {self.ODOMETRY_TOPIC}")

        odom_type = self._get_message_type(topic_types[self.ODOMETRY_TOPIC])

        storage_filter = rosbag2_py.StorageFilter(topics=[self.ODOMETRY_TOPIC])
        reader.set_filter(storage_filter)

        pose_count = 0
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if topic == self.ODOMETRY_TOPIC:
                msg = deserialize_message(data, odom_type)
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                z = msg.pose.pose.position.z
                qx = msg.pose.pose.orientation.x
                qy = msg.pose.pose.orientation.y
                qz = msg.pose.pose.orientation.z
                qw = msg.pose.pose.orientation.w
                roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

                self.trajectory.add_pose(Pose(
                    timestamp_ns=timestamp_ns, x=x, y=y, z=z,
                    roll=roll, pitch=pitch, yaw=yaw,
                    qx=qx, qy=qy, qz=qz, qw=qw
                ))
                pose_count += 1

        print(f"\n  Extracted {pose_count} poses")
        print(f"  Duration: {self.trajectory.duration_sec():.1f} seconds")
        print(f"  Distance: {self.trajectory.total_distance():.1f} meters")

        if self.trajectory.poses:
            first, last = self.trajectory.poses[0], self.trajectory.poses[-1]
            print(f"\n  Start: ({first.x:.1f}, {first.y:.1f}, {first.z:.1f})")
            print(f"  End:   ({last.x:.1f}, {last.y:.1f}, {last.z:.1f})")
            print(f"  Avg frequency: {pose_count / max(self.trajectory.duration_sec(), 0.1):.1f} Hz")

        return self.trajectory

    def extract_gnss(self) -> List[dict]:
        """Extract GNSS data from the rosbag."""
        import rosbag2_py
        from rclpy.serialization import deserialize_message

        print(f"\n{'='*60}")
        print(" Extracting GNSS Data")
        print(f"{'='*60}")

        reader = self._create_bag_reader()
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        if self.GNSS_TOPIC not in topic_types:
            print("  GNSS topic not found, skipping...")
            return []

        gnss_type = self._get_message_type(topic_types[self.GNSS_TOPIC])
        storage_filter = rosbag2_py.StorageFilter(topics=[self.GNSS_TOPIC])
        reader.set_filter(storage_filter)

        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            if topic == self.GNSS_TOPIC:
                msg = deserialize_message(data, gnss_type)
                self.gnss_data.append({
                    'timestamp_ns': timestamp_ns,
                    'latitude': msg.latitude,
                    'longitude': msg.longitude,
                    'altitude': msg.altitude
                })

        print(f"  Extracted {len(self.gnss_data)} GNSS readings")
        if self.gnss_data:
            first = self.gnss_data[0]
            print(f"  First: ({first['latitude']:.6f}, {first['longitude']:.6f})")

        return self.gnss_data

    def extract_semantic_data(self, sample_rate: int = 1) -> List[SemanticFrame]:
        """
        Step B: Extract semantic masks from ground truth camera.

        Reads semantic segmentation images and extracts:
        - Road mask (class ID = 1)
        - Lane marking pixel coordinates (class ID = 24)

        Args:
            sample_rate: Process every Nth frame (1 = all frames, 5 = every 5th frame)

        Returns:
            List of SemanticFrame objects
        """
        import rosbag2_py
        from rclpy.serialization import deserialize_message

        print(f"\n{'='*60}")
        print(" Step B: Extracting Semantic Data (Road & Lane Markings)")
        print(f"{'='*60}")

        reader = self._create_bag_reader()
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

        if self.SEMANTIC_TOPIC not in topic_types:
            print(f"  WARNING: Semantic topic not found: {self.SEMANTIC_TOPIC}")
            print("  Make sure to record with semantic camera enabled.")
            return []

        img_type = self._get_message_type(topic_types[self.SEMANTIC_TOPIC])

        storage_filter = rosbag2_py.StorageFilter(topics=[self.SEMANTIC_TOPIC])
        reader.set_filter(storage_filter)

        frame_count = 0
        processed_count = 0
        total_lane_pixels = 0
        total_road_coverage = 0.0

        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()

            if topic == self.SEMANTIC_TOPIC:
                frame_count += 1

                # Sample rate - skip frames if needed
                if (frame_count - 1) % sample_rate != 0:
                    continue

                msg = deserialize_message(data, img_type)

                # Convert ROS Image to numpy array
                # Bridge sends mono8 (single channel, each pixel = class ID)
                height = msg.height
                width = msg.width

                # Handle different encodings
                if msg.encoding == 'mono8':
                    # Expected format from bridge - single channel class IDs
                    class_ids = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                elif msg.encoding == 'bgra8':
                    # Raw CARLA format - class ID is in R channel (index 2)
                    img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 4)
                    class_ids = img_data[:, :, 2]
                elif msg.encoding == 'rgb8':
                    img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
                    class_ids = img_data[:, :, 0]
                else:
                    # Try to parse as single channel
                    try:
                        class_ids = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                    except ValueError:
                        print(f"  Warning: Unknown encoding {msg.encoding}, skipping frame")
                        continue

                # Extract road mask (class ID = 1)
                road_mask = (class_ids == self.CLASS_ROAD).astype(np.uint8)

                # Extract lane marking pixels (class ID = 24)
                lane_mask = (class_ids == self.CLASS_LANE_MARKING)
                lane_pixel_coords = np.argwhere(lane_mask)  # Nx2 array of (row, col)

                # Create semantic frame
                semantic_frame = SemanticFrame(
                    timestamp_ns=timestamp_ns,
                    road_mask=road_mask,
                    lane_pixels=lane_pixel_coords
                )
                self.semantic_frames.append(semantic_frame)

                processed_count += 1
                total_lane_pixels += semantic_frame.num_lane_pixels
                total_road_coverage += semantic_frame.road_coverage

                # Progress update every 100 frames
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} frames...")

        # Summary
        print(f"\n  Total frames in bag: {frame_count}")
        print(f"  Processed frames: {processed_count} (sample_rate={sample_rate})")

        if processed_count > 0:
            avg_lane_pixels = total_lane_pixels / processed_count
            avg_road_coverage = total_road_coverage / processed_count
            print(f"  Avg lane pixels per frame: {avg_lane_pixels:.0f}")
            print(f"  Avg road coverage: {avg_road_coverage:.1f}%")

            # Show first and last frame stats
            first_frame = self.semantic_frames[0]
            last_frame = self.semantic_frames[-1]
            print(f"\n  First frame: {first_frame.num_lane_pixels} lane pixels, {first_frame.road_coverage:.1f}% road")
            print(f"  Last frame:  {last_frame.num_lane_pixels} lane pixels, {last_frame.road_coverage:.1f}% road")

        return self.semantic_frames

    def extract_lidar_data(self, sample_rate: int = 1) -> List[LidarFrame]:
        """
        Extract LiDAR point clouds from rosbag.

        Args:
            sample_rate: Process every Nth frame (1 = all frames)

        Returns:
            List of LidarFrame objects
        """
        print(f"\n{'='*60}")
        print(" Extracting LiDAR Data")
        print(f"{'='*60}")
        print(f"  Topic: {self.LIDAR_TOPIC}")

        from rclpy.serialization import deserialize_message

        reader = self._create_bag_reader()
        topic_types = {info.name: info.type for info in reader.get_all_topics_and_types()}

        if self.LIDAR_TOPIC not in topic_types:
            print(f"  Warning: LiDAR topic not found in bag")
            return []

        frame_count = 0
        processed_count = 0
        total_points = 0

        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()

            if topic != self.LIDAR_TOPIC:
                continue

            frame_count += 1
            if frame_count % sample_rate != 0:
                continue

            msg_type = self._get_message_type(topic_types[topic])
            msg = deserialize_message(data, msg_type)

            # Parse PointCloud2 message
            points = self._parse_pointcloud2(msg)
            if points is None or len(points) == 0:
                continue

            lidar_frame = LidarFrame(
                timestamp_ns=timestamp_ns,
                points=points
            )
            self.lidar_frames.append(lidar_frame)
            processed_count += 1
            total_points += lidar_frame.num_points

            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} frames...")

        print(f"\n  Total frames: {frame_count}")
        print(f"  Processed: {processed_count} (sample_rate={sample_rate})")
        if processed_count > 0:
            avg_pts = total_points / processed_count
            print(f"  Avg points per frame: {avg_pts:,.0f}")

        return self.lidar_frames

    def _parse_pointcloud2(self, msg) -> np.ndarray:
        """Parse PointCloud2 message into Nx4 array (x, y, z, intensity)."""
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

    def _find_closest_lidar_frame(self, timestamp_ns: int, tolerance_ns: int = 100_000_000) -> Optional[LidarFrame]:
        """Find the closest LiDAR frame to a given timestamp."""
        if not self.lidar_frames:
            return None
        timestamps = np.array([f.timestamp_ns for f in self.lidar_frames])
        idx = np.searchsorted(timestamps, timestamp_ns)
        candidates = []
        if idx > 0:
            candidates.append((abs(timestamps[idx-1] - timestamp_ns), idx-1))
        if idx < len(timestamps):
            candidates.append((abs(timestamps[idx] - timestamp_ns), idx))
        if not candidates:
            return None
        best_diff, best_idx = min(candidates)
        return self.lidar_frames[best_idx] if best_diff <= tolerance_ns else None

    def project_lane_pixels_to_3d(self, max_distance: float = 50.0,
                                    lidar_search_radius: int = 5,
                                    ground_fit_radius: float = 5.0) -> List[LanePoints3D]:
        """
        Step C: Project 2D lane pixels to 3D using hybrid LiDAR + ground plane approach.

        Pipeline:
            1. Primary: Use LiDAR depth where available
               - Project LiDAR points to camera image
               - For each lane pixel, find nearest LiDAR point within search radius
               - Use actual LiDAR 3D coordinates
            2. Fallback: Local ground plane estimation
               - If no LiDAR point nearby, fit plane to surrounding LiDAR ground points
               - Project pixel ray onto fitted plane

        Args:
            max_distance: Filter out points beyond this distance (meters)
            lidar_search_radius: Search radius in pixels for LiDAR depth lookup
            ground_fit_radius: Radius in meters for local ground plane fitting

        Returns:
            List of LanePoints3D objects, one per semantic frame
        """
        print(f"\n{'='*60}")
        print(" Step C: Projecting Lane Pixels to 3D (Hybrid LiDAR + Ground Plane)")
        print(f"{'='*60}")

        if not self.semantic_frames:
            print("  No semantic frames to project. Run extract_semantic_data first.")
            return []

        if not self.lidar_frames:
            print("  No LiDAR frames available. Run extract_lidar_data first.")
            print("  Falling back to IPM (flat ground) for all pixels.")

        ci = self.camera_intrinsics
        total_points = 0
        lidar_projected = 0
        ground_plane_projected = 0
        no_lidar_frames = 0

        for frame_idx, sem_frame in enumerate(self.semantic_frames):
            if sem_frame.num_lane_pixels == 0:
                self.lane_points_3d.append(LanePoints3D(
                    timestamp_ns=sem_frame.timestamp_ns,
                    points=np.empty((0, 3), dtype=np.float32)
                ))
                continue

            # Find closest LiDAR frame
            lidar_frame = self._find_closest_lidar_frame(sem_frame.timestamp_ns)

            # lane_pixels is (row, col) = (v, u)
            v = sem_frame.lane_pixels[:, 0].astype(np.float32)
            u = sem_frame.lane_pixels[:, 1].astype(np.float32)
            n_pixels = len(u)

            # Output arrays
            x_3d = np.full(n_pixels, np.nan, dtype=np.float32)
            y_3d = np.full(n_pixels, np.nan, dtype=np.float32)
            z_3d = np.full(n_pixels, np.nan, dtype=np.float32)

            if lidar_frame is not None and lidar_frame.num_points > 0:
                # Project LiDAR points to image plane
                lidar_xyz = lidar_frame.get_xyz()
                lidar_uv, lidar_depths, valid_proj = self._project_lidar_to_image(lidar_xyz)

                if np.any(valid_proj):
                    lidar_uv_valid = lidar_uv[valid_proj]
                    lidar_xyz_valid = lidar_xyz[valid_proj]

                    # Create KD-tree for fast nearest neighbor lookup
                    from scipy.spatial import cKDTree
                    tree = cKDTree(lidar_uv_valid)

                    # Query nearest LiDAR points for each lane pixel
                    lane_uv = np.column_stack([u, v])
                    distances, indices = tree.query(lane_uv, k=1, distance_upper_bound=lidar_search_radius)

                    # Mask for pixels with valid LiDAR matches
                    lidar_match_mask = distances < lidar_search_radius

                    if np.any(lidar_match_mask):
                        matched_indices = indices[lidar_match_mask]
                        matched_xyz = lidar_xyz_valid[matched_indices]

                        # Transform LiDAR points to vehicle frame
                        # LiDAR is at (LIDAR_X, LIDAR_Y, LIDAR_Z) in vehicle frame
                        x_3d[lidar_match_mask] = matched_xyz[:, 0] + self.LIDAR_X
                        y_3d[lidar_match_mask] = matched_xyz[:, 1] + self.LIDAR_Y
                        z_3d[lidar_match_mask] = matched_xyz[:, 2] + self.LIDAR_Z

                        lidar_projected += np.sum(lidar_match_mask)

                    # Fallback: Local ground plane for unmatched pixels
                    unmatched_mask = ~lidar_match_mask & np.isnan(x_3d)
                    if np.any(unmatched_mask):
                        # Filter LiDAR points that are likely ground (low z in vehicle frame)
                        ground_z_thresh = 0.3  # 30cm above ground level
                        ground_mask = (lidar_xyz_valid[:, 2] + self.LIDAR_Z) < ground_z_thresh
                        ground_points = lidar_xyz_valid[ground_mask]

                        if len(ground_points) >= 3:
                            # For unmatched pixels, use ground plane projection
                            unmatched_indices = np.where(unmatched_mask)[0]
                            for idx in unmatched_indices:
                                px_u, px_v = u[idx], v[idx]
                                result = self._project_pixel_to_ground_plane(
                                    px_u, px_v, ground_points, ci, ground_fit_radius
                                )
                                if result is not None:
                                    x_3d[idx], y_3d[idx], z_3d[idx] = result
                                    ground_plane_projected += 1
            else:
                no_lidar_frames += 1
                # No LiDAR available - use flat ground assumption (IPM fallback)
                cam_height = self.CAMERA_Z
                norm_x = (u - ci.cx) / ci.fx
                norm_y = (v - ci.cy) / ci.fy
                ray_z = -norm_y  # up direction in camera link frame

                valid_down = ray_z < -1e-6
                s = np.zeros_like(ray_z)
                s[valid_down] = -cam_height / ray_z[valid_down]

                valid_fwd = valid_down & (s > 0)
                x_3d[valid_fwd] = self.CAMERA_X + s[valid_fwd]
                y_3d[valid_fwd] = self.CAMERA_Y + s[valid_fwd] * (-norm_x[valid_fwd])
                z_3d[valid_fwd] = 0.0  # Ground plane
                ground_plane_projected += np.sum(valid_fwd)

            # Filter valid points
            valid = ~np.isnan(x_3d)
            distance = np.sqrt(x_3d**2 + y_3d**2)
            valid &= (distance < max_distance) & (x_3d > 0)

            points_3d = np.column_stack([
                x_3d[valid], y_3d[valid], z_3d[valid]
            ]).astype(np.float32)

            self.lane_points_3d.append(LanePoints3D(
                timestamp_ns=sem_frame.timestamp_ns,
                points=points_3d
            ))
            total_points += len(points_3d)

            if (frame_idx + 1) % 100 == 0:
                print(f"  Processed {frame_idx + 1}/{len(self.semantic_frames)} frames...")

        # Summary
        print(f"\n  Processed {len(self.semantic_frames)} frames")
        print(f"  Total 3D points: {total_points:,}")
        print(f"  LiDAR-projected: {lidar_projected:,}")
        print(f"  Ground-plane-projected: {ground_plane_projected:,}")
        if no_lidar_frames > 0:
            print(f"  Frames without LiDAR: {no_lidar_frames}")

        return self.lane_points_3d

    def _project_lidar_to_image(self, lidar_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project LiDAR points to camera image coordinates.

        Args:
            lidar_xyz: Nx3 array of (x, y, z) in LiDAR frame

        Returns:
            uv: Nx2 array of (u, v) pixel coordinates
            depths: N array of depths (forward distance)
            valid: N boolean mask for points in front of camera and within image
        """
        ci = self.camera_intrinsics

        # Transform LiDAR to camera frame (translation only, no rotation)
        x_cam = lidar_xyz[:, 0] + self.t_lidar_to_cam[0]
        y_cam = lidar_xyz[:, 1] + self.t_lidar_to_cam[1]
        z_cam = lidar_xyz[:, 2] + self.t_lidar_to_cam[2]

        # Filter points in front of camera
        min_forward = 0.5
        front_mask = x_cam > min_forward

        # Camera frame to optical frame
        # z_opt = x_cam (forward), x_opt = -y_cam (right), y_opt = -z_cam (down)
        z_opt = x_cam
        x_opt = -y_cam
        y_opt = -z_cam

        # Pinhole projection
        u = np.full(len(lidar_xyz), -1, dtype=np.float32)
        v = np.full(len(lidar_xyz), -1, dtype=np.float32)

        u[front_mask] = ci.fx * x_opt[front_mask] / z_opt[front_mask] + ci.cx
        v[front_mask] = ci.fy * y_opt[front_mask] / z_opt[front_mask] + ci.cy

        # Check image bounds
        valid = front_mask & (u >= 0) & (u < ci.width) & (v >= 0) & (v < ci.height)

        uv = np.column_stack([u, v])
        depths = z_opt  # forward distance

        return uv, depths, valid

    def _project_pixel_to_ground_plane(self, u: float, v: float,
                                        ground_points: np.ndarray,
                                        ci: CameraIntrinsics,
                                        radius: float) -> Optional[Tuple[float, float, float]]:
        """
        Project a pixel to 3D using local ground plane fit.

        Args:
            u, v: Pixel coordinates
            ground_points: Nx3 array of ground LiDAR points in LiDAR frame
            ci: Camera intrinsics
            radius: Search radius for plane fitting

        Returns:
            (x, y, z) in vehicle frame or None if fit failed
        """
        # First, get approximate ground position using flat assumption
        norm_x = (u - ci.cx) / ci.fx
        norm_y = (v - ci.cy) / ci.fy
        ray_z = -norm_y

        if ray_z >= -1e-6:
            return None  # Ray not pointing down

        s = -self.CAMERA_Z / ray_z
        if s <= 0:
            return None

        # Approximate position in vehicle frame
        approx_x = self.CAMERA_X + s
        approx_y = self.CAMERA_Y + s * (-norm_x)

        # Find nearby ground points (in LiDAR frame, need to adjust)
        # Ground points are already filtered to be near ground level
        ground_xy = ground_points[:, :2] + np.array([self.LIDAR_X, self.LIDAR_Y])
        distances = np.sqrt((ground_xy[:, 0] - approx_x)**2 + (ground_xy[:, 1] - approx_y)**2)
        nearby_mask = distances < radius

        if np.sum(nearby_mask) < 3:
            # Not enough points for plane fit, use flat ground
            return (approx_x, approx_y, 0.0)

        nearby_points = ground_points[nearby_mask]
        # Transform to vehicle frame
        nearby_xyz = nearby_points + np.array([self.LIDAR_X, self.LIDAR_Y, self.LIDAR_Z])

        # Fit plane: ax + by + c = z
        # Using least squares: [x, y, 1] @ [a, b, c]^T = z
        A = np.column_stack([nearby_xyz[:, 0], nearby_xyz[:, 1], np.ones(len(nearby_xyz))])
        z = nearby_xyz[:, 2]

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b, c = coeffs

            # Now project ray onto this plane
            # Ray: origin (cam_x, cam_y, cam_z), direction (1, -norm_x, -norm_y)
            # Plane: z = ax + by + c
            # Solving: cam_z + t*(-norm_y) = a*(cam_x + t) + b*(cam_y + t*(-norm_x)) + c
            # cam_z - norm_y*t = a*cam_x + a*t + b*cam_y - b*norm_x*t + c
            # t*(a - b*norm_x + norm_y) = cam_z - a*cam_x - b*cam_y - c
            denom = a - b * norm_x + norm_y
            if abs(denom) < 1e-6:
                return (approx_x, approx_y, 0.0)

            t = (self.CAMERA_Z - a * self.CAMERA_X - b * self.CAMERA_Y - c) / denom
            if t <= 0:
                return None

            x_3d = self.CAMERA_X + t
            y_3d = self.CAMERA_Y + t * (-norm_x)
            z_3d = a * x_3d + b * y_3d + c

            return (x_3d, y_3d, z_3d)

        except np.linalg.LinAlgError:
            return (approx_x, approx_y, 0.0)

        return self.lane_points_3d

    def visualize_semantic_frame(self, frame_idx: int = 0, output_path: Optional[str] = None):
        """Visualize a single semantic frame."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if not self.semantic_frames:
            print("No semantic data to visualize")
            return

        if frame_idx >= len(self.semantic_frames):
            frame_idx = len(self.semantic_frames) - 1

        frame = self.semantic_frames[frame_idx]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Semantic Frame {frame_idx} - t={frame.timestamp_sec:.2f}s', fontsize=14)

        # Road mask
        ax1 = axes[0]
        ax1.imshow(frame.road_mask, cmap='Greens')
        ax1.set_title(f'Road Mask ({frame.road_coverage:.1f}% coverage)')
        ax1.axis('off')

        # Lane markings
        ax2 = axes[1]
        lane_img = np.zeros_like(frame.road_mask)
        if frame.num_lane_pixels > 0:
            lane_img[frame.lane_pixels[:, 0], frame.lane_pixels[:, 1]] = 1
        ax2.imshow(lane_img, cmap='Blues')
        ax2.set_title(f'Lane Markings ({frame.num_lane_pixels} pixels)')
        ax2.axis('off')

        # Combined view
        ax3 = axes[2]
        combined = np.zeros((*frame.road_mask.shape, 3), dtype=np.uint8)
        combined[:, :, 1] = frame.road_mask * 100  # Green for road
        if frame.num_lane_pixels > 0:
            combined[frame.lane_pixels[:, 0], frame.lane_pixels[:, 1], 0] = 255  # Red for lanes
            combined[frame.lane_pixels[:, 0], frame.lane_pixels[:, 1], 1] = 255  # Yellow lanes
        ax3.imshow(combined)
        ax3.set_title('Combined (Green=Road, Yellow=Lane)')
        ax3.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved semantic visualization to: {output_path}")
        else:
            plt.show()
        plt.close()

    def visualize_semantic_summary(self, output_path: Optional[str] = None):
        """Visualize summary statistics of semantic extraction."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if not self.semantic_frames:
            print("No semantic data to visualize")
            return

        # Collect statistics
        timestamps = np.array([f.timestamp_ns for f in self.semantic_frames]) / 1e9
        timestamps = timestamps - timestamps[0]  # Relative time
        lane_counts = np.array([f.num_lane_pixels for f in self.semantic_frames])
        road_coverages = np.array([f.road_coverage for f in self.semantic_frames])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Semantic Extraction Summary - {self.bag_path.name}', fontsize=14)

        # Lane pixel count over time
        ax1 = axes[0, 0]
        ax1.plot(timestamps, lane_counts, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Lane Pixels')
        ax1.set_title(f'Lane Marking Pixels Over Time (avg: {np.mean(lane_counts):.0f})')
        ax1.grid(True, alpha=0.3)

        # Road coverage over time
        ax2 = axes[0, 1]
        ax2.plot(timestamps, road_coverages, 'g-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Road Coverage (%)')
        ax2.set_title(f'Road Coverage Over Time (avg: {np.mean(road_coverages):.1f}%)')
        ax2.grid(True, alpha=0.3)

        # Lane pixel histogram
        ax3 = axes[1, 0]
        ax3.hist(lane_counts, bins=50, color='blue', alpha=0.7)
        ax3.set_xlabel('Lane Pixels per Frame')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Lane Pixel Counts')
        ax3.grid(True, alpha=0.3)

        # Sample frames - first, middle, last
        ax4 = axes[1, 1]
        sample_indices = [0, len(self.semantic_frames) // 2, len(self.semantic_frames) - 1]
        combined_lanes = np.zeros((*self.semantic_frames[0].road_mask.shape,), dtype=np.float32)

        for idx in sample_indices:
            frame = self.semantic_frames[idx]
            if frame.num_lane_pixels > 0:
                combined_lanes[frame.lane_pixels[:, 0], frame.lane_pixels[:, 1]] += 1

        ax4.imshow(combined_lanes, cmap='hot')
        ax4.set_title('Lane Pixels Overlay (3 sample frames)')
        ax4.axis('off')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved semantic summary to: {output_path}")
        else:
            plt.show()
        plt.close()

    def visualize_3d_lane_points(self, output_path: Optional[str] = None, sample_frames: int = 10):
        """
        Visualize 3D lane points from Step C.

        Shows a bird's-eye view of projected lane marking points in vehicle frame.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if not self.lane_points_3d:
            print("No 3D lane points to visualize. Run project_lane_pixels_to_3d first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'3D Lane Points (IPM) - {self.bag_path.name}', fontsize=14)

        # Collect all points from sampled frames
        step = max(1, len(self.lane_points_3d) // sample_frames)
        all_x, all_y = [], []
        colors = []
        cmap = plt.cm.viridis

        for i, frame in enumerate(self.lane_points_3d[::step]):
            if frame.num_points > 0:
                all_x.extend(frame.points[:, 0])
                all_y.extend(frame.points[:, 1])
                color = cmap(i / max(1, len(self.lane_points_3d) // step - 1))
                colors.extend([color] * frame.num_points)

        if not all_x:
            print("  No valid 3D points found")
            return

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        # Left plot: Single frame BEV
        ax1 = axes[0]
        # Find a frame with good lane points
        mid_idx = len(self.lane_points_3d) // 2
        sample_frame = self.lane_points_3d[mid_idx]
        if sample_frame.num_points > 0:
            ax1.scatter(sample_frame.points[:, 1], sample_frame.points[:, 0],
                       c='blue', s=1, alpha=0.5)
            ax1.scatter(0, 0, c='red', s=100, marker='^', label='Vehicle')
            ax1.set_xlabel('Y (m) - Left')
            ax1.set_ylabel('X (m) - Forward')
            ax1.set_title(f'Single Frame BEV (frame {mid_idx})\n{sample_frame.num_points} points')
            ax1.legend()
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-20, 20)
            ax1.set_ylim(0, 50)
        else:
            ax1.text(0.5, 0.5, 'No points in sample frame', ha='center', va='center')

        # Right plot: Multiple frames overlaid
        ax2 = axes[1]
        ax2.scatter(all_y, all_x, c=colors, s=1, alpha=0.3)
        ax2.scatter(0, 0, c='red', s=100, marker='^', label='Vehicle')
        ax2.set_xlabel('Y (m) - Left')
        ax2.set_ylabel('X (m) - Forward')
        ax2.set_title(f'Multiple Frames BEV ({len(self.lane_points_3d[::step])} frames)\n{len(all_x):,} points')
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-20, 20)
        ax2.set_ylim(0, 50)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved 3D lane points visualization to: {output_path}")
        else:
            plt.show()
        plt.close()

    def register_to_global_frame(self) -> GlobalLaneMap:
        """
        Step D: Transform local 3D lane points to global frame using trajectory.

        For each frame's 3D points (in vehicle frame), finds the corresponding
        vehicle pose and transforms the points to global/map frame.

        Transform: P_global = R @ P_local + t
        where R is rotation matrix from pose and t is pose position.

        Returns:
            GlobalLaneMap with all accumulated points in global frame
        """
        print(f"\n{'='*60}")
        print(" Step D: Registering to Global Frame")
        print(f"{'='*60}")

        if not self.lane_points_3d:
            print("  No 3D points to register. Run project_lane_pixels_to_3d first.")
            return self.global_lane_map

        if not self.trajectory.poses:
            print("  No trajectory data. Run extract_trajectory first.")
            return self.global_lane_map

        total_registered = 0
        frames_without_pose = 0
        frames_with_points = 0

        for lane_frame in self.lane_points_3d:
            if lane_frame.num_points == 0:
                continue

            frames_with_points += 1

            # Find corresponding pose by timestamp
            pose = self.trajectory.get_pose_at_time(lane_frame.timestamp_ns)

            if pose is None:
                frames_without_pose += 1
                continue

            # Transform points from vehicle frame to global frame
            # P_global = R @ P_local + t
            local_points = lane_frame.points  # Nx3

            # Get rotation matrix and position from pose
            R = pose.rotation_matrix  # 3x3
            t = pose.position         # (3,)

            # Apply transformation: each row of local_points is transformed
            # global_points[i] = R @ local_points[i] + t
            global_points = (R @ local_points.T).T + t

            # Add to global map
            self.global_lane_map.add_points(global_points)
            total_registered += len(global_points)

        # Summary
        print(f"\n  Frames with 3D points: {frames_with_points}")
        print(f"  Frames without matching pose: {frames_without_pose}")
        print(f"  Total registered points: {total_registered:,}")
        print(f"  Global map bounds: {self.global_lane_map.get_bounds()}")

        return self.global_lane_map

    def cluster_and_fit_splines(self, eps: float = 1.5, min_samples: int = 5,
                                  min_lane_points: int = 20,
                                  spline_resolution: float = 0.5,
                                  voxel_size: float = 1.0) -> HDMapLanes:
        """
        Step E: Cluster global lane points and fit splines to each lane marking.

        Pipeline:
            1. Downsample points using voxel grid (for efficiency)
            2. Cluster using DBSCAN (handles arbitrary lane shapes)
            3. Order points along each cluster's principal direction
            4. Fit B-spline to ordered points
            5. Resample spline at fixed resolution

        Args:
            eps: DBSCAN max distance between points in same cluster (meters)
            min_samples: DBSCAN min points to form a cluster
            min_lane_points: Minimum points for a cluster to be considered a lane
            spline_resolution: Distance between sampled spline points (meters)

        Returns:
            HDMapLanes containing fitted lane splines
        """
        print(f"\n{'='*60}")
        print(" Step E: Clustering and Fitting Splines")
        print(f"{'='*60}")

        if self.global_lane_map.num_points == 0:
            print("  No global points. Run register_to_global_frame first.")
            return self.hd_map_lanes

        from sklearn.cluster import DBSCAN
        from scipy.interpolate import splprep, splev

        points = self.global_lane_map.points
        print(f"  Input points: {len(points):,}")

        # Step 1: Voxel-based downsampling for efficiency
        # Using a large voxel size collapses lateral noise/smearing into
        # single points, forcing the greedy nearest-neighbor to step longitudinally.
        print(f"  Downsampling with voxel size: {voxel_size:.2f}m")
        downsampled = self._voxel_downsample(points, voxel_size)
        print(f"  After downsampling: {len(downsampled):,} points")

        # Step 2: DBSCAN clustering
        print(f"  Clustering with DBSCAN (eps={eps}m, min_samples={min_samples})")
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clustering.fit_predict(downsampled[:, :2])  # Cluster on XY only

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        print(f"  Found {n_clusters} clusters, {n_noise:,} noise points")

        # Step 3 & 4: Process each cluster
        lane_id = 0
        valid_lanes = 0
        total_length = 0.0

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            cluster_mask = labels == label
            cluster_points = downsampled[cluster_mask]

            if len(cluster_points) < min_lane_points:
                continue  # Too few points

            # Order points along principal direction
            ordered_pts = self._order_points_along_curve(cluster_points)

            if ordered_pts is None or len(ordered_pts) < 4:
                continue

            # Fit spline and resample
            spline_pts, length = self._fit_and_resample_spline(
                ordered_pts, spline_resolution
            )

            if spline_pts is None or length < 1.0:  # Skip very short lanes
                continue

            lane = LaneSpline(
                lane_id=lane_id,
                points=cluster_points,
                spline_points=spline_pts,
                length=length,
                num_raw_points=len(cluster_points)
            )
            self.hd_map_lanes.add_lane(lane)
            lane_id += 1
            valid_lanes += 1
            total_length += length

        # Summary
        print(f"\n  Valid lanes extracted: {valid_lanes}")
        print(f"  Total lane length: {total_length:.1f}m")
        if self.hd_map_lanes.lanes:
            lengths = [l.length for l in self.hd_map_lanes.lanes]
            print(f"  Lane lengths: min={min(lengths):.1f}m, max={max(lengths):.1f}m, "
                  f"avg={np.mean(lengths):.1f}m")

        return self.hd_map_lanes

    def _voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """Downsample points using voxel grid (keep one point per voxel)."""
        if len(points) == 0:
            return points

        # Quantize to voxel grid
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Use unique voxels to keep one point per voxel
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return points[unique_idx]

    def _order_points_along_curve(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Order 3D points along a curve using trajectory projection.
        This collapses lateral noise by slicing points perpendicular to the path.
        """
        if len(points) < 2:
            return points

        traj_data = self.trajectory.to_numpy()
        if traj_data and 'x' in traj_data and len(traj_data['x']) > 0:
            from scipy.spatial import cKDTree
            traj_xy = np.column_stack([traj_data['x'], traj_data['y']])
            
            # Map lane cluster points to the nearest vehicle trajectory node
            tree = cKDTree(traj_xy)
            _, indices = tree.query(points[:, :2])
            
            # Group points by nearest trajectory index, advancing straight down the lane length
            unique_indices = np.unique(indices)
            
            ordered_pts = []
            for idx in np.sort(unique_indices):
                mask = indices == idx
                # Take the geometric center of this cross-section to eliminate width/zig-zag
                ordered_pts.append(np.mean(points[mask], axis=0))
                
            ordered_array = np.array(ordered_pts)
            
            # Remove duplicated or excessively close points to prevent spline math errors
            if len(ordered_array) >= 2:
                diffs = np.linalg.norm(np.diff(ordered_array, axis=0), axis=1)
                valid_mask = np.concatenate(([True], diffs > 0.05)) # At least 5cm apart
                ordered_array = ordered_array[valid_mask]
                
            return ordered_array

        # -- Fallback (If no trajectory) --
        # Use PCA to find principal direction
        xy = points[:, :2]
        center = np.mean(xy, axis=0)
        centered = xy - center

        # Compute principal direction
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_dir = eigenvectors[:, -1]  # Largest eigenvalue direction

        # Project points onto principal direction to find endpoints
        projections = centered @ principal_dir
        start_idx = np.argmin(projections)

        # Greedy nearest-neighbor ordering
        n = len(points)
        ordered_indices = [start_idx]
        remaining = set(range(n)) - {start_idx}

        from scipy.spatial import cKDTree
        tree = cKDTree(xy)

        current_idx = start_idx
        while remaining:
            # Find nearest unvisited neighbor
            distances, indices = tree.query(xy[current_idx], k=min(20, n))

            next_idx = None
            for idx in indices:
                if idx in remaining:
                    next_idx = idx
                    break

            if next_idx is None:
                # Find globally nearest
                min_dist = float('inf')
                for r in remaining:
                    d = np.linalg.norm(xy[current_idx] - xy[r])
                    if d < min_dist:
                        min_dist = d
                        next_idx = r

            ordered_indices.append(next_idx)
            remaining.remove(next_idx)
            current_idx = next_idx

        return points[ordered_indices]

    def _fit_and_resample_spline(self, points: np.ndarray,
                                   resolution: float) -> Tuple[Optional[np.ndarray], float]:
        """
        Fit B-spline to ordered points and resample at fixed resolution.

        Returns:
            (spline_points, arc_length) or (None, 0) if fitting fails
        """
        from scipy.interpolate import splprep, splev

        if len(points) < 4:
            return None, 0.0

        try:
            # Fit B-spline (use smoothing to handle noise)
            # High smoothing factor acts as a low-pass filter to aggressively flatten micro-jitters
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # Adjust smoothing factor based on number of points.
            # Using a much higher constant to heavily smooth out the remaining jitter.
            s = len(points) * 5.0  # Very aggressive smoothing for highway-like curves

            # Fit 3rd degree B-spline
            tck, u = splprep([x, y, z], s=s, k=min(3, len(points) - 1))

            # Compute arc length by sampling densely
            dense_u = np.linspace(0, 1, 1000)
            dense_pts = np.array(splev(dense_u, tck)).T
            arc_lengths = np.cumsum(np.sqrt(np.sum(np.diff(dense_pts, axis=0)**2, axis=1)))
            total_length = arc_lengths[-1] if len(arc_lengths) > 0 else 0

            if total_length < 1.0:
                return None, 0.0

            # Resample at fixed resolution
            n_samples = max(2, int(total_length / resolution))
            u_new = np.linspace(0, 1, n_samples)
            spline_pts = np.array(splev(u_new, tck)).T

            return spline_pts.astype(np.float32), total_length

        except Exception:
            # If spline fitting fails, return simplified polyline
            return self._simplify_polyline(points, resolution)

    def _simplify_polyline(self, points: np.ndarray,
                            resolution: float) -> Tuple[np.ndarray, float]:
        """Fallback: simplify polyline by resampling."""
        if len(points) < 2:
            return points, 0.0

        # Compute cumulative arc length
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = arc_lengths[-1]

        if total_length < 1.0:
            return points, total_length

        # Resample at fixed intervals
        n_samples = max(2, int(total_length / resolution))
        target_lengths = np.linspace(0, total_length, n_samples)

        resampled = []
        for target in target_lengths:
            idx = np.searchsorted(arc_lengths, target)
            idx = min(idx, len(points) - 1)
            if idx == 0:
                resampled.append(points[0])
            else:
                # Linear interpolation
                t = (target - arc_lengths[idx-1]) / max(segment_lengths[idx-1], 1e-6)
                t = np.clip(t, 0, 1)
                pt = points[idx-1] + t * (points[idx] - points[idx-1])
                resampled.append(pt)

        return np.array(resampled, dtype=np.float32), total_length

    def visualize_lane_splines(self, output_path: Optional[str] = None,
                                 show_trajectory: bool = True,
                                 show_raw_points: bool = False):
        """
        Visualize fitted lane splines (Step E result).

        Args:
            output_path: Save plot to file if provided
            show_trajectory: Overlay vehicle trajectory
            show_raw_points: Show raw clustered points behind splines
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if len(self.hd_map_lanes) == 0:
            print("No lane splines to visualize. Run cluster_and_fit_splines first.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle(f'HD Map Lane Splines (LiDAR) - {self.bag_path.name}', fontsize=14)

        cmap = plt.cm.tab20
        colors = [cmap(i % 20) for i in range(len(self.hd_map_lanes))]

        # Left: Full map view
        ax1 = axes[0]

        if show_raw_points:
            pts = self.global_lane_map.points
            ax1.scatter(pts[:, 0], pts[:, 1], c='lightgray', s=0.5, alpha=0.3, label='Raw points')

        for i, lane in enumerate(self.hd_map_lanes.lanes):
            sp = lane.spline_points
            ax1.plot(sp[:, 0], sp[:, 1], c=colors[i], linewidth=2, alpha=0.8)

        if show_trajectory and self.trajectory.poses:
            traj = self.trajectory.to_numpy()
            ax1.plot(traj['x'], traj['y'], 'r-', linewidth=1, alpha=0.5, label='Vehicle path')
            ax1.plot(traj['x'][0], traj['y'][0], 'go', markersize=8, label='Start')
            ax1.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=8, label='End')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Lane Splines ({len(self.hd_map_lanes)} lanes, {self.hd_map_lanes.total_length():.0f}m total)')
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Right: Zoomed view on a section
        ax2 = axes[1]

        # Find center of trajectory for zoom
        if self.trajectory.poses:
            traj = self.trajectory.to_numpy()
            mid_idx = len(traj['x']) // 2
            zoom_center = (traj['x'][mid_idx], traj['y'][mid_idx])
            zoom_size = 50  # meters
        else:
            bounds = self.hd_map_lanes.get_bounds()
            zoom_center = ((bounds['x_min'] + bounds['x_max']) / 2,
                           (bounds['y_min'] + bounds['y_max']) / 2)
            zoom_size = 50

        if show_raw_points:
            pts = self.global_lane_map.points
            ax2.scatter(pts[:, 0], pts[:, 1], c='lightgray', s=1, alpha=0.5)

        for i, lane in enumerate(self.hd_map_lanes.lanes):
            sp = lane.spline_points
            # Draw with varying width for visibility
            ax2.plot(sp[:, 0], sp[:, 1], c=colors[i], linewidth=3, alpha=0.8)
            # Add lane ID label at midpoint
            mid = len(sp) // 2
            ax2.annotate(f'L{lane.lane_id}', (sp[mid, 0], sp[mid, 1]),
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        if show_trajectory and self.trajectory.poses:
            ax2.plot(traj['x'], traj['y'], 'r-', linewidth=2, alpha=0.7)

        ax2.set_xlim(zoom_center[0] - zoom_size, zoom_center[0] + zoom_size)
        ax2.set_ylim(zoom_center[1] - zoom_size, zoom_center[1] + zoom_size)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Zoomed View (50m window)')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # Add info box
        info_text = (f"Lanes: {len(self.hd_map_lanes)}\n"
                     f"Total length: {self.hd_map_lanes.total_length():.1f}m\n"
                     f"Raw points: {self.global_lane_map.num_points:,}")
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved lane splines visualization to: {output_path}")
        else:
            plt.show()
        plt.close()

    def save_hd_map(self, output_path: str):
        """Save HD map lanes to JSON."""
        output = {
            'bag_path': str(self.bag_path),
            'num_lanes': len(self.hd_map_lanes),
            'total_length_m': self.hd_map_lanes.total_length(),
            'bounds': self.hd_map_lanes.get_bounds(),
            'lanes': self.hd_map_lanes.to_json()
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  Saved HD map to: {output_path}")

    def visualize_global_map(self, output_path: Optional[str] = None,
                              show_trajectory: bool = True,
                              point_size: float = 0.5):
        """
        Visualize the global lane map (bird's eye view).

        Args:
            output_path: Save plot to file if provided
            show_trajectory: Overlay vehicle trajectory
            point_size: Size of lane points in plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if self.global_lane_map.num_points == 0:
            print("No global map data to visualize. Run register_to_global_frame first.")
            return

        fig, ax = plt.subplots(figsize=(12, 12))
        fig.suptitle(f'Global Lane Map (LiDAR) - {self.bag_path.name}', fontsize=14)

        # Plot lane points
        points = self.global_lane_map.points
        ax.scatter(points[:, 0], points[:, 1], c='blue', s=point_size,
                   alpha=0.3, label=f'Lane points ({len(points):,})')

        # Plot trajectory if requested
        if show_trajectory and self.trajectory.poses:
            traj = self.trajectory.to_numpy()
            ax.plot(traj['x'], traj['y'], 'r-', linewidth=1, alpha=0.7, label='Vehicle path')
            ax.plot(traj['x'][0], traj['y'][0], 'go', markersize=10, label='Start')
            ax.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=10, label='End')

            # Draw heading arrows at intervals
            step = max(1, len(traj['x']) // 20)
            for i in range(0, len(traj['x']), step):
                yaw = traj['yaw'][i]
                dx = np.cos(yaw) * 2
                dy = np.sin(yaw) * 2
                ax.arrow(traj['x'][i], traj['y'][i], dx, dy,
                        head_width=0.5, head_length=0.3, fc='red', ec='red', alpha=0.5)

        ax.set_xlabel('X (m) - Global Frame')
        ax.set_ylabel('Y (m) - Global Frame')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add bounds info
        bounds = self.global_lane_map.get_bounds()
        info_text = (f"X: [{bounds['x_min']:.1f}, {bounds['x_max']:.1f}] m\n"
                     f"Y: [{bounds['y_min']:.1f}, {bounds['y_max']:.1f}] m\n"
                     f"Points: {self.global_lane_map.num_points:,}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved global map visualization to: {output_path}")
        else:
            plt.show()
        plt.close()

    def visualize_trajectory(self, output_path: Optional[str] = None):
        """Visualize the extracted trajectory."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping visualization")
            return

        if not self.trajectory.poses:
            print("No trajectory data to visualize")
            return

        data = self.trajectory.to_numpy()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Vehicle Trajectory - {self.bag_path.name}', fontsize=14)

        # Top-down view
        ax1 = axes[0, 0]
        scatter = ax1.scatter(data['x'], data['y'], c=data['timestamps']/1e9,
                             cmap='viridis', s=2, alpha=0.7)
        ax1.plot(data['x'][0], data['y'][0], 'go', markersize=10, label='Start')
        ax1.plot(data['x'][-1], data['y'][-1], 'ro', markersize=10, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Top-Down View (XY)')
        ax1.set_aspect('equal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Time (s)')

        # Heading
        ax2 = axes[0, 1]
        time_sec = (data['timestamps'] - data['timestamps'][0]) / 1e9
        ax2.plot(time_sec, np.degrees(data['yaw']), 'b-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Yaw (degrees)')
        ax2.set_title('Vehicle Heading')
        ax2.grid(True, alpha=0.3)

        # Elevation
        ax3 = axes[1, 0]
        ax3.plot(time_sec, data['z'], 'g-', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Elevation Profile')
        ax3.grid(True, alpha=0.3)

        # Speed
        ax4 = axes[1, 1]
        if len(data['x']) > 1:
            dt = np.diff(data['timestamps']) / 1e9
            dx, dy = np.diff(data['x']), np.diff(data['y'])
            speed_kmh = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6) * 3.6
            ax4.plot(time_sec[1:], speed_kmh, 'r-', linewidth=0.5)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Speed (km/h)')
            ax4.set_title(f'Speed Profile (avg: {np.mean(speed_kmh):.1f} km/h)')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n  Saved trajectory plot to: {output_path}")
        else:
            plt.show()
        plt.close()

    def save_trajectory(self, output_path: str):
        """Save trajectory to JSON."""
        output = {
            'bag_path': str(self.bag_path),
            'num_poses': len(self.trajectory),
            'duration_sec': self.trajectory.duration_sec(),
            'distance_m': self.trajectory.total_distance(),
            'trajectory': self.trajectory.to_json()
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved trajectory to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def process_bag(bag_path: str, output_dir: Optional[str] = None, visualize: bool = False, sample_rate: int = 1):
    """Process a rosbag to generate HD map."""
    bag_path = Path(bag_path)

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = bag_path.parent / f"{bag_path.name}_hdmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(" HD Map Processor")
    print(f"{'='*60}")
    print(f"  Bag: {bag_path}")
    print(f"  Output: {out_dir}")

    processor = HDMapProcessor(str(bag_path))

    # Step A: Extract trajectory
    processor.extract_trajectory()
    processor.extract_gnss()

    # Save trajectory
    traj_path = out_dir / "trajectory.json"
    processor.save_trajectory(str(traj_path))

    # Visualize trajectory if requested
    if visualize:
        plot_path = out_dir / "trajectory.png"
        processor.visualize_trajectory(str(plot_path))

    # Step B: Extract semantic data
    processor.extract_semantic_data(sample_rate=sample_rate)

    # Visualize semantic data if requested
    if visualize and processor.semantic_frames:
        # Save summary visualization
        semantic_summary_path = out_dir / "semantic_summary.png"
        processor.visualize_semantic_summary(str(semantic_summary_path))

        # Save sample frame visualizations
        for idx in [0, len(processor.semantic_frames) // 2, len(processor.semantic_frames) - 1]:
            if idx < len(processor.semantic_frames):
                frame_path = out_dir / f"semantic_frame_{idx}.png"
                processor.visualize_semantic_frame(idx, str(frame_path))

    # Extract LiDAR data for Step C
    processor.extract_lidar_data(sample_rate=sample_rate)

    # Step C: Project 2D lane pixels to 3D (hybrid LiDAR + ground plane)
    if processor.semantic_frames:
        processor.project_lane_pixels_to_3d()

        # Visualize 3D lane points if requested
        if visualize and processor.lane_points_3d:
            lane_3d_path = out_dir / "lane_points_3d.png"
            processor.visualize_3d_lane_points(str(lane_3d_path))

    # Step D: Register to global frame
    if processor.lane_points_3d and processor.trajectory.poses:
        processor.register_to_global_frame()

        # Visualize global map if requested
        if visualize and processor.global_lane_map.num_points > 0:
            global_map_path = out_dir / "global_lane_map.png"
            processor.visualize_global_map(str(global_map_path))

    # Step E: Cluster and fit splines
    if processor.global_lane_map.num_points > 0:
        processor.cluster_and_fit_splines()

        # Save HD map
        hd_map_path = out_dir / "hd_map_lanes.json"
        processor.save_hd_map(str(hd_map_path))

        # Visualize lane splines if requested
        if visualize and len(processor.hd_map_lanes) > 0:
            splines_path = out_dir / "lane_splines.png"
            processor.visualize_lane_splines(str(splines_path), show_raw_points=True)

    print(f"\n{'='*60}")
    print(" HD Map Processing Complete (Steps A-E)")
    print(f"{'='*60}")
    print(f"\n  Trajectory: {len(processor.trajectory)} poses")
    print(f"  Semantic frames: {len(processor.semantic_frames)}")
    print(f"  LiDAR frames: {len(processor.lidar_frames)}")
    if processor.lane_points_3d:
        total_3d_pts = sum(f.num_points for f in processor.lane_points_3d)
        print(f"  3D lane points (local): {total_3d_pts:,}")
    if processor.global_lane_map.num_points > 0:
        print(f"  Global map points: {processor.global_lane_map.num_points:,}")
        bounds = processor.global_lane_map.get_bounds()
        print(f"  Map extent: X=[{bounds['x_min']:.1f}, {bounds['x_max']:.1f}]m, "
              f"Y=[{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]m")
    if len(processor.hd_map_lanes) > 0:
        print(f"  Extracted lanes: {len(processor.hd_map_lanes)}")
        print(f"  Total lane length: {processor.hd_map_lanes.total_length():.1f}m")
    print()

    return processor


def main():
    parser = argparse.ArgumentParser(
        description='HD Map Processor - Generate HD maps from rosbags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-04_29_03
  python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-04_29_03 --visualize
  python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-04_29_03 --sample-rate 5
        """
    )

    parser.add_argument('--bag', required=True, help='Path to rosbag directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory for HD map data')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--sample-rate', type=int, default=1,
                        help='Process every Nth semantic frame (default: 1 = all frames)')

    args = parser.parse_args()

    try:
        process_bag(args.bag, args.output, args.visualize, args.sample_rate)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
