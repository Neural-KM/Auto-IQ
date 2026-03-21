#!/usr/bin/env python3
"""
HD Map Processor - Offline Rosbag Processing
=============================================

Processes recorded rosbags to generate HD maps.

Pipeline Steps:
    A. Extract trajectory from odometry (IMPLEMENTED)
    B. Extract semantic masks (road/lanes) from ground truth camera (IMPLEMENTED)
    C. Project 2D lane pixels to 3D using IPM (IMPLEMENTED)
    D. Transform to global frame using trajectory
    E. Cluster and fit splines

Usage:
    # Basic processing (Steps A, B & C)
    python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-05_52_15

    # With visualizations
    python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-05_52_15 --visualize

    # Process every 5th frame (faster)
    python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-05_52_15 --visualize --sample-rate 5

Output:
    Creates <bag_name>_hdmap/ directory with:
    - trajectory.json       : Vehicle poses over time
    - trajectory.png        : Trajectory visualization (if --visualize)
    - semantic_summary.png  : Lane/road statistics (if --visualize)
    - semantic_frame_*.png  : Sample frame visualizations (if --visualize)
    - lane_points_3d.png    : 3D lane points BEV visualization (if --visualize)
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

    def __init__(self, bag_path: str):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {bag_path}")
        self.trajectory = Trajectory()
        self.gnss_data: List[dict] = []
        self.semantic_frames: List[SemanticFrame] = []  # Step B data
        self.lane_points_3d: List[LanePoints3D] = []    # Step C data
        self._msg_type_cache: Dict[str, type] = {}

        # Initialize camera intrinsics
        self.camera_intrinsics = CameraIntrinsics.from_fov(
            self.CAMERA_FOV, self.CAMERA_WIDTH, self.CAMERA_HEIGHT
        )

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

    def project_lane_pixels_to_3d(self, max_distance: float = 50.0) -> List[LanePoints3D]:
        """
        Step C: Project 2D lane pixels to 3D using IPM (flat ground assumption).

        Uses Inverse Perspective Mapping to project lane marking pixels from
        image coordinates to 3D vehicle frame coordinates (x=forward, y=left, z=0).

        This approach assumes lane markings are on a flat ground plane (z=0),
        which is accurate for typical road surfaces.

        Math (same as ipm_funcs.py):
            1. Pixel to normalized ray: ray = [(u-cx)/fx, (v-cy)/fy, 1] (optical frame)
            2. Convert optical to camera link: link = [1, -norm_x, -norm_y]
            3. Project ray to ground plane (z=0): solve for intersection

        Args:
            max_distance: Filter out points beyond this distance (meters)

        Returns:
            List of LanePoints3D objects, one per semantic frame
        """
        print(f"\n{'='*60}")
        print(" Step C: Projecting Lane Pixels to 3D (IPM)")
        print(f"{'='*60}")

        if not self.semantic_frames:
            print("  No semantic frames to project. Run extract_semantic_data first.")
            return []

        ci = self.camera_intrinsics
        cam_height = self.CAMERA_Z  # Camera height above ground

        total_points = 0
        total_filtered = 0

        for frame in self.semantic_frames:
            if frame.num_lane_pixels == 0:
                # No lane pixels in this frame
                self.lane_points_3d.append(LanePoints3D(
                    timestamp_ns=frame.timestamp_ns,
                    points=np.empty((0, 3), dtype=np.float32)
                ))
                continue

            # lane_pixels is (row, col) = (v, u)
            v = frame.lane_pixels[:, 0].astype(np.float32)
            u = frame.lane_pixels[:, 1].astype(np.float32)

            # 1. Unproject to normalized ray in optical frame
            # Optical: X=right, Y=down, Z=forward
            norm_x = (u - ci.cx) / ci.fx
            norm_y = (v - ci.cy) / ci.fy

            # 2. Convert from Optical to Camera Link Frame
            # CARLA camera_link: X=forward, Y=left, Z=up
            # link = [1, -norm_x, -norm_y] (unit ray direction)
            ray_x = np.ones_like(norm_x)  # forward
            ray_y = -norm_x                # left
            ray_z = -norm_y                # up

            # 3. Intersect ray with ground plane (Z = 0)
            # Camera is at height cam_height above ground
            # Ray origin: (CAMERA_X, CAMERA_Y, cam_height)
            # Ray direction: (ray_x, ray_y, ray_z)
            # Find s such that: cam_height + s * ray_z = 0
            # s = -cam_height / ray_z

            # Only valid for rays pointing down (ray_z < 0)
            valid_mask = ray_z < -1e-6  # Looking down at road

            # Calculate intersection parameter
            s = np.zeros_like(ray_z)
            s[valid_mask] = -cam_height / ray_z[valid_mask]

            # Intersection must be in front (s > 0)
            valid_mask &= (s > 0)

            # Calculate 3D points in vehicle frame
            # (adding camera offset to get vehicle frame coords)
            x_3d = self.CAMERA_X + s * ray_x
            y_3d = self.CAMERA_Y + s * ray_y
            z_3d = np.zeros_like(x_3d)  # Ground plane

            # Filter by distance
            distance = np.sqrt(x_3d**2 + y_3d**2)
            valid_mask &= (distance < max_distance)
            valid_mask &= (x_3d > 0)  # Only forward points

            # Extract valid points
            points_3d = np.column_stack([
                x_3d[valid_mask],
                y_3d[valid_mask],
                z_3d[valid_mask]
            ]).astype(np.float32)

            self.lane_points_3d.append(LanePoints3D(
                timestamp_ns=frame.timestamp_ns,
                points=points_3d
            ))

            total_points += len(points_3d)
            total_filtered += frame.num_lane_pixels - len(points_3d)

        # Summary
        print(f"\n  Processed {len(self.semantic_frames)} frames")
        print(f"  Total 3D points: {total_points:,}")
        print(f"  Filtered out: {total_filtered:,} (behind camera or too far)")
        if len(self.lane_points_3d) > 0:
            valid_frames = [f for f in self.lane_points_3d if f.num_points > 0]
            if valid_frames:
                avg_points = total_points / len(valid_frames)
                print(f"  Avg points per frame: {avg_points:.0f}")
                print(f"  Frames with lane points: {len(valid_frames)}/{len(self.lane_points_3d)}")

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

    # Step C: Project 2D lane pixels to 3D using IPM
    if processor.semantic_frames:
        processor.project_lane_pixels_to_3d()

        # Visualize 3D lane points if requested
        if visualize and processor.lane_points_3d:
            lane_3d_path = out_dir / "lane_points_3d.png"
            processor.visualize_3d_lane_points(str(lane_3d_path))

    print(f"\n{'='*60}")
    print(" Steps A, B & C Complete!")
    print(f"{'='*60}")
    print(f"\n  Trajectory: {len(processor.trajectory)} poses")
    print(f"  Semantic frames: {len(processor.semantic_frames)}")
    if processor.lane_points_3d:
        total_3d_pts = sum(f.num_points for f in processor.lane_points_3d)
        print(f"  3D lane points: {total_3d_pts:,}")
    print("\n  Next steps to implement:")
    print("    D. Global Registration (local -> global frame)")
    print("    E. Accumulation & Filtering (cluster + fit splines)")
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
