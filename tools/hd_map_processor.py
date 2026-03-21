#!/usr/bin/env python3
"""
HD Map Processor - Offline Rosbag Processing
=============================================

Processes recorded rosbags to generate HD maps.

Rosbags/hdmap_2026_03_21-05_52_15Pipeline Steps:
    A. Extract trajectory from odometry
    B. Extract semantic masks (road/lanes) from ground truth camera
    C. Project 2D lane pixels to 3D using LiDAR
    D. Transform to global frame using trajectory
    E. Cluster and fit splines

Usage:
    python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-04_29_03
    python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-04_29_03 --visualize
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
        C. Project 2D lane pixels to 3D using LiDAR
        D. Transform to global frame using trajectory
        E. Cluster and fit splines
    """

    ODOMETRY_TOPIC = '/carla/ego_vehicle/ground_truth/odometry'
    GNSS_TOPIC = '/carla/ego_vehicle/sensor/gnss/gnss/fix'
    SEMANTIC_TOPIC = '/carla/ego_vehicle/sensor/camera/semantic/front/image'
    LIDAR_TOPIC = '/carla/ego_vehicle/sensor/lidar/at128_front/point_cloud'

    CLASS_ROAD = 1
    CLASS_LANE_MARKING = 24

    def __init__(self, bag_path: str):
        self.bag_path = Path(bag_path)
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {bag_path}")
        self.trajectory = Trajectory()
        self.gnss_data: List[dict] = []
        self._msg_type_cache: Dict[str, type] = {}

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

def process_bag(bag_path: str, output_dir: Optional[str] = None, visualize: bool = False):
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

    # Visualize if requested
    if visualize:
        plot_path = out_dir / "trajectory.png"
        processor.visualize_trajectory(str(plot_path))

    print(f"\n{'='*60}")
    print(" Step A Complete!")
    print(f"{'='*60}")
    print("\n  Next steps to implement:")
    print("    B. Semantic Extraction (lane/road from ground truth)")
    print("    C. 2D-to-3D Projection (camera pixels -> LiDAR depth)")
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
  python tools/hd_map_processor.py --bag Rosbags/hdmap_2026_03_21-04_29_03 --output /tmp/hdmap_output
        """
    )

    parser.add_argument('--bag', required=True, help='Path to rosbag directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory for HD map data')
    parser.add_argument('--visualize', action='store_true', help='Visualize trajectory')

    args = parser.parse_args()

    try:
        process_bag(args.bag, args.output, args.visualize)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
