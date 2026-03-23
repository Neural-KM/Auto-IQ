#!/usr/bin/env python3
"""Visualize HD Map lanes and vehicle trajectory from JSON files."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def visualize_hdmap(trajectory_path, lanes_path, output_path=None):
    """Visualize trajectory and HD map lanes."""

    # Load data
    traj_data = load_json(trajectory_path)
    lanes_data = load_json(lanes_path)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Colors for lanes
    colors = plt.cm.tab20(np.linspace(0, 1, lanes_data['num_lanes']))

    # --- Plot 1: Trajectory only ---
    ax1 = axes[0]
    traj_x = [p['x'] for p in traj_data['trajectory']]
    traj_y = [p['y'] for p in traj_data['trajectory']]

    ax1.plot(traj_x, traj_y, 'b-', linewidth=2, label='Vehicle Trajectory')
    ax1.scatter(traj_x[0], traj_y[0], c='green', s=100, marker='o', zorder=5, label='Start')
    ax1.scatter(traj_x[-1], traj_y[-1], c='red', s=100, marker='s', zorder=5, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'Vehicle Trajectory\n{traj_data["num_poses"]} poses, {traj_data["distance_m"]:.1f}m')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: HD Map Lanes only ---
    ax2 = axes[1]
    for i, lane in enumerate(lanes_data['lanes']):
        points = np.array(lane['spline_points'])
        ax2.plot(points[:, 0], points[:, 1], color=colors[i], linewidth=1.5,
                label=f"Lane {lane['lane_id']}" if i < 10 else None)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'HD Map Lanes\n{lanes_data["num_lanes"]} lanes, {lanes_data["total_length_m"]:.1f}m total')
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Combined view ---
    ax3 = axes[2]

    # Plot lanes first (background)
    for i, lane in enumerate(lanes_data['lanes']):
        points = np.array(lane['spline_points'])
        ax3.plot(points[:, 0], points[:, 1], color=colors[i], linewidth=1.5, alpha=0.7)

    # Plot trajectory on top
    ax3.plot(traj_x, traj_y, 'b-', linewidth=3, label='Vehicle Trajectory')
    ax3.scatter(traj_x[0], traj_y[0], c='green', s=150, marker='o', zorder=5, label='Start')
    ax3.scatter(traj_x[-1], traj_y[-1], c='red', s=150, marker='s', zorder=5, label='End')

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Combined View\nTrajectory + HD Map Lanes')
    ax3.legend()
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")

    plt.show()

    # Print summary
    print("\n=== Summary ===")
    print(f"Trajectory: {traj_data['num_poses']} poses, {traj_data['duration_sec']:.1f}s, {traj_data['distance_m']:.1f}m")
    print(f"HD Map: {lanes_data['num_lanes']} lanes, {lanes_data['total_length_m']:.1f}m total")
    print(f"Bounds: X [{lanes_data['bounds']['x_min']:.1f}, {lanes_data['bounds']['x_max']:.1f}]")
    print(f"        Y [{lanes_data['bounds']['y_min']:.1f}, {lanes_data['bounds']['y_max']:.1f}]")


if __name__ == "__main__":
    # Default paths
    base_dir = Path(__file__).parent.parent
    rosbag_dir = base_dir / "Rosbags" / "hdmap_2026_03_21-14_32_38_hdmap"

    trajectory_path = rosbag_dir / "trajectory.json"
    lanes_path = rosbag_dir / "hd_map_lanes.json"
    output_path = rosbag_dir / "hdmap_visualization.png"

    visualize_hdmap(trajectory_path, lanes_path, output_path)
