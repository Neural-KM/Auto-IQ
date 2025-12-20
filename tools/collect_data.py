#!/usr/bin/env python3
"""
Data Collection Tool - Collects sensor data with ground truth for training.

Usage:
    python tools/collect_data.py --spawn 0 --obstacle-spawns 5 10 --frames 100
"""

import argparse
import sys
import os
import time
import json
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import carla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CARLA, SENSORS
from tools.spawn_obstacle import ObstacleSpawner, _spawned_actors, cleanup_actors


@dataclass
class CollectorArgs:
    """Arguments for AutoIQ when used for data collection."""
    host: str = CARLA['connection']['host']
    port: int = CARLA['connection']['port']
    spawn: int = 0
    dest: int = None
    speed: float = 0.0
    sync: bool = True
    visualize: set = None
    no_lane_detection: bool = True
    fake_perception: bool = True
    controller: str = 'mpc'
    trajectory_mode: int = 0
    
    def __post_init__(self):
        if self.visualize is None:
            self.visualize = set()


class DataCollector:
    """Collects sensor data with ground truth annotations."""
    
    def __init__(self, spawn_index: int, output_dir: str):
        self.spawn_index = spawn_index
        self.output_dir = Path(output_dir)
        self.auto_iq = None
        self.obstacle_spawner = None
        
    def setup(self, obstacles: list):
        from main import AutoIQ
        
        args = CollectorArgs(spawn=self.spawn_index)
        self.auto_iq = AutoIQ(args)
        self.auto_iq._setup_carla()
        
        # Keep ego stationary
        self.auto_iq.vehicle.apply_control(
            carla.VehicleControl(hand_brake=True, throttle=0.0, brake=1.0)
        )
        
        # Spawn obstacles using original spawner API
        self.obstacle_spawner = ObstacleSpawner()
        ego = self.auto_iq.vehicle
        for obs in obstacles:
            transform = self.obstacle_spawner.get_spawn_transform(ego, obs['distance'], obs['lane'])
            self.obstacle_spawner.spawn_vehicle(transform)
        for _ in range(20):
                self.auto_iq.world.tick()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'frames').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        for config in SENSORS:
            (self.output_dir / config['type'] / config['name']).mkdir(parents=True, exist_ok=True)
        
        print(f"[DataCollector] Ego at spawn {self.spawn_index}, {len(obstacles)} obstacles")
    
    def _get_next_frame_id(self):
        """Get the next frame ID based on existing files."""
        frames_dir = self.output_dir / 'frames'
        existing = list(frames_dir.glob('*.json'))
        if not existing:
            return 0
        ids = [int(f.stem) for f in existing]
        return max(ids) + 1
        
    def _get_obstacle_annotations(self):
        ego_transform = self.auto_iq.vehicle.get_transform()
        world = self.auto_iq.world
        annotations = []
        
        for actor_id in _spawned_actors:
            actor = world.get_actor(actor_id)
            if actor is None:
                continue
            actor_transform = actor.get_transform()
            bbox = actor.bounding_box
            
            bbox_center_world = actor_transform.transform(bbox.location)
            bbox_center_relative = ego_transform.inverse_transform(bbox_center_world)
            
            annotations.append({
                'id': actor.id,
                'type_id': actor.type_id,
                'world': {
                    'location': {'x': actor_transform.location.x, 'y': actor_transform.location.y, 'z': actor_transform.location.z},
                    'rotation': {'pitch': actor_transform.rotation.pitch, 'yaw': actor_transform.rotation.yaw, 'roll': actor_transform.rotation.roll},
                },
                'ego_relative': {'x': bbox_center_relative.x, 'y': bbox_center_relative.y, 'z': bbox_center_relative.z},
                'bbox': {'length': bbox.extent.x * 2, 'width': bbox.extent.y * 2, 'height': bbox.extent.z * 2},
                'distance': math.sqrt(bbox_center_relative.x**2 + bbox_center_relative.y**2 + bbox_center_relative.z**2),
            })
            
        annotations.sort(key=lambda x: x['distance'])
        return annotations
        
    def _save_frame(self, frame_id: int):
        frame_str = f"{frame_id:06d}"
        ego_transform = self.auto_iq.vehicle.get_transform()
        
        frame_data = {
            'frame_id': frame_id,
            'timestamp': time.time(),
            'ego': {
                'location': {'x': ego_transform.location.x, 'y': ego_transform.location.y, 'z': ego_transform.location.z},
                'rotation': {'pitch': ego_transform.rotation.pitch, 'yaw': ego_transform.rotation.yaw, 'roll': ego_transform.rotation.roll},
            },
            'obstacles': self._get_obstacle_annotations(),
        }
        
        with open(self.output_dir / 'frames' / f"{frame_str}.json", 'w') as f:
            json.dump(frame_data, f, indent=2)
            
        sensors = self.auto_iq.sensors
        for config in SENSORS:
            sensor = sensors.get_sensor(config['name'])
            data = sensor.get_data()
            
            if config['type'] == 'rgb':
                cv2.imwrite(str(self.output_dir / 'rgb' / config['name'] / f"{frame_str}.png"),
                           cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            elif config['type'] == 'depth':
                np.save(str(self.output_dir / 'depth' / config['name'] / f"{frame_str}.npy"),
                       sensor.get_depth_meters())
            elif config['type'] == 'semantic':
                cv2.imwrite(str(self.output_dir / 'semantic' / config['name'] / f"{frame_str}.png"),
                           cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            elif config['type'] == 'lidar':
                np.save(str(self.output_dir / 'lidar' / config['name'] / f"{frame_str}.npy"),
                       sensor.get_point_cloud())
                    
    def collect(self):
        world = self.auto_iq.world
        
        # Tick once to ensure sensors have data
        world.tick()
        
        frame_id = self._get_next_frame_id()
        frame_str = f"{frame_id:06d}"
        
        metadata = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'map': self.auto_iq.world_map.name,
            'ego_vehicle': self.auto_iq.vehicle.type_id,
            'ego_spawn_index': self.spawn_index,
            'sensors': SENSORS,
            'obstacles': self._get_obstacle_annotations()
        }
        with open(self.output_dir / 'metadata' / f"{frame_str}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._save_frame(frame_id)
        print(f"[DataCollector] Saved frame {frame_id} to {self.output_dir}")
        
    def cleanup(self):
        cleanup_actors()
        if self.auto_iq:
            self.auto_iq.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Collect sensor data for perception training')
    parser.add_argument('--spawn', type=int, required=True)
    parser.add_argument('--obstacle', nargs=2, action='append', required=True,
                        metavar=('DISTANCE', 'LANE'), help='Obstacle: distance (m) and lane (left/center/right)')
    parser.add_argument('--output', type=str, default='./perception/data')
    args = parser.parse_args()
    
    obstacles = [{'distance': float(d), 'lane': l} for d, l in args.obstacle]
    collector = DataCollector(spawn_index=args.spawn, output_dir=args.output)
    
    try:
        collector.setup(obstacles=obstacles)
        collector.collect()
    finally:
        collector.cleanup()


if __name__ == '__main__':
    main()
