#!/usr/bin/env python3
"""
HD Map Data Collection Script (Standalone)
==========================================

Spawns a vehicle with sensors in CARLA, drives between spawn points using
autopilot, and records a rosbag with all data needed for HD map generation.

This script allows:
    - Automated, hands-free data collection (no manual driving required)
    - Reproducible routes between specific spawn points (e.g., --start 10 --end 50)
    - Standalone operation without running the full autoiq_bridge
    - Automatic stop when destination is reached
    - Clean shutdown (stops recording + destroys spawned actors on Ctrl+C)

Recorded Topics:
    - /carla/ego_vehicle/ground_truth/odometry                    : Vehicle pose
    - /carla/ego_vehicle/sensor/gnss/gnss/fix                     : GPS coordinates
    - /carla/ego_vehicle/sensor/imu/imu/data                      : IMU data
    - /carla/ego_vehicle/sensor/lidar/*/point_cloud               : LiDAR point clouds
    - /carla/ego_vehicle/sensor/camera/rgb/front/image            : RGB camera
    - /carla/ego_vehicle/ground_truth/camera/semantic/front/image : Semantic segmentation (mono8)
    - /tf, /tf_static                                             : Transforms

Usage:
    Run first autoiq_bridge and carla bridge launch to record
    # Record data while driving from spawn 10 to spawn 50
    python tools/hd_map_data_collector.py --start 10 --end 50 --record

    # List all spawn points
    python tools/hd_map_data_collector.py --list-spawns

    # Visualize spawn points in CARLA window
    python tools/hd_map_data_collector.py --show-spawns

    # Use with existing bridge (attach to running vehicle)
    python tools/hd_map_data_collector.py --use-bridge --record

Output:
    Rosbags saved to: Rosbags/hdmap_YYYY_MM_DD-HH_MM_SS/

Process recorded bags with:
    python tools/hd_map_processor.py --bag Rosbags/hdmap_<timestamp> --visualize
"""

import argparse
import json
import math
import os
import signal
import sys
import time
import atexit
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla

# Global cleanup tracking
_spawned_actors: List[int] = []
_world: Optional[carla.World] = None
_collector_instance = None
_cleanup_done = False


def cleanup():
    """Clean up spawned actors on exit."""
    global _spawned_actors, _world, _collector_instance, _cleanup_done

    if _cleanup_done:
        return
    _cleanup_done = True

    if _collector_instance and hasattr(_collector_instance, 'stop_vehicle'):
        try:
            _collector_instance.stop_vehicle()
        except Exception:
            pass

    if _world is None:
        return

    # Delete sensors first, then vehicle, avoiding stop() since CARLA C++ handles it inside destroy()
    for actor_id in reversed(_spawned_actors):
        try:
            actor = _world.get_actor(actor_id)
            if actor and actor.is_alive:
                actor.destroy()
        except Exception:
            pass
            
    # Extra fallback: Ensure the main vehicle itself is destroyed even if we are using the bridge
    # or if it was somehow skipped.
    if _collector_instance and _collector_instance.vehicle:
        try:
            if _collector_instance.vehicle.is_alive:
                _collector_instance.vehicle.destroy()
        except Exception:
            pass
        _collector_instance.vehicle = None

    _spawned_actors.clear()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal. Cleaning up...")
    cleanup()
    sys.exit(0)


class HDMapDataCollector:
    """
    Standalone HD map data collector.

    Spawns vehicle with sensors from config, drives to destination.
    """

    # Default config file path (repo root is 3 levels up from tools/HD_map/)
    DEFAULT_CONFIG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'src/autoiq_bridge/config/objects.json'
    )

    def __init__(self, host='127.0.0.1', port=2000, timeout=10.0):
        """Connect to CARLA server."""
        global _world, _collector_instance
        _collector_instance = self

        print(f"Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)

        self.world = self.client.get_world()
        _world = self.world

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.traffic_manager = self.client.get_trafficmanager()

        # Actors
        self.vehicle = None
        self.sensors: Dict[str, carla.Actor] = {}

        # Route state
        self.start_location = None
        self.end_location = None
        self.route_complete = False
        self.rosbag_process = None

        print(f"Connected to CARLA, map: {self.map.name}")

    def get_spawn_points(self) -> List[Tuple[int, carla.Transform]]:
        """Get all spawn points with indices."""
        return list(enumerate(self.map.get_spawn_points()))

    def list_spawn_points(self):
        """Print all available spawn points."""
        spawn_points = self.get_spawn_points()

        print(f"\n{'='*60}")
        print(f" Available Spawn Points ({len(spawn_points)} total)")
        print(f"{'='*60}")
        print(f"{'Index':<8}{'X':>10}{'Y':>10}{'Z':>10}{'Yaw':>10}")
        print(f"{'-'*60}")

        for idx, sp in spawn_points:
            loc = sp.location
            rot = sp.rotation
            print(f"{idx:<8}{loc.x:>10.1f}{loc.y:>10.1f}{loc.z:>10.1f}{rot.yaw:>10.1f}")

        print(f"{'='*60}\n")

    def visualize_spawn_points(self, lifetime=60.0):
        """Draw spawn points in CARLA with index numbers."""
        spawn_points = self.get_spawn_points()
        debug = self.world.debug

        print(f"\nDrawing {len(spawn_points)} spawn points (visible for {lifetime}s)...")

        for idx, sp in spawn_points:
            debug.draw_point(
                sp.location + carla.Location(z=0.5),
                size=0.15,
                color=carla.Color(0, 255, 255),
                life_time=lifetime
            )
            debug.draw_string(
                sp.location + carla.Location(z=1.2),
                str(idx),
                draw_shadow=True,
                color=carla.Color(255, 255, 255),
                life_time=lifetime
            )

        print("Check the CARLA simulator window to see spawn points.")

    def load_config(self) -> dict:
        """Load and return ego_vehicle config."""
        if not os.path.exists(self.DEFAULT_CONFIG):
            raise FileNotFoundError(f"Config file not found: {self.DEFAULT_CONFIG}")

        with open(self.DEFAULT_CONFIG, 'r') as f:
            config = json.load(f)

        for obj in config.get('objects', []):
            if obj.get('id') == 'ego_vehicle':
                return obj

        raise ValueError("No ego_vehicle found in config file")

    def spawn_vehicle(self, spawn_point_index: int, use_existing=False) -> carla.Actor:
        """Spawn vehicle at specified spawn point using config, or use existing ego_vehicle."""
        spawn_points = self.map.get_spawn_points()

        if spawn_point_index >= len(spawn_points):
            raise ValueError(f"Spawn point {spawn_point_index} out of range (max: {len(spawn_points)-1})")

        spawn_point = spawn_points[spawn_point_index]
        self.start_location = spawn_point.location

        if use_existing:
            print("Looking for existing ego_vehicle in the world instead of spawning a new one...")
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'hero' or actor.attributes.get('role_name') == 'ego_vehicle':
                    self.vehicle = actor
                    self.start_location = self.vehicle.get_location()
                    print(f"Found existing vehicle: {self.vehicle.type_id}")
                    return self.vehicle
            raise RuntimeError("Could not find an existing ego_vehicle (role_name 'hero' or 'ego_vehicle') in the world.")

        # Get vehicle type from config
        ego_config = self.load_config()
        vehicle_type = ego_config.get('type', 'vehicle.tesla.model3')

        vehicle_bp = self.blueprint_library.find(vehicle_type)
        if vehicle_bp is None:
            vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
            print(f"Warning: {vehicle_type} not found, using {vehicle_bp.id}")

        vehicle_bp.set_attribute('role_name', 'hero')

        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        _spawned_actors.append(self.vehicle.id)

        print(f"Spawned {self.vehicle.type_id} at spawn point {spawn_point_index}")
        return self.vehicle

    def spawn_sensors(self) -> Dict[str, carla.Actor]:
        """Spawn sensors from config file attached to vehicle."""
        if self.vehicle is None:
            raise RuntimeError("Vehicle must be spawned before sensors")

        ego_config = self.load_config()
        sensor_configs = ego_config.get('sensors', [])

        print(f"Spawning {len(sensor_configs)} sensors from config...")

        sensors = {}
        for sensor_cfg in sensor_configs:
            sensor_type = sensor_cfg['type']
            sensor_id = sensor_cfg['id']
            sp = sensor_cfg.get('spawn_point', {})

            bp = self.blueprint_library.find(sensor_type)
            if bp is None:
                print(f"  Warning: {sensor_type} not found, skipping")
                continue

            # Set attributes
            for attr in ['image_size_x', 'image_size_y', 'fov', 'sensor_tick',
                         'channels', 'range', 'points_per_second', 'rotation_frequency',
                         'upper_fov', 'lower_fov', 'horizontal_fov']:
                if attr in sensor_cfg and bp.has_attribute(attr):
                    bp.set_attribute(attr, str(sensor_cfg[attr]))

            transform = carla.Transform(
                carla.Location(x=sp.get('x', 0), y=sp.get('y', 0), z=sp.get('z', 0)),
                carla.Rotation(pitch=sp.get('pitch', 0), yaw=sp.get('yaw', 0), roll=sp.get('roll', 0))
            )

            sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
            _spawned_actors.append(sensor.id)
            sensors[sensor_id] = sensor

        self.sensors = sensors
        print(f"  Spawned {len(sensors)} sensors")
        return sensors

    def set_destination(self, end_spawn_point_index: int):
        """Set destination spawn point."""
        spawn_points = self.map.get_spawn_points()

        if end_spawn_point_index >= len(spawn_points):
            raise ValueError(f"End spawn point {end_spawn_point_index} out of range")

        self.end_location = spawn_points[end_spawn_point_index].location

        if self.start_location:
            distance = self.start_location.distance(self.end_location)
            print(f"Destination: spawn point {end_spawn_point_index} ({distance:.0f}m away)")

    def enable_autopilot(self, target_speed_kmh=30.0, ignore_lights=True):
        """Enable autopilot with Traffic Manager."""
        if self.vehicle is None:
            raise RuntimeError("Vehicle must be spawned first")

        tm_port = self.traffic_manager.get_port()
        self.vehicle.set_autopilot(True, tm_port)

        self.traffic_manager.set_desired_speed(self.vehicle, target_speed_kmh)
        self.traffic_manager.auto_lane_change(self.vehicle, False)

        if ignore_lights:
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 100.0)
            self.traffic_manager.ignore_signs_percentage(self.vehicle, 100.0)

        if self.end_location:
            self.traffic_manager.set_path(self.vehicle, [self.end_location])

        print(f"Autopilot enabled at {target_speed_kmh} km/h")

    def check_destination_reached(self) -> bool:
        """Check if vehicle has reached the destination, or passing it."""
        if self.vehicle is None or self.end_location is None:
            return False
            
        current_distance = self.vehicle.get_location().distance(self.end_location)
        
        # Check against a reasonably loose threshold (35 meters)
        if current_distance < 35.0:
            return True
            
        # Optional: track if we were getting closer but now are moving away
        if not hasattr(self, '_min_distance'):
            self._min_distance = current_distance
        else:
            if current_distance < self._min_distance:
                self._min_distance = current_distance
            elif current_distance > self._min_distance + 15.0 and self._min_distance < 50.0:
                # We were within 50 meters, and now we are 15 meters further away tracking from min distance (overshot)
                print(f"\nOvershot destination (min distance was {self._min_distance:.1f}m)")
                return True
                
        return False

    def stop_vehicle(self):
        """Stop the vehicle by disabling autopilot and applying brake."""
        if self.rosbag_process:
            print("\nStopping ros2 bag record cleanly...")
            self.rosbag_process.send_signal(signal.SIGINT)
            try:
                self.rosbag_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print("rosbag process did not stop in time, killing it...")
                self.rosbag_process.kill()
            self.rosbag_process = None

        if self.vehicle is None:
            return
        self.vehicle.set_autopilot(False)
        control = carla.VehicleControl()
        control.brake = 1.0
        control.throttle = 0.0
        self.vehicle.apply_control(control)

    def run(self, duration_limit=600.0, record_rosbag=False, bag_dir="Rosbags"):
        """Run until destination reached or duration limit."""
        if self.vehicle is None:
            raise RuntimeError("Vehicle must be spawned first")

        if record_rosbag:
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            bag_name = os.path.join(bag_dir, f"hdmap_{timestamp}")
            os.makedirs(bag_dir, exist_ok=True)
            print(f"Starting ROS 2 bag record at {bag_name}...")
            # Record only topics needed for HD map creation
            hdmap_topics = [
                '/carla/ego_vehicle/sensor/lidar/top/point_cloud',
                '/carla/ego_vehicle/sensor/lidar/at128_front/point_cloud',
                '/carla/ego_vehicle/sensor/gnss/gnss/fix',
                '/carla/ego_vehicle/sensor/imu/imu/data',
                '/carla/ego_vehicle/ground_truth/odometry',
                '/carla/ego_vehicle/sensor/camera/rgb/front/image',
                '/carla/ego_vehicle/ground_truth/camera/semantic/front/image',  # Ground truth segmentation (mono8)
                '/tf',
                '/tf_static',
            ]
            cmd = [
                'ros2', 'bag', 'record',
                '-o', bag_name,
                '--storage', 'mcap',
                '--max-cache-size', '200000000'
            ]
            cmd.extend(hdmap_topics)
            self.rosbag_process = subprocess.Popen(cmd)

        print(f"\n{'='*60}")
        print(" Running... (Ctrl+C to stop)")
        print(f"{'='*60}\n")

        start_time = time.time()
        last_status = 0

        while True:
            elapsed = time.time() - start_time

            if elapsed > duration_limit:
                print(f"\nDuration limit ({duration_limit}s) reached.")
                self.stop_vehicle()
                break

            if self.check_destination_reached():
                print(f"\nDestination reached!")
                self.stop_vehicle()
                self.route_complete = True
                break

            # Status every 5 seconds
            if elapsed - last_status >= 5.0:
                velocity = self.vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
                pos = self.vehicle.get_location()

                dist = "N/A"
                if self.end_location:
                    dist = f"{pos.distance(self.end_location):.0f}m"

                print(f"[{elapsed:5.0f}s] {speed:4.1f} km/h | Pos: ({pos.x:.0f}, {pos.y:.0f}) | To dest: {dist}")
                last_status = elapsed

            time.sleep(0.1)

        print(f"\nDone! Duration: {time.time() - start_time:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Standalone HD Map Data Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/hd_map_data_collector.py --start 10 --end 50 --record
  python tools/hd_map_data_collector.py --list-spawns
  python tools/hd_map_data_collector.py --show-spawns
        """
    )

    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--start', type=int, default=0, help='Start spawn point index')
    parser.add_argument('--end', type=int, default=None, help='End spawn point index')
    parser.add_argument('--speed', type=float, default=30.0, help='Target speed in km/h')
    parser.add_argument('--duration', type=float, default=600.0, help='Max duration (seconds)')
    parser.add_argument('--respect-lights', action='store_true', help='Respect traffic lights')
    parser.add_argument('--list-spawns', action='store_true', help='List spawn points and exit')
    parser.add_argument('--show-spawns', action='store_true', help='Visualize spawn points in CARLA')
    parser.add_argument('--record', action='store_true', help='Record a ROS 2 bag of all topics during the run')
    # Default bag dir at repo root (3 levels up from tools/HD_map/)
    default_bag_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'Rosbags'
    )
    parser.add_argument('--bag-dir', type=str, default=default_bag_dir, help='Directory to store the recorded rosbag')
    parser.add_argument('--use-bridge', action='store_true', help='Do not spawn vehicle/sensors. Attach to existing vehicle spawned by autoiq_bridge')

    args = parser.parse_args()

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        collector = HDMapDataCollector(args.host, args.port)

        if args.list_spawns:
            collector.list_spawn_points()
            return 0

        if args.show_spawns:
            collector.visualize_spawn_points()
            input("\nPress Enter to exit...")
            return 0

        # Spawn vehicle and sensors
        collector.spawn_vehicle(args.start, use_existing=args.use_bridge)
        if not args.use_bridge:
            collector.spawn_sensors()

        if args.end is not None:
            collector.set_destination(args.end)

        time.sleep(1.0)  # Let sensors initialize

        collector.enable_autopilot(
            target_speed_kmh=args.speed,
            ignore_lights=not args.respect_lights
        )

        collector.run(duration_limit=args.duration, record_rosbag=args.record, bag_dir=args.bag_dir)
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1
    finally:
        cleanup()


if __name__ == '__main__':
    sys.exit(main())
