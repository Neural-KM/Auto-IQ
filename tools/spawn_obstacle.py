#!/usr/bin/env python3
"""
Obstacle Spawner Tool
=====================

Spawns actors (vehicles, pedestrians, props) in CARLA for testing.
Supports both absolute world coordinates and relative-to-ego positions.

Usage Examples:
    # Single obstacle 20m ahead of ego, stationary
    python tools/spawn_obstacle.py --obstacles "vehicle:rel:20,0,0"
    
    # Multiple obstacles with different configs
    python tools/spawn_obstacle.py --obstacles \
        "vehicle:rel:20,0,0:autopilot" \
        "vehicle:rel:30,3.5,0" \
        "pedestrian:rel:25,-2,0"
    
    # Absolute world coordinates
    python tools/spawn_obstacle.py --obstacles "vehicle:abs:100,50,0.5"
    
    # Using a config file
    python tools/spawn_obstacle.py --config obstacles.yaml
    
Obstacle Format:
    type:coord_mode:x,y,z[:options]
    
    type: vehicle, car, truck, motorcycle, pedestrian, cone, barrier, 
          or any CARLA blueprint filter (e.g., vehicle.tesla.model3)
    
    coord_mode:
        rel - Relative to ego (x=forward, y=left, z=up)
        abs - Absolute world coordinates
        lane - Lane-relative (x=distance ahead, y=lane offset as integer: -1=right, 0=center, +1=left)
    
    x,y,z: Position coordinates
    
    options (optional, colon-separated):
        autopilot - Enable autopilot (vehicles only)
        yaw=N - Yaw offset in degrees
        speed=N - Initial speed in m/s (requires autopilot or physics)
        color=R,G,B - Vehicle color (0-255 each)

Examples:
    vehicle:rel:20,0,0                    # Stationary car 20m ahead
    vehicle:rel:30,3.5,0:autopilot        # Autopilot car 30m ahead, left lane
    vehicle:lane:25,1,0:autopilot         # Autopilot car 25m ahead in left lane
    pedestrian:rel:15,-2,0                # Pedestrian 15m ahead, 2m right
    cone:rel:10,0,0                       # Traffic cone 10m ahead
    vehicle.tesla.model3:rel:20,0,0       # Specific vehicle blueprint
"""

import argparse
import sys
import os
import time
import math
import signal
import atexit
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla

# Global tracking for cleanup
_spawned_actors: List[int] = []
_world: Optional[carla.World] = None


def cleanup_actors():
    """Remove all spawned actors on exit."""
    global _spawned_actors, _world
    if _world is None:
        return
    
    for actor_id in _spawned_actors:
        try:
            actor = _world.get_actor(actor_id)
            if actor:
                actor.destroy()
                print(f"Cleaned up actor {actor_id}")
        except Exception as e:
            print(f"Failed to clean up actor {actor_id}. Error: {e}")
            pass
    _spawned_actors.clear()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nCleaning up obstacles...")
    cleanup_actors()
    sys.exit(0)


@dataclass
class ObstacleSpec:
    """Specification for a single obstacle to spawn."""
    actor_type: str  # vehicle, pedestrian, cone, or blueprint filter
    coord_mode: str  # 'rel', 'abs', or 'lane'
    x: float
    y: float
    z: float = 0.0
    autopilot: bool = False
    yaw_offset: float = 0.0
    speed: float = 0.0
    color: Optional[tuple] = None
    
    @classmethod
    def from_string(cls, spec_str: str) -> 'ObstacleSpec':
        """
        Parse obstacle spec from string format.
        
        Format: type:coord_mode:x,y,z[:options]
        Options: autopilot, yaw=N, speed=N, color=R,G,B
        """
        parts = spec_str.strip().split(':')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid obstacle spec: {spec_str}. "
                           f"Expected format: type:coord_mode:x,y,z[:options]")
        
        actor_type = parts[0].lower()
        coord_mode = parts[1].lower()
        
        if coord_mode not in ['rel', 'abs', 'lane']:
            raise ValueError(f"Invalid coord_mode: {coord_mode}. Must be 'rel', 'abs', or 'lane'")
        
        # Parse coordinates
        coords = parts[2].split(',')
        if len(coords) < 2:
            raise ValueError(f"Invalid coordinates: {parts[2]}. Expected x,y or x,y,z")
        
        x = float(coords[0])
        y = float(coords[1])
        z = float(coords[2]) if len(coords) > 2 else 0.0
        
        # Parse options
        autopilot = False
        yaw_offset = 0.0
        speed = 0.0
        color = None
        
        for opt in parts[3:]:
            opt = opt.strip().lower()
            if opt == 'autopilot':
                autopilot = True
            elif opt.startswith('yaw='):
                yaw_offset = float(opt.split('=')[1])
            elif opt.startswith('speed='):
                speed = float(opt.split('=')[1])
            elif opt.startswith('color='):
                rgb = opt.split('=')[1].split(',')
                color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        return cls(
            actor_type=actor_type,
            coord_mode=coord_mode,
            x=x, y=y, z=z,
            autopilot=autopilot,
            yaw_offset=yaw_offset,
            speed=speed,
            color=color
        )
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ObstacleSpec':
        """Parse obstacle spec from dictionary (YAML config)."""
        return cls(
            actor_type=d.get('type', 'vehicle'),
            coord_mode=d.get('coord_mode', 'rel'),
            x=float(d.get('x', 20)),
            y=float(d.get('y', 0)),
            z=float(d.get('z', 0)),
            autopilot=d.get('autopilot', False),
            yaw_offset=float(d.get('yaw', 0)),
            speed=float(d.get('speed', 0)),
            color=tuple(d['color']) if 'color' in d else None
        )


class ObstacleSpawner:
    """Spawns test obstacles in CARLA."""
    
    def __init__(self, host='127.0.0.1', port=2000):
        """Connect to CARLA."""
        global _world
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        _world = self.world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.ego_vehicle = None
        
    def get_ego_vehicle(self) -> Optional[carla.Actor]:
        """Find the ego vehicle."""
        if self.ego_vehicle is not None:
            return self.ego_vehicle
            
        vehicles = self.world.get_actors().filter('vehicle.*')
        
        # First look for 'hero' role
        for vehicle in vehicles:
            if vehicle.attributes.get('role_name') == 'hero':
                self.ego_vehicle = vehicle
                return vehicle
        
        # Fallback to Tesla or first vehicle
        for vehicle in vehicles:
            if 'tesla' in vehicle.type_id.lower():
                self.ego_vehicle = vehicle
                return vehicle
        
        if len(vehicles) > 0:
            self.ego_vehicle = vehicles[0]
            return self.ego_vehicle
            
        return None
    
    def compute_spawn_transform(self, spec: ObstacleSpec) -> Optional[carla.Transform]:
        """
        Compute world transform from obstacle specification.
        
        Relative coordinates (rel):
            x = forward distance from ego
            y = left distance from ego (positive = left, negative = right)
            z = height offset
            
        Lane coordinates (lane):
            x = forward distance along road
            y = lane offset as integer (-1 = right lane, 0 = current, +1 = left lane)
            z = ignored (uses road height)
            
        Absolute coordinates (abs):
            x, y, z = world coordinates
        """
        if spec.coord_mode == 'abs':
            # Absolute world coordinates
            return carla.Transform(
                carla.Location(x=spec.x, y=spec.y, z=spec.z + 0.5),
                carla.Rotation(yaw=spec.yaw_offset)
            )
        
        ego = self.get_ego_vehicle()
        if ego is None:
            print("ERROR: No ego vehicle found for relative spawn")
            return None
        
        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        ego_yaw = ego_transform.rotation.yaw
        yaw_rad = math.radians(ego_yaw)
        
        if spec.coord_mode == 'rel':
            # Relative to ego: x=forward, y=left (ROS convention)
            # Transform to CARLA world coordinates
            # Forward vector
            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)
            # Left vector (perpendicular)
            left_x = -math.sin(yaw_rad)
            left_y = math.cos(yaw_rad)
            
            # Compute world position
            world_x = ego_location.x + spec.x * forward_x + spec.y * left_x
            world_y = ego_location.y + spec.x * forward_y + spec.y * left_y
            world_z = ego_location.z + spec.z + 0.5
            
            spawn_yaw = ego_yaw + spec.yaw_offset
            
            return carla.Transform(
                carla.Location(x=world_x, y=world_y, z=world_z),
                carla.Rotation(yaw=spawn_yaw)
            )
        
        elif spec.coord_mode == 'lane':
            # Lane-relative: use waypoints
            current_wp = self.map.get_waypoint(ego_location)
            if current_wp is None:
                print("Warning: Could not get waypoint, falling back to relative")
                spec.coord_mode = 'rel'
                return self.compute_spawn_transform(spec)
            
            # Get waypoint at specified distance ahead
            waypoints = current_wp.next(spec.x)
            if not waypoints:
                print("Warning: No waypoints ahead, falling back to relative")
                spec.coord_mode = 'rel'
                return self.compute_spawn_transform(spec)
            
            target_wp = waypoints[0]
            
            # Apply lane offset (y = lane index)
            lane_offset = int(spec.y)
            for _ in range(abs(lane_offset)):
                if lane_offset > 0:  # Left
                    new_wp = target_wp.get_left_lane()
                else:  # Right
                    new_wp = target_wp.get_right_lane()
                
                if new_wp and new_wp.lane_type == carla.LaneType.Driving:
                    target_wp = new_wp
                else:
                    print(f"Warning: Lane offset {lane_offset} not fully available")
                    break
            
            spawn_transform = target_wp.transform
            spawn_transform.location.z += 0.5
            spawn_transform.rotation.yaw += spec.yaw_offset
            
            return spawn_transform
        
        return None
    
    def get_blueprint(self, actor_type: str) -> Optional[carla.ActorBlueprint]:
        """Get appropriate blueprint for actor type."""
        import random
        
        # Handle common aliases
        type_mappings = {
            'vehicle': 'vehicle.*',
            'car': 'vehicle.audi.*',
            'truck': 'vehicle.carlamotors.*',
            'motorcycle': 'vehicle.harley*',
            'pedestrian': 'walker.pedestrian.*',
            'cone': 'static.prop.constructioncone',
            'barrier': 'static.prop.streetbarrier',
            'box': 'static.prop.box*',
        }
        
        filter_str = type_mappings.get(actor_type, actor_type)
        blueprints = list(self.blueprint_library.filter(filter_str))
        
        # For generic 'vehicle', exclude bikes for stability
        if actor_type == 'vehicle':
            blueprints = [bp for bp in blueprints 
                         if 'bike' not in bp.id and 'motorcycle' not in bp.id]
        
        if not blueprints:
            print(f"No blueprints found for: {actor_type}")
            return None
        
        return random.choice(blueprints)
    
    def spawn_actor(self, spec: ObstacleSpec) -> Optional[carla.Actor]:
        """
        Spawn an actor based on specification.
        
        Returns spawned actor or None on failure.
        """
        import random
        
        transform = self.compute_spawn_transform(spec)
        if transform is None:
            return None
        
        blueprint = self.get_blueprint(spec.actor_type)
        if blueprint is None:
            return None
        
        # Apply color if specified
        if spec.color and blueprint.has_attribute('color'):
            r, g, b = spec.color
            blueprint.set_attribute('color', f'{r},{g},{b}')
        elif blueprint.has_attribute('color'):
            colors = blueprint.get_attribute('color').recommended_values
            blueprint.set_attribute('color', random.choice(colors))
        
        # Try spawning with height adjustments
        for z_offset in [0, 0.5, 1.0, 1.5]:
            try:
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=transform.location.x,
                        y=transform.location.y,
                        z=transform.location.z + z_offset
                    ),
                    transform.rotation
                )
                actor = self.world.spawn_actor(blueprint, spawn_transform)
                _spawned_actors.append(actor.id)
                
                # Configure actor behavior
                self._configure_actor(actor, spec)
                
                print(f"Spawned {spec.actor_type}: {blueprint.id} at "
                      f"({spawn_transform.location.x:.1f}, {spawn_transform.location.y:.1f})"
                      f"{' [autopilot]' if spec.autopilot else ' [stationary]'}")
                return actor
                
            except Exception as e:
                continue
        
        print(f"Failed to spawn {spec.actor_type}")
        return None
    
    def _configure_actor(self, actor: carla.Actor, spec: ObstacleSpec):
        """Configure actor after spawning (autopilot, speed, etc.)."""
        actor_type = actor.type_id
        
        if 'vehicle' in actor_type:
            if spec.autopilot:
                # Enable autopilot
                actor.set_autopilot(True)
                
                # Set initial speed if specified
                if spec.speed > 0:
                    # Get forward vector
                    transform = actor.get_transform()
                    yaw_rad = math.radians(transform.rotation.yaw)
                    velocity = carla.Vector3D(
                        x=spec.speed * math.cos(yaw_rad),
                        y=spec.speed * math.sin(yaw_rad),
                        z=0
                    )
                    actor.set_target_velocity(velocity)
            else:
                # Stationary - apply handbrake
                try:
                    actor.apply_control(carla.VehicleControl(hand_brake=True))
                except Exception as e:
                    print(f"Warning: Could not apply handbrake to vehicle: {e}")
                    pass
        
        elif 'walker' in actor_type:
            if spec.autopilot or spec.speed > 0:
                # For pedestrians with autopilot, spawn AI controller
                try:
                    walker_bp = self.blueprint_library.find('controller.ai.walker')
                    controller = self.world.spawn_actor(walker_bp, carla.Transform(), actor)
                    _spawned_actors.append(controller.id)
                    controller.start()
                    controller.set_max_speed(spec.speed if spec.speed > 0 else 1.4)
                    # Walk to a point ahead
                    controller.go_to_location(
                        self.world.get_random_location_from_navigation()
                    )
                except Exception as e:
                    print(f"Warning: Could not spawn pedestrian controller: {e}")


def load_config(config_path: str) -> List[ObstacleSpec]:
    """Load obstacle specifications from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    obstacles = []
    for obs_dict in config.get('obstacles', []):
        obstacles.append(ObstacleSpec.from_dict(obs_dict))
    
    return obstacles


def main():
    parser = argparse.ArgumentParser(
        description='Spawn test obstacles in CARLA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Obstacle Format: type:coord_mode:x,y,z[:options]

Examples:
  vehicle:rel:20,0,0                 Stationary car 20m ahead
  vehicle:rel:30,3.5,0:autopilot     Autopilot car in left lane
  vehicle:lane:25,1,0:autopilot      Autopilot car in left lane (using waypoints)
  pedestrian:rel:15,-2,0             Pedestrian 15m ahead, 2m right
  cone:rel:10,0,0                    Traffic cone 10m ahead
  vehicle:abs:100,50,1               Vehicle at world coords (100, 50, 1)
  vehicle:rel:20,0,0:yaw=45          Vehicle angled 45 degrees
  vehicle:rel:20,0,0:speed=10:autopilot  Autopilot at 10 m/s
        """
    )
    
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    
    parser.add_argument('--obstacles', '-o', nargs='+', type=str,
                       help='Obstacle specs (type:coord_mode:x,y,z[:options])')
    
    parser.add_argument('--config', '-c', type=str,
                       help='YAML config file with obstacle list')
    
    parser.add_argument('--persist', action='store_true',
                       help='Keep obstacles after script exits')
    
    parser.add_argument('--duration', type=float, default=60.0,
                       help='How long to keep obstacles (seconds, default: 60)')
    
    # Quick shortcuts for common scenarios
    parser.add_argument('--ahead', type=float, metavar='DIST',
                       help='Quick: spawn stationary vehicle DIST meters ahead')
    
    parser.add_argument('--autopilot-ahead', type=float, metavar='DIST',
                       help='Quick: spawn autopilot vehicle DIST meters ahead')
    
    args = parser.parse_args()
    
    # Build obstacle list
    obstacles: List[ObstacleSpec] = []
    
    # From --obstacles argument
    if args.obstacles:
        for spec_str in args.obstacles:
            try:
                obstacles.append(ObstacleSpec.from_string(spec_str))
            except ValueError as e:
                print(f"Error parsing obstacle spec: {e}")
                return 1
    
    # From config file
    if args.config:
        try:
            obstacles.extend(load_config(args.config))
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1
    
    # Quick shortcuts
    if args.ahead:
        obstacles.append(ObstacleSpec(
            actor_type='vehicle',
            coord_mode='rel',
            x=args.ahead, y=0, z=0,
            autopilot=False
        ))
    
    if args.autopilot_ahead:
        obstacles.append(ObstacleSpec(
            actor_type='vehicle',
            coord_mode='rel',
            x=args.autopilot_ahead, y=0, z=0,
            autopilot=True
        ))
    
    # Default if nothing specified
    if not obstacles:
        print("No obstacles specified. Use --obstacles, --config, --ahead, or --autopilot-ahead")
        print("Example: python tools/spawn_obstacle.py --obstacles 'vehicle:rel:20,0,0'")
        parser.print_help()
        return 1
    
    # Register cleanup handlers
    if not args.persist:
        atexit.register(cleanup_actors)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("Connecting to CARLA...")
    try:
        spawner = ObstacleSpawner(args.host, args.port)
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        return 1
    
    # Find ego vehicle for relative spawns
    ego = spawner.get_ego_vehicle()
    if ego:
        print(f"Found ego vehicle: {ego.type_id}")
        ego_loc = ego.get_location()
        print(f"Ego location: ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")
    else:
        # Check if any obstacles need ego
        needs_ego = any(o.coord_mode in ['rel', 'lane'] for o in obstacles)
        if needs_ego:
            print("ERROR: No ego vehicle found! Required for relative spawning.")
            return 1
    
    # Spawn all obstacles
    spawned_count = 0
    for spec in obstacles:
        actor = spawner.spawn_actor(spec)
        if actor:
            spawned_count += 1
    
    if spawned_count == 0:
        print("\nFailed to spawn any obstacles.")
        return 1
    
    print(f"\n{spawned_count}/{len(obstacles)} obstacles spawned successfully.")
    
    if args.persist:
        print("Obstacles will PERSIST until manually removed or CARLA restarts.")
    else:
        print(f"Will be removed in {args.duration}s or on Ctrl+C...")
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            pass
        print("Removing obstacles...")
        cleanup_actors()
        print("Done.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
