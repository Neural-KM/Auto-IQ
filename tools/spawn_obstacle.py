#!/usr/bin/env python3
"""
Obstacle Spawner Tool
=====================

Spawns vehicles or static obstacles in CARLA for testing obstacle detection.
Obstacles are automatically removed when the script exits (unless --persist).

Usage:
    python tools/spawn_obstacle.py                    # Spawn vehicle 20m ahead
    python tools/spawn_obstacle.py --distance 30     # Spawn 30m ahead
    python tools/spawn_obstacle.py --type pedestrian # Spawn pedestrian
    python tools/spawn_obstacle.py --type cone       # Spawn traffic cone
    python tools/spawn_obstacle.py --lane left       # Spawn in left lane
    python tools/spawn_obstacle.py --persist         # Keep obstacle after script exits
"""

import argparse
import sys
import os
import time
import math
import signal
import atexit

# Add parent directory to path (since we're in tools/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
from config.settings import Settings


# Global list to track spawned actors for cleanup
_spawned_actors = []
_world = None


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
            pass
    _spawned_actors.clear()


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nCleaning up obstacles...")
    cleanup_actors()
    sys.exit(0)


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
        
    def get_ego_vehicle(self):
        """Find the ego vehicle."""
        vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            if 'tesla' in vehicle.type_id.lower():
                return vehicle
        # If no Tesla, return any vehicle with 'hero' role or first vehicle
        for vehicle in vehicles:
            if vehicle.attributes.get('role_name') == 'hero':
                return vehicle
        if len(vehicles) > 0:
            return vehicles[0]
        return None
    
    def get_spawn_transform(self, ego_vehicle, distance=20.0, lane_offset='center'):
        """
        Calculate spawn location ahead of ego vehicle using waypoints.
        
        This follows the road ahead rather than just using forward vector,
        so it works correctly on curves.
        
        Args:
            ego_vehicle: The reference vehicle
            distance: Distance ahead in meters
            lane_offset: 'left', 'right', or 'center'
            
        Returns:
            carla.Transform for spawning
        """
        ego_location = ego_vehicle.get_location()
        
        # Get current waypoint
        current_wp = self.map.get_waypoint(ego_location)
        if current_wp is None:
            print("Warning: Could not get waypoint for ego vehicle")
            return self._fallback_transform(ego_vehicle, distance, lane_offset)
        
        # Follow waypoints ahead for the specified distance
        waypoints = current_wp.next(distance)
        
        if not waypoints:
            print("Warning: Could not find waypoints ahead")
            return self._fallback_transform(ego_vehicle, distance, lane_offset)
        
        # Use the first waypoint (straight ahead)
        target_wp = waypoints[0]
        
        # Apply lane offset if needed
        if lane_offset == 'left':
            left_wp = target_wp.get_left_lane()
            if left_wp and left_wp.lane_type == carla.LaneType.Driving:
                target_wp = left_wp
            else:
                print("Warning: No left lane available, using center")
        elif lane_offset == 'right':
            right_wp = target_wp.get_right_lane()
            if right_wp and right_wp.lane_type == carla.LaneType.Driving:
                target_wp = right_wp
            else:
                print("Warning: No right lane available, using center")
        
        spawn_transform = target_wp.transform
        spawn_transform.location.z += 0.5  # Lift slightly above ground
        
        return spawn_transform
    
    def _fallback_transform(self, ego_vehicle, distance, lane_offset):
        """Fallback spawn calculation using simple forward vector."""
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        
        yaw_rad = math.radians(ego_rotation.yaw)
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)
        
        spawn_x = ego_location.x + forward_x * distance
        spawn_y = ego_location.y + forward_y * distance
        spawn_z = ego_location.z + 0.5
        
        if lane_offset == 'left':
            spawn_x -= forward_y * 3.5
            spawn_y += forward_x * 3.5
        elif lane_offset == 'right':
            spawn_x += forward_y * 3.5
            spawn_y -= forward_x * 3.5
        
        return carla.Transform(
            carla.Location(x=spawn_x, y=spawn_y, z=spawn_z),
            carla.Rotation(yaw=ego_rotation.yaw)
        )
    
    def spawn_vehicle(self, transform, vehicle_type='random'):
        """
        Spawn a vehicle obstacle.
        
        Args:
            transform: Where to spawn
            vehicle_type: 'random', 'car', 'truck', 'motorcycle', or specific blueprint
            
        Returns:
            Spawned actor or None
        """
        import random
        
        if vehicle_type == 'random':
            blueprints = list(self.blueprint_library.filter('vehicle.*'))
            blueprints = [bp for bp in blueprints if 'bike' not in bp.id and 'motorcycle' not in bp.id]
        elif vehicle_type == 'car':
            blueprints = list(self.blueprint_library.filter('vehicle.audi.*')) + \
                        list(self.blueprint_library.filter('vehicle.bmw.*')) + \
                        list(self.blueprint_library.filter('vehicle.dodge.*')) + \
                        list(self.blueprint_library.filter('vehicle.mercedes.*'))
        elif vehicle_type == 'truck':
            blueprints = list(self.blueprint_library.filter('vehicle.carlamotors.*'))
        elif vehicle_type == 'motorcycle':
            blueprints = list(self.blueprint_library.filter('vehicle.harley*')) + \
                        list(self.blueprint_library.filter('vehicle.kawasaki*')) + \
                        list(self.blueprint_library.filter('vehicle.yamaha*'))
        else:
            blueprints = list(self.blueprint_library.filter(vehicle_type))
        
        if not blueprints:
            # Fallback to any vehicle
            blueprints = list(self.blueprint_library.filter('vehicle.*'))
            blueprints = [bp for bp in blueprints if 'bike' not in bp.id]
        
        if not blueprints:
            print(f"No blueprints found for vehicle type: {vehicle_type}")
            return None
        
        blueprint = random.choice(blueprints)
        
        if blueprint.has_attribute('color'):
            colors = blueprint.get_attribute('color').recommended_values
            blueprint.set_attribute('color', random.choice(colors))
        
        # Try spawning with slight adjustments if needed
        for z_offset in [0, 0.5, 1.0]:
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
                print(f"Spawned vehicle: {blueprint.id}")
                return actor
            except Exception as e:
                continue
        
        print(f"Failed to spawn vehicle at location")
        return None
    
    def spawn_pedestrian(self, transform):
        """Spawn a pedestrian obstacle."""
        import random
        
        blueprints = list(self.blueprint_library.filter('walker.pedestrian.*'))
        if not blueprints:
            print("No pedestrian blueprints found")
            return None
        
        blueprint = random.choice(blueprints)
        
        # Try multiple positions
        for z_offset in [0.5, 1.0, 1.5]:
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
                print(f"Spawned pedestrian: {blueprint.id}")
                return actor
            except:
                continue
        
        print("Failed to spawn pedestrian")
        return None
    
    def spawn_static_prop(self, transform, prop_type='cone'):
        """Spawn a static prop obstacle."""
        import random
        
        prop_filters = {
            'cone': 'static.prop.constructioncone',
            'barrier': 'static.prop.streetbarrier',
            'box': 'static.prop.box*',
            'trash': 'static.prop.trashbag',
        }
        
        filter_str = prop_filters.get(prop_type, prop_type)
        blueprints = list(self.blueprint_library.filter(filter_str))
        
        if not blueprints:
            print(f"Prop type '{prop_type}' not found")
            return None
        
        blueprint = random.choice(blueprints)
        transform.location.z = 0.1
        
        try:
            actor = self.world.spawn_actor(blueprint, transform)
            _spawned_actors.append(actor.id)
            print(f"Spawned prop: {blueprint.id}")
            return actor
        except Exception as e:
            print(f"Failed to spawn prop: {e}")
            return None
    
    def spawn_multiple_cones(self, ego_vehicle, start_distance=15, count=5, spacing=3):
        """Spawn a line of traffic cones ahead."""
        print(f"Spawning {count} cones starting at {start_distance}m...")
        
        for i in range(count):
            distance = start_distance + i * spacing
            transform = self.get_spawn_transform(ego_vehicle, distance, 'center')
            self.spawn_static_prop(transform, 'cone')
            time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description='Spawn test obstacles in CARLA')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--distance', type=float, nargs='+', default=[20.0], help='Distance(s) ahead (meters)')
    parser.add_argument('--type', choices=['vehicle', 'car', 'truck', 'motorcycle', 
                                           'pedestrian', 'cone', 'barrier', 'cones'],
                       default='vehicle', help='Type of obstacle')
    parser.add_argument('--lane', choices=['left', 'center', 'right'], nargs='+',
                       default=['center'], help='Which lane(s) to spawn in')
    parser.add_argument('--persist', action='store_true', 
                       help='Keep obstacle after script exits (default: auto-remove)')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='How long to keep obstacle (seconds, default: 30)')
    
    args = parser.parse_args()
    
    # Register cleanup handlers (unless --persist)
    if not args.persist:
        atexit.register(cleanup_actors)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("Connecting to CARLA...")
    spawner = ObstacleSpawner(args.host, args.port)
    
    # Find ego vehicle
    ego = spawner.get_ego_vehicle()
    if not ego:
        print("ERROR: No ego vehicle found! Start main.py first.")
        return
    
    print(f"Found ego vehicle: {ego.type_id}")
    ego_loc = ego.get_location()
    print(f"Ego location: ({ego_loc.x:.1f}, {ego_loc.y:.1f}, {ego_loc.z:.1f})")
    
    spawned_count = 0
    
    for dist in args.distance:
        for lane in args.lane:
            # Get spawn location
            transform = spawner.get_spawn_transform(ego, dist, lane)
            print(f"Spawn location ({dist}m, {lane}): ({transform.location.x:.1f}, {transform.location.y:.1f}, {transform.location.z:.1f})")
            
            # Calculate actual distance
            actual_dist = ego_loc.distance(transform.location)
            print(f"Actual distance from ego: {actual_dist:.1f}m")
            
            # Spawn based on type
            actor = None
            if args.type == 'vehicle' or args.type == 'car':
                actor = spawner.spawn_vehicle(transform, 'car')
                if actor:
                    actor.apply_control(carla.VehicleControl(hand_brake=True))
                    print("Vehicle set to stationary (handbrake on)")
                    
            elif args.type == 'truck':
                actor = spawner.spawn_vehicle(transform, 'truck')
                if actor:
                    actor.apply_control(carla.VehicleControl(hand_brake=True))
                    
            elif args.type == 'motorcycle':
                actor = spawner.spawn_vehicle(transform, 'motorcycle')
                if actor:
                    actor.apply_control(carla.VehicleControl(hand_brake=True))
                    
            elif args.type == 'pedestrian':
                actor = spawner.spawn_pedestrian(transform)
                
            elif args.type == 'cone':
                actor = spawner.spawn_static_prop(transform, 'cone')
                
            elif args.type == 'barrier':
                actor = spawner.spawn_static_prop(transform, 'barrier')
                
            elif args.type == 'cones':
                spawner.spawn_multiple_cones(ego, dist, count=5)
                actor = True
            
            if actor:
                spawned_count += 1
            else:
                print(f"Failed to spawn obstacle at {dist}m in {lane} lane.")
    
    if spawned_count == 0:
        print("\nFailed to spawn any obstacles.")
        return
    
    if args.persist:
        print(f"\n{spawned_count} obstacles spawned and will PERSIST until manually removed.")
    else:
        print(f"\n{spawned_count} obstacles spawned! Will be removed in {args.duration}s or on Ctrl+C...")
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            pass
        print("Removing obstacles...")
        cleanup_actors()
        print("Done.")


if __name__ == '__main__':
    main()
