#!/usr/bin/env python

"""
AutoIQ Demo Runner
==================

Main entry point for running the autonomous vehicle demo.

Usage:
    python run_demo.py [--host HOST] [--port PORT] [--speed SPEED]
"""

import sys
import os
import argparse
import time

# Add carla module path (one level up from Auto-IQ, then into carla)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'carla'))

try:
    import carla
except ImportError:
    raise RuntimeError('Cannot import carla module. Make sure CARLA PythonAPI is in the path.')

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_q, K_UP, K_DOWN, K_SPACE, K_LEFT, K_RIGHT, K_b
except ImportError:
    raise RuntimeError('Cannot import pygame. pip install pygame')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('Cannot import numpy. pip install numpy')

# AutoIQ modules
from config import Settings
from control import VehicleController
from sensors import SensorSuite
from visualization import HUD, TiledDisplay
from navigation import NavigationManager
from utils.helpers import spawn_vehicle, get_vehicle_telemetry


class AutoIQDemo:
    """Main demo application."""
    
    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.vehicle = None
        self.sensors = None
        self.nav_manager = None
        self.tiled_display = None
        self.display = None
        self.clock = None
        
        # State
        self.target_speed = args.speed
        self.running = True
        self.layout_mode = 'default'  # 'default', 'minimal', 'debug'
    
    def setup(self):
        """Initialize everything."""
        # Connect to CARLA
        print(f"[AutoIQ] Connecting to CARLA at {self.args.host}:{self.args.port}...")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(Settings.CARLA_TIMEOUT)
        
        self.world = self.client.get_world()
        self.world_map = self.world.get_map()
        print(f"[AutoIQ] Connected to map: {self.world_map.name}")
        
        # Set synchronous mode
        if self.args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = Settings.FIXED_DELTA_SECONDS
            self.world.apply_settings(settings)
            print("[AutoIQ] Synchronous mode enabled")
        
        # Spawn vehicle
        self.vehicle = spawn_vehicle(self.world, spawn_index=self.args.spawn)
        print(f"[AutoIQ] Spawned: {self.vehicle.type_id} at spawn point {self.args.spawn}")
        
        # Setup sensors
        self.sensors = SensorSuite(self.vehicle, self.world)
        self.sensors.add_rgb_camera('main', position='third_person')
        self.sensors.add_rgb_camera('front_rgb', position='front')  # Front RGB for debug
        self.sensors.add_semantic_camera('semantic', position='front')
        self.sensors.add_depth_camera('depth', position='front')
        print("[AutoIQ] Sensors attached")
        
        # Setup navigation manager with sensor integration
        self.nav_manager = NavigationManager(
            self.vehicle, self.world, 
            sensor_suite=self.sensors,
            draw_on_simulator=self.args.draw_on_simulator
        )
        self.nav_manager.set_target_speed(self.target_speed)
        
        # Set destination if provided
        if self.args.dest is not None:
            self.nav_manager.set_destination_by_index(self.args.dest)
        
        # Setup pygame display
        pygame.init()
        pygame.display.set_caption("AutoIQ - Autonomous Demo")
        self.display = pygame.display.set_mode(
            (self.args.width, self.args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.clock = pygame.time.Clock()
        self.tiled_display = TiledDisplay(self.args.width, self.args.height)
        
        # Wait for sensors
        print("[AutoIQ] Waiting for sensors...")
        if self.args.sync:
            for _ in range(20):
                self.world.tick()
        else:
            time.sleep(1.0)
        
        print("[AutoIQ] Ready!")
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    self.running = False
                elif event.key == K_UP:
                    self.target_speed = min(self.target_speed + 5, 100)
                    self.nav_manager.set_target_speed(self.target_speed)
                    print(f"[AutoIQ] Target speed: {self.target_speed} km/h")
                elif event.key == K_DOWN:
                    self.target_speed = max(self.target_speed - 5, 10)
                    self.nav_manager.set_target_speed(self.target_speed)
                    print(f"[AutoIQ] Target speed: {self.target_speed} km/h")
                elif event.key == K_SPACE:
                    if self.nav_manager.is_navigating:
                        self.nav_manager.stop()
                        print("[AutoIQ] Navigation paused")
                    else:
                        self.nav_manager.resume()
                        print("[AutoIQ] Navigation resumed")
                elif event.key == K_LEFT:
                    # Force lane change left
                    self.nav_manager.force_lane_change('left')
                elif event.key == K_RIGHT:
                    # Force lane change right
                    self.nav_manager.force_lane_change('right')
                elif event.key == K_b:
                    # Reset behavior to lane following
                    self.nav_manager.reset_behavior()
                    print("[AutoIQ] Behavior reset")
                elif event.key == pygame.K_1:
                    # Default layout
                    self.tiled_display.reconfigure_layout('default')
                    print("[AutoIQ] Layout: Default")
                elif event.key == pygame.K_2:
                    # Minimal layout
                    self.tiled_display.reconfigure_layout('minimal')
                    print("[AutoIQ] Layout: Minimal")
                elif event.key == pygame.K_3:
                    # Debug layout
                    self.tiled_display.reconfigure_layout('debug')
                    print("[AutoIQ] Layout: Debug")
                elif event.key == pygame.K_4:
                    # Perception debug layout
                    self.tiled_display.reconfigure_layout('perception')
                    print("[AutoIQ] Layout: Perception Debug")
    
    def run_step(self):
        """Execute one simulation step."""
        control = self.nav_manager.run_step()
        
        if control:
            self.vehicle.apply_control(control)
        else:
            # Not navigating - apply brakes
            self.vehicle.apply_control(carla.VehicleControl(brake=0.5))
        
        return control
    
    def render(self, control):
        """Render the display using tiled layout."""
        # Get sensor data
        main_cam = self.sensors.get_sensor('main')
        front_rgb_cam = self.sensors.get_sensor('front_rgb')
        semantic_cam = self.sensors.get_sensor('semantic')
        depth_cam = self.sensors.get_sensor('depth')
        
        # Get telemetry
        telemetry = get_vehicle_telemetry(self.vehicle, self.world_map)
        telemetry['target_speed'] = self.target_speed
        telemetry['control'] = control
        
        # Add navigation info
        nav_status = self.nav_manager.get_status()
        telemetry['nav_progress'] = f"{nav_status['progress']*100:.1f}%"
        telemetry['nav_remaining'] = f"{nav_status['remaining_distance']:.0f}m"
        telemetry['behavior_state'] = nav_status.get('behavior_state', 'N/A')
        
        # Build obstacle detection dict
        obstacle_detection = {
            'obstacle_detected': nav_status.get('obstacle_detected', False),
            'distance': nav_status.get('obstacle_distance', float('inf')),
            'position': nav_status.get('obstacle_position', None),
            'center': nav_status.get('obstacle_center', None),
            'pixel_count': nav_status.get('obstacle_pixels', 0),
            'bbox': nav_status.get('obstacle_bbox', None),
            'obstacle_type': nav_status.get('obstacle_type', 'unknown')
        }
        
        # Get debug images from obstacle detector
        debug_images = self.nav_manager.obstacle_detector.get_debug_images()
        
        # Prepare panel data
        panel_data = {
            'main_camera': {
                'image': main_cam.get_data() if main_cam else None
            },
            'front_rgb': {
                'image': front_rgb_cam.get_data() if front_rgb_cam else None
            },
            'semantic': {
                'image': semantic_cam.get_colored_image() if semantic_cam else None,
                'obstacle_detection': obstacle_detection
            },
            'depth': {
                'image': depth_cam.get_normalized_image() if depth_cam else None,
                'obstacle_detection': obstacle_detection
            },
            'telemetry': telemetry,
            'obstacle': {
                'obstacle_detection': obstacle_detection
            },
            # Debug panels for perception layout
            'roi_semantic': {
                'image': debug_images.get('roi_semantic')
            },
            'vehicle_mask': {
                'image': debug_images.get('vehicle_mask')
            },
            'raw_obstacles': {
                'image': debug_images.get('raw_obstacles'),
                'pixel_count': nav_status.get('obstacle_pixels', 0)
            },
            'filtered_obstacles': {
                'image': debug_images.get('filtered_obstacles'),
                'pixel_count': nav_status.get('obstacle_pixels', 0)
            },
            'combined_debug': {
                'image': debug_images.get('combined')
            }
        }
        
        # Render tiled display
        self.tiled_display.render(self.display, panel_data)
        
        pygame.display.flip()
    
    def run(self):
        """Main loop."""
        print("\n[AutoIQ] Starting autonomous demo...")
        print("[AutoIQ] Controls:")
        print("  UP/DOWN   : Adjust target speed")
        print("  LEFT/RIGHT: Force lane change")
        print("  SPACE     : Pause/Resume navigation")
        print("  B         : Reset behavior state")
        print("  1/2/3/4   : Switch layout (Default/Minimal/Debug/Perception)")
        print("  ESC/Q     : Quit\n")
        
        try:
            while self.running:
                self.clock.tick(60)
                
                if self.args.sync:
                    self.world.tick()
                
                self.handle_events()
                control = self.run_step()
                self.render(control)
        
        except KeyboardInterrupt:
            print("\n[AutoIQ] Interrupted")
    
    def cleanup(self):
        """Clean up resources."""
        print("[AutoIQ] Cleaning up...")
        
        if self.sensors:
            self.sensors.destroy_all()
        
        if self.vehicle:
            self.vehicle.destroy()
        
        if self.args.sync and self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        
        pygame.quit()
        print("[AutoIQ] Done.")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='AutoIQ Autonomous Demo')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--speed', type=float, default=30.0, help='Target speed km/h')
    parser.add_argument('--sync', action='store_true', help='Synchronous mode')
    parser.add_argument('--res', default='1280x860', help='Window resolution (default: 1280x860)')
    parser.add_argument('--spawn', type=int, default=0, help='Spawn point index for vehicle (default: 0)')
    parser.add_argument('--dest', type=int, default=None, help='Destination spawn point index')
    parser.add_argument('--list', action='store_true', help='List spawn points and exit')
    parser.add_argument('--draw-on-simulator', action='store_true',
                        help='Draw debug waypoints and markers in the simulator world')
    args = parser.parse_args()
    
    # Parse resolution and ensure minimum size for tiled display
    req_width, req_height = [int(x) for x in args.res.split('x')]
    args.width, args.height = TiledDisplay.get_recommended_size(req_width, req_height)
    
    if (args.width, args.height) != (req_width, req_height):
        print(f"[AutoIQ] Adjusted window size to {args.width}x{args.height} for tiled layout")
    
    # List spawn points if requested
    if args.list:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        spawn_points = client.get_world().get_map().get_spawn_points()
        print(f"\n{len(spawn_points)} spawn points:")
        for i, sp in enumerate(spawn_points):
            print(f"  [{i:3d}] x={sp.location.x:8.1f}, y={sp.location.y:8.1f}")
        return
    
    demo = AutoIQDemo(args)
    
    try:
        demo.setup()
        demo.run()
    finally:
        demo.cleanup()


if __name__ == '__main__':
    main()
