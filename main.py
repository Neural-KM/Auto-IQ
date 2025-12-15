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
import signal
import atexit

# Add carla module path (one level up from Auto-IQ, then into carla)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'carla'))

import carla
import pygame
from pygame.locals import K_ESCAPE, K_q, K_UP, K_DOWN, K_SPACE, K_LEFT, K_RIGHT, K_b, K_g, K_n
import numpy as np
import cv2

# AutoIQ modules
from config.config import CARLA, CONTROL, SENSORS
from sensors.sensor_suite import SensorSuite
from visualization.tiled_display import TiledDisplay
from navigation.navigation_manager import NavigationManager
from perception.lane_detector import LaneDetector, LaneDetectorDebugWindow
from perception.fake_perception import FakePerception
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
        self.enable_lane_detection = not args.no_lane_detection
        self.visualize_options = args.visualize
        
        # Lane detection
        self.lane_detector = None
        self.lane_debug_window = None
        self.lane_detection_result = None
        
        # Fake perception
        self.fake_perception = None
    
    def setup(self):
        """Initialize everything."""
        # Connect to CARLA
        print(f"[AutoIQ] Connecting to CARLA at {self.args.host}:{self.args.port}...")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(CARLA['connection']['timeout'])
        
        self.world = self.client.get_world()
        self.world_map = self.world.get_map()
        print(f"[AutoIQ] Connected to map: {self.world_map.name}")
        
        # Set synchronous mode
        if self.args.sync:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = CARLA['simulation']['fixed_delta_seconds']
            self.world.apply_settings(settings)
            print("[AutoIQ] Synchronous mode enabled")
        
        # Spawn vehicle
        self.vehicle = spawn_vehicle(self.world, spawn_index=self.args.spawn)
        print(f"[AutoIQ] Spawned: {self.vehicle.type_id} at spawn point {self.args.spawn}")
        
        # Initialize fake perception
        fake_perception_debug = ('all' in self.args.visualize) or ('bbox' in self.args.visualize)
        self.fake_perception = FakePerception(self.vehicle, self.world, debug=fake_perception_debug)
        print("[AutoIQ] Fake perception initialized")
        
        # Setup sensors from config
        self.sensors = SensorSuite(self.vehicle, self.world)
        self.sensors.add_sensors_from_config()
        print("[AutoIQ] Sensors attached")
        
        # Setup navigation manager with sensor integration
        self.nav_manager = NavigationManager(
            vehicle=self.vehicle, 
            world=self.world, 
            sensor_suite=self.sensors,
            controller_type=self.args.controller,
            visualize_options=self.args.visualize
        )
        self.nav_manager.set_target_speed(self.target_speed)
        
        # Store destination but don't start navigation yet - wait for G key
        self._pending_destination = self.args.dest
        
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
        
        # Setup lane detection
        semantic_sensor = self.sensors.get_sensor('semantic')
        if self.enable_lane_detection and semantic_sensor:
            # Get camera parameters for proper BEV transform
            camera_fov = semantic_sensor.fov
            camera_pitch = semantic_sensor.position.get('pitch', 0.0)
            camera_height = semantic_sensor.position.get('z', 1.5)
            
            self.lane_detector = LaneDetector(
                image_width=semantic_sensor.width,
                image_height=semantic_sensor.height,
                camera_fov=camera_fov,
                camera_pitch=camera_pitch,
                camera_height=camera_height
            )
            self.lane_debug_window = LaneDetectorDebugWindow("Lane Detection Debug")
            print(f"[AutoIQ] Lane detector initialized (FOV={camera_fov}°, pitch={camera_pitch}°, height={camera_height}m)")
        else:
            if not self.enable_lane_detection:
                print("[AutoIQ] Lane detection disabled via CLI flag")
            else:
                print("[AutoIQ] Warning: No semantic sensor found, lane detection disabled")
        
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
                elif event.key == K_g:
                    # Plan route to destination (but don't start driving yet)
                    if self._pending_destination is not None:
                        print(f"[AutoIQ] Planning route to spawn point {self._pending_destination}...")
                        self.nav_manager.plan_route_by_index(self._pending_destination)
                    else:
                        print("[AutoIQ] No destination set. Use --dest argument.")
                elif event.key == K_n:
                    # Start navigation (begin driving the planned route)
                    if self.nav_manager.is_route_planned():
                        print("[AutoIQ] Starting navigation...")
                        self.nav_manager.start_navigation()
                    else:
                        print("[AutoIQ] No route planned. Press G first to plan a route.")
                elif event.key == pygame.K_1:
                    # Default layout
                    self.tiled_display.reconfigure_layout('default')
                    print("[AutoIQ] Layout: Default")
                elif event.key == pygame.K_2:
                    # Surround layout (all sensors)
                    self.tiled_display.reconfigure_layout('surround')
                    print("[AutoIQ] Layout: Surround")
                elif event.key == pygame.K_3:
                    # Fullscreen layout with dropdown
                    self.tiled_display.reconfigure_layout('fullscreen')
                    print("[AutoIQ] Layout: Fullscreen (click dropdown to select sensor)")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.tiled_display.handle_click(event.pos)
    
    def run_step(self):
        """Execute one simulation step."""
        # Run fake perception
        if self.fake_perception:
            obstacles = self.fake_perception.detect_obstacles(radius=50.0)
            if obstacles:
                print(f"\n[FakePerception] Detected {len(obstacles)} obstacles:")
                for obs in obstacles:
                    bbox = obs['bounding_box']
                    rel_loc = bbox['relative_location']
                    dims = bbox['dimensions']
                    print(f"  - {obs['type']} (ID: {obs['id']}) at {obs['distance']:.1f}m | Speed: {obs['speed']:.1f} m/s")
                    print(f"    Relative Pos: (x={rel_loc.x:.2f}, y={rel_loc.y:.2f}, z={rel_loc.z:.2f})")
                    print(f"    Dimensions:   (l={dims.x:.2f}, w={dims.y:.2f}, h={dims.z:.2f})")
        
        # Run lane detection pipeline first so nav manager has latest data
        if self.enable_lane_detection and self.lane_detector:
            semantic_sensor = self.sensors.get_sensor('semantic')
            if semantic_sensor:
                semantic_image = semantic_sensor.get_data()
                if semantic_image is not None:
                    self.lane_detection_result = self.lane_detector.detect(semantic_image)
                    
                    # Update navigation manager with perception data
                    if self.nav_manager:
                        self.nav_manager.update_perception(self.lane_detection_result)
                    
                    # Print lane detection control inferences
                    if self.lane_detection_result['valid']:
                        confidence = self.lane_detection_result['confidence']
                        print(f"[Lane] Detected | Conf: {confidence:.2f}")

        control = self.nav_manager.run_step()
        
        if control:
            self.vehicle.apply_control(control)
        else:
            # Not navigating - apply brakes
            self.vehicle.apply_control(carla.VehicleControl(brake=0.5))
        
        # Draw sensor markers in simulator
        if self.visualize_options and ('all' in self.visualize_options or 'sensors' in self.visualize_options):
            self.sensors.draw_sensor_markers()
        return control
    
    def render(self, control):
        """Render the display using tiled layout."""
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
        
        # Build panel data dynamically from configured sensors
        panel_data = {
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
        
        # Dynamically add sensor data based on config
        for sensor_config in SENSORS:
            sensor_name = sensor_config['name']
            sensor_type = sensor_config['type']
            sensor = self.sensors.get_sensor(sensor_name)
            
            if sensor is None:
                continue
            
            # Get appropriate data based on sensor type
            if sensor_type == 'rgb':
                panel_data[sensor_name] = {
                    'image': sensor.get_data()
                }
            elif sensor_type == 'semantic':
                panel_data[sensor_name] = {
                    'image': sensor.get_colored_image(),
                    'obstacle_detection': obstacle_detection
                }
            elif sensor_type == 'depth':
                panel_data[sensor_name] = {
                    'image': sensor.get_normalized_image(),
                    'obstacle_detection': obstacle_detection
                }
            else:
                panel_data[sensor_name] = {
                    'image': sensor.get_data()
                }
        
        # Also add main_camera alias (for layouts that use main_camera panel name)
        main_sensor = self.sensors.get_sensor('main')
        if main_sensor:
            panel_data['main_camera'] = {'image': main_sensor.get_data()}
        
        # Render tiled display
        self.tiled_display.render(self.display, panel_data)
        
        pygame.display.flip()
        
        # Show lane detection debug window (separate OpenCV window)
        if self.lane_debug_window and self.lane_detector:
            semantic_sensor = self.sensors.get_sensor('semantic')
            if semantic_sensor:
                semantic_image = semantic_sensor.get_data()
                self.lane_debug_window.show(
                    self.lane_detector,
                    semantic_image,
                    self.lane_detection_result
                )
                # Process OpenCV window events
                LaneDetectorDebugWindow.wait_key(1)
    
    def run(self):
        """Main loop."""
        print("\n[AutoIQ] Starting autonomous demo...")
        print("[AutoIQ] Controls:")
        print("  G         : Start navigation to destination")
        print("  UP/DOWN   : Adjust target speed")
        print("  LEFT/RIGHT: Force lane change")
        print("  SPACE     : Pause/Resume navigation")
        print("  B         : Reset behavior state")
        print("  1/2/3     : Switch layout (Default/Surround/Fullscreen)")
        print("  ESC/Q     : Quit\n")
        
        if self._pending_destination is not None:
            print(f"[AutoIQ] Press G to start navigation to spawn point {self._pending_destination}")
        
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
        """Clean up resources. Safe to call multiple times."""
        if hasattr(self, '_cleaned_up') and self._cleaned_up:
            return
        self._cleaned_up = True
        
        print("[AutoIQ] Cleaning up...")
        
        # Stop navigation and clear debug drawings
        try:
            if self.nav_manager:
                self.nav_manager.cleanup()
        except Exception as e:
            print(f"[AutoIQ] Warning: Error cleaning up navigation: {e}")
        
        # Destroy sensors
        try:
            if self.sensors:
                self.sensors.destroy_all()
                self.sensors = None
        except Exception as e:
            print(f"[AutoIQ] Warning: Error destroying sensors: {e}")
        
        # Destroy vehicle
        try:
            if self.vehicle:
                self.vehicle.destroy()
                self.vehicle = None
        except Exception as e:
            print(f"[AutoIQ] Warning: Error destroying vehicle: {e}")
        
        # Restore world settings
        try:
            if self.args.sync and self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except Exception as e:
            print(f"[AutoIQ] Warning: Error restoring world settings: {e}")
        
        # Close lane detection debug window
        try:
            if self.lane_debug_window:
                self.lane_debug_window.close()
                self.lane_debug_window = None
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[AutoIQ] Warning: Error closing debug windows: {e}")
        
        # Quit pygame
        try:
            pygame.quit()
        except Exception as e:
            print(f"[AutoIQ] Warning: Error quitting pygame: {e}")
        
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
    parser.add_argument('--visualize', type=str, default='',
                        help='Comma-separated items to draw in simulator (e.g., "route,waypoints,bbox,all")')
    parser.add_argument('--no-lane-detection', action='store_true',
                        help='Disable lane detection pipeline (use map waypoints only)')
    parser.add_argument('--controller', type=str, default='mpc',
                        choices=['mpc', 'pid'], help='Type of vehicle controller to use (default: mpc)')
    args = parser.parse_args()

    # Parse visualize options
    args.visualize = set([item.strip().lower() for item in args.visualize.split(',') if item.strip()])
    if 'all' in args.visualize:
        args.visualize = {'all'}
    
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
    
    # Register cleanup handlers for various exit scenarios
    def signal_handler(sig, frame):
        print(f"\n[AutoIQ] Received signal {sig}, shutting down...")
        demo.running = False
        demo.cleanup()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register atexit handler as backup
    atexit.register(demo.cleanup)
    
    try:
        demo.setup()
        demo.run()
    except Exception as e:
        print(f"[AutoIQ] Error: {e}")
    finally:

        demo.cleanup()


if __name__ == '__main__':
    main()
