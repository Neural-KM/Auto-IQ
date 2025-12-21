#!/usr/bin/env python

"""
AutoIQ Runner
==================

Main entry point for running the autonomous vehicle demo.

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
from pygame.locals import K_ESCAPE, K_q, K_UP, K_DOWN, K_SPACE, K_g, K_n
import numpy as np
import cv2

# AutoIQ modules
from config.config import CARLA, CONTROL, SENSORS, DISPLAY
from config.visualization import DISPLAY_WIDTH, DISPLAY_HEIGHT

from sensors.sensor_suite import SensorSuite
from visualization.tiled_display import TiledDisplay
from navigation.navigation_manager import NavigationManager
from perception.perception_manager import PerceptionManager
from utils.helpers import spawn_vehicle, get_vehicle_telemetry
from navigation.obstacle_state import ObstacleState
from navigation.lane_state import LaneState

class AutoIQ:
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
        self.visualize_options = args.visualize
        
        # Perception manager (handles obstacle & lane detection)
        self.perception_manager = None
        
        # Shared perception state
        self.obstacle_state = None
        self.lane_state = None
    
    def _setup_carla(self):
        """Setup CARLA client, world, and vehicle."""

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
        # Print initial pose and speed

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_ms = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        print(f"[AutoIQ] Initial vehicle pose: x={transform.location.x:.2f}, y={transform.location.y:.2f}, yaw={transform.rotation.yaw:.2f}°, v={speed_ms:.2f} m/s")

        # Setup sensors from config
        self.sensors = SensorSuite(self.vehicle, self.world)
        self.sensors.add_sensors_from_config()
        print("[AutoIQ] Sensors attached")

        # Wait for sensors
        print("[AutoIQ] Waiting for sensors...")
        if self.args.sync:
            for _ in range(20):
                self.world.tick()
        else:
            time.sleep(1.0)

    def setup(self):

        self._setup_carla()

        # Initialize shared perception state
        self.obstacle_state = ObstacleState()
        self.lane_state = LaneState()
        # Initialize perception manager
        self.perception_manager = PerceptionManager(
            vehicle=self.vehicle,
            world=self.world,
            sensor_suite=self.sensors,
            perception_mode=self.args.perception,
            enable_lane_detection= (self.args.lane_detection == 'on'),
            visualize_options=self.visualize_options,
        )
        print("[AutoIQ] Perception manager initialized")
        
        # Setup navigation manager with shared state
        self.nav_manager = NavigationManager(
            self.vehicle, self.world,
            obstacle_state=self.obstacle_state,
            lane_state=self.lane_state,
            controller_type=self.args.controller,
            visualize_options=self.args.visualize,
            trajectory_mode=self.args.trajectory_mode
        )
        
        # Store destination but don't start navigation yet - wait for G key
        self.destination_index_carla = self.args.dest
        
        # Setup pygame display
        pygame.init()
        pygame.display.set_caption("AutoIQ - Autonomous Demo")
        self.display = pygame.display.set_mode(
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.clock = pygame.time.Clock()
        self.tiled_display = TiledDisplay()
        
        # Apply initial layout based on perception mode
        if self.args.perception == 'fake':
            self._apply_display_layout('default', 'fake_perception')
        else:
            self._apply_display_layout('default', 'real_perception')
        
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
                elif event.key == K_g:
                    # Plan route to destination (but don't start driving yet)
                    print(f"[AutoIQ] Planning route to spawn point {self.destination_index_carla}...")
                    self.nav_manager.plan_route_by_index(self.destination_index_carla)
                elif event.key == K_n:
                    # Start navigation (begin driving the planned route)
                    if self.nav_manager.is_route_planned():
                        print("[AutoIQ] Starting navigation...")
                        self.nav_manager.start_navigation()
                    else:
                        print("[AutoIQ] No route planned. Press G first to plan a route.")
                elif event.key == pygame.K_1:
                    # Default layout (choose based on perception mode)
                    if self.args.perception == 'fake':
                        self._apply_display_layout('default', 'fake_perception')
                        print("[AutoIQ] Layout: Default")
                    else:
                        self._apply_display_layout('default', 'real_perception')
                        print("[AutoIQ] Layout: YOLO Detection")
                elif event.key == pygame.K_2:
                    # Grid layout (all sensors)
                    self._apply_display_layout('grid', 'all_sensors')
                    print("[AutoIQ] Layout: Grid")
                elif event.key == pygame.K_3:
                    # Fullscreen layout
                    self._apply_display_layout('fullscreen', 'third_person')
                    print("[AutoIQ] Layout: Fullscreen")
                elif event.key == pygame.K_4:
                    # Dual layout (main + lidar)
                    self._apply_display_layout('dual', 'lidar')
                    print("[AutoIQ] Layout: Dual (Camera + LiDAR)")
    
    def _apply_display_layout(self, layout_type, option):
        """
        Apply a display layout from config.
        
        Args:
            layout_type: Layout type ('default', 'dual', 'grid', 'fullscreen')
            option: Option name within that layout type
        """
        if layout_type not in DISPLAY:
            print(f"[AutoIQ] Warning: Layout type '{layout_type}' not found, using 'default'")
            layout_type = 'default'
        
        layout = DISPLAY[layout_type]
        options = layout['options']
        
        if option not in options:
            print(f"[AutoIQ] Warning: Option '{option}' not found in '{layout_type}', using first available")
            option = list(options.keys())[0]
        
        slot_mapping = options[option]['slot_mapping']
        
        self.tiled_display.setup_layout(layout_type)
        self._slot_mapping = slot_mapping
    
    def run_step(self):
        """Execute one simulation step."""
        obstacles, lanes_result = self.perception_manager.run_step()
        self.obstacle_state.update(obstacles)
        self.lane_state.update(lanes_result)

        control = self.nav_manager.run_step()

        if control:
            self.vehicle.apply_control(control)
        else:
            self.vehicle.apply_control(carla.VehicleControl(brake=0.5))

        if self.visualize_options and ('all' in self.visualize_options or 'sensors' in self.visualize_options):
            self.sensors.draw_sensor_markers()

        return control
    
    def render(self, control):
        """Render the display using tiled layout."""
        # Get telemetry
        telemetry = get_vehicle_telemetry(self.vehicle, self.world_map)
        telemetry['target_speed'] = self.target_speed
        telemetry['control'] = control
        telemetry['fps'] = self.clock.get_fps()
        
        # Add navigation info
        nav_status = self.nav_manager.get_status()
        telemetry['nav_progress'] = f"{nav_status['progress']*100:.1f}%"
        telemetry['nav_remaining'] = f"{nav_status['remaining_distance']:.0f}m"
        telemetry['behavior_state'] = nav_status.get('behavior_state', 'N/A')
        
        # Add MPC solve time if using MPC
        mpc_info = self.nav_manager.controller.get_mpc_info()
        if mpc_info:
            telemetry['mpc_solve_time_ms'] = mpc_info.get('solve_time_ms', 0)
        
        # Build obstacle detection dict
        obstacle_detection = {
            'obstacle_detected': nav_status.get('obstacle_detected', False),
            'distance': nav_status.get('obstacle_distance', float('inf')),
            'obstacle_type': nav_status.get('obstacle_type', 'unknown')
        }
        
        # Get all images from perception
        perception_images = self.perception_manager.get_images()
        
        # Build source data - collect all available images by source name
        source_data = {}
        
        # Add sensor images
        for sensor_config in SENSORS:
            sensor_name = sensor_config['name']
            sensor_type = sensor_config['type']
            sensor = self.sensors.get_sensor(sensor_name)
            
            if sensor is None:
                continue
            
            if sensor_type == 'rgb':
                source_data[sensor_name] = {'image': sensor.get_data(), 'title': sensor_name.replace('_', ' ').title()}
            elif sensor_type == 'semantic':
                source_data[sensor_name] = {'image': sensor.get_colored_image(), 'title': 'Semantic'}
            elif sensor_type == 'depth':
                source_data[sensor_name] = {'image': sensor.get_normalized_image(), 'title': 'Depth'}
            elif sensor_type == 'lidar':
                source_data[sensor_name] = {'image': sensor.get_bev_image(), 'title': 'LiDAR BEV'}
            else:
                source_data[sensor_name] = {'image': sensor.get_data(), 'title': sensor_name}
        
        # Add perception images
        for name, image in perception_images.items():
            source_data[name] = {'image': image, 'title': name.replace('_', ' ').title()}
        
        # Map slots to source data using slot mapping
        panel_data = {
            'telemetry': telemetry,
            'obstacle': {'obstacle_detection': obstacle_detection},
        }
        
        slot_mapping = getattr(self, '_slot_mapping', {})
        for slot_name, source_name in slot_mapping.items():
            if source_name in source_data:
                panel_data[slot_name] = source_data[source_name]
        
        # Render tiled display
        self.tiled_display.render(self.display, panel_data)
        
        pygame.display.flip()
            
    def run(self):
        """Main loop."""
        print("\n[AutoIQ] Starting autonomous demo...")
        print("[AutoIQ] Controls:")
        print("  G         : Start navigation to destination")
        print("  UP/DOWN   : Adjust target speed")
        print("  LEFT/RIGHT: Force lane change")
        print("  SPACE     : Pause/Resume navigation")
        print("  B         : Reset behavior state")
        print("  1/2/3/4   : Switch layout (Default/Surround/Fullscreen/LiDAR)")
        print("  ESC/Q     : Quit\n")
        
        if self.destination_index_carla is not None:
            print(f"[AutoIQ] Press G to start navigation to spawn point {self.destination_index_carla}")
        
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
        
        # Close perception debug windows
        try:
            if self.perception_manager:
                self.perception_manager.close()
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
    parser.add_argument('--sync', type=bool, default=True,
                        help='Enable synchronous mode (True/False)')
    parser.add_argument('--spawn', type=int, default=50, help='Spawn point index for vehicle (default: 0)')
    parser.add_argument('--dest', type=int, default=250, help='Destination spawn point index')
    parser.add_argument('--visualize', type=str, default='waypoints,sensors,route,live,bbox',
                        help='Comma-separated items to draw in simulator (e.g., "route,waypoints,bbox,all")')
    parser.add_argument('--lane-detection', type=str, default='off', choices=['off', 'on'],
                        help='Lane detection mode: off or on')
    parser.add_argument('--perception', type=str, default='fake',
                        choices=['real', 'fake'], help='Perception mode: real (YOLO+Depth) or fake (ground truth)')
    parser.add_argument('--controller', type=str, default='mpc',
                        choices=['mpc', 'pid'], help='Type of vehicle controller to use (default: mpc)')
    parser.add_argument('--trajectory-mode', type=int, default=1,
                        help='Trajectory generation mode: 0=Heuristic, 1=Trapezoidal (default: 0)')
    args = parser.parse_args()

    args.visualize = set([item.strip().lower() for item in args.visualize.split(',') if item.strip()])
    if 'all' in args.visualize:
        args.visualize = {'all'}

    demo = AutoIQ(args)
    
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
    
    demo.setup()
    demo.run()
    demo.cleanup()


if __name__ == '__main__':
    main()
