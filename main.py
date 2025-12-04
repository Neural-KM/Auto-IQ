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
    from pygame.locals import K_ESCAPE, K_q, K_UP, K_DOWN, K_SPACE
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
from visualization import HUD
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
        self.hud = None
        self.display = None
        self.clock = None
        
        # State
        self.target_speed = args.speed
        self.running = True
        self.show_perception = True
    
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
        self.sensors.add_semantic_camera('semantic', position='front')
        self.sensors.add_depth_camera('depth', position='front')
        print("[AutoIQ] Sensors attached")
        
        # Setup navigation manager
        self.nav_manager = NavigationManager(self.vehicle, self.world)
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
        self.hud = HUD(self.args.width, self.args.height)
        
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
        """Render the display."""
        # Get camera image
        rgb_camera = self.sensors.get_sensor('main')
        surface = rgb_camera.get_pygame_surface()
        if surface:
            self.display.blit(surface, (0, 0))
        
        # Get telemetry
        telemetry = get_vehicle_telemetry(self.vehicle, self.world_map)
        telemetry['target_speed'] = self.target_speed
        
        # Add navigation info
        nav_status = self.nav_manager.get_status()
        telemetry['nav_progress'] = f"{nav_status['progress']*100:.1f}%"
        telemetry['nav_remaining'] = f"{nav_status['remaining_distance']:.0f}m"
        
        # Render HUD
        self.hud.render(self.display, telemetry, control)
        
        # Render perception views if enabled
        if self.show_perception:
            semantic_cam = self.sensors.get_sensor('semantic')
            depth_cam = self.sensors.get_sensor('depth')
            
            semantic_img = semantic_cam.get_colored_image() if semantic_cam else None
            depth_img = depth_cam.get_normalized_image() if depth_cam else None
            
            self.hud.render_perception_view(
                self.display, semantic_img, depth_img,
                x=self.args.width - 340, y=10
            )
        
        pygame.display.flip()
    
    def run(self):
        """Main loop."""
        print("\n[AutoIQ] Starting autonomous demo...")
        print("[AutoIQ] UP/DOWN: Adjust speed | SPACE: Pause/Resume | R: Random dest | ESC: Quit\n")
        
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
    parser.add_argument('--res', default='1280x720', help='Window resolution')
    parser.add_argument('--spawn', type=int, default=0, help='Spawn point index for vehicle (default: 0)')
    parser.add_argument('--dest', type=int, default=None, help='Destination spawn point index')
    parser.add_argument('--list', action='store_true', help='List spawn points and exit')
    args = parser.parse_args()
    
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
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
