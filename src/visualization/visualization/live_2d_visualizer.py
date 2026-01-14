"""
Live 2D Visualizer (ROS2 Version)
=================================

Matplotlib-based 2D visualizer that receives data via ROS2 topics.
Runs in a separate thread and displays:
- Current vehicle pose (triangle with heading)
- Local trajectory (cyan)
- Global route (green)
- MPC predicted trajectory (orange)
- Obstacles (red boxes)

This version is decoupled from WorldState and works purely with ROS2 data.
"""

import threading
import time
import math
import numpy as np
from typing import Optional, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2


class Live2DVisualizerROS:
    """
    ROS2-compatible 2D visualizer using matplotlib + OpenCV for display.
    
    All data is set via update methods (called from ROS callbacks).
    The visualizer runs its own thread for rendering.
    
    Supports interactive zoom controls:
    - '+' / '=' : Zoom in (smaller view window, more detail)
    - '-' / '_' : Zoom out (larger view window, less detail)
    - '0' : Reset to auto-fit mode (shows all data)
    - 'v' : Vehicle-centered mode (follows vehicle with current zoom)
    """
    
    # Zoom levels in meters (half-width of view window)
    ZOOM_LEVELS = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    
    def __init__(self, title: str = "Auto-IQ Live 2D Visualizer", refresh_hz: float = 10.0):
        self.title = title
        self.refresh_hz = refresh_hz
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Data stores (all in world/CARLA coordinates)
        self._vehicle_pose = None  # (x, y, yaw)
        self._vehicle_speed = 0.0
        self._trajectory = None  # List of (x, y)
        self._global_route = None  # List of (x, y)
        self._predicted_trajectory = None  # List of (x, y, yaw, speed)
        self._obstacles = []  # List of dicts with x, y, yaw, length, width, type
        
        # Zoom control
        self._zoom_index = 3  # Default to 30m half-width
        self._auto_fit_mode = False  # Start in vehicle-centered mode
        self._vehicle_centered = True  # Follow vehicle
        
        # Plot handles
        self._fig = None
        self._ax = None
        self._vehicle_patch = None
        self._trajectory_line = None
        self._global_route_line = None
        self._predicted_trajectory_line = None
        self._obstacle_patches = []
    
    def start(self) -> bool:
        """Start the visualization thread."""
        if self._thread and self._thread.is_alive():
            return True
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True
    
    def stop(self):
        """Stop the visualization thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
    
    def _run(self):
        """Main visualization loop (runs in separate thread)."""
        # Initialize matplotlib figure
        self._fig = plt.figure(figsize=(8, 8))
        self._ax = self._fig.add_subplot(111)
        self._fig.canvas = FigureCanvasAgg(self._fig)
        
        self._ax.set_aspect('equal', adjustable='box')
        self._ax.grid(True, alpha=0.3)
        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_title(self.title)
        
        # Initialize plot elements
        self._global_route_line, = self._ax.plot([], [], '-', color='green', 
                                                  linewidth=1.5, alpha=0.5, label='Global Route')
        self._trajectory_line, = self._ax.plot([], [], '-', color='cyan', 
                                                linewidth=2, label='Trajectory')
        self._predicted_trajectory_line, = self._ax.plot([], [], '-', color='orange', 
                                               linewidth=2, label='MPC Predicted')
        self._vehicle_patch = Polygon([[0, 0], [0, 0], [0, 0]], color='blue', alpha=0.8)
        self._ax.add_patch(self._vehicle_patch)
        self._ax.legend(loc='upper left')
        
        last_update = 0
        while self._running:
            t = time.time()
            if t - last_update >= 1.0 / self.refresh_hz:
                self._draw_frame()
                last_update = t
            time.sleep(0.01)
        
        # Cleanup
        plt.close(self._fig)
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)
    
    def _draw_frame(self):
        """Draw a single frame."""
        with self._lock:
            vehicle = self._vehicle_pose
            speed = self._vehicle_speed
            trajectory = self._trajectory.copy() if self._trajectory else None
            global_route = self._global_route.copy() if self._global_route else None
            predicted_trajectory = self._predicted_trajectory.copy() if self._predicted_trajectory else None
            obstacles = self._obstacles.copy()
            zoom_idx = self._zoom_index
            auto_fit = self._auto_fit_mode
            vehicle_centered = self._vehicle_centered
        
        xs, ys = [], []
        
        # Draw global route
        if global_route and len(global_route) > 1:
            route = np.array(global_route)
            self._global_route_line.set_data(route[:, 0], route[:, 1])
            xs.extend(route[:, 0].tolist())
            ys.extend(route[:, 1].tolist())
        else:
            self._global_route_line.set_data([], [])
        
        # Draw trajectory
        if trajectory and len(trajectory) > 1:
            traj = np.array(trajectory)
            self._trajectory_line.set_data(traj[:, 0], traj[:, 1])
            xs.extend(traj[:, 0].tolist())
            ys.extend(traj[:, 1].tolist())
        else:
            self._trajectory_line.set_data([], [])
        
        # Draw predicted trajectory
        if predicted_trajectory and len(predicted_trajectory) > 1:
            pred = np.array(predicted_trajectory)
            self._predicted_trajectory_line.set_data(pred[:, 0], pred[:, 1])
            xs.extend(pred[:, 0].tolist())
            ys.extend(pred[:, 1].tolist())
        else:
            self._predicted_trajectory_line.set_data([], [])
        
        # Remove old obstacle patches
        for patch in self._obstacle_patches:
            patch.remove()
        self._obstacle_patches = []
        
        # Draw obstacles
        for obs in obstacles:
            pts = self._create_obstacle_polygon(obs)
            if pts is not None:
                is_vehicle = obs.get('type', 'vehicle') == 'vehicle'
                color = 'red' if is_vehicle else 'orange'
                patch = Polygon(pts, color=color, alpha=0.6)
                self._ax.add_patch(patch)
                self._obstacle_patches.append(patch)
                xs.extend(pts[:, 0].tolist())
                ys.extend(pts[:, 1].tolist())
        
        # Draw vehicle
        if vehicle is not None:
            vx, vy, vyaw = vehicle
            pts = self._create_vehicle_triangle(vx, vy, vyaw)
            self._vehicle_patch.set_xy(pts)
            xs.extend(pts[:, 0].tolist())
            ys.extend(pts[:, 1].tolist())
        else:
            self._vehicle_patch.set_xy([[0, 0], [0, 0], [0, 0]])
        
        # Update plot bounds based on zoom mode
        if auto_fit and xs and ys:
            # Auto-fit mode: show all data with margin
            margin = 20
            self._ax.set_xlim(min(xs) - margin, max(xs) + margin)
            self._ax.set_ylim(min(ys) - margin, max(ys) + margin)
        elif vehicle_centered and vehicle is not None:
            # Vehicle-centered mode: fixed zoom around vehicle
            zoom_half_width = self.ZOOM_LEVELS[zoom_idx]
            vx, vy, _ = vehicle
            self._ax.set_xlim(vx - zoom_half_width, vx + zoom_half_width)
            self._ax.set_ylim(vy - zoom_half_width, vy + zoom_half_width)
        elif xs and ys:
            # Fallback to auto-fit
            margin = 20
            self._ax.set_xlim(min(xs) - margin, max(xs) + margin)
            self._ax.set_ylim(min(ys) - margin, max(ys) + margin)
        
        # Build zoom status string
        if auto_fit:
            zoom_str = "Auto-fit"
        else:
            zoom_str = f"Zoom: {self.ZOOM_LEVELS[zoom_idx]}m"
        
        # Update title with speed and zoom info
        if vehicle is not None:
            self._ax.set_title(f"{self.title} | Speed: {speed * 3.6:.1f} km/h | {zoom_str}\n[+/-] Zoom  [0] Auto-fit  [V] Vehicle-center")
        
        # Render to OpenCV window and handle keyboard
        try:
            self._fig.canvas.draw()
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            cv2.imshow(self.title, img)
            
            # Handle keyboard input for zoom control
            key = cv2.waitKey(1) & 0xFF
            self._handle_key(key)
            cv2.waitKey(1)
        except Exception:
            pass
    
    def _create_vehicle_triangle(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Create triangle polygon for vehicle."""
        length = 4.5
        width = 2.0
        
        # Triangle pointing in heading direction
        pts = np.array([
            [length * 0.6, 0],
            [-length * 0.4, -width * 0.5],
            [-length * 0.4, width * 0.5],
        ])
        
        # Rotate and translate
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pts = pts.dot(R.T) + np.array([x, y])
        
        return pts
    
    def _create_obstacle_polygon(self, obs: dict) -> Optional[np.ndarray]:
        """Create rectangle polygon for obstacle."""
        x = obs.get('x', 0)
        y = obs.get('y', 0)
        yaw = obs.get('yaw', 0)
        length = obs.get('length', 4.0)
        width = obs.get('width', 2.0)
        
        half_l, half_w = length / 2, width / 2
        pts = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])
        
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pts = pts.dot(R.T) + np.array([x, y])
        
        return pts
    
    def _handle_key(self, key: int):
        """
        Handle keyboard input for zoom control.
        
        Keys:
            '+' / '=' : Zoom in (smaller view window)
            '-' / '_' : Zoom out (larger view window)
            '0'       : Reset to auto-fit mode
            'v' / 'V' : Toggle vehicle-centered mode
        """
        if key == 255:  # No key pressed
            return
        
        with self._lock:
            # Zoom in: '+' or '='
            if key in [ord('+'), ord('=')]:
                if self._zoom_index > 0:
                    self._zoom_index -= 1
                    self._auto_fit_mode = False
                    self._vehicle_centered = True
            
            # Zoom out: '-' or '_'
            elif key in [ord('-'), ord('_')]:
                if self._zoom_index < len(self.ZOOM_LEVELS) - 1:
                    self._zoom_index += 1
                    self._auto_fit_mode = False
                    self._vehicle_centered = True
            
            # Auto-fit mode: '0'
            elif key == ord('0'):
                self._auto_fit_mode = True
                self._vehicle_centered = False
            
            # Vehicle-centered mode: 'v' or 'V'
            elif key in [ord('v'), ord('V')]:
                self._auto_fit_mode = False
                self._vehicle_centered = True
    
    # ==================== Update Methods (called from ROS callbacks) ====================
    
    def update_vehicle(self, x: float, y: float, yaw: float, speed: float):
        """Update vehicle pose and speed."""
        with self._lock:
            self._vehicle_pose = (x, y, yaw)
            self._vehicle_speed = speed
    
    def update_trajectory(self, waypoints: List[Tuple[float, float]]):
        """Update local trajectory."""
        with self._lock:
            self._trajectory = list(waypoints) if waypoints else None
    
    def update_global_route(self, waypoints: List[Tuple[float, float]]):
        """Update global route."""
        with self._lock:
            self._global_route = list(waypoints) if waypoints else None
    
    def update_predicted_trajectory(self, states: List[Tuple[float, float, float, float]]):
        """Update MPC predicted trajectory."""
        with self._lock:
            self._predicted_trajectory = list(states) if states else None
    
    def update_obstacles(self, obstacles: List[dict]):
        """
        Update obstacles.
        
        Args:
            obstacles: List of dicts with keys: x, y, yaw, length, width, type
        """
        with self._lock:
            self._obstacles = list(obstacles) if obstacles else []
