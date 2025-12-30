"""
Live 2D visualizer using Matplotlib.

This visualizer opens a window in a separate thread and draws:
- Current vehicle pose (triangle with heading)
- Waypoints (blue)
- Reference trajectory (cyan)
- Predicted MPC trajectory (green) if provided
- Ground truth obstacles (green)
- Perceived obstacles (red)

Usage: 
- Create a visualizer with a WorldState reference.
- Call `visualizer.start()` once. 
- Call `visualizer.update(waypoints, reference, predicted)` each tick.

The visualizer runs a timed loop at ~10Hz to keep the window responsive.
"""

import threading
import time
import warnings
import numpy as np
from typing import Optional

from state.world_state import WorldState

# Use Agg backend to avoid thread issues - must be set before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2

class Live2DVisualizer:
    def __init__(self, world_state: WorldState = None, title="Auto-IQ Live 2D Visualizer", refresh_hz=10):
        self.title = title
        self.refresh_hz = refresh_hz
        self.world_state = world_state
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # Data stores for drawing (waypoints, reference, predicted still passed via update)
        self._waypoints = None  # list of (x,y)
        self._reference = None  # Nx4 array
        self._predicted = None  # Nx4 array

        # Plot handles
        self._fig = None
        self._ax = None
        self._vehicle_patch = None
        self._waypoints_line = None
        self._reference_line = None
        self._predicted_line = None
        self._obstacle_patches = []  # list of Polygon patches for obstacles
    
    def set_world_state(self, world_state: WorldState):
        """Set or update the world state reference."""
        self.world_state = world_state

    def start(self):
        if plt is None:
            print("[Live2DVisualizer] Matplotlib not available - visualizer disabled")
            return False

        if self._thread and self._thread.is_alive():
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self):
        # Initialize plot with Agg backend (thread-safe)
        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111)
        self._fig.canvas = FigureCanvasAgg(self._fig)
        
        self._ax.set_aspect('equal', adjustable='datalim')
        self._ax.grid(True)
        self._ax.set_xlabel('X (m)')
        self._ax.set_ylabel('Y (m)')
        self._ax.set_title(self.title)

        # Initialize empty lines/patches
        self._waypoints_line, = self._ax.plot([], [], 'o-', color='blue', label='Waypoints')
        self._reference_line, = self._ax.plot([], [], '-.', color='cyan', label='Reference')
        self._predicted_line, = self._ax.plot([], [], '-', color='green', label='MPC Predicted')
        self._vehicle_patch = Polygon([[0,0],[0,0],[0,0]], color='red')
        self._ax.add_patch(self._vehicle_patch)
        # Obstacle patches will be created dynamically
        self._obstacle_patches = []
        self._ax.legend()

        last_update = 0
        while self._running:
            t = time.time()
            if t - last_update >= 1.0 / self.refresh_hz:
                self._draw_once()
                last_update = t
            # Use short sleep for responsiveness
            time.sleep(0.01)

        # Cleanup within this thread
        plt.close(self._fig)
        cv2.destroyWindow(self.title)
        cv2.waitKey(1)  # Process the destroy event

    def _draw_once(self):
        with self._lock:
            waypoints = self._waypoints
            reference = self._reference
            predicted = self._predicted
        
        # Get vehicle state from world_state
        vehicle = None
        if self.world_state and self.world_state.vehicle_state:
            vs = self.world_state.vehicle_state
            vehicle = (vs.x, vs.y, vs.yaw)

        # Update bounds
        xs = []
        ys = []
        if waypoints:
            ws = np.array(waypoints)
            xs.extend(ws[:, 0].tolist())
            ys.extend(ws[:, 1].tolist())
            self._waypoints_line.set_data(ws[:, 0], ws[:, 1])
        else:
            self._waypoints_line.set_data([], [])

        if reference is not None and len(reference) > 0:
            ref = np.array(reference)
            xs.extend(ref[:, 0].tolist())
            ys.extend(ref[:, 1].tolist())
            self._reference_line.set_data(ref[:, 0], ref[:, 1])
        else:
            self._reference_line.set_data([], [])

        if predicted is not None and len(predicted) > 0:
            pred = np.array(predicted)
            xs.extend(pred[:, 0].tolist())
            ys.extend(pred[:, 1].tolist())
            self._predicted_line.set_data(pred[:, 0], pred[:, 1])
        else:
            self._predicted_line.set_data([], [])

        # Draw obstacles
        # Remove old obstacle patches
        for patch in self._obstacle_patches:
            patch.remove()
        self._obstacle_patches = []

        # Draw obstacles directly from world_state
        if self.world_state and vehicle is not None:
            vx, vy, vpsi = vehicle[0], vehicle[1], vehicle[2]
            c, s = np.cos(vpsi), np.sin(vpsi)
            R = np.array([[c, -s], [s, c]])
            
            # Draw ground truth obstacles (green)
            for obs in self.world_state.ground_truth_obstacles:
                pts = self._draw_obstacle(obs, vx, vy, c, s, R, 'green' if obs.obstacle_type == 'vehicle' else 'lime')
                if pts is not None:
                    xs.extend(pts[:, 0].tolist())
                    ys.extend(pts[:, 1].tolist())
            
            # Draw perceived obstacles (red)
            for obs in self.world_state.perceived_obstacles:
                pts = self._draw_obstacle(obs, vx, vy, c, s, R, 'red' if obs.obstacle_type == 'vehicle' else 'coral')
                if pts is not None:
                    xs.extend(pts[:, 0].tolist())
                    ys.extend(pts[:, 1].tolist())

        if vehicle is not None:
            vx, vy, vpsi = vehicle[0], vehicle[1], vehicle[2]
            # Draw a triangular vehicle for heading
            l = 1.5  # vehicle length
            w = 0.8  # vehicle width
            pts = np.array([
                [l*0.5, 0],
                [-l*0.5, -w*0.5],
                [-l*0.5, w*0.5],
            ])
            # Rotate and translate
            c, s = np.cos(vpsi), np.sin(vpsi)
            R = np.array([[c, -s], [s, c]])
            pts = (pts.dot(R.T)) + np.array([vx, vy])
            self._vehicle_patch.set_xy(pts)
            xs.extend(pts[:, 0].tolist())
            ys.extend(pts[:, 1].tolist())
        else:
            self._vehicle_patch.set_xy([[0, 0], [0, 0], [0, 0]])

        if len(xs) > 0 and len(ys) > 0:
            self._ax.relim()
            self._ax.autoscale_view()

        try:
            # Render to buffer using Agg backend
            self._fig.canvas.draw()
            
            # Get the RGBA buffer from the figure
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            # Convert RGBA to BGR for OpenCV
            img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
            cv2.imshow(self.title, img)
            cv2.waitKey(1)
        except Exception:
            pass

    def _draw_obstacle(self, obs, vx, vy, c, s, R, color):
        """
        Draw a single obstacle and return its world-frame corner points.
        
        Args:
            obs: Obstacle object with x, y, length, width, yaw attributes
            vx, vy: Vehicle world position
            c, s: cos and sin of vehicle heading (for position transform)
            R: Rotation matrix (unused, kept for API compatibility)
            color: Color for the obstacle patch
            
        Returns:
            Corner points in world frame, or None if obstacle is invalid
        """
        # Obstacle coordinates are ego-relative (x=forward, y=left)
        ox_ego, oy_ego = obs.x, obs.y
        ol, ow = obs.length, obs.width
        
        # Transform ego-relative position to world coordinates
        ox_world = vx + ox_ego * c - oy_ego * s
        oy_world = vy + ox_ego * s + oy_ego * c
        
        # Compute obstacle's world yaw (ego yaw + relative yaw)
        ego_yaw = np.arctan2(s, c)
        obs_world_yaw = ego_yaw + obs.yaw
        
        # Create rotation matrix for obstacle's own orientation
        c_obs, s_obs = np.cos(obs_world_yaw), np.sin(obs_world_yaw)
        R_obs = np.array([[c_obs, -s_obs], [s_obs, c_obs]])
        
        # Create rectangle for obstacle in world coordinates
        half_l, half_w = ol / 2, ow / 2
        local_pts = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])
        pts = local_pts.dot(R_obs.T) + np.array([ox_world, oy_world])
        
        obs_patch = Polygon(pts, color=color, alpha=0.7)
        self._ax.add_patch(obs_patch)
        self._obstacle_patches.append(obs_patch)
        return pts

    def update(self, waypoints=None, reference=None, predicted=None):
        """
        Update visualization data.
        
        Vehicle state and obstacles are read directly from world_state.
        """
        with self._lock:
            if waypoints is not None:
                self._waypoints = [(float(w[0]), float(w[1])) for w in waypoints]
            if reference is not None:
                self._reference = [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in reference]
            if predicted is not None:
                self._predicted = [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in predicted]


# Module-level convenience instance
visualizer = Live2DVisualizer()


def start():
    visualizer.start()


def stop():
    visualizer.stop()


def update(waypoints=None, reference=None, predicted=None):
    visualizer.update(waypoints, reference, predicted)
