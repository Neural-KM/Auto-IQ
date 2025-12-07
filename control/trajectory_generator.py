"""
Trajectory Generator
====================

Generates smooth reference trajectories from sparse waypoints using spline interpolation.
Converts A* waypoints into dense, differentiable paths suitable for MPC.

Output: Dense reference states [x_ref, y_ref, psi_ref, v_ref] for prediction horizon.
"""

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.ndimage import gaussian_filter1d


class TrajectoryGenerator:
    """
    Generates smooth trajectories from sparse waypoints.
    
    Takes waypoints from global planner (A*) and produces:
    1. Smooth path via cubic spline interpolation
    2. Heading angles computed from path tangent
    3. Velocity profile based on curvature
    """
    
    def __init__(self, target_speed=30.0, min_speed=5.0, max_curvature_speed=15.0):
        """
        Initialize trajectory generator.
        
        Args:
            target_speed: Default target speed in km/h
            min_speed: Minimum speed in km/h (for sharp turns)
            max_curvature_speed: Speed at maximum curvature in km/h
        """
        self.target_speed = target_speed / 3.6  # Convert to m/s
        self.min_speed = min_speed / 3.6
        self.max_curvature_speed = max_curvature_speed / 3.6
        
        # Spline parameters
        self.spline_smoothing = 0.1  # Smoothing factor
        self.sample_spacing = 0.5    # meters between reference points
        
        # Cached spline
        self._spline = None
        self._spline_length = 0
        self._waypoints = None
        
    def set_waypoints(self, waypoints):
        """
        Set waypoints and compute spline.
        
        Args:
            waypoints: List of carla.Waypoint or (x, y) tuples or carla.Location
        """
        # Extract x, y coordinates
        points = []
        for wp in waypoints:
            if hasattr(wp, 'transform'):
                # carla.Waypoint
                points.append((wp.transform.location.x, wp.transform.location.y))
            elif hasattr(wp, 'x'):
                # carla.Location or similar
                points.append((wp.x, wp.y))
            else:
                # Tuple
                points.append((wp[0], wp[1]))
        
        if len(points) < 4:
            print(f"[TrajectoryGen] Not enough waypoints ({len(points)}), need at least 4")
            self._spline = None
            return False
        
        self._waypoints = np.array(points)
        
        # Fit parametric spline
        try:
            # Remove duplicate points
            unique_points = [self._waypoints[0]]
            for p in self._waypoints[1:]:
                if np.linalg.norm(p - unique_points[-1]) > 0.1:
                    unique_points.append(p)
            
            if len(unique_points) < 4:
                print("[TrajectoryGen] Not enough unique waypoints after deduplication")
                return False
            
            self._waypoints = np.array(unique_points)
            
            # Parametric spline (u parameter from 0 to 1)
            tck, u = splprep([self._waypoints[:, 0], self._waypoints[:, 1]], 
                            s=self.spline_smoothing, k=min(3, len(self._waypoints)-1))
            self._spline = tck
            
            # Compute approximate path length
            u_dense = np.linspace(0, 1, 1000)
            x_dense, y_dense = splev(u_dense, tck)
            distances = np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2)
            self._spline_length = np.sum(distances)
            
            print(f"[TrajectoryGen] Spline fitted: {len(self._waypoints)} waypoints, "
                  f"{self._spline_length:.1f}m path length")
            return True
            
        except Exception as e:
            print(f"[TrajectoryGen] Spline fitting failed: {e}")
            self._spline = None
            return False
    
    def generate_reference(self, current_pos, horizon_length, dt, target_speed=None):
        """
        Generate reference trajectory for MPC horizon.
        
        Args:
            current_pos: Current position (x, y) or [x, y, psi, v]
            horizon_length: Number of steps in prediction horizon (N)
            dt: Time step (seconds)
            target_speed: Override target speed (m/s), None uses default
            
        Returns:
            Reference trajectory as numpy array [N+1, 4] with columns [x, y, psi, v]
            Returns None if no valid trajectory
        """
        if self._spline is None:
            return None
        
        # Get current position
        if len(current_pos) >= 2:
            curr_x, curr_y = current_pos[0], current_pos[1]
        else:
            return None
        
        speed = target_speed if target_speed else self.target_speed
        
        # Find closest point on spline
        u_start = self._find_closest_u(curr_x, curr_y)
        
        # Generate reference points along spline
        total_distance = speed * dt * (horizon_length + 5)  # Extra buffer
        
        # Sample points along path
        reference = []
        u_current = u_start
        
        for i in range(horizon_length + 1):
            # Get position at current u
            if u_current >= 1.0:
                u_current = 0.9999  # Clamp to end
            
            x, y = splev(u_current, self._spline)
            
            # Compute heading from spline tangent
            dx, dy = splev(u_current, self._spline, der=1)
            psi = np.arctan2(dy, dx)
            
            # Compute curvature for velocity profile
            d2x, d2y = splev(u_current, self._spline, der=2)
            curvature = abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-6)**1.5
            
            # Velocity based on curvature (slow down for sharp turns)
            v_ref = self._speed_from_curvature(curvature, speed)
            
            reference.append([float(x), float(y), float(psi), float(v_ref)])
            
            # Advance along spline by distance = v * dt
            step_distance = v_ref * dt
            u_current = self._advance_u(u_current, step_distance)
        
        return np.array(reference)
    
    def _find_closest_u(self, x, y, num_samples=100):
        """Find spline parameter u closest to point (x, y)."""
        u_samples = np.linspace(0, 1, num_samples)
        x_samples, y_samples = splev(u_samples, self._spline)
        
        distances = (x_samples - x)**2 + (y_samples - y)**2
        closest_idx = np.argmin(distances)
        
        # Refine search around closest point
        if closest_idx > 0 and closest_idx < num_samples - 1:
            u_fine = np.linspace(u_samples[closest_idx - 1], 
                                u_samples[closest_idx + 1], 50)
            x_fine, y_fine = splev(u_fine, self._spline)
            distances_fine = (x_fine - x)**2 + (y_fine - y)**2
            return u_fine[np.argmin(distances_fine)]
        
        return u_samples[closest_idx]
    
    def _advance_u(self, u_start, distance):
        """Advance u parameter by approximately 'distance' meters along spline."""
        if distance <= 0:
            return u_start
        
        # Approximate: du ≈ distance / path_length
        du_approx = distance / (self._spline_length + 1e-6)
        u_new = u_start + du_approx
        
        return min(u_new, 1.0)
    
    def _speed_from_curvature(self, curvature, max_speed):
        """Compute safe speed based on curvature."""
        # Higher curvature = lower speed
        # curvature of 0.1 rad/m is a ~10m radius turn
        if curvature < 0.01:
            return max_speed
        elif curvature > 0.2:
            return self.min_speed
        else:
            # Linear interpolation
            t = (curvature - 0.01) / (0.2 - 0.01)
            return max_speed * (1 - t) + self.min_speed * t
    
    def get_path_points(self, num_points=100):
        """Get dense path points for visualization."""
        if self._spline is None:
            return None
        
        u = np.linspace(0, 1, num_points)
        x, y = splev(u, self._spline)
        return np.column_stack([x, y])
    
    def get_remaining_distance(self, current_pos):
        """Get remaining distance to end of path."""
        if self._spline is None:
            return float('inf')
        
        u = self._find_closest_u(current_pos[0], current_pos[1])
        return self._spline_length * (1 - u)


class SimpleTrajectoryGenerator:
    """
    Simpler trajectory generator using linear interpolation.
    Fallback when spline fitting fails or for short paths.
    """
    
    def __init__(self, target_speed=30.0):
        """
        Args:
            target_speed: Target speed in km/h
        """
        self.target_speed = target_speed / 3.6  # m/s
        self._waypoints = None
        self._cumulative_dist = None
        
    def set_waypoints(self, waypoints):
        """Set waypoints from global planner."""
        points = []
        for wp in waypoints:
            if hasattr(wp, 'transform'):
                points.append([wp.transform.location.x, wp.transform.location.y])
            elif hasattr(wp, 'x'):
                points.append([wp.x, wp.y])
            else:
                points.append([wp[0], wp[1]])
        
        if len(points) < 2:
            return False
        
        self._waypoints = np.array(points)
        
        # Compute cumulative distances
        dists = [0.0]
        for i in range(1, len(self._waypoints)):
            d = np.linalg.norm(self._waypoints[i] - self._waypoints[i-1])
            dists.append(dists[-1] + d)
        self._cumulative_dist = np.array(dists)
        
        return True
    
    def generate_reference(self, current_pos, horizon_length, dt, target_speed=None):
        """Generate reference trajectory by interpolating waypoints."""
        if self._waypoints is None or len(self._waypoints) < 2:
            return None
        
        speed = target_speed if target_speed else self.target_speed
        curr_x, curr_y = current_pos[0], current_pos[1]
        
        # Find closest waypoint
        distances = np.linalg.norm(self._waypoints - [curr_x, curr_y], axis=1)
        closest_idx = np.argmin(distances)
        
        # Generate reference points
        reference = []
        dist_along_path = self._cumulative_dist[closest_idx]
        
        for i in range(horizon_length + 1):
            # Find waypoint at this distance
            target_dist = dist_along_path + speed * dt * i
            
            # Interpolate position
            if target_dist >= self._cumulative_dist[-1]:
                # Past end of path
                x, y = self._waypoints[-1]
                if len(self._waypoints) >= 2:
                    dx = self._waypoints[-1, 0] - self._waypoints[-2, 0]
                    dy = self._waypoints[-1, 1] - self._waypoints[-2, 1]
                    psi = np.arctan2(dy, dx)
                else:
                    psi = 0.0
            else:
                # Find segment
                idx = np.searchsorted(self._cumulative_dist, target_dist) - 1
                idx = max(0, min(idx, len(self._waypoints) - 2))
                
                # Interpolate within segment
                seg_start = self._cumulative_dist[idx]
                seg_end = self._cumulative_dist[idx + 1]
                t = (target_dist - seg_start) / (seg_end - seg_start + 1e-6)
                t = np.clip(t, 0, 1)
                
                x = self._waypoints[idx, 0] * (1-t) + self._waypoints[idx+1, 0] * t
                y = self._waypoints[idx, 1] * (1-t) + self._waypoints[idx+1, 1] * t
                
                dx = self._waypoints[idx+1, 0] - self._waypoints[idx, 0]
                dy = self._waypoints[idx+1, 1] - self._waypoints[idx, 1]
                psi = np.arctan2(dy, dx)
            
            reference.append([x, y, psi, speed])
        
        return np.array(reference)


# Test
if __name__ == '__main__':
    # Create some test waypoints (curved path)
    t = np.linspace(0, 2*np.pi, 20)
    waypoints = [(10*np.cos(t_i), 10*np.sin(t_i)) for t_i in t]
    
    gen = TrajectoryGenerator(target_speed=30.0)
    gen.set_waypoints(waypoints)
    
    # Generate reference from starting position
    ref = gen.generate_reference([10, 0], horizon_length=20, dt=0.1)
    
    if ref is not None:
        print("Generated reference trajectory:")
        print(f"  Shape: {ref.shape}")
        print(f"  First point: x={ref[0,0]:.2f}, y={ref[0,1]:.2f}, "
              f"psi={np.degrees(ref[0,2]):.1f}°, v={ref[0,3]:.1f}m/s")
        print(f"  Last point: x={ref[-1,0]:.2f}, y={ref[-1,1]:.2f}, "
              f"psi={np.degrees(ref[-1,2]):.1f}°, v={ref[-1,3]:.1f}m/s")
