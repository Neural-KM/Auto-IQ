"""
Frenet Frame Utilities
======================

Utilities for Frenet coordinate conversion and polynomial trajectory generation.
Includes smoothing and robust projection for stable path following.
"""

import math
import numpy as np
import bisect
from scipy import interpolate
from scipy.ndimage import uniform_filter1d


class QuarticPolynomial:
    """
    Quartic Polynomial for Frenet Path Planning (Longitudinal)
    s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4
    """

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        if time < 1e-6:
            self.a3 = 0
            self.a4 = 0
            return

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        
        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
        except np.linalg.LinAlgError:
            self.a3 = 0
            self.a4 = 0

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2


class QuinticPolynomial:
    """
    Quintic Polynomial for Frenet Path Planning (Lateral)
    d(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    """

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        if time < 1e-6:
            self.a3 = 0
            self.a4 = 0
            self.a5 = 0
            return

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        
        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
        except np.linalg.LinAlgError:
            self.a3 = 0
            self.a4 = 0
            self.a5 = 0

    def calc_point(self, t):
        return self.a0 + self.a1 * t + self.a2 * t ** 2 + \
               self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

    def calc_first_derivative(self, t):
        return self.a1 + 2 * self.a2 * t + \
               3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

    def calc_second_derivative(self, t):
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3


class FrenetPath:
    """Container for a Frenet trajectory."""
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.x = []
        self.y = []
        self.yaw = []


class CubicSpline2D:
    """
    2D Cubic Spline for smooth reference path representation.
    
    Uses parametric splines for x(s) and y(s) to avoid issues
    with non-monotonic x values.
    """
    
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        
        # Compute arc length parameter
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.hypot(dx, dy)
        self.s = np.zeros(len(x))
        self.s[1:] = np.cumsum(ds)
        
        # Fit cubic splines
        # Use 'not-a-knot' boundary condition for smooth ends
        self.sx = interpolate.CubicSpline(self.s, x, bc_type='not-a-knot')
        self.sy = interpolate.CubicSpline(self.s, y, bc_type='not-a-knot')
        
    def calc_position(self, s):
        """Get (x, y) at arc length s."""
        s = np.clip(s, self.s[0], self.s[-1])
        return float(self.sx(s)), float(self.sy(s))
    
    def calc_yaw(self, s):
        """Get heading angle at arc length s."""
        s = np.clip(s, self.s[0], self.s[-1])
        dx = float(self.sx(s, 1))  # First derivative
        dy = float(self.sy(s, 1))
        return math.atan2(dy, dx)
    
    def calc_curvature(self, s):
        """Get curvature at arc length s."""
        s = np.clip(s, self.s[0], self.s[-1])
        dx = float(self.sx(s, 1))
        dy = float(self.sy(s, 1))
        ddx = float(self.sx(s, 2))
        ddy = float(self.sy(s, 2))
        
        denom = (dx**2 + dy**2)**1.5
        if abs(denom) < 1e-6:
            return 0.0
        return (dx * ddy - dy * ddx) / denom
    
    def get_total_length(self):
        """Get total arc length of the spline."""
        return self.s[-1]


class WaypointConverter:
    """
    Robust converter between Cartesian (x,y) and Frenet (s,d) coordinates.
    
    Uses cubic spline interpolation for smooth reference path and
    includes filtering to reduce noise in state estimation.
    """
    
    def __init__(self, waypoints_x, waypoints_y, smoothing_window=5):
        """
        Initialize with reference path waypoints.
        
        Args:
            waypoints_x: List of x coordinates
            waypoints_y: List of y coordinates  
            smoothing_window: Window size for path smoothing
        """
        wx = np.array(waypoints_x)
        wy = np.array(waypoints_y)
        
        # Remove duplicate points
        keep = [0]
        for i in range(1, len(wx)):
            if math.hypot(wx[i] - wx[keep[-1]], wy[i] - wy[keep[-1]]) > 0.1:
                keep.append(i)
        wx = wx[keep]
        wy = wy[keep]
        
        if len(wx) < 4:
            # Not enough points for cubic spline, use linear
            self.spline = None
            self.wx = wx
            self.wy = wy
            self.s_values = self._compute_s_values(wx, wy)
        else:
            # Smooth the path slightly to reduce waypoint noise
            if smoothing_window > 1 and len(wx) > smoothing_window:
                wx = uniform_filter1d(wx, size=min(smoothing_window, len(wx)//2))
                wy = uniform_filter1d(wy, size=min(smoothing_window, len(wy)//2))
            
            self.spline = CubicSpline2D(wx, wy)
            self.wx = wx
            self.wy = wy
            self.s_values = self.spline.s
        
        # State filtering
        self._prev_s = None
        self._prev_d = None
        self._filter_alpha = 0.3  # Low-pass filter coefficient (0-1, lower = more smoothing)
        
    def _compute_s_values(self, wx, wy):
        """Compute cumulative arc length for waypoints."""
        s = np.zeros(len(wx))
        for i in range(1, len(wx)):
            s[i] = s[i-1] + math.hypot(wx[i] - wx[i-1], wy[i] - wy[i-1])
        return s

    def xy_to_frenet(self, x, y, filter_output=True):
        """
        Convert Cartesian (x, y) to Frenet (s, d) coordinates.
        
        Uses Newton-Raphson iteration for accurate projection onto spline.
        Includes optional low-pass filtering for stable output.
        
        Args:
            x, y: Cartesian coordinates
            filter_output: Whether to apply smoothing filter
            
        Returns:
            (s, d): Frenet coordinates
        """
        if len(self.wx) == 0:
            return 0.0, 0.0
            
        if self.spline is None:
            return self._xy_to_frenet_linear(x, y)
        
        # Find closest point using search + refinement
        s, d = self._xy_to_frenet_spline(x, y)
        
        # Apply low-pass filter for stability
        if filter_output and self._prev_s is not None:
            # For s: only filter if not jumping (e.g., route reset)
            ds = s - self._prev_s
            if abs(ds) < 10.0:  # Reasonable continuity
                s = self._prev_s + self._filter_alpha * ds
            
            # For d: always filter to reduce lateral jitter
            d = self._prev_d + self._filter_alpha * (d - self._prev_d)
        
        self._prev_s = s
        self._prev_d = d
        
        return s, d
    
    def _xy_to_frenet_spline(self, x, y):
        """Project point onto spline using search + Newton refinement."""
        # Coarse search: sample spline and find closest
        # Only search within +-10 meters of previous s if available
        total_length = self.spline.get_total_length()
        search_radius = 10.0  # meters
        
        if self._prev_s is not None:
            s_min = max(0, self._prev_s - search_radius)
            s_max = min(total_length, self._prev_s + search_radius)
        else:
            s_min = 0
            s_max = total_length
        
        n_samples = min(100, len(self.wx) * 2)
        s_samples = np.linspace(s_min, s_max, n_samples)
        
        min_dist = float('inf')
        best_s = s_min
        
        for s_test in s_samples:
            px, py = self.spline.calc_position(s_test)
            dist = (x - px)**2 + (y - py)**2
            if dist < min_dist:
                min_dist = dist
                best_s = s_test
        
        # Refine with Newton-Raphson (a few iterations)
        s = best_s
        for _ in range(5):
            px, py = self.spline.calc_position(s)
            dx_ds = float(self.spline.sx(s, 1))
            dy_ds = float(self.spline.sy(s, 1))
            
            # Error vector from spline point to query point
            ex = x - px
            ey = y - py
            
            # Project error onto tangent direction
            tangent_proj = ex * dx_ds + ey * dy_ds
            tangent_norm = dx_ds**2 + dy_ds**2
            
            if tangent_norm < 1e-6:
                break
                
            ds = tangent_proj / tangent_norm
            s = np.clip(s + ds, 0, self.spline.get_total_length())
            
            if abs(ds) < 0.01:
                break
        
        # Compute d (signed lateral distance)
        px, py = self.spline.calc_position(s)
        yaw = self.spline.calc_yaw(s)
        
        # Vector from spline to query point
        dx = x - px
        dy = y - py
        
        # d is positive to the left of the path
        # Use cross product: tangent x (point - spline_point)
        # tangent = (cos(yaw), sin(yaw))
        d = -math.sin(yaw) * dx + math.cos(yaw) * dy
        
        return s, d
    
    def _xy_to_frenet_linear(self, x, y):
        """Fallback linear interpolation for short paths."""
        # Find closest segment
        min_dist = float('inf')
        best_s = 0
        best_d = 0
        
        for i in range(len(self.wx) - 1):
            x0, y0 = self.wx[i], self.wy[i]
            x1, y1 = self.wx[i+1], self.wy[i+1]
            
            # Project onto segment
            dx = x1 - x0
            dy = y1 - y0
            seg_len_sq = dx*dx + dy*dy
            
            if seg_len_sq < 1e-6:
                continue
                
            t = max(0, min(1, ((x - x0)*dx + (y - y0)*dy) / seg_len_sq))
            
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            
            dist = math.hypot(x - proj_x, y - proj_y)
            
            if dist < min_dist:
                min_dist = dist
                seg_len = math.sqrt(seg_len_sq)
                best_s = self.s_values[i] + t * seg_len
                
                # Signed distance
                cross = dx * (y - y0) - dy * (x - x0)
                best_d = math.copysign(dist, cross)
        
        return best_s, best_d

    def frenet_to_xy(self, s, d):
        """
        Convert Frenet (s, d) to Cartesian (x, y, yaw).
        
        Args:
            s: Arc length along reference path
            d: Lateral offset (positive = left)
            
        Returns:
            (x, y, yaw): Cartesian position and heading
        """
        if self.spline is None:
            return self._frenet_to_xy_linear(s, d)
        
        # Clamp s to valid range
        s = np.clip(s, 0, self.spline.get_total_length())
        
        # Get reference point and heading
        rx, ry = self.spline.calc_position(s)
        yaw = self.spline.calc_yaw(s)
        
        # Normal vector (perpendicular to tangent, pointing left)
        nx = -math.sin(yaw)
        ny = math.cos(yaw)
        
        # Apply lateral offset
        x = rx + d * nx
        y = ry + d * ny
        
        return x, y, yaw
    
    def _frenet_to_xy_linear(self, s, d):
        """Fallback linear interpolation for short paths."""
        # Find segment
        idx = bisect.bisect_right(self.s_values, s) - 1
        idx = max(0, min(idx, len(self.s_values) - 2))
        
        x0, y0 = self.wx[idx], self.wy[idx]
        x1, y1 = self.wx[idx+1], self.wy[idx+1]
        s0, s1 = self.s_values[idx], self.s_values[idx+1]
        
        # Interpolation ratio
        if s1 - s0 < 1e-6:
            t = 0
        else:
            t = (s - s0) / (s1 - s0)
        
        # Reference point
        rx = x0 + t * (x1 - x0)
        ry = y0 + t * (y1 - y0)
        
        # Heading
        yaw = math.atan2(y1 - y0, x1 - x0)
        
        # Normal and offset
        nx = -math.sin(yaw)
        ny = math.cos(yaw)
        
        x = rx + d * nx
        y = ry + d * ny
        
        return x, y, yaw
    
    def reset_filter(self):
        """Reset the internal filter state."""
        self._prev_s = None
        self._prev_d = None
