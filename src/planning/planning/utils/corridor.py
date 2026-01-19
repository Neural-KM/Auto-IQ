"""
Free-Space Corridor Generator
=============================

Converts a binary 2D occupancy grid (local costmap) into a smooth reference path
and safe lateral boundaries for a Frenet-based planner.

Pipeline:
1. Compute Medial Axis Transform (skeletonization) of free space
2. Extract and sort skeleton points by longitudinal distance
3. Fit a Cubic Spline parameterized by arc length s
4. Use ray casting to compute lateral boundaries at any point along the path

Usage:
    generator = CorridorGenerator(resolution=0.1)
    x_spline, y_spline, s_max = generator.generate_reference_line(grid)
    d_min, d_max = generator.get_constraints(s, x_spline, y_spline, grid)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from skimage.morphology import medial_axis
from typing import Tuple, Optional


class CorridorGenerator:
    """
    Generates free-space corridors from occupancy grids for Frenet-based planning.
    
    The corridor consists of:
    - A reference line (centerline) computed via the medial axis transform
    - Lateral boundaries computed via ray casting perpendicular to the reference
    
    Attributes:
        resolution: Grid cell size in meters (default: 0.1m)
        min_skeleton_points: Minimum points required for valid skeleton (default: 10)
        ray_step: Step size for ray casting in grid cells (default: 0.5)
        spline_smoothing: Number of points for spline smoothing (default: 50)
    """
    
    def __init__(
        self,
        resolution: float = 0.1,
        min_skeleton_points: int = 10,
        ray_step: float = 0.5,
        spline_smoothing: int = 50
    ):
        """
        Initialize the corridor generator.
        
        Args:
            resolution: Grid cell size in meters
            min_skeleton_points: Minimum skeleton points for valid reference line
            ray_step: Step size for ray casting (in grid cells)
            spline_smoothing: Number of points to use for spline fitting
        """
        self.resolution = resolution
        self.min_skeleton_points = min_skeleton_points
        self.ray_step = ray_step
        self.spline_smoothing = spline_smoothing
        
        # Cached data from last reference line generation
        self._last_grid = None
        self._last_skeleton = None
    
    def generate_reference_line(
        self,
        grid: np.ndarray,
        start_point: Optional[Tuple[float, float]] = None,
        goal_point: Optional[Tuple[float, float]] = None
    ) -> Tuple[CubicSpline, CubicSpline, float]:
        """
        Generate a smooth reference line from the occupancy grid.
        
        Computes the Medial Axis Transform (skeleton) of the free space and fits
        a cubic spline parameterized by arc length.
        
        Args:
            grid: 2D numpy array where 0 is free/drivable and non-zero is obstacle
            start_point: Optional (x, y) in grid coordinates to start the path
                         If None, uses the bottom-center of the grid
            goal_point: Optional (x, y) in grid coordinates for navigation target
                        Used to guide skeleton traversal at junctions/exits
        
        Returns:
            Tuple of (x_spline, y_spline, s_max):
                - x_spline: CubicSpline for x(s) in grid coordinates, or None on failure
                - y_spline: CubicSpline for y(s) in grid coordinates, or None on failure
                - s_max: Maximum arc length of the reference line (0 on failure)
        """
        # Create binary mask for medial_axis: 1 = free space (foreground), 0 = obstacle
        free_space = (grid == 0).astype(np.uint8)
        
        # Compute medial axis (skeleton) of free space
        skeleton, distance = medial_axis(free_space, return_distance=True)
        self._last_skeleton = skeleton
        self._last_grid = grid
        
        # Extract skeleton points (row, col format)
        skeleton_points = np.argwhere(skeleton)
        
        if len(skeleton_points) < self.min_skeleton_points:
            raise ValueError(
                f"Insufficient skeleton points: {len(skeleton_points)} < {self.min_skeleton_points}. "
                "Grid may be too small or too obstructed."
            )
        
        # Convert to (x, y) format: x = col, y = row
        points_xy = skeleton_points[:, ::-1].astype(float)  # (col, row) -> (x, y)
        
        # Determine start point for sorting
        if start_point is None:
            # Default: bottom-center of grid (vehicle typically starts here)
            start_point = np.array([grid.shape[1] / 2, grid.shape[0] - 1])
        else:
            start_point = np.array(start_point)
        
        # Sort points by distance along the path using greedy nearest neighbor
        sorted_points = self._sort_skeleton_points(points_xy, start_point, goal_point)
        
        
        # Subsample for spline fitting if we have enough points
        # This acts as another layer of smoothing and reduces spline oscillation
        if len(sorted_points) > self.spline_smoothing:
            # Subsample for smoother spline
            indices = np.linspace(0, len(sorted_points) - 1, self.spline_smoothing, dtype=int)
            sorted_points = sorted_points[indices]
        
        # Compute arc length parameterization
        arc_lengths = self._compute_arc_lengths(sorted_points)
        s_max = arc_lengths[-1]
        
        if s_max < 1e-6:
            raise ValueError("Arc length is too small, points may be coincident.")
        
        # Fit cubic splines for x(s) and y(s)
        x_spline = CubicSpline(arc_lengths, sorted_points[:, 0], bc_type='natural')
        y_spline = CubicSpline(arc_lengths, sorted_points[:, 1], bc_type='natural')
        
        return x_spline, y_spline, s_max
    
    def get_constraints(
        self,
        s: float,
        x_spline: CubicSpline,
        y_spline: CubicSpline,
        grid: np.ndarray,
        max_width: float = 50.0
    ) -> Tuple[float, float]:
        """
        Compute lateral driveable boundaries at a given arc length.
        
        Performs ray casting perpendicular to the reference line in both
        directions (left and right) to find obstacle boundaries.
        
        Args:
            s: Arc length along the reference line
            x_spline: CubicSpline for x(s) from generate_reference_line
            y_spline: CubicSpline for y(s) from generate_reference_line
            grid: 2D occupancy grid (same grid used for reference line)
            max_width: Maximum search distance for boundaries (grid cells)
        
        Returns:
            Tuple (d_min, d_max) in grid cell units:
                - d_min: Distance to left boundary (negative value)
                - d_max: Distance to right boundary (positive value)
            
            In Frenet coordinates, the driveable lateral range is [d_min, d_max]
        """
        # Get position on reference line
        x = float(x_spline(s))
        y = float(y_spline(s))
        
        # Compute tangent vector (first derivative)
        dx_ds = float(x_spline(s, 1))  # dx/ds
        dy_ds = float(y_spline(s, 1))  # dy/ds
        
        # Normalize tangent
        tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)
        if tangent_norm < 1e-9:
            # Fallback: use finite difference
            ds = 0.1
            dx_ds = float(x_spline(s + ds) - x_spline(s)) / ds
            dy_ds = float(y_spline(s + ds) - y_spline(s)) / ds
            tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)
        
        if tangent_norm < 1e-9:
            # Cannot compute normal, return conservative bounds
            return -1.0, 1.0
        
        tx = dx_ds / tangent_norm
        ty = dy_ds / tangent_norm
        
        # Normal vector (perpendicular, 90° counterclockwise from tangent)
        # Right-hand rule: normal points to the right of the path
        nx = ty   # Rotate tangent 90° clockwise for "right" direction
        ny = -tx
        
        # Ray cast in positive normal direction (right)
        d_max = self._ray_cast(x, y, nx, ny, grid, max_width)
        
        # Ray cast in negative normal direction (left)
        d_min = -self._ray_cast(x, y, -nx, -ny, grid, max_width)
        
        return d_min, d_max
    
    def get_corridor(
        self,
        x_spline: CubicSpline,
        y_spline: CubicSpline,
        s_max: float,
        grid: np.ndarray,
        num_points: int = 50,
        max_width: float = 50.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the full corridor along the reference line.
        
        Convenience method that computes constraints at evenly spaced points.
        
        Args:
            x_spline: CubicSpline for x(s)
            y_spline: CubicSpline for y(s)
            s_max: Maximum arc length
            grid: 2D occupancy grid
            num_points: Number of points to sample along the path
            max_width: Maximum search distance for boundaries
        
        Returns:
            Tuple (s_values, d_min_values, d_max_values):
                - s_values: Arc length values (num_points,)
                - d_min_values: Left boundary at each s (num_points,)
                - d_max_values: Right boundary at each s (num_points,)
        """
        s_values = np.linspace(0, s_max, num_points)
        d_min_values = np.zeros(num_points)
        d_max_values = np.zeros(num_points)
        
        for i, s in enumerate(s_values):
            d_min_values[i], d_max_values[i] = self.get_constraints(
                s, x_spline, y_spline, grid, max_width
            )
        
        return s_values, d_min_values, d_max_values
    
    def to_world_coordinates(
        self,
        grid_x: float,
        grid_y: float,
        origin_x: float,
        origin_y: float
    ) -> Tuple[float, float]:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            grid_x, grid_y: Position in grid coordinates
            origin_x, origin_y: World coordinates of grid origin (0, 0)
        
        Returns:
            (world_x, world_y) in meters
        """
        world_x = origin_x + grid_x * self.resolution
        world_y = origin_y + grid_y * self.resolution
        return world_x, world_y
    
    def to_grid_coordinates(
        self,
        world_x: float,
        world_y: float,
        origin_x: float,
        origin_y: float
    ) -> Tuple[float, float]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            world_x, world_y: Position in world coordinates (meters)
            origin_x, origin_y: World coordinates of grid origin
        
        Returns:
            (grid_x, grid_y) in grid cell units
        """
        grid_x = (world_x - origin_x) / self.resolution
        grid_y = (world_y - origin_y) / self.resolution
        return grid_x, grid_y
    # ==================== Private Methods ====================
    
    def _sort_skeleton_points(
        self,
        points: np.ndarray,
        start: np.ndarray,
        goal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Sort skeleton points along the path using goal-directed greedy traversal.
        
        Starts from the point closest to 'start' and iteratively adds the
        nearest unvisited point, with a bias toward the goal if provided.
        This ensures the path follows navigation intent at junctions/exits.
        """
        n = len(points)
        if n == 0:
            return points
        
        # Find starting point (closest to start)
        distances = np.linalg.norm(points - start, axis=1)
        start_idx = np.argmin(distances)
        
        # Greedy nearest neighbor traversal with goal bias
        visited = np.zeros(n, dtype=bool)
        sorted_indices = [start_idx]
        visited[start_idx] = True
        current_idx = start_idx
        
        for _ in range(n - 1):
            current_point = points[current_idx]
            
            # Compute distances to unvisited points
            unvisited_mask = ~visited
            unvisited_indices = np.where(unvisited_mask)[0]
            
            if len(unvisited_indices) == 0:
                break
            
            dists = np.linalg.norm(points[unvisited_indices] - current_point, axis=1)
            
            # Add goal-directed bias: prefer points that reduce distance to goal
            if goal is not None:
                goal_dists = np.linalg.norm(points[unvisited_indices] - goal, axis=1)
                current_goal_dist = np.linalg.norm(current_point - goal)
                # Reward points that get closer to goal (negative = closer)
                goal_progress = goal_dists - current_goal_dist
                # Combine: short distance + progress toward goal
                # Normalize goal_progress to similar scale as dists
                dists = dists + 0.5 * goal_progress
            
            # Select nearest (with goal bias applied)
            nearest_local_idx = np.argmin(dists)
            nearest_idx = unvisited_indices[nearest_local_idx]
            
            # Check for large jumps (discontinuities in medial axis)
            # If the nearest point is too far, it means we've reached the end of a branch
            # and are about to jump to a disconnected segment.
            # 5.0 grid cells = 1.0m at 0.2m resolution.
            dist_to_next = np.linalg.norm(points[nearest_idx] - current_point)
            if dist_to_next > 5.0:
                break
            
            sorted_indices.append(nearest_idx)
            visited[nearest_idx] = True
            current_idx = nearest_idx
        
        return points[sorted_indices]
    
    def _compute_arc_lengths(self, points: np.ndarray) -> np.ndarray:
        """
        Compute cumulative arc length along the path.
        
        Returns array where s[i] is the arc length from start to point i.
        """
        if len(points) < 2:
            return np.array([0.0])
        
        # Compute segment lengths
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        # Cumulative sum starting from 0
        arc_lengths = np.zeros(len(points))
        arc_lengths[1:] = np.cumsum(segment_lengths)
        
        return arc_lengths
    
    def _ray_cast(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        grid: np.ndarray,
        max_distance: float
    ) -> float:
        """
        Cast a ray from (x, y) in direction (dx, dy) until hitting an obstacle.
        
        Returns the distance traveled before hitting an obstacle or grid edge.
        """
        height, width = grid.shape
        distance = 0.0
        
        while distance < max_distance:
            # Compute current position along ray
            curr_x = x + distance * dx
            curr_y = y + distance * dy
            
            # Convert to grid indices
            col = int(round(curr_x))
            row = int(round(curr_y))
            
            # Check bounds
            if row < 0 or row >= height or col < 0 or col >= width:
                return distance  # Hit grid edge
            
            # Check for obstacle
            if grid[row, col] > 0:
                return distance  # Hit obstacle
            
            # Advance along ray
            distance += self.ray_step
        
        return max_distance
    
    def get_reference_path_points(
        self,
        x_spline: CubicSpline,
        y_spline: CubicSpline,
        s_max: float,
        num_points: int = 100
    ) -> np.ndarray:
        """
        Sample points along the reference path.
        
        Args:
            x_spline, y_spline: Spline functions from generate_reference_line
            s_max: Maximum arc length
            num_points: Number of points to sample
        
        Returns:
            Array of shape (num_points, 2) with (x, y) coordinates
        """
        s_values = np.linspace(0, s_max, num_points)
        x_values = x_spline(s_values)
        y_values = y_spline(s_values)
        return np.column_stack([x_values, y_values])
    
    def get_curvature(
        self,
        s: float,
        x_spline: CubicSpline,
        y_spline: CubicSpline
    ) -> float:
        """
        Compute the signed curvature at arc length s.
        
        Curvature κ = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
        
        Args:
            s: Arc length position
            x_spline, y_spline: Spline functions
        
        Returns:
            Signed curvature (1/radius, positive = turning left)
        """
        dx = float(x_spline(s, 1))
        dy = float(y_spline(s, 1))
        ddx = float(x_spline(s, 2))
        ddy = float(y_spline(s, 2))
        
        denom = (dx**2 + dy**2)**1.5
        if denom < 1e-9:
            return 0.0
        
        return (dx * ddy - dy * ddx) / denom
    
    def get_heading(
        self,
        s: float,
        x_spline: CubicSpline,
        y_spline: CubicSpline
    ) -> float:
        """
        Compute the heading angle at arc length s.
        
        Args:
            s: Arc length position
            x_spline, y_spline: Spline functions
        
        Returns:
            Heading angle in radians (0 = +x direction, π/2 = +y direction)
        """
        dx = float(x_spline(s, 1))
        dy = float(y_spline(s, 1))
        return np.arctan2(dy, dx)
