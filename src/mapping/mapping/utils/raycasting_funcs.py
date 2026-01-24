import numpy as np
from numba import jit
@jit(nopython=True, cache=True)
def bresenham_line_fast(x0, y0, x1, y1):
    """JIT-compiled Bresenham algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    # Pre-allocate output arrays
    # Max length of a line is dx + dy
    max_len = dx + dy + 1
    x_arr = np.empty(max_len, dtype=np.int32)
    y_arr = np.empty(max_len, dtype=np.int32)
    
    x, y = x0, y0
    idx = 0
    
    while True:
        if x == x1 and y == y1:
            break
            
        x_arr[idx] = x
        y_arr[idx] = y
        idx += 1
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
            
    return x_arr[:idx], y_arr[:idx]

@jit(nopython=True, cache=True)
def compute_raycast_batch(robot_ix, robot_iy, hit_ix_arr, hit_iy_arr, ray_mask, max_ray_len):
    """
    Batch process all raycasts in a single JIT-compiled loop.
    Avoids Python interpreter overhead for thousands of rays.
    """
    n_hits = len(hit_ix_arr)
    
    # Heuristic allocation: assume avg ray is half of max range
    # 5000 hits * 100 cells = 500k ints = ~2MB RAM (Safe to allocate)
    est_size = n_hits * max_ray_len
    
    all_free_x = np.empty(est_size, dtype=np.int32)
    all_free_y = np.empty(est_size, dtype=np.int32)
    
    count = 0
    
    for i in range(n_hits):
        # Skip if mask is false (too far)
        if not ray_mask[i]:
            continue
            
        hx = hit_ix_arr[i]
        hy = hit_iy_arr[i]
        
        # Call JIT bresenham
        xs, ys = bresenham_line_fast(robot_ix, robot_iy, hx, hy)
        n_pts = len(xs)
        
        # Safety check for buffer overflow (rare, but good practice)
        if count + n_pts >= len(all_free_x):
             break # Or handle resize, but with generous est_size this won't happen
            
        all_free_x[count : count + n_pts] = xs
        all_free_y[count : count + n_pts] = ys
        count += n_pts
        
    return all_free_x[:count], all_free_y[:count]


@jit(nopython=True, cache=True)
def compute_shadow_buffer(obstacle_x: np.ndarray, obstacle_y: np.ndarray,
                          robot_x: float, robot_y: float, 
                          angular_res: float) -> np.ndarray:
    """
    Build an angular depth buffer (shadow buffer) from obstacle points.
    
    For each angular bin (360°), stores the minimum distance to an obstacle.
    This creates a "shadow" behind each obstacle that will block raycasting.
    
    Args:
        obstacle_x: X coordinates of obstacle points (world frame)
        obstacle_y: Y coordinates of obstacle points (world frame)
        robot_x, robot_y: Robot position (world frame)
        angular_res: Angular resolution in radians (e.g., 1° = 0.0175)
        
    Returns:
        1D array of size (2*pi / angular_res) with minimum distance per bin.
        Bins with no obstacles have value np.inf.
    """
    n_bins = int(2.0 * np.pi / angular_res) + 1
    shadow_buffer = np.full(n_bins, np.inf, dtype=np.float32)
    
    n_obstacles = len(obstacle_x)
    
    for i in range(n_obstacles):
        dx = obstacle_x[i] - robot_x
        dy = obstacle_y[i] - robot_y
        
        # Calculate angle and distance
        angle = np.arctan2(dy, dx)  # [-pi, pi]
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Convert angle to bin index [0, n_bins)
        bin_idx = int((angle + np.pi) / angular_res)
        
        # Keep minimum distance (closest obstacle)
        if distance < shadow_buffer[bin_idx]:
            shadow_buffer[bin_idx] = distance
    
    return shadow_buffer


@jit(nopython=True, cache=True)
def filter_rays_by_shadows(target_x: np.ndarray, target_y: np.ndarray,
                           robot_x: float, robot_y: float,
                           shadow_buffer: np.ndarray, angular_res: float,
                           tolerance: float) -> np.ndarray:
    """
    Filter target points that are occluded by obstacles (behind the shadow).
    
    A point is occluded if its distance is greater than the shadow buffer
    distance for its angular bin (plus a small tolerance).
    
    Args:
        target_x, target_y: Target point coordinates (world frame)
        robot_x, robot_y: Robot position (world frame)
        shadow_buffer: Angular depth buffer from compute_shadow_buffer
        angular_res: Angular resolution in radians (must match shadow buffer)
        tolerance: Distance tolerance in meters (points within tolerance pass through)
        
    Returns:
        Boolean mask - True for points that should be raycasted (not occluded)
    """
    n_points = len(target_x)
    n_bins = len(shadow_buffer)
    valid_mask = np.ones(n_points, dtype=np.bool_)
    
    for i in range(n_points):
        dx = target_x[i] - robot_x
        dy = target_y[i] - robot_y
        
        # Calculate angle and distance
        angle = np.arctan2(dy, dx)  # [-pi, pi]
        distance = np.sqrt(dx * dx + dy * dy)
        
        # Convert angle to bin index
        bin_idx = int((angle + np.pi) / angular_res)
        
        # Check if point is behind an obstacle (occluded)
        # If target distance > shadow distance + tolerance, it's behind an obstacle
        if distance > shadow_buffer[bin_idx] + tolerance:
            valid_mask[i] = False
    
    return valid_mask