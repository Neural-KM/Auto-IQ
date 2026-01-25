from numba import jit, prange
import numpy as np
@jit(nopython=True, parallel=True, cache=True)
def reverse_project_grid(
    grid_rows: int,
    grid_cols: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
    R_base2cam: np.ndarray,
    t_base2cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    semantic_img: np.ndarray,
    road_label: int,
    road_line_label: int,
    UNKNOWN: int,
    FREE: int,
    OCCUPIED: int,
) -> np.ndarray:
    """
    Reverse projection: iterate over grid cells and sample camera image.
    
    CARLA camera_link frame: X=forward, Y=left, Z=up
    Optical frame (for pinhole): X=right, Y=down, Z=forward
    
    Conversion: optical = [link_Y negated, link_Z negated, link_X]
                optical_x = -link_y
                optical_y = -link_z  
                optical_z = link_x
    
    For each grid cell:
    1. Grid to Base: Convert (r, c) to vehicle coordinates
    2. Base to Camera Link: Transform point to camera_link frame
    3. Camera Link to Optical: Convert frame conventions
    4. Check Z: If Z_optical <= 0, point is behind camera -> UNKNOWN
    5. Project to Pixel: Use intrinsics to get (u, v)
    6. Bounds Check & Sample: If in bounds, sample semantic image
    
    Args:
        grid_rows: Number of rows in grid
        grid_cols: Number of columns in grid
        origin_x: Grid origin X in base frame (meters)
        origin_y: Grid origin Y in base frame (meters)
        resolution: Grid resolution (meters per cell)
        R_base2cam: Rotation matrix from base_link to camera_link [3x3]
        t_base2cam: Translation vector from base_link to camera_link [3]
        fx, fy: Focal lengths
        cx, cy: Principal point
        img_height, img_width: Image dimensions
        semantic_img: Semantic label image [H, W]
        road_label: Label ID for road
        road_line_label: Label ID for road line
        
    Returns:
        grid: Occupancy grid [grid_rows, grid_cols] with values UNKNOWN/FREE/OCCUPIED
    """
    grid = np.full((grid_rows, grid_cols), UNKNOWN, dtype=np.int8)
    
    # Use prange for parallel execution over rows
    for r in prange(grid_rows):
        for c in range(grid_cols):
            # 1. Grid to Base Frame
            # Grid cell (r, c) -> base_link coordinates (x, y, z=0)
            # c is along X (forward), r is along Y (left)
            x_base = origin_x + c * resolution
            y_base = origin_y + r * resolution
            z_base = 0.0  # Ground plane
            
            # 2. Base to Camera Link Frame
            # P_link = R * P_base + t
            x_link = R_base2cam[0, 0] * x_base + R_base2cam[0, 1] * y_base + R_base2cam[0, 2] * z_base + t_base2cam[0]
            y_link = R_base2cam[1, 0] * x_base + R_base2cam[1, 1] * y_base + R_base2cam[1, 2] * z_base + t_base2cam[1]
            z_link = R_base2cam[2, 0] * x_base + R_base2cam[2, 1] * y_base + R_base2cam[2, 2] * z_base + t_base2cam[2]
            
            # 3. Camera Link to Optical Frame conversion
            # Link: X=forward, Y=left, Z=up
            # Optical: X=right, Y=down, Z=forward
            # optical_x = -link_y (right = -left)
            # optical_y = -link_z (down = -up)
            # optical_z = link_x  (forward = forward)
            x_optical = -y_link
            y_optical = -z_link
            z_optical = x_link
            
            # 4. Check Z: Point must be in front of camera (Z_optical > 0)
            if z_optical <= 0.0:
                # Point is behind camera
                continue  # Already UNKNOWN
            
            # 5. Project to Pixel using pinhole model
            # u = fx * (x/z) + cx
            # v = fy * (y/z) + cy
            u = fx * (x_optical / z_optical) + cx
            v = fy * (y_optical / z_optical) + cy
            
            # 6. Bounds Check
            u_int = int(u + 0.5)  # Round to nearest
            v_int = int(v + 0.5)
            
            if u_int < 0 or u_int >= img_width or v_int < 0 or v_int >= img_height:
                # Outside image bounds
                continue  # Already UNKNOWN
            
            # 7. Sample semantic label
            label = semantic_img[v_int, u_int]
            
            if label == road_label or label == road_line_label:
                grid[r, c] = FREE
            else:
                grid[r, c] = OCCUPIED
    
    return grid

@jit(nopython=True, cache=True)
def apply_column_drop_skirt(semantic_img, obstacle_ids, drop_pixels, hood_mask_rows):
    """
    Scans every column to find the lowest obstacle pixel and extends it downwards.
    Solves the 'Ghost Road' undercarriage problem without merging adjacent lanes.
    
    Args:
        semantic_img: Semantic label image [H, W]
        obstacle_ids: Array of label IDs to treat as obstacles
        drop_pixels: Number of pixels to extend obstacle downward
        hood_mask_rows: Ignore bottom N rows (ego vehicle hood)
    """
    rows, cols = semantic_img.shape
    
    # Don't scan the hood region at the bottom
    scan_limit = rows - hood_mask_rows
    
    # Iterate over every column
    for c in range(cols):
        # Scan from bottom of valid region up to find the first obstacle
        lowest_obs_row = -1
        fill_val = 0
        
        for r in range(scan_limit - 1, -1, -1):
            pixel_val = semantic_img[r, c]
            
            # Check if this is an obstacle
            is_obstacle = False
            for i in range(len(obstacle_ids)):
                if pixel_val == obstacle_ids[i]:
                    is_obstacle = True
                    break
            
            if is_obstacle:
                lowest_obs_row = r
                fill_val = pixel_val
                break # Found the bottom edge
        
        # Extend downwards (can fill into hood region)
        if lowest_obs_row != -1:
            start_r = lowest_obs_row + 1
            end_r = min(rows, start_r + drop_pixels)
            # print(f'Column {c}: Extending obstacle from row {lowest_obs_row} down to {end_r - 1} with label {fill_val}')
            for r in range(start_r, end_r):
                semantic_img[r, c] = fill_val

    return semantic_img