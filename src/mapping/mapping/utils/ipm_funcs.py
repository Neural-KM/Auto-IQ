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

@jit(nopython=True, cache=True)
def project_pixels_to_ground(u_arr, v_arr, fx, fy, cx, cy, 
                             R_cam2map, t_cam2map, ground_z):
    """
    Project image pixels to ground plane (Z=ground_z) in map frame.
    
    CARLA uses camera_link frame convention (X=forward, Y=left, Z=up),
    but pixel unprojection assumes optical frame (X=right, Y=down, Z=forward).
    
    We convert: optical [norm_x, norm_y, 1] -> link [1, -norm_x, -norm_y]
    
    Math:
    1. Pixel to Ray (Optical Frame): ray_optical = [(u-cx)/fx, (v-cy)/fy, 1]
    2. Convert to Camera Link Frame: ray_link = [1, -norm_x, -norm_y]
    3. Ray to Map Frame: ray_map = R * ray_link
    4. Intersection with Ground (Z=ground_z):
       s = (ground_z - tz) / dz
    """
    n_points = len(u_arr)
    out_x = np.empty(n_points, dtype=np.float32)
    out_y = np.empty(n_points, dtype=np.float32)
    valid_mask = np.zeros(n_points, dtype=np.bool_)
    
    # Extract translation (camera origin in map frame)
    tx, ty, tz = t_cam2map[0], t_cam2map[1], t_cam2map[2]
    
    for i in range(n_points):
        u = u_arr[i]
        v = v_arr[i]
        
        # 1. Unproject to normalized ray in Camera Optical Frame
        # Optical: X=right, Y=down, Z=forward
        norm_x = (u - cx) / fx
        norm_y = (v - cy) / fy
        
        # 2. Convert from Optical Frame to Camera Link Frame
        # CARLA camera_link: X=forward, Y=left, Z=up
        # Optical X (right) = -Link Y (left)
        # Optical Y (down) = -Link Z (up)  
        # Optical Z (forward) = Link X (forward)
        # So: link = [optical_z, -optical_x, -optical_y] = [1, -norm_x, -norm_y]
        link_x = 1.0
        link_y = -norm_x
        link_z = -norm_y
        
        # 3. Rotate Ray Direction to Map Frame
        # dir = R * [link_x, link_y, link_z]
        dx = R_cam2map[0,0]*link_x + R_cam2map[0,1]*link_y + R_cam2map[0,2]*link_z
        dy = R_cam2map[1,0]*link_x + R_cam2map[1,1]*link_y + R_cam2map[1,2]*link_z
        dz = R_cam2map[2,0]*link_x + R_cam2map[2,1]*link_y + R_cam2map[2,2]*link_z
        
        # 4. Intersect with Ground Plane (Z=ground_z)
        # We need s such that: tz + s * dz = ground_z  =>  s = (ground_z - tz) / dz
        # For rays pointing down (dz < 0) with camera above ground (tz > ground_z), s > 0
        
        if abs(dz) > 1e-4:  # Avoid division by zero (horizon rays)
            s = (ground_z - tz) / dz
            
            # Intersection must be in front of the ray (s > 0)
            if s > 0:
                gx = tx + s * dx
                gy = ty + s * dy
                
                # Filter points extremely far away which cause grid artifacts
                if s < 100.0: 
                    out_x[i] = gx
                    out_y[i] = gy
                    valid_mask[i] = True
                    
    return out_x, out_y, valid_mask

@jit(nopython=True, cache=True)
def fill_grid_from_points(x_arr, y_arr, valid_mask, grid, 
                          origin_x, origin_y, resolution, width, height, FREE):
    """
    Mark grid cells as FREE (0) based on ground points.
    Initializes grid with UNKNOWN (-1) or OBSTACLE (100) before filling.
    """
    n_points = len(x_arr)
    
    for i in range(n_points):
        if not valid_mask[i]:
            continue
            
        # Convert World (x, y) to Grid (c, r)
        c = int((x_arr[i] - origin_x) / resolution)
        r = int((y_arr[i] - origin_y) / resolution)
        
        if 0 <= c < width and 0 <= r < height:
            # Mark as FREE
            grid[r, c] = FREE
           
            # if c+1 < width: grid[r, c+1] = FREE
            # if r+1 < height: grid[r+1, c] = FREE
