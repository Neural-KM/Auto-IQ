"""
Geometric utilities for occupancy grid operations.
"""
import numpy as np


def mark_ego_footprint(
    grid: np.ndarray,
    resolution: float,
    vehicle_length: float,
    vehicle_width: float,
    rear_to_base: float,
    FREE: int
) -> np.ndarray:
    """
    Mark the ego vehicle footprint as FREE in the occupancy grid.
    
    Assumes vehicle (base_link) is at the center of the grid.
    
    Args:
        grid: Occupancy grid [rows, cols]
        resolution: Grid resolution (meters per cell)
        vehicle_length: Total vehicle length (meters)
        vehicle_width: Total vehicle width (meters)
        rear_to_base: Distance from rear end to base_link (meters)
        FREE: Value for free cells
        
    Returns:
        Modified grid with ego footprint marked as FREE
    """
    grid_rows, grid_cols = grid.shape
    
    # Grid center (vehicle base_link position)
    center_r = grid_rows // 2
    center_c = grid_cols // 2
    
    # Vehicle extent in cells from base_link
    half_width_cells = int((vehicle_width / 2.0) / resolution)
    rear_cells = int(rear_to_base / resolution)
    front_cells = int((vehicle_length - rear_to_base) / resolution)
    
    # Grid indices
    r_min = max(0, center_r - half_width_cells)
    r_max = min(grid_rows - 1, center_r + half_width_cells)
    c_min = max(0, center_c - rear_cells)
    c_max = min(grid_cols - 1, center_c + front_cells)
    
    # Mark footprint as FREE
    grid[r_min:r_max + 1, c_min:c_max + 1] = FREE
    
    return grid


def mark_circular_clear_zone(
    grid: np.ndarray,
    radius: float,
    resolution: float,
    FREE: int
) -> np.ndarray:
    """
    Mark a circular region around the vehicle center as FREE.
    
    Args:
        grid: Occupancy grid [rows, cols]
        radius: Radius in meters
        resolution: Grid resolution (meters per cell)
        FREE: Value for free cells
        
    Returns:
        Modified grid
    """
    rows, cols = grid.shape
    center_r, center_c = rows // 2, cols // 2
    
    radius_cells = int(radius / resolution)
    
    # Create coordinate grids suitable for mask
    # y corresponds to rows (0 is center), x to cols (0 is center)
    y, x = np.ogrid[-center_r:rows-center_r, -center_c:cols-center_c]
    
    # Boolean mask where distance <= radius
    mask = x*x + y*y <= radius_cells*radius_cells
    
    # Apply to grid
    grid[mask] = FREE
    
    return grid
