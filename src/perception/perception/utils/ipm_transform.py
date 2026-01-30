import cv2
import numpy as np
from typing import Tuple, Optional
def convert_to_ipm(
    image: np.ndarray,
    fov: float = 110.0,
    pitch: float = 0.0,
    camera_height: float = 1.2,
    bev_x_range: Tuple[float, float] = None,
    bev_y_range: Tuple[float, float] = None,
    pixels_per_meter: float = None,
    return_transform: bool = False
) -> Tuple[np.ndarray, Tuple[int, int], Optional[np.ndarray]]:
    """
    Convert an image to Bird's Eye View using camera intrinsics and extrinsics.
    
    This function computes the perspective transform matrix from camera parameters
    and applies it to transform the image to a Bird's Eye View. The output BEV 
    dimensions are AUTO-CALCULATED from the x/y ranges and pixels_per_meter to 
    ensure equal scale in both X and Y directions (no distortion).
    
    Args:
        image: Input image (H, W, 3) or (H, W)
        fov: Camera horizontal field of view in degrees
        pitch: Camera pitch angle in degrees (positive = looking down)
        camera_height: Camera height above ground in meters
        bev_x_range: (x_min, x_max) range in meters (left/right from vehicle center).
                     Uses DEFAULT_BEV_X_RANGE if None.
        bev_y_range: (y_min, y_max) range in meters (forward distance from camera).
                     Uses DEFAULT_BEV_Y_RANGE if None.
        pixels_per_meter: Resolution of BEV output (pixels per meter). Higher = more detail.
                         Uses DEFAULT_PIXELS_PER_METER if None.
        return_transform: If True, also returns the 3x3 perspective transform matrix.
        
    Returns:
        bev_img: BEV transformed image
        bev_size: (width, height) of output BEV image (auto-calculated)
        M: 3x3 perspective transform matrix (only if return_transform=True, else None)
        
    Example:
        # Basic usage
        bev_img, bev_size, _ = convert_to_bev(image, fov=110, pitch=5, camera_height=1.2)
        
        # With custom ranges at 30 px/m: bev_x_range=(-15, 15), bev_y_range=(3, 30)
        # X span = 30m -> bev_width = 900 pixels
        # Y span = 27m -> bev_height = 810 pixels
        # Scale is equal: 1 pixel = 3.33 cm in both directions
    """
    height, width = image.shape[:2]
    
    # Use defaults if not specified
    if bev_x_range is None:
        bev_x_range = DEFAULT_BEV_X_RANGE
    if bev_y_range is None:
        bev_y_range = DEFAULT_BEV_Y_RANGE
    if pixels_per_meter is None:
        pixels_per_meter = DEFAULT_PIXELS_PER_METER
    
    x_min, x_max = bev_x_range
    y_min, y_max = bev_y_range
    
    # Calculate real-world spans
    x_span = x_max - x_min  # Total width in meters
    y_span = y_max - y_min  # Total depth in meters
    
    # Auto-calculate BEV dimensions to maintain equal scale
    bev_width = int(x_span * pixels_per_meter)
    bev_height = int(y_span * pixels_per_meter)
    
    # Ensure minimum size
    bev_width = max(bev_width, 100)
    bev_height = max(bev_height, 100)
    
    # Compute camera intrinsics
    fov_rad = np.radians(fov)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    pitch_rad = np.radians(pitch)
    
    # Ground plane points: (X, Y, Z) where Z=0 (ground level)
    # Order: far-left, far-right, near-right, near-left
    ground_points_3d = np.array([
        [x_min, y_max, 0],  # Far left
        [x_max, y_max, 0],  # Far right
        [x_max, y_min, 0],  # Near right
        [x_min, y_min, 0],  # Near left
    ], dtype=np.float64)
    
    # Rotation: pitch around x-axis
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ], dtype=np.float64)
    
    src_pts = []
    for gp in ground_points_3d:
        # Convert vehicle coords (X_v=right, Y_v=forward, Z_v=up) to 
        # camera coords (X_c=right, Y_c=down, Z_c=forward)
        
        # Point relative to camera position (camera at height h)
        p_vehicle = np.array([gp[0], gp[1], gp[2] - camera_height])
        
        # Transform to camera frame
        # X_c = X_v, Y_c = -Z_v, Z_c = Y_v
        p_cam_unrotated = np.array([p_vehicle[0], -p_vehicle[2], p_vehicle[1]])
        
        p_cam = R_pitch @ p_cam_unrotated
        
        # Project to image using pinhole model
        if p_cam[2] > 0:  # Only if in front of camera
            u = fx * p_cam[0] / p_cam[2] + cx
            v = fy * p_cam[1] / p_cam[2] + cy
            src_pts.append([u, v])
        else:
            src_pts.append([cx, 0])
    
    src_pts = np.float32(src_pts)
    
    # Destination points in BEV image
    dst_pts = np.float32([
        [0, 0],                           # Far left -> top-left
        [bev_width, 0],                   # Far right -> top-right
        [bev_width, bev_height],          # Near right -> bottom-right
        [0, bev_height],                  # Near left -> bottom-left
    ])
    
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply transform to image
    bev_img = cv2.warpPerspective(image, M, (bev_width, bev_height), flags=cv2.INTER_LINEAR)
    
    bev_size = (bev_width, bev_height)
    
    if return_transform:
        return bev_img, bev_size, M
    else:
        return bev_img, bev_size, None

