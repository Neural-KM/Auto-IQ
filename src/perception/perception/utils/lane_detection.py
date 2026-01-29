import cv2
import numpy as np

def _extract_lanes_from_bev(
        bev_image: np.ndarray, 
        bev_road_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract lane markings from BEV image using improved multi-scale adaptive thresholding.
        
        Args:
            bev_image: BEV transformed RGB image
            bev_road_mask: Binary road mask in BEV space
            
        Returns:
            Binary lane mask (H, W) where 255 = lane markings
        """
        # Extract road ROI
        road_only = cv2.bitwise_and(bev_image, bev_image, mask=bev_road_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(road_only, cv2.COLOR_RGB2GRAY)
        
        # =====================================================================
        # IMPROVEMENT 1: Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This enhances local contrast making lane markings more visible
        # against varying road surfaces (shadows, different asphalt colors)
        # =====================================================================
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Fill non-road areas with mean road intensity (prevents edge artifacts)
        road_pixels = enhanced_gray[bev_road_mask == 255]
        if len(road_pixels) == 0:
            raise ValueError("No road pixels found in BEV road mask. Cannot extract lane markings.")
        mean_val = np.mean(road_pixels)
        filled_gray = enhanced_gray.copy()
        filled_gray[bev_road_mask == 0] = int(mean_val)
        
        # =====================================================================
        # IMPROVEMENT 2: Multi-scale Gaussian blur for noise reduction
        # Light blur preserves lane edges, heavier blur removes small noise
        # =====================================================================
        blurred_light = cv2.GaussianBlur(filled_gray, (3, 3), 0)  # Preserve fine details
        blurred_medium = cv2.GaussianBlur(filled_gray, (5, 5), 0)  # Balance
        
        # =====================================================================
        # IMPROVEMENT 3: Multi-scale Adaptive Thresholding
        # Different block sizes capture lanes at different distances/scales
        # Small block (11): Captures thin/distant lane markings
        # Medium block (21): General lane detection  
        # Large block (31): Captures wide lane markings, less affected by noise
        # =====================================================================
        
        # Small scale - good for thin lane markings
        lane_mask_small = cv2.adaptiveThreshold(
            blurred_light,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,   # Small block size
            -8    # C value: detect pixels brighter than local mean + 8
        )
        
        # Medium scale - balanced detection
        lane_mask_medium = cv2.adaptiveThreshold(
            blurred_medium,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,   # Medium block size
            -12   # C value
        )
        
        # Large scale - for thick lane markings, robust to noise
        lane_mask_large = cv2.adaptiveThreshold(
            blurred_medium,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,   # Large block size
            -15   # C value
        )
        
        # =====================================================================
        # IMPROVEMENT 4: Statistical thresholding with Otsu
        # Otsu automatically finds optimal threshold - good for high contrast lanes
        # =====================================================================
        if len(road_pixels) > 100:
            # Apply Otsu only on road pixels for better threshold estimation
            _, lane_mask_otsu = cv2.threshold(
                filled_gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            lane_mask_otsu = np.zeros_like(gray)
        
        # =====================================================================
        # IMPROVEMENT 5: Top-hat transform to extract bright regions
        # This is excellent for extracting lane markings (bright on dark road)
        # Removes large-scale illumination variations
        # =====================================================================
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(filled_gray, cv2.MORPH_TOPHAT, kernel_tophat)
        
        # Threshold the top-hat result
        if len(road_pixels) > 0:
            tophat_thresh = np.mean(tophat[bev_road_mask == 255]) + 2.0 * np.std(tophat[bev_road_mask == 255])
            _, lane_mask_tophat = cv2.threshold(tophat, max(tophat_thresh, 20), 255, cv2.THRESH_BINARY)
        else:
            lane_mask_tophat = np.zeros_like(gray)
        
        # =====================================================================
        # IMPROVEMENT 6: Smart combination using weighted voting
        # Instead of simple OR, use voting - pixel is lane if detected by
        # at least 2 methods (reduces false positives)
        # =====================================================================
        # Convert to binary 0/1 for voting
        votes = (
            (lane_mask_small > 0).astype(np.uint8) +
            (lane_mask_medium > 0).astype(np.uint8) +
            (lane_mask_large > 0).astype(np.uint8) +
            (lane_mask_otsu > 0).astype(np.uint8) +
            (lane_mask_tophat > 0).astype(np.uint8)
        )
        
        # Require at least 3 votes (detected by 3+ methods)
        lane_mask = ((votes >= 2) * 255).astype(np.uint8)
        
        # =====================================================================
        # IMPROVEMENT 7: Enhanced morphological cleanup
        # =====================================================================
        # Remove small isolated noise (opening)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Connect nearby lane segments (closing with rectangular kernel for lanes)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))  # Vertical bias for lanes
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # =====================================================================
        # IMPROVEMENT 8: Remove small blobs using connected components
        # Lanes have minimum area - remove tiny noise blobs
        # =====================================================================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
        min_area = 50  # Minimum lane marking area in pixels
        
        # Create clean mask keeping only components above minimum area
        lane_mask_clean = np.zeros_like(lane_mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                lane_mask_clean[labels == i] = 255
        
        # Enforce road mask
        lane_mask_clean = cv2.bitwise_and(lane_mask_clean, bev_road_mask)
        
        return lane_mask_clean