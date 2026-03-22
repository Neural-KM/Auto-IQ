"""
Skeleton Extractor
==================

Extracts the medial-axis skeleton of the local lane data from an 
occupancy grid (IPM costmap).

The skeleton represents the topological centerline(s) of free space,
which can later be used as a reference path or for topology-aware planning.

Usage:
    extractor = SkeletonExtractor()
    skeleton = extractor.extract(occupancy_grid)
"""

import numpy as np
from skimage.morphology import skeletonize, remove_small_objects


class SkeletonExtractor:
    """
    Extracts a 1-pixel-wide skeleton from the local lane data of an
    occupancy grid.

    Pipeline:
        1. Threshold the grid to get a binary drivable mask
        2. (Optional) Remove small spurious free regions
        3. Morphological skeletonization (Zhang-Suen thinning)
    """

    def __init__(
        self,
        free_threshold: int = 50,
        min_region_size: int = 200,
    ):
        """
        Args:
            free_threshold: Cells with value <= this are considered drivable.
                            OccupancyGrid convention: 0 = free, 100 = occupied.
            min_region_size: Connected free regions smaller than this (in
                             pixels) are discarded before skeletonization to
                             suppress noise.
        """
        self.free_threshold = free_threshold
        self.min_region_size = min_region_size

    def extract(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute the skeleton of the local lane data.

        Args:
            grid: 2-D int8/uint8 occupancy grid (H × W).
                  Values follow OccupancyGrid convention:
                      0        = free
                      1-100    = increasing cost / occupied
                      -1 (255) = unknown

        Returns:
            skeleton: 2-D bool array (H × W) where True marks the
                      skeleton pixels.
        """
        # 1. Binary drivable mask  (free <= threshold, treat unknown/-1 as occupied)
        drivable = self._threshold(grid)

        # 2. Remove small noise blobs
        if self.min_region_size > 0:
            drivable = self._clean(drivable)

        # 3. Skeletonize (Zhang-Suen thinning, returns bool array)
        skeleton = skeletonize(drivable)

        return skeleton

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _threshold(self, grid: np.ndarray) -> np.ndarray:
        """Create a binary drivable mask from the occupancy grid."""
        # OccupancyGrid int8: -1 (unknown) wraps to 127 in uint8 view,
        # or stays -1 in int8.  Both are > free_threshold, so they become
        # "not drivable" automatically.
        return (grid >= 0) & (grid <= self.free_threshold)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected regions smaller than min_region_size."""
        # remove_small_objects needs a bool array
        return remove_small_objects(
            mask.astype(bool),
            min_size=self.min_region_size,
            connectivity=2,          # 8-connected
        )

    # ------------------------------------------------------------------
    # Convenience: skeleton → OccupancyGrid data (for publishing)
    # ------------------------------------------------------------------

    @staticmethod
    def skeleton_to_occupancy_data(skeleton: np.ndarray, value: int = 100) -> np.ndarray:
        """
        Convert a boolean skeleton array to int8 occupancy values
        suitable for publishing as an OccupancyGrid.

        Skeleton pixels → `value` (default 100, shows as lethal/purple in RViz).
        Everything else → 0.

        Args:
            skeleton: 2-D bool array.
            value:    Occupancy value assigned to skeleton pixels.

        Returns:
            1-D int8 array ready for OccupancyGrid.data.
        """
        out = np.zeros(skeleton.shape, dtype=np.int8)
        out[skeleton] = value
        return out.flatten()
