# Localization & Mapping (SLAM) Subsystem

**Goal:** Determine precise vehicle position and build/use a map of the world.

**Deliverables:**
*   **Pose:** Position ($x,y,z$) and Orientation ($roll, pitch, yaw$).
*   **Velocity/Acceleration:** For control feedback.
*   **Map:** Static representation of the world (if not pre-loaded).

**Implementation options:**

1.  **Ground Truth:** Get transform from simulator. (*Implemented*)
2.  **Dead Reckoning:** Estimate pose by integrating motion over time.
3.  **GNSS + IMU Fusion:** Fuse GNSS (absolute but noisy) with IMU/odometry (smooth but drifting) to keep errors bounded.
4.  **Map Matching (Pure Localization):** Assumes a pre-built map exists. The goal is only to determine the vehicle's pose by aligning live sensor data (LiDAR/camera) to the map.
5.  **Visual SLAM (V-SLAM):** Build a map and localize simultaneously using only camera images. This involves extracting and matching visual features (e.g., ORB, SIFT) between frames to estimate camera motion and build a 3D map of feature points.
6.  **Lidar SLAM:** Build a map and localize simultaneously using only LiDAR scans. Map matching is a core component of this process, used to align new scans against the map being built. This typically involves scan-to-scan or scan-to-map matching to estimate motion and build a point cloud map.

**Algorithms (brief):**
*   **Kalman Filter:** Fuses a motion model with noisy measurements to estimate state and uncertainty.
*   **EKF:** A Kalman Filter that linearizes non-linear models around the current estimate.
*   **UKF:** A Kalman Filter that uses sigma points to handle non-linear models without explicit linearization.
*   **ICP:** Aligns two point sets by iteratively minimizing distances between corresponding points.
*   **NDT:** Aligns scans to a map represented as local Gaussian distributions by maximizing likelihood.
*   **ORB-SLAM:** A popular feature-based Visual SLAM system that uses ORB (Oriented FAST and Rotated BRIEF) features for tracking, mapping, relocalization, and loop closure. It builds a sparse map of 3D points.
*   **LIO-SAM / FAST-LIO:** Tightly-coupled LiDAR-Inertial Odometry systems. They fuse IMU data for high-frequency motion prediction and use LiDAR scans for correction, achieving high accuracy and robustness. The process involves:
    *   **IMU Pre-integration:** Summarize high-frequency IMU measurements between LiDAR scans into a single relative motion constraint.
    *   **Feature Extraction:** Extract planar and edge features from the point cloud.
    *   **Scan Matching:** Match features to a local map of recently observed features.
    *   **Factor Graph Optimization:** Use a factor graph (e.g., GTSAM) to jointly optimize the vehicle's trajectory by incorporating constraints from IMU, LiDAR odometry, and loop closures.

**Where 3D Gaussian Splatting fits:**
*   **3D Gaussian Splatting (3DGS):** A scene representation that reconstructs a dense 3D model from many images and renders it fast. It can be used as a *mapping* representation (build a 3DGS map) and sometimes for *localization* by finding the camera pose that best matches a rendered view to the live camera image. This approach is promising for creating photorealistic, high-fidelity maps for localization and simulation.

**Supporting algorithms (small but useful):**
*   **Motion compensation:** Corrects scan/image distortion using IMU/odometry (important when the platform moves).
*   **Loop closure:** Detects revisiting the same place and reduces accumulated drift by adding constraints.
*   **Outlier rejection (RANSAC):** Filters bad correspondences when matching features/points.
*   **Time synchronization:** Aligns sensor timestamps so fusion doesn’t inject artificial errors.
*   **Pose Graph Optimization:** A backend optimization technique used in many SLAM systems. The vehicle's trajectory is represented as a graph where nodes are poses at different times and edges are relative motion constraints between poses (from odometry or loop closures). A non-linear least squares solver adjusts all poses to minimize the error in the constraints, correcting for accumulated drift.
---

## Tasks

- [ ] Basic IMU + dead reckoning localization
- [ ] IMU + GPS fusion localization
- [ ] Visual SLAM implementation
- [ ] Map matching localization with LiDAR
- [ ] 