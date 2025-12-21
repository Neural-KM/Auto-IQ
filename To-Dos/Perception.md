# Perception Subsystem

**Goal:** Provide a complete understanding of the environment, including obstacles, traffic rules, and road geometry.

**Deliverables:**
*   **Dynamic Objects:** Vehicles, pedestrians, cyclists (Position, Velocity, Class).
*   **Static Objects:** Traffic cones, barriers, debris.
*   **Traffic Controls:** Traffic lights (State: Red/Green/Yellow), Stop signs, Speed limit signs.
*   **Road Geometry:** Lane lines, drivable area, crosswalks.

**Implementation options:**

1.  **Ground Truth (Simulation Only):** Query the simulator API for all actor positions and traffic light states.
2.  **2D Camera + Depth:** Detect objects in the image and use depth to estimate distance/relative position; detect traffic lights from the image.
3.  **Lidar-Only 3D Detection:** Use point clouds to detect and size obstacles in 3D.
4.  **Camera–Lidar Fusion:** Combine camera classification (what) with lidar geometry (where).
5.  **Bird’s Eye View (BEV) Fusion:** Fuse multiple sensors into a single top-down representation for detection and road understanding.
6.  **End-to-End Transformer Models:** Use a single, large transformer model to process multi-camera images (and optionally LiDAR) to directly predict perception outputs, often in a BEV space.

**Algorithms (brief):**
*   **YOLO:** Predicts object classes and 2D bounding boxes from an image in a single pass.
*   **Color thresholding:** Segments pixels by color range (often HSV) as a baseline for traffic light state.
*   **PointPillars:** Converts point clouds into vertical “pillars” and predicts 3D boxes efficiently.
*   **VoxelNet:** Voxelizes point clouds into a 3D grid and learns voxel features to regress 3D boxes.
*   **PointPainting:** Projects camera semantics onto LiDAR points before running 3D detection.
*   **BEVFormer / BEV-based Transformers:** These models use a transformer architecture to convert 2D image features from multiple cameras into a unified Bird's-Eye-View (BEV) representation. They work by using learnable BEV queries that "ask" the multi-camera images for relevant information, effectively projecting and fusing features into a top-down grid without explicit geometry.
*   **Occupancy Networks:** Instead of predicting bounding boxes, these models predict the probability of each point in a 3D space being occupied. This creates a dense, fine-grained 3D map of the environment, capturing complex shapes that boxes cannot. They typically use sensor data to query a neural network for the occupancy value at any continuous 3D coordinate.
*   **Vision Transformer (ViT):** A model that treats an image as a sequence of patches, similar to how text transformers treat a sequence of words. It serves as a powerful backbone for extracting features from images, which can then be fed into other models like BEV transformers.

**Supporting algorithms (small but useful):**
*   **Semantic segmentation:** Pixel-wise labeling (road, lane markings, curb, sidewalk) to produce drivable area and lane-marking masks.
*   **Instance segmentation:** Separates “which pixels belong to which object,” useful when boxes are not enough.
*   **Optical flow:** Estimates per-pixel motion; helpful for detecting moving objects and relative motion cues.
*   **Multi-object tracking (SORT/DeepSORT):** Assigns persistent IDs to detections and estimates velocity over time. A common approach involves:
    *   **State Estimation:** Using a Kalman Filter to predict the next position of a tracked object.
    *   **Data Association:** Using the Hungarian algorithm or Joint Probability Data Association (JPDA) to match new detections to existing tracks based on a cost metric (e.g., IoU or distance).
*   **Camera calibration & projection:** Intrinsics/extrinsics management; critical when projecting between image/depth/3D.

---

## Tasks
- [*] Implement rgb+depth camera based object detection
    * Basic functionality works
    * But bounding box accuracy is very less
    * Need to test robustness in different lighting/weather conditions.
- [*] Lane segmentation and lane line detection
    * Works if only one lane present in the camera view
    * Need to modify for multiple lanes and curved roads
    * Is lane detection necessary if HD map + localization is available?
- [ ] Use Lidar point cloud instead of depth camera for depth estimation in YOLO-based detection.
- [ ] Use all cameras for multi-view object detection
- [ ] Traffic light detection
- [ ] Lidar only 3D object detection
- [ ] 
