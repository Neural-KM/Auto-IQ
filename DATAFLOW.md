# Auto-IQ System Interfaces & Data Flow

This document details the complete data flow pipeline from sensor inputs to control actuation. The system operates in a distributed ROS 2 architecture (typically 10-20 Hz).

---

## 1. Bridge (Interface Layer)
**Responsibility**: Abstraction layer between the System and the CARLA Simulator.
- **Role**: Publishes sensor data, subscribes to control commands.
- **Node**: `carla_bridge_node`

**Outputs (Topics Published)**
- `/carla/ego_vehicle/sensor/camera/rgb/front/image` (sensor_msgs/Image): Raw RGB camera feed.
- `/carla/ego_vehicle/sensor/lidar/top/point_cloud` (sensor_msgs/PointCloud2): Raw LiDAR data.
- `/carla/ego_vehicle/ground_truth/odometry` (nav_msgs/Odometry): Ground-truth vehicle state (Position, Orientation, Velocity).

**Inputs (Topics Subscribed)**
- `/carla/ego_vehicle/vehicle_control_cmd` (carla_msgs/CarlaEgoVehicleControl): Final Actuation commands (Steering, Throttle, Brake).

---

## 2. Perception (Vision Pipeline)
**Responsibility**: Converts raw pixels into a complete semantic map (including lane markings) and detects objects.
**Launch**: `ros2 launch perception perception.launch.py segmentation_source:=segformer`

### A. Object Detection
- **Node**: `perception_node`
- **Algorithm**: YOLO (Deep Learning).
- **Input**: `/carla/ego_vehicle/sensor/camera/rgb/front/image`
- **Output**: Object detections (bounding boxes, classes, confidence).

### B. Semantic Segmentation
- **Node**: `semantic_segmentation_node` 
- **Algorithm**: SegFormer (Deep Learning) or Ground Truth lookup.
- **Input**: `/carla/ego_vehicle/sensor/camera/rgb/front/image`
- **Output**: `/perception/semantic_map` (sensor_msgs/Image - mono8): Pixel-wise class labels (Road, Vehicle, etc.). **Note: SegFormer does NOT produce road_line (class 24).**

### C. Road Extraction + Semantic Fusion
- **Node**: `road_extraction_node`
- **Algorithm**: Adaptive thresholding within road mask (class 1) to detect bright lane markings from the RGB image. Detected lane pixels are overwritten with class 24 (road_line). Also provides optional IPM-based lane visualization.
- **Inputs**: 
    - `/perception/semantic_map` (mono8, from SegFormer)
    - `/carla/ego_vehicle/sensor/camera/rgb/front/image` (RGB camera)
- **Outputs**: 
    - `/perception/semantic_map/fused` (sensor_msgs/Image - mono8): Complete semantic map with road_line class injected. **Drop-in replacement for CARLA ground truth.**
    - `/perception/semantic_map/fused/colored` (sensor_msgs/Image - rgb8): Colored visualization
    - `/perception/road_mask`, `/perception/lane_markings` (debug masks)
    - `/perception/lane_markings/ipm` (sensor_msgs/Image - rgb8): Top-down bird's eye view (optional debug visualization)

### D. Semantic PointCloud
- **Node**: `semantic_pointcloud_node`
- **Algorithm**: Projects semantic labels onto LiDAR point cloud.
- **Inputs**:
    - `/carla/ego_vehicle/sensor/lidar/top/point_cloud` (LiDAR)
    - `/carla/ego_vehicle/ground_truth/camera/semantic/front/image` (Semantic camera)
    - `/carla/ego_vehicle/sensor/camera/rgb/front/camera_info` (Camera intrinsics)
- **Output**: `/perception/semantic_pointcloud` (sensor_msgs/PointCloud2): Labeled 3D point cloud.

---

## 3. Mapping (Constraint Layer)
**Responsibility**: Obstacle mapping and free-space determination.

- **Node**: `local_costmap_node` (running `inverse_perspective_mapper_node_v2.py`)
- **Algorithm**: Inverse Perspective Mapping (Reverse Projection) using Semantic Images.
- **Inputs**: 
    - `/perception/semantic_map/fused` (Fused Semantic Map from Perception — or CARLA ground truth as fallback).
    - `/carla/ego_vehicle/sensor/camera/rgb/front/camera_info` (Camera Intrinsics).
- **Semantic Classes Used**:
    - Class 1 (`road`) + Class 24 (`road_line`) → **FREE** (0)
    - Everything else → **OCCUPIED** (100)
- **Output**: `/mapping/ipm_costmap` (nav_msgs/OccupancyGrid):
    - 2D Grid fixed to vehicle frame.
    - **0**: Free Space.
    - **100**: Lethal Obstacle.

---

## 4. Planning (Decision Layer)
**Responsibility**: Trajectory generation and cost optimization.

- **Node**: `planning_node` (using `planner_unified.py`).
- **Inputs**: 
    1. `/carla/ego_vehicle/ground_truth/odometry` (Current State).
    2. `/mapping/ipm_costmap` (Obstacle Constraints).
    
- **Process (Cost Function)**:
    - **Ref Cost**: Distance of trajectory endpoint from **Global Route** (Map Centerline).
    - **Lane Center Cost**: Mean-squared distance of *entire* trajectory from Global Route.
    - **Collision Cost**: Sum of costmap values under the trajectory footprint.
    
- **Output**: `/planning/trajectory` (autoiq_msgs/Trajectory): List of waypoints $(x, y, v, t)$ for the controller to follow.

---

## 5. Control (Actuation Layer)
**Responsibility**: Path tracking and command generation.

- **Node**: `pure_pursuit_node` (or `mpc_node`).
- **Inputs**:
    - `/planning/trajectory` (Target Path).
    - `/carla/ego_vehicle/ground_truth/odometry` (Current State).
    
- **Process**: 
    - Finds lookahead point on the trajectory.
    - Calculates curvature/steering angle required to reach it.
    - PID control for velocity tracking.
    
- **Output**: `/carla/ego_vehicle/vehicle_control_cmd` -> **Sent back to Bridge**.
