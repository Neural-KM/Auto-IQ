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
**Responsibility**: Converts raw pixels into geometric lane information.

### A. Semantic Segmentation
- **Node**: `semantic_segmentation_node` 
- **Algorithm**: SegFormer (Deep Learning) or Ground Truth lookup.
- **Input**: `/carla/ego_vehicle/sensor/camera/rgb/front/image`
- **Output**: `/perception/semantic_map` (sensor_msgs/Image - mono8): Pixel-wise class labels (Road, Vehicle, etc.).

### B. Road Extraction
- **Node**: `road_extraction_node`
- **Algorithm**: Color thresholding, Inverse Perspective Mapping (IPM).
- **Inputs**: 
    - `/perception/semantic_map`
    - `/carla/ego_vehicle/sensor/camera/rgb/front/image`
- **Output**: `/perception/lane_markings/ipm` (sensor_msgs/Image): Top-down bird's eye view of lane lines.

### C. Lane Center Estimator
- **Node**: `lane_center_estimator_node`
- **Algorithm**: Sliding window polynomial fitting on IPM image.
- **Inputs**: 
    - `/perception/lane_markings/ipm`
    - `/carla/ego_vehicle/ground_truth/odometry` (To transform local pixels to global map frame).
- **Output**: `/planning/perception_reference_path` (nav_msgs/Path): The computed geometric center line of the current lane.

---

## 3. Mapping (Constraint Layer)
**Responsibility**: Obstacle mapping and free-space determination.

- **Node**: `local_costmap_node`
- **Algorithm**: Inverse Perspective Mapping / Raycasting.
- **Inputs**: Sensor Data (LiDAR/Camera based).
- **Output**: `/mapping/ipm_costmap` (nav_msgs/OccupancyGrid):
    - 2D Grid fixed to vehicle or map frame.
    - **0**: Free Space.
    - **100**: Lethal Obstacle.

---

## 4. Planning (Decision Layer)
**Responsibility**: Trajectory generation and cost optimization.

- **Node**: `planning_node` (using `planner_unified.py`).
- **Inputs**: 
    1. `/carla/ego_vehicle/ground_truth/odometry` (Current State).
    2. `/planning/perception_reference_path` (Reference Line - defines $d=0$).
    3. `/mapping/ipm_costmap` (Obstacle Constraints).
    
- **Process (Cost Function)**:
    - **Ref Cost**: Distance of trajectory endpoint from Reference Path.
    - **Lane Center Cost**: Mean-squared distance of *entire* trajectory from Reference Path.
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
