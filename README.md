# Auto-IQ

A Level 4 autonomous vehicle simulation framework built on CARLA

## Requirements
```bash
pip install -r requirements.txt
```

## Quick Start (ROS2)

1. **Install CARLA Python bindings**
   ```bash
   pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-linux_x86_64.whl
   ```

2. **Start CARLA simulator**

3. **Build the workspace**
   ```bash
   cd ~/Auto-IQ
   colcon build
   source install/setup.bash
   ```

4. **Launch the full stack**
   ```bash
   ros2 launch autoiq_bridge autoiq.launch.py
   ```

## Launch Arguments

### Relevant Config params
| Argument | Default | Description |
|----------|---------|-------------|
| `controller` | `pure_pursuit` | Control algorithm: `mpc` or `pure_pursuit` |
| `use_ground_truth` | `true` | Use ground truth perception |

### Usage

```bash
# Full stack with defaults
ros2 launch autoiq_bridge autoiq.launch.py

# Test planning separately (disable it, run manually)
ros2 launch autoiq_bridge autoiq.launch.py enable_planning:=false

# Use Pure Pursuit controller
ros2 launch autoiq_bridge autoiq.launch.py controller:=pure_pursuit

# Use real perception (YOLO) instead of ground truth
ros2 launch autoiq_bridge autoiq.launch.py use_ground_truth:=false
```

## Tools

```bash
# Spawn static obstacle
python3 tools/spawn_obstacle.py --ahead 30

# Set navigation goal by spawn point index
ros2 topic pub /planning/goal_index std_msgs/Int32 "data: 250" --once
```

## Perception Nodes

### Semantic Segmentation + Lane Extraction Pipeline

```bash
# Semantic segmentation (choose source: ground_truth or segformer)
ros2 run perception semantic_segmentation_node --ros-args -p segmentation_source:=ground_truth

# Lane extraction (subscribes to semantic map)
ros2 run perception road_extraction_node
```

| Node | Subscribes | Publishes |
|------|------------|-----------|
| `semantic_segmentation_node` | RGB or GT semantic camera | `/perception/semantic_map`, `/perception/semantic_map/colored` |
| `road_extraction_node` | `/perception/semantic_map` (mono8), RGB camera topic | `/perception/road_mask`, `/perception/lane_markings` |

## Datasets

```bash
# Carla image datasets of every spawn point
wget "https://huggingface.co/datasets/nakul004/imagessim/resolve/main/Data.zip"

# Rosbags 15/1 bot at transit
wget "https://huggingface.co/datasets/nakul004/imagessim/resolve/main/rosbag2_1970_01_01-05_49_48.zip"
wget "https://huggingface.co/datasets/nakul004/imagessim/resolve/main/rosbag2_1970_01_01-05_54_59.zip"
wget "https://huggingface.co/datasets/nakul004/imagessim/resolve/main/rosbag2_1970_01_01-05_50_52.zip" 

#Rosbag images 15/1 bot at transit
wget "https://huggingface.co/datasets/nakul004/imagessim/resolve/main/Images_15jan_bot_transit.zip" 
```

## License

Apache License 2.0
