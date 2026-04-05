# HD Map Generation Pipeline

This document describes the step-by-step process to generate an HD Map from the CARLA simulator using the tools in this directory and `fast_lio`.

## Step 1: Record a Rosbag

First, collect the necessary sensor data from CARLA. Ensure `autoiq_bridge` and `carla_ros_bridge` are running, or use the collector standalone.

Run the data collector script to spawn a vehicle, navigate between spawn points, and record a rosbag:

```bash
python3 /home/autoiq/nakul/Auto-IQ/tools/HD_map/hd_map_data_collector.py --start <START_SPAWN_ID> --end <END_SPAWN_ID> --record
```
*(Example: `--start 10 --end 50`)*

Outputs are saved in the `Rosbags/hdmap_YYYY_MM_DD-HH_MM_SS/` directory.

## Step 2: Run Fast-LIO

To generate accurate odometry/mapping from the recorded LiDAR and IMU data, run `fast_lio` on the recorded rosbag:

```bash
# Terminal 1: Launch Fast-LIO
ros2 launch fast_lio mapping.launch.py config_file:=carla.yaml rviz:=true

# Terminal 2: Record the Fast-LIO output
ros2 bag record -o Rosbags/hdmap_fastlio_odom /Odometry /path --use-sim-time

# Terminal 3: Play the recorded rosbag (make sure to use the correct timestamp folder)
ros2 bag play Rosbags/hdmap_<timestamp> --clock --qos-profile-overrides-path qos_override.yaml
```

## Step 3: Merge Rosbags

Merge the Fast-LIO output rosbag with your original sensor rosbag so the processor has synchronized access to both odometry and camera/semantic data.

You can use the merge tool configured via `Rosbags/merge_config.yaml` or a standard ROS 2 bag merging approach to combine the files into a single dataset (e.g., `Rosbags/hdmap_merged_new/`).

## Step 4: Process into an HD Map

Run the HD map processor on the merged rosbag to generate the final map format. 

```bash
python3 /home/autoiq/nakul/Auto-IQ/tools/HD_map/hd_map_processor.py --bag Rosbags/hdmap_merged_new/ --visualize
```

This will run the necessary processing steps on the synchronized data to output HD map assets.
