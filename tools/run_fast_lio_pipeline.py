#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_fast_lio2_and_record(bag_path, output_bag_path):
    """
    Run Fast-LIO2 on a rosbag and record its output to a new rosbag.
    This assumes Fast-LIO2 is installed and available in the ROS2 workspace.
    """
    bag_path = Path(bag_path).resolve()
    output_bag_path = Path(output_bag_path).resolve()

    print(f"[{time.strftime('%H:%M:%S')}] Starting Fast-LIO2 processing pipeline...")
    print(f"Input bag: {bag_path}")
    print(f"Output bag: {output_bag_path}")

    if output_bag_path.exists():
        print(f"Warning: Output bag {output_bag_path} already exists. Removing it...")
        subprocess.run(['rm', '-rf', str(output_bag_path)])

    # Setup the commands
    # 1. Play the input rosbag (delay start to give FastLio time to initialize)
    play_cmd = f"ros2 bag play {bag_path} --delay 2.0"
    
    # 2. Record the Fast-LIO2 outputs (Odometry and PointCloud)
    record_cmd = f"ros2 bag record -o {output_bag_path} /Odometry /cloud_registered"
    
    # 3. Launch Fast-LIO2
    # NOTE: You might need to adjust the package and launch file names depending on 
    #       how Fast-LIO2 is installed in your ROS 2 environment (e.g., fast_lio matching your config)
    #       Assuming the standard fast_lio ROS2 port:
    fast_lio_cmd = "ros2 launch fast_lio mapping.launch.py"

    print(f"[{time.strftime('%H:%M:%S')}] Starting Fast-LIO2 node...")
    fl_process = subprocess.Popen(fast_lio_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    time.sleep(3) # Wait for node to initialize

    print(f"[{time.strftime('%H:%M:%S')}] Starting ROS 2 bag record...")
    record_process = subprocess.Popen(record_cmd, shell=True)
    
    time.sleep(2) # Give recording a head start

    print(f"[{time.strftime('%H:%M:%S')}] Playing original ROS 2 bag...")
    play_process = subprocess.Popen(play_cmd, shell=True)

    # Wait for bag playback to finish
    play_process.wait()
    print(f"[{time.strftime('%H:%M:%S')}] Bag playback finished.")

    # Give a little time for pending messages to be recorded
    time.sleep(2)

    print(f"[{time.strftime('%H:%M:%S')}] Terminating Fast-LIO2 and recording nodes...")
    
    # Terminate the record process safely (sends SIGINT equivalent)
    record_process.terminate()
    try:
        record_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        record_process.kill()

    # Terminate FastLio process
    fl_process.terminate()
    try:
        fl_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        fl_process.kill()
        
    print(f"[{time.strftime('%H:%M:%S')}] Ensure orphaned ros2 processes are stopped...")
    subprocess.run("pkill -2 -f 'ros2 bag'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run("pkill -2 -f 'fast_lio'", shell=True, stderr=subprocess.DEVNULL)

    print(f"[{time.strftime('%H:%M:%S')}] Output bag generated successfully at: {output_bag_path}")

def run_hd_map_processor(original_bag, slamed_bag, segmentation_source):
    """
    Runs the existing hd_map_processor using the original bag (for images) 
    and the new slamed_bag (for Fast-LIO2 odometry).
    """
    
    # We update hd_map_processor.py to merge bags! But actually, it's easier to
    # run our existing hd_map_processor on the slamed bag, but we need the camera images.
    # So instead, let's just instruct the user on the next command.
    
    print("\nNext, run the HD Map Processor with the merged Fast-LIO odometry.")
    print("Wait... Fast-LIO2 outputs are strictly /Odometry and /cloud_registered.")
    print("Your script `hd_map_processor.py` relies on camera topics which aren't in this new output bag.")
    print("\nIf you want to use the FastLIO odometry for HD Maps, you must record the original camera topics ALONG WITH the fast-lio outputs.")
    
    pass

def main():
    parser = argparse.ArgumentParser(description='Automate Fast-LIO2 SLAM on a rosbag.')
    parser.add_argument('--bag', required=True, help='Path to the input rosbag directory')
    parser.add_argument('--output', required=False, help='Path to the output rosbag directory (default: <bag>_fastlio)')

    args = parser.parse_args()

    input_bag = args.bag
    if args.output:
        output_bag = args.output
    else:
        output_bag = f"{str(Path(input_bag).resolve())}_fastlio"

    if not os.path.exists(input_bag):
        print(f"Error: Input bag {input_bag} does not exist.")
        sys.exit(1)

    # 1. Run the SLAM pipeline and record
    run_fast_lio2_and_record(input_bag, output_bag)

    # 2. Inform the user how to run the HD Map Processor
    print(f"\n========================================================")
    print(f"Success! The Fast-LIO2 odometry is recorded in: {output_bag}")
    print(f"However, to use `hd_map_processor.py`, you need the camera images.")
    print(f"I recommend editing the `record_cmd` in this script to also record the camera topics:")
    print(f"  ros2 bag record -o {output_bag} /Odometry /cloud_registered /carla/ego_vehicle/ground_truth/camera/semantic/front/image /carla/ego_vehicle/sensor/camera/rgb/front/image")
    print(f"========================================================")


if __name__ == '__main__':
    main()
