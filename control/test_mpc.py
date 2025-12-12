"""
Simple MPC integration test harness.

This script constructs an MPCController, a TrajectoryGenerator, and a small
set of waypoints. It then computes a reference and solves the MPC problem,
printing the results.

Edit the initial conditions and waypoints below as needed.
"""

import numpy as np
import math

from control.mpc_controller import MPCController
from control.trajectory_generator import TrajectoryGenerator
import matplotlib.pyplot as plt

# ---- EDIT THESE INITIAL CONDITIONS ----
init_x = 405.7
init_y = -95.4
init_yaw_deg = -89.4  # degrees
init_v = 1.0        # m/s
horizon = 21
mpc_dt = 0.05
# --------------------------------------



def main():
    yaw_rad = math.radians(init_yaw_deg)

    # Build a small straight set of waypoints ahead of the initial pose
    waypoints = [
        (405.74182129, -97.58068848),
        (405.76269531, -99.58058167),
        (405.78359985, -101.58046722),
        (405.80450439, -103.58035278),
        (405.82537842, -105.58024597),
        (405.84628296, -107.58013916),
        (405.8671875, -109.58003235),
        (405.88806152, -111.57992554),
        (405.90896606, -113.5798111),
        (405.92987061, -115.57970428),
        (405.95074463, -117.57958984),
        (405.97164917, -119.57948303),
        (405.99255371, -121.57937622),
        (406.01342773, -123.57926941),
        (406.03433228, -125.5791626),
        (406.05523682, -127.57904816),
        (406.07611084, -129.57893372),
        (406.09701538, -131.5788269),
        (406.11791992, -133.57872009),
        (406.13879395, -135.57861328),
        (406.15969849, -137.57850647),
        (406.18060303, -139.57839966),
        (406.20147705, -141.57829285),
        (406.22238159, -143.57817078),
        (406.24328613, -145.57806396),
        (406.26416016, -147.57795715),
        (406.2850647, -149.57785034),
        (406.30596924, -151.57774353),
        (406.32684326, -153.57763672),
        (406.3477478, -155.57751465)
    ]

    print(f"[test_mpc] Using {len(waypoints)} waypoints: first={waypoints[0]}, last={waypoints[-1]}")

    # Initialize trajectory generator and fit waypoints
    traj_gen = TrajectoryGenerator(target_speed=10.0)
    success = traj_gen.set_waypoints(waypoints)
    if not success:
        print("[test_mpc] Failed to fit spline to waypoints")
        return

    # Current state
    current_state = np.array([init_x,init_y, yaw_rad, init_v])

    # Generate reference from trajectory generator
    reference = traj_gen.generate_reference(current_state, horizon_length=horizon, dt=mpc_dt)
    if reference is None:
        print("[test_mpc] Reference generation failed")
        return

    print(f"[test_mpc] Reference length: {len(reference)}")
    print(f"[test_mpc] First ref point: x={reference[0][0]:.2f}, y={reference[0][1]:.2f}, psi={math.degrees(reference[0][2]):.2f}°, v={reference[0][3]:.2f}")

    # Initialize MPC
    mpc = MPCController(dt=mpc_dt, horizon=horizon)

    try:
        control = mpc.solve(current_state, reference, obstacles=[])
    except Exception as exc:
        print(f"[test_mpc] MPC solver failed to run: {exc}")
        return

    if control is None:
        print("[test_mpc] MPC returned no control (failed)")
    else:
        print(f"[test_mpc] First control: delta={control[0]:.4f} rad, accel={control[1]:.4f} m/s^2")

    # Get predicted trajectory
    pred = mpc.get_predicted_trajectory()
    if pred is not None:
        print(f"[test_mpc] Predicted trajectory length: {len(pred)}")
        for i, p in enumerate(pred):
            print(f"  {i:02d}: x={p[0]:.2f}, y={p[1]:.2f}, psi={math.degrees(p[2]):.2f}°")

    # Plot waypoints, reference, and predicted trajectory
    way_x, way_y = zip(*waypoints)
    ref_x, ref_y = reference[:,0], reference[:,1]
    
    plt.figure(figsize=(8,6))
    plt.plot(way_x, way_y, 'bo-', label='Waypoints')
    plt.plot(ref_x, ref_y, 'c.-', label='Reference Trajectory')
    
    if pred is not None:
        pred_x = [p[0] for p in pred]
        pred_y = [p[1] for p in pred]
        plt.plot(pred_x, pred_y, 'g.-', linewidth=2, label='MPC Prediction')

    plt.scatter([current_state[0]], [current_state[1]], color='red', s=100, label='Initial State')


    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.title('MPC Test: Waypoints, Reference, and Prediction')
    plt.show()


if __name__ == '__main__':
    main()
