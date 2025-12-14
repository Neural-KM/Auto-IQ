"""
Simple MPC integration test harness.

This script constructs an MPCController (which now includes trajectory generation)
and a small set of waypoints. It then solves the MPC problem, printing the results.

Edit the initial conditions and waypoints below as needed.
"""

import numpy as np
import math

from control.mpc_controller import MPCController
import matplotlib.pyplot as plt

# ---- EDIT THESE INITIAL CONDITIONS ----
init_x = 405.7
init_y = -95.4
init_yaw_deg = -89.4  # degrees
init_v = 0        # m/s
horizon = 30
mpc_dt = 0.05
target_speed_kmh = 36.0  # km/h
# --------------------------------------

def main():
    yaw_rad = math.radians(init_yaw_deg)

    # Build a small straight set of waypoints ahead of the initial pose
    waypoints = [
        # (405.7, -95.4),
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
    print(f"Car initial state: x={init_x}, y={init_y}, yaw={init_yaw_deg}°, v={init_v}m/s")
    print(f"[test_mpc] Using {len(waypoints)} waypoints: first={waypoints[0]}, last={waypoints[-1]}")

    # Initialize MPC (includes trajectory generation)
    mpc = MPCController(dt=mpc_dt, horizon=horizon, target_speed=target_speed_kmh)
    
    # Set waypoints (fits spline internally)
    success = mpc.set_waypoints(waypoints)
    if not success:
        print("[test_mpc] Failed to fit spline to waypoints")
        return

    # Current state
    current_state = np.array([init_x, init_y, yaw_rad, init_v])
    print(f"[test_mpc] Initial state: x={init_x:.2f}, y={init_y:.2f}, yaw={init_yaw_deg:.2f}°, v={init_v:.2f}")

    # Solve MPC (generates reference internally)
    target_speed_ms = target_speed_kmh / 3.6
    try:
        control = mpc.solve(current_state, target_speed_ms)
    except Exception as exc:
        print(f"[test_mpc] MPC solver failed to run: {exc}")
        return

    if control is None:
        print("[test_mpc] MPC returned no control (failed)")
    else:
        print(f"[test_mpc] First control: delta={control[0]:.4f} rad ({math.degrees(control[0]):.2f}°), accel={control[1]:.4f} m/s^2")
        print(f"[test_mpc] Solve time: {mpc.get_avg_solve_time():.1f} ms")

    # Get predicted trajectory, reference, and planned controls
    pred = mpc.get_predicted_trajectory()
    reference = mpc.get_reference_trajectory()
    controls = mpc.get_planned_controls()
    
    # Print control values for entire horizon
    if controls is not None:
        print(f"\n[test_mpc] Planned controls for entire horizon ({len(controls)} steps):")
        for i, c in enumerate(controls):
            print(f"  k={i:02d}: delta={c[0]:.4f} rad ({math.degrees(c[0]):6.2f}°), accel={c[1]:6.3f} m/s^2")
    
    # Print reference trajectory
    if reference is not None:
        print(f"\n[test_mpc] Reference trajectory ({len(reference)} points):")
        for i, r in enumerate(reference):
            print(f"  k={i:02d}: x={r[0]:.2f}, y={r[1]:.2f}, psi={math.degrees(r[2]):6.2f}°, v={r[3]:.2f} m/s")
    
    if pred is not None:
        print(f"\n[test_mpc] Predicted trajectory length: {len(pred)}")
        for i, p in enumerate(pred): 
            print(f"  {i:02d}: x={p[0]:.2f}, y={p[1]:.2f}, psi={math.degrees(p[2]):.2f}°, v={p[3]:.2f}")

    # Plot waypoints, reference, and predicted trajectory
    way_x, way_y = zip(*waypoints)
    
    plt.figure(figsize=(8,6))
    plt.plot(way_x, way_y, 'bo-', label='Waypoints')
    
    if reference is not None:
        ref_x, ref_y = reference[:,0], reference[:,1]
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
