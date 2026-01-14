"""
MPC Controller
==============

Nonlinear Model Predictive Controller using CasADi for optimization.
Includes integrated vehicle model and trajectory generation.

State: [x, y, psi, v]
Control: [delta, a]
"""

import numpy as np
import time
import casadi as ca
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import argparse
import math

class MPCController:
    """
    Nonlinear MPC Controller with integrated vehicle model and trajectory generation.
    
    Solves optimization problem at each timestep:
    
    min  Σ (w1*||p - p_ref||² + w2*||v - v_ref||² + w3*||Δu||²)
    s.t. x_{k+1} = f(x_k, u_k)           # Vehicle dynamics
         u_min ≤ u ≤ u_max               # Actuator limits
    """
    
    # Vehicle parameters (Tesla Model 3 approximation)
    WHEELBASE = 2.875           # meters
    MAX_STEER = np.radians(35)  # ~35 degrees
    MAX_STEER_RATE = np.radians(60)
    MAX_ACCEL = 4.0             # m/s^2
    MAX_DECEL = -5.0            # m/s^2
    MAX_SPEED = 50.0            # m/s
    MIN_SPEED = 0.0             # m/s
    
    def __init__(self, dt, horizon, trajectory_mode):
        """
        Initialize MPC controller.
        
        Args:
            dt: Control timestep (seconds)
            horizon: Prediction horizon (number of steps)
            trajectory_mode: 0 for heuristic (curvature-based), 1 for trapezoidal (forward-backward)
        """
        
        self.dt = dt
        self.horizon = horizon
        self.L = self.WHEELBASE
        self.trajectory_mode = trajectory_mode
        
        # Speed settings
        self.min_speed = 0.0
        
        # State dimensions
        self.n_states = 4   # [x, y, psi, v]
        self.n_controls = 2  # [delta, a]
        
        # Cost weights
        self.w_pos = 1.0
        self.w_heading = 1.0
        self.w_vel = 1.0
        self.w_delta = 1.0
        self.w_accel = 0.1
        self.w_delta_rate = 0.5
        self.w_accel_rate = 0.5
        
        # Spline for trajectory
        self._spline = None
        self._spline_length = 0
        self._waypoints = None
        
        # Previous solution for warm start
        self._prev_u = None
        self._prev_x = None
        self._last_solve_time_ms = 0.0
        
        # Fail counts
        self._solve_fail_count = 0
        self._generation_fail_count = 0
        
        # Build solver
        self._build_solver()
        
        print(f"[MPC] Initialized: dt={dt}s, horizon={horizon}, prediction={horizon*dt:.1f}s ahead")
    
    # ========================
    # Vehicle Dynamics
    # ========================
    
    def _get_dynamics(self):
        """Build CasADi dynamics function using Euler integration."""
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        v = ca.SX.sym('v')
        state = ca.vertcat(x, y, psi, v)
        
        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        control = ca.vertcat(delta, a)
        
        # Euler integration of kinematic bicycle model
        x_next = x + self.dt * v * ca.cos(psi)
        y_next = y + self.dt * v * ca.sin(psi)
        psi_next = psi + self.dt * (v / self.L) * ca.tan(delta)
        v_next = v + self.dt * a
        
        next_state = ca.vertcat(x_next, y_next, psi_next, v_next)
        
        return ca.Function('dynamics', [state, control], [next_state])
    
    # ========================
    # Trajectory Generation
    # ========================
    
    def set_waypoints(self, waypoints):
        """Fit a parametric B-spline to provided waypoints."""
        if len(waypoints) < 4:
            print(f"[MPC] Not enough waypoints ({len(waypoints)}), need at least 4")
            self._spline = None
            return False
        
        self._waypoints = np.array(waypoints)
        
        # Remove duplicate points
        unique_points = [self._waypoints[0]]
        for p in self._waypoints[1:]:
            if np.linalg.norm(p - unique_points[-1]) > 0.1:
                unique_points.append(p)
        
        if len(unique_points) < 4:
            print("[MPC] Not enough unique waypoints after deduplication")
            return False
        
        self._waypoints = np.array(unique_points)
        # Fit parametric spline
        tck, _ = splprep([self._waypoints[:, 0], self._waypoints[:, 1]], 
                         s=0.1, k=min(3, len(self._waypoints)-1))
        self._spline = tck
        
        # Compute path length
        u_dense = np.linspace(0, 1, 1000)
        x_dense, y_dense = splev(u_dense, tck)
        distances = np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2)
        self._spline_length = np.sum(distances)
        
        return True
    
    def _generate_reference(self, current_state, target_speed):
        """Generate reference trajectory from current state."""
        if self._spline is None:
            return None
            
        if self.trajectory_mode == 1:
            return self._generate_reference_trapezoidal(current_state, target_speed)
        else:
            return self._generate_reference_heuristic(current_state, target_speed)

    def _generate_reference_heuristic(self, current_state, target_speed):
        """Original heuristic generation (Curvature-based only)."""
        curr_x, curr_y = current_state[0], current_state[1]
        
        # Find closest point on spline
        u_start = self._find_closest_u(curr_x, curr_y)
        
        # Generate reference points
        reference = []
        u_current = u_start
        
        for i in range(self.horizon + 1):
            if u_current >= 1.0:
                u_current = 0.9999
            
            x, y = splev(u_current, self._spline)
            
            # Heading from spline tangent
            dx, dy = splev(u_current, self._spline, der=1)
            psi = np.arctan2(dy, dx)
            
            # Curvature-based speed
            d2x, d2y = splev(u_current, self._spline, der=2)
            curvature = abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-6)**1.5
            v_ref = self._speed_from_curvature(curvature, target_speed)
            
            reference.append([float(x), float(y), float(psi), float(v_ref)])
            
            # Advance along spline
            step_distance = v_ref * self.dt
            u_current = min(u_current + step_distance / (self._spline_length + 1e-6), 1.0)
        
        return np.array(reference)

    def _generate_reference_trapezoidal(self, current_state, target_speed):
        """Trapezoidal profile generation (Forward-Backward Integration)."""
        curr_x, curr_y = current_state[0], current_state[1]
        curr_v = current_state[3]
        
        # 1. Generate dense spatial path
        u_start = self._find_closest_u(curr_x, curr_y)
        
        # Look ahead distance: horizon * max_speed (approx) + buffer
        # We need enough spatial points to cover the horizon
        lookahead_dist = max(self.horizon * self.dt * target_speed * 1.5, 50.0)
        
        # Sample points along spline
        # Step size approx 1 meter
        num_points = int(lookahead_dist) + 10
        
        points = []
        u_curr = u_start
        dist_accum = 0
        
        # Create spatial points
        for _ in range(num_points):
            x, y = splev(u_curr, self._spline)
            dx, dy = splev(u_curr, self._spline, der=1)
            d2x, d2y = splev(u_curr, self._spline, der=2)
            psi = np.arctan2(dy, dx)
            curvature = abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-6)**1.5
            
            points.append({'x': x, 'y': y, 'psi': psi, 'k': curvature, 'u': u_curr})
            
            # Advance u
            step = 1.0 # 1 meter approx
            u_next = min(u_curr + step / (self._spline_length + 1e-6), 1.0)
            dist_accum += step
            u_curr = u_next
            if u_curr >= 1.0 and u_next >= 1.0: break
            
        if not points: return None
        
        # 2. Curvature Limit Pass
        max_lat_accel = 3.0 # m/s^2

        for p in points:
            # v_max = sqrt(a_lat_max / k)
            v_k = np.sqrt(max_lat_accel / (p['k'] + 1e-3))
            p['v_lim'] = min(target_speed, v_k)
            
        # 3. Backward Pass (Deceleration)
        max_decel = abs(self.MAX_DECEL) # Use class constant
        points[-1]['v_bw'] = points[-1]['v_lim'] # End condition: assume we can be at limit
        
        for i in range(len(points)-2, -1, -1):
            ds = np.sqrt((points[i+1]['x']-points[i]['x'])**2 + (points[i+1]['y']-points[i]['y'])**2)
            # v_i = sqrt(v_{i+1}^2 + 2 * a_dec * ds)
            v_bw = np.sqrt(points[i+1]['v_bw']**2 + 2 * max_decel * ds)
            points[i]['v_bw'] = min(points[i]['v_lim'], v_bw)
            
        # 4. Forward Pass (Acceleration)
        max_accel = self.MAX_ACCEL # Use class constant
        points[0]['v_fw'] = curr_v
        points[0]['v_final'] = min(points[0]['v_bw'], curr_v)
        
        for i in range(1, len(points)):
            ds = np.sqrt((points[i]['x']-points[i-1]['x'])**2 + (points[i]['y']-points[i-1]['y'])**2)
            # v_i = sqrt(v_{i-1}^2 + 2 * a_acc * ds)
            v_fw_pure = np.sqrt(points[i-1]['v_fw']**2 + 2 * max_accel * ds)
            points[i]['v_fw'] = min(points[i]['v_lim'], v_fw_pure)
            
            points[i]['v_final'] = min(points[i]['v_bw'], points[i]['v_fw'])
            
        # Store debug info
        self._last_trapezoidal_points = points
        # print("GENERATE")
        # for i, point in enumerate(points):
        #     print(f"{i}: {point}")
        # 5. Resample to Time Domain for MPC
        reference = []
        
        # Create interpolators
        t_vals = [0.0]
        s_vals = [0.0]
        v_vals = [points[0]['v_final']]
        u_vals_list = [points[0]['u']]
        
        for i in range(1, len(points)):
            ds = np.sqrt((points[i]['x']-points[i-1]['x'])**2 + (points[i]['y']-points[i-1]['y'])**2)
            v_start = points[i-1]['v_final']
            v_end = points[i]['v_final']
            
            # Avoid division by zero
            if v_start + v_end < 1e-4:
                # If velocity is effectively 0 for a segment, time delta is large
                # But since we strip duplicates, ds > 0. 
                # Theoretically dt -> infinity, but we cap it or assume small creep speed
                dt_segment = 1.0 # Fallback, shouldn't happen often due to fwd pass
            else:
                # Average velocity assumption for constant acceleration segment
                dt_segment = 2.0 * ds / (v_start + v_end)
            
            t_vals.append(t_vals[-1] + dt_segment)
            s_vals.append(s_vals[-1] + ds)
            v_vals.append(v_end)
            u_vals_list.append(points[i]['u'])

        # ==========================================================
        # 6. Resample to MPC Time Domain
        # ==========================================================
        
        reference = []
        
        for k in range(self.horizon + 1):
            target_t = k * self.dt
            
            # We interpolate based on TIME, not distance integration
            # If target_t exceeds our profile generation, interp creates a stop (const value)
            
            u_ref = np.interp(target_t, t_vals, u_vals_list)
            v_ref = np.interp(target_t, t_vals, v_vals)
            
            x, y = splev(u_ref, self._spline)
            dx, dy = splev(u_ref, self._spline, der=1)
            psi = np.arctan2(dy, dx)
            
            reference.append([float(x), float(y), float(psi), float(v_ref)])
            
        return np.array(reference)
    
    def _find_closest_u(self, x, y, num_samples=100):
        """Find spline parameter u closest to (x,y)."""
        u_samples = np.linspace(0, 1, num_samples)
        x_samples, y_samples = splev(u_samples, self._spline)
        
        distances = (x_samples - x)**2 + (y_samples - y)**2
        closest_idx = np.argmin(distances)
        
        # Refine search
        if 0 < closest_idx < num_samples - 1:
            u_fine = np.linspace(u_samples[closest_idx - 1], u_samples[closest_idx + 1], 50)
            x_fine, y_fine = splev(u_fine, self._spline)
            distances_fine = (x_fine - x)**2 + (y_fine - y)**2
            return u_fine[np.argmin(distances_fine)]
        
        return u_samples[closest_idx]
    
    def _speed_from_curvature(self, curvature, max_speed):
        """Speed profile based on curvature."""
        if curvature < 0.01:
            return max_speed
        elif curvature > 0.2:
            return self.min_speed
        else:
            t = (curvature - 0.01) / (0.2 - 0.01)
            return max_speed * (1 - t) + self.min_speed * t
    
    # ========================
    # MPC Solver
    # ========================
    
    def _build_solver(self):
        """Build the CasADi NLP solver."""
        X = ca.SX.sym('X', self.n_states, self.horizon + 1)
        U = ca.SX.sym('U', self.n_controls, self.horizon)
        P = ca.SX.sym('P', self.n_states + self.n_states * (self.horizon + 1))
        
        x0 = P[0:self.n_states]
        f = self._get_dynamics()
        
        cost = 0
        g = [X[:, 0] - x0]
        
        for k in range(self.horizon):
            xk = X[:, k]
            uk = U[:, k]
            
            ref_start = self.n_states + k * self.n_states
            ref_k = P[ref_start:ref_start + self.n_states]
            
            # Position tracking
            cost += self.w_pos * ((xk[0] - ref_k[0])**2 + (xk[1] - ref_k[1])**2)
            
            # Heading tracking
            heading_error = xk[2] - ref_k[2]
            cost += self.w_heading * ca.sin(heading_error/2)**2
            
            # Velocity tracking
            cost += self.w_vel * (xk[3] - ref_k[3])**2
            
            # Control effort
            cost += self.w_delta * uk[0]**2
            cost += self.w_accel * uk[1]**2
            
            # Control rate
            if k > 0:
                uk_prev = U[:, k-1]
                cost += self.w_delta_rate * (uk[0] - uk_prev[0])**2
                cost += self.w_accel_rate * (uk[1] - uk_prev[1])**2
            
            # Dynamics constraint
            g.append(X[:, k+1] - f(xk, uk))
        
        # Terminal cost
        ref_final = P[self.n_states + self.horizon * self.n_states:
                      self.n_states + (self.horizon + 1) * self.n_states]
        cost += self.w_pos * 2 * ((X[0, self.horizon] - ref_final[0])**2 + 
                                   (X[1, self.horizon] - ref_final[1])**2)
        
        g = ca.vertcat(*g)
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        nlp = {'x': opt_vars, 'f': cost, 'g': g, 'p': P}
        
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'print_time': 0,
            'ipopt.sb': 'yes',
        }
        
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
        
        # Bounds
        self.lbx = [-1e6] * self.n_states * (self.horizon + 1)
        self.ubx = [1e6] * self.n_states * (self.horizon + 1)
        
        for k in range(self.horizon + 1):
            self.lbx[k * self.n_states + 3] = self.MIN_SPEED
            self.ubx[k * self.n_states + 3] = self.MAX_SPEED
        
        for k in range(self.horizon):
            self.lbx.append(-self.MAX_STEER)
            self.ubx.append(self.MAX_STEER)
            self.lbx.append(self.MAX_DECEL)
            self.ubx.append(self.MAX_ACCEL)
        
        n_eq = self.n_states * (self.horizon + 1)
        self.lbg = [0] * n_eq
        self.ubg = [0] * n_eq
        
        print("[MPC] Solver built successfully")
    
    def generate_reference(self, current_state, target_speed=None):
        """
        Generate reference trajectory from current state.
        
        This is a public method that should be called before solve().
        
        Args:
            current_state: Current vehicle state [x, y, psi, v]
            target_speed: Target speed in m/s (optional)
            
        Returns:
            Reference trajectory as numpy array of shape (horizon+1, 4) or None if failed
        """
        reference = self._generate_reference(current_state, target_speed)
        
        if reference is None or len(reference) < self.horizon + 1:
            self._generation_fail_count += 1
            print("[MPC] Failed to generate reference trajectory")
            return None
        
        # Store for visualization
        self._last_reference = reference
        return reference
    
    def solve(self, current_state, reference):
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current vehicle state [x, y, psi, v]
            reference: Reference trajectory from generate_reference(), shape (horizon+1, 4)
            
        Returns:
            Optimal control [delta, a] or None if failed
        """
        t_start = time.time()
        
        if reference is None or len(reference) < self.horizon + 1:
            print("[MPC] Invalid reference trajectory")
            return None
        
        # Build parameter vector
        p = list(current_state)
        for k in range(self.horizon + 1):
            p.extend(reference[k] if k < len(reference) else reference[-1])
        
        # Initial guess
        if self._prev_x is not None and self._prev_u is not None:
            x0_guess = np.zeros((self.n_states, self.horizon + 1))
            x0_guess[:, :-1] = self._prev_x[:, 1:]
            x0_guess[:, -1] = self._prev_x[:, -1]
            
            u0_guess = np.zeros((self.n_controls, self.horizon))
            u0_guess[:, :-1] = self._prev_u[:, 1:]
            u0_guess[:, -1] = self._prev_u[:, -1]
        else:
            x0_guess = reference[:self.horizon+1].T
            u0_guess = np.zeros((self.n_controls, self.horizon))
        
        # Flatten with column-major order to match CasADi's reshape
        x0 = np.concatenate([x0_guess.flatten(order='F'), u0_guess.flatten(order='F')])
        
        try:
            sol = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, 
                             lbg=self.lbg, ubg=self.ubg, p=p)
            
            opt = np.array(sol['x']).flatten()
            n_state_vars = self.n_states * (self.horizon + 1)
            X_opt = opt[:n_state_vars].reshape((self.n_states, self.horizon + 1), order='F')
            U_opt = opt[n_state_vars:].reshape((self.n_controls, self.horizon), order='F')
            
            self._prev_x = X_opt
            self._prev_u = U_opt
            
            self._last_solve_time_ms = (time.time() - t_start) * 1000
            
            return U_opt[:, 0]
            
        except Exception as e:
            print(f"[MPC] Solver failed: {e}")
            self._solve_fail_count += 1
            return None
    
    # ========================
    # Accessors
    # ========================
    
    def get_predicted_trajectory(self):
        """Get the predicted trajectory from last solve."""
        if self._prev_x is None:
            return None
        return self._prev_x.T
    
    def get_reference_trajectory(self):
        """Get the reference trajectory from last solve."""
        return getattr(self, '_last_reference', None)
    
    def get_planned_controls(self):
        """Get the planned control sequence from last solve."""
        if self._prev_u is None:
            return None
        return self._prev_u.T
    
    def get_last_solve_time_ms(self):
        """Get the last solve time in milliseconds."""
        return self._last_solve_time_ms
    
    def plot_velocity_profile(self):
        """Plot the last generated trapezoidal velocity profile."""
        
        points = self._last_trapezoidal_points
        
        # Calculate distances
        dists = [0]
        for i in range(1, len(points)):
            d = np.sqrt((points[i]['x']-points[i-1]['x'])**2 + (points[i]['y']-points[i-1]['y'])**2)
            dists.append(dists[-1] + d)
            
        v_lim = [p['v_lim'] * 3.6 for p in points] # km/h
        v_bw = [p['v_bw'] * 3.6 for p in points]
        v_fw = [p['v_fw'] * 3.6 for p in points]
        v_final = [p['v_final'] * 3.6 for p in points]
        
        plt.figure(figsize=(10, 6))
        plt.plot(dists, v_lim, 'r--', label='Curvature Limit', alpha=0.5)
        plt.plot(dists, v_bw, 'g--', label='Backward Pass (Decel)', alpha=0.5)
        plt.plot(dists, v_fw, 'b--', label='Forward Pass (Accel)', alpha=0.5)
        plt.plot(dists, v_final, 'k-', label='Final Profile', linewidth=2)
        
        plt.title('Trapezoidal Velocity Profile Generation')
        plt.xlabel('Distance (m)')
        plt.ylabel('Speed (km/h)')
        plt.grid(True)
        plt.legend()
        plt.show()




# Test
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test MPC Controller")
    parser.add_argument('--trajectory-mode', type=int, default=0, choices=[0,1],
                        help='Trajectory mode: 0=heuristic (curvature-based), 1=trapezoidal (forward-backward)')
    args = parser.parse_args()

    # ---- EDIT THESE INITIAL CONDITIONS ----
    init_x = 405.7
    init_y = -95.4
    init_yaw_deg = -89.4  # degrees
    init_v = 0        # m/s
    horizon = 40
    mpc_dt = 0.05
    target_speed_kmh = 36.0  # km/h
    # --------------------------------------

    yaw_rad = math.radians(init_yaw_deg)

    # Build a small straight set of waypoints ahead of the initial pose
    waypoints = [
        (405.7, -95.4),
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
    mpc = MPCController(dt=mpc_dt, 
                        horizon=horizon, 
                        trajectory_mode=args.trajectory_mode)

    # Set waypoints (fits spline internally)
    success = mpc.set_waypoints(waypoints)
    if not success:
        print("[test_mpc] Failed to fit spline to waypoints")
        exit(1)

    # Current state
    current_state = np.array([init_x, init_y, yaw_rad, init_v])
    print(f"[test_mpc] Initial state: x={init_x:.2f}, y={init_y:.2f}, yaw={init_yaw_deg:.2f}°, v={init_v:.2f}")

    # Generate reference trajectory
    target_speed_ms = target_speed_kmh / 3.6
    reference = mpc.generate_reference(current_state, target_speed_ms)
    
    if reference is None:
        print("[test_mpc] Failed to generate reference trajectory")
        exit(1)
    
    # Solve MPC
    try:
        control = mpc.solve(current_state, reference)
    except Exception as exc:
        print(f"[test_mpc] MPC solver failed to run: {exc}")
        exit(1)

    if control is None:
        print("[test_mpc] MPC returned no control (failed)")
    else:
        print(f"[test_mpc] First control: delta={control[0]:.4f} rad ({math.degrees(control[0]):.2f}°), accel={control[1]:.4f} m/s^2")
        print(f"[test_mpc] Solve time: {mpc.get_last_solve_time_ms():.1f} ms")
        
        # Plot the velocity profile
        print("[test_mpc] Plotting velocity profile...")
        mpc.plot_velocity_profile()

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
    
    ref_x, ref_y = reference[:,0], reference[:,1]
    plt.plot(ref_x, ref_y, 'c.-', label='Reference Trajectory')

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
