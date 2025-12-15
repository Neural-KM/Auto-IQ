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
    MAX_DECEL = -6.0            # m/s^2
    MAX_SPEED = 50.0            # m/s
    MIN_SPEED = 0.0             # m/s
    
    def __init__(self, dt=0.05, horizon=20, target_speed=30.0):
        """
        Initialize MPC controller.
        
        Args:
            dt: Control timestep (seconds)
            horizon: Prediction horizon (number of steps)
            target_speed: Target speed in km/h
        """
        self.dt = dt
        self.horizon = horizon
        self.L = self.WHEELBASE
        
        # Speed settings
        self.target_speed = target_speed / 3.6  # Convert to m/s
        self.min_speed = 5.0 / 3.6
        
        # State dimensions
        self.n_states = 4   # [x, y, psi, v]
        self.n_controls = 2  # [delta, a]
        
        # Cost weights
        self.w_pos = 1.0
        self.w_heading = 1.0
        self.w_vel = 1.0
        self.w_delta = 0.1
        self.w_accel = 0.1
        self.w_delta_rate = 0.1
        self.w_accel_rate = 0.1
        
        # Spline for trajectory
        self._spline = None
        self._spline_length = 0
        self._waypoints = None
        
        # Previous solution for warm start
        self._prev_u = None
        self._prev_x = None
        self._last_solve_time_ms = 0.0
        
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
    
    def _generate_reference(self, current_state, target_speed=None):
        """Generate reference trajectory from current state."""
        if self._spline is None:
            return None
        
        curr_x, curr_y = current_state[0], current_state[1]
        speed = target_speed if target_speed else self.target_speed
        
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
            v_ref = self._speed_from_curvature(curvature, speed)
            
            reference.append([float(x), float(y), float(psi), float(v_ref)])
            
            # Advance along spline
            step_distance = v_ref * self.dt
            u_current = min(u_current + step_distance / (self._spline_length + 1e-6), 1.0)
        
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
    
    def solve(self, current_state, target_speed=None):
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current vehicle state [x, y, psi, v]
            target_speed: Target speed in m/s (optional)
            
        Returns:
            Optimal control [delta, a] or None if failed
        """
        t_start = time.time()
        
        # Generate reference trajectory
        reference = self._generate_reference(current_state, target_speed)
        if reference is None or len(reference) < self.horizon + 1:
            print("[MPC] Failed to generate reference trajectory")
            return None
        
        # Store for visualization
        self._last_reference = reference
        
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
    
    def reset(self):
        """Reset controller state."""
        self._prev_x = None
        self._prev_u = None
        self._last_solve_time_ms = 0.0
        self._spline = None
        self._spline_length = 0
        self._waypoints = None
    
    def set_weights(self, pos=None, heading=None, vel=None, delta=None, 
                    accel=None, delta_rate=None, accel_rate=None):
        """Update cost function weights. Requires rebuilding solver."""
        if pos is not None:
            self.w_pos = pos
        if heading is not None:
            self.w_heading = heading
        if vel is not None:
            self.w_vel = vel
        if delta is not None:
            self.w_delta = delta
        if accel is not None:
            self.w_accel = accel
        if delta_rate is not None:
            self.w_delta_rate = delta_rate
        if accel_rate is not None:
            self.w_accel_rate = accel_rate
        
        self._build_solver()
        self.reset()


# Test
if __name__ == '__main__':
    print("Testing MPC Controller...")
    
    mpc = MPCController(dt=0.05, horizon=20, target_speed=36.0)
    
    # Set waypoints
    waypoints = [(i * 2.0, 0.0) for i in range(30)]
    mpc.set_waypoints(waypoints)
    
    # Current state
    current_state = np.array([0.0, 0.0, 0.0, 10.0])
    
    # Solve
    print("\nSolving MPC...")
    control = mpc.solve(current_state)
    
    if control is not None:
        print(f"Optimal control: steering={np.degrees(control[0]):.2f}°, accel={control[1]:.2f} m/s²")
        print(f"Solve time: {mpc.get_avg_solve_time():.1f} ms")
        
        pred = mpc.get_predicted_trajectory()
        if pred is not None:
            print(f"Predicted trajectory (first 5 steps):")
            for i in range(min(5, len(pred))):
                print(f"  t={i*mpc.dt:.2f}s: x={pred[i,0]:.2f}, y={pred[i,1]:.2f}")
    else:
        print("MPC solve failed!")
