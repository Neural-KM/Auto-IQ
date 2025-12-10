"""
MPC Controller
==============

Nonlinear Model Predictive Controller using CasADi for optimization.
Replaces PID controller for smoother, predictive vehicle control.

Features:
- Kinematic bicycle model
- Obstacle avoidance constraints
- Smooth control outputs (penalizes jerk)
- Velocity tracking with curvature-based limits
"""

import numpy as np
import time
import casadi as ca

from control.vehicle_model import VehicleModel


class MPCController:
    """
    Nonlinear MPC Controller using CasADi.
    
    Solves optimization problem at each timestep:
    
    min  Σ (w1*||p - p_ref||² + w2*||v - v_ref||² + w3*||Δu||²)
    s.t. x_{k+1} = f(x_k, u_k)           # Vehicle dynamics
         u_min ≤ u ≤ u_max               # Actuator limits
         ||p - p_obs||² ≥ r_safe²        # Obstacle avoidance
    """
    
    def __init__(self, dt=0.05, horizon=20):
        """
        Initialize MPC controller.
        
        Args:
            dt: Control timestep (seconds), default 50ms = 20Hz
            horizon: Prediction horizon (number of steps)
        """

        self.dt = dt
        self.N = horizon  # Prediction horizon
        
        # Vehicle model
        self.model = VehicleModel()
        self.constraints = self.model.get_constraints()
        
        # State dimensions
        self.n_states = 4   # [x, y, psi, v]
        self.n_controls = 2  # [delta, a]
        
        # Cost weights (tunable)
        self.w_pos = 10.0       # Position tracking weight
        self.w_heading = 5.0    # Heading tracking weight
        self.w_vel = 2.0        # Velocity tracking weight
        self.w_delta = 50.0     # Steering effort weight
        self.w_accel = 10.0     # Acceleration effort weight
        self.w_delta_rate = 200.0  # Steering rate weight (smoothness)
        self.w_accel_rate = 50.0   # Acceleration rate weight
        
        # Obstacle avoidance
        self.obstacle_weight = 1000.0  # Soft constraint penalty
        self.max_obstacles = 10        # Maximum obstacles to consider
        
        # Previous solution for warm start
        self._prev_u = None
        self._prev_x = None
        
        # Build the solver
        self._build_solver()
        
        # Performance tracking
        self._solve_times = []
        
        print(f"[MPC] Initialized: dt={dt}s, horizon={horizon}, "
              f"prediction={horizon*dt:.1f}s ahead")
    
    def _build_solver(self):
        """Build the CasADi NLP solver."""
        
        # Symbolic variables
        # States over horizon: X = [x0, x1, ..., xN]
        X = ca.SX.sym('X', self.n_states, self.N + 1)
        # Controls over horizon: U = [u0, u1, ..., u_{N-1}]
        U = ca.SX.sym('U', self.n_controls, self.N)
        # Reference trajectory: [x_ref, y_ref, psi_ref, v_ref] for each step
        P = ca.SX.sym('P', self.n_states + self.n_states * (self.N + 1) + self.max_obstacles * 3)
        # P structure: [x0 (4)] [ref_0...ref_N (4*(N+1))] [obs_0...obs_M (3*M)]
        
        # Initial state from P
        x0 = P[0:self.n_states]
        
        # Get vehicle dynamics function
        f = self.model.get_casadi_dynamics_simple(self.dt)
        
        # Cost function
        cost = 0
        
        # Equality constraints (dynamics)
        g = []
        
        # Initial state constraint
        g.append(X[:, 0] - x0)
        
        for k in range(self.N):
            # Current state and control
            xk = X[:, k]
            uk = U[:, k]
            
            # Reference for this step (offset in P)
            ref_start = self.n_states + k * self.n_states
            ref_k = P[ref_start:ref_start + self.n_states]
            
            # Position tracking cost
            cost += self.w_pos * ((xk[0] - ref_k[0])**2 + (xk[1] - ref_k[1])**2)
            
            # Heading tracking cost (handle angle wrapping)
            heading_error = xk[2] - ref_k[2]
            cost += self.w_heading * ca.sin(heading_error/2)**2
            
            # Velocity tracking cost
            cost += self.w_vel * (xk[3] - ref_k[3])**2
            
            # Control effort cost
            cost += self.w_delta * uk[0]**2
            cost += self.w_accel * uk[1]**2
            
            # Control rate cost (smoothness)
            if k > 0:
                uk_prev = U[:, k-1]
                cost += self.w_delta_rate * (uk[0] - uk_prev[0])**2
                cost += self.w_accel_rate * (uk[1] - uk_prev[1])**2
            
            # Obstacle avoidance (soft constraints)
            obs_start = self.n_states + self.n_states * (self.N + 1)
            for i in range(self.max_obstacles):
                obs_idx = obs_start + i * 3
                obs_x = P[obs_idx]
                obs_y = P[obs_idx + 1]
                obs_r = P[obs_idx + 2]  # Safety radius (0 if no obstacle)
                
                # Distance to obstacle
                dist_sq = (xk[0] - obs_x)**2 + (xk[1] - obs_y)**2
                # Soft constraint: penalize if inside safety radius
                violation = ca.fmax(0, obs_r**2 - dist_sq)
                cost += self.obstacle_weight * violation
            
            # Dynamics constraint
            xk_next = f(xk, uk)
            g.append(X[:, k+1] - xk_next)
        
        # Terminal cost (final state tracking)
        ref_final = P[self.n_states + self.N * self.n_states:
                      self.n_states + (self.N + 1) * self.n_states]
        cost += self.w_pos * 2 * ((X[0, self.N] - ref_final[0])**2 + 
                                   (X[1, self.N] - ref_final[1])**2)
        
        # Flatten constraints
        g = ca.vertcat(*g)
        
        # Decision variables
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        
        # NLP problem
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': g,
            'p': P
        }
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'print_time': 0,
            'ipopt.sb': 'yes',  # Suppress banner
        }
        
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)
        
        # Constraint bounds
        n_vars = self.n_states * (self.N + 1) + self.n_controls * self.N
        
        # State bounds (loose)
        self.lbx = [-1e6] * self.n_states * (self.N + 1)
        self.ubx = [1e6] * self.n_states * (self.N + 1)
        
        # Velocity bounds
        for k in range(self.N + 1):
            self.lbx[k * self.n_states + 3] = self.constraints['v_min']
            self.ubx[k * self.n_states + 3] = self.constraints['v_max']
        
        # Control bounds
        for k in range(self.N):
            # Steering
            self.lbx.append(self.constraints['delta_min'])
            self.ubx.append(self.constraints['delta_max'])
            # Acceleration
            self.lbx.append(self.constraints['a_min'])
            self.ubx.append(self.constraints['a_max'])
        
        # Equality constraint bounds (dynamics must be satisfied exactly)
        n_eq = self.n_states * (self.N + 1)
        self.lbg = [0] * n_eq
        self.ubg = [0] * n_eq
        
        print("[MPC] Solver built successfully")
    
    def solve(self, current_state, reference_trajectory, obstacles=None):
        """
        Solve MPC optimization problem.
        
        Args:
            current_state: Current vehicle state [x, y, psi, v]
            reference_trajectory: Reference trajectory array [N+1, 4]
            obstacles: List of obstacles [(x, y, radius), ...] or None
            
        Returns:
            Optimal control [delta, a] or None if failed
        """
        t_start = time.time()
        
        # Validate inputs
        if reference_trajectory is None or len(reference_trajectory) < self.N + 1:
            print("[MPC] Invalid reference trajectory")
            return None
        
        # Build parameter vector P
        p = list(current_state)  # Initial state
        
        # Reference trajectory
        for k in range(self.N + 1):
            if k < len(reference_trajectory):
                p.extend(reference_trajectory[k])
            else:
                p.extend(reference_trajectory[-1])  # Repeat last if short
        
        # Obstacles (pad with zeros if fewer than max)
        if obstacles is None:
            obstacles = []
        
        for i in range(self.max_obstacles):
            if i < len(obstacles):
                p.extend([obstacles[i][0], obstacles[i][1], obstacles[i][2]])
            else:
                p.extend([0, 0, 0])  # No obstacle (radius 0)
        
        # Initial guess (warm start)
        if self._prev_x is not None and self._prev_u is not None:
            # Shift previous solution
            x0_guess = np.zeros((self.n_states, self.N + 1))
            x0_guess[:, :-1] = self._prev_x[:, 1:]
            x0_guess[:, -1] = self._prev_x[:, -1]
            
            u0_guess = np.zeros((self.n_controls, self.N))
            u0_guess[:, :-1] = self._prev_u[:, 1:]
            u0_guess[:, -1] = self._prev_u[:, -1]
        else:
            # Cold start: use reference trajectory
            x0_guess = reference_trajectory[:self.N+1].T
            u0_guess = np.zeros((self.n_controls, self.N))
        
        x0 = np.concatenate([x0_guess.flatten(), u0_guess.flatten()])
        
        # Solve
        try:
            sol = self.solver(
                x0=x0,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p
            )
            
            # Extract solution
            opt = np.array(sol['x']).flatten()
            
            # Parse states and controls
            n_state_vars = self.n_states * (self.N + 1)
            X_opt = opt[:n_state_vars].reshape(self.n_states, self.N + 1)
            U_opt = opt[n_state_vars:].reshape(self.n_controls, self.N)
            
            # Save for warm start
            self._prev_x = X_opt
            self._prev_u = U_opt
            
            # Get first control
            optimal_control = U_opt[:, 0]
            
            # Track solve time
            solve_time = time.time() - t_start
            self._solve_times.append(solve_time)
            if len(self._solve_times) > 100:
                self._solve_times.pop(0)
            
            return optimal_control
            
        except Exception as e:
            print(f"[MPC] Solver failed: {e}")
            return None
    
    def get_predicted_trajectory(self):
        """Get the predicted trajectory from last solve."""
        if self._prev_x is None:
            return None
        return self._prev_x.T  # [N+1, 4]
    
    def get_planned_controls(self):
        """Get the planned control sequence from last solve."""
        if self._prev_u is None:
            return None
        return self._prev_u.T  # [N, 2]
    
    def get_avg_solve_time(self):
        """Get average solve time in milliseconds."""
        if not self._solve_times:
            return 0
        return np.mean(self._solve_times) * 1000
    
    def reset(self):
        """Reset controller state (warm start cache)."""
        self._prev_x = None
        self._prev_u = None
        self._solve_times = []
    
    def set_weights(self, pos=None, heading=None, vel=None, delta=None, 
                    accel=None, delta_rate=None, accel_rate=None, obstacle=None):
        """
        Update cost function weights. Requires rebuilding solver.
        
        Args:
            pos: Position tracking weight
            heading: Heading tracking weight
            vel: Velocity tracking weight
            delta: Steering effort weight
            accel: Acceleration effort weight
            delta_rate: Steering rate weight
            accel_rate: Acceleration rate weight
            obstacle: Obstacle avoidance weight
        """
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
        if obstacle is not None:
            self.obstacle_weight = obstacle
        
        # Rebuild solver with new weights
        self._build_solver()
        self.reset()


class MPCControllerSimple:
    """
    Simplified MPC for faster computation when obstacles aren't present.
    Uses quadratic programming instead of full NLP.
    """
    
    def __init__(self, dt=0.05, horizon=15):
        """Initialize simplified MPC."""
        self.dt = dt
        self.N = horizon
        self.model = VehicleModel()
        
        # Just use the full MPC for now
        # Could implement QP-based linear MPC later for speed
        self._mpc = MPCController(dt=dt, horizon=horizon)
    
    def solve(self, current_state, reference_trajectory, obstacles=None):
        """Solve MPC problem."""
        return self._mpc.solve(current_state, reference_trajectory, obstacles)
    
    def reset(self):
        """Reset controller."""
        self._mpc.reset()


# Test
if __name__ == '__main__':
    if not CASADI_AVAILABLE:
        print("CasADi not available, cannot test MPC")
        exit(1)
    
    print("Testing MPC Controller...")
    
    # Create controller
    mpc = MPCController(dt=0.05, horizon=20)
    
    # Current state: at origin, heading east, 10 m/s
    current_state = np.array([0.0, 0.0, 0.0, 10.0])
    
    # Reference: straight line ahead
    reference = np.zeros((21, 4))
    for i in range(21):
        reference[i] = [i * 0.5, 0.0, 0.0, 10.0]  # Points ahead on x-axis
    
    # No obstacles
    obstacles = []
    
    # Solve
    print("\nSolving MPC (no obstacles)...")
    control = mpc.solve(current_state, reference, obstacles)
    
    if control is not None:
        print(f"Optimal control: steering={np.degrees(control[0]):.2f}°, accel={control[1]:.2f} m/s²")
        print(f"Solve time: {mpc.get_avg_solve_time():.1f} ms")
    else:
        print("MPC solve failed!")
    
    # Test with obstacle
    print("\nSolving MPC (with obstacle at x=5, y=0)...")
    obstacles = [(5.0, 0.0, 3.0)]  # Obstacle ahead
    control = mpc.solve(current_state, reference, obstacles)
    
    if control is not None:
        print(f"Optimal control: steering={np.degrees(control[0]):.2f}°, accel={control[1]:.2f} m/s²")
        print(f"Solve time: {mpc.get_avg_solve_time():.1f} ms")
        
        # Show predicted trajectory
        pred = mpc.get_predicted_trajectory()
        if pred is not None:
            print(f"Predicted trajectory (first 5 steps):")
            for i in range(min(5, len(pred))):
                print(f"  t={i*mpc.dt:.2f}s: x={pred[i,0]:.2f}, y={pred[i,1]:.2f}")
