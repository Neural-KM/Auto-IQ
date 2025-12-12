"""
Vehicle Model
=============

Kinematic Bicycle Model for Tesla Model 3 approximation.
Used by MPC controller for prediction.

State: [x, y, psi, v]
    x   - x position (meters)
    y   - y position (meters)  
    psi - heading angle (radians)
    v   - velocity (m/s)

Control: [delta, a]
    delta - steering angle (radians)
    a     - acceleration (m/s^2)
"""

import numpy as np
import casadi as ca
CASADI_AVAILABLE = True


class VehicleModel:
    """
    Kinematic Bicycle Model approximating Tesla Model 3.
    
    The bicycle model simplifies a 4-wheeled vehicle to 2 wheels
    (front and rear) along the centerline.
    """
    
    # Tesla Model 3 approximate parameters
    WHEELBASE = 2.875           # meters (distance between axles)
    MAX_STEER = np.radians(35)  # max steering angle ~35 degrees
    MAX_STEER_RATE = np.radians(60)  # max steering rate deg/s
    MAX_ACCEL = 4.0             # m/s^2 (comfortable acceleration)
    MAX_DECEL = -6.0            # m/s^2 (comfortable braking)
    MAX_SPEED = 50.0            # m/s (~180 km/h)
    MIN_SPEED = 0.0             # m/s
    
    # Vehicle dimensions for collision avoidance
    LENGTH = 4.694              # meters
    WIDTH = 1.849               # meters
    SAFETY_RADIUS = 2.5         # meters (circle encompassing vehicle + margin)
    
    def __init__(self, wheelbase=None):
        """
        Initialize vehicle model.
        
        Args:
            wheelbase: Vehicle wheelbase in meters (default: Model 3)
        """
        self.L = wheelbase if wheelbase else self.WHEELBASE
        
    def continuous_dynamics(self, state, control):
        """
        Continuous-time dynamics: dx/dt = f(x, u)
        
        Args:
            state: [x, y, psi, v]
            control: [delta, a]
            
        Returns:
            State derivative [dx/dt, dy/dt, dpsi/dt, dv/dt]
        """
        x, y, psi, v = state
        delta, a = control
        
        # Kinematic bicycle model equations
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = (v / self.L) * np.tan(delta)
        dv = a
        
        return np.array([dx, dy, dpsi, dv])
    
    def discrete_dynamics(self, state, control, dt):
        """
        Discrete-time dynamics using Euler integration.
        
        Args:
            state: Current state [x, y, psi, v]
            control: Control input [delta, a]
            dt: Time step (seconds)
            
        Returns:
            Next state [x, y, psi, v]
        """
        derivative = self.continuous_dynamics(state, control)
        next_state = state + derivative * dt
        
        # Normalize heading to [-pi, pi]
        next_state[2] = self._normalize_angle(next_state[2])
        
        # Clamp velocity
        next_state[3] = np.clip(next_state[3], self.MIN_SPEED, self.MAX_SPEED)
        
        return next_state
    
    def discrete_dynamics_rk4(self, state, control, dt):
        """
        Discrete-time dynamics using RK4 integration (more accurate).
        
        Args:
            state: Current state [x, y, psi, v]
            control: Control input [delta, a]
            dt: Time step (seconds)
            
        Returns:
            Next state [x, y, psi, v]
        """
        k1 = self.continuous_dynamics(state, control)
        k2 = self.continuous_dynamics(state + 0.5 * dt * k1, control)
        k3 = self.continuous_dynamics(state + 0.5 * dt * k2, control)
        k4 = self.continuous_dynamics(state + dt * k3, control)
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize heading
        next_state[2] = self._normalize_angle(next_state[2])
        
        # Clamp velocity
        next_state[3] = np.clip(next_state[3], self.MIN_SPEED, self.MAX_SPEED)
        
        return next_state
    
    def get_casadi_dynamics(self, dt):
        """
        Get CasADi symbolic dynamics function for MPC.
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            CasADi Function: f(state, control) -> next_state
        """
        if not CASADI_AVAILABLE:
            raise RuntimeError("CasADi not available")
        
        # Symbolic state and control
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        v = ca.SX.sym('v')
        state = ca.vertcat(x, y, psi, v)
        
        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        control = ca.vertcat(delta, a)
        
        # Continuous dynamics
        dx = v * ca.cos(psi)
        dy = v * ca.sin(psi)
        dpsi = (v / self.L) * ca.tan(delta)
        dv = a
        
        # RK4 integration
        k1 = ca.vertcat(dx, dy, dpsi, dv)
        
        x2 = x + 0.5 * dt * v * ca.cos(psi)
        y2 = y + 0.5 * dt * v * ca.sin(psi)
        psi2 = psi + 0.5 * dt * (v / self.L) * ca.tan(delta)
        v2 = v + 0.5 * dt * a
        k2 = ca.vertcat(
            v2 * ca.cos(psi2),
            v2 * ca.sin(psi2),
            (v2 / self.L) * ca.tan(delta),
            a
        )
        
        x3 = x + 0.5 * dt * k2[0]
        y3 = y + 0.5 * dt * k2[1]
        psi3 = psi + 0.5 * dt * k2[2]
        v3 = v + 0.5 * dt * a
        k3 = ca.vertcat(
            v3 * ca.cos(psi3),
            v3 * ca.sin(psi3),
            (v3 / self.L) * ca.tan(delta),
            a
        )
        
        x4 = x + dt * k3[0]
        y4 = y + dt * k3[1]
        psi4 = psi + dt * k3[2]
        v4 = v + dt * a
        k4 = ca.vertcat(
            v4 * ca.cos(psi4),
            v4 * ca.sin(psi4),
            (v4 / self.L) * ca.tan(delta),
            a
        )
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Create function
        f = ca.Function('vehicle_dynamics', [state, control], [next_state],
                        ['state', 'control'], ['next_state'])
        
        return f
    
    def get_casadi_dynamics_simple(self, dt):
        """
        Get simplified CasADi dynamics (Euler integration, faster).
        
        Args:
            dt: Time step (seconds)
            
        Returns:
            CasADi Function: f(state, control) -> next_state
        """
        if not CASADI_AVAILABLE:
            raise RuntimeError("CasADi not available")
        
        # Symbolic state and control
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        v = ca.SX.sym('v')
        state = ca.vertcat(x, y, psi, v)
        
        delta = ca.SX.sym('delta')
        a = ca.SX.sym('a')
        control = ca.vertcat(delta, a)
        
        # Euler integration
        x_next = x + dt * v * ca.cos(psi)
        y_next = y + dt * v * ca.sin(psi)
        psi_next = psi + dt * (v / self.L) * ca.tan(delta)
        v_next = v + dt * a
        
        next_state = ca.vertcat(x_next, y_next, psi_next, v_next)
        
        f = ca.Function('vehicle_dynamics', [state, control], [next_state],
                        ['state', 'control'], ['next_state'])
        
        return f
    
    @staticmethod
    def _normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def predict_trajectory(self, initial_state, controls, dt):
        """
        Predict trajectory given a sequence of controls.
        
        Args:
            initial_state: Starting state [x, y, psi, v]
            controls: List of controls [[delta, a], ...]
            dt: Time step (seconds)
            
        Returns:
            List of states including initial state
        """
        trajectory = [np.array(initial_state)]
        state = np.array(initial_state)
        
        for control in controls:
            state = self.discrete_dynamics_rk4(state, np.array(control), dt)
            trajectory.append(state.copy())
        
        return trajectory
    
    def get_constraints(self):
        """
        Get vehicle constraints for MPC.
        
        Returns:
            Dictionary with constraint bounds
        """
        return {
            'delta_min': -self.MAX_STEER,
            'delta_max': self.MAX_STEER,
            'a_min': self.MAX_DECEL,
            'a_max': self.MAX_ACCEL,
            'v_min': self.MIN_SPEED,
            'v_max': self.MAX_SPEED,
            'delta_rate_max': self.MAX_STEER_RATE,
        }


# Quick test
if __name__ == '__main__':
    model = VehicleModel()
    
    # Test trajectory prediction
    state = np.array([0.0, 0.0, 0.0, 10.0])  # Start at origin, heading east, 10 m/s
    controls = [[0.1, 0.0]] * 20  # Slight right turn, constant speed
    
    dt = 0.05
    trajectory = model.predict_trajectory(state, controls, dt)
    
    print("Predicted trajectory (first 5 states):")
    for i, s in enumerate(trajectory[:5]):
        print(f"  t={i*dt:.2f}s: x={s[0]:.2f}, y={s[1]:.2f}, psi={np.degrees(s[2]):.1f}°, v={s[3]:.1f}m/s")
    
    if CASADI_AVAILABLE:
        print("\nCasADi dynamics available ✓")
        f = model.get_casadi_dynamics_simple(dt)
        print(f"  Function: {f}")
