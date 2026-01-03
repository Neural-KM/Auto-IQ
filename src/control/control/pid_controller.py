"""
PID Controller
==============

Generic PID controller with anti-windup for vehicle control.
"""


class PIDController:
    """Generic PID Controller with anti-windup."""
    
    def __init__(self, k_p, k_i, k_d, dt, output_limits=(-1.0, 1.0)):
        """
        Initialize PID controller.
        
        Args:
            k_p: Proportional gain
            k_i: Integral gain
            k_d: Derivative gain
            dt: Time step (seconds)
            output_limits: Tuple of (min, max) output limits for anti-windup
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.dt = dt
        self.output_min, self.output_max = output_limits
        
        self.integral = 0.0
        self.prev_error = 0.0
    
    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def step(self, error):
        """
        Compute PID control output with anti-windup.
        
        Args:
            error: Current error value
            
        Returns:
            Control output (clamped to output_limits)
        """
        # Proportional term
        p_term = self.k_p * error
        
        # Derivative term
        d_term = self.k_d * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # Integral term (before anti-windup)
        self.integral += error * self.dt
        i_term = self.k_i * self.integral
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Anti-windup: clamp output and back-calculate integral
        if output > self.output_max:
            self.integral -= (output - self.output_max) / self.k_i if self.k_i != 0 else 0
            output = self.output_max
        elif output < self.output_min:
            self.integral -= (output - self.output_min) / self.k_i if self.k_i != 0 else 0
            output = self.output_min
        
        return output
