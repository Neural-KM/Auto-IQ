"""
PID Controller
==============

Generic PID controller for vehicle control.
"""

from collections import deque


class PIDController:
    """Generic PID Controller."""
    
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, dt=0.05):
        """
        Initialize PID controller.
        
        Args:
            k_p: Proportional gain
            k_i: Integral gain
            k_d: Derivative gain
            dt: Time step (seconds)
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.dt = dt
        self.error_buffer = deque(maxlen=20)
    
    def reset(self):
        """Reset the error buffer."""
        self.error_buffer.clear()
    
    def step(self, error):
        """
        Compute PID control output.
        
        Args:
            error: Current error value
            
        Returns:
            Control output
        """
        self.error_buffer.append(error)
        
        # Proportional term
        p_term = self.k_p * error
        
        # Derivative term
        if len(self.error_buffer) >= 2:
            d_term = self.k_d * (self.error_buffer[-1] - self.error_buffer[-2]) / self.dt
        else:
            d_term = 0.0
        
        # Integral term
        i_term = self.k_i * sum(self.error_buffer) * self.dt
        
        return p_term + i_term + d_term
    
    def set_gains(self, k_p=None, k_i=None, k_d=None):
        """Update PID gains."""
        if k_p is not None:
            self.k_p = k_p
        if k_i is not None:
            self.k_i = k_i
        if k_d is not None:
            self.k_d = k_d
