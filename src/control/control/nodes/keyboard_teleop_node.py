#!/usr/bin/env python3
"""
Keyboard Teleop Node (Pygame)
=============================

ROS2 node for controlling the vehicle using keyboard arrow keys.
Uses pygame window for reliable key capture.

Controls:
    UP/W    - Accelerate (increase throttle)
    DOWN/S  - Brake / Reverse
    LEFT/A  - Steer left
    RIGHT/D - Steer right
    SPACE   - Emergency brake (hand brake)
    R       - Toggle reverse
    Q/ESC   - Quit

Publishes:
    /carla/ego_vehicle/control (autoiq_msgs/VehicleControl)
"""

import pygame
import rclpy
from rclpy.node import Node

from autoiq_msgs.msg import VehicleControl


class KeyboardTeleopNode(Node):
    """
    Keyboard teleoperation node for vehicle control using pygame.
    """
    
    def __init__(self):
        super().__init__('keyboard_teleop_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('control_topic', '/carla/ego_vehicle/control')
        self.declare_parameter('publish_rate', 30.0)  # Hz
        self.declare_parameter('throttle_step', 0.05)
        self.declare_parameter('brake_step', 0.05)
        self.declare_parameter('steer_step', 0.08)
        self.declare_parameter('steer_decay', 0.06)  # Auto-center steering
        self.declare_parameter('throttle_decay', 0.03)  # Throttle decay when no input
        
        control_topic = self.get_parameter('control_topic').value
        publish_rate = self.get_parameter('publish_rate').value
        self.throttle_step = self.get_parameter('throttle_step').value
        self.brake_step = self.get_parameter('brake_step').value
        self.steer_step = self.get_parameter('steer_step').value
        self.steer_decay = self.get_parameter('steer_decay').value
        self.throttle_decay = self.get_parameter('throttle_decay').value
        
        # ==================== Control State ====================
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self.hand_brake = False
        self.reverse = False
        self.running = True
        
        # ==================== Pygame Setup ====================
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption('Auto-IQ Keyboard Teleop')
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.TEXT_COLOR = (220, 220, 220)
        self.ACCENT_COLOR = (0, 200, 100)
        self.WARN_COLOR = (255, 180, 0)
        self.BRAKE_COLOR = (255, 80, 80)
        
        # ==================== Publisher ====================
        self.control_pub = self.create_publisher(
            VehicleControl, control_topic, 10)
        
        # ==================== Timer ====================
        self.timer = self.create_timer(1.0 / publish_rate, self.control_loop)
        
        self.get_logger().info('Keyboard Teleop Node started (pygame)')
        self.get_logger().info(f'Publishing to: {control_topic}')
    
    def process_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.hand_brake = not self.hand_brake
                    self.get_logger().info(f'Hand brake: {"ON" if self.hand_brake else "OFF"}')
                elif event.key == pygame.K_r:
                    self.reverse = not self.reverse
                    self.get_logger().info(f'Gear: {"REVERSE" if self.reverse else "FORWARD"}')
        
        return True
    
    def update_control(self):
        """Update control values based on pressed keys."""
        keys = pygame.key.get_pressed()
        
        # Throttle / Brake
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.throttle = min(1.0, self.throttle + self.throttle_step)
            self.brake = 0.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            if self.throttle > 0.01:
                self.throttle = max(0.0, self.throttle - self.throttle_step * 2)
            else:
                self.throttle = 0.0
                self.brake = min(1.0, self.brake + self.brake_step)
        else:
            self.throttle = max(0.0, self.throttle - self.throttle_decay)
            self.brake = max(0.0, self.brake - self.brake_step * 0.5)
        
        # Steering
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.steer = min(1.0, self.steer + self.steer_step)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.steer = max(-1.0, self.steer - self.steer_step)
        else:
            # Auto-center steering
            if abs(self.steer) < self.steer_decay:
                self.steer = 0.0
            elif self.steer > 0:
                self.steer -= self.steer_decay
            else:
                self.steer += self.steer_decay
    
    def draw_ui(self):
        """Draw the pygame UI."""
        self.screen.fill(self.BG_COLOR)
        
        # Title
        title = self.font.render('Auto-IQ Teleop', True, self.TEXT_COLOR)
        self.screen.blit(title, (120, 15))
        
        # Controls info
        y_offset = 55
        controls = [
            '↑/W: Accelerate   ↓/S: Brake',
            '←/A: Left   →/D: Right',
            'SPACE: Hand Brake   R: Reverse',
            'Q/ESC: Quit'
        ]
        for text in controls:
            surf = self.small_font.render(text, True, (150, 150, 150))
            self.screen.blit(surf, (20, y_offset))
            y_offset += 22
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 60), (20, 145), (380, 145), 1)
        
        # Status display
        y_offset = 160
        
        # Throttle bar
        throttle_text = self.small_font.render(f'Throttle: {self.throttle:.2f}', True, self.TEXT_COLOR)
        self.screen.blit(throttle_text, (20, y_offset))
        pygame.draw.rect(self.screen, (60, 60, 60), (150, y_offset + 2, 200, 16))
        pygame.draw.rect(self.screen, self.ACCENT_COLOR, (150, y_offset + 2, int(200 * self.throttle), 16))
        y_offset += 30
        
        # Brake bar
        brake_text = self.small_font.render(f'Brake:    {self.brake:.2f}', True, self.TEXT_COLOR)
        self.screen.blit(brake_text, (20, y_offset))
        pygame.draw.rect(self.screen, (60, 60, 60), (150, y_offset + 2, 200, 16))
        pygame.draw.rect(self.screen, self.BRAKE_COLOR, (150, y_offset + 2, int(200 * self.brake), 16))
        y_offset += 30
        
        # Steering bar (centered)
        steer_text = self.small_font.render(f'Steer:   {self.steer:+.2f}', True, self.TEXT_COLOR)
        self.screen.blit(steer_text, (20, y_offset))
        pygame.draw.rect(self.screen, (60, 60, 60), (150, y_offset + 2, 200, 16))
        # Center marker
        pygame.draw.line(self.screen, (100, 100, 100), (250, y_offset + 2), (250, y_offset + 18), 2)
        # Steering indicator
        steer_pos = 250 + int(100 * self.steer)
        pygame.draw.rect(self.screen, self.WARN_COLOR, (min(steer_pos, 250), y_offset + 2, abs(steer_pos - 250), 16))
        y_offset += 35
        
        # Status indicators
        gear_color = self.WARN_COLOR if self.reverse else self.ACCENT_COLOR
        gear_text = 'REVERSE' if self.reverse else 'FORWARD'
        gear_surf = self.small_font.render(f'Gear: {gear_text}', True, gear_color)
        self.screen.blit(gear_surf, (20, y_offset))
        
        hb_color = self.BRAKE_COLOR if self.hand_brake else (100, 100, 100)
        hb_text = 'ON' if self.hand_brake else 'OFF'
        hb_surf = self.small_font.render(f'Hand Brake: {hb_text}', True, hb_color)
        self.screen.blit(hb_surf, (200, y_offset))
        
        pygame.display.flip()
    
    def control_loop(self):
        """Main control loop."""
        if not self.running:
            self.stop_vehicle()
            rclpy.shutdown()
            return
        
        # Process pygame events
        if not self.process_events():
            self.stop_vehicle()
            rclpy.shutdown()
            return
        
        # Update control based on keys
        self.update_control()
        
        # Draw UI
        self.draw_ui()
        
        # Publish control message
        msg = VehicleControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.throttle = self.throttle
        msg.brake = self.brake
        msg.steer = self.steer
        msg.hand_brake = self.hand_brake
        msg.reverse = self.reverse
        msg.manual_gear_shift = False
        msg.gear = 0
        
        self.control_pub.publish(msg)
    
    def stop_vehicle(self):
        """Send stop command to vehicle."""
        msg = VehicleControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.throttle = 0.0
        msg.brake = 1.0
        msg.steer = 0.0
        msg.hand_brake = True
        msg.reverse = False
        self.control_pub.publish(msg)
        self.get_logger().info('Vehicle stopped')
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        pygame.quit()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = KeyboardTeleopNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        pygame.quit()


if __name__ == '__main__':
    main()
