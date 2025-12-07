"""
Heads-Up Display
================

On-screen display for telemetry and debug info.
"""

import os

try:
    import pygame
except ImportError:
    pygame = None

try:
    import numpy as np
except ImportError:
    np = None

from config import Settings


class HUD:
    """Heads-up display for rendering telemetry."""
    
    # Semantic label names for display
    SEMANTIC_LABEL_NAMES = {
        0: 'Unlabeled',
        1: 'Building',
        2: 'Fence',
        3: 'Other',
        4: 'Pedestrian',
        5: 'Pole',
        6: 'Road Line',
        7: 'Road',
        8: 'Sidewalk',
        9: 'Vegetation',
        10: 'Vehicle',
        11: 'Wall',
        12: 'Traffic Sign',
        13: 'Sky',
        14: 'Ground',
        15: 'Bridge',
        16: 'Rail Track',
        17: 'Guard Rail',
        18: 'Traffic Light',
        19: 'Static',
        20: 'Dynamic',
        21: 'Water',
        22: 'Terrain',
    }
    
    # Colors for obstacle warning (RGB)
    OBSTACLE_COLORS = {
        'clear': (0, 255, 0),      # Green
        'warning': (255, 255, 0),   # Yellow
        'danger': (255, 0, 0),      # Red
    }
    
    def __init__(self, width=None, height=None):
        """
        Initialize HUD.
        
        Args:
            width: Display width
            height: Display height
        """
        self.width = width or Settings.WINDOW_WIDTH
        self.height = height or Settings.WINDOW_HEIGHT
        
        if pygame is None:
            raise ImportError("pygame is required for HUD")
        
        # Initialize fonts
        pygame.font.init()
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else (fonts[0] if fonts else None)
        mono = pygame.font.match_font(mono) if mono else None
        
        self.font = pygame.font.Font(mono, 18)
        self.font_big = pygame.font.Font(mono, 24)
        self.font_small = pygame.font.Font(mono, 14)
    
    def render(self, display, telemetry, control=None, behavior_state=None):
        """
        Render HUD elements.
        
        Args:
            display: Pygame display surface
            telemetry: Dict with telemetry data
            control: carla.VehicleControl object (optional)
            behavior_state: Current behavior state string (optional)
        """
        # Semi-transparent background
        hud_surface = pygame.Surface((300, 280))
        hud_surface.set_alpha(180)
        hud_surface.fill((0, 0, 0))
        display.blit(hud_surface, (10, 10))
        
        # Title
        title = self.font_big.render("AutoIQ Autonomous", True, (0, 200, 255))
        display.blit(title, (20, 15))
        
        # Build info lines
        y_offset = 50
        info_lines = [
            f"Speed: {telemetry.get('speed_kmh', 0):.1f} km/h",
            f"Target: {telemetry.get('target_speed', 0):.1f} km/h",
        ]
        
        if 'road_id' in telemetry:
            info_lines.append(f"Road ID: {telemetry['road_id']}")
        
        if 'lane_id' in telemetry:
            info_lines.append(f"Lane ID: {telemetry['lane_id']}")
        
        if 'is_junction' in telemetry:
            info_lines.append(f"Junction: {'Yes' if telemetry['is_junction'] else 'No'}")
        
        if control is not None:
            info_lines.append(f"Steer: {control.steer:+.2f}")
            info_lines.append(f"Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f}")
        
        # Add behavior state from telemetry
        if 'behavior_state' in telemetry:
            behavior_state = telemetry['behavior_state']
        
        if behavior_state:
            # Color code behavior state
            state_color = (255, 255, 255)
            if 'LANE_CHANGING' in behavior_state:
                state_color = (255, 255, 0)  # Yellow for lane change
            elif 'EMERGENCY' in behavior_state:
                state_color = (255, 0, 0)  # Red for emergency
            elif 'FOLLOWING' in behavior_state:
                state_color = (0, 255, 0)  # Green for normal
            
            state_text = self.font.render(f"State: {behavior_state}", True, state_color)
            display.blit(state_text, (20, y_offset))
            y_offset += 22
        
        # Render lines
        for line in info_lines:
            text = self.font.render(line, True, (255, 255, 255))
            display.blit(text, (20, y_offset))
            y_offset += 22
        
        # Render obstacle detection info
        if 'obstacle' in telemetry:
            obstacle_info = telemetry['obstacle']
            if 'DETECTED' in obstacle_info:
                obstacle_color = (255, 0, 0)  # Red
            else:
                obstacle_color = (0, 255, 0)  # Green
            obstacle_text = self.font.render(f"Obstacle: {obstacle_info}", True, obstacle_color)
            display.blit(obstacle_text, (20, y_offset))
            y_offset += 22
    def render_perception_view(self, display, semantic_img, depth_img=None, x=10, y=240,
                                obstacle_detection=None, rgb_img=None):
        """
        Render small perception windows with obstacle detection overlay.
        
        Args:
            display: Pygame display surface
            semantic_img: Colored semantic image (numpy array)
            depth_img: Normalized depth image (numpy array)
            x, y: Position to render
            obstacle_detection: Dict with obstacle detection results (optional)
            rgb_img: Front RGB camera image (numpy array, optional)
        """
        scale = 0.25  # Scale down perception views
        
        # ===== RGB CAMERA VIEW (shows what camera actually sees) =====
        if rgb_img is not None:
            h, w = rgb_img.shape[:2]
            small_w, small_h = int(w * scale), int(h * scale)
            
            # Convert to pygame surface (RGB)
            if len(rgb_img.shape) == 3 and rgb_img.shape[2] >= 3:
                surface = pygame.surfarray.make_surface(
                    rgb_img[:, :, :3].swapaxes(0, 1)
                )
            else:
                surface = pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, (small_w, small_h))
            
            # Draw border and label
            pygame.draw.rect(display, (0, 255, 255), (x-2, y-2, small_w+4, small_h+4), 2)
            display.blit(surface, (x, y))
            
            label = self.font.render("RGB Front", True, (0, 255, 255))
            display.blit(label, (x, y + small_h + 5))
            
            y += small_h + 30
        
        # ===== SEMANTIC VIEW =====
        if semantic_img is not None:
            h, w = semantic_img.shape[:2]
            small_w, small_h = int(w * scale), int(h * scale)
            
            # Convert to pygame surface
            surface = pygame.surfarray.make_surface(
                semantic_img[:, :, ::-1].swapaxes(0, 1)
            )
            surface = pygame.transform.scale(surface, (small_w, small_h))
            
            # Draw border and label
            pygame.draw.rect(display, (255, 255, 255), (x-2, y-2, small_w+4, small_h+4), 2)
            display.blit(surface, (x, y))
            
            # Draw obstacle detection ROI overlay
            if obstacle_detection is not None:
                self._draw_obstacle_overlay(display, x, y, small_w, small_h, 
                                           scale, obstacle_detection)
            
            label = self.font.render("Semantic", True, (255, 255, 255))
            display.blit(label, (x, y + small_h + 5))
            
            y += small_h + 30
        
        if depth_img is not None:
            h, w = depth_img.shape[:2]
            small_w, small_h = int(w * scale), int(h * scale)
            
            surface = pygame.surfarray.make_surface(
                depth_img.swapaxes(0, 1)
            )
            surface = pygame.transform.scale(surface, (small_w, small_h))
            
            pygame.draw.rect(display, (255, 255, 255), (x-2, y-2, small_w+4, small_h+4), 2)
            display.blit(surface, (x, y))
            
            # Draw obstacle bounding box on depth view too
            if obstacle_detection and obstacle_detection.get('obstacle_detected'):
                bbox = obstacle_detection.get('bbox')
                if bbox:
                    x1, y1, x2, y2 = bbox
                    scaled_x1 = int(x1 * scale) + x
                    scaled_y1 = int(y1 * scale) + y
                    scaled_x2 = int(x2 * scale) + x
                    scaled_y2 = int(y2 * scale) + y
                    bbox_rect = pygame.Rect(scaled_x1, scaled_y1, 
                                            scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)
                    pygame.draw.rect(display, (255, 0, 0), bbox_rect, 2)
                
                center = obstacle_detection.get('center')
                if center:
                    scaled_cx = int(center[0] * scale) + x
                    scaled_cy = int(center[1] * scale) + y
                    pygame.draw.circle(display, (255, 0, 0), (scaled_cx, scaled_cy), 6, 2)
            
            label = self.font.render("Depth", True, (255, 255, 255))
            display.blit(label, (x, y + small_h + 5))
            
            y += small_h + 30
        
        # Render obstacle detection panel
        if obstacle_detection is not None:
            self._render_obstacle_panel(display, x, y, obstacle_detection)
    
    def _draw_obstacle_overlay(self, display, view_x, view_y, view_w, view_h, 
                               scale, obstacle_detection):
        """
        Draw obstacle detection overlay on semantic view.
        
        Args:
            display: Pygame display surface
            view_x, view_y: View position
            view_w, view_h: View dimensions
            scale: Scale factor
            obstacle_detection: Detection results
        """
        # Draw ROI rectangle (detection region)
        roi_left = int(Settings.CAMERA_WIDTH * 0.3 * scale)
        roi_right = int(Settings.CAMERA_WIDTH * 0.7 * scale)
        roi_top = int(Settings.CAMERA_HEIGHT * 0.4 * scale)
        roi_bottom = int(Settings.CAMERA_HEIGHT * 0.8 * scale)
        
        roi_rect = pygame.Rect(
            view_x + roi_left,
            view_y + roi_top,
            roi_right - roi_left,
            roi_bottom - roi_top
        )
        
        # Color based on detection status
        if obstacle_detection.get('obstacle_detected'):
            distance = obstacle_detection.get('distance', float('inf'))
            if distance < 10:
                roi_color = (255, 0, 0)  # Red - danger
            elif distance < 20:
                roi_color = (255, 255, 0)  # Yellow - warning
            else:
                roi_color = (255, 165, 0)  # Orange
        else:
            roi_color = (0, 255, 0)  # Green - clear
        
        pygame.draw.rect(display, roi_color, roi_rect, 2)
        
        # Draw obstacle center marker if detected
        if obstacle_detection.get('obstacle_detected'):
            # Draw bounding box around obstacle
            bbox = obstacle_detection.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                scaled_x1 = int(x1 * scale) + view_x
                scaled_y1 = int(y1 * scale) + view_y
                scaled_x2 = int(x2 * scale) + view_x
                scaled_y2 = int(y2 * scale) + view_y
                
                # Draw bounding box
                bbox_rect = pygame.Rect(scaled_x1, scaled_y1, 
                                        scaled_x2 - scaled_x1, scaled_y2 - scaled_y1)
                pygame.draw.rect(display, (255, 0, 0), bbox_rect, 2)
                
                # Draw label above bbox
                obstacle_type = obstacle_detection.get('obstacle_type', 'obstacle')
                distance = obstacle_detection.get('distance', 0)
                label_text = f"{obstacle_type.upper()} {distance:.1f}m"
                label_surface = self.font_small.render(label_text, True, (255, 255, 0))
                
                # Background for label
                label_bg = pygame.Surface((label_surface.get_width() + 4, label_surface.get_height()))
                label_bg.fill((0, 0, 0))
                label_bg.set_alpha(180)
                display.blit(label_bg, (scaled_x1, scaled_y1 - label_surface.get_height() - 2))
                display.blit(label_surface, (scaled_x1 + 2, scaled_y1 - label_surface.get_height() - 2))
            
            # Also draw crosshair at center
            center = obstacle_detection.get('center')
            if center:
                scaled_x = int(center[0] * scale) + view_x
                scaled_y = int(center[1] * scale) + view_y
                
                # Draw crosshair
                pygame.draw.line(display, (255, 0, 0), 
                               (scaled_x - 10, scaled_y), (scaled_x + 10, scaled_y), 2)
                pygame.draw.line(display, (255, 0, 0), 
                               (scaled_x, scaled_y - 10), (scaled_x, scaled_y + 10), 2)
    
    def _render_obstacle_panel(self, display, x, y, obstacle_detection):
        """
        Render obstacle detection info panel.
        
        Args:
            display: Pygame display surface
            x, y: Panel position
            obstacle_detection: Detection results
        """
        # Panel background
        panel_w, panel_h = 320, 100
        panel_surface = pygame.Surface((panel_w, panel_h))
        panel_surface.set_alpha(200)
        panel_surface.fill((30, 30, 30))
        display.blit(panel_surface, (x, y))
        
        # Panel border
        border_color = (255, 0, 0) if obstacle_detection.get('obstacle_detected') else (0, 255, 0)
        pygame.draw.rect(display, border_color, (x, y, panel_w, panel_h), 2)
        
        # Title
        title_color = (255, 100, 100) if obstacle_detection.get('obstacle_detected') else (100, 255, 100)
        title = self.font.render("OBSTACLE DETECTION", True, title_color)
        display.blit(title, (x + 10, y + 5))
        
        y_offset = y + 30
        
        if obstacle_detection.get('obstacle_detected'):
            # Detected - show details
            distance = obstacle_detection.get('distance', 0)
            position = obstacle_detection.get('position', 'unknown')
            pixel_count = obstacle_detection.get('pixel_count', 0)
            obstacle_type = obstacle_detection.get('obstacle_type', 'unknown')
            
            # Distance with color coding
            if distance < 10:
                dist_color = (255, 0, 0)
                status = "DANGER!"
            elif distance < 20:
                dist_color = (255, 255, 0)
                status = "WARNING"
            else:
                dist_color = (255, 165, 0)
                status = "CAUTION"
            
            status_text = self.font.render(f"Status: {status}", True, dist_color)
            display.blit(status_text, (x + 10, y_offset))
            y_offset += 20
            
            type_text = self.font.render(f"Type: {obstacle_type.upper()}", True, (255, 255, 255))
            display.blit(type_text, (x + 10, y_offset))
            y_offset += 20
            
            dist_text = self.font.render(f"Distance: {distance:.1f}m", True, dist_color)
            display.blit(dist_text, (x + 10, y_offset))
            
            pos_text = self.font.render(f"Pos: {position.upper()}", True, (200, 200, 200))
            display.blit(pos_text, (x + 170, y_offset))
            y_offset += 20
            
            # Show pixel count
            pixels_text = self.font_small.render(f"Pixels: {pixel_count}", True, (150, 150, 150))
            display.blit(pixels_text, (x + 10, y_offset))
        else:
            # No obstacle
            clear_text = self.font.render("Status: CLEAR", True, (0, 255, 0))
            display.blit(clear_text, (x + 10, y_offset))
            y_offset += 20
            
            info_text = self.font.render("No obstacles detected", True, (150, 150, 150))
            display.blit(info_text, (x + 10, y_offset))
