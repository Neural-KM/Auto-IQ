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

from config import Settings


class HUD:
    """Heads-up display for rendering telemetry."""
    
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
        hud_surface = pygame.Surface((300, 220))
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
        
        if behavior_state:
            info_lines.append(f"State: {behavior_state}")
        
        # Render lines
        for line in info_lines:
            text = self.font.render(line, True, (255, 255, 255))
            display.blit(text, (20, y_offset))
            y_offset += 22
        
        # Help text at bottom
        help_surface = pygame.Surface((450, 30))
        help_surface.set_alpha(180)
        help_surface.fill((0, 0, 0))
        display.blit(help_surface, (10, self.height - 40))
        
        help_text = self.font.render(
            "UP/DOWN: Speed | SPACE: Toggle View | ESC: Quit",
            True, (200, 200, 200)
        )
        display.blit(help_text, (20, self.height - 35))
    
    def render_perception_view(self, display, semantic_img, depth_img=None, x=10, y=240):
        """
        Render small perception windows.
        
        Args:
            display: Pygame display surface
            semantic_img: Colored semantic image (numpy array)
            depth_img: Normalized depth image (numpy array)
            x, y: Position to render
        """
        scale = 0.25  # Scale down perception views
        
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
            
            label = self.font.render("Depth", True, (255, 255, 255))
            display.blit(label, (x, y + small_h + 5))
