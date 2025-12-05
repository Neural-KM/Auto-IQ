"""
Tiled Display System
====================

Modular visualization system with configurable tiled panels.
Allows easy addition/removal of visualization components.
"""

import pygame
import numpy as np
from config import Settings


class Panel:
    """Base class for visualization panels."""
    
    def __init__(self, name, width, height):
        """
        Initialize panel.
        
        Args:
            name: Panel identifier
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.name = name
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.visible = True
        self.border_color = (100, 100, 100)
        self.title_color = (200, 200, 200)
        self.bg_color = (20, 20, 20)
        
        # Font setup
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
    
    def render(self, data=None):
        """
        Render panel content. Override in subclasses.
        
        Args:
            data: Data dict for rendering
            
        Returns:
            pygame.Surface
        """
        self.surface.fill(self.bg_color)
        return self.surface
    
    def draw_border(self, color=None):
        """Draw panel border."""
        color = color or self.border_color
        pygame.draw.rect(self.surface, color, (0, 0, self.width, self.height), 2)
    
    def draw_title(self, title, color=None):
        """Draw panel title."""
        color = color or self.title_color
        text = self.font.render(title, True, color)
        self.surface.blit(text, (5, 5))


class CameraPanel(Panel):
    """Panel for displaying camera feed."""
    
    def __init__(self, name, width, height, camera_name="Camera"):
        super().__init__(name, width, height)
        self.camera_name = camera_name
        self.border_color = (0, 200, 255)
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        if data and 'image' in data and data['image'] is not None:
            img = data['image']
            
            # Convert numpy array to surface
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3:
                    # Scale image to fit panel (leave room for title)
                    content_height = self.height - 25
                    h, w = img.shape[:2]
                    scale = min((self.width - 10) / w, content_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    # Handle BGR vs RGB
                    if img.shape[2] >= 3:
                        rgb_img = img[:, :, :3]
                        if data.get('bgr', False):
                            rgb_img = rgb_img[:, :, ::-1]
                        surface = pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
                    else:
                        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                    
                    surface = pygame.transform.scale(surface, (new_w, new_h))
                    
                    # Center image in panel
                    x_offset = (self.width - new_w) // 2
                    self.surface.blit(surface, (x_offset, 22))
        
        self.draw_title(self.camera_name, self.border_color)
        self.draw_border(self.border_color)
        return self.surface


class SemanticPanel(Panel):
    """Panel for semantic segmentation with obstacle overlay."""
    
    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.border_color = (255, 255, 255)
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        content_height = self.height - 25
        img_y_offset = 22
        
        if data and 'image' in data and data['image'] is not None:
            img = data['image']
            
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                scale = min((self.width - 10) / w, content_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Image is already in RGB format
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    surface = pygame.surfarray.make_surface(img[:, :, :3].swapaxes(0, 1))
                else:
                    surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                
                surface = pygame.transform.scale(surface, (new_w, new_h))
                x_offset = (self.width - new_w) // 2
                self.surface.blit(surface, (x_offset, img_y_offset))
                
                # Draw obstacle overlay
                obstacle = data.get('obstacle_detection')
                if obstacle:
                    self._draw_obstacle_overlay(x_offset, img_y_offset, new_w, new_h, 
                                                w, h, obstacle)
        
        self.draw_title("Semantic", self.border_color)
        self.draw_border(self.border_color)
        return self.surface
    
    def _draw_obstacle_overlay(self, view_x, view_y, view_w, view_h, 
                                orig_w, orig_h, obstacle):
        """Draw ROI and obstacle markers."""
        scale_x = view_w / orig_w
        scale_y = view_h / orig_h
        
        # Draw ROI rectangle
        roi_left = int(orig_w * 0.3 * scale_x)
        roi_right = int(orig_w * 0.7 * scale_x)
        roi_top = int(orig_h * 0.4 * scale_y)
        roi_bottom = int(orig_h * 0.8 * scale_y)
        
        roi_color = (0, 255, 0) if not obstacle.get('obstacle_detected') else (255, 0, 0)
        pygame.draw.rect(self.surface, roi_color,
                        (view_x + roi_left, view_y + roi_top,
                         roi_right - roi_left, roi_bottom - roi_top), 1)
        
        # Draw bounding box if obstacle detected
        if obstacle.get('obstacle_detected'):
            bbox = obstacle.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                sx1 = int(x1 * scale_x) + view_x
                sy1 = int(y1 * scale_y) + view_y
                sx2 = int(x2 * scale_x) + view_x
                sy2 = int(y2 * scale_y) + view_y
                pygame.draw.rect(self.surface, (255, 0, 0),
                               (sx1, sy1, sx2 - sx1, sy2 - sy1), 2)
                
                # Label
                obs_type = obstacle.get('obstacle_type', 'OBJ')
                dist = obstacle.get('distance', 0)
                label = f"{obs_type.upper()} {dist:.1f}m"
                label_surf = self.font_small.render(label, True, (255, 255, 0))
                self.surface.blit(label_surf, (sx1, sy1 - 12))
            
            # Crosshair at center
            center = obstacle.get('center')
            if center:
                cx = int(center[0] * scale_x) + view_x
                cy = int(center[1] * scale_y) + view_y
                pygame.draw.line(self.surface, (255, 0, 0), (cx - 8, cy), (cx + 8, cy), 2)
                pygame.draw.line(self.surface, (255, 0, 0), (cx, cy - 8), (cx, cy + 8), 2)


class DepthPanel(Panel):
    """Panel for depth visualization."""
    
    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.border_color = (150, 150, 255)
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        content_height = self.height - 25
        img_y_offset = 22
        
        if data and 'image' in data and data['image'] is not None:
            img = data['image']
            
            if isinstance(img, np.ndarray):
                h, w = img.shape[:2]
                scale = min((self.width - 10) / w, content_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                surface = pygame.transform.scale(surface, (new_w, new_h))
                x_offset = (self.width - new_w) // 2
                self.surface.blit(surface, (x_offset, img_y_offset))
                
                # Draw obstacle marker
                obstacle = data.get('obstacle_detection')
                if obstacle and obstacle.get('obstacle_detected'):
                    bbox = obstacle.get('bbox')
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        sx1 = int(x1 * (new_w / w)) + x_offset
                        sy1 = int(y1 * (new_h / h)) + img_y_offset
                        sx2 = int(x2 * (new_w / w)) + x_offset
                        sy2 = int(y2 * (new_h / h)) + img_y_offset
                        pygame.draw.rect(self.surface, (255, 0, 0),
                                       (sx1, sy1, sx2 - sx1, sy2 - sy1), 2)
        
        self.draw_title("Depth", self.border_color)
        self.draw_border(self.border_color)
        return self.surface


class DebugMaskPanel(Panel):
    """Panel for showing debug masks from obstacle detection."""
    
    def __init__(self, name, width, height, title="Debug", border_color=(255, 165, 0)):
        super().__init__(name, width, height)
        self.title = title
        self.border_color = border_color
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        content_height = self.height - 25
        img_y_offset = 22
        
        if data and 'image' in data and data['image'] is not None:
            img = data['image']
            
            if isinstance(img, np.ndarray) and img.size > 0:
                h, w = img.shape[:2]
                if h > 0 and w > 0:
                    scale = min((self.width - 10) / w, content_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    if new_w > 0 and new_h > 0:
                        # Handle both grayscale and color
                        if len(img.shape) == 2:
                            img = np.stack([img, img, img], axis=-1)
                        
                        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        surface = pygame.transform.scale(surface, (new_w, new_h))
                        x_offset = (self.width - new_w) // 2
                        self.surface.blit(surface, (x_offset, img_y_offset))
        
        # Show pixel count if available
        if data and 'pixel_count' in data:
            count_text = self.font_small.render(f"Pixels: {data['pixel_count']}", True, (200, 200, 200))
            self.surface.blit(count_text, (5, self.height - 18))
        
        self.draw_title(self.title, self.border_color)
        self.draw_border(self.border_color)
        return self.surface


class TelemetryPanel(Panel):
    """Panel for vehicle telemetry display."""
    
    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.border_color = (0, 200, 255)
        self.font = pygame.font.Font(None, 20)
        self.font_big = pygame.font.Font(None, 26)
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        y = 5
        
        # Title
        title = self.font_big.render("AutoIQ Autonomous", True, (0, 200, 255))
        self.surface.blit(title, (10, y))
        y += 28
        
        if data:
            # Behavior state with color
            state = data.get('behavior_state', 'N/A')
            state_color = (255, 255, 255)
            if 'LANE_CHANGING' in state:
                state_color = (255, 255, 0)
            elif 'EMERGENCY' in state:
                state_color = (255, 0, 0)
            elif 'FOLLOWING' in state:
                state_color = (0, 255, 0)
            
            state_text = self.font.render(f"State: {state}", True, state_color)
            self.surface.blit(state_text, (10, y))
            y += 22
            
            # Speed
            speed = data.get('speed_kmh', 0)
            target = data.get('target_speed', 0)
            speed_text = self.font.render(f"Speed: {speed:.1f} / {target:.1f} km/h", True, (255, 255, 255))
            self.surface.blit(speed_text, (10, y))
            y += 22
            
            # Road info
            road_id = data.get('road_id', 'N/A')
            lane_id = data.get('lane_id', 'N/A')
            road_text = self.font.render(f"Road: {road_id} | Lane: {lane_id}", True, (200, 200, 200))
            self.surface.blit(road_text, (10, y))
            y += 22
            
            # Control
            control = data.get('control')
            if control:
                ctrl_text = self.font.render(
                    f"Steer: {control.steer:+.2f} | Thr: {control.throttle:.2f} | Brk: {control.brake:.2f}",
                    True, (200, 200, 200)
                )
                self.surface.blit(ctrl_text, (10, y))
                y += 22
            
            # Navigation progress
            progress = data.get('nav_progress', 'N/A')
            remaining = data.get('nav_remaining', 'N/A')
            nav_text = self.font.render(f"Progress: {progress} | Remaining: {remaining}", True, (150, 200, 255))
            self.surface.blit(nav_text, (10, y))
        
        self.draw_border(self.border_color)
        return self.surface


class ObstaclePanel(Panel):
    """Panel for obstacle detection status."""
    
    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.font = pygame.font.Font(None, 20)
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        obstacle = data.get('obstacle_detection', {}) if data else {}
        detected = obstacle.get('obstacle_detected', False)
        
        # Set colors based on status
        if detected:
            distance = obstacle.get('distance', float('inf'))
            if distance < 10:
                self.border_color = (255, 0, 0)
                status = "DANGER!"
                status_color = (255, 0, 0)
            elif distance < 20:
                self.border_color = (255, 255, 0)
                status = "WARNING"
                status_color = (255, 255, 0)
            else:
                self.border_color = (255, 165, 0)
                status = "CAUTION"
                status_color = (255, 165, 0)
        else:
            self.border_color = (0, 255, 0)
            status = "CLEAR"
            status_color = (0, 255, 0)
        
        y = 5
        title = self.font.render("OBSTACLE DETECTION", True, status_color)
        self.surface.blit(title, (10, y))
        y += 24
        
        status_text = self.font.render(f"Status: {status}", True, status_color)
        self.surface.blit(status_text, (10, y))
        y += 22
        
        if detected:
            obs_type = obstacle.get('obstacle_type', 'unknown')
            type_text = self.font.render(f"Type: {obs_type.upper()}", True, (255, 255, 255))
            self.surface.blit(type_text, (10, y))
            y += 22
            
            distance = obstacle.get('distance', 0)
            dist_text = self.font.render(f"Distance: {distance:.1f}m", True, status_color)
            self.surface.blit(dist_text, (10, y))
            y += 22
            
            position = obstacle.get('position', 'unknown')
            pos_text = self.font.render(f"Position: {position.upper()}", True, (200, 200, 200))
            self.surface.blit(pos_text, (10, y))
        else:
            clear_text = self.font.render("No obstacles detected", True, (100, 100, 100))
            self.surface.blit(clear_text, (10, y))
        
        self.draw_border(self.border_color)
        return self.surface


class HelpPanel(Panel):
    """Panel for keyboard shortcuts."""
    
    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.border_color = (80, 80, 80)
        self.font = pygame.font.Font(None, 16)
    
    def render(self, data=None):
        self.surface.fill((30, 30, 30))
        
        help_text = "UP/DOWN: Speed | LEFT/RIGHT: Lane | SPACE: Pause | B: Reset | ESC: Quit"
        text = self.font.render(help_text, True, (180, 180, 180))
        self.surface.blit(text, (10, (self.height - text.get_height()) // 2))
        
        return self.surface


class TiledDisplay:
    """
    Manages a tiled display layout with multiple panels.
    
    Layout:
    +-------------------+------------------+
    |                   |  Camera/Semantic |
    |   Main Camera     +------------------+
    |                   |      Depth       |
    +-------------------+------------------+
    |  Telemetry  | Obstacle  |   Help     |
    +-------------+-----------+------------+
    """
    
    # Minimum panel heights
    MIN_BOTTOM_HEIGHT = 140
    MIN_SIDE_PANEL_HEIGHT = 150
    
    def __init__(self, width, height):
        """
        Initialize tiled display.
        
        Args:
            width: Total display width
            height: Total display height
        """
        self.width = width
        self.height = height
        self.panels = {}
        self.layout = []
        
        pygame.font.init()
        
        # Calculate panel sizes
        self._setup_default_layout()
    
    @staticmethod
    def get_recommended_size(base_width=1280, base_height=720):
        """
        Get recommended window size for the tiled layout.
        
        Args:
            base_width: Base width (default 1280)
            base_height: Base height (default 720)
            
        Returns:
            Tuple (width, height) with minimum required size
        """
        # Ensure minimum height for bottom panels + main content
        min_height = max(base_height, TiledDisplay.MIN_BOTTOM_HEIGHT + 480)
        # Ensure minimum width for side panels
        min_width = max(base_width, 1024)
        return (min_width, min_height)
    
    def resize(self, width, height):
        """
        Resize the display and reconfigure layout.
        
        Args:
            width: New width
            height: New height
        """
        self.width = width
        self.height = height
        # Reconfigure with current layout style
        current_layout = getattr(self, '_current_layout', 'default')
        self.reconfigure_layout(current_layout)
    
    def _setup_default_layout(self):
        """Setup default panel layout."""
        # Bottom panels need fixed minimum height for text
        bottom_h = 140
        
        # Main view takes remaining height
        main_h = self.height - bottom_h
        main_w = int(self.width * 0.72)
        
        # Side panels
        side_w = self.width - main_w
        side_h = main_h // 3
        
        # Create panels
        self.add_panel('main_camera', CameraPanel('main_camera', main_w, main_h, "Main View"), 
                      0, 0)
        
        self.add_panel('front_rgb', CameraPanel('front_rgb', side_w, side_h, "RGB Front"),
                      main_w, 0)
        
        self.add_panel('semantic', SemanticPanel('semantic', side_w, side_h),
                      main_w, side_h)
        
        self.add_panel('depth', DepthPanel('depth', side_w, side_h),
                      main_w, side_h * 2)
        
        # Bottom row
        telem_w = int(self.width * 0.45)
        obstacle_w = int(self.width * 0.30)
        help_w = self.width - telem_w - obstacle_w
        
        self.add_panel('telemetry', TelemetryPanel('telemetry', telem_w, bottom_h),
                      0, main_h)
        
        self.add_panel('obstacle', ObstaclePanel('obstacle', obstacle_w, bottom_h),
                      telem_w, main_h)
        
        self.add_panel('help', HelpPanel('help', help_w, bottom_h),
                      telem_w + obstacle_w, main_h)
    
    def add_panel(self, name, panel, x, y):
        """
        Add a panel to the display.
        
        Args:
            name: Panel identifier
            panel: Panel instance
            x, y: Position in display
        """
        self.panels[name] = {
            'panel': panel,
            'x': x,
            'y': y
        }
    
    def remove_panel(self, name):
        """Remove a panel from the display."""
        if name in self.panels:
            del self.panels[name]
    
    def get_panel(self, name):
        """Get a panel by name."""
        if name in self.panels:
            return self.panels[name]['panel']
        return None
    
    def toggle_panel(self, name):
        """Toggle panel visibility."""
        if name in self.panels:
            panel = self.panels[name]['panel']
            panel.visible = not panel.visible
    
    def render(self, display, data=None):
        """
        Render all panels to the display.
        
        Args:
            display: Pygame display surface
            data: Dict with data for each panel, keyed by panel name
        """
        # Clear display
        display.fill((0, 0, 0))
        
        # Render each panel
        for name, panel_info in self.panels.items():
            panel = panel_info['panel']
            if not panel.visible:
                continue
            
            # Get panel-specific data
            panel_data = data.get(name, {}) if data else {}
            
            # Render panel
            surface = panel.render(panel_data)
            
            # Blit to display
            display.blit(surface, (panel_info['x'], panel_info['y']))
    
    def reconfigure_layout(self, layout_name='default'):
        """
        Reconfigure panel layout.
        
        Args:
            layout_name: Name of layout preset
        """
        # Clear existing panels
        self.panels.clear()
        self._current_layout = layout_name
        
        if layout_name == 'default':
            self._setup_default_layout()
        elif layout_name == 'minimal':
            self._setup_minimal_layout()
        elif layout_name == 'debug':
            self._setup_debug_layout()
        elif layout_name == 'perception':
            self._setup_perception_debug_layout()
    
    def _setup_minimal_layout(self):
        """Setup minimal layout with just main camera."""
        main_h = self.height - 35
        self.add_panel('main_camera', CameraPanel('main_camera', self.width, main_h, "Main View"),
                      0, 0)
        self.add_panel('help', HelpPanel('help', self.width, 35),
                      0, main_h)
    
    def _setup_debug_layout(self):
        """Setup debug layout with all perception views."""
        # Bottom bar needs fixed height
        bottom_h = 140
        
        # 2x2 grid of cameras
        half_w = self.width // 2
        half_h = (self.height - bottom_h) // 2
        
        self.add_panel('main_camera', CameraPanel('main_camera', half_w, half_h, "Main View"),
                      0, 0)
        self.add_panel('front_rgb', CameraPanel('front_rgb', half_w, half_h, "RGB Front"),
                      half_w, 0)
        self.add_panel('semantic', SemanticPanel('semantic', half_w, half_h),
                      0, half_h)
        self.add_panel('depth', DepthPanel('depth', half_w, half_h),
                      half_w, half_h)
        
        # Bottom info bar
        bottom_y = half_h * 2
        
        self.add_panel('telemetry', TelemetryPanel('telemetry', self.width // 2, bottom_h),
                      0, bottom_y)
        self.add_panel('obstacle', ObstaclePanel('obstacle', self.width // 2, bottom_h),
                      self.width // 2, bottom_y)
    
    def _setup_perception_debug_layout(self):
        """Setup layout showing obstacle detection intermediate steps."""
        # This layout shows: Main cam, ROI semantic, Vehicle mask, Road mask, Raw obstacles, Filtered
        bottom_h = 100
        
        # 3x2 grid for debug views
        third_w = self.width // 3
        half_h = (self.height - bottom_h) // 2
        
        # Top row: Main camera, ROI Semantic, Combined debug
        self.add_panel('main_camera', CameraPanel('main_camera', third_w, half_h, "Main View"),
                      0, 0)
        self.add_panel('roi_semantic', DebugMaskPanel('roi_semantic', third_w, half_h, 
                                                       "ROI Semantic", (255, 255, 0)),
                      third_w, 0)
        self.add_panel('combined_debug', DebugMaskPanel('combined_debug', third_w, half_h,
                                                         "Combined", (0, 255, 0)),
                      third_w * 2, 0)
        
        # Bottom row: Vehicle mask, Raw obstacles, Filtered obstacles
        self.add_panel('vehicle_mask', DebugMaskPanel('vehicle_mask', third_w, half_h,
                                                       "Vehicles (raw)", (0, 100, 255)),
                      0, half_h)
        self.add_panel('raw_obstacles', DebugMaskPanel('raw_obstacles', third_w, half_h,
                                                        "Before Filter", (0, 255, 255)),
                      third_w, half_h)
        self.add_panel('filtered_obstacles', DebugMaskPanel('filtered_obstacles', third_w, half_h,
                                                             "After Filter", (0, 255, 0)),
                      third_w * 2, half_h)
        
        # Info bar
        bottom_y = half_h * 2
        self.add_panel('obstacle', ObstaclePanel('obstacle', self.width, bottom_h),
                      0, bottom_y)
