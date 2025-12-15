"""
Tiled Display System
====================

Modular visualization system with configurable tiled panels.
Automatically adapts to available sensors from config.
"""

import pygame
import numpy as np
from config.config import SENSORS
from config.visualization import get_layout_config, get_all_sensor_names


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


class LidarPanel(Panel):
    """Panel for LiDAR bird's eye view visualization."""
    
    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.border_color = (255, 255, 0)  # Yellow for LiDAR
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        content_height = self.height - 25
        img_y_offset = 22
        
        if data and 'image' in data and data['image'] is not None:
            img = data['image']
            
            if isinstance(img, np.ndarray) and img.size > 0:
                h, w = img.shape[:2]
                if h > 0 and w > 0:
                    # Scale to fit panel while maintaining aspect ratio
                    scale = min((self.width - 10) / w, content_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    if new_w > 0 and new_h > 0:
                        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                        surface = pygame.transform.scale(surface, (new_w, new_h))
                        x_offset = (self.width - new_w) // 2
                        self.surface.blit(surface, (x_offset, img_y_offset))
        
        # Draw legend
        legend_y = self.height - 18
        legend_text = self.font_small.render("BEV: Blue=Low  Green=Mid  Red=High", True, (150, 150, 150))
        self.surface.blit(legend_text, (5, legend_y))
        
        self.draw_title("LiDAR BEV", self.border_color)
        self.draw_border(self.border_color)
        return self.surface


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
        self.font_small = pygame.font.Font(None, 16)
    
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

            # Location (car position)
            loc = data.get('location') or {}
            lx = loc.get('x', 0.0)
            ly = loc.get('y', 0.0)
            lz = loc.get('z', 0.0)
            loc_text = self.font.render(f"Pos: x={lx:7.1f}  y={ly:7.1f}  z={lz:5.1f}", True, (180, 220, 255))
            self.surface.blit(loc_text, (10, y))
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
                
                # Visual control bars
                bar_width = min(self.width - 20, 300)
                bar_height = 12
                bar_x = 10
                center_x = bar_x + bar_width // 2
                
                # Steering bar (cyan indicator on gray bar)
                pygame.draw.rect(self.surface, (60, 60, 60), (bar_x, y, bar_width, bar_height))
                pygame.draw.line(self.surface, (100, 100, 100), (center_x, y), (center_x, y + bar_height), 1)
                steer_offset = int(control.steer * (bar_width // 2))
                steer_x = center_x + steer_offset
                pygame.draw.rect(self.surface, (0, 200, 255), (steer_x - 3, y, 6, bar_height))
                steer_label = self.font_small.render("STEER", True, (150, 150, 150))
                self.surface.blit(steer_label, (bar_x + bar_width + 5, y))
                y += bar_height + 4
                
                # Throttle/Brake bar (green right for throttle, red left for brake)
                pygame.draw.rect(self.surface, (60, 60, 60), (bar_x, y, bar_width, bar_height))
                pygame.draw.line(self.surface, (100, 100, 100), (center_x, y), (center_x, y + bar_height), 1)
                if control.throttle > 0:
                    throttle_w = int(control.throttle * (bar_width // 2))
                    pygame.draw.rect(self.surface, (0, 200, 0), (center_x, y, throttle_w, bar_height))
                if control.brake > 0:
                    brake_w = int(control.brake * (bar_width // 2))
                    pygame.draw.rect(self.surface, (200, 0, 0), (center_x - brake_w, y, brake_w, bar_height))
                tb_label = self.font_small.render("THR/BRK", True, (150, 150, 150))
                self.surface.blit(tb_label, (bar_x + bar_width + 5, y))
                y += bar_height + 4
            
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
    MIN_BOTTOM_HEIGHT = 175
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
        self._current_layout = 'default'
        self._fullscreen_sensor = '3rd_person'
        self._dropdown_open = False
        
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
        """Setup default panel layout dynamically based on available sensors."""
        # Get layout config (filters to only existing sensors)
        layout = get_layout_config('default')
        
        # Bottom panels need fixed minimum height for text
        bottom_h = 175
        
        # Main view takes remaining height
        main_h = self.height - bottom_h
        
        # Get side sensors that exist
        side_sensors = layout.get('side', [])
        
        # If we have side panels, split the width
        if side_sensors:
            main_w = int(self.width * 0.72)
            side_w = self.width - main_w
            side_h = main_h // max(len(side_sensors), 1)
        else:
            main_w = self.width
            side_w = 0
            side_h = 0
        
        # Create main camera panel
        main_sensor = layout.get('main')
        if main_sensor:
            panel = self._create_panel_for_sensor(main_sensor, main_w, main_h)
            self.add_panel(main_sensor, panel, 0, 0)
        
        # Create side panels dynamically
        for i, sensor_name in enumerate(side_sensors):
            panel = self._create_panel_for_sensor(sensor_name, side_w, side_h)
            self.add_panel(sensor_name, panel, main_w, i * side_h)
        
        # Bottom bar
        self._add_bottom_bar(main_h)
    
    def _get_sensor_type(self, sensor_name):
        """Get sensor type by name from config."""
        for sensor in SENSORS:
            if sensor['name'] == sensor_name:
                return sensor['type']
        return 'rgb'  # default
    
    def _create_panel_for_sensor(self, sensor_name, width, height):
        """Create appropriate panel type for a sensor."""
        sensor_type = self._get_sensor_type(sensor_name)
        
        if sensor_type == 'semantic':
            return SemanticPanel(sensor_name, width, height)
        elif sensor_type == 'depth':
            return DepthPanel(sensor_name, width, height)
        elif sensor_type == 'lidar':
            return LidarPanel(sensor_name, width, height)
        else:  # rgb or unknown
            return CameraPanel(sensor_name, width, height, sensor_name.replace('_', ' ').title())
    
    def _add_bottom_bar(self, y_position, height=175):
        """
        Add telemetry and obstacle panels at the bottom.
        Called by all layouts.
        
        Args:
            y_position: Y position for the bottom bar
            height: Height of the bottom bar
        """
        telem_w = self.width // 2
        obstacle_w = self.width - telem_w
        
        self.add_panel('telemetry', TelemetryPanel('telemetry', telem_w, height),
                      0, y_position)
        self.add_panel('obstacle', ObstaclePanel('obstacle', obstacle_w, height),
                      telem_w, y_position)
    
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
    
    def reconfigure_layout(self, layout_name='default'):
        """
        Reconfigure panel layout.
        
        Args:
            layout_name: Name of layout preset ('default', 'surround', or 'fullscreen')
        """
        # Clear existing panels
        self.panels.clear()
        self._current_layout = layout_name
        
        if layout_name == 'default':
            self._setup_default_layout()
        elif layout_name == 'surround':
            self._setup_surround_layout()
        elif layout_name == 'fullscreen':
            self._setup_fullscreen_layout()
        elif layout_name == 'lidar':
            self._setup_lidar_layout()
    
    def _setup_lidar_layout(self):
        """Setup layout with LiDAR BEV view alongside main camera."""
        # Bottom bar needs fixed height
        bottom_h = 175
        main_h = self.height - bottom_h
        
        # Split horizontally: main camera (left) and LiDAR BEV (right)
        main_w = int(self.width * 0.6)
        lidar_w = self.width - main_w
        
        # Main camera panel (3rd person view)
        main_panel = CameraPanel('3rd_person', main_w, main_h, '3rd Person')
        self.add_panel('3rd_person', main_panel, 0, 0)
        
        # LiDAR BEV panel
        lidar_panel = LidarPanel('lidar', lidar_w, main_h)
        self.add_panel('lidar', lidar_panel, main_w, 0)
        
        # Bottom bar
        self._add_bottom_bar(main_h)
    
    def _setup_surround_layout(self):
        """Setup surround layout showing all sensors in a grid."""
        layout = get_layout_config('surround')
        grid_sensors = layout.get('grid', [])
        
        # If no grid sensors defined, fall back to all sensors
        if not grid_sensors:
            grid_sensors = [s['name'] for s in SENSORS]
        
        num_sensors = len(grid_sensors)
        
        if num_sensors == 0:
            self._setup_default_layout()
            return
        
        # Bottom bar needs fixed height
        bottom_h = 175
        available_h = self.height - bottom_h
        
        # Calculate grid dimensions
        if num_sensors <= 2:
            cols, rows = num_sensors, 1
        elif num_sensors <= 4:
            cols, rows = 2, 2
        elif num_sensors <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 4, 2
        
        cell_w = self.width // cols
        cell_h = available_h // rows
        
        # Create panels in grid
        for i, sensor_name in enumerate(grid_sensors):
            row = i // cols
            col = i % cols
            x = col * cell_w
            y = row * cell_h
            
            panel = self._create_panel_for_sensor(sensor_name, cell_w, cell_h)
            self.add_panel(sensor_name, panel, x, y)
        
        # Bottom bar
        self._add_bottom_bar(rows * cell_h)
    
    def _setup_fullscreen_layout(self):
        """Setup fullscreen layout with dropdown sensor selector."""
        # Initialize selected sensor if not set
        if not hasattr(self, '_fullscreen_sensor'):
            self._fullscreen_sensor = '3rd_person'
        if not hasattr(self, '_dropdown_open'):
            self._dropdown_open = False
        
        # Header bar for dropdown
        header_h = 40
        bottom_h = 175
        content_h = self.height - header_h - bottom_h
        
        # Create fullscreen panel for current sensor
        panel = self._create_panel_for_sensor(self._fullscreen_sensor, self.width, content_h)
        self.add_panel('fullscreen_view', panel, 0, header_h)
        
        # Bottom bar
        self._add_bottom_bar(header_h + content_h)
    
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
            
            # For fullscreen layout, map fullscreen_view to selected sensor
            if name == 'fullscreen_view' and hasattr(self, '_fullscreen_sensor'):
                panel_data = data.get(self._fullscreen_sensor, {}) if data else {}
            
            # Render panel
            surface = panel.render(panel_data)
            
            # Blit to display
            display.blit(surface, (panel_info['x'], panel_info['y']))
        
        # Render dropdown if in fullscreen mode
        if self._current_layout == 'fullscreen':
            self._render_dropdown(display)
    
    def _render_dropdown(self, display):
        """Render the sensor dropdown selector."""
        sensors = get_all_sensor_names()
        
        # Dropdown button
        btn_w, btn_h = 200, 30
        btn_x, btn_y = 10, 5
        
        # Draw button background
        btn_color = (60, 60, 60) if not self._dropdown_open else (80, 80, 80)
        pygame.draw.rect(display, btn_color, (btn_x, btn_y, btn_w, btn_h))
        pygame.draw.rect(display, (100, 100, 100), (btn_x, btn_y, btn_w, btn_h), 1)
        
        # Draw current selection text
        font = pygame.font.Font(None, 24)
        text = font.render(f"Sensor: {self._fullscreen_sensor}", True, (255, 255, 255))
        display.blit(text, (btn_x + 10, btn_y + 7))
        
        # Draw arrow
        arrow_x = btn_x + btn_w - 20
        arrow_y = btn_y + btn_h // 2
        if self._dropdown_open:
            pygame.draw.polygon(display, (255, 255, 255), 
                              [(arrow_x, arrow_y + 5), (arrow_x + 10, arrow_y + 5), (arrow_x + 5, arrow_y - 5)])
        else:
            pygame.draw.polygon(display, (255, 255, 255), 
                              [(arrow_x, arrow_y - 5), (arrow_x + 10, arrow_y - 5), (arrow_x + 5, arrow_y + 5)])
        
        # Draw dropdown list if open
        if self._dropdown_open:
            list_y = btn_y + btn_h
            for i, sensor in enumerate(sensors):
                item_rect = pygame.Rect(btn_x, list_y + i * btn_h, btn_w, btn_h)
                
                # Highlight hovered item
                mouse_pos = pygame.mouse.get_pos()
                if item_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(display, (80, 100, 120), item_rect)
                else:
                    pygame.draw.rect(display, (50, 50, 50), item_rect)
                
                pygame.draw.rect(display, (100, 100, 100), item_rect, 1)
                
                # Draw sensor name
                item_text = font.render(sensor, True, (255, 255, 255))
                display.blit(item_text, (btn_x + 10, list_y + i * btn_h + 7))
    
    def handle_click(self, pos):
        """
        Handle mouse click for dropdown interaction.
        Call this from main event loop when in fullscreen layout.
        
        Args:
            pos: (x, y) mouse position
            
        Returns:
            True if click was handled, False otherwise
        """
        if self._current_layout != 'fullscreen':
            return False
        
        sensors = get_all_sensor_names()
        btn_w, btn_h = 200, 30
        btn_x, btn_y = 10, 5
        
        # Check if clicked on dropdown button
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        if btn_rect.collidepoint(pos):
            self._dropdown_open = not self._dropdown_open
            return True
        
        # Check if clicked on dropdown item
        if self._dropdown_open:
            list_y = btn_y + btn_h
            for i, sensor in enumerate(sensors):
                item_rect = pygame.Rect(btn_x, list_y + i * btn_h, btn_w, btn_h)
                if item_rect.collidepoint(pos):
                    self._fullscreen_sensor = sensor
                    self._dropdown_open = False
                    # Rebuild the layout with new sensor
                    self._setup_fullscreen_layout()
                    return True
            
            # Clicked outside dropdown - close it
            self._dropdown_open = False
            return True
        
        return False
