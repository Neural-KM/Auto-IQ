"""
Tiled Display System - Modular visualization system with configurable panels.

This module is completely agnostic to data sources. It only manages panel
layout, sizing, and positioning. The caller (main.py) decides what data
goes into each slot.
"""

import pygame
import numpy as np
from config.visualization import (
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    BOTTOM_BAR_HEIGHT,
)


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


class ImagePanel(Panel):
    """
    Generic panel for displaying any image.
    
    Data format:
        {
            'image': np.ndarray,  # The image (RGB, BGR, or grayscale)
            'format': str,        # Optional: 'rgb', 'bgr', or 'grayscale' (default: 'rgb')
            'title': str,         # Optional: override panel title
        }
    """
    
    def __init__(self, name, width, height, title=None, border_color=(0, 200, 255)):
        super().__init__(name, width, height)
        self.default_title = title or name.replace('_', ' ').title()
        self.border_color = border_color
    
    def render(self, data=None):
        self.surface.fill(self.bg_color)
        
        # Get title (can be overridden per-frame)
        title = self.default_title
        if data and 'title' in data:
            title = data['title']
        
        if data and 'image' in data and data['image'] is not None:
            img = data['image']
            
            content_height = self.height - 25
            h, w = img.shape[:2]
            scale = min((self.width - 10) / w, content_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Handle grayscale images
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            
            # Handle color format: 'bgr', 'rgb' (default), or legacy 'bgr' boolean
            img_format = data.get('format', 'rgb')
            is_bgr = img_format == 'bgr' or data.get('bgr', False)
            
            if len(img.shape) == 3 and img.shape[2] >= 3:
                rgb_img = img[:, :, :3]
                if is_bgr:
                    rgb_img = rgb_img[:, :, ::-1]
                surface = pygame.surfarray.make_surface(rgb_img.swapaxes(0, 1))
            else:
                surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            
            surface = pygame.transform.scale(surface, (new_w, new_h))
            x_offset = (self.width - new_w) // 2
            self.surface.blit(surface, (x_offset, 22))
        
        self.draw_title(title, self.border_color)
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
        
        # FPS and MPC solve time on right side of title row
        if data:
            fps = data.get('fps', 0.0)
            mpc_solve_time = data.get('mpc_solve_time_ms')
            
            right_text = f"FPS: {fps:.1f}"
            if mpc_solve_time is not None:
                right_text += f" | MPC: {mpc_solve_time:.1f} ms"
                
            info_text = self.font.render(right_text, True, (200, 200, 200))
            info_x = self.width - info_text.get_width() - 10
            self.surface.blit(info_text, (info_x, y + 4))
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


class TiledDisplay:
    """
    Manages a tiled display layout with multiple panels.
    
    This class is completely agnostic to data sources. It only manages:
    - Panel layout (positions and sizes)
    - Panel rendering
    
    The caller provides:
    - Layout configuration (which slots to create)
    - Data for each slot (images with metadata)
    
    Layout types:
    - 'default': 1 main panel + N side panels + bottom bar
    - 'dual': 2 equal panels side by side + bottom bar
    - 'grid': NxM grid of panels + bottom bar
    - 'fullscreen': 1 large panel + bottom bar
    """
    
    def __init__(self):
        """Initialize tiled display with standard dimensions."""
        self.width = DISPLAY_WIDTH
        self.height = DISPLAY_HEIGHT
        self.panels = {}
        self._current_layout = 'default'
        self._layout_config = {}
        
        pygame.font.init()
        
        # Setup default layout with generic slot names
        self.setup_layout('default', {'main': 'main', 'side': ['side_0', 'side_1', 'side_2']})
    
    def setup_layout(self, layout_type, config=None):
        """
        Setup panel layout.
        
        Args:
            layout_type: 'default', 'dual', 'grid', or 'fullscreen'
            config: Layout-specific configuration dict:
                - 'default': {'main': 'slot_name', 'side': ['slot1', 'slot2', ...]}
                - 'dual': {'left': 'slot_name', 'right': 'slot_name'}
                - 'grid': {'slots': ['slot1', 'slot2', ...], 'cols': 2}
                - 'fullscreen': {'main': 'slot_name'}
        """
        self.panels.clear()
        self._current_layout = layout_type
        self._layout_config = config or {}
        
        if layout_type == 'default':
            self._setup_default_layout()
        elif layout_type == 'dual':
            self._setup_dual_layout()
        elif layout_type == 'grid':
            self._setup_grid_layout()
        elif layout_type == 'fullscreen':
            self._setup_fullscreen_layout()
        else:
            self._setup_default_layout()
    
    def _setup_default_layout(self):
        """Setup default layout: main panel + side panels + bottom bar."""
        config = self._layout_config
        main_slot = config.get('main', 'main')
        side_slots = config.get('side', [])
        
        bottom_h = BOTTOM_BAR_HEIGHT
        main_h = self.height - bottom_h
        
        if side_slots:
            main_w = int(self.width * 0.72)
            side_w = self.width - main_w
            side_h = main_h // max(len(side_slots), 1)
        else:
            main_w = self.width
            side_w = 0
            side_h = 0
        
        # Main panel
        panel = ImagePanel(main_slot, main_w, main_h)
        self.add_panel(main_slot, panel, 0, 0)
        
        # Side panels
        for i, slot_name in enumerate(side_slots):
            panel = ImagePanel(slot_name, side_w, side_h)
            self.add_panel(slot_name, panel, main_w, i * side_h)
        
        self._add_bottom_bar(main_h)
    
    def _setup_dual_layout(self):
        """Setup dual layout: two equal panels side by side + bottom bar."""
        config = self._layout_config
        left_slot = config.get('left', 'left')
        right_slot = config.get('right', 'right')
        
        bottom_h = BOTTOM_BAR_HEIGHT
        main_h = self.height - bottom_h
        panel_w = self.width // 2
        
        left_panel = ImagePanel(left_slot, panel_w, main_h)
        self.add_panel(left_slot, left_panel, 0, 0)
        
        right_panel = ImagePanel(right_slot, self.width - panel_w, main_h)
        self.add_panel(right_slot, right_panel, panel_w, 0)
        
        self._add_bottom_bar(main_h)
    
    def _setup_grid_layout(self):
        """Setup grid layout: NxM grid of panels + bottom bar."""
        config = self._layout_config
        slots = config.get('slots', ['slot_0', 'slot_1', 'slot_2', 'slot_3'])
        cols = config.get('cols', 2)
        
        num_slots = len(slots)
        if num_slots == 0:
            self._setup_default_layout()
            return
        
        rows = (num_slots + cols - 1) // cols
        
        bottom_h = BOTTOM_BAR_HEIGHT
        available_h = self.height - bottom_h
        
        cell_w = self.width // cols
        cell_h = available_h // rows
        
        for i, slot_name in enumerate(slots):
            row = i // cols
            col = i % cols
            x = col * cell_w
            y = row * cell_h
            
            # Last column may be wider to fill space
            w = cell_w if col < cols - 1 else self.width - x
            
            panel = ImagePanel(slot_name, w, cell_h)
            self.add_panel(slot_name, panel, x, y)
        
        self._add_bottom_bar(rows * cell_h)
    
    def _setup_fullscreen_layout(self):
        """Setup fullscreen layout: one large panel + bottom bar."""
        config = self._layout_config
        main_slot = config.get('main', 'main')
        
        bottom_h = BOTTOM_BAR_HEIGHT
        content_h = self.height - bottom_h
        
        panel = ImagePanel(main_slot, self.width, content_h)
        self.add_panel(main_slot, panel, 0, 0)
        
        self._add_bottom_bar(content_h)
    
    def _add_bottom_bar(self, y_position):
        """Add telemetry and obstacle panels at the bottom."""
        telem_w = self.width // 2
        obstacle_w = self.width - telem_w
        
        self.add_panel('telemetry', TelemetryPanel('telemetry', telem_w, BOTTOM_BAR_HEIGHT),
                      0, y_position)
        self.add_panel('obstacle', ObstaclePanel('obstacle', obstacle_w, BOTTOM_BAR_HEIGHT),
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
    
    def get_slot_names(self):
        """Get list of current slot names (excluding telemetry/obstacle)."""
        return [name for name in self.panels.keys() if name not in ('telemetry', 'obstacle')]
    
    def get_current_layout(self):
        """Get current layout type."""
        return self._current_layout
    
    def render(self, display, data=None):
        """
        Render all panels to the display.
        
        Args:
            display: Pygame display surface
            data: Dict with data for each panel slot, keyed by slot name.
                  Each entry should be: {'image': np.ndarray, 'format': 'rgb', 'title': 'Optional Title'}
        """
        display.fill((0, 0, 0))
        
        for name, panel_info in self.panels.items():
            panel = panel_info['panel']
            if not panel.visible:
                continue
            
            panel_data = data.get(name, {}) if data else {}
            surface = panel.render(panel_data)
            display.blit(surface, (panel_info['x'], panel_info['y']))
