"""
Global Route Planner
====================

High-level route planning from origin to destination using CARLA's road network.
Provides waypoint-based routes with navigation instructions (turn left, right, etc.)
"""

import math
from enum import IntEnum
from collections import deque

import carla
import numpy as np

try:
    import networkx as nx
except ImportError:
    raise RuntimeError('networkx required. pip install networkx')


class RoadOption(IntEnum):
    """
    Represents possible navigation decisions at waypoints.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class GlobalPlanner:
    """
    Global route planner using CARLA's road network topology.
    
    Computes routes from origin to destination and provides:
    - List of waypoints along the route
    - Navigation instructions (RoadOption) at each waypoint
    - Route progress tracking
    """
    
    def __init__(self, world_map, sampling_resolution=2.0):
        """
        Initialize global planner.
        
        Args:
            world_map: CARLA map instance
            sampling_resolution: Distance between waypoints in meters
        """
        self.map = world_map
        self.sampling_resolution = sampling_resolution
        
        # Graph representation of road network
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None
        
        # Route state
        self._route = []
        self._route_index = 0
        self._destination = None
        self._origin = None
        
        # For turn decision tracking
        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID
        
        # Build the road network graph
        print("[GlobalPlanner] Building road network graph...")
        self._build_topology()
        self._build_graph()
        self._find_loose_ends()
        self._lane_change_link()
        print(f"[GlobalPlanner] Graph built with {self._graph.number_of_nodes()} nodes")
    
    def set_destination(self, origin, destination):
        """
        Compute route from origin to destination.
        
        Args:
            origin: carla.Location or (x, y, z) tuple
            destination: carla.Location or (x, y, z) tuple
            
        Returns:
            True if route found, False otherwise
        """
        # Convert to carla.Location if needed
        if isinstance(origin, (tuple, list)):
            origin = carla.Location(x=origin[0], y=origin[1], z=origin[2] if len(origin) > 2 else 0)
        if isinstance(destination, (tuple, list)):
            destination = carla.Location(x=destination[0], y=destination[1], z=destination[2] if len(destination) > 2 else 0)
        
        self._origin = origin
        self._destination = destination
        self._route_index = 0
        
        # Compute route
        self._route = self.trace_route(origin, destination)
        
        if not self._route:
            print(f"[GlobalPlanner] No route found from {origin} to {destination}")
            return False
        
        print(f"[GlobalPlanner] Route computed: {len(self._route)} waypoints")
        return True
    
    def trace_route(self, origin, destination):
        """
        Compute route trace from origin to destination.
        
        Args:
            origin: carla.Location - start point
            destination: carla.Location - end point
            
        Returns:
            List of (carla.Waypoint, RoadOption) tuples
        """
        route_trace = []
        
        try:
            route = self._path_search(origin, destination)
        except nx.NetworkXNoPath:
            return []
        
        if not route:
            return []
        
        current_waypoint = self.map.get_waypoint(origin)
        destination_waypoint = self.map.get_waypoint(destination)
        
        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)
            edge = self._graph.edges[route[i], route[i + 1]]
            path = []
            
            if edge['type'] not in (RoadOption.LANEFOLLOW, RoadOption.VOID):
                # Junction/intersection handling
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge['exit_waypoint']
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._graph.edges[n1, n2]
                
                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path']) - 1, closest_index + 5)
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))
            else:
                # Lane following
                path = [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                closest_index = self._find_closest_in_list(current_waypoint, path)
                
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    
                    # Check if we're close to destination
                    if len(route) - i <= 2:
                        dist_to_dest = waypoint.transform.location.distance(destination)
                        if dist_to_dest < 2 * self.sampling_resolution:
                            break
                        if (current_waypoint.road_id == destination_waypoint.road_id and
                            current_waypoint.section_id == destination_waypoint.section_id and
                            current_waypoint.lane_id == destination_waypoint.lane_id):
                            destination_index = self._find_closest_in_list(destination_waypoint, path)
                            if closest_index > destination_index:
                                break
        
        return route_trace
    
    def get_route(self):
        """Get the complete computed route."""
        return self._route
    
    def get_next_waypoint(self):
        """
        Get next waypoint in route.
        
        Returns:
            Tuple of (carla.Waypoint, RoadOption) or (None, None) if route complete
        """
        if self._route_index >= len(self._route):
            return None, None
        return self._route[self._route_index]
    
    def get_upcoming_waypoints(self, count=10):
        """
        Get next N waypoints from current position.
        
        Args:
            count: Number of waypoints to return
            
        Returns:
            List of (carla.Waypoint, RoadOption) tuples
        """
        end_index = min(self._route_index + count, len(self._route))
        return self._route[self._route_index:end_index]
    
    def advance(self, vehicle_location, threshold=2.0):
        """
        Advance route index based on vehicle position.
        
        Args:
            vehicle_location: carla.Location of vehicle
            threshold: Distance threshold to consider waypoint reached
            
        Returns:
            True if route updated, False if route complete
        """
        if not self._route or self._route_index >= len(self._route):
            return False
        
        # Find closest waypoint in upcoming route
        min_dist = float('inf')
        closest_index = self._route_index
        
        # Only search within reasonable range ahead
        search_range = min(self._route_index + 20, len(self._route))
        
        for i in range(self._route_index, search_range):
            wp, _ = self._route[i]
            dist = vehicle_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        # If we've passed waypoints, advance index
        if closest_index > self._route_index:
            self._route_index = closest_index
        
        # Also advance if very close to current waypoint
        if min_dist < threshold and self._route_index < len(self._route) - 1:
            self._route_index += 1
        
        return self._route_index < len(self._route)
    
    def get_progress(self):
        """
        Get route progress as percentage.
        
        Returns:
            Float between 0.0 and 1.0
        """
        if not self._route:
            return 0.0
        return self._route_index / len(self._route)
    
    def get_remaining_distance(self):
        """
        Estimate remaining distance to destination.
        
        Returns:
            Distance in meters
        """
        if not self._route or self._route_index >= len(self._route):
            return 0.0
        
        remaining = 0.0
        for i in range(self._route_index, len(self._route) - 1):
            wp1, _ = self._route[i]
            wp2, _ = self._route[i + 1]
            remaining += wp1.transform.location.distance(wp2.transform.location)
        
        return remaining
    
    def is_destination_reached(self, vehicle_location, threshold=5.0):
        """
        Check if destination has been reached.
        
        Args:
            vehicle_location: carla.Location of vehicle
            threshold: Distance threshold in meters
            
        Returns:
            True if destination reached
        """
        if self._destination is None:
            return False
        return vehicle_location.distance(self._destination) < threshold
    
    def get_current_road_option(self):
        """Get the road option for current waypoint."""
        if not self._route or self._route_index >= len(self._route):
            return RoadOption.VOID
        return self._route[self._route_index][1]
    
    def draw_route(self, world, life_time=60.0, color=None):
        """
        Draw the route in the CARLA world.
        
        Args:
            world: CARLA world instance
            life_time: How long to display (seconds)
            color: carla.Color (default: green)
        """
        if not self._route:
            return
        
        if color is None:
            color = carla.Color(0, 255, 0)  # Green
        
        debug = world.debug
        
        for wp, road_option in self._route:
            debug.draw_point(
                wp.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=color,
                life_time=life_time
            )
    
    # =========================================================================
    # Internal methods for graph building and path search
    # =========================================================================
    
    def _build_topology(self):
        """Build topology from CARLA map."""
        self._topology = []
        
        for segment in self.map.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            
            # Round off to avoid floating point issues
            x1, y1, z1 = round(l1.x, 0), round(l1.y, 0), round(l1.z, 0)
            x2, y2, z2 = round(l2.x, 0), round(l2.y, 0), round(l2.z, 0)
            
            seg_dict = {
                'entry': wp1,
                'exit': wp2,
                'entryxyz': (x1, y1, z1),
                'exitxyz': (x2, y2, z2),
                'path': []
            }
            
            # Generate intermediate waypoints
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self.sampling_resolution:
                next_wps = wp1.next(self.sampling_resolution)
                if next_wps:
                    w = next_wps[0]
                    while w.transform.location.distance(endloc) > self.sampling_resolution:
                        seg_dict['path'].append(w)
                        next_ws = w.next(self.sampling_resolution)
                        if not next_ws:
                            break
                        w = next_ws[0]
            else:
                next_wps = wp1.next(self.sampling_resolution)
                if next_wps:
                    seg_dict['path'].append(next_wps[0])
            
            self._topology.append(seg_dict)
    
    def _build_graph(self):
        """Build networkx graph from topology."""
        self._graph = nx.DiGraph()
        self._id_map = {}
        self._road_id_to_edge = {}
        
        for segment in self._topology:
            entry_xyz = segment['entryxyz']
            exit_xyz = segment['exitxyz']
            path = segment['path']
            entry_wp = segment['entry']
            exit_wp = segment['exit']
            intersection = entry_wp.is_junction
            road_id = entry_wp.road_id
            section_id = entry_wp.section_id
            lane_id = entry_wp.lane_id
            
            # Add vertices
            for vertex in [entry_xyz, exit_xyz]:
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            
            # Store road_id to edge mapping
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = {}
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = {}
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
            
            # Compute edge vectors
            entry_loc = entry_wp.transform.location
            exit_loc = exit_wp.transform.location
            
            entry_vec = entry_wp.transform.get_forward_vector()
            exit_vec = exit_wp.transform.get_forward_vector()
            
            net_vec = np.array([
                exit_loc.x - entry_loc.x,
                exit_loc.y - entry_loc.y,
                0
            ])
            norm = np.linalg.norm(net_vec)
            if norm > 0.001:
                net_vec = net_vec / norm
            
            # Add edge
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1,
                path=path,
                entry_waypoint=entry_wp,
                exit_waypoint=exit_wp,
                entry_vector=np.array([entry_vec.x, entry_vec.y, 0]),
                exit_vector=np.array([exit_vec.x, exit_vec.y, 0]),
                net_vector=net_vec,
                intersection=intersection,
                type=RoadOption.LANEFOLLOW
            )
    
    def _find_loose_ends(self):
        """Find and connect loose ends in the graph."""
        count_loose_ends = 0
        hop_resolution = self.sampling_resolution
        
        for segment in self._topology:
            end_wp = segment['exit']
            exit_xyz = segment['exitxyz']
            road_id = end_wp.road_id
            section_id = end_wp.section_id
            lane_id = end_wp.lane_id
            
            if road_id in self._road_id_to_edge and \
               section_id in self._road_id_to_edge[road_id] and \
               lane_id in self._road_id_to_edge[road_id][section_id]:
                continue
            
            # Try to find connection
            next_wps = end_wp.next(hop_resolution)
            if next_wps:
                next_wp = next_wps[0]
                next_road_id = next_wp.road_id
                next_section_id = next_wp.section_id
                next_lane_id = next_wp.lane_id
                
                if next_road_id in self._road_id_to_edge and \
                   next_section_id in self._road_id_to_edge[next_road_id] and \
                   next_lane_id in self._road_id_to_edge[next_road_id][next_section_id]:
                    
                    n1 = self._id_map[exit_xyz]
                    n2, _ = self._road_id_to_edge[next_road_id][next_section_id][next_lane_id]
                    
                    if not self._graph.has_edge(n1, n2):
                        self._graph.add_edge(
                            n1, n2,
                            length=1,
                            path=[],
                            entry_waypoint=end_wp,
                            exit_waypoint=next_wp,
                            entry_vector=None,
                            exit_vector=None,
                            net_vector=None,
                            intersection=False,
                            type=RoadOption.LANEFOLLOW
                        )
                        count_loose_ends += 1
    
    def _lane_change_link(self):
        """Add lane change connections to the graph."""
        for segment in self._topology:
            left_found, right_found = False, False
            
            for waypoint in segment['path']:
                if not waypoint.is_junction:
                    # Check left lane
                    if not left_found:
                        left_wp = waypoint.get_left_lane()
                        if left_wp and left_wp.lane_type == carla.LaneType.Driving and \
                           waypoint.lane_id * left_wp.lane_id > 0:
                            self._add_lane_change_edge(segment, left_wp, RoadOption.CHANGELANELEFT)
                            left_found = True
                    
                    # Check right lane
                    if not right_found:
                        right_wp = waypoint.get_right_lane()
                        if right_wp and right_wp.lane_type == carla.LaneType.Driving and \
                           waypoint.lane_id * right_wp.lane_id > 0:
                            self._add_lane_change_edge(segment, right_wp, RoadOption.CHANGELANERIGHT)
                            right_found = True
                
                if left_found and right_found:
                    break
    
    def _add_lane_change_edge(self, segment, target_wp, road_option):
        """Add a lane change edge to the graph."""
        target_road_id = target_wp.road_id
        target_section_id = target_wp.section_id
        target_lane_id = target_wp.lane_id
        
        if target_road_id not in self._road_id_to_edge:
            return
        if target_section_id not in self._road_id_to_edge[target_road_id]:
            return
        if target_lane_id not in self._road_id_to_edge[target_road_id][target_section_id]:
            return
        
        next_node, _ = self._road_id_to_edge[target_road_id][target_section_id][target_lane_id]
        current_node = self._id_map[segment['entryxyz']]
        
        if not self._graph.has_edge(current_node, next_node):
            self._graph.add_edge(
                current_node, next_node,
                length=50,  # Higher cost for lane changes
                path=[],
                entry_waypoint=segment['entry'],
                exit_waypoint=target_wp,
                entry_vector=None,
                exit_vector=None,
                net_vector=None,
                intersection=False,
                type=road_option
            )
    
    def _path_search(self, origin, destination):
        """
        Find shortest path using A* algorithm.
        
        Args:
            origin: carla.Location
            destination: carla.Location
            
        Returns:
            List of node IDs representing the path
        """
        start = self.map.get_waypoint(origin)
        end = self.map.get_waypoint(destination)
        
        start_road_id = start.road_id
        start_section_id = start.section_id
        start_lane_id = start.lane_id
        
        end_road_id = end.road_id
        end_section_id = end.section_id
        end_lane_id = end.lane_id
        
        # Find start and end nodes
        start_edge = self._road_id_to_edge.get(start_road_id, {}).get(start_section_id, {}).get(start_lane_id)
        end_edge = self._road_id_to_edge.get(end_road_id, {}).get(end_section_id, {}).get(end_lane_id)
        
        if start_edge is None or end_edge is None:
            raise nx.NetworkXNoPath("No valid start or end edge found")
        
        start_node = start_edge[0]
        end_node = end_edge[0]
        
        # A* search with euclidean heuristic
        def heuristic(n1, n2):
            v1 = self._graph.nodes[n1]['vertex']
            v2 = self._graph.nodes[n2]['vertex']
            return math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
        
        route = nx.astar_path(
            self._graph, start_node, end_node,
            heuristic=heuristic,
            weight='length'
        )
        
        return route
    
    def _turn_decision(self, index, route):
        """
        Determine the turn decision at a route waypoint.
        
        Args:
            index: Index in route
            route: List of node IDs
            
        Returns:
            RoadOption for the turn decision
        """
        decision = None
        previous_node = route[index]
        current_node = route[index + 1]
        
        # Get edge type
        edge = self._graph.edges[previous_node, current_node]
        edge_type = edge['type']
        
        if edge_type in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            return edge_type
        
        if edge_type == RoadOption.LANEFOLLOW and edge['intersection']:
            # At intersection, compute turn based on angle
            if index + 2 < len(route):
                next_node = route[index + 2]
                next_edge = self._graph.edges[current_node, next_node]
                
                if edge['exit_vector'] is not None and next_edge['net_vector'] is not None:
                    cross = np.cross(edge['exit_vector'][:2], next_edge['net_vector'][:2])
                    dot = np.dot(edge['exit_vector'][:2], next_edge['net_vector'][:2])
                    
                    if cross > 0.5:
                        decision = RoadOption.LEFT
                    elif cross < -0.5:
                        decision = RoadOption.RIGHT
                    elif dot > 0:
                        decision = RoadOption.STRAIGHT
                    else:
                        decision = RoadOption.LANEFOLLOW
                else:
                    decision = RoadOption.LANEFOLLOW
            else:
                decision = RoadOption.LANEFOLLOW
        else:
            decision = RoadOption.LANEFOLLOW
        
        self._previous_decision = decision
        return decision
    
    def _find_closest_in_list(self, waypoint, waypoint_list):
        """Find index of closest waypoint in list."""
        min_distance = float('inf')
        closest_index = 0
        
        for i, wp in enumerate(waypoint_list):
            dist = waypoint.transform.location.distance(wp.transform.location)
            if dist < min_distance:
                min_distance = dist
                closest_index = i
        
        return closest_index


# =============================================================================
# Spawn Point Utilities
# =============================================================================

def get_spawn_points(world):
    """Get all spawn points in the map."""
    return world.get_map().get_spawn_points()


def get_random_destination(world, current_location=None, min_distance=50.0):
    """
    Get a random destination that's at least min_distance away.
    
    Args:
        world: CARLA world
        current_location: carla.Location to measure distance from
        min_distance: Minimum distance for destination
        
    Returns:
        carla.Transform of destination spawn point
    """
    import random
    
    spawn_points = get_spawn_points(world)
    
    if current_location is None:
        return random.choice(spawn_points)
    
    # Filter spawn points by distance
    valid_points = [
        sp for sp in spawn_points
        if current_location.distance(sp.location) >= min_distance
    ]
    
    if valid_points:
        return random.choice(valid_points)
    return random.choice(spawn_points)


def print_spawn_points(world):
    """Print all spawn points with indices."""
    spawn_points = get_spawn_points(world)
    print(f"\n{len(spawn_points)} spawn points available:")
    for i, sp in enumerate(spawn_points):
        print(f"  [{i:3d}] x={sp.location.x:7.1f}, y={sp.location.y:7.1f}, z={sp.location.z:5.1f}")


if __name__ == '__main__':
    import sys
    import os
    import argparse
    
    # Add parent directory to path for imports when running as standalone
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    
    parser = argparse.ArgumentParser(description='Global Planner - List spawn points')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    parser.add_argument('--no-draw', action='store_true', help='Do not draw spawn points on map')
    parser.add_argument('--life-time', type=float, default=30.0, help='How long to display points (seconds)')
    args = parser.parse_args()
    
    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    
    world = client.get_world()
    map_name = world.get_map().name
    print(f"Connected to map: {map_name}")
    
    print_spawn_points(world)
    
    # Draw spawn points on the map
    if not args.no_draw:
        spawn_points = get_spawn_points(world)
        debug = world.debug
        print(f"\nDrawing {len(spawn_points)} spawn points on map (visible for {args.life_time}s)...")
        
        for i, sp in enumerate(spawn_points):
            # Draw point
            debug.draw_point(
                sp.location + carla.Location(z=0.5),
                size=0.15,
                color=carla.Color(0, 255, 255),  # Cyan
                life_time=args.life_time
            )
            # Draw index number
            debug.draw_string(
                sp.location + carla.Location(z=1.0),
                str(i),
                draw_shadow=True,
                color=carla.Color(255, 255, 255),  # White
                life_time=args.life_time
            )
        
        print("Done! Check the CARLA simulator window.")
