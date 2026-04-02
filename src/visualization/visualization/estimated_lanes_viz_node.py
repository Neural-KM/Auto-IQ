#!/usr/bin/env python3
"""
Estimated Lanes Visualizer Node
===============================

ROS2 node for visualizing EstimatedLanes messages in RViz.

Subscribes to:
- /mapping/estimated_lanes (autoiq_msgs/EstimatedLanes)

Publishes:
- /mapping/estimated_lanes/visualization (visualization_msgs/MarkerArray)

Visualizes:
- Road boundaries (red)
- Lane centerlines (green)
- Lane boundaries (blue for inner, black for intermediate)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from autoiq_msgs.msg import EstimatedLanes


# Colors (RGBA)
ROAD_BOUNDARY_COLOR = (1.0, 0.0, 0.0, 1.0)       # Red - drivable area edges
INNER_LANE_BOUNDARY_COLOR = (0.0, 0.0, 1.0, 1.0) # Blue - inner lane boundaries
LANE_BOUNDARY_COLOR = (0.0, 0.0, 0.0, 0.8)       # Black - intermediate lane boundaries
LANE_CENTERLINE_COLOR = (0.0, 1.0, 0.0, 1.0)     # Green - lane centerlines


class EstimatedLanesVisualizerNode(Node):
    """
    ROS2 Node for visualizing EstimatedLanes in RViz.
    """

    def __init__(self):
        super().__init__('estimated_lanes_visualizer_node')

        # Parameters
        self.declare_parameter('marker_lifetime', 0.2)  # seconds
        self.declare_parameter('centerline_width', 0.15)  # meters
        self.declare_parameter('boundary_width', 0.1)  # meters
        self.declare_parameter('inner_boundary_width', 0.12)  # meters
        self.declare_parameter('road_boundary_width', 0.15)  # meters

        self.marker_lifetime = self.get_parameter('marker_lifetime').value
        self.centerline_width = self.get_parameter('centerline_width').value
        self.boundary_width = self.get_parameter('boundary_width').value
        self.inner_boundary_width = self.get_parameter('inner_boundary_width').value
        self.road_boundary_width = self.get_parameter('road_boundary_width').value

        # QoS for estimated lanes subscription
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber - follows pattern: subscribe to source topic
        self.estimated_lanes_sub = self.create_subscription(
            EstimatedLanes,
            '/mapping/estimated_lanes',
            self._estimated_lanes_cb,
            qos
        )

        # Publisher - follows pattern: <source_topic>/visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/mapping/estimated_lanes/visualization',
            10
        )

        self.get_logger().info('Estimated Lanes Visualizer Node started')

    def _estimated_lanes_cb(self, msg: EstimatedLanes):
        """Handle incoming EstimatedLanes message and publish markers."""
        marker_array = MarkerArray()
        stamp = msg.header.stamp
        frame_id = msg.header.frame_id

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.ns = 'estimated_lanes'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if not msg.valid:
            self.marker_pub.publish(marker_array)
            return

        marker_id = 1
        lifetime_sec = int(self.marker_lifetime)
        lifetime_nsec = int((self.marker_lifetime - lifetime_sec) * 1e9)

        # 1. Draw road boundaries (RED - drivable area edges)
        if len(msg.left_boundary) >= 2:
            left_marker = self._create_line_strip(
                points=msg.left_boundary,
                frame_id=frame_id,
                stamp=stamp,
                ns='road_boundary',
                marker_id=marker_id,
                color=ROAD_BOUNDARY_COLOR,
                width=self.road_boundary_width,
                lifetime_sec=lifetime_sec,
                lifetime_nsec=lifetime_nsec,
                z_offset=0.05
            )
            marker_array.markers.append(left_marker)
            marker_id += 1

        if len(msg.right_boundary) >= 2:
            right_marker = self._create_line_strip(
                points=msg.right_boundary,
                frame_id=frame_id,
                stamp=stamp,
                ns='road_boundary',
                marker_id=marker_id,
                color=ROAD_BOUNDARY_COLOR,
                width=self.road_boundary_width,
                lifetime_sec=lifetime_sec,
                lifetime_nsec=lifetime_nsec,
                z_offset=0.05
            )
            marker_array.markers.append(right_marker)
            marker_id += 1

        # 2. Draw lane boundaries and centerlines
        num_lanes = len(msg.lanes)
        for i, lane in enumerate(msg.lanes):
            lane_idx = lane.lane_index

            # Determine if this is an inner boundary (BLUE) or intermediate (BLACK)
            # Inner right = rightmost lane's right edge (i=0, right boundary)
            # Inner left = leftmost lane's left edge (i=num_lanes-1, left boundary)

            # Right boundary
            if len(lane.right_boundary) >= 2:
                is_inner_right = (i == 0)
                color = INNER_LANE_BOUNDARY_COLOR if is_inner_right else LANE_BOUNDARY_COLOR
                width = self.inner_boundary_width if is_inner_right else self.boundary_width

                right_marker = self._create_line_strip(
                    points=lane.right_boundary,
                    frame_id=frame_id,
                    stamp=stamp,
                    ns=f'lane_{lane_idx}_boundary',
                    marker_id=marker_id,
                    color=color,
                    width=width,
                    lifetime_sec=lifetime_sec,
                    lifetime_nsec=lifetime_nsec,
                    z_offset=0.08 if is_inner_right else 0.02
                )
                marker_array.markers.append(right_marker)
                marker_id += 1

            # Left boundary
            if len(lane.left_boundary) >= 2:
                is_inner_left = (i == num_lanes - 1)
                color = INNER_LANE_BOUNDARY_COLOR if is_inner_left else LANE_BOUNDARY_COLOR
                width = self.inner_boundary_width if is_inner_left else self.boundary_width

                left_marker = self._create_line_strip(
                    points=lane.left_boundary,
                    frame_id=frame_id,
                    stamp=stamp,
                    ns=f'lane_{lane_idx}_boundary',
                    marker_id=marker_id,
                    color=color,
                    width=width,
                    lifetime_sec=lifetime_sec,
                    lifetime_nsec=lifetime_nsec,
                    z_offset=0.08 if is_inner_left else 0.02
                )
                marker_array.markers.append(left_marker)
                marker_id += 1

            # Lane centerline (GREEN)
            if len(lane.centerline) >= 2:
                centerline_marker = self._create_line_strip(
                    points=lane.centerline,
                    frame_id=frame_id,
                    stamp=stamp,
                    ns=f'lane_{lane_idx}_centerline',
                    marker_id=marker_id,
                    color=LANE_CENTERLINE_COLOR,
                    width=self.centerline_width,
                    lifetime_sec=lifetime_sec,
                    lifetime_nsec=lifetime_nsec,
                    z_offset=0.1
                )
                marker_array.markers.append(centerline_marker)
                marker_id += 1

            # Lane label (text marker)
            if len(lane.centerline) > 0:
                mid_idx = len(lane.centerline) // 2
                mid_pt = lane.centerline[mid_idx]

                text_marker = Marker()
                text_marker.header.frame_id = frame_id
                text_marker.header.stamp = stamp
                text_marker.ns = 'lane_labels'
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = mid_pt.x
                text_marker.pose.position.y = mid_pt.y
                text_marker.pose.position.z = 0.5
                text_marker.pose.orientation.w = 1.0
                text_marker.scale.z = 0.8
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
                text_marker.text = f'L{lane_idx}'
                if lane.is_partial:
                    text_marker.text += ' (P)'
                text_marker.lifetime.sec = lifetime_sec
                text_marker.lifetime.nanosec = lifetime_nsec
                marker_array.markers.append(text_marker)
                marker_id += 1

        # 3. Add status text
        status_marker = Marker()
        status_marker.header.frame_id = frame_id
        status_marker.header.stamp = stamp
        status_marker.ns = 'status'
        status_marker.id = marker_id
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.pose.position.x = 2.0
        status_marker.pose.position.y = 0.0
        status_marker.pose.position.z = 1.5
        status_marker.pose.orientation.w = 1.0
        status_marker.scale.z = 0.5
        status_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
        status_marker.text = f'Lanes: {msg.lane_count} | Width: {msg.road_width_avg:.1f}m'
        status_marker.lifetime.sec = lifetime_sec
        status_marker.lifetime.nanosec = lifetime_nsec
        marker_array.markers.append(status_marker)

        self.marker_pub.publish(marker_array)

    def _create_line_strip(
        self,
        points,
        frame_id: str,
        stamp,
        ns: str,
        marker_id: int,
        color: tuple,
        width: float,
        lifetime_sec: int,
        lifetime_nsec: int,
        z_offset: float = 0.0
    ) -> Marker:
        """Create a LINE_STRIP marker from a list of Points."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width

        marker.color = ColorRGBA(
            r=float(color[0]),
            g=float(color[1]),
            b=float(color[2]),
            a=float(color[3])
        )

        marker.lifetime.sec = lifetime_sec
        marker.lifetime.nanosec = lifetime_nsec

        for pt in points:
            p = Point()
            p.x = pt.x
            p.y = pt.y
            p.z = z_offset
            marker.points.append(p)

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = EstimatedLanesVisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
