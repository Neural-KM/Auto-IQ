#!/usr/bin/env python3
"""
Local Lane Data Visualizer Node
=============================

ROS2 node for visualizing LocalLaneData messages in RViz.

Subscribes to:
- /mapping/local_lane_data (autoiq_msgs/LocalLaneData)

Publishes:
- /visualization/local_lane_data (visualization_msgs/MarkerArray)

Visualizes:
- Road boundaries (red)
- Lane centerlines (green)
- Lane boundaries (black)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

from autoiq_msgs.msg import LocalLaneData


# Colors (RGBA)
ROAD_BOUNDARY_COLOR = (1.0, 0.0, 0.0, 1.0)    # Red
LANE_CENTERLINE_COLOR = (0.0, 1.0, 0.0, 1.0)  # Green
LANE_BOUNDARY_COLOR = (0.0, 0.0, 0.0, 0.8)    # Black


class LocalLaneDataVisualizerNode(Node):
    """
    ROS2 Node for visualizing LocalLaneData in RViz.
    """

    def __init__(self):
        super().__init__('local_lane_data_visualizer_node')

        # Parameters
        self.declare_parameter('marker_lifetime', 0.2)  # seconds
        self.declare_parameter('centerline_width', 0.15)  # meters
        self.declare_parameter('boundary_width', 0.1)  # meters
        self.declare_parameter('road_boundary_width', 0.2)  # meters

        self.marker_lifetime = self.get_parameter('marker_lifetime').value
        self.centerline_width = self.get_parameter('centerline_width').value
        self.boundary_width = self.get_parameter('boundary_width').value
        self.road_boundary_width = self.get_parameter('road_boundary_width').value

        # QoS for local lane data subscription
        drivable_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.local_lane_data_sub = self.create_subscription(
            LocalLaneData,
            '/mapping/local_lane_data',
            self._local_lane_data_cb,
            drivable_qos
        )

        # Publisher
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visualization/local_lane_data',
            10
        )

        self.get_logger().info('Local Lane Data Visualizer Node started')

    def _local_lane_data_cb(self, msg: LocalLaneData):
        """Handle incoming LocalLaneData message and publish markers."""
        marker_array = MarkerArray()
        stamp = msg.header.stamp
        frame_id = msg.header.frame_id  # Should be base_link or ego_vehicle

        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.ns = 'local_lane_data'
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if not msg.valid:
            # Publish just the clear marker
            self.marker_pub.publish(marker_array)
            return

        marker_id = 1
        lifetime_ns = int(self.marker_lifetime * 1e9)

        # 1. Draw road boundaries (red)
        if len(msg.left_boundary) >= 2:
            left_marker = self._create_line_strip(
                points=msg.left_boundary,
                frame_id=frame_id,
                stamp=stamp,
                ns='road_boundary',
                marker_id=marker_id,
                color=ROAD_BOUNDARY_COLOR,
                width=self.road_boundary_width,
                lifetime_ns=lifetime_ns,
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
                lifetime_ns=lifetime_ns,
                z_offset=0.05
            )
            marker_array.markers.append(right_marker)
            marker_id += 1

        # 2. Draw lane centerlines (green) and boundaries (black)
        for lane in msg.lanes:
            lane_idx = lane.lane_index

            # Lane centerline (green)
            if len(lane.centerline) >= 2:
                centerline_marker = self._create_line_strip(
                    points=lane.centerline,
                    frame_id=frame_id,
                    stamp=stamp,
                    ns=f'lane_{lane_idx}_centerline',
                    marker_id=marker_id,
                    color=LANE_CENTERLINE_COLOR,
                    width=self.centerline_width,
                    lifetime_ns=lifetime_ns,
                    z_offset=0.1
                )
                marker_array.markers.append(centerline_marker)
                marker_id += 1

            # Lane boundaries (black)
            if len(lane.left_boundary) >= 2:
                left_lane_marker = self._create_line_strip(
                    points=lane.left_boundary,
                    frame_id=frame_id,
                    stamp=stamp,
                    ns=f'lane_{lane_idx}_boundary',
                    marker_id=marker_id,
                    color=LANE_BOUNDARY_COLOR,
                    width=self.boundary_width,
                    lifetime_ns=lifetime_ns,
                    z_offset=0.02
                )
                marker_array.markers.append(left_lane_marker)
                marker_id += 1

            if len(lane.right_boundary) >= 2:
                right_lane_marker = self._create_line_strip(
                    points=lane.right_boundary,
                    frame_id=frame_id,
                    stamp=stamp,
                    ns=f'lane_{lane_idx}_boundary',
                    marker_id=marker_id,
                    color=LANE_BOUNDARY_COLOR,
                    width=self.boundary_width,
                    lifetime_ns=lifetime_ns,
                    z_offset=0.02
                )
                marker_array.markers.append(right_lane_marker)
                marker_id += 1

            # Lane label (text marker)
            if len(lane.centerline) > 0:
                # Place label at middle of centerline
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
                text_marker.lifetime.nanosec = lifetime_ns
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
        status_marker.lifetime.nanosec = lifetime_ns
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
        lifetime_ns: int,
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
        marker.scale.x = width  # Line width

        marker.color = ColorRGBA(
            r=float(color[0]),
            g=float(color[1]),
            b=float(color[2]),
            a=float(color[3])
        )

        marker.lifetime.nanosec = lifetime_ns

        for pt in points:
            p = Point()
            p.x = pt.x
            p.y = pt.y
            p.z = z_offset
            marker.points.append(p)

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = LocalLaneDataVisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
