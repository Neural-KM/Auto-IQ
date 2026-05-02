#!/usr/bin/env python3
"""
cmd_to_can_node
===============

Bridge ``autoiq_msgs/VehicleControl`` (published by the ``control`` package)
to the Teensy 4.1 over the CAN bus.

This node mirrors the wire protocol implemented in
``firmware/main_project/src/main.cpp`` and demonstrated by
``firmware/main_project/src/compute_side_tui.py``:

* ``0x300`` THROTTLE_CMD       -- 2 bytes, big-endian uint16, value = throttle[%] * 10
* ``0x301`` MODE_CMD           -- 3 bytes:
                                  byte 0 = autonomous (1=AUTO, 0=MANUAL)
                                  byte 1 = steer mode (0=DIRECT, 1=ANGLE_PID)
                                  byte 2 = throttle mode (0=DIRECT, 1=VELOCITY_PID)
* ``0x302`` STEER_TORQUE_CMD   -- 2 bytes, big-endian int16 (DIRECT steer)
* ``0x303`` STEER_ANGLE_CMD    -- 2 bytes, big-endian int16, value = angle[deg] * 10
* ``0x304`` VELOCITY_CMD       -- 2 bytes, big-endian int16, value = velocity * 100
                                  (in the same engineering units the firmware's
                                  VEL_FB_SCALE produces -- typically m/s)

For now this node intentionally **bypasses** the ``throttle`` and ``brake``
fields produced by the control package. The Teensy runs an on-board velocity
PID that closes the loop against the OBD feedback frame (extended ID
``0x0C08A7F0``, bytes [2..3] little-endian uint16, offset 0x7D00, scaled by
the firmware's ``VEL_FB_SCALE``). The host publishes the velocity setpoint
on a separate ROS topic -- ``/target_velocity`` (``std_msgs/Float32``) by
default -- in the same scaled engineering units the firmware's
``VEL_FB_SCALE`` produces (typically m/s). Until a message arrives the
setpoint is hardcoded to ``0.0`` ("stay at idle"), so a fresh launch never
silently asks the car to move.

The steering pass-through maps ``VehicleControl.steer`` (left positive, right
negative, range [-1, 1]) to a firmware steer-angle setpoint in degrees. The
firmware feedback convention has positive angle = right, so the default
``steer_sign = -1.0`` flips it. Override per vehicle if the physical wiring
differs.

Frames go out on a periodic timer (default 50 ms / 20 Hz) so the firmware's
500 ms command-timeout watchdog is never tripped just because no ROS message
arrived in the meantime; the most recent steer setpoint is re-sent each tick.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import can
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from autoiq_msgs.msg import VehicleControl
from std_msgs.msg import Float32


# ---------------------------------------------------------------------------
# Wire protocol constants. Keep these in sync with main.cpp -- if any of these
# change in the firmware, this file must be updated to match.
# ---------------------------------------------------------------------------

THROTTLE_CMD_CAN_ID = 0x300
MODE_CMD_CAN_ID = 0x301
STEER_TORQUE_CMD_CAN_ID = 0x302
STEER_ANGLE_CMD_CAN_ID = 0x303
VELOCITY_CMD_CAN_ID = 0x304

# Every standard-frame ID this node ever transmits. Used for the blacklist
# safety check at startup (see ``compute_side_tui.py``).
COMMAND_IDS: tuple[tuple[bool, int], ...] = (
    (False, MODE_CMD_CAN_ID),
    (False, STEER_ANGLE_CMD_CAN_ID),
    (False, VELOCITY_CMD_CAN_ID),
)

STEER_ANGLE_LIMIT_DEG = 15.0

# Wire format for the 0x303 steering-angle setpoint frame: int16 BE carrying
# the desired angle in **degrees × STEER_ANGLE_CMD_TX_X10**. 10 → 0.1°
# resolution and a wire range of ±3276.7°, way more than the physical
# ±STEER_ANGLE_LIMIT_DEG, so the clamp in ``angle_deg_to_raw_x10`` catches
# malformed/runaway commands at the edge before they ever hit the bus. Both
# the encode side here and the firmware decode side
# (``STEER_ANGLE_CMD_TX_X10`` in ``main.cpp``) MUST use the same factor --
# the unit-pipeline contract is exactly the same as VEL_CMD_TX_X100 below.
STEER_ANGLE_CMD_TX_X10 = 10.0

# Wire format for the 0x304 velocity setpoint frame: int16 BE carrying the
# scaled velocity setpoint multiplied by this factor. Keep in sync with
# ``VEL_CMD_TX_X100`` in firmware/main_project/src/main.cpp.
VEL_CMD_TX_X100 = 100.0


class SteerMode(IntEnum):
    DIRECT = 0
    ANGLE_PID = 1


class ThrottleMode(IntEnum):
    DIRECT = 0
    VELOCITY_PID = 1


# ---------------------------------------------------------------------------
# Helpers shared with compute_side_tui.py. Inlined here (rather than imported)
# so the ROS package has no source dependency on the firmware tree.
# ---------------------------------------------------------------------------

def angle_deg_to_raw_x10(angle_deg: float) -> int:
    """Encode a desired steering angle in degrees as the int16 wire value.

    Single source of truth for the host-side steering encode. The firmware
    inverts this exactly: ``targetDeg = strAngleCmdX10 / STEER_ANGLE_CMD_TX_X10``.
    Clamping happens here (in degrees) and again in firmware (in the
    derived 0.1° integer limits) so a bad input gets bounded once at the
    boundary it crosses, in the units that boundary is naturally expressed
    in. Both clamps fall out of the same ``STEER_ANGLE_LIMIT_DEG`` so they
    never disagree.
    """
    bounded = max(-STEER_ANGLE_LIMIT_DEG, min(STEER_ANGLE_LIMIT_DEG, angle_deg))
    return int(round(bounded * STEER_ANGLE_CMD_TX_X10))


def clamp_int(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def load_used_can_id_blacklist(
    path: Optional[Path],
) -> tuple[frozenset[tuple[bool, int]], str]:
    """Parse the blacklist file produced by ``scan_used_can_ids.py``.

    Returns ``(blacklist, status_text)``. A None or missing path is treated
    as an empty blacklist with a clear status message, so the node still
    works the very first time before any scan has been run.
    """
    if path is None:
        return frozenset(), "DISABLED (blacklist_file empty)"
    if not path.exists():
        return frozenset(), f"NONE ({path} missing)"

    entries: set[tuple[bool, int]] = set()
    parse_errors = 0
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            parse_errors += 1
            continue
        try:
            arb = int(parts[0], 16)
        except ValueError:
            parse_errors += 1
            continue
        ext = parts[1].lower().startswith("ext")
        entries.add((ext, arb))

    suffix = f", {parse_errors} bad lines" if parse_errors else ""
    return frozenset(entries), f"OK ({len(entries)} IDs from {path.name}{suffix})"


def default_blacklist_path() -> Optional[Path]:
    """Best-effort lookup of the blacklist shipped with this package.

    Returns the installed share path if available, otherwise None so the
    caller can decide whether to warn or just silently disable the
    blacklist.
    """
    try:
        share = Path(get_package_share_directory("cmd_to_can"))
    except Exception:  # noqa: BLE001 -- ament not initialized yet, etc.
        return None
    candidate = share / "config" / "used_can_ids.txt"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass
class _ControlSnapshot:
    """Most recent ``VehicleControl`` + target-velocity values we've received,
    with thread-safe access.

    The ROS subscriber callbacks run on the executor thread; the periodic
    send timer also runs on the executor thread but in a separate callback
    invocation, so technically a lock isn't strictly required for CPython
    here. We use one anyway because (a) it's cheap, (b) it makes the
    invariant explicit, and (c) it keeps us correct if the executor type
    ever changes.

    ``target_velocity`` defaults to 0.0 ("stay at idle") so the very first
    tick before any /target_velocity message arrives never asks the car to
    move. ``have_velocity_msg`` is exposed so the startup banner / hot path
    can tell the difference between "explicit zero from the publisher" and
    "no publisher yet".
    """

    steer: float = 0.0
    target_velocity: float = 0.0
    have_msg: bool = False
    have_velocity_msg: bool = False
    _lock: threading.Lock = threading.Lock()

    def update(self, msg: VehicleControl) -> None:
        with self._lock:
            self.steer = float(msg.steer)
            self.have_msg = True

    def update_velocity(self, value: float) -> None:
        with self._lock:
            self.target_velocity = float(value)
            self.have_velocity_msg = True

    def snapshot(self) -> tuple[float, float, bool, bool]:
        with self._lock:
            return (
                self.steer,
                self.target_velocity,
                self.have_msg,
                self.have_velocity_msg,
            )


# Threshold below which a change in throttle/brake/steer is treated as
# "the same value" for change-detection logging. Picked to be smaller than
# the 0.05 step the TUI uses, so a single keypress always prints.
_FIELD_CHANGE_EPS = 0.005


class CmdToCanNode(Node):
    def __init__(self) -> None:
        super().__init__("cmd_to_can_node")

        # ==================== Parameters ====================
        # ROS-side
        self.declare_parameter("control_topic", "/carla/ego_vehicle/control")
        # Topic carrying the velocity setpoint as std_msgs/Float32, in the
        # same scaled engineering units the firmware's VEL_FB_SCALE produces
        # (typically m/s). Until a message arrives, the setpoint is held at
        # 0.0 -- safe default, "stay at idle".
        self.declare_parameter("target_velocity_topic", "/target_velocity")
        self.declare_parameter("publish_rate_hz", 20.0)

        # CAN bus
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("can_bitrate", 250000)

        # Blacklist. The default is "" -- the resolution-with-fallback to the
        # package-share copy happens below, so launch files that pass an empty
        # default for ``blacklist_file`` still get the shipped blacklist
        # instead of silently disabling it.
        self.declare_parameter("blacklist_file", "")
        self.declare_parameter("disable_blacklist", False)

        # Steering mapping
        # ``VehicleControl.steer`` is +1 for left, -1 for right (per the msg
        # docstring). Firmware feedback convention puts positive angle on the
        # right side of center, so by default we flip the sign. Set
        # steer_sign=+1.0 if your wiring already matches.
        self.declare_parameter("steer_sign", -1.0)
        self.declare_parameter("max_steer_angle_deg", STEER_ANGLE_LIMIT_DEG)

        # Engagement
        # ``autonomous_mode`` controls the Teensy's relay (HIGH = DAC drives
        # throttle). Default False so a fresh launch never silently engages
        # the actuator. Flip to True at runtime via ``ros2 param set`` once
        # you've verified the bus and the velocity setpoint.
        self.declare_parameter("autonomous_mode", False)
        self.declare_parameter("steer_mode", int(SteerMode.ANGLE_PID))
        self.declare_parameter("throttle_mode", int(ThrottleMode.VELOCITY_PID))

        # Resolve parameters into local handles for the hot path.
        self._control_topic = self.get_parameter("control_topic").value
        self._target_velocity_topic = str(
            self.get_parameter("target_velocity_topic").value
        )
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._can_interface = self.get_parameter("can_interface").value
        self._can_channel = self.get_parameter("can_channel").value
        self._can_bitrate = int(self.get_parameter("can_bitrate").value)

        # Cache parameter names that get re-read every send tick. We re-read
        # them each tick (rather than caching the value) so ``ros2 param
        # set`` takes effect without restarting the node.
        self._param_names_hot = (
            "steer_sign",
            "max_steer_angle_deg",
            "autonomous_mode",
            "steer_mode",
            "throttle_mode",
        )

        # ==================== Blacklist ====================
        if bool(self.get_parameter("disable_blacklist").value):
            self._blacklist: frozenset[tuple[bool, int]] = frozenset()
            blacklist_status = "DISABLED (disable_blacklist=true)"
        else:
            bl_param = str(self.get_parameter("blacklist_file").value).strip()
            if bl_param:
                bl_path: Optional[Path] = Path(bl_param)
            else:
                # Empty parameter -> fall back to the file shipped in the
                # package's share/ tree, if any. Keeps launch files that
                # default the arg to "" working out of the box.
                bl_path = default_blacklist_path()
            self._blacklist, blacklist_status = load_used_can_id_blacklist(bl_path)

        conflicts = [
            arb for ext, arb in COMMAND_IDS if (ext, arb) in self._blacklist
        ]
        if conflicts:
            self.get_logger().error(
                "BLACKLIST CONFLICT: command IDs "
                f"{', '.join(f'0x{c:X}' for c in conflicts)} are listed in the "
                "bus blacklist; those frames will be DROPPED at send time. "
                "Re-run scan_used_can_ids.py with this node off, or change "
                "the firmware command IDs."
            )

        # ==================== ROS I/O ====================
        self._snapshot = _ControlSnapshot()
        # Last logged values, used to suppress duplicate "same as last frame"
        # log lines (the TUI republishes at 20 Hz, we don't want a wall of
        # identical text). Initialized to NaN so the very first message is
        # always treated as a change.
        self._last_logged_throttle = float("nan")
        self._last_logged_brake = float("nan")
        self._last_logged_steer = float("nan")
        self._last_logged_hand_brake: Optional[bool] = None
        self._last_logged_reverse: Optional[bool] = None
        self._last_logged_gear: Optional[int] = None

        self._sub = self.create_subscription(
            VehicleControl,
            self._control_topic,
            self._on_control,
            10,
        )
        # Velocity setpoint subscription. Same depth as the control topic;
        # the firmware re-uses the most recent value at every send tick so
        # we don't need a long backlog. NaN/Inf is rejected at callback time
        # to keep a malformed publisher from poisoning the hot path.
        self._velocity_sub = self.create_subscription(
            Float32,
            self._target_velocity_topic,
            self._on_target_velocity,
            10,
        )
        self._last_logged_target_velocity = float("nan")

        # ==================== CAN bus ====================
        self._bus: Optional[can.BusABC] = None
        try:
            self._bus = can.Bus(
                interface=self._can_interface,
                channel=self._can_channel,
                bitrate=self._can_bitrate,
            )
            self.get_logger().info(
                f"CAN bus connected ({self._can_interface}:{self._can_channel}, "
                f"{self._can_bitrate} bps)"
            )
        except Exception as exc:  # noqa: BLE001 -- python-can raises a zoo of types
            self.get_logger().error(
                f"Failed to open CAN bus ({self._can_interface}:"
                f"{self._can_channel}): {exc}. The node will keep running so "
                "you can fix the bus and restart, but no frames will be sent."
            )

        # ==================== Send timer ====================
        publish_rate_hz = max(1.0, publish_rate_hz)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._tick)

        self.get_logger().info(
            "cmd_to_can_node started.\n"
            f"  Subscribing to    : {self._control_topic}\n"
            f"  Velocity topic    : {self._target_velocity_topic}"
            " (std_msgs/Float32, defaults to 0.0 until first msg)\n"
            f"  Publish rate      : {publish_rate_hz:.1f} Hz\n"
            f"  CAN               : {self._can_interface}:{self._can_channel}"
            f" @ {self._can_bitrate} bps\n"
            f"  Blacklist         : {blacklist_status}\n"
            f"  Autonomous mode   : {self._param('autonomous_mode')}\n"
            f"  Steer mode        : {SteerMode(self._param('steer_mode')).name}\n"
            f"  Throttle mode     : "
            f"{ThrottleMode(self._param('throttle_mode')).name}"
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _on_control(self, msg: VehicleControl) -> None:
        self._snapshot.update(msg)

        # Decide whether anything actually changed since the last log line.
        # We treat float fields with an epsilon (smaller than the TUI's step
        # size) so a single keypress is always loud; bool/int fields are
        # exact-compare.
        changed = (
            abs(float(msg.throttle) - self._last_logged_throttle) > _FIELD_CHANGE_EPS
            or abs(float(msg.brake) - self._last_logged_brake) > _FIELD_CHANGE_EPS
            or abs(float(msg.steer) - self._last_logged_steer) > _FIELD_CHANGE_EPS
            or bool(msg.hand_brake) != self._last_logged_hand_brake
            or bool(msg.reverse) != self._last_logged_reverse
            or int(msg.gear) != self._last_logged_gear
        )

        line = (
            f"RX VehicleControl  throttle={float(msg.throttle):+.3f}  "
            f"brake={float(msg.brake):+.3f}  steer={float(msg.steer):+.3f}  "
            f"hand_brake={bool(msg.hand_brake)}  reverse={bool(msg.reverse)}  "
            f"gear={int(msg.gear)}"
        )

        if changed:
            # Print every change immediately so the operator sees keypresses
            # land with no perceived latency.
            self.get_logger().info(line)
            self._last_logged_throttle = float(msg.throttle)
            self._last_logged_brake = float(msg.brake)
            self._last_logged_steer = float(msg.steer)
            self._last_logged_hand_brake = bool(msg.hand_brake)
            self._last_logged_reverse = bool(msg.reverse)
            self._last_logged_gear = int(msg.gear)
        else:
            # Steady-state heartbeat at 1 Hz so it's still obvious the topic
            # is alive even when the operator isn't pressing anything.
            self.get_logger().info(line, throttle_duration_sec=1.0)

    def _on_target_velocity(self, msg: Float32) -> None:
        value = float(msg.data)
        # NaN / Inf would otherwise propagate into the int16 wire encoding
        # below and silently wrap; reject upstream so the operator sees the
        # bad publisher in the log instead of weird actuator behavior.
        if value != value or value in (float("inf"), float("-inf")):
            self.get_logger().warn(
                f"Ignoring non-finite target_velocity={msg.data!r}",
                throttle_duration_sec=1.0,
            )
            return
        self._snapshot.update_velocity(value)

        # Same change-vs-heartbeat pattern as _on_control above so a
        # publisher hammering the same value at 50 Hz doesn't drown the log.
        if abs(value - self._last_logged_target_velocity) > _FIELD_CHANGE_EPS:
            self.get_logger().info(f"RX target_velocity={value:+.3f}")
            self._last_logged_target_velocity = value
        else:
            self.get_logger().info(
                f"RX target_velocity={value:+.3f}",
                throttle_duration_sec=1.0,
            )

    # ------------------------------------------------------------------
    # Periodic CAN tick. Always sends MODE + STEER_ANGLE + VELOCITY so the
    # firmware watchdog (500 ms) never trips and so the most recent steer
    # setpoint keeps the wheels tracking even if upstream messages stall.
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self._bus is None:
            return

        steer, target_velocity, _, _ = self._snapshot.snapshot()

        steer_sign = float(self._param("steer_sign"))
        max_angle_deg = float(self._param("max_steer_angle_deg"))
        auto = bool(self._param("autonomous_mode"))
        steer_mode = SteerMode(int(self._param("steer_mode")))
        throttle_mode = ThrottleMode(int(self._param("throttle_mode")))

        # Map ROS steer in [-1, 1] to firmware angle in degrees.
        steer_clamped = max(-1.0, min(1.0, steer))
        target_angle_deg = steer_sign * steer_clamped * max_angle_deg
        # max_steer_angle_deg can be set lower than the firmware's hardware
        # limit; clamp again here so we don't push the Teensy past its
        # protected range.
        target_angle_deg = max(-STEER_ANGLE_LIMIT_DEG,
                               min(STEER_ANGLE_LIMIT_DEG, target_angle_deg))

        msgs = [
            self._build_mode_frame(auto, steer_mode, throttle_mode),
            self._build_angle_frame(target_angle_deg),
            self._build_velocity_frame(target_velocity),
        ]

        for m in msgs:
            self._safe_send(m)

    # ------------------------------------------------------------------
    # Frame builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mode_frame(
        auto: bool,
        steer_mode: SteerMode,
        throttle_mode: ThrottleMode,
    ) -> can.Message:
        return can.Message(
            arbitration_id=MODE_CMD_CAN_ID,
            data=[1 if auto else 0, int(steer_mode), int(throttle_mode)],
            is_extended_id=False,
        )

    @staticmethod
    def _build_angle_frame(angle_deg: float) -> can.Message:
        angle_x10 = angle_deg_to_raw_x10(angle_deg)
        # Firmware also clamps to the same range (derived from
        # STEER_ANGLE_LIMIT_DEG × STEER_ANGLE_CMD_TX_X10), but enforce it
        # here too -- defense in depth, and the source of truth for the
        # numeric limit lives in this same module.
        angle_x10_limit = int(round(STEER_ANGLE_LIMIT_DEG * STEER_ANGLE_CMD_TX_X10))
        angle_x10 = clamp_int(angle_x10, -angle_x10_limit, angle_x10_limit)
        payload = angle_x10.to_bytes(2, byteorder="big", signed=True)
        return can.Message(
            arbitration_id=STEER_ANGLE_CMD_CAN_ID,
            data=[payload[0], payload[1]],
            is_extended_id=False,
        )

    @staticmethod
    def _build_velocity_frame(target_velocity: float) -> can.Message:
        # Encode in the firmware's wire format: int16 BE = scaled velocity ×
        # VEL_CMD_TX_X100. Clamp to int16 so an out-of-range request from a
        # bad parameter set doesn't wrap silently.
        velocity_x100 = int(round(target_velocity * VEL_CMD_TX_X100))
        velocity_x100 = clamp_int(velocity_x100, -32768, 32767)
        payload = velocity_x100.to_bytes(2, byteorder="big", signed=True)
        return can.Message(
            arbitration_id=VELOCITY_CMD_CAN_ID,
            data=[payload[0], payload[1]],
            is_extended_id=False,
        )

    # ------------------------------------------------------------------
    # Send + housekeeping
    # ------------------------------------------------------------------

    def _safe_send(self, msg: can.Message) -> None:
        key = (bool(msg.is_extended_id), int(msg.arbitration_id))
        if key in self._blacklist:
            # Don't spam the log on every tick; the startup banner already
            # warned about conflicts.
            return
        try:
            self._bus.send(msg)
        except can.CanError as exc:
            # Throttled warn so a bus-off condition doesn't flood the log.
            self.get_logger().warn(
                f"CAN send failed for ID 0x{msg.arbitration_id:X}: {exc}",
                throttle_duration_sec=1.0,
            )

    def _param(self, name: str):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------
    # Shutdown -- send a safe-state frame so the Teensy parks the actuators
    # even if the firmware's command-timeout watchdog hasn't fired yet.
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._bus is None:
            return
        try:
            safe_msgs = [
                self._build_mode_frame(False, SteerMode.DIRECT, ThrottleMode.DIRECT),
                self._build_angle_frame(0.0),
                self._build_velocity_frame(0.0),
            ]
            for m in safe_msgs:
                try:
                    self._bus.send(m)
                except can.CanError:
                    pass
        finally:
            try:
                self._bus.shutdown()
            except Exception:  # noqa: BLE001 -- best-effort cleanup
                pass
            self._bus = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdToCanNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Clean Ctrl-C / SIGTERM. Fall through to the shutdown path so the
        # safe-state CAN frames still go out and the bus is closed.
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
