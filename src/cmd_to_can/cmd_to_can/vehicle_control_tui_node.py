#!/usr/bin/env python3
"""
vehicle_control_tui_node
========================

Dummy publisher of ``autoiq_msgs/VehicleControl`` driven by a ``curses`` TUI.

Purpose
-------
The real producer of ``/carla/ego_vehicle/control`` is the CARLA bridge /
control package. When you just want to verify the rest of the pipeline --
``cmd_to_can_node`` -> CAN -> Teensy -> actuators -- you don't want to be
forced to spin CARLA up. This node lets you publish ``VehicleControl``
messages by hand from the keyboard so you can confirm that your control
command actually walks all the way out to the bus.

Topic / message convention matches what ``cmd_to_can_node`` subscribes to:

* Control topic : ``/carla/ego_vehicle/control`` (override with the
                  ``--ros-args -p control_topic:=<name>`` parameter).
* Control type  : ``autoiq_msgs/VehicleControl``.
* Velocity topic: ``/target_velocity`` (override with
                  ``--ros-args -p target_velocity_topic:=<name>``).
* Velocity type : ``std_msgs/Float32``, in the same scaled engineering units
                  the firmware's ``VEL_FB_SCALE`` produces (typically m/s).
                  Bypasses ``VehicleControl.throttle/brake`` since the
                  Teensy runs the longitudinal PID itself.
* Steer convention follows the .msg docstring: ``+1.0`` = full left,
  ``-1.0`` = full right. The arrow keys are wired so the visual direction
  matches: LEFT arrow drives ``steer`` more positive, RIGHT arrow drives it
  more negative.

Run it (in its own terminal -- curses needs a real TTY)::

    ros2 run cmd_to_can vehicle_control_tui

In a different terminal, leave ``cmd_to_can_node`` running. It will
re-broadcast the most recent steer onto CAN at its own ``publish_rate_hz``,
so you'll see the wheels respond as soon as you nudge the arrow keys.
"""

from __future__ import annotations

import curses
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from autoiq_msgs.msg import VehicleControl
from std_msgs.msg import Float32


# ---------------------------------------------------------------------------
# Step sizes for the arrow keys. Small enough that a single press feels safe
# on a real vehicle, large enough that holding the key down ramps quickly.
# ---------------------------------------------------------------------------

THROTTLE_STEP = 0.05  # range [0.0, 1.0]
BRAKE_STEP = 0.05     # range [0.0, 1.0]
STEER_STEP = 0.05     # range [-1.0, 1.0]
# Velocity is in scaled engineering units (typically m/s). 0.1 is small
# enough that a single keypress is safe even on a real vehicle, and the cap
# is well below the int16 wire range (±327.67) but high enough for a normal
# road test. Override TARGET_VEL_MAX in code if you need more.
TARGET_VEL_STEP = 0.1
TARGET_VEL_MAX = 30.0


@dataclass
class ControlState:
    """Mutable state mirrored 1:1 onto a ``VehicleControl`` message, plus the
    out-of-band ``target_velocity`` field that goes to its own topic.

    ``target_velocity`` lives here for convenience -- it's not part of
    ``VehicleControl`` -- so the panic / safe-state paths can zero it
    alongside the rest of the state without bookkeeping in two places.
    """

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    hand_brake: bool = False
    reverse: bool = False
    manual_gear_shift: bool = False
    gear: int = 0
    target_velocity: float = 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class VehicleControlTuiNode(Node):
    """Owns the publisher; the curses loop owns the keyboard + state."""

    def __init__(self) -> None:
        super().__init__("vehicle_control_tui_node")

        # Same defaults the cmd_to_can bridge subscribes to. Override at
        # runtime with ``--ros-args -p control_topic:=/your/topic`` (and
        # likewise for ``target_velocity_topic``).
        self.declare_parameter("control_topic", "/carla/ego_vehicle/control")
        self.declare_parameter("target_velocity_topic", "/target_velocity")
        # Re-publish the latest state on a timer too, so the cmd_to_can
        # bridge always has a fresh sample even when the user isn't pressing
        # keys. 20 Hz matches cmd_to_can_node's default tick rate.
        self.declare_parameter("publish_rate_hz", 20.0)

        self._control_topic = str(self.get_parameter("control_topic").value)
        self._target_velocity_topic = str(
            self.get_parameter("target_velocity_topic").value
        )
        self._publish_period = 1.0 / max(1.0, float(
            self.get_parameter("publish_rate_hz").value
        ))

        self._publisher = self.create_publisher(
            VehicleControl,
            self._control_topic,
            10,
        )
        # Velocity setpoint goes on its own topic; cmd_to_can_node holds the
        # last value until the next message arrives, so we publish at the
        # same periodic rate as VehicleControl to keep both streams in lock-
        # step on the consumer side.
        self._velocity_publisher = self.create_publisher(
            Float32,
            self._target_velocity_topic,
            10,
        )

        self.get_logger().info(
            f"vehicle_control_tui_node publishing on {self._control_topic} "
            f"and {self._target_velocity_topic} at {1.0 / self._publish_period:.1f} Hz"
        )

    def publish_state(self, state: ControlState) -> None:
        msg = VehicleControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "vehicle_control_tui"
        msg.throttle = float(state.throttle)
        msg.brake = float(state.brake)
        msg.steer = float(state.steer)
        msg.hand_brake = bool(state.hand_brake)
        msg.reverse = bool(state.reverse)
        msg.manual_gear_shift = bool(state.manual_gear_shift)
        msg.gear = int(state.gear)
        self._publisher.publish(msg)

        vmsg = Float32()
        vmsg.data = float(state.target_velocity)
        self._velocity_publisher.publish(vmsg)


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------


HELP_LINES = (
    "Vehicle Control TUI -- publishes autoiq_msgs/VehicleControl + std_msgs/Float32",
    "",
    "  UP / DOWN     : throttle +/-          (range 0.00 .. 1.00)",
    "  B  / N        : brake +/-             (range 0.00 .. 1.00)",
    "  LEFT / RIGHT  : steer left / right    (+left, -right; range -1.00 .. +1.00)",
    "  ,  / .        : target velocity -/+   (step 0.10, range -30.00 .. +30.00)",
    "  0             : zero target velocity",
    "  C             : center steer",
    "  H             : toggle hand brake",
    "  R             : toggle reverse gear",
    "  SPACE         : PANIC -- zero throttle/steer/velocity, hit brake, drop reverse",
    "  P             : publish once now (in addition to the periodic timer)",
    "  Q             : quit",
)


def draw_ui(
    stdscr: "curses.window",
    state: ControlState,
    control_topic: str,
    velocity_topic: str,
    publish_hz: float,
    last_publish_status: str,
) -> None:
    stdscr.erase()
    for row, line in enumerate(HELP_LINES):
        stdscr.addstr(row, 0, line)

    base = len(HELP_LINES) + 1
    stdscr.addstr(base + 0, 0, f"Control topic   : {control_topic}")
    stdscr.addstr(base + 1, 0, f"Velocity topic  : {velocity_topic}")
    stdscr.addstr(base + 2, 0, f"Publish rate    : {publish_hz:.1f} Hz")
    stdscr.addstr(base + 3, 0, "")

    if state.steer > 0.001:
        steer_label = f"{state.steer:+.2f}  (LEFT)"
    elif state.steer < -0.001:
        steer_label = f"{state.steer:+.2f}  (RIGHT)"
    else:
        steer_label = f"{state.steer:+.2f}  (CENTER)"

    stdscr.addstr(base + 4, 0, f"throttle        : {state.throttle:.2f}")
    stdscr.addstr(base + 5, 0, f"brake           : {state.brake:.2f}")
    stdscr.addstr(base + 6, 0, f"steer           : {steer_label}")
    stdscr.addstr(base + 7, 0, f"target velocity : {state.target_velocity:+.2f}  (scaled units, typ. m/s)")
    stdscr.addstr(base + 8, 0, f"hand_brake      : {'ON' if state.hand_brake else 'off'}")
    stdscr.addstr(base + 9, 0, f"reverse         : {'ON' if state.reverse else 'off'}")
    stdscr.addstr(base + 11, 0, f"Last publish    : {last_publish_status}")

    stdscr.refresh()


def run_tui(stdscr: "curses.window", node: VehicleControlTuiNode) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    state = ControlState()
    last_publish_time = 0.0
    last_publish_status = "(none yet)"

    publish_hz = 1.0 / node._publish_period  # noqa: SLF001 -- internal helper, fine

    try:
        while rclpy.ok():
            now = time.monotonic()

            key = stdscr.getch()
            should_publish_now = False

            if key != -1:
                if key == curses.KEY_UP:
                    state.throttle = round(clamp(state.throttle + THROTTLE_STEP, 0.0, 1.0), 3)
                elif key == curses.KEY_DOWN:
                    state.throttle = round(clamp(state.throttle - THROTTLE_STEP, 0.0, 1.0), 3)
                elif key in (ord("b"), ord("B")):
                    state.brake = round(clamp(state.brake + BRAKE_STEP, 0.0, 1.0), 3)
                elif key in (ord("n"), ord("N")):
                    state.brake = round(clamp(state.brake - BRAKE_STEP, 0.0, 1.0), 3)
                elif key == curses.KEY_LEFT:
                    # +1.0 is left per the VehicleControl.msg convention.
                    state.steer = round(clamp(state.steer + STEER_STEP, -1.0, 1.0), 3)
                elif key == curses.KEY_RIGHT:
                    state.steer = round(clamp(state.steer - STEER_STEP, -1.0, 1.0), 3)
                elif key == ord(","):
                    # Comma/period are positionally adjacent on every layout
                    # we care about and don't collide with the existing
                    # bindings. Comma = down, period = up; round to a clean
                    # 0.1 grid so float drift doesn't accumulate over many
                    # presses.
                    state.target_velocity = round(
                        clamp(state.target_velocity - TARGET_VEL_STEP,
                              -TARGET_VEL_MAX, TARGET_VEL_MAX),
                        3,
                    )
                elif key == ord("."):
                    state.target_velocity = round(
                        clamp(state.target_velocity + TARGET_VEL_STEP,
                              -TARGET_VEL_MAX, TARGET_VEL_MAX),
                        3,
                    )
                elif key == ord("0"):
                    state.target_velocity = 0.0
                elif key in (ord("c"), ord("C")):
                    state.steer = 0.0
                elif key in (ord("h"), ord("H")):
                    state.hand_brake = not state.hand_brake
                elif key in (ord("r"), ord("R")):
                    state.reverse = not state.reverse
                elif key == ord(" "):
                    # Panic: drop the throttle, slam brake, center steer,
                    # zero the velocity setpoint so the firmware velocity
                    # PID stops asking for any motion either, leave the
                    # gear selector alone so the driver isn't surprised by
                    # a phantom shift.
                    state.throttle = 0.0
                    state.brake = 1.0
                    state.steer = 0.0
                    state.target_velocity = 0.0
                    state.reverse = False
                elif key in (ord("p"), ord("P")):
                    should_publish_now = True
                elif key in (ord("q"), ord("Q")):
                    break

            # Periodic publish so the bridge always has a fresh sample, plus
            # an immediate publish on the explicit ``P`` keypress so the
            # operator can confirm round-trip latency.
            if should_publish_now or (now - last_publish_time) >= node._publish_period:  # noqa: SLF001
                node.publish_state(state)
                last_publish_time = now
                last_publish_status = (
                    f"OK @ {time.strftime('%H:%M:%S')} "
                    f"(t={state.throttle:.2f} b={state.brake:.2f} "
                    f"s={state.steer:+.2f} v={state.target_velocity:+.2f})"
                )

            draw_ui(  # noqa: SLF001 -- internal helpers, fine
                stdscr,
                state,
                node._control_topic,
                node._target_velocity_topic,
                publish_hz,
                last_publish_status,
            )
            time.sleep(0.02)

    finally:
        # On exit, push out one explicit "everything off" frame so the bridge
        # immediately stops asking the actuators to do anything. This mirrors
        # the safe-state shutdown in cmd_to_can_node itself. ControlState's
        # field defaults already give us throttle=brake=steer=target_velocity=0
        # so just emit a fresh default.
        try:
            node.publish_state(ControlState())
        except Exception:  # noqa: BLE001 -- best-effort on shutdown
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VehicleControlTuiNode()
    try:
        curses.wrapper(run_tui, node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
