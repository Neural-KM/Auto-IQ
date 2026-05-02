#!/usr/bin/env python3
"""
pid_tuner_gui
=============

Standalone pygame GUI for live-tuning the Teensy 4.1's steering-angle and
velocity PID gains over CAN. **Not** a ROS 2 node -- it talks to the firmware
directly with python-can so it can run on a laptop with a USB-CAN dongle and
no ROS environment, or on the same Jetson next to ``cmd_to_can_node`` since
the gain channel uses its own CAN IDs (0x333 / 0x334).

Wire protocol (mirrors firmware/main_project/src/main.cpp)
----------------------------------------------------------

Two paired standard-frame IDs, each carrying a 5-byte payload::

    byte 0    : selector  (uint8, see PID_GAIN_SELECTORS below)
    bytes 1..4: value     (IEEE-754 float, little-endian / Teensy native)

* ``0x333`` PID_GAINS_SET_CAN_ID   -- host -> Teensy, "store this value".
* ``0x334`` PID_GAINS_FB_CAN_ID    -- Teensy -> host, "this is what I have".

Confirmation strategy
---------------------

After every successful 0x333 write, the firmware echoes the **stored** value
back on 0x334. The firmware also round-robin broadcasts every gain on 0x334
at ~150 ms / selector, so:

* a freshly-started GUI populates without having to ask, and
* every typed-in value gets a green confirmation within one round-trip
  (typically <50 ms) -- if it doesn't, we flag it as PENDING (yellow) and
  eventually FAILED (red), which means either the bus is wedged, the
  selector is wrong, or the firmware rejected the value (e.g. NaN).

Usage
-----

::

    # native Python (no ROS env needed):
    python3 pid_tuner_gui.py --can-channel can0

    # via the ament-installed entry point:
    ros2 run cmd_to_can pid_tuner_gui --can-channel can0

Click a field (or use Tab / Shift-Tab) to focus it, type a number, press
Enter to send. Esc clears the input. The "Current" column shows the live
value the Teensy reports; the "Status" column shows whether your last write
to that gain was confirmed.
"""

from __future__ import annotations

import argparse
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import can

try:
    import pygame
except ImportError as exc:  # pragma: no cover -- import-time install hint
    raise SystemExit(
        "pygame is required for the PID tuner GUI. Install it with:\n"
        "  pip install pygame\n"
        f"(import error: {exc})"
    )


# ---------------------------------------------------------------------------
# Wire protocol -- keep in sync with firmware/main_project/src/main.cpp
# ---------------------------------------------------------------------------

PID_GAINS_SET_CAN_ID = 0x333
PID_GAINS_FB_CAN_ID = 0x334


class PidGainSelector(IntEnum):
    STEER_KP = 0x00
    STEER_KI = 0x01
    STEER_KD = 0x02
    VEL_KP = 0x03
    VEL_KI = 0x04
    VEL_KD = 0x05


# Display order + human-readable labels. Order also drives the Tab cycle.
PID_GAIN_ROWS: tuple[tuple[PidGainSelector, str, str], ...] = (
    (PidGainSelector.STEER_KP, "Steer Kp", "deg-error -> %torque"),
    (PidGainSelector.STEER_KI, "Steer Ki", "integrated deg-error"),
    (PidGainSelector.STEER_KD, "Steer Kd", "deg-error rate"),
    (PidGainSelector.VEL_KP,   "Vel   Kp", "vel-error -> %throttle"),
    (PidGainSelector.VEL_KI,   "Vel   Ki", "integrated vel-error"),
    (PidGainSelector.VEL_KD,   "Vel   Kd", "vel-error rate"),
)

# Confirmation tolerance. The firmware stores a float verbatim, so a perfect
# round-trip would compare equal -- but we keep a tiny epsilon so a value
# typed as "1.23" doesn't trip on the parse->float32->parse asymmetry.
_CONFIRM_EPS = 1e-5

# How long to wait for a confirmation echo before declaring the write FAILED.
# The firmware acks immediately on 0x333 RX, so anything past a few hundred
# ms means the frame never landed (bus down, wrong selector, etc.).
_CONFIRM_TIMEOUT_S = 0.75


def encode_gain_frame(selector: PidGainSelector, value: float) -> can.Message:
    """Pack a (selector, value) pair into the wire format."""
    payload = bytes([int(selector)]) + struct.pack("<f", float(value))
    return can.Message(
        arbitration_id=PID_GAINS_SET_CAN_ID,
        data=payload,
        is_extended_id=False,
    )


def decode_gain_frame(msg: can.Message) -> Optional[tuple[int, float]]:
    """Return ``(selector, value)`` from a 0x334 frame, or None if malformed.

    We don't raise on garbage -- the bus may carry a frame with the same
    arbitration ID from some other device and we'd rather drop it quietly
    than crash the GUI thread.
    """
    if msg.dlc < 5 or len(msg.data) < 5:
        return None
    selector = msg.data[0]
    try:
        (value,) = struct.unpack("<f", bytes(msg.data[1:5]))
    except struct.error:
        return None
    return selector, float(value)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class WriteStatus(IntEnum):
    IDLE = 0       # No write in flight; "Current" reflects the firmware
    PENDING = 1    # Sent on 0x333, waiting for matching echo on 0x334
    CONFIRMED = 2  # Echo matched the value we sent (within _CONFIRM_EPS)
    FAILED = 3     # Echo never came (timeout) or value mismatched


@dataclass
class GainRowState:
    """Per-row mutable state. Touched from both the UI and CAN-RX threads."""

    selector: PidGainSelector
    label: str
    units: str
    current_value: Optional[float] = None  # last value the Teensy reported
    last_update_s: float = 0.0             # monotonic timestamp of that report
    pending_value: Optional[float] = None  # value waiting for confirmation
    pending_sent_s: float = 0.0
    status: WriteStatus = WriteStatus.IDLE
    input_text: str = ""                   # what the user is typing right now


@dataclass
class TunerState:
    """Whole-app state. The lock guards every field below."""

    rows: list[GainRowState]
    lock: threading.Lock = field(default_factory=threading.Lock)
    bus_status: str = "CAN bus: not yet connected"
    last_rx_s: float = 0.0


# ---------------------------------------------------------------------------
# CAN RX thread
# ---------------------------------------------------------------------------


def can_rx_loop(
    bus: can.BusABC,
    state: TunerState,
    stop_event: threading.Event,
) -> None:
    """Pump 0x334 frames into ``state``. Runs until ``stop_event`` is set."""
    while not stop_event.is_set():
        try:
            msg = bus.recv(timeout=0.1)
        except can.CanError as exc:
            with state.lock:
                state.bus_status = f"CAN bus: RX error ({exc})"
            time.sleep(0.1)
            continue
        if msg is None:
            continue
        if msg.is_extended_id or msg.arbitration_id != PID_GAINS_FB_CAN_ID:
            continue
        decoded = decode_gain_frame(msg)
        if decoded is None:
            continue
        selector_byte, value = decoded
        now = time.monotonic()
        with state.lock:
            state.last_rx_s = now
            for row in state.rows:
                if int(row.selector) != selector_byte:
                    continue
                row.current_value = value
                row.last_update_s = now
                # If we had a write outstanding for this selector, this echo
                # is the round-trip ack. Match within the same epsilon the UI
                # uses for color-coding so a 1.23-vs-1.230000048 float32
                # rounding asymmetry doesn't show as FAILED.
                if (
                    row.status == WriteStatus.PENDING
                    and row.pending_value is not None
                    and abs(value - row.pending_value) <= _CONFIRM_EPS
                ):
                    row.status = WriteStatus.CONFIRMED
                    row.pending_value = None
                break


def confirmation_watchdog(state: TunerState, stop_event: threading.Event) -> None:
    """Promote PENDING rows whose echo never arrived to FAILED."""
    while not stop_event.is_set():
        now = time.monotonic()
        with state.lock:
            for row in state.rows:
                if (
                    row.status == WriteStatus.PENDING
                    and (now - row.pending_sent_s) > _CONFIRM_TIMEOUT_S
                ):
                    row.status = WriteStatus.FAILED
        # Cheap loop -- 50 ms is plenty for a human-perceptible UI update.
        stop_event.wait(0.05)


# ---------------------------------------------------------------------------
# Send path
# ---------------------------------------------------------------------------


def submit_gain(
    bus: can.BusABC,
    state: TunerState,
    row: GainRowState,
    raw_text: str,
) -> Optional[str]:
    """Parse ``raw_text``, send it, mark the row PENDING. Returns error text or None."""
    text = raw_text.strip()
    if not text:
        return "empty"
    try:
        value = float(text)
    except ValueError:
        return f"not a number: {text!r}"
    # Reject silly values that would brick the PID; the firmware also
    # rejects NaN/Inf, but it's friendlier to fail in the UI before sending.
    if not _is_finite(value):
        return "non-finite"

    msg = encode_gain_frame(row.selector, value)
    try:
        bus.send(msg)
    except can.CanError as exc:
        return f"send failed: {exc}"

    with state.lock:
        row.pending_value = value
        row.pending_sent_s = time.monotonic()
        row.status = WriteStatus.PENDING
        # Clear the input now that we've taken it; the user can keep typing
        # immediately for the next iteration.
        row.input_text = ""
    return None


def _is_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# Pygame UI
# ---------------------------------------------------------------------------

# Colors. Picked for legibility on a dark background; tweak to taste.
_COLOR_BG = (18, 18, 22)
_COLOR_PANEL = (30, 30, 38)
_COLOR_PANEL_FOCUS = (44, 44, 60)
_COLOR_TEXT = (220, 220, 230)
_COLOR_MUTED = (140, 140, 155)
_COLOR_LABEL = (160, 200, 255)
_COLOR_INPUT_BG = (10, 10, 14)
_COLOR_INPUT_BG_FOCUS = (10, 30, 50)
_COLOR_BORDER = (70, 70, 90)
_COLOR_BORDER_FOCUS = (110, 170, 255)

_COLOR_STATUS_IDLE = (130, 130, 145)
_COLOR_STATUS_PENDING = (240, 200, 60)
_COLOR_STATUS_CONFIRMED = (90, 210, 120)
_COLOR_STATUS_FAILED = (240, 100, 100)

_STATUS_LABELS = {
    WriteStatus.IDLE: "live",
    WriteStatus.PENDING: "pending",
    WriteStatus.CONFIRMED: "confirmed",
    WriteStatus.FAILED: "FAILED",
}
_STATUS_COLORS = {
    WriteStatus.IDLE: _COLOR_STATUS_IDLE,
    WriteStatus.PENDING: _COLOR_STATUS_PENDING,
    WriteStatus.CONFIRMED: _COLOR_STATUS_CONFIRMED,
    WriteStatus.FAILED: _COLOR_STATUS_FAILED,
}


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "  -- waiting --"
    return f"{value:+.6f}"


def run_ui(
    bus: can.BusABC,
    state: TunerState,
    stop_event: threading.Event,
) -> None:
    pygame.init()
    pygame.display.set_caption("Auto-IQ PID tuner")
    screen = pygame.display.set_mode((760, 520))
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("dejavusansmono", 22, bold=True)
    font_row = pygame.font.SysFont("dejavusansmono", 18)
    font_small = pygame.font.SysFont("dejavusansmono", 14)
    font_input = pygame.font.SysFont("dejavusansmono", 18)

    focused_idx = 0  # index into state.rows
    last_error: Optional[tuple[str, float]] = None  # (text, monotonic_ts)

    # Geometry constants. Recomputed inside the draw loop so the row
    # rectangles can drive both rendering and click hit-testing without
    # duplicating the layout math.
    margin_x = 24
    row_height = 56
    row_gap = 6
    rows_top_y = 80

    def row_rect(idx: int) -> pygame.Rect:
        return pygame.Rect(
            margin_x,
            rows_top_y + idx * (row_height + row_gap),
            screen.get_width() - 2 * margin_x,
            row_height,
        )

    def input_rect(rect: pygame.Rect) -> pygame.Rect:
        # Input box: right-aligned third of the row. Same on every row so
        # tabbing through them is visually stable.
        w = 240
        return pygame.Rect(rect.right - w - 12, rect.y + 12, w, rect.height - 24)

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                with state.lock:
                    for idx in range(len(state.rows)):
                        if row_rect(idx).collidepoint(event.pos):
                            focused_idx = idx
                            break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    with state.lock:
                        state.rows[focused_idx].input_text = ""
                elif event.key == pygame.K_TAB:
                    n = len(state.rows)
                    shift = bool(event.mod & pygame.KMOD_SHIFT)
                    focused_idx = (focused_idx + (-1 if shift else 1)) % n
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    with state.lock:
                        row = state.rows[focused_idx]
                        text = row.input_text
                    err = submit_gain(bus, state, row, text)
                    if err is not None:
                        last_error = (
                            f"{row.label}: {err}",
                            time.monotonic(),
                        )
                elif event.key == pygame.K_BACKSPACE:
                    with state.lock:
                        row = state.rows[focused_idx]
                        if row.input_text:
                            row.input_text = row.input_text[:-1]
                else:
                    ch = event.unicode
                    # Allow only characters that can appear in a float
                    # literal; rejecting at keystroke time is friendlier
                    # than letting them type junk and then rejecting on
                    # Enter. Also ban whitespace -- you can't tab past a
                    # space-padded number into the next field.
                    if ch and ch in "0123456789.+-eE":
                        with state.lock:
                            state.rows[focused_idx].input_text += ch

        # ------------------------------------------------------------------
        # Draw
        # ------------------------------------------------------------------
        screen.fill(_COLOR_BG)

        # Title + bus status row
        title = font_title.render("PID gain tuner  (Tab to switch, Enter to send)", True, _COLOR_TEXT)
        screen.blit(title, (margin_x, 18))

        with state.lock:
            bus_status = state.bus_status
            last_rx = state.last_rx_s
        if last_rx > 0.0:
            age = time.monotonic() - last_rx
            rx_text = f"last 0x{PID_GAINS_FB_CAN_ID:03X} {age:.2f}s ago"
        else:
            rx_text = "no 0x{:03X} frames seen yet".format(PID_GAINS_FB_CAN_ID)
        sub = font_small.render(f"{bus_status}   |   {rx_text}", True, _COLOR_MUTED)
        screen.blit(sub, (margin_x, 50))

        # Per-gain rows
        with state.lock:
            rows_snapshot = [
                (
                    idx,
                    row.label,
                    row.units,
                    row.current_value,
                    row.input_text,
                    row.status,
                    int(row.selector),
                )
                for idx, row in enumerate(state.rows)
            ]

        for idx, label, units, current_value, input_text, status, selector_byte in rows_snapshot:
            rect = row_rect(idx)
            focused = idx == focused_idx
            pygame.draw.rect(
                screen,
                _COLOR_PANEL_FOCUS if focused else _COLOR_PANEL,
                rect,
                border_radius=6,
            )
            pygame.draw.rect(
                screen,
                _COLOR_BORDER_FOCUS if focused else _COLOR_BORDER,
                rect,
                width=1,
                border_radius=6,
            )

            label_surf = font_row.render(label, True, _COLOR_LABEL)
            screen.blit(label_surf, (rect.x + 12, rect.y + 6))

            units_surf = font_small.render(
                f"sel=0x{selector_byte:02X}  {units}",
                True,
                _COLOR_MUTED,
            )
            screen.blit(units_surf, (rect.x + 12, rect.y + 30))

            cur_surf = font_row.render(
                f"current: {_format_value(current_value)}",
                True,
                _COLOR_TEXT,
            )
            screen.blit(cur_surf, (rect.x + 220, rect.y + 18))

            irect = input_rect(rect)
            pygame.draw.rect(
                screen,
                _COLOR_INPUT_BG_FOCUS if focused else _COLOR_INPUT_BG,
                irect,
                border_radius=4,
            )
            pygame.draw.rect(
                screen,
                _COLOR_BORDER_FOCUS if focused else _COLOR_BORDER,
                irect,
                width=1,
                border_radius=4,
            )

            display_text = input_text
            if focused:
                # Soft blink at 2 Hz so it's obvious where focus is.
                if int(time.monotonic() * 2) % 2 == 0:
                    display_text = display_text + "_"
            placeholder = not input_text and not focused
            text_color = _COLOR_MUTED if placeholder else _COLOR_TEXT
            shown = display_text if input_text or focused else "type new value..."
            input_surf = font_input.render(shown, True, text_color)
            screen.blit(input_surf, (irect.x + 8, irect.y + 4))

            status_label = _STATUS_LABELS[status]
            status_color = _STATUS_COLORS[status]
            status_surf = font_small.render(status_label, True, status_color)
            screen.blit(
                status_surf,
                (irect.right - status_surf.get_width() - 4, irect.bottom + 2),
            )

        # Footer: last error, fades after a few seconds
        if last_error is not None:
            err_text, err_ts = last_error
            age = time.monotonic() - err_ts
            if age < 4.0:
                err_surf = font_small.render(err_text, True, _COLOR_STATUS_FAILED)
                screen.blit(err_surf, (margin_x, screen.get_height() - 24))
            else:
                last_error = None

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="pygame GUI for tuning Teensy steering/velocity PID gains "
                    "over CAN (IDs 0x333 set / 0x334 feedback).",
    )
    p.add_argument(
        "--can-interface",
        default="socketcan",
        help="python-can interface (default: socketcan).",
    )
    p.add_argument(
        "--can-channel",
        default="can0",
        help="python-can channel (default: can0).",
    )
    p.add_argument(
        "--can-bitrate",
        type=int,
        default=250000,
        help="CAN bitrate in bits/s (default: 250000). Ignored by socketcan.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    rows = [
        GainRowState(selector=sel, label=label, units=units)
        for sel, label, units in PID_GAIN_ROWS
    ]
    state = TunerState(rows=rows)

    try:
        bus = can.Bus(
            interface=args.can_interface,
            channel=args.can_channel,
            bitrate=args.can_bitrate,
        )
    except Exception as exc:  # noqa: BLE001 -- python-can raises a zoo of types
        print(
            f"Failed to open CAN bus ({args.can_interface}:{args.can_channel}): {exc}",
            file=sys.stderr,
        )
        return 1

    state.bus_status = (
        f"CAN bus: {args.can_interface}:{args.can_channel} @ {args.can_bitrate} bps"
    )

    stop_event = threading.Event()
    rx_thread = threading.Thread(
        target=can_rx_loop,
        args=(bus, state, stop_event),
        name="pid-tuner-rx",
        daemon=True,
    )
    wd_thread = threading.Thread(
        target=confirmation_watchdog,
        args=(state, stop_event),
        name="pid-tuner-wd",
        daemon=True,
    )
    rx_thread.start()
    wd_thread.start()

    try:
        run_ui(bus, state, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        rx_thread.join(timeout=1.0)
        wd_thread.join(timeout=1.0)
        try:
            bus.shutdown()
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
