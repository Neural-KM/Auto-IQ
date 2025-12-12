#!/usr/bin/env python3
"""
Clear World (Simple)
====================

Minimal utility to:
- Connect to CARLA
- Destroy all dynamic actors (vehicles, walkers, sensors, props)
- Reload the current map to clear any lingering debug drawings

No arguments, no options. Just run it.
"""

import sys

import carla


def main() -> int:
    host = "127.0.0.1"
    port = 2000
    timeout = 5.0

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
    except Exception as e:
        print(f"[ClearWorld] Failed to connect to CARLA at {host}:{port}: {e}")
        return 1

    actors = world.get_actors()
    to_destroy = []

    for actor in actors:
        try:
            type_id = actor.type_id or ""
        except Exception:
            type_id = ""
        if type_id.startswith("vehicle.") or type_id.startswith("walker.") or type_id.startswith("sensor.") or type_id.startswith("static.prop"):
            to_destroy.append(actor)

    # Destroy sensors first
    sensors = [a for a in to_destroy if a.type_id.startswith("sensor.")]
    others = [a for a in to_destroy if not a.type_id.startswith("sensor.")]

    destroyed = 0
    for a in sensors + others:
        try:
            a.destroy()
            destroyed += 1
        except Exception as e:
            print(f"[ClearWorld] Failed to destroy id={a.id} type={a.type_id}: {e}")

    print(f"[ClearWorld] Destroyed {destroyed} actors.")

    print("[ClearWorld] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
