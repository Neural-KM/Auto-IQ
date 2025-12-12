# Auto-IQ

A Level 4 autonomous vehicle simulation framework built on CARLA.

## Overview

Auto-IQ is a modular autonomous driving system that integrates perception, planning, and control modules for self-driving vehicle simulation.

## Requirements
1. Download the relevant wheels from PythonAPI/carla/dist/
>**Note** - Carla0.9.16 only has the library wheels for python3.10,3.11,3.12
2. ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start main.py
   ```bash
   python main.py --spawn 50 --dest 250 --sync --visualize waypoints,sensors,route,live --controller pid
   ```
   Use relevant visualize option to avoid crowding
2. Generate static obstacle
   ```bash
   python3 tools/spawn_obstacle.py --type car --distance 40 30 --lane  right center
   ```
### Key Controls

| Key | Action |
|-----|--------|
| `G` | Plan route to destination |
| `N` | Start navigation (drive planned route) |
| `↑/↓` | Adjust target speed |
| `←/→` | Force lane change |
| `B` | Reset behavior |
| `1/2/3` | Switch display layouts |
| `ESC/Q` | Quit |

## License

Apache License 2.0
