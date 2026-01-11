## Planning

**Goal:** Decide *what* to do and *how* to move.

**Deliverables:**
*   **Behavior:** High-level decisions (Change Lane, Stop, Yield, Overtake).
*   **Trajectory:** A smooth path of points ($x, y, v, t$) to follow.

**TO-DOs:**
* Lane change timeout won't work if you again see a vehicle in the new lane. There can't be a cooldown if the lane is occupied. Need to rethink this logic.
* Modify spawn obstacle to spawn vehicles in autopilot mode in a particular configuration around the ego vehicle for testing behavior planner.
**Implementation**

1.  **Waypoint Following:** Follow a fixed line. Stop if blocked. (*Implemented*)
2.  **Finite State Machine (FSM):** Switch between pre-defined behaviors (lane keep/change/stop) using rules. (*Implemented*)
3.  **Road-Frame Trajectory Generation:** Generate multiple smooth candidate trajectories and choose the best one with a cost function.
4.  **Space–Time Planning:** Plan while explicitly accounting for time and moving obstacles.
5.  **End-to-End Policies:** Use a single neural network to map sensor data and a high-level goal directly to a trajectory or control commands, bypassing the traditional modular pipeline.
6.   **FSM:** A behavior model with discrete states and rule-based transitions.
7.   **Frenet frame:** Road-aligned coordinates where $s$ is along-track distance and $d$ is lateral offset.
8.   **Quintic polynomials:** Smooth 5th-order curves used to generate comfortable trajectories.
9.   **Lattice planner:** Samples a structured set of candidate trajectories and selects by cost + collision checks.
10.   **Hybrid A\*:** A search method that produces drivable paths while respecting vehicle kinematics.
11.   **Reinforcement Learning (RL) for Behavior:** Train an RL agent to make high-level decisions (e.g., lane change, yield). The state can be a summarized representation of the environment (e.g., BEV grid), and the action space is discrete (e.g., "keep lane", "change left"). The reward function is designed to encourage safe and efficient driving.
12.   **Diffusion Models for Planning:** These models learn to reverse a noising process. For planning, they can be trained on a dataset of expert trajectories. At inference time, they start with random noise and iteratively "denoise" it into a realistic, feasible trajectory that respects the surrounding environment. This allows for generating diverse and context-aware plans.
13.   **Transformer-based End-to-End Models:** Models like UniAD or VAD use a transformer architecture to ingest sensor data and produce a full stack of outputs, including trajectories and control commands. They learn the entire driving task from data, often using imitation learning on a large dataset of human driving.

**Supporting algorithms (small but useful):**
*   **Costmap / occupancy grid:** Converts perception into a grid cost representation for fast collision checking.
*   **Collision checking primitives:** Circle/box overlaps, swept-volume checks, time-to-collision (TTC) checks.
*   **Speed profiling:** Builds a target speed along the path using curvature limits and obstacle distances.
*   **Rule handling:** Stop-line handling, yielding logic, right-of-way heuristics (even before full prediction).
*   **Optimization-based planning (OCP/MPC):** Plan by solving a constrained optimization directly (lane bounds, obstacle clearance, speed/accel/jerk, curvature) rather than picking from a fixed trajectory set.
*   **Planner+tracker unification:** Solve for the trajectory and controls together as one optimal control problem, so the controller always tracks something dynamically feasible.

---