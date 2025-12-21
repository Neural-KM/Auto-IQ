# Planning & Control Subsystems

## Planning

**Goal:** Decide *what* to do and *how* to move.

**Deliverables:**
*   **Behavior:** High-level decisions (Change Lane, Stop, Yield, Overtake).
*   **Trajectory:** A smooth path of points ($x, y, v, t$) to follow.

**Implementation options:**

1.  **Waypoint Following:** Follow a fixed line. Stop if blocked. (*Implemented*)
2.  **Finite State Machine (FSM):** Switch between pre-defined behaviors (lane keep/change/stop) using rules. (*Implemented*)
3.  **Road-Frame Trajectory Generation:** Generate multiple smooth candidate trajectories and choose the best one with a cost function.
4.  **Space–Time Planning:** Plan while explicitly accounting for time and moving obstacles.
5.  **End-to-End Policies:** Use a single neural network to map sensor data and a high-level goal directly to a trajectory or control commands, bypassing the traditional modular pipeline.

**Algorithms (brief):**
*   **FSM:** A behavior model with discrete states and rule-based transitions.
*   **Frenet frame:** Road-aligned coordinates where $s$ is along-track distance and $d$ is lateral offset.
*   **Quintic polynomials:** Smooth 5th-order curves used to generate comfortable trajectories.
*   **Lattice planner:** Samples a structured set of candidate trajectories and selects by cost + collision checks.
*   **Hybrid A\*:** A search method that produces drivable paths while respecting vehicle kinematics.
*   **Reinforcement Learning (RL) for Behavior:** Train an RL agent to make high-level decisions (e.g., lane change, yield). The state can be a summarized representation of the environment (e.g., BEV grid), and the action space is discrete (e.g., "keep lane", "change left"). The reward function is designed to encourage safe and efficient driving.
*   **Diffusion Models for Planning:** These models learn to reverse a noising process. For planning, they can be trained on a dataset of expert trajectories. At inference time, they start with random noise and iteratively "denoise" it into a realistic, feasible trajectory that respects the surrounding environment. This allows for generating diverse and context-aware plans.
*   **Transformer-based End-to-End Models:** Models like UniAD or VAD use a transformer architecture to ingest sensor data and produce a full stack of outputs, including trajectories and control commands. They learn the entire driving task from data, often using imitation learning on a large dataset of human driving.

**Supporting algorithms (small but useful):**
*   **Costmap / occupancy grid:** Converts perception into a grid cost representation for fast collision checking.
*   **Collision checking primitives:** Circle/box overlaps, swept-volume checks, time-to-collision (TTC) checks.
*   **Speed profiling:** Builds a target speed along the path using curvature limits and obstacle distances.
*   **Rule handling:** Stop-line handling, yielding logic, right-of-way heuristics (even before full prediction).
*   **Optimization-based planning (OCP/MPC):** Plan by solving a constrained optimization directly (lane bounds, obstacle clearance, speed/accel/jerk, curvature) rather than picking from a fixed trajectory set.
*   **Planner+tracker unification:** Solve for the trajectory and controls together as one optimal control problem, so the controller always tracks something dynamically feasible.

---

## Control

**Goal:** Execute the planned trajectory.

**Deliverables:**
*   **Steering:** Wheel angle command.
*   **Throttle/Brake:** Acceleration/Deceleration commands.

**Implementation options:**

1.  **PID Control:** Simple error correction (reactive). (*Implemented*)
2.  **Geometric Tracking:** Track a path using geometry (lookahead / cross-track error). (*Implemented*)
3.  **Model Predictive Control (MPC):** Optimize steering/throttle/brake over a short horizon with constraints. (*Implemented*)
4.  **Learning-based Control:** Train a policy that maps state to actions from data. This can be done via:
    *   **Imitation Learning (IL):** Train a network to directly clone the behavior of an expert (e.g., a human driver or a powerful MPC). A common technique is DAgger (Dataset Aggregation), where the model is iteratively trained on data collected from its own rollouts to correct for compounding errors.
    *   **Reinforcement Learning (RL):** Train an agent to maximize a reward signal through trial and error. The state includes vehicle dynamics and trajectory error, and the action is the control command.

**Algorithms (brief):**
*   **PID:** Uses proportional/integral/derivative error terms to compute actuation.
*   **Pure Pursuit:** Steers toward a lookahead point on the path by commanding curvature.
*   **Stanley:** Combines heading error and cross-track error to compute steering.
*   **MPC:** Solves an optimization over a short horizon to minimize tracking error under constraints.
*   **PPO:** A policy-gradient RL method with constrained updates for stability. It's a robust, general-purpose RL algorithm often used as a strong baseline for continuous control tasks.
*   **SAC:** An RL method that optimizes reward while encouraging exploration via entropy. It is known for being highly sample-efficient and stable, making it a popular choice for training control policies from scratch.
*   **Implicit Behavioral Cloning (IBC):** A modern imitation learning technique that learns an implicit energy-based model of the expert's policy. At inference, it uses optimization to find the action that minimizes the learned energy function, resulting in more robust and multi-modal behavior compared to direct regression.

**Supporting algorithms (small but useful):**
*   **Feedforward + feedback blending:** Adds a feedforward steering/accel term from curvature/speed profile and uses feedback to correct errors.
*   **Actuator limits & rate limiting:** Prevents unrealistic command jumps (steering rate, throttle/brake limits).
*   **Anti-windup (for PID):** Prevents the integral term from accumulating when actuators saturate.
*   **Friction-aware limits:** Adjusts target speed/accel based on estimated tire-road friction.
*   **Friction circle (combined slip):** Couples longitudinal and lateral feasibility, e.g. $\sqrt{F_x^2 + F_y^2} \le \mu F_z$. In practice this is often enforced as $a_x^2 + a_y^2 \le (\mu g)^2$ inside MPC so braking/accel and steering are coordinated.
*   **Real-time OCP solvers (acados + HPIPM):** Implement the combined planner+tracker as an OCP solved with acados (real-time SQP/RTI), typically using HPIPM as the QP backend.

---

## Tasks
- [x] Implement basic PID + Pure Pursuit controller
    * Works better than mpc at currrent progress
- [x] Implement Model Predictive Control (MPC) controller
    * Jittery throttle and steerings. Needs tuning(if tuning is a problem that is)
- [ ] Check why is mpc misbehaving on lane change maneuvers
- [ ] Use a dynamic bicycle model in MPC instead of kinematic
- [ ] Smmothen lane change trajectory generation
- [ ] Implement friction circle constraints while generating velocity profiles from paths
- [ ] Can the model of the car itself be used to generate trajectories? Is this equivlent to planning+tracking unification?
- [ ] Find new methods to generate path+trajectory together for stronger feasibility
- [ ] 
