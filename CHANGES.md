# Controller Changes Log

Running log of changes to the controller, plus steps to run/test each version.
Newest entries on top. Uncommitted in git as of this entry — see `git status`.

---

## 2026-07-27 — `--shoot_mode push`: side-gripped stick shepherds the puck into the goal

Alternative shooting strategy for robot 6 (swing stays the default and the fallback).
Instead of spinning to whack the puck, the shooter grips the stick **sideways** so the
ground-touching blade rides at a forward-lateral offset from the chassis, then drives
behind the puck and pushes it goalward, releasing near the mouth so it slides in.
Removes the swing's ~100 ms contact-timing problem entirely and works on a moving puck.

### `robot.py`
- **`--shoot_mode {swing,push}`** (default swing — nothing changes unless passed).
- Push mode changes `MOVE_TO_STICK`: docks from the stick's **right side** (standoff
  and alignment rotate −90° off the stick axis). The stick-frame
  `sideways/vertical_offset`s still apply to the grab point unchanged.
- **Tool-point control**: the controller servos the blade point `b = (b_fwd, b_lat)`
  fixed in the body frame (`--push_b_fwd` 0.45, `--push_b_lat` −0.25 = right side) via
  `[v,w] = B⁻¹ R(θ)ᵀ ṗ`, `B = [[1, −b_lat],[0, b_fwd]]` — the 2D generalization of
  the look-ahead `L_inv`. `b_fwd` must be > 0 (singular otherwise; enforced).
  **These values must match the sim's `SIDE_BLADE_BODY`; measure them on the real
  gripped stick before the lab.**
- **New sequence `PUSH_PUCK`** (route: … MOVE_TO_WAIT → LOWER_STICK → WAIT_FOR_PASS →
  PUSH_PUCK → HIT_DONE): stage blade `--push_follow_gap` (0.5 m) behind the puck on
  the puck→goal line (target recomputed every tick, so a still-moving pass is
  intercepted) → align heading to goal → creep at `--push_speed` (0.35 m/s) →
  release at `--push_release_dist` (0.4 m; sim slide ≈ speed/0.8 = 0.44 m) →
  re-engage automatically if the puck stalls short.
- WAIT_FOR_PASS: push mode drops the puck-stopped condition (it can chase); swing
  keeps it. The virtual-puck CBF obstacle is lifted during PUSH_PUCK only.

### `~/multi_robomaster_ros_sim`
- **Carry style is inferred from grab geometry**: face-on grab → legacy centerline
  carry (swing physics); perpendicular grab → side carry, blade at body-frame
  (0.45, ∓0.25) (`SIDE_BLADE_BODY`), stick drawn diagonally, VRPN pose follows.
- **Pushing contact physics** for side carries: the blade transfers only its
  approaching (normal) velocity component — shepherds a moving puck, never pulls or
  brakes it, no resting-puck gate. Swing impulse physics unchanged for center carry.
- **Stick 2 moved to (−0.9, −0.9, 90°)** so both the right-side dock (push) and the
  face-on dock (swing fallback) stay in-bounds and clear of the puck.

### How to run — pass + push-shoot
```bash
# T1: cd ~/multi_robomaster_ros_sim && sudo bash run.sh
# T2 (shooter, start first):
python3 /ece687_robot_prod/robot.py --robot_id 6 --sim_mode --hit_mode --wait_for_pass \
    --shoot_mode push --hockey_stick_id 2 --standoff_distance 2.0
# T3 (passer, unchanged):
python3 /ece687_robot_prod/robot.py --robot_id 1 --sim_mode --hit_mode \
    --pass_to_robot 6 --standoff_distance 1.0 --hit_spin_speed 3.0
```
Swing fallback for robot 6: same command with `--shoot_mode swing` (or omit) —
recommend `--standoff_distance 1.0` there so the face-on stick standoff stays clear.

Known sim-to-real caveats: the sim's point-contact push holds the puck on the blade
perfectly; a real puck slips sideways off a straight blade (push slowly, expect
corrections). The pass still ghosts through the waiting robot's chassis in the sim
(no puck-body collision); in the lab the puck may bounce off the shooter instead.

---

## 2026-07-25 — Two-robot pass-and-shoot: `--hit_mode` / `--wait_for_pass` + sim game physics

Implements the actual project spec: robot 1 grabs stick 1 and **passes** the puck to
robot 6; robot 6 grabs stick 2, waits at the goal standoff, and **shoots** the puck
into a goal gate. Swing geometry follows the MATLAB reference sim
(`swingCenterTarget` perpendicular-offset + rotation hit).

### `robot.py`
- **New sequences** (10–14): `MOVE_TO_WAIT`, `WAIT_FOR_PASS`, `ALIGN_HIT`,
  `SPIN_HIT`, `HIT_DONE`. Sequence order is now a per-role **route list**
  (`advance_sequence` walks the route); without the new flags the route is the
  original 0–9 chain, so existing lab/mock behavior is unchanged.
- **`--hit_mode`**: `MOVE_TO_PUCK` drives to a *swing center* — offset
  `--swing_offset` (0.55 m) perpendicular to the puck→aim line, side auto-picked
  (nearest, then locked) — instead of the puck itself. Then `ALIGN_HIT` slowly points
  the stick *away* from the puck (ω capped 0.6 rad/s so the tip can't launch it
  accidentally), and `SPIN_HIT` sweeps `--hit_swing_angle` (4.71 rad) at
  `--hit_spin_speed` (4.0 rad/s); the tip crosses the puck ~π in, launching it along
  the aim line. Aim = live mocap pose of `--pass_to_robot`, or the goal
  (`--goal_x/y/yaw_deg`, default (0, −1.75) facing 90°) when passing to 0.
  Launch speed ≈ spin·0.55 m; sim puck range ≈ launch/0.8.
- **`--wait_for_pass`** (shooter role): after `MOVE_BACK_ROTATE`, drives to
  `goal + standoff_distance·(goal facing)` — **note the dual use** of
  `--standoff_distance`: it is both the stick-approach staging distance and the goal
  waiting distance — then holds until the puck has (a) moved >0.3 m from its initial
  position, (b) come within `--wait_radius` (3.0 m), and (c) stopped (<0.15 m/s,
  estimated by differencing mocap poses).
- **Hit mode safety**: the puck is injected as a CBF obstacle (`virtual_puck`) so no
  driving leg ever rolls over it; only the swinging stick tip may touch it.
- **sim_mode gripper**: gripper open/close now publishes `std_msgs/Bool` on
  `/robot{id}/gripper_sim` so the simulator attaches/releases sticks.

### `~/multi_robomaster_ros_sim` (separate repo)
- Roster is now robots **[1, 2, 3, 6]** (1 = passer, 6 = shooter, 2–3 obstacles).
- **Two sticks** (`hockey_sticks_1` at (1.2, 1.2, −90°), `hockey_sticks_2` at
  (−1.2, −1.2, 0°), flags `--stick2_x/y/theta_deg`). Stick facings deliberately point
  *away* from the puck: the approach standoff lies along the facing, and a facing
  aimed at the puck sends the robot "to the puck first" (and the virtual-puck CBF
  makes `get_valid_standoff_distance` march the standoff through and past it). Gripper close within 0.5 m of a
  stick junction attaches it; the carried stick tracks the robot (T junction 0.10 m
  ahead, tip 0.55 m ahead) and is drawn accordingly.
- **Puck physics**: a resting puck hit by a stick tip moving >0.5 m/s launches along
  the tip's velocity (capped 2.5 m/s), decays at 0.8 /s, stops at walls.
  Puck–robot-body collisions are NOT modeled.
- **Goal gate** at (0, −1.75) facing north (+x of its frame = +y world, flags
  `--goal_x/y/yaw_deg/width/depth`); logs `GOAL!` and sets the figure title when the
  puck crosses the mouth.
- **Trails** for every movable object (robot 1 red, robot 6 green, obstacles gray,
  puck dotted blue).
- Spawn resampling now also keeps clear of stick 2, the goal, and the shooter's
  waiting spot.
- **In-place game reset**: press `r` in the sim window, or
  `ros2 topic pub --once /sim/reset std_msgs/msg/Empty "{}"` — restores sticks/puck,
  respawns robots, clears trails, no process restart. Stop the controllers first.
- **`run.sh` now keeps the container alive**: first run creates a persistent
  container (`sleep infinity`) and execs the simulator into it; Ctrl+C kills only the
  simulator, and rerunning `run.sh` rebuilds + relaunches in seconds. Full teardown:
  `sudo docker rm -f dji_robomaster_ros_simulator`.

### How to run — two-robot pass-and-shoot (WSL Ubuntu)
```bash
# Terminal 1 — simulator
cd ~/multi_robomaster_ros_sim && sudo bash run.sh

# Terminal 2 — robot 1, passer
sudo docker exec -it dji_robomaster_ros_simulator bash
source /opt/ros/humble/setup.bash && source /opt/ros/ws/setup.bash
python3 /ece687_robot_prod/robot.py --robot_id 1 --sim_mode --hit_mode \
    --pass_to_robot 6 --standoff_distance 1.0 --hit_spin_speed 3.0

# Terminal 3 — robot 6, shooter
sudo docker exec -it dji_robomaster_ros_simulator bash
source /opt/ros/humble/setup.bash && source /opt/ros/ws/setup.bash
python3 /ece687_robot_prod/robot.py --robot_id 6 --sim_mode --hit_mode --wait_for_pass \
    --hockey_stick_id 2 --standoff_distance 2.0
```
Known coordination caveat: robot 1 aims at robot 6's *live* pose, so if the pass is
launched while robot 6 is still driving to its waiting spot, the puck lands where
robot 6 *was* — the 3 m `wait_radius` absorbs this, and robot 6 drives to wherever
the puck actually stops. Start robot 6 first (its stick leg is longer) if you want
the pass received cleanly at the standoff.

---

## 2026-07-25 — `--sim_mode`: run `robot.py` against the Docker multi-robot simulator

### What changed
- **`robot.py`**: new `--sim_mode` CLI flag (mutually exclusive with `--mock_mode`).
  Fakes the gripper/arm actions exactly like mock mode, but subscribes to the **real**
  `/vrpn_mocap/...` topic names. Use it to run the controller against the Docker sim
  (`~/multi_robomaster_ros_sim`, image `dji_robomaster_ros:1.0`), which fakes the
  Robohub VRPN system. `--mock_mode` behavior is unchanged.
- **`~/multi_robomaster_ros_sim` repo** (separate repo):
  - `simulator.py` now also publishes static `/vrpn_mocap/hockey_sticks_1/pose` and
    `/vrpn_mocap/hockey_puck_blue/pose`, drawing the stick as a "T" (leg = the stick's
    +x facing direction; the controller's standoff point lies along the leg) and the
    puck as a blue disc. Configurable via `--stick_x/--stick_y/--stick_theta_deg`,
    `--puck_x/--puck_y`, `--hockey_stick_id`, `--puck_color`.
    Defaults: stick (1.2, 1.2, 180°) → standoff at (1.2 − d, 1.2); puck (−1.2, 1.2).
  - Robots reduced to IDs 1–5, random spawns kept (robot 1 = controlled,
    2–5 = static obstacles for the CBF).
  - `run.sh` additionally mounts this repo at `/ece687_robot_prod` in the container.

### How to run — Docker sim rehearsal (WSL Ubuntu)
```bash
# Terminal 1 — simulator
cd ~/multi_robomaster_ros_sim && sudo bash run.sh

# Terminal 2 — controller, inside the same container
sudo docker exec -it dji_robomaster_ros_simulator bash
source /opt/ros/humble/setup.bash && source /opt/ros/ws/setup.bash
python3 -c "import scipy" || python3 -m pip install scipy   # one-time check; container is --rm
python3 /ece687_robot_prod/robot.py --robot_id 1 --sim_mode \
    --standoff_distance 1.0 --r_safety 0.35 \
    --sideways_offset 0.0 --vertical_offset 0.0
```
The arena is real-scale 4×4 m (x, y ∈ [−2, 2]) — use hardware-scale tunings
(standoff ≈ 1.0–1.5 m), not turtlesim-scale values. The sim zeroes a robot's velocity
if no `cmd_vel` arrives for 500 ms; the controller's 10 Hz loop clears that fine.

---

## 2026-07-25 — Reference version: full pick-and-place sequence + CLF-CBF obstacle avoidance

This supersedes the 2026-07-20 entry below. The offset design changed shape (see
"Sideways / vertical offset" section) — read that before reusing old tuning numbers.

Files: [`robot.py`](robot.py), [`mock_robot.py`](mock_robot.py),
[`turtlesim/turtle_sim_robot.py`](turtlesim/turtle_sim_robot.py),
[`turtlesim/turtle_sim_mock.py`](turtlesim/turtle_sim_mock.py)

### Architecture: 10-stage `Sequence` state machine
Both `robot.py` and `turtle_sim_robot.py` now share the same `Sequence` enum and
top-level `control_loop()` structure:

| # | Sequence | Real robot (`robot.py`) | Turtlesim (`turtle_sim_robot.py`) |
|---|---|---|---|
| 0 | OPEN_GRIPPER | dispatches `GripperControl` action | 2s timed placeholder |
| 1 | MOVE_EE_TO_ORIGIN | dispatches `MoveArm(x=0, z=0)` action | 2s timed placeholder |
| 2 | MOVE_EE_TO_REF_POS | dispatches `MoveArm(x=0.15, z=0.15)` action | 2s timed placeholder |
| 3 | MOVE_TO_STICK | 4-stage NID + CLF-CBF approach (see below) | same logic |
| 4 | CLOSE_GRIPPER | dispatches `GripperControl` action | 2s timed placeholder |
| 5 | LIFT_STICK | publishes `Vector3(z=+0.10)` to `cmd_arm`, waits 2s | 2s timed placeholder |
| 6 | MOVE_BACK_ROTATE | reverses `linear.x` for 3s | reverses `linear.x` for 3s |
| 7 | MOVE_TO_PUCK | 2-stage NID + CLF-CBF approach (see below) | same logic |
| 8 | LOWER_STICK | publishes `Vector3(z=-0.10)` to `cmd_arm`, waits 2s | 2s timed placeholder |
| 9 | RELEASE_PUCK | **log only**, no hardware action | 2s timed placeholder |

Gripper/arm are real `ActionClient` calls now (`GripperControl`, `MoveArm` from
`robomaster_msgs.action`) instead of the old log-only stubs — except Sequence 9
(release/shoot), which is still a stub in both files. If the field session needs an
actual shoot/release mechanism triggered, that's the remaining gap.

`--mock_mode` (robot.py) skips waiting for the real action servers and fakes success
immediately — use it for desk-testing against `mock_robot.py` without hardware.
**Without `--mock_mode`, `robot.py` blocks in `wait_for_server()` at startup** until the
gripper and arm action servers are actually up — don't be surprised if it hangs there
in the lab if those nodes aren't running yet.

### MOVE_TO_STICK: 4-stage standoff approach (new)
Old controller drove straight at the target with a 2-phase (translate-then-rotate)
loop. Now `nid_to_move_robot()` / `nid_kinematics()` run a 4-stage sub-machine
(`seq1_stage` 0-3) for the stick:
0. Rotate in place to face a **standoff point** — a point offset `standoff_distance`
   back from the (offset-adjusted) target along the stick's facing direction.
   `get_valid_standoff_distance()` auto-extends this distance in 0.1 m steps if the
   standoff point would land inside `r_safety` of a tracked obstacle.
1. Drive to the standoff point with NID + CLF-CBF-filtered velocity, capped at `v_max`.
2. Rotate in place to align the tool with the stick (approaches facing *opposite* the
   stick's own facing direction — `target_theta + π`).
3. Drive the final approach from standoff to the actual (offset-adjusted) target.

### MOVE_TO_PUCK: 2-stage direct approach (new)
Simpler `seq4_stage` 0-1: rotate in place to face the puck directly, then drive
straight in with NID + CLF-CBF. No standoff and **no offset** is applied to the puck
target (offsets only affect the stick target — see below).

### CLF-CBF obstacle avoidance (new — `solve_clf_cbf_qp`)
Every driving leg (stick stages 1 & 3, puck stage 1) routes its nominal NID velocity
through a QP (`scipy.optimize.minimize`, method `SLSQP`) that enforces:
- a **CLF constraint** guaranteeing Lyapunov decrease toward the target,
- a **CBF constraint** per nearby obstacle (tracked robots within `r_safety * 1.6`)
  keeping the look-ahead point outside `r_safety` of each one,
- a pre-QP tangent nudge that picks a dodge side per obstacle and remembers it
  (`chosen_tangent_sign`) so the robot doesn't flip-flop which way it swerves,
- low-pass filtering (`alpha=0.4`) of the output command frame-to-frame.

Obstacles come from mocap/mock pose topics for other robots on the field (see
"Topics" below). **This is a new hard dependency on `scipy`** — install with
`sudo apt install python3-scipy` if a fresh machine hits
`ModuleNotFoundError: No module named 'scipy'` (this is what happened tonight; fixed
via apt to stay consistent with the apt-installed numpy/ROS Python).

### Sideways / vertical offset — replaces the old `sideway_offset` design
**This is a behavior change from the 2026-07-20 entry below, not just a rename.**

Old design: offset was rotated relative to the stick's own facing angle
(`target_theta`) — a "left/right of the stick, from the stick's point of view" shift.

Current design (`sideways_offset`, `vertical_offset`): applied directly in **world
frame**, only to the stick target, in both the standoff point and the final approach
point:
```python
target_x = p_xg + vertical_offset   # shifts along the room's world X axis
target_y = p_yg + sideways_offset   # shifts along the room's world Y axis
```
Practical implication: which physical direction "sideways" nudges the robot now
depends on how the stick happens to be oriented in the room, **not** on the stick's
own facing direction like before. Figure out the sign empirically for each stick
placement rather than assuming a fixed left/right convention.

### Other new tunables
Declared as ROS parameters, most also exposed as CLI flags on `robot.py` /
`turtle_sim_robot.py` (flag noted where present):
- `standoff_distance` (`--standoff_distance`, default 2.5 m)
- `r_safety` (`--r_safety`, default 0.35 m) — obstacle safety radius for CBF
- `v_max` (param only, default 1.0 m/s) — velocity saturation cap
- `gamma_cbf` (1.5), `gamma_clf` (1.0), `clf_penalty` (1e3) — param only, QP tuning
- `control_frequency` (param only, robot.py, default 10.0 Hz)
- `start_sequence` (param only, default 0) — lets you jump straight into e.g.
  `MOVE_TO_STICK` for bench-testing without replaying gripper/arm stages every time
- `kp_v` default raised 0.5→1.2, `kp_w` default raised ~1.0-1.7→2.0
- `tolerance` default **differs between files**: 0.15 m in `robot.py`, 0.20 m in
  `turtle_sim_robot.py` — worth aligning before trusting sim-tuned values on hardware

### Mock/sim files rewritten
- **`mock_robot.py`**: now a real closed-loop unicycle simulator. Integrates the
  robot's pose from `/robot{id}/cmd_vel` at 50 Hz and publishes mock VRPN topics for
  self pose, stick, puck, and up to 9 synthetic "obstacle" robots auto-placed along
  the stick→puck path (alternating lateral offsets) so the CLF-CBF logic has
  something to dodge. This is what lets you run `robot.py --mock_mode` end-to-end with
  zero hardware.
- **`turtlesim/turtle_sim_mock.py`**: now a CLI-configurable target/obstacle
  publisher (`--stick_x/y/angle_rad`, `--puck_x/y/angle_rad`, `--standoff_distance`,
  and up to 7 `--obsN_x`/`--obsN_y` pairs), spawning a marker turtle for the stick,
  puck, and every obstacle. **Known issue:** it also accepts `--sideways_offset` /
  `--vertical_offset` flags and stores them, but never actually uses them anywhere —
  the spawned stick/puck markers do **not** move to reflect those offsets. Only
  `turtle_sim_robot.py`'s own offset flags affect where the robot actually drives;
  don't expect the visual marker to move if you only pass offsets to the mock.

### Topics (for reference)
- Real: `/vrpn_mocap/hockey_sticks_{id}/pose`, `/vrpn_mocap/dji_robot_{robot_id}/pose`
  (self), `/vrpn_mocap/dji_robot_{i}/pose` for other robots (obstacles),
  `/vrpn_mocap/hockey_puck_{color}/pose`
- Mock (robot.py `--mock_mode`): same names prefixed with `/mock/...`, published by
  `mock_robot.py`
- Turtlesim: `/vrpn_mocap/hockey_sticks_1/pose`, `/vrpn_mocap/puck_1/pose`,
  `/vrpn_mocap/robot_obstacle_{1..7}/pose`, published by `turtle_sim_mock.py`

### How to run — turtlesim rehearsal
```bash
# Terminal 1
ros2 run turtlesim turtlesim_node

# Terminal 2 — targets + obstacles (offsets here are cosmetic only, see known issue above)
python3  ~/ece687_robot_prod/turtlesim/turtle_sim_mock.py --obs1_x 4.0 --obs1_y 5.0

# Terminal 3 — controller; this is where offset/standoff/safety flags actually matter
python3  ~/ece687_robot_prod/turtlesim/turtle_sim_robot.py --sideways_offset 0.3 --vertical_offset 0.0 \
    --standoff_distance 2.0 --r_safety 0.35
```

### How to run — real robot code, no hardware (desk test)
```bash
# Terminal 1 — simulated VRPN + physics
python3 mock_robot.py --robot_id 1

# Terminal 2 — controller in mock mode
python3 robot.py --robot_id 1 --mock_mode \
    --sideways_offset 0.05 --vertical_offset 0.0 --standoff_distance 1.5
```

### How to run — real robot, real hardware (lab)
```bash
python3 robot.py --robot_id 1 \
    --hockey_stick_id 1 --puck_color blue \
    --sideways_offset 0.05 --vertical_offset 0.0 \
    --standoff_distance 1.5 --r_safety 0.35
```
Add `--pass_to_robot <id>` to pass instead of shoot at the goal. Confirm
`--hockey_stick_id` / `--puck_color` match the actual VRPN tags in the room before
starting. Non-CLI params (`gamma_cbf`, `gamma_clf`, `clf_penalty`,
`control_frequency`) can still be overridden with `--ros-args -p name:=value` appended
after the script's own flags.

Start with small `sideways_offset`/`vertical_offset` magnitudes (0.02-0.05 m) and
iterate sign/magnitude based on where the EE actually lands relative to the stick.

### Note: `setup.md` is missing
The repo's `setup.md` (package-creation walkthrough) is gone from disk as of this
edit — `git status` shows it deleted, not just modified. Flagging in case that wasn't
intentional; it hasn't been restored here since it's unclear whether the deletion was
deliberate cleanup or accidental.

---

## 2026-07-20 — Added `sideway_offset` parameter (superseded above)

### What changed
Added a tunable ROS parameter, `sideway_offset`, to the NID controller in `robot.py`
and `turtlesim/turtle_sim_robot.py` (files have since been substantially rewritten —
see the entry above for current behavior).

### Why
The hockey sticks sit on a base, off-center from the base's tracked pose. Driving
straight at the base's center didn't line up the arm end effector with the actual
stick. `sideway_offset` shifted the approach goal point sideways, perpendicular to the
stick's facing direction, so the EE alignment could be tuned in the field instead of
by editing code.

### How it worked (no longer current — see 2026-07-25 entry)
Default `0.0`. Nudged the goal point perpendicular to `target_theta`:
```python
p_xg += sideway_offset * math.sin(target_theta)
p_yg -= sideway_offset * math.cos(target_theta)
```
Positive = target's right side (facing along `target_theta`), negative = left.

### Known issue noticed at the time (since resolved differently)
`robot.py`'s controller didn't apply the same `l`-forward look-ahead offset to the
goal point that the turtlesim files applied, so the two controllers weren't driving to
quite the same goal point. The 2026-07-25 rewrite restructured the goal-point math
enough that this specific discrepancy no longer applies in the same form — worth a
fresh look rather than assuming it's still the same issue.
