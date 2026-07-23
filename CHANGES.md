# Controller Changes Log

Running log of changes made to the NID controller code to prep for lab sessions,
plus the steps to run/test each change. Newest entries on top.

---

## 2026-07-20 — Added `sideway_offset` parameter

### What changed
Added a new tunable ROS parameter, `sideway_offset`, to the NID controller in:
- [`robot.py`](robot.py) — `Robot.nid_to_move_robot()`
- [`turtlesim/turtle_sim_robot.py`](turtlesim/turtle_sim_robot.py) — `TurtlesimSequenceController.nid_kinematics()`

### Why
The hockey sticks sit on a base, off-center from the base's tracked pose. Driving
straight at the base's center doesn't line up the arm end effector with the actual
stick. `sideway_offset` shifts the approach goal point sideways (perpendicular to the
stick's facing direction) so the EE can be aligned by tuning one number in the field,
instead of editing code.

### How it works
Declared next to the other tunables (`l`, `kp_v`, `kp_w`, `tolerance`), default `0.0`
(no behavior change unless set). After the goal point `(p_xg, p_yg)` and `target_theta`
are computed, the goal is nudged perpendicular to `target_theta`:

```python
sideway_offset = self.get_parameter('sideway_offset').value
p_xg += sideway_offset * math.sin(target_theta)
p_yg -= sideway_offset * math.cos(target_theta)
```

Sign convention: **positive = target's right side** (i.e. the right side when facing
along `target_theta`), **negative = left**. Applies to every target in the sequence
(hockey stick and puck), not just the stick.

### How to test in sim (turtlesim) first
```bash
# Terminal 1
ros2 run turtlesim turtlesim_node

# Terminal 2
python3 turtlesim/turtle_sim_mock.py

# Terminal 3 — try a nonzero offset and watch which way the approach point shifts
python3 turtlesim/turtle_sim_robot.py --ros-args -p sideway_offset:=0.05
```
Increase/flip the sign of `0.05` and re-run to confirm which direction is "right"
matches what you expect before trying it on the real robot.

### How to run on the real robot
Pass the parameter override on the command line — no code edit needed between runs:
```bash
ros2 run robot_controller robot_node --ros-args -p sideway_offset:=0.05 -- --robot_id 1
```
(Combine with the existing `--robot_id` / `--pass_to_robot` / `--mock_mode` /
`--orient_to_stick` args as usual — see [setup.md](setup.md) for the base run command.
ROS parameter overrides go in a `--ros-args -p key:=value` block, separated from the
script's own argparse flags by `--`.)

Start with small magnitudes (e.g. `0.02`–`0.05` m) and adjust based on how far off the
EE lands from the stick, iterating sign/magnitude in the lab.

### Known issue noticed while making this change (not fixed)
`robot.py`'s `nid_to_move_robot()` does **not** apply the `l`-forward look-ahead offset
to the goal point (`p_xg += l * cos(target_theta)`, `p_yg += l * sin(target_theta)`)
that both `turtle_sim_robot.py` and `move_turtle_sim.py` apply. So the real-robot
controller and the turtlesim controller aren't driving to quite the same goal point
today. Worth confirming with the team whether that's intentional before the lab
session, since it affects where the robot physically stops relative to the stick.
