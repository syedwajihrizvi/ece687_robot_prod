# Pass-play turtlesim demo — run guide

Two robots + a puck + a goal. First robot **drags** the puck into a receiving
region; the Second robot (random start) drives to a waiting spot, then **drags**
the settled puck into the goal.

> turtlesim has no physics/collisions, and the puck cannot be shot. The puck is a
> passive object owned by `world.py`: it only moves while a robot is in contact
> and "carrying" it (the puck rides at the robot's stick tip), and it freezes the
> instant the robot releases. It never moves on its own.

## Prerequisites
- The pass-play folder is at `/hockey/ece687_robot_prod/turtlesim/pass_play/`
  inside the container.
- Every shell: `sudo docker exec -it dji_robomaster_ros bash`, then
  `source /opt/ros/humble/setup.bash` and `source /opt/ros/ws/setup.bash`.

## Launch order (4 terminals, each sourced)

**T1 — turtlesim (start it fresh so the arena is clean):**
```bash
ros2 run turtlesim turtlesim_node
```

**T2 — the world (spawns props, draws net + region, runs the puck):**
```bash
python3 /hockey/ece687_robot_prod/turtlesim/pass_play/world.py
```

**T3 — First robot:**
```bash
python3 /hockey/ece687_robot_prod/turtlesim/pass_play/first_robot.py
```

**T4 — Second robot:**
```bash
python3 /hockey/ece687_robot_prod/turtlesim/pass_play/second_robot.py
```

## What you should see
1. A dark net box near the top, a green region circle in the middle, a goal
   turtle inside the net, the puck at lower-left, the First robot near it, and
   the Second robot at a random lower spot.
2. First robot backs to a point behind the puck, sits perpendicular to the pass
   line, pivots to face the region, makes contact, and **drags** the puck (blue
   trail) into the green region, then backs off and leaves it there.
3. Second robot drives up to its waiting spot, waits, then goes behind the
   settled puck, faces the goal, makes contact, and **drags** the puck into the
   net. World logs `GOAL!`. You confirm visually (puck inside the net box). The
   puck only ever moves while a robot is pushing it.

## Restart
Ctrl-C T2/T3/T4, then Ctrl-C + re-run **T1** (turtlesim) so the drawn net/region
and killed default turtle reset cleanly, then relaunch T2 -> T3 -> T4.
Do NOT use `/clear` or `/reset` — they erase the drawn net and region.

## Tuning knobs
- `world.py`: `STRIKE_SPEED`, `FRICTION` (range = v^2 / 2*FRICTION), `CONTACT_R`,
  layout constants (`GOAL_POS`, `NET_BOX`, `REGION_CENTER/RADIUS`, `PUCK_START`).
- controllers: `STANDOFF`, `KV`/`KW`, `POS_TOL`/`ANG_TOL`, `ADVANCE_SPEED`.
- `L` must stay equal to `world.py`'s `L_EE` (end-effector look-ahead).
