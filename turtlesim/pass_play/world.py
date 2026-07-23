#!/usr/bin/env python3
"""
Pass-play WORLD / puck node  --  DRAG model.

turtlesim has no physics, and (per the requirement) the puck CANNOT be shot: it
only moves while a robot is in contact and pushing it, and it stops the instant
contact is lost. So the puck is a passive object that a robot "captures" when its
stick tip (end-effector) touches it while that robot is carrying; it then rides
at the stick tip and is dragged along until the robot releases. It never moves on
its own.

Subscribes:
  /robot1/pose, /robot2/pose            (turtlesim/Pose)
  /robot1/carrying, /robot2/carrying    (std_msgs/Bool)  "engaging / pushing"
Publishes:
  /puck/speed                           (std_msgs/Float64) so robots can tell when
                                        the puck is moving vs. settled.
"""
import math
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, Kill, SetPen, TeleportAbsolute

# ---- Layout (turtlesim arena is ~[0, 11.08]) --------------------------------
GOAL_POS      = (5.5, 10.0)
GOAL_THETA    = 0.0                     # goal x-axis runs along the net mouth (horizontal)
NET_BOX       = (4.0, 7.0, 9.4, 10.6)   # xmin, xmax, ymin, ymax
REGION_CENTER = (5.5, 6.5)
REGION_RADIUS = 1.5
PUCK_START    = (2.5, 2.5)
ROBOT1_START  = (1.0, 1.0)              # First robot; Second robot start is random.

# ---- Contact / carry model --------------------------------------------------
L_EE      = 0.4      # robot stick-tip look-ahead (MUST match the controllers)
CAPTURE_R = 0.30     # stick tip within this of the puck (while carrying) captures it
TICK      = 0.033    # ~30 Hz


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class HockeyWorld(Node):
    def __init__(self):
        super().__init__('hockey_world')
        self.spawn_cli = self.create_client(Spawn, '/spawn')
        self.kill_cli = self.create_client(Kill, '/kill')
        self.get_logger().info('Waiting for turtlesim (/spawn, /kill)...')
        self.spawn_cli.wait_for_service()
        self.kill_cli.wait_for_service()

        # Puck state (this node is the authority on the puck's position).
        self.px, self.py = float(PUCK_START[0]), float(PUCK_START[1])
        self.puck_speed = 0.0
        self.attached = None          # None | 'robot1' | 'robot2'
        self.scored = False

        # Robot state
        self.r1 = None
        self.r2 = None
        self.r1_carry = False
        self.r2_carry = False

        self._build_scene()

        self.puck_tele = self.create_client(TeleportAbsolute, '/puck/teleport_absolute')
        self.puck_tele.wait_for_service()

        self.create_subscription(Pose, '/robot1/pose', lambda m: setattr(self, 'r1', m), 10)
        self.create_subscription(Pose, '/robot2/pose', lambda m: setattr(self, 'r2', m), 10)
        self.create_subscription(Bool, '/robot1/carrying', lambda m: setattr(self, 'r1_carry', m.data), 10)
        self.create_subscription(Bool, '/robot2/carrying', lambda m: setattr(self, 'r2_carry', m.data), 10)
        self.speed_pub = self.create_publisher(Float64, '/puck/speed', 10)

        self.timer = self.create_timer(TICK, self.step)
        self.get_logger().info('Hockey world ready. Puck at rest — it moves only when a robot pushes it.')

    # ---- blocking service helper (only used during setup, before the timer) --
    def _call(self, client, req):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result()

    def _spawn(self, name, x, y, th):
        req = Spawn.Request()
        req.x, req.y, req.theta, req.name = float(x), float(y), float(th), name
        return self._call(self.spawn_cli, req)

    def _kill(self, name):
        req = Kill.Request()
        req.name = name
        return self._call(self.kill_cli, req)

    def _set_pen(self, name, r, g, b, width, off):
        cli = self.create_client(SetPen, f'/{name}/set_pen')
        if not cli.wait_for_service(timeout_sec=5.0):
            return
        req = SetPen.Request()
        req.r, req.g, req.b, req.width, req.off = r, g, b, width, off
        self._call(cli, req)

    # ---- scene ---------------------------------------------------------------
    def _build_scene(self):
        try:
            self._kill('turtle1')          # remove the default turtle
        except Exception:
            pass
        self._spawn('robot1', ROBOT1_START[0], ROBOT1_START[1], 0.0)
        rx = random.uniform(1.0, 10.0)
        ry = random.uniform(0.5, 3.5)
        self._spawn('robot2', rx, ry, random.uniform(-math.pi, math.pi))
        self._spawn('puck', PUCK_START[0], PUCK_START[1], 0.0)
        self._spawn('goal', GOAL_POS[0], GOAL_POS[1], GOAL_THETA)
        # Robots leave no ink; puck leaves a blue trail so you can see its drag path.
        self._set_pen('robot1', 0, 0, 0, 0, 1)
        self._set_pen('robot2', 0, 0, 0, 0, 1)
        self._set_pen('goal', 0, 0, 0, 0, 1)
        self._set_pen('puck', 60, 60, 255, 2, 0)
        self._draw_static()
        self.get_logger().info(f'Scene built. Second robot spawned at ({rx:.1f}, {ry:.1f}).')

    def _draw_static(self):
        """Trace the goal net box and the green region with a throwaway pen turtle."""
        self._spawn('pen', 0.5, 0.5, 0.0)
        tele = self.create_client(TeleportAbsolute, '/pen/teleport_absolute')
        setpen = self.create_client(SetPen, '/pen/set_pen')
        if not tele.wait_for_service(timeout_sec=5.0) or not setpen.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Pen services unavailable; skipping scene drawing.')
            return

        def pen(r, g, b, w, off):
            req = SetPen.Request()
            req.r, req.g, req.b, req.width, req.off = r, g, b, w, off
            self._call(setpen, req)

        def goto(x, y):
            req = TeleportAbsolute.Request()
            req.x, req.y, req.theta = float(x), float(y), 0.0
            self._call(tele, req)

        # goal net box (dark grey)
        x0, x1, y0, y1 = NET_BOX
        pen(0, 0, 0, 1, 1); goto(x0, y0)
        pen(30, 30, 30, 3, 0); goto(x1, y0); goto(x1, y1); goto(x0, y1); goto(x0, y0)
        # receiving region (green circle)
        cx, cy = REGION_CENTER
        pen(0, 0, 0, 1, 1); goto(cx + REGION_RADIUS, cy)
        pen(0, 180, 0, 2, 0)
        for i in range(1, 37):
            a = 2 * math.pi * i / 36.0
            goto(cx + REGION_RADIUS * math.cos(a), cy + REGION_RADIUS * math.sin(a))
        pen(0, 0, 0, 1, 1)
        self._kill('pen')

    # ---- puck physics (drag only) -------------------------------------------
    def step(self):
        prev = (self.px, self.py)
        if not self.scored:
            self._update_carry()
        self._check_goal()
        self.puck_speed = math.hypot(self.px - prev[0], self.py - prev[1]) / TICK
        m = Float64()
        m.data = self.puck_speed
        self.speed_pub.publish(m)

    def _update_carry(self):
        robots = {'robot1': (self.r1, self.r1_carry),
                  'robot2': (self.r2, self.r2_carry)}
        # capture: a carrying robot whose stick tip touches the puck grabs it
        if self.attached is None:
            for name, (pose, carry) in robots.items():
                if pose is None or not carry:
                    continue
                ex = pose.x + L_EE * math.cos(pose.theta)
                ey = pose.y + L_EE * math.sin(pose.theta)
                if math.hypot(ex - self.px, ey - self.py) < CAPTURE_R:
                    self.attached = name
                    self.get_logger().info(f'{name} captured the puck.')
                    break
        # carry: while held, the puck rides at the stick tip; release freezes it
        if self.attached is not None:
            pose, carry = robots[self.attached]
            if carry and pose is not None:
                self.px = min(10.9, max(0.3, pose.x + L_EE * math.cos(pose.theta)))
                self.py = min(10.9, max(0.3, pose.y + L_EE * math.sin(pose.theta)))
                self._teleport_puck(pose.theta)
            else:
                self.get_logger().info(
                    f'{self.attached} released the puck at ({self.px:.2f}, {self.py:.2f}).')
                self.attached = None

    def _teleport_puck(self, theta):
        req = TeleportAbsolute.Request()
        req.x, req.y, req.theta = float(self.px), float(self.py), float(theta)
        self.puck_tele.call_async(req)   # fire-and-forget: never block inside the timer

    def _check_goal(self):
        if self.scored:
            return
        x0, x1, y0, y1 = NET_BOX
        if x0 <= self.px <= x1 and y0 <= self.py <= y1:
            self.scored = True
            self.get_logger().info('===============   GOAL!   ===============')


def main(args=None):
    rclpy.init(args=args)
    node = HockeyWorld()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
