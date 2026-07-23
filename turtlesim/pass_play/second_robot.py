#!/usr/bin/env python3
"""
SECOND robot controller  --  DRAG model.

1. GO_WAIT : drive from its random start to the waiting spot inside the region.
2. WAIT    : hold there, oriented so its x-axis is PERPENDICULAR to the goal
             turtle's x-axis (world frame), until the puck has arrived in the
             region and settled (not being carried, speed ~ 0).
3. Then line up behind the settled puck and DRAG it into the net (no shooting):
   NAVIGATE -> ORIENT_PERP -> AIM -> ENGAGE -> PUSH -> RELEASE -> HOLD.

The /robot2/carrying flag is True during ENGAGE + PUSH.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64
from turtlesim.msg import Pose

REGION_CENTER = (5.5, 6.5)       # must match world.py
REGION_RADIUS = 1.5
WAIT_POINT    = (6.8, 6.5)       # inside the region, off to the side of the puck's arrival
NET_BOX       = (4.0, 7.0, 9.4, 10.6)   # must match world.py; puck inside = delivered

L        = 0.4
KV       = 0.8
KW       = 2.5
VMAX     = 1.5
WMAX     = 3.0
POS_TOL  = 0.20
ANG_TOL  = 0.04
STANDOFF = 1.0
ENGAGE_SPEED = 0.7
CAPTURE_DETECT = 0.15
RELEASE_BACK = 0.6


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class SecondRobot(Node):
    def __init__(self):
        super().__init__('second_robot')
        self.name = 'robot2'
        self.pose = None
        self.puck = None
        self.goal = None
        self.puck_speed = 0.0
        self.state = 'GO_WAIT'
        self.release_start = None
        self.create_subscription(Pose, f'/{self.name}/pose', lambda m: setattr(self, 'pose', m), 10)
        self.create_subscription(Pose, '/puck/pose', lambda m: setattr(self, 'puck', m), 10)
        self.create_subscription(Pose, '/goal/pose', lambda m: setattr(self, 'goal', m), 10)
        self.create_subscription(Float64, '/puck/speed', lambda m: setattr(self, 'puck_speed', m.data), 10)
        self.cmd = self.create_publisher(Twist, f'/{self.name}/cmd_vel', 10)
        self.carry_pub = self.create_publisher(Bool, f'/{self.name}/carrying', 10)
        self.timer = self.create_timer(0.033, self.loop)
        self.get_logger().info('Second robot ready.')

    def drive_to(self, gx, gy):
        x, y, th = self.pose.x, self.pose.y, self.pose.theta
        pxl = x + L * math.cos(th)
        pyl = y + L * math.sin(th)
        ex, ey = gx - pxl, gy - pyl
        dist = math.hypot(ex, ey)
        v = KV * (ex * math.cos(th) + ey * math.sin(th))
        w = KV * (-ex * math.sin(th) + ey * math.cos(th)) / L
        return max(-VMAX, min(VMAX, v)), max(-WMAX, min(WMAX, w)), dist

    def rotate_to(self, target):
        err = wrap(target - self.pose.theta)
        return max(-WMAX, min(WMAX, KW * err)), abs(err)

    def wait_heading(self):
        """Heading whose x-axis is perpendicular to the goal turtle's x-axis,
        choosing the side that faces toward the goal."""
        tg = self.goal.theta
        c1 = wrap(tg + math.pi / 2.0)
        c2 = wrap(tg - math.pi / 2.0)
        to_goal = math.atan2(self.goal.y - WAIT_POINT[1], self.goal.x - WAIT_POINT[0])
        return c1 if abs(wrap(c1 - to_goal)) < abs(wrap(c2 - to_goal)) else c2

    def puck_in_region(self):
        return math.hypot(self.puck.x - REGION_CENTER[0],
                          self.puck.y - REGION_CENTER[1]) < REGION_RADIUS

    def puck_in_net(self):
        x0, x1, y0, y1 = NET_BOX
        return x0 <= self.puck.x <= x1 and y0 <= self.puck.y <= y1

    def loop(self):
        if self.pose is None or self.puck is None or self.goal is None:
            return
        tw = Twist()

        if self.state == 'GO_WAIT':
            v, w, dist = self.drive_to(*WAIT_POINT)
            tw.linear.x, tw.angular.z = v, w
            if dist < POS_TOL:
                self.state = 'WAIT'
                self.get_logger().info('At waiting spot — aligning to the goal frame, waiting for the puck.')
        elif self.state == 'WAIT':
            w, _ = self.rotate_to(self.wait_heading())
            tw.angular.z = w
            if self.puck_in_region() and self.puck_speed < 0.05:
                self.state = 'NAVIGATE'
                self.get_logger().info('Puck received and settled — going to drag it into the goal.')
        elif self.state in ('NAVIGATE', 'ORIENT_PERP', 'AIM', 'ENGAGE', 'PUSH'):
            px, py = self.puck.x, self.puck.y
            goal_xy = (self.goal.x, self.goal.y)
            push_dir = math.atan2(goal_xy[1] - py, goal_xy[0] - px)
            standoff = (px - math.cos(push_dir) * STANDOFF, py - math.sin(push_dir) * STANDOFF)
            perp = wrap(push_dir + math.pi / 2.0)
            ee_x = self.pose.x + L * math.cos(self.pose.theta)
            ee_y = self.pose.y + L * math.sin(self.pose.theta)
            ee_to_puck = math.hypot(ee_x - px, ee_y - py)

            if self.state == 'NAVIGATE':
                v, w, dist = self.drive_to(*standoff)
                tw.linear.x, tw.angular.z = v, w
                if dist < POS_TOL:
                    self.state = 'ORIENT_PERP'
                    self.get_logger().info('Behind the puck — orienting onto the perpendicular line.')
            elif self.state == 'ORIENT_PERP':
                w, aerr = self.rotate_to(perp)
                tw.angular.z = w
                if aerr < ANG_TOL:
                    self.state = 'AIM'
                    self.get_logger().info('Pivoting to face the goal.')
            elif self.state == 'AIM':
                w, aerr = self.rotate_to(push_dir)
                tw.angular.z = w
                if aerr < ANG_TOL:
                    self.state = 'ENGAGE'
                    self.get_logger().info('Aimed — moving in to make contact with the puck.')
            elif self.state == 'ENGAGE':
                ang_to_puck = math.atan2(py - self.pose.y, px - self.pose.x)
                tw.linear.x = ENGAGE_SPEED
                tw.angular.z = max(-2.0, min(2.0, KW * wrap(ang_to_puck - self.pose.theta)))
                if ee_to_puck < CAPTURE_DETECT:
                    self.state = 'PUSH'
                    self.get_logger().info('Contacted the puck — dragging it into the goal.')
            else:  # PUSH
                v, w, _ = self.drive_to(*goal_xy)
                tw.linear.x, tw.angular.z = v, w
                if self.puck_in_net():
                    self.state = 'RELEASE'
                    self.release_start = (self.pose.x, self.pose.y)
                    self.get_logger().info('Puck is in the net — releasing.')
        elif self.state == 'RELEASE':
            tw.linear.x = -0.6
            if math.hypot(self.pose.x - self.release_start[0],
                          self.pose.y - self.release_start[1]) > RELEASE_BACK:
                self.state = 'HOLD'
                self.get_logger().info('Backed off. Play complete.')
        # HOLD: tw stays zero

        self.cmd.publish(tw)
        b = Bool()
        b.data = self.state in ('ENGAGE', 'PUSH')
        self.carry_pub.publish(b)


def main(args=None):
    rclpy.init(args=args)
    node = SecondRobot()
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
