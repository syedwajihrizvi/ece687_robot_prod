#!/usr/bin/env python3
"""
FIRST robot controller  --  DRAG model.

The puck can't be shot; it must be pushed. So the First robot lines up BEHIND the
puck along the pass line (arriving on the perpendicular "blue" line, then pivoting
to aim), makes contact, and DRAGS the puck into the receiving region, then lets
go and backs off.

States: NAVIGATE -> ORIENT_PERP -> AIM -> ENGAGE -> PUSH -> RELEASE -> HOLD
The /robot1/carrying flag is True during ENGAGE + PUSH; the world captures the
puck onto the stick tip while it is True.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from turtlesim.msg import Pose

DEST = (5.5, 6.5)              # where to drag the puck (region center; match world.py)

L        = 0.4                 # stick-tip look-ahead (must match world.py L_EE)
KV       = 0.8
KW       = 2.5
VMAX     = 1.5
WMAX     = 3.0
POS_TOL  = 0.20
ANG_TOL  = 0.03
STANDOFF = 1.0                 # how far behind the puck to line up
ENGAGE_SPEED = 0.7
CAPTURE_DETECT = 0.15          # stick tip this close to puck => world has captured it
REACH_TOL = 0.30               # puck within this of DEST => delivered
RELEASE_BACK = 0.6             # how far to reverse after releasing


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class FirstRobot(Node):
    def __init__(self):
        super().__init__('first_robot')
        self.name = 'robot1'
        self.pose = None
        self.puck = None
        self.state = 'NAVIGATE'
        self.release_start = None
        self.create_subscription(Pose, f'/{self.name}/pose', lambda m: setattr(self, 'pose', m), 10)
        self.create_subscription(Pose, '/puck/pose', lambda m: setattr(self, 'puck', m), 10)
        self.cmd = self.create_publisher(Twist, f'/{self.name}/cmd_vel', 10)
        self.carry_pub = self.create_publisher(Bool, f'/{self.name}/carrying', 10)
        self.timer = self.create_timer(0.033, self.loop)
        self.get_logger().info('First robot ready.')

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

    def loop(self):
        if self.pose is None or self.puck is None:
            return
        px, py = self.puck.x, self.puck.y
        push_dir = math.atan2(DEST[1] - py, DEST[0] - px)
        standoff = (px - math.cos(push_dir) * STANDOFF, py - math.sin(push_dir) * STANDOFF)
        perp = wrap(push_dir + math.pi / 2.0)
        ee_x = self.pose.x + L * math.cos(self.pose.theta)
        ee_y = self.pose.y + L * math.sin(self.pose.theta)
        ee_to_puck = math.hypot(ee_x - px, ee_y - py)
        tw = Twist()

        if self.state == 'NAVIGATE':
            v, w, dist = self.drive_to(*standoff)
            tw.linear.x, tw.angular.z = v, w
            if dist < POS_TOL:
                self.state = 'ORIENT_PERP'
                self.get_logger().info('At standoff — orienting onto the perpendicular (blue) line.')
        elif self.state == 'ORIENT_PERP':
            w, aerr = self.rotate_to(perp)
            tw.angular.z = w
            if aerr < ANG_TOL:
                self.state = 'AIM'
                self.get_logger().info('On the blue line — pivoting to face the region.')
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
                self.get_logger().info('Contacted the puck — dragging it to the region.')
        elif self.state == 'PUSH':
            v, w, _ = self.drive_to(*DEST)          # drive the stick tip (with the puck) to DEST
            tw.linear.x, tw.angular.z = v, w
            if math.hypot(px - DEST[0], py - DEST[1]) < REACH_TOL:
                self.state = 'RELEASE'
                self.release_start = (self.pose.x, self.pose.y)
                self.get_logger().info('Puck delivered to the region — releasing.')
        elif self.state == 'RELEASE':
            tw.linear.x = -0.6
            if math.hypot(self.pose.x - self.release_start[0],
                          self.pose.y - self.release_start[1]) > RELEASE_BACK:
                self.state = 'HOLD'
                self.get_logger().info('Backed off. Handing over to the Second robot.')
        # HOLD: tw stays zero

        self.cmd.publish(tw)
        b = Bool()
        b.data = self.state in ('ENGAGE', 'PUSH')   # carrying only while engaging/pushing
        self.carry_pub.publish(b)


def main(args=None):
    rclpy.init(args=args)
    node = FirstRobot()
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
