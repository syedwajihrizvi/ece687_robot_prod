import argparse
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist


class MoveRobotNode(Node):
    def __init__(self, robot_id):
        super().__init__('move_robot_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.robot_pose = None
        self.hockey_stick_pose = None
        # Declare Parameters
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('kp', 1.5)
        self.declare_parameter('l', 0.20)
        self.declare_parameter('tolerance', 0.05)

        timer_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(timer_period, self.control_loop)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Need subscription to the robot pose wrt to the world frame
        # Need subscription of the hcokey stick rectangle
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, qos)
        # Subscription to the robot pose wrt to the world frame
        self.create_subscription(PoseStamped, f'/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)  
         # Publishers
        self.pub_cmd_vel = self.create_publisher(Twist, f'{self.robot_name}/cmd_vel', qos)

        self.get_logger().info(f'MoveRobotNode initialized for robot ID: {robot_id} with control frequency: {self.get_parameter("control_frequency").value} Hz @ {self.robot_name}')

    def control_loop(self):
        # Implement the control logic here
        msg = Twist()
        if self.hockey_stick_pose is not None and self.robot_pose is not None:
            x = self.robot_pose.pose.position.x
            y = self.robot_pose.pose.position.y
            theta = self.get_yaw_from_quaternion(self.robot_pose.pose.orientation)

            goal_x = self.hockey_stick_pose.pose.position.x
            goal_y = self.hockey_stick_pose.pose.position.y
            l = self.get_parameter('l').value
            kp = self.get_parameter('kp').value
            tolerance = self.get_parameter('tolerance').value

            p_x = x + l*math.cos(theta)
            p_y = y + l*math.sin(theta)

            error_x = goal_x - p_x
            error_y = goal_y - p_y
            distance = math.sqrt(error_x**2 + error_y**2)

            if distance < tolerance:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.pub_cmd_vel.publish(msg)
                self.get_logger().info('Robot has reached the goal position.')
                return
            p_dot_x = kp * error_x
            p_dot_y = kp * error_y

            v = p_dot_x * math.cos(theta) + p_dot_y * math.sin(theta)
            omega = (-p_dot_x * math.sin(theta) + p_dot_y * math.cos(theta)) / l
            msg.linear.x = v
            msg.angular.z = omega
            self.pub_cmd_vel.publish(msg)
        else:
            self.get_logger().warn('Waiting for both robot and hockey stick poses to be available.')
    
        
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2*(q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2*(q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
            
    def hockey_stick_pos_callback(self, msg):
        # Handle the hockey stick position message
        self.hockey_stick_pose = msg

    def robot_pos_callback(self, msg):
        # Handle the robot position message
        self.robot_pose = msg

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_id', type=int, required=True, help='ID of the robot to control')
    args, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    node = MoveRobotNode(robot_id=args.robot_id)
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