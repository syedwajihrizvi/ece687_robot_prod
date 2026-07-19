import argparse
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from robomaster_msgs.action import GripperControl
from geometry_msgs.msg import PoseStamped, Twist

# TODO: Add initial manupilator positiion

class MoveRobotNode(Node):
    def __init__(self, robot_id):
        super().__init__('move_robot_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.is_gripper_open = False
        self.target_pos_reached = False
        # Declare Parameters
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('kw', 0.2)
        self.declare_parameter('l', 0.10)
        self.declare_parameter('tolerance', 0.4)

        timer_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(timer_period, self.control_loop)
        self.action_group = ReentrantCallbackGroup()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Need subscription to the robot pose wrt to the world frame
        # Need subscription of the hockey stick rectangle
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, qos)
        # Subscription to the robot pose wrt to the world frame
        self.create_subscription(PoseStamped, f'/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)  
        # TODO: Subscribe to the gripper state as well
         # Publishers
        self.pub_cmd_vel = self.create_publisher(Twist, f'{self.robot_name}/cmd_vel',10)
        self.gripper_client = ActionClient(self, GripperControl, f'{self.robot_name}/gripper', callback_group=self.action_group)
        self.get_logger().info("Waiting for client")
        # self.gripper_client.wait_for_server()
        self.get_logger().info("Waited for client successfully")
        self.get_logger().info(f'MoveRobotNode initialized for robot ID: {robot_id} with control frequency: {self.get_parameter("control_frequency").value} Hz @ {self.robot_name}')

    def operate_on_gripper(self, open):
        """
        Open or Close the gripper
        """
        goal_msg = GripperControl.Goal()
        goal_msg.target_state = 1 if open else 2
        goal_msg.power = 0.5
        future = self.gripper_client.send_goal_async(goal_msg)
        future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected')
            return

        self.get_logger().info('Gripper goal accepted')
        goal_handle.get_result_async().add_done_callback(self.gripper_result_callback)

    def gripper_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Gripper Result: {result}")

    def control_loop(self):
        # Implement the control logic here
        # If not at the target position and gripper not open, open the gripper
        if not self.target_pos_reached and not self.is_gripper_open:
            self.get_logger().info('Opening gripper...')
            self.operate_on_gripper(open=True)
            self.is_gripper_open = True
            return
        elif self.target_pos_reached and self.is_gripper_open:
            self.get_logger().info('Closing gripper...')
            self.operate_on_gripper(open=False)
            self.is_gripper_open = False
            return

        msg = Twist()
        if self.hockey_stick_pose is not None and self.robot_pose is not None:
            self.get_logger().info("Running if block")
            x = self.robot_pose.pose.position.x
            y = self.robot_pose.pose.position.y
            theta = self.get_yaw_from_quaternion(self.robot_pose.pose.orientation)

            goal_x = self.hockey_stick_pose.pose.position.x
            goal_y = self.hockey_stick_pose.pose.position.y
            l = self.get_parameter('l').value
            kp = self.get_parameter('kp').value
            kw = self.get_parameter('kw').value
            tolerance = self.get_parameter('tolerance').value

            p_x = x + l*math.cos(theta)
            p_y = y + l*math.sin(theta)

            error_x = goal_x - p_x
            error_y = goal_y - p_y
            distance = math.sqrt(error_x**2 + error_y**2)
            self.get_logger().info(f"Distance: {distance}")
            if distance < tolerance:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.pub_cmd_vel.publish(msg)
                self.get_logger().info('Robot has reached the goal position.')
                self.target_pos_reached = True
                return
            p_dot_x = kp * error_x
            p_dot_y = kw * error_y

            v = p_dot_x * math.cos(theta) + p_dot_y * math.sin(theta)
            omega = (-p_dot_x * math.sin(theta) + p_dot_y * math.cos(theta)) / l
            msg.linear.x = v
            msg.angular.z = omega
            self.get_logger().info(f"Publishing Robot Msg: {msg}")
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
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()