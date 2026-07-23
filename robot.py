import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped

class Robot(Node):
    def __init__(self, robot_id, pass_to_robot, mock_mode=False, orient_to_stick=False):
        super().__init__(f'robot_{robot_id}_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.pass_to_robot = pass_to_robot
        self.mock_mode = mock_mode
        self.orient_to_stick = orient_to_stick

        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None

        self.current_target_pose = None
        self.rotation_phase = False
        self.state_start_time = None

        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('kp_v', 0.6)
        self.declare_parameter('kp_w', 0.5)
        # TODO: Update to be the distance from Horizontal the robot's sensor to the manipulator EE
        self.declare_parameter('l', 0.15)
        # TODO: Determine a good tolerance value
        self.declare_parameter('tolerance', 0.15)
        # Perpendicular shift of the approach goal point, relative to the target's facing
        # direction (target_theta). Positive = robot's right of target_theta, negative = left.
        # Tune this in the lab to compensate for the hockey stick sitting off-center on its base.
        self.declare_parameter('sideway_offset', 0.0)
        self.declare_parameter('start_sequence', 1)

        self.current_sequence = self.get_parameter('start_sequence').value

        self.L_inv = np.array([[1, 0], [0, 1/self.get_parameter('l').value]])
        time_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(time_period, self.control_loop)
        self.action_group = ReentrantCallbackGroup()
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        if self.mock_mode:
            self.create_subscription(PoseStamped, '/mock/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/mock/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)
            self.create_subscription(PoseStamped, '/mock/vrpn_mocap/puck_1/pose', self.puck_pos_callback, qos)
        else:
            self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)
            self.create_subscription(PoseStamped, '/vrpn_mocap/puck_1/pose', self.puck_pos_callback, qos)
        
        self.pub_cmd_vel = self.create_publisher(Twist, f'{self.robot_name}/cmd_vel', 10)
        self.get_logger().info(f'Robot node initialized at sequence state: {self.current_sequence}')

    def get_rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def hockey_stick_pos_callback(self, msg):
        self.hockey_stick_pose = msg.pose

    def robot_pos_callback(self, msg):
        self.robot_pose = msg.pose

    def puck_pos_callback(self, msg):
        self.puck_pose = msg.pose

    def control_loop(self):
        if self.robot_pose is None:
            self.get_logger().warn("Waiting for robot pose...", throttle_duration_sec=2.0)
            return
        
        if self.current_sequence in [1, 4]:
            self.current_target_pose = self.hockey_stick_pose if self.current_sequence == 1 else self.puck_pose
            
            if self.current_target_pose is None:
                self.get_logger().warn(f"Sequence {self.current_sequence}: Awaiting target data...", throttle_duration_sec=2.0)
                return

            cmd = Twist()
            v, w = self.nid_to_move_robot()

            if v == 0.0 and w == 0.0:
                self.pub_cmd_vel.publish(cmd)
                self.get_logger().info(f"Sequence {self.current_sequence} completed!")
                self.current_sequence += 1
                self.rotation_phase = False
                return
                
            cmd.linear.x = v
            cmd.angular.z = w
            self.get_logger().info(f"Sequence {self.current_sequence}: v={v:.3f}, w={w:.3f}")
            self.pub_cmd_vel.publish(cmd)

        elif self.current_sequence == 2:
            # Force the robot to stand completely still
            self.pub_cmd_vel.publish(Twist())

            # Capture the start time when we first enter state 2
            if self.state_start_time is None:
                self.state_start_time = self.get_clock().now()
                self.get_logger().info("Gripper Operation triggered! Sitting idle for 5 seconds...")
                self.pickup_hockey_stick()

            # Calculate how much time has passed
            elapsed_time = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            
            if elapsed_time >= 5.0:
                self.get_logger().info("5 seconds elapsed. Advancing to Sequence 3 (Move Backwards).")
                self.current_sequence += 1
                self.state_start_time = None  # Clear timer tracking for the next state     
            
        elif self.current_sequence == 3:
            # After the robot picks up the stick, we need to move it backwards a bit and rotate
            cmd = Twist()
            if self.state_start_time is None:
                self.state_start_time = self.get_clock().now()
                self.get_logger().info("Sequence 3: Moving backwards and rotating for 5 seconds...")
            elapsed_time = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed_time < 5.0:
                cmd.linear.x = -0.1  # Move backwards
                self.get_logger().info(f"Sequence 3: Moving backwards. Elapsed time: {elapsed_time:.2f}s")
                self.pub_cmd_vel.publish(cmd)
            else:
                self.get_logger().info("Sequence 3 completed. Advancing to Sequence 4.")
                self.current_sequence += 1
                self.state_start_time = None
        elif self.current_sequence == 5:
            self.release_puck()
            self.current_sequence += 1
            
        else:
            self.get_logger().info("All sequences completed. Robot is now idle.")
            self.pub_cmd_vel.publish(Twist())

    def nid_to_move_robot(self):
        l = self.get_parameter('l').value
        Kp_v = self.get_parameter('kp_v').value
        Kp_w = self.get_parameter('kp_w').value
        
        x = self.robot_pose.position.x
        y = self.robot_pose.position.y
        theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)
        
        p_xg = self.current_target_pose.position.x
        p_yg = self.current_target_pose.position.y
        target_theta = self.get_yaw_from_quaternion(self.current_target_pose.orientation)

        target_theta = np.arctan2(np.sin(target_theta), np.cos(target_theta))

        # Shift the goal sideways off the target center so the arm EE lines up with the
        # actual stick, which sits off-center on its base (positive = target's right side).
        sideway_offset = self.get_parameter('sideway_offset').value
        p_xg += sideway_offset * math.sin(target_theta)
        p_yg -= sideway_offset * math.cos(target_theta)

        p_xl = x + l * math.cos(theta)
        p_yl = y + l * math.sin(theta)

        distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
        angle_error = target_theta - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        v, w = 0.0, 0.0
        if self.rotation_phase or distance_to_target <= self.get_parameter('tolerance').value:
            if self.current_sequence == 1 and not self.orient_to_stick:
                v, w = 0.0, 0.0
            else:
                self.rotation_phase = True
                if self.current_sequence == 1:
                    flipped_target_theta = np.arctan2(np.sin(target_theta + np.pi), np.cos(target_theta + np.pi))
                    angle_error = np.arctan2(np.sin(flipped_target_theta - theta), np.cos(flipped_target_theta - theta))
                if abs(angle_error) > 0.02:
                    v = 0.0
                    w = Kp_w * angle_error
                else:
                    v, w = 0.0, 0.0
        else:
            e_x = p_xg - p_xl
            e_y = p_yg - p_yl
            p_dot_x = Kp_v * e_x
            p_dot_y = Kp_v * e_y

            control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
            v, w = control_inputs[0, 0], control_inputs[1, 0]
        self.get_logger().info(f"Distance to Target: {distance_to_target:.3f}, Angle Error: {angle_error:.3f}")
        return float(v), float(w)
    
    def pickup_hockey_stick(self):
        self.get_logger().info("Gripper Operation running to pick up the stick...")  

    def release_puck(self):
        dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the goal"
        self.get_logger().info(f"Releasing / Shooting the puck to {dest}...")

def main(args=None):
    parser = argparse.ArgumentParser(description='Move Robot Node')
    parser.add_argument('--robot_id', type=int, required=True, help='ID of the robot to control')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of ally robot to pass to (0 for goal)')
    parser.add_argument('--mock_mode', action='store_true', help='Enable mock mode for testing without real VRPN data')
    parser.add_argument('--orient_to_stick', action='store_true', help='Enable terminal angle orientation alignment for the hockey stick')

    args, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    
    node = Robot(robot_id=args.robot_id, pass_to_robot=args.pass_to_robot, mock_mode=args.mock_mode, orient_to_stick=args.orient_to_stick)
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