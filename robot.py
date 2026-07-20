import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from robomaster_msgs.action import GripperControl
from geometry_msgs.msg import Twist, PoseStamped

class Robot(Node):
    def __init__(self, 
                 robot_id, 
                 pass_to_robot, 
                 hockey_stick_id=1, 
                 puck_color='blue', 
                 mock_mode=False, 
                 orient_to_stick=False, 
                 l_default=0.15, 
                 tolerance_default=0.15, 
                 sideways_offset=0.0, 
                 vertical_offset=0.0, 
                 standoff_distance=2.5):
        super().__init__(f'robot_{robot_id}_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.gripper_action = f'/robot{robot_id}/gripper'
        self.pass_to_robot = pass_to_robot
        self.hockey_stick_id = hockey_stick_id
        self.puck_color = puck_color
        self.mock_mode = mock_mode
        self.orient_to_stick = orient_to_stick
        self.gripper_action_running = False
        self.gripper_action_accepted = False
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None

        self.current_target_pose = None
        self.rotation_phase = False
        self.state_start_time = None

        # Staging flags for Sequence 1 and Sequence 4
        self.seq1_stage = 0 
        self.seq1_completed = False 
        self.seq4_stage = 0
        self.seq4_completed = False

        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('kp_v', 0.6)
        self.declare_parameter('kp_w', 1.2)
        self.declare_parameter('l', l_default)
        self.declare_parameter('tolerance', tolerance_default)
        self.declare_parameter('standoff_distance', standoff_distance)
        self.declare_parameter('start_sequence', 0) 
        self.declare_parameter('sideways_offset', sideways_offset)
        self.declare_parameter('vertical_offset', vertical_offset)
        self.current_sequence = self.get_parameter('start_sequence').value

        self.L_inv = np.array([[1, 0], [0, 1/self.get_parameter('l').value]])
        self._action_group = ReentrantCallbackGroup()
        self.gripper_action_client = None
        if not self.mock_mode:
            self.gripper_action_client = ActionClient(
                self,
                GripperControl,
                self.gripper_action,
                callback_group=self._action_group
            )
            self.get_logger().info("Waiting for gripper action server...")
            self.gripper_action_client.wait_for_server()
            self.get_logger().info("Gripper action server is available.")
        time_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(time_period, self.control_loop)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        if self.mock_mode:
            self.create_subscription(PoseStamped, f'/mock/vrpn_mocap/hockey_sticks_{self.hockey_stick_id}/pose', self.hockey_stick_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/mock/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)
            self.create_subscription(PoseStamped, '/mock/vrpn_mocap/puck_1/pose', self.puck_pos_callback, qos)
        else:
            self.create_subscription(PoseStamped, f'/vrpn_mocap/hockey_sticks_{self.hockey_stick_id}/pose', self.hockey_stick_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/vrpn_mocap/hockey_puck_{self.puck_color}/pose', self.puck_pos_callback, qos)
            
        self.pub_cmd_vel = self.create_publisher(Twist, f'{self.robot_name}/cmd_vel', 10)
        self.get_logger().info(f'Robot node initialized at sequence state: {self.current_sequence} with stick ID: {self.hockey_stick_id} & puck color: {self.puck_color}')

    def get_rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
                        
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

            if v == 0.0 and w == 0.0 and (
                (self.current_sequence == 1 and self.seq1_completed) or 
                (self.current_sequence == 4 and self.seq4_completed)
            ):
                self.pub_cmd_vel.publish(cmd)
                self.get_logger().info(f"Sequence {self.current_sequence} completed!")
                self.current_sequence += 1
                self.rotation_phase = False
                self.state_start_time = None 
                return

            cmd.linear.x = v
            cmd.angular.z = w
            self.get_logger().info(f"Sequence {self.current_sequence}: v={v:.3f}, w={w:.3f}", throttle_duration_sec=1.0)
            self.pub_cmd_vel.publish(cmd)

        elif self.current_sequence == 0:
            now = self.get_clock().now()
            if self.state_start_time is None:
                elapsed_retry_time = 3.0 
            else:
                elapsed_retry_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_retry_time >= 3.0 and not self.gripper_action_running:
                self.get_logger().info("Sequence 0: Dispatching gripper OPEN request...")
                self.state_start_time = now 
                self.gripper_action_running = True
                self.gripper_controller(open=True)
                
        elif self.current_sequence == 2:
            self.pub_cmd_vel.publish(Twist()) 
            now = self.get_clock().now()
            if self.state_start_time is None:
                elapsed_retry_time = 3.0 
            else:
                elapsed_retry_time = (now - self.state_start_time).nanoseconds / 1e9

            if elapsed_retry_time >= 3.0 and not self.gripper_action_running:
                self.get_logger().info(f"Sequence 2: Dispatching gripper CLOSE request... (Running: {self.gripper_action_running})")
                self.state_start_time = now 
                self.gripper_action_running = True
                self.gripper_controller(open=False) 

        elif self.current_sequence == 3:
            cmd = Twist()
            if self.state_start_time is None:
                self.state_start_time = self.get_clock().now()
                self.get_logger().info("Sequence 3: Moving backwards and rotating for 5 seconds...")
            elapsed_time = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed_time < 5.0:
                cmd.linear.x = -0.15
                self.get_logger().info(f"Sequence 3: Moving backwards. Elapsed time: {elapsed_time:.2f}s", throttle_duration_sec=1.0)
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
        tolerance = self.get_parameter('tolerance').value
        Kp_v = self.get_parameter('kp_v').value
        Kp_w = self.get_parameter('kp_w').value
        standoff_dist = self.get_parameter('standoff_distance').value

        x = self.robot_pose.position.x
        y = self.robot_pose.position.y
        theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)

        p_xg = self.current_target_pose.position.x
        p_yg = self.current_target_pose.position.y
        target_theta = self.get_yaw_from_quaternion(self.current_target_pose.orientation)
        target_theta = np.arctan2(np.sin(target_theta), np.cos(target_theta))

        p_xl = x + l * math.cos(theta)
        p_yl = y + l * math.sin(theta)

        v, w = 0.0, 0.0

        # --- MULTI-STAGE TRAJECTORY CONTROL FOR SEQUENCE 1 ---
        if self.current_sequence == 1:
            # Apply static (x, y) offset translation directly to the base target point
            target_x = p_xg + self.get_parameter('vertical_offset').value
            target_y = p_yg + self.get_parameter('sideways_offset').value

            # Universal Translation: Extends offset target point along the stick's vector angle
            standoff_x = target_x + standoff_dist * math.cos(target_theta)
            standoff_y = target_y + standoff_dist * math.sin(target_theta)

            if self.seq1_stage == 0:
                bearing_to_standoff = np.arctan2(standoff_y - p_yl, standoff_x - p_xl)
                angle_error = np.arctan2(np.sin(bearing_to_standoff - theta), np.cos(bearing_to_standoff - theta))
                
                if abs(angle_error) > 0.03:
                    w = Kp_w * angle_error
                    return 0.0, float(w)
                else:
                    self.seq1_stage = 1
                    self.get_logger().info("[Seq 1 - Stage 0] Heading aligned to standoff vector. Advancing to Stage 1.")

            if self.seq1_stage == 1:
                dist = np.sqrt((standoff_x - p_xl)**2 + (standoff_y - p_yl)**2)
                if dist <= tolerance:
                    self.seq1_stage = 2
                    self.get_logger().info("[Seq 1 - Stage 1] Arrived at standoff location. Advancing to Stage 2 (Orientation).")
                else:
                    e_x, e_y = standoff_x - p_xl, standoff_y - p_yl
                    p_dot_x, p_dot_y = Kp_v * e_x, Kp_v * e_y
                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

            if self.seq1_stage == 2:
                flipped_target_theta = np.arctan2(np.sin(target_theta + np.pi), np.cos(target_theta + np.pi))
                angle_error = np.arctan2(np.sin(flipped_target_theta - theta), np.cos(flipped_target_theta - theta))
                
                if abs(angle_error) > 0.02:
                    w = Kp_w * angle_error
                    return 0.0, float(w)
                else:
                    self.seq1_stage = 3
                    self.get_logger().info("[Seq 1 - Stage 2] Alignment complete! Advancing to Stage 3 (Final Move).")

            if self.seq1_stage == 3:
                # Target the translated offset point rather than the raw stick coordinate
                dist = np.sqrt((target_x - p_xl)**2 + (target_y - p_yl)**2)
                if dist <= tolerance:
                    self.seq1_completed = True 
                    return 0.0, 0.0  
                else:
                    e_x, e_y = target_x - p_xl, target_y - p_yl
                    p_dot_x, p_dot_y = Kp_v * e_x, Kp_v * e_y
                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

        # --- MULTI-STAGE TRAJECTORY CONTROL FOR SEQUENCE 4 ---
        elif self.current_sequence == 4:
            if self.seq4_stage == 0:
                distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
                if distance_to_target <= tolerance:
                    self.seq4_stage = 1
                    self.get_logger().info("[Seq 4 - Stage 0] Arrived at puck location. Advancing to Stage 1 (Orientation Alignment).")
                    return 0.0, 0.0
                else:
                    e_x, e_y = p_xg - p_xl, p_yg - p_yl
                    p_dot_x, p_dot_y = Kp_v * e_x, Kp_v * e_y
                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])
            
            if self.seq4_stage == 1:
                angle_error = np.arctan2(np.sin(target_theta - theta), np.cos(target_theta - theta))
                if abs(angle_error) > 0.02:
                    w = Kp_w * angle_error
                    return 0.0, float(w)
                else:
                    self.seq4_completed = True
                    self.get_logger().info("[Seq 4 - Stage 1] Puck orientation alignment complete!")
                    return 0.0, 0.0

        return 0.0, 0.0

    def gripper_controller(self, open=False):
        if self.mock_mode:
            self.get_logger().info(f"Mock mode active: {'Opening' if open else 'Closing'} gripper simulated.")
            self.gripper_action_running = False
            self.current_sequence += 1
            return
        self.get_logger().info("Gripper Operation running to pick up the stick...") 
        goal = GripperControl.Goal()
        self.power = 1
        goal.target_state = 1 if open else 2
        future = self.gripper_action_client.send_goal_async(goal)
        self.get_logger().info("Goal has been sent")
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        self.get_logger().info(f"Goal Handle Result: {goal_handle}")
        if not goal_handle.accepted:
            self.get_logger().warn("Goal Client rejected by server! Resetting flag for retry...")
            self.gripper_action_running = False 
            return
        self.get_logger().info("Goal accepted by server. Awaiting execution result...")
        goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, future):
        try:
            result = future.result()
            self.current_sequence += 1
            self.get_logger().info(f'Gripper operation succeeded. Moving to Sequence {self.current_sequence}')
        except Exception as e:
            self.get_logger().error(f'Gripper execution tracking faulted: {e}. Will retry...')
        finally:
            self.gripper_action_running = False
            self.state_start_time = None 

    def release_puck(self):
        dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the goal"
        self.get_logger().info(f"Releasing / Shooting the puck to {dest}...")

def main(args=None):
    parser = argparse.ArgumentParser(description='Move Robot Node')
    parser.add_argument('--robot_id', type=int, required=True, help='ID of the robot to control')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of ally robot to pass to (0 for goal)')
    parser.add_argument('--hockey_stick_id', type=int, default=1, help='ID tag integer for the hockey stick VRPN tracking topic')
    parser.add_argument('--puck_color', type=str, default='blue', help='Color tag string for the puck VRPN tracking topic (e.g., blue, red, green)')
    parser.add_argument('--mock_mode', action='store_true', help='Enable mock mode for testing without real VRPN data')
    parser.add_argument('--orient_to_stick', action='store_true', help='Enable terminal angle orientation alignment for the hockey stick')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help="Sideways offset for hockeystick pose")
    parser.add_argument('--vertical_offset', type=float, default=0.0, help="Vertical offset for hockeystick pose")
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along the vector field line')
    parser.add_argument('--l', type=float, default=0.15, help='Look-ahead center to manipulator end-effector displacement distance')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Target proximity threshold radius for spatial sequence handoffs')

    args, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    node = Robot(
        robot_id=args.robot_id, 
        pass_to_robot=args.pass_to_robot, 
        hockey_stick_id=args.hockey_stick_id,
        puck_color=args.puck_color,
        mock_mode=args.mock_mode, 
        orient_to_stick=args.orient_to_stick,
        l_default=args.l,
        tolerance_default=args.tolerance,
        sideways_offset=args.sideways_offset,
        vertical_offset=args.vertical_offset,
        standoff_distance=args.standoff_distance
    )
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