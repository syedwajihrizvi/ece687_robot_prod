import argparse
import math
import numpy as np
import rclpy
from enum import Enum
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from robomaster_msgs.action import GripperControl, MoveArm
from geometry_msgs.msg import Twist, PoseStamped, Vector3
from scipy.optimize import minimize

"""
Following Sequences
0: Open Gripper
1: Move Arm to Origin Position (0.0, 0.0)
2: Move Arm to Reference Position (0.15, 0.15)
3: Move to Hockey Stick
4: Close Gripper
5: Lift Stick in the Air to remove from platform
6: Move Backwards and Rotate
7: Move to Puck
8: Bring Stick to the Ground
9: Release Puck
"""
class Sequence(Enum):
    OPEN_GRIPPER = 0
    MOVE_EE_TO_ORIGIN = 1
    MOVE_EE_TO_REF_POS = 2
    MOVE_TO_STICK = 3
    CLOSE_GRIPPER = 4
    LIFT_STICK = 5
    MOVE_BACK_ROTATE = 6
    MOVE_TO_PUCK = 7
    LOWER_STICK = 8
    RELEASE_PUCK = 9

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
                 standoff_distance=2.5,
                 r_safety=0.35):
        super().__init__(f'robot_{robot_id}_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.gripper_action = f'/robot{robot_id}/gripper'
        self.arm_action = f'/robot{robot_id}/arm'
        self.pass_to_robot = pass_to_robot
        self.hockey_stick_id = hockey_stick_id
        self.puck_color = puck_color
        self.mock_mode = mock_mode
        self.orient_to_stick = orient_to_stick
        
        # Action tracking flags
        self.gripper_action_running = False
        self.arm_action_running = False
        
        # Pose storage structures
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None
        self.obstacle_poses = {}

        # Optimization & Safety Parameters
        self.declare_parameter('r_safety', r_safety)
        self.declare_parameter('gamma_cbf', 1.5)
        self.declare_parameter('gamma_clf', 1.0)
        self.declare_parameter('clf_penalty', 1e3)

        self.current_target_pose = None
        self.rotation_phase = False
        self.state_start_time = None

        # Filter and Tangent memory variables
        self.chosen_tangent_sign = {}
        self.filtered_u_p = None

        # Sub-stages trackers
        self.seq1_stage = 0 
        self.seq1_completed = False 
        self.seq4_stage = 0
        self.seq4_completed = False

        # Controller tunings & parameters
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('kp_v', 1.2)
        self.declare_parameter('kp_w', 2.0)
        self.declare_parameter('v_max', 1.0)  # Maximum workspace velocity cap (m/s)
        self.declare_parameter('l', l_default)
        self.declare_parameter('tolerance', tolerance_default)
        self.declare_parameter('standoff_distance', standoff_distance)
        self.declare_parameter('start_sequence', 0) 
        self.declare_parameter('sideways_offset', sideways_offset)
        self.declare_parameter('vertical_offset', vertical_offset)
        
        self.current_sequence = Sequence(self.get_parameter('start_sequence').value)

        self.L_inv = np.array([[1, 0], [0, 1 / self.get_parameter('l').value]])
        self._action_group = ReentrantCallbackGroup()
        self.gripper_action_client = None
        self.arm_action_client = None

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
            
            self.arm_action_client = ActionClient(
                self,
                MoveArm,
                self.arm_action,
                callback_group=self._action_group
            )
            self.get_logger().info("Waiting for arm action server...")
            self.arm_action_client.wait_for_server()
            self.get_logger().info("Arm action server is available.")

        time_period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(time_period, self.control_loop)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Topic subscriptions
        if self.mock_mode:
            self.create_subscription(PoseStamped, f'/mock/vrpn_mocap/hockey_sticks_{self.hockey_stick_id}/pose', self.hockey_stick_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/mock/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)
            self.create_subscription(PoseStamped, '/mock/vrpn_mocap/puck_1/pose', self.puck_pos_callback, qos)
            for i in range(1, 11):
                if i == self.robot_id:
                    continue  # Skip subscribing to own robot's obstacle topic
                topic_name = f'/mock/vrpn_mocap/dji_robot_{i}/pose'
                key = f'obstacle_robot_{i}'
                self.create_subscription(PoseStamped, topic_name, self.obstacle_pos_callback(key), qos)
        else:
            self.create_subscription(PoseStamped, f'/vrpn_mocap/hockey_sticks_{self.hockey_stick_id}/pose', self.hockey_stick_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)
            self.create_subscription(PoseStamped, f'/vrpn_mocap/hockey_puck_{self.puck_color}/pose', self.puck_pos_callback, qos)
            for i in range(1, 11):
                if i == self.robot_id:
                    continue  # Skip subscribing to own robot's obstacle topic
                topic_name = f'/vrpn_mocap/dji_robot_{i}/pose'
                key = f'obstacle_robot_{i}'
                self.create_subscription(PoseStamped, topic_name, self.obstacle_pos_callback(key), qos)
            
        self.pub_cmd_vel = self.create_publisher(Twist, f'{self.robot_name}/cmd_vel', 10)
        self.pub_cmd_arm = self.create_publisher(Vector3, f'{self.robot_name}/cmd_arm', 10)
        self.get_logger().info(f'Robot node initialized at sequence state: {self.current_sequence.name} with stick ID: {self.hockey_stick_id} & puck color: {self.puck_color}')

    def advance_sequence(self):
        """Advances state machine to the next Sequence enum member and resets velocity filter memory."""
        next_val = self.current_sequence.value + 1
        if next_val in [s.value for s in Sequence]:
            self.current_sequence = Sequence(next_val)
        else:
            self.current_sequence = None
        self.filtered_u_p = None
        self.chosen_tangent_sign.clear()

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

    def obstacle_pos_callback(self, key):
        def callback(msg):
            self.obstacle_poses[key] = msg.pose
        return callback

    def get_valid_standoff_distance(self, target_x, target_y, target_theta, initial_standoff):
        """
        Checks if computed standoff position overlaps with any obstacle's safety radius.
        Dynamically increases standoff distance until it is completely clear of obstacles.
        """
        r_safety = self.get_parameter('r_safety').value
        current_standoff = initial_standoff
        step_increment = 0.1
        max_standoff = initial_standoff + 3.0

        adjusted = False
        blocking_obs_key = None

        while current_standoff <= max_standoff:
            st_x = target_x + current_standoff * math.cos(target_theta)
            st_y = target_y + current_standoff * math.sin(target_theta)

            collision_detected = False

            for obs_key, obs_pose in self.obstacle_poses.items():
                if obs_pose is None:
                    continue
                obs_x = obs_pose.position.x
                obs_y = obs_pose.position.y
                dist = math.sqrt((st_x - obs_x)**2 + (st_y - obs_y)**2)

                if dist <= (r_safety + 0.05):
                    collision_detected = True
                    blocking_obs_key = obs_key
                    break

            if collision_detected:
                adjusted = True
                current_standoff += step_increment
            else:
                if adjusted:
                    self.get_logger().warn(
                        f"[STANDOFF ADJUSTED] Standoff distance overlapped with {blocking_obs_key}! "
                        f"Increased from {initial_standoff:.2f}m to {current_standoff:.2f}m due to obstacle overlap.",
                        throttle_duration_sec=2.0
                    )
                return current_standoff, st_x, st_y

        return current_standoff, st_x, st_y

    def solve_clf_cbf_qp(self, p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, target_x, target_y):
        """
        QP Filter enforcing Control Lyapunov Functions (CLF) and Control Barrier Functions (CBF).
        """
        r_safety = self.get_parameter('r_safety').value
        gamma_cbf = self.get_parameter('gamma_cbf').value
        gamma_clf = self.get_parameter('gamma_clf').value
        clf_penalty = self.get_parameter('clf_penalty').value

        if self.filtered_u_p is None:
            self.filtered_u_p = np.array([p_dot_x_nom, p_dot_y_nom])

        u_nom = np.array([p_dot_x_nom, p_dot_y_nom])
        active_obstacle_keys = []
        
        for obs_key, obs_pose in self.obstacle_poses.items():
            if obs_pose is None:
                continue
            
            obs_p = np.array([obs_pose.position.x, obs_pose.position.y])
            p_rel = np.array([p_xl, p_yl]) - obs_p
            dist = np.linalg.norm(p_rel)

            if dist < (r_safety * 1.6) and dist > 1e-4:
                active_obstacle_keys.append(obs_key)
                normal = p_rel / dist
                base_tangent = np.array([-normal[1], normal[0]])
                
                if obs_key not in self.chosen_tangent_sign:
                    sign = 1.0 if np.dot(base_tangent, u_nom) >= 0 else -1.0
                    self.chosen_tangent_sign[obs_key] = sign
                
                tangent = self.chosen_tangent_sign[obs_key] * base_tangent
                influence_factor = max(0.0, (r_safety * 1.6 - dist) / (r_safety * 0.6))
                u_nom = u_nom + (1.5 * influence_factor) * tangent

        for k in list(self.chosen_tangent_sign.keys()):
            if k not in active_obstacle_keys:
                del self.chosen_tangent_sign[k]

        p_dot_x_nom, p_dot_y_nom = u_nom[0], u_nom[1]

        def objective(z):
            ux, uy, delta = z[0], z[1], z[2]
            u_diff = (ux - p_dot_x_nom)**2 + (uy - p_dot_y_nom)**2
            return 0.5 * u_diff + 0.5 * clf_penalty * (delta**2)

        def objective_jacobian(z):
            ux, uy, delta = z[0], z[1], z[2]
            return np.array([ux - p_dot_x_nom, uy - p_dot_y_nom, clf_penalty * delta])

        constraints = []

        # 1. CLF Constraint
        e_x = p_xl - target_x
        e_y = p_yl - target_y
        V = 0.5 * (e_x**2 + e_y**2)

        def clf_constraint(z):
            ux, uy, delta = z[0], z[1], z[2]
            return delta - (e_x * ux + e_y * uy + gamma_clf * V)

        constraints.append({'type': 'ineq', 'fun': clf_constraint})

        # 2. CBF Constraints
        for obs_key in active_obstacle_keys:
            obs_pose = self.obstacle_poses[obs_key]
            obs_x = obs_pose.position.x
            obs_y = obs_pose.position.y

            dist_sq = (p_xl - obs_x)**2 + (p_yl - obs_y)**2
            h = dist_sq - (r_safety**2)

            def cbf_constraint(z, ox=obs_x, oy=obs_y, h_val=h):
                ux, uy, _ = z[0], z[1], z[2]
                dh_dot = 2 * (p_xl - ox) * ux + 2 * (p_yl - oy) * uy
                return dh_dot + gamma_cbf * h_val

            constraints.append({'type': 'ineq', 'fun': cbf_constraint})

        bounds = [(None, None), (None, None), (0, None)]
        initial_guess = np.array([p_dot_x_nom, p_dot_y_nom, 0.0])

        res = minimize(
            objective,
            initial_guess,
            jac=objective_jacobian,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if res.success:
            raw_u = np.array([float(res.x[0]), float(res.x[1])])
        else:
            raw_u = np.array([p_dot_x_nom, p_dot_y_nom])

        alpha = 0.4
        self.filtered_u_p = alpha * raw_u + (1.0 - alpha) * self.filtered_u_p
        return float(self.filtered_u_p[0]), float(self.filtered_u_p[1])

    def control_loop(self):
        if self.robot_pose is None:
            self.get_logger().warn("Waiting for robot pose...", throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()

        # Sequence 0: Open Gripper Action
        if self.current_sequence == Sequence.OPEN_GRIPPER:
            if self.state_start_time is None:
                elapsed_retry_time = 3.0 
            else:
                elapsed_retry_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_retry_time >= 3.0 and not self.gripper_action_running:
                self.get_logger().info("Sequence 0: Dispatching gripper OPEN request...")
                self.state_start_time = now 
                self.gripper_action_running = True
                self.gripper_controller(open=True)

        # Sequence 1: Move Arm to Origin Action (0.0, 0.0)
        elif self.current_sequence == Sequence.MOVE_EE_TO_ORIGIN:
            if self.state_start_time is None:
                elapsed_retry_time = 3.0
            else:
                elapsed_retry_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_retry_time >= 3.0 and not self.arm_action_running:
                self.get_logger().info("Sequence 1: Dispatching arm move to origin request...")
                self.state_start_time = now 
                self.arm_action_running = True
                self.move_arm_using_action(x=0.0, z=0.0, relative=False)

        # Sequence 2: Move Arm to Ref Pos Action (0.15, 0.15)
        elif self.current_sequence == Sequence.MOVE_EE_TO_REF_POS:
            if self.state_start_time is None:
                elapsed_retry_time = 3.0
            else:
                elapsed_retry_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_retry_time >= 3.0 and not self.arm_action_running:
                self.get_logger().info("Sequence 2: Dispatching arm move to reference position request...")
                self.state_start_time = now 
                self.arm_action_running = True
                self.move_arm_using_action(x=0.15, z=0.15, relative=False)

        # Sequence 3 & 7: Spatial Tracking with CLF-CBF
        elif self.current_sequence in [Sequence.MOVE_TO_STICK, Sequence.MOVE_TO_PUCK]:
            self.current_target_pose = self.hockey_stick_pose if self.current_sequence == Sequence.MOVE_TO_STICK else self.puck_pose
            if self.current_target_pose is None:
                self.get_logger().warn(f"Sequence {self.current_sequence.name}: Awaiting target data...", throttle_duration_sec=2.0)
                return

            cmd = Twist()
            v, w = self.nid_to_move_robot()

            if v == 0.0 and w == 0.0 and (
                (self.current_sequence == Sequence.MOVE_TO_STICK and self.seq1_completed) or 
                (self.current_sequence == Sequence.MOVE_TO_PUCK and self.seq4_completed)
            ):
                self.pub_cmd_vel.publish(cmd)
                self.get_logger().info(f"Sequence {self.current_sequence.name} completed!")
                self.advance_sequence()
                self.rotation_phase = False
                self.state_start_time = None 
                return

            cmd.linear.x = v
            cmd.angular.z = w
            self.get_logger().info(f"Sequence {self.current_sequence.name}: v={v:.3f}, w={w:.3f}", throttle_duration_sec=1.0)
            self.pub_cmd_vel.publish(cmd)

        # Sequence 4: Close Gripper Action
        elif self.current_sequence == Sequence.CLOSE_GRIPPER:
            self.pub_cmd_vel.publish(Twist()) 
            if self.state_start_time is None:
                elapsed_retry_time = 3.0 
            else:
                elapsed_retry_time = (now - self.state_start_time).nanoseconds / 1e9

            if elapsed_retry_time >= 3.0 and not self.gripper_action_running:
                self.get_logger().info(f"Sequence 4: Dispatching gripper CLOSE request...")
                self.state_start_time = now 
                self.gripper_action_running = True
                self.gripper_controller(open=False) 

        # Sequence 5: Lift Stick
        elif self.current_sequence == Sequence.LIFT_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("Sequence 5: Dispatching arm LIFT command (waiting 2s)...")
                self.arm_controller(direction=1)

            elapsed_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_time >= 2.0:
                self.get_logger().info("Sequence 5: Arm lift complete! Advancing sequence.")
                self.state_start_time = None
                self.advance_sequence()
            else:
                self.get_logger().info(f"Sequence 5: Lifting stick... {elapsed_time:.1f}s", throttle_duration_sec=1.0)

        # Sequence 6: Move Backwards
        elif self.current_sequence == Sequence.MOVE_BACK_ROTATE:
            cmd = Twist()
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("Sequence 6: Executing reverse safety clearance step (3s)...")
            elapsed_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_time < 3.0:
                cmd.linear.x = -0.15
                self.get_logger().info(f"Sequence 6: Moving backwards. Elapsed time: {elapsed_time:.2f}s", throttle_duration_sec=1.0)
                self.pub_cmd_vel.publish(cmd)
            else:
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().info("Sequence 6 completed. Advancing to MOVE_TO_PUCK.")
                self.advance_sequence()
                self.state_start_time = None

        # Sequence 8: Lower Stick
        elif self.current_sequence == Sequence.LOWER_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("Sequence 8: Dispatching arm LOWER command (waiting 2s)...")
                self.arm_controller(direction=-1)

            elapsed_time = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed_time >= 2.0:
                self.get_logger().info("Sequence 8: Arm lower complete! Advancing sequence.")
                self.state_start_time = None
                self.advance_sequence()
            else:
                self.get_logger().info(f"Sequence 8: Lowering stick... {elapsed_time:.1f}s", throttle_duration_sec=1.0)

        # Sequence 9: Release Puck
        elif self.current_sequence == Sequence.RELEASE_PUCK:
            self.get_logger().info("Sequence 9: Releasing puck.")
            self.release_puck()
            self.advance_sequence()

        else:
            self.get_logger().info("All sequences completed. Robot is now idle.", throttle_duration_sec=3.0)
            self.pub_cmd_vel.publish(Twist())

    def nid_to_move_robot(self):
        l = self.get_parameter('l').value
        tolerance = self.get_parameter('tolerance').value
        Kp_v = self.get_parameter('kp_v').value
        Kp_w = self.get_parameter('kp_w').value
        v_max = self.get_parameter('v_max').value
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

        # --- MULTI-STAGE CONTROL FOR MOVE_TO_STICK ---
        if self.current_sequence == Sequence.MOVE_TO_STICK:
            target_x = p_xg + self.get_parameter('vertical_offset').value
            target_y = p_yg + self.get_parameter('sideways_offset').value

            valid_standoff_dist, standoff_x, standoff_y = self.get_valid_standoff_distance(
                target_x, target_y, target_theta, standoff_dist
            )

            # Stage 0: Rotate to face standoff location
            if self.seq1_stage == 0:
                bearing_to_standoff = np.arctan2(standoff_y - y, standoff_x - x)
                angle_error = np.arctan2(np.sin(bearing_to_standoff - theta), np.cos(bearing_to_standoff - theta))
                
                if abs(angle_error) > 0.02:
                    return 0.0, float(Kp_w * angle_error)
                else:
                    self.seq1_stage = 1
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 3 - Stage 0] Heading aligned to standoff vector. Advancing to Stage 1.")

            # Stage 1: Drive to standoff position with CLF-CBF
            elif self.seq1_stage == 1:
                dist = np.sqrt((standoff_x - p_xl)**2 + (standoff_y - p_yl)**2)
                if dist <= tolerance:
                    self.seq1_stage = 2
                    self.get_logger().info("[Seq 3 - Stage 1] Arrived at standoff location. Advancing to Stage 2 (Orientation).")
                    return 0.0, 0.0
                else:
                    e_x, e_y = standoff_x - p_xl, standoff_y - p_yl
                    p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y

                    # Velocity saturation
                    p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                    if p_dot_norm > v_max:
                        p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                        p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max

                    p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, standoff_x, standoff_y)

                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

            # Stage 2: Align with Tool Orientation
            elif self.seq1_stage == 2:
                flipped_target_theta = np.arctan2(np.sin(target_theta + np.pi), np.cos(target_theta + np.pi))
                angle_error = np.arctan2(np.sin(flipped_target_theta - theta), np.cos(flipped_target_theta - theta))
                
                if abs(angle_error) > 0.02:
                    return 0.0, float(Kp_w * angle_error)
                else:
                    self.seq1_stage = 3
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 3 - Stage 2] Alignment complete! Advancing to Stage 3 (Final Move).")

            # Stage 3: Drive final approach to stick
            elif self.seq1_stage == 3:
                dist = np.sqrt((target_x - p_xl)**2 + (target_y - p_yl)**2)
                if dist <= tolerance:
                    self.seq1_completed = True 
                    return 0.0, 0.0  
                else:
                    e_x, e_y = target_x - p_xl, target_y - p_yl
                    p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y

                    p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                    if p_dot_norm > v_max:
                        p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                        p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max

                    p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, target_x, target_y)

                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

        # --- STREAMLINED CONTROL FOR MOVE_TO_PUCK ---
        elif self.current_sequence == Sequence.MOVE_TO_PUCK:
            # Stage 0: Clean stationary rotation to face puck directly
            if self.seq4_stage == 0:
                bearing_to_puck = np.arctan2(p_yg - y, p_xg - x)
                angle_error = np.arctan2(np.sin(bearing_to_puck - theta), np.cos(bearing_to_puck - theta))
                
                if abs(angle_error) > 0.02:
                    return 0.0, float(Kp_w * angle_error)
                else:
                    self.seq4_stage = 1
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 7 - Stage 0] Heading aligned to puck position. Advancing to Stage 1 (Direct NID Drive).")
                    return 0.0, 0.0

            # Stage 1: Drive directly to puck location using NID + CBF (Finishes upon arrival)
            elif self.seq4_stage == 1:
                distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
                if distance_to_target <= tolerance:
                    self.seq4_completed = True
                    self.get_logger().info("[Seq 7 - Stage 1] Arrived at puck location! Sequence complete.")
                    return 0.0, 0.0
                else:
                    e_x, e_y = p_xg - p_xl, p_yg - p_yl
                    p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y

                    # Velocity saturation
                    p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                    if p_dot_norm > v_max:
                        p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                        p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max

                    p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, p_xg, p_yg)

                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

        return 0.0, 0.0

    def gripper_controller(self, open=False):
        if self.mock_mode:
            self.get_logger().info(f"Mock mode active: {'Opening' if open else 'Closing'} gripper simulated.")
            self.gripper_action_running = False
            self.state_start_time = None
            self.advance_sequence()
            return
        self.get_logger().info("Gripper Operation running...") 
        goal = GripperControl.Goal()
        goal.target_state = 1 if open else 2
        future = self.gripper_action_client.send_goal_async(goal)
        self.get_logger().info("Gripper goal request dispatched.")
        future.add_done_callback(self._goal_response_cb)

    def move_arm_using_action(self, x, z, relative=False):
        if self.mock_mode:
            self.get_logger().info(f"Mock mode active: Arm move to ({x}, {z}) simulated.")
            self.arm_action_running = False
            self.state_start_time = None
            self.advance_sequence()
            return
        self.get_logger().info(f"Moving arm to pose ({x}, {z})...")
        goal = MoveArm.Goal()
        goal.x = x
        goal.z = z
        goal.relative = relative
        future = self.arm_action_client.send_goal_async(goal)
        self.get_logger().info("Sending arm move goal request...")
        future.add_done_callback(self._arm_goal_response_cb)

    def arm_controller(self, direction=1):
        if self.mock_mode:
            self.get_logger().info(f"Mock mode active: Arm {'lifting' if direction == 1 else 'lowering'} simulated.")
            return
        cmd = Vector3()
        cmd.x = 0.0
        cmd.z = 0.10 * direction
        self.pub_cmd_arm.publish(cmd)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        self.get_logger().info(f"Gripper Goal Handle Result: {goal_handle}")
        if not goal_handle.accepted:
            self.get_logger().warn("Gripper Goal rejected by server! Will retry after cooldown...")
            self.gripper_action_running = False 
            return
        self.get_logger().info("Gripper Goal accepted by server. Awaiting execution result...")
        goal_handle.get_result_async().add_done_callback(self._result_cb)

    def _arm_goal_response_cb(self, future):
        goal_handle = future.result()
        self.get_logger().info(f"Arm Goal Handle Result: {goal_handle}")
        if not goal_handle.accepted:
            self.get_logger().warn("Arm Goal rejected by server! Will retry after cooldown...")
            self.arm_action_running = False 
            return
        self.get_logger().info("Arm Goal accepted by server. Awaiting execution result...")
        goal_handle.get_result_async().add_done_callback(self._arm_result_cb)

    def _result_cb(self, future):
        try:
            result = future.result()
            self.get_logger().info(f'Gripper operation succeeded. Moving to Sequence {self.current_sequence.name}')
            self.advance_sequence()
        except Exception as e:
            self.get_logger().error(f'Gripper execution tracking faulted: {e}. Retrying...')
        finally:
            self.gripper_action_running = False
            self.state_start_time = None 

    def _arm_result_cb(self, future):
        try:
            result = future.result()
            self.get_logger().info(f'Arm operation succeeded. Moving to Sequence {self.current_sequence.name}')
            self.advance_sequence()
        except Exception as e:
            self.get_logger().error(f'Arm execution tracking faulted: {e}. Retrying...')
        finally:
            self.arm_action_running = False
            self.state_start_time = None

    def release_puck(self):
        dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the goal"
        self.get_logger().info(f"Releasing / Shooting the puck to {dest}...")

def main(args=None):
    parser = argparse.ArgumentParser(description='Move Robot Node with CLF-CBF Obstacle Avoidance')
    parser.add_argument('--robot_id', type=int, required=True, help='ID of the robot to control')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of ally robot to pass to (0 for goal)')
    parser.add_argument('--hockey_stick_id', type=int, default=1, help='ID tag integer for the hockey stick VRPN tracking topic')
    parser.add_argument('--puck_color', type=str, default='blue', help='Color tag string for the puck VRPN tracking topic')
    parser.add_argument('--mock_mode', action='store_true', help='Enable mock mode for testing without real VRPN data')
    parser.add_argument('--orient_to_stick', action='store_true', help='Enable terminal angle orientation alignment for the hockey stick')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help="Sideways offset for hockey stick pose")
    parser.add_argument('--vertical_offset', type=float, default=0.0, help="Vertical offset for hockey stick pose")
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along the vector field line')
    parser.add_argument('--r_safety', type=float, default=0.35, help='Safety radius for obstacle avoidance')
    parser.add_argument('--l', type=float, default=0.15, help='Look-ahead center to end-effector displacement distance')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Target proximity threshold radius')

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
        standoff_distance=args.standoff_distance,
        r_safety=args.r_safety
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