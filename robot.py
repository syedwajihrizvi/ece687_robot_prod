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
from std_msgs.msg import Bool
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
    # hit-mode sequences (pass/shoot by spinning the carried stick into the puck)
    MOVE_TO_WAIT = 10
    WAIT_FOR_PASS = 11
    ALIGN_HIT = 12
    SPIN_HIT = 13
    HIT_DONE = 14

class Robot(Node):
    def __init__(self, 
                 robot_id, 
                 pass_to_robot, 
                 hockey_stick_id=1, 
                 puck_color='blue',
                 mock_mode=False,
                 sim_mode=False,
                 orient_to_stick=False,
                 l_default=0.15, 
                 tolerance_default=0.15, 
                 sideways_offset=0.0, 
                 vertical_offset=0.0, 
                 standoff_distance=2.5,
                 r_safety=0.35,
                 hit_mode=False,
                 wait_for_pass=False,
                 swing_offset=0.55,
                 wait_radius=3.0,
                 hit_spin_speed=4.0,
                 hit_swing_angle=4.71,
                 goal_x=0.0,
                 goal_y=-1.75,
                 goal_yaw_deg=90.0):
        super().__init__(f'robot_{robot_id}_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.gripper_action = f'/robot{robot_id}/gripper'
        self.arm_action = f'/robot{robot_id}/arm'
        self.pass_to_robot = pass_to_robot
        self.hockey_stick_id = hockey_stick_id
        self.puck_color = puck_color
        self.mock_mode = mock_mode
        # sim_mode: fake gripper/arm actions like mock_mode, but keep the real /vrpn_mocap
        # topics — for running against the Docker multi_robomaster_ros_sim simulator
        self.sim_mode = sim_mode
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

        # Hit-mode state (pass/shoot by spinning the carried stick into the puck)
        self.hit_mode = hit_mode
        self.wait_for_pass = wait_for_pass
        self.hit_side = None       # +1/-1: which side of the puck->aim line the robot swings from
        self.spin_accum = 0.0      # accumulated rotation during SPIN_HIT
        self.wait_stage = 0        # MOVE_TO_WAIT sub-stage (0 rotate, 1 drive)
        self.puck_speed = 0.0      # low-pass filtered puck speed estimate from mocap
        self._puck_prev_time = None
        self._initial_puck_pos = None  # first observed puck position (pass detection reference)

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

        # Hit-mode tunables
        self.declare_parameter('swing_offset', swing_offset)      # perpendicular park distance from the puck (~= sim stick tip length)
        self.declare_parameter('wait_radius', wait_radius)        # puck within this range of the shooter triggers the shot phase
        self.declare_parameter('hit_spin_speed', hit_spin_speed)  # rad/s during SPIN_HIT; launch speed ~= this * swing_offset
        self.declare_parameter('hit_swing_angle', hit_swing_angle) # total SPIN_HIT sweep (rad); contact happens ~pi in
        self.declare_parameter('goal_x', goal_x)
        self.declare_parameter('goal_y', goal_y)
        self.declare_parameter('goal_yaw', goal_yaw_deg * math.pi / 180.0)

        self.current_sequence = Sequence(self.get_parameter('start_sequence').value)

        # Sequence route: hit-mode replaces the RELEASE_PUCK stub with the swing-hit tail,
        # and the shooter inserts the wait-at-goal-standoff phase before approaching the puck
        base_route = [Sequence(i) for i in range(7)]  # OPEN_GRIPPER .. MOVE_BACK_ROTATE
        if self.hit_mode:
            hit_tail = [Sequence.MOVE_TO_PUCK, Sequence.LOWER_STICK,
                        Sequence.ALIGN_HIT, Sequence.SPIN_HIT, Sequence.HIT_DONE]
            if self.wait_for_pass:
                self.sequence_route = base_route + [Sequence.MOVE_TO_WAIT, Sequence.WAIT_FOR_PASS] + hit_tail
            else:
                self.sequence_route = base_route + hit_tail
        else:
            self.sequence_route = [Sequence(i) for i in range(10)]  # original behavior

        self.L_inv = np.array([[1, 0], [0, 1 / self.get_parameter('l').value]])
        self._action_group = ReentrantCallbackGroup()
        self.gripper_action_client = None
        self.arm_action_client = None

        if not (self.mock_mode or self.sim_mode):
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
        # sim_mode: tells the simulator to attach/release the stick on gripper close/open
        self.pub_gripper_sim = self.create_publisher(Bool, f'{self.robot_name}/gripper_sim', 10) if self.sim_mode else None
        self.get_logger().info(f'Robot node initialized at sequence state: {self.current_sequence.name} with stick ID: {self.hockey_stick_id} & puck color: {self.puck_color}')

    def advance_sequence(self):
        """Advances state machine along the active sequence route and resets velocity filter memory."""
        try:
            idx = self.sequence_route.index(self.current_sequence)
            self.current_sequence = self.sequence_route[idx + 1]
        except (ValueError, IndexError):
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
        now = self.get_clock().now()
        if self._initial_puck_pos is None:
            self._initial_puck_pos = (msg.pose.position.x, msg.pose.position.y)
        if self.puck_pose is not None and self._puck_prev_time is not None:
            dt = (now - self._puck_prev_time).nanoseconds / 1e9
            if dt > 1e-3:
                dx = msg.pose.position.x - self.puck_pose.position.x
                dy = msg.pose.position.y - self.puck_pose.position.y
                self.puck_speed = 0.5 * (math.sqrt(dx * dx + dy * dy) / dt) + 0.5 * self.puck_speed
        self._puck_prev_time = now
        self.puck_pose = msg.pose

    def get_aim_point(self):
        """Where the puck should be sent: the ally robot's live pose (pass) or the goal mouth."""
        if self.pass_to_robot:
            pose = self.obstacle_poses.get(f'obstacle_robot_{self.pass_to_robot}')
            if pose is None:
                return None
            return pose.position.x, pose.position.y
        return self.get_parameter('goal_x').value, self.get_parameter('goal_y').value

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

        # Hit mode: treat the puck as a CBF obstacle so no driving leg rolls over it —
        # only the spinning stick tip is supposed to touch the puck
        if self.hit_mode and self.puck_pose is not None:
            self.obstacle_poses['virtual_puck'] = self.puck_pose

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
                self.get_logger().info("Sequence 6 completed. Advancing.")
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

        # Sequence 10: Park at the goal standoff point to await the pass (shooter)
        elif self.current_sequence == Sequence.MOVE_TO_WAIT:
            l = self.get_parameter('l').value
            tolerance = self.get_parameter('tolerance').value
            Kp_v = self.get_parameter('kp_v').value
            Kp_w = self.get_parameter('kp_w').value
            v_max = self.get_parameter('v_max').value
            gyaw = self.get_parameter('goal_yaw').value
            wait_x = self.get_parameter('goal_x').value + self.get_parameter('standoff_distance').value * math.cos(gyaw)
            wait_y = self.get_parameter('goal_y').value + self.get_parameter('standoff_distance').value * math.sin(gyaw)

            x = self.robot_pose.position.x
            y = self.robot_pose.position.y
            theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)
            p_xl = x + l * math.cos(theta)
            p_yl = y + l * math.sin(theta)

            cmd = Twist()
            # Stage 0: rotate in place to face the waiting point
            if self.wait_stage == 0:
                bearing = np.arctan2(wait_y - y, wait_x - x)
                angle_error = np.arctan2(np.sin(bearing - theta), np.cos(bearing - theta))
                if abs(angle_error) > 0.02:
                    cmd.angular.z = float(Kp_w * angle_error)
                else:
                    self.wait_stage = 1
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 10 - Stage 0] Heading aligned to waiting point. Advancing to Stage 1.")
            # Stage 1: drive to the waiting point with CLF-CBF
            else:
                dist = np.sqrt((wait_x - p_xl)**2 + (wait_y - p_yl)**2)
                if dist <= tolerance:
                    self.pub_cmd_vel.publish(Twist())
                    self.get_logger().info("[Seq 10] Arrived at goal standoff waiting point.")
                    self.advance_sequence()
                    self.state_start_time = None
                    return
                e_x, e_y = wait_x - p_xl, wait_y - p_yl
                p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y
                p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                if p_dot_norm > v_max:
                    p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                    p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max
                p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, wait_x, wait_y)
                self.L_inv[1, 1] = 1.0 / l
                control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                cmd.linear.x = float(control_inputs[0, 0])
                cmd.angular.z = float(control_inputs[1, 0])
                self.get_logger().info(f"Sequence MOVE_TO_WAIT: v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}", throttle_duration_sec=1.0)
            self.pub_cmd_vel.publish(cmd)

        # Sequence 11: Hold position until the pass arrives (puck moved, close, and stopped)
        elif self.current_sequence == Sequence.WAIT_FOR_PASS:
            self.pub_cmd_vel.publish(Twist())
            if self.puck_pose is None or self._initial_puck_pos is None:
                return
            px, py = self.puck_pose.position.x, self.puck_pose.position.y
            displacement = math.hypot(px - self._initial_puck_pos[0], py - self._initial_puck_pos[1])
            dist_to_me = math.hypot(px - self.robot_pose.position.x, py - self.robot_pose.position.y)
            if displacement > 0.3 and dist_to_me <= self.get_parameter('wait_radius').value and self.puck_speed < 0.15:
                self.get_logger().info(f"[Seq 11] Pass received: puck at ({px:.2f}, {py:.2f}), {dist_to_me:.2f} m away. Moving to shoot.")
                self.advance_sequence()
            else:
                self.get_logger().info(
                    f"[Seq 11] Waiting for pass (moved {displacement:.2f} m, dist {dist_to_me:.2f} m, speed {self.puck_speed:.2f} m/s)...",
                    throttle_duration_sec=2.0)

        # Sequence 12: Rotate the stick to point AWAY from the puck (slow, so the tip can't launch it)
        elif self.current_sequence == Sequence.ALIGN_HIT:
            if self.puck_pose is None:
                return
            x = self.robot_pose.position.x
            y = self.robot_pose.position.y
            theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)
            bearing_to_puck = np.arctan2(self.puck_pose.position.y - y, self.puck_pose.position.x - x)
            away_heading = np.arctan2(np.sin(bearing_to_puck + np.pi), np.cos(bearing_to_puck + np.pi))
            angle_error = np.arctan2(np.sin(away_heading - theta), np.cos(away_heading - theta))
            cmd = Twist()
            if abs(angle_error) > 0.03:
                Kp_w = self.get_parameter('kp_w').value
                cmd.angular.z = float(np.clip(Kp_w * angle_error, -0.6, 0.6))  # capped: tip stays below launch speed
                self.pub_cmd_vel.publish(cmd)
            else:
                self.pub_cmd_vel.publish(Twist())
                self.spin_accum = 0.0
                self.get_logger().info("[Seq 12] Stick wound up (pointing away from puck). Starting swing.")
                self.advance_sequence()

        # Sequence 13: Fast swing — the stick tip sweeps through the puck, launching it toward the aim point
        elif self.current_sequence == Sequence.SPIN_HIT:
            if self.hit_side is None:
                self.hit_side = 1.0
            spin_speed = self.get_parameter('hit_spin_speed').value
            cmd = Twist()
            cmd.angular.z = float(-self.hit_side * spin_speed)
            self.spin_accum += spin_speed / self.get_parameter('control_frequency').value
            if self.spin_accum >= self.get_parameter('hit_swing_angle').value:
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().info("[Seq 13] Swing complete.")
                self.advance_sequence()
            else:
                self.pub_cmd_vel.publish(cmd)

        # Sequence 14: Hit finished
        elif self.current_sequence == Sequence.HIT_DONE:
            self.pub_cmd_vel.publish(Twist())
            dest = f"robot {self.pass_to_robot}" if self.pass_to_robot else "the goal"
            self.get_logger().info(f"[Seq 14] Puck sent toward {dest}. Task complete.")
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
            # Hit mode: don't drive at the puck itself — park at the swing center, a point
            # offset perpendicular to the puck->aim line so the spinning stick tip sweeps
            # through the puck in the aim direction (MATLAB swingCenterTarget geometry)
            if self.hit_mode:
                aim = self.get_aim_point()
                if aim is None:
                    self.get_logger().warn("Hit mode: awaiting pass-target pose...", throttle_duration_sec=2.0)
                    return 0.0, 0.0
                d_aim = np.hypot(aim[0] - p_xg, aim[1] - p_yg)
                if d_aim < 1e-3:
                    self.get_logger().warn("Hit mode: aim point coincides with puck, waiting...", throttle_duration_sec=2.0)
                    return 0.0, 0.0
                aim_unit = np.array([aim[0] - p_xg, aim[1] - p_yg]) / d_aim
                normal = np.array([aim_unit[1], -aim_unit[0]])
                swing_offset = self.get_parameter('swing_offset').value
                if self.hit_side is None:
                    c_plus = np.array([p_xg, p_yg]) + swing_offset * normal
                    c_minus = np.array([p_xg, p_yg]) - swing_offset * normal
                    d_plus = np.hypot(c_plus[0] - x, c_plus[1] - y)
                    d_minus = np.hypot(c_minus[0] - x, c_minus[1] - y)
                    self.hit_side = 1.0 if d_plus <= d_minus else -1.0
                    self.get_logger().info(f"Hit mode: swing side {self.hit_side:+.0f}, aim point ({aim[0]:.2f}, {aim[1]:.2f})")
                p_xg = p_xg + self.hit_side * swing_offset * normal[0]
                p_yg = p_yg + self.hit_side * swing_offset * normal[1]

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
        if self.mock_mode or self.sim_mode:
            if self.sim_mode:
                grip_msg = Bool()
                grip_msg.data = not open  # True = closed
                self.pub_gripper_sim.publish(grip_msg)
            self.get_logger().info(f"Mock/sim mode active: {'Opening' if open else 'Closing'} gripper simulated.")
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
        if self.mock_mode or self.sim_mode:
            self.get_logger().info(f"Mock/sim mode active: Arm move to ({x}, {z}) simulated.")
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
        if self.mock_mode or self.sim_mode:
            self.get_logger().info(f"Mock/sim mode active: Arm {'lifting' if direction == 1 else 'lowering'} simulated.")
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
    parser.add_argument('--sim_mode', action='store_true', help='Fake gripper/arm actions but use real /vrpn_mocap topics (for the Docker multi_robomaster_ros_sim simulator)')
    parser.add_argument('--orient_to_stick', action='store_true', help='Enable terminal angle orientation alignment for the hockey stick')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help="Sideways offset for hockey stick pose")
    parser.add_argument('--vertical_offset', type=float, default=0.0, help="Vertical offset for hockey stick pose")
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along the vector field line')
    parser.add_argument('--r_safety', type=float, default=0.35, help='Safety radius for obstacle avoidance')
    parser.add_argument('--l', type=float, default=0.15, help='Look-ahead center to end-effector displacement distance')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Target proximity threshold radius')
    parser.add_argument('--hit_mode', action='store_true', help='Pass/shoot by spinning the carried stick into the puck (replaces the RELEASE_PUCK stub)')
    parser.add_argument('--wait_for_pass', action='store_true', help='Shooter role: park at the goal standoff and wait for the pass before approaching the puck')
    parser.add_argument('--swing_offset', type=float, default=0.55, help='Perpendicular park distance from the puck when preparing a hit (~ carried stick tip length)')
    parser.add_argument('--wait_radius', type=float, default=3.0, help='Puck arriving within this range of the shooter triggers the shot phase')
    parser.add_argument('--hit_spin_speed', type=float, default=4.0, help='Angular speed (rad/s) of the hit swing; launch speed ~= this * swing_offset')
    parser.add_argument('--hit_swing_angle', type=float, default=4.71, help='Total swing sweep angle (rad); contact happens ~pi in')
    parser.add_argument('--goal_x', type=float, default=0.0, help='Goal mouth center x (m)')
    parser.add_argument('--goal_y', type=float, default=-1.75, help='Goal mouth center y (m)')
    parser.add_argument('--goal_yaw_deg', type=float, default=90.0, help='Goal facing direction (deg); mouth opens along it')

    args, remaining = parser.parse_known_args(args)
    if args.mock_mode and args.sim_mode:
        parser.error('--mock_mode and --sim_mode are mutually exclusive')
    rclpy.init(args=remaining)
    node = Robot(
        robot_id=args.robot_id, 
        pass_to_robot=args.pass_to_robot, 
        hockey_stick_id=args.hockey_stick_id,
        puck_color=args.puck_color,
        mock_mode=args.mock_mode,
        sim_mode=args.sim_mode,
        orient_to_stick=args.orient_to_stick,
        l_default=args.l,
        tolerance_default=args.tolerance,
        sideways_offset=args.sideways_offset,
        vertical_offset=args.vertical_offset,
        standoff_distance=args.standoff_distance,
        r_safety=args.r_safety,
        hit_mode=args.hit_mode,
        wait_for_pass=args.wait_for_pass,
        swing_offset=args.swing_offset,
        wait_radius=args.wait_radius,
        hit_spin_speed=args.hit_spin_speed,
        hit_swing_angle=args.hit_swing_angle,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_yaw_deg=args.goal_yaw_deg
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