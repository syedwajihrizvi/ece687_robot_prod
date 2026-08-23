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
from geometry_msgs.msg import Twist, PoseStamped, Vector3, Point
from std_msgs.msg import Bool
from scipy.optimize import minimize
from nav_recorder import NavRecorder, NullRecorder

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
10: Move to Wait Position
11: Wait for Pass
12: Align Hit
13: Spin Hit
14: Hit Done
16: Shoot Setup (optional: stand on the line through the puck perpendicular to the shot)
17: Aim Shot (freeze the shooting direction vector and the swing plan)
18: Wind Up (rotate back by undershoot_w)
19: Swing (open-loop sweep of undershoot_w + overshoot_w through the puck)
20: Shoot Done

Open-loop shooting geometry
---------------------------
The blade tip rides a circle centred on the chassis, so when the tip reaches the
puck its velocity is tangential: the puck leaves perpendicular to the
chassis->puck line ("puck-robot-origin line"). Sequences 17-19 therefore:
  * take the shooting direction vector = puck -> hockey_goal_<id> (mocap),
  * pick the spin sense whose tangent points down that vector,
  * wind up undershoot_w BEFORE the puck-robot-origin line,
  * sweep undershoot_w + overshoot_w open loop, so contact happens on the line
    and the follow-through carries overshoot_w past it.
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
    # Hit Mode Sequences
    MOVE_TO_WAIT = 10
    WAIT_FOR_PASS = 11
    ALIGN_HIT = 12
    SPIN_HIT = 13
    HIT_DONE = 14
    # Open-Loop Goal Shooting Sequences
    SHOOT_SETUP = 16
    AIM_SHOT = 17
    WIND_UP = 18
    SWING = 19
    SHOOT_DONE = 20

class Robot(Node):
    def __init__(self, 
                 robot_id, 
                 pass_to_robot, 
                 hockey_stick_id=1, 
                 puck_color='blue', 
                 mock_mode=False, 
                 sim_mode=False,
                 orient_to_stick=False, 
                 l_default=0.50, 
                 tolerance_default=0.10, 
                 sideways_offset=0.1, 
                 vertical_offset=0.15, 
                 standoff_distance=2.5,
                 r_safety=0.5,
                 hit_mode=False,
                 wait_for_pass=False,
                 swing_offset=0.55,
                 puck_contact_offset=0.0,
                 wait_radius=3.0,
                 hit_spin_speed=4.0,
                 hit_swing_angle=4.71,
                 goal_x=0.0,
                 goal_y=-1.75,
                 goal_yaw_deg=90.0,
                 shoot=False,
                 goal_id=1,
                 aim_target='goal',
                 undershoot_w_deg=60.0,
                 overshoot_w_deg=45.0,
                 swing_speed=6.0,
                 swing_rate=100.0,
                 windup_speed=1.5,
                 shot_setup=False,
                 shot_standoff=1.2,
                 windup_lift=True,
                 lower_before_aim=False,
                 windup_lift_z=0.12,
                 swing_arm_z=-0.08,
                 arm_settle_time=1.5,
                 pre_swing_pause=0.5,
                 swing_stop_time=0.5,
                 report_delay=2.0,
                 max_aim_error_deg=25.0,
                 strict_aim=False,
                 record=False,
                 log_dir='records',
                 trial_id='1',
                 start_sequence=0):
        super().__init__(f'robot_{robot_id}_node')
        self.robot_id = robot_id
        self.robot_name = f'/robot{robot_id}'
        self.gripper_action = f'/robot{robot_id}/gripper'
        self.arm_action = f'/robot{robot_id}/move_arm'
        self.pass_to_robot = pass_to_robot
        self.hockey_stick_id = hockey_stick_id
        self.puck_color = puck_color
        self.mock_mode = mock_mode
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

        # Hit-mode state
        self.hit_mode = hit_mode
        self.wait_for_pass = wait_for_pass
        self.hit_side = None
        self.spin_accum = 0.0
        self.wait_stage = 0
        self.puck_speed = 0.0
        self._puck_prev_time = None
        self._initial_puck_pos = None

        # Open-loop shooting state
        self.shoot = shoot
        self.goal_id = goal_id
        self.aim_target = aim_target
        self.shot_setup = shot_setup
        self.windup_lift = windup_lift
        self.lower_before_aim = lower_before_aim
        self.strict_aim = strict_aim
        self.goal_poses = {}
        self.shot_side = None
        self.shot_setup_stage = 0
        self.windup_stage = 0
        self.shot_plan = None
        self._shot_puck_pos = None
        self.swing_started = False
        self.swing_yaw_prev = None
        self.swing_yaw_accum = 0.0
        self._swing_timer = None
        self._swing_done = False
        self._swing_omega = 0.0
        self._swing_duration = 0.0
        self._swing_t0 = None

        # Controller tunings & parameters
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('kp_v', 1.2)
        self.declare_parameter('kp_w', 1.0)
        self.declare_parameter('v_max', 1.0)  # Maximum workspace velocity cap (m/s)
        self.l = l_default
        self.declare_parameter('l', l_default)
        self.declare_parameter('tolerance', tolerance_default)
        self.tolerace = tolerance_default
        self.declare_parameter('standoff_distance', standoff_distance)
        self.declare_parameter('start_sequence', 0) 
        self.declare_parameter('sideways_offset', sideways_offset)
        self.declare_parameter('vertical_offset', vertical_offset)

        # Extended mode tunables
        self.declare_parameter('swing_offset', swing_offset)
        self.declare_parameter('puck_contact_offset', puck_contact_offset)  # tip passes this far (m) from the puck CENTER at contact; = puck radius -> grazes the edge
        self.declare_parameter('wait_radius', wait_radius)
        self.declare_parameter('hit_spin_speed', hit_spin_speed)
        self.declare_parameter('hit_swing_angle', hit_swing_angle)
        self.declare_parameter('goal_x', goal_x)
        self.declare_parameter('goal_y', goal_y)
        self.declare_parameter('goal_yaw', goal_yaw_deg * math.pi / 180.0)

        # Open-loop shooting tunables (all angles held in radians)
        self.declare_parameter('undershoot_w', undershoot_w_deg * math.pi / 180.0)
        self.declare_parameter('overshoot_w', overshoot_w_deg * math.pi / 180.0)
        self.declare_parameter('swing_speed', swing_speed)
        self.declare_parameter('swing_rate', swing_rate)
        self.declare_parameter('windup_speed', windup_speed)
        self.declare_parameter('shot_standoff', shot_standoff)
        self.declare_parameter('windup_lift_z', windup_lift_z)
        self.declare_parameter('swing_arm_z', swing_arm_z)
        self.declare_parameter('arm_settle_time', arm_settle_time)
        self.declare_parameter('pre_swing_pause', pre_swing_pause)
        self.declare_parameter('swing_stop_time', swing_stop_time)
        self.declare_parameter('report_delay', report_delay)
        self.declare_parameter('max_aim_error', max_aim_error_deg * math.pi / 180.0)

        self.current_sequence = Sequence(start_sequence)

        # Dynamic Route Construction
        base_route = [Sequence(i) for i in range(7)]
        if self.shoot:
            # The blade parks straddling the puck (that is what mid-blade contact looks like),
            # so lowering it there would shove the puck. Leave it raised and let the wind-up
            # set it down once it has rotated clear - unless the wind-up lift is switched off,
            # in which case something has to put the blade down first.
            self.lower_before_aim = lower_before_aim or not self.windup_lift
            shoot_tail = [Sequence.MOVE_TO_PUCK]
            if self.lower_before_aim:
                shoot_tail.append(Sequence.LOWER_STICK)
            shoot_tail += [Sequence.AIM_SHOT, Sequence.WIND_UP, Sequence.SWING, Sequence.SHOOT_DONE]
            if self.shot_setup:
                self.sequence_route = base_route + [Sequence.SHOOT_SETUP] + shoot_tail
            else:
                self.sequence_route = base_route + shoot_tail
        elif self.hit_mode:
            hit_tail = [Sequence.MOVE_TO_PUCK, Sequence.LOWER_STICK,
                        Sequence.ALIGN_HIT, Sequence.SPIN_HIT, Sequence.HIT_DONE]
            if self.wait_for_pass:
                self.sequence_route = base_route + [Sequence.MOVE_TO_WAIT, Sequence.WAIT_FOR_PASS] + hit_tail
            else:
                self.sequence_route = base_route + hit_tail
        else:
            self.sequence_route = [Sequence(i) for i in range(10)]

        # Navigation convergence recording. nid_to_move_robot() stashes the geometry
        # it just solved in _nav_sample; the control loop pairs it with the (v, w) that
        # came back and hands the row to the recorder.
        self._nav_sample = None
        if record:
            self.recorder = NavRecorder(
                log_dir, trial_id, self.get_clock(), logger=self.get_logger(),
                meta={'robot_id': robot_id, 'hockey_stick_id': hockey_stick_id,
                      'l': l_default, 'tolerance': tolerance_default,
                      'standoff_distance': standoff_distance, 'r_safety': r_safety,
                      'sideways_offset': sideways_offset, 'vertical_offset': vertical_offset,
                      'kp_v': self.get_parameter('kp_v').value,
                      'kp_w': self.get_parameter('kp_w').value,
                      'v_max': self.get_parameter('v_max').value,
                      'control_frequency': self.get_parameter('control_frequency').value,
                      'sim_mode': sim_mode, 'mock_mode': mock_mode})
        else:
            self.recorder = NullRecorder()

        self.L_inv = np.array([[1, 0], [0, 1 / self.l]])
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
            for gid in (1, 2):
                # Goals are shot targets, never CBF obstacles - kept out of obstacle_poses.
                self.create_subscription(PoseStamped, f'/mock/vrpn_mocap/hockey_goal_{gid}/pose', self.goal_pos_callback(gid), qos)
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
            for gid in (1, 2):
                # Goals are shot targets, never CBF obstacles - kept out of obstacle_poses.
                self.create_subscription(PoseStamped, f'/vrpn_mocap/hockey_goal_{gid}/pose', self.goal_pos_callback(gid), qos)

        self.pub_cmd_vel = self.create_publisher(Twist, f'{self.robot_name}/cmd_vel', 10)
        self.pub_cmd_arm = self.create_publisher(Point, f'{self.robot_name}/target_arm_position', 10)
        self.pub_gripper_sim = self.create_publisher(Bool, f'{self.robot_name}/gripper_sim', 10) if self.sim_mode else None
        self.pub_ready_to_pass_puck = self.create_publisher(Bool, f'{self.robot_name}/ready_to_pass_puck', 10)
        self.pub_ready_to_receive_puck = self.create_publisher(Bool, f'{self.robot_name}/ready_to_receive_puck', 10)

        # Publish initial false for passing or receiving puck readiness
        self.publish_ready_to_pass_puck(False)
        self.publish_ready_to_receive_puck(False)
        self.get_logger().info(f'Robot node initialized at sequence state: {self.current_sequence.name} with stick ID: {self.hockey_stick_id} & puck color: {self.puck_color}')
        if self.current_sequence not in self.sequence_route:
            self.get_logger().error(
                f'--start_sequence {start_sequence} ({self.current_sequence.name}) is not on this route: '
                f'{" -> ".join(s.name for s in self.sequence_route)}. The node will idle immediately.')
        if self.shoot:
            aim = f"robot {self.pass_to_robot}" if (self.aim_target == 'ally' and self.pass_to_robot) else f"hockey_goal_{self.goal_id}"
            self.get_logger().info(
                f'Open-loop shooting enabled: aiming at {aim}, undershoot_w={undershoot_w_deg:.0f} deg, '
                f'overshoot_w={overshoot_w_deg:.0f} deg, swing {swing_speed:.1f} rad/s, '
                f'shot_setup={"on" if self.shot_setup else "off"}, '
                f'blade {"lowered on arrival" if self.lower_before_aim else "kept up until the wind-up clears the puck"}. '
                f'Route: {" -> ".join(s.name for s in self.sequence_route)}')

    def advance_sequence(self):
        """Advances state machine along sequence_route and resets velocity filter memory."""
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

    def goal_pos_callback(self, goal_id):
        def callback(msg):
            self.goal_poses[goal_id] = msg.pose
        return callback

    @staticmethod
    def wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    def get_shot_target(self):
        """Point the puck must be sent to, as ((x, y), label). None while data is missing."""
        if self.aim_target == 'ally' and self.pass_to_robot:
            pose = self.obstacle_poses.get(f'obstacle_robot_{self.pass_to_robot}')
            if pose is None:
                self.get_logger().warn(f"Awaiting mocap for ally robot {self.pass_to_robot}...", throttle_duration_sec=2.0)
                return None
            return (pose.position.x, pose.position.y), f"robot {self.pass_to_robot}"

        pose = self.goal_poses.get(self.goal_id)
        if pose is not None:
            return (pose.position.x, pose.position.y), f"hockey_goal_{self.goal_id}"
        self.get_logger().warn(
            f"No mocap on hockey_goal_{self.goal_id}; falling back to static goal_x/goal_y.",
            throttle_duration_sec=5.0)
        return (self.get_parameter('goal_x').value, self.get_parameter('goal_y').value), "static goal_x/goal_y"

    def compute_shot_geometry(self):
        """Shooting direction vector and the spin that delivers the puck along it.

        The blade tip orbits the chassis, so at contact the puck leaves along the
        tangent - perpendicular to the chassis->puck ("puck-robot-origin") line.
        Only the spin sense is free; the aim itself comes from where the chassis
        stands, which is what 'aim_error' below reports. Returns None if a pose is
        still missing.
        """
        if self.puck_pose is None or self.robot_pose is None:
            return None
        target = self.get_shot_target()
        if target is None:
            return None
        (tx, ty), label = target

        px, py = self.puck_pose.position.x, self.puck_pose.position.y
        shot = np.array([tx - px, ty - py])
        shot_dist = float(np.linalg.norm(shot))
        if shot_dist < 1e-3:
            self.get_logger().warn("Shot target coincides with the puck; cannot aim.", throttle_duration_sec=2.0)
            return None
        u = shot / shot_dist  # unit shooting direction vector (puck -> target)

        x = self.robot_pose.position.x
        y = self.robot_pose.position.y
        theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)
        d = np.array([px - x, py - y])  # chassis -> puck: the swing arm at contact
        radius = float(np.linalg.norm(d))
        if radius < 1e-3:
            return None
        d_hat = d / radius

        # Spin sense whose tangential tip velocity at the puck points down the shot line.
        cross = float(d_hat[0] * u[1] - d_hat[1] * u[0])
        spin_sign = 1.0 if cross >= 0.0 else -1.0
        tangent = spin_sign * np.array([-d_hat[1], d_hat[0]])  # tip velocity direction at contact
        tangent_heading = math.atan2(tangent[1], tangent[0])
        shot_heading = math.atan2(u[1], u[0])

        return {
            'puck': (px, py),
            'target': (tx, ty),
            'label': label,
            'u': u,
            'shot_dist': shot_dist,
            'shot_heading': shot_heading,
            'd_hat': d_hat,
            'radius': radius,
            'theta': theta,
            # Yaw at which the blade (assumed on the chassis forward axis) points at the puck.
            'bearing': math.atan2(d[1], d[0]),
            'spin_sign': spin_sign,
            'tangent_heading': tangent_heading,
            # Where the puck will actually go minus where we want it to go.
            'aim_error': self.wrap_angle(tangent_heading - shot_heading),
        }

    def _start_swing_timer(self):
        """Runs the open-loop sweep on its own fast timer so the stop instant is not
        quantised by the 10 Hz control loop (34 deg per tick at 6 rad/s)."""
        rate = max(10.0, float(self.get_parameter('swing_rate').value))
        if self._swing_timer is not None:
            self.destroy_timer(self._swing_timer)
            self._swing_timer = None
        self._swing_done = False
        self._swing_t0 = self.get_clock().now()
        self._swing_timer = self.create_timer(1.0 / rate, self._swing_tick, callback_group=self._action_group)

    def _swing_tick(self):
        timer = self._swing_timer
        if timer is None or self._swing_done:
            return
        elapsed = (self.get_clock().now() - self._swing_t0).nanoseconds / 1e9
        if elapsed < self._swing_duration:
            cmd = Twist()
            cmd.angular.z = self._swing_omega
            self.pub_cmd_vel.publish(cmd)
            return
        self.pub_cmd_vel.publish(Twist())  # stop exactly at the end of the sweep
        self._swing_done = True
        timer.cancel()

    def get_valid_standoff_distance(self, target_x, target_y, target_theta, initial_standoff):
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
        self.get_logger().info(f"Current Sequence: {self.current_sequence}")
        if self.robot_pose is None:
            self.get_logger().warn("Waiting for robot pose...", throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()

        # Dynamic CBF Obstacle logic for puck in hit mode
        if self.hit_mode and self.puck_pose is not None:
            self.obstacle_poses['virtual_puck'] = self.puck_pose

        # Shoot mode: the puck is the target, never a CBF obstacle - a safety bubble
        # around it would fight the approach and the shot-line setup.
        if self.shoot:
            self.obstacle_poses.pop('virtual_puck', None)

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
            self.move_arm_using_publisher(x=0.2, z=-0.05)
            self.advance_sequence()

        # Sequence 2: Move Arm to Ref Pos Action (0.15, 0.15)
        elif self.current_sequence == Sequence.MOVE_EE_TO_REF_POS:
            self.advance_sequence()

        # Sequence 3 & 7: Spatial Tracking with CLF-CBF
        elif self.current_sequence in [Sequence.MOVE_TO_STICK, Sequence.MOVE_TO_PUCK]:
            self.current_target_pose = self.hockey_stick_pose if self.current_sequence == Sequence.MOVE_TO_STICK else self.puck_pose
            if self.current_target_pose is None:
                self.get_logger().warn(f"Sequence {self.current_sequence.name}: Awaiting target data...", throttle_duration_sec=2.0)
                return

            cmd = Twist()
            v, w = self.nid_to_move_robot()

            # nid_to_move_robot() left the geometry it solved in _nav_sample; pair it
            # with the (v, w) it returned so every logged row is one control tick.
            if self._nav_sample is not None:
                self.recorder.sample(v=v, w=w, **self._nav_sample)

            if v == 0.0 and w == 0.0 and (
                (self.current_sequence == Sequence.MOVE_TO_STICK and self.seq1_completed) or
                (self.current_sequence == Sequence.MOVE_TO_PUCK and self.seq4_completed)
            ):
                self.pub_cmd_vel.publish(cmd)
                if self._nav_sample is not None:
                    # err here is the tolerance the sequence was actually marked
                    # complete at - the per-trial figure the convergence table reports.
                    s = self._nav_sample
                    self.recorder.mark(s['sequence'], s['stage'], 'complete', s['err'],
                                       err_stage=s['err_stage'], x=s['x'], y=s['y'],
                                       target_x=s['target_x'], target_y=s['target_y'])
                    self._nav_sample = None
                self.get_logger().info(f"Sequence {self.current_sequence.name} completed!")
                if (self.pass_to_robot):
                    self.publish_ready_to_pass_puck(True)
                else:
                    self.publish_ready_to_receive_puck(True)
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
                self.move_arm_using_publisher(0.2, 0.2)

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
                # Move Look ahead to the stick
                self.l += 0.06
                self.tolerace = 0.05
                self.state_start_time = None

        # Sequence 8: Lower Stick
        elif self.current_sequence == Sequence.LOWER_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("Sequence 8: Dispatching arm LOWER command (waiting 2s)...")
                self.move_arm_using_publisher(0.2, -0.08)

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

        # Sequence 10: Move to Wait Position
        elif self.current_sequence == Sequence.MOVE_TO_WAIT:
            l = self.l
            tolerance = self.tolerace
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
            if self.wait_stage == 0:
                bearing = np.arctan2(wait_y - y, wait_x - x)
                angle_error = np.arctan2(np.sin(bearing - theta), np.cos(bearing - theta))
                if abs(angle_error) > 0.02:
                    cmd.angular.z = float(Kp_w * angle_error)
                else:
                    self.wait_stage = 1
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 10 - Stage 0] Heading aligned to waiting point. Advancing to Stage 1.")
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

        # Sequence 11: Wait For Pass
        elif self.current_sequence == Sequence.WAIT_FOR_PASS:
            self.pub_cmd_vel.publish(Twist())
            if self.puck_pose is None or self._initial_puck_pos is None:
                return
            px, py = self.puck_pose.position.x, self.puck_pose.position.y
            displacement = math.hypot(px - self._initial_puck_pos[0], py - self._initial_puck_pos[1])
            dist_to_me = math.hypot(px - self.robot_pose.position.x, py - self.robot_pose.position.y)
            speed_ok = self.puck_speed < 0.15
            if displacement > 0.3 and dist_to_me <= self.get_parameter('wait_radius').value and speed_ok:
                self.get_logger().info(f"[Seq 11] Pass received: puck at ({px:.2f}, {py:.2f}), {dist_to_me:.2f} m away. Moving to shoot.")
                self.advance_sequence()
            else:
                self.get_logger().info(
                    f"[Seq 11] Waiting for pass (moved {displacement:.2f} m, dist {dist_to_me:.2f} m, speed {self.puck_speed:.2f} m/s)...",
                    throttle_duration_sec=2.0)

        # Sequence 12: Align Hit
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
                cmd.angular.z = float(np.clip(Kp_w * angle_error, -0.6, 0.6))
                self.pub_cmd_vel.publish(cmd)
            else:
                self.pub_cmd_vel.publish(Twist())
                self.spin_accum = 0.0
                self.get_logger().info("[Seq 12] Stick wound up (pointing away from puck). Starting swing.")
                self.advance_sequence()

        # Sequence 13: Spin Hit
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

        # Sequence 16: Shoot Setup - park on the line through the puck that is perpendicular
        # to the shooting direction, so the straight-in approach of Sequence 7 leaves the
        # chassis square to the shot and the swing tangent pointing at the goal.
        elif self.current_sequence == Sequence.SHOOT_SETUP:
            geom = self.compute_shot_geometry()
            if geom is None:
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().warn("[Seq 16] Awaiting puck / goal mocap...", throttle_duration_sec=2.0)
                return

            Kp_v = self.get_parameter('kp_v').value
            Kp_w = self.get_parameter('kp_w').value
            v_max = self.get_parameter('v_max').value
            l = self.get_parameter('l').value
            tolerance = max(self.get_parameter('tolerance').value, 0.10)
            D = self.get_parameter('shot_standoff').value

            px, py = geom['puck']
            u = geom['u']
            n = np.array([-u[1], u[0]])  # left normal of the shot line
            x = self.robot_pose.position.x
            y = self.robot_pose.position.y
            theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)

            if self.shot_side is None:
                d_plus = math.hypot(px + D * n[0] - x, py + D * n[1] - y)
                d_minus = math.hypot(px - D * n[0] - x, py - D * n[1] - y)
                self.shot_side = 1.0 if d_plus <= d_minus else -1.0
                self.get_logger().info(
                    f"[Seq 16] Setting up {D:.2f} m off the puck on side {self.shot_side:+.0f} of the "
                    f"shot line to {geom['label']} (shot heading {geom['shot_heading'] * 180.0 / math.pi:+.1f} deg).")
            wx = px + self.shot_side * D * n[0]
            wy = py + self.shot_side * D * n[1]

            cmd = Twist()
            if self.shot_setup_stage == 0:
                bearing = np.arctan2(wy - y, wx - x)
                angle_error = self.wrap_angle(bearing - theta)
                if abs(angle_error) > 0.02:
                    cmd.angular.z = float(Kp_w * angle_error)
                else:
                    self.shot_setup_stage = 1
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 16 - Stage 0] Heading aligned to the shot-line standoff. Advancing to Stage 1.")
            else:
                # The CHASSIS (not the look-ahead point) has to end up on the shot line,
                # because Sequence 7 then rotates in place and drives straight in.
                e_x, e_y = wx - x, wy - y
                dist = math.hypot(e_x, e_y)
                if dist <= tolerance:
                    self.pub_cmd_vel.publish(Twist())
                    self.get_logger().info("[Seq 16 - Stage 1] On the shot line. Handing over to the puck approach.")
                    self.advance_sequence()
                    self.state_start_time = None
                    return

                p_xl = x + l * math.cos(theta)
                p_yl = y + l * math.sin(theta)
                p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y
                p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                if p_dot_norm > v_max:
                    p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                    p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max
                # Asking the look-ahead point for the same displacement moves the chassis to (wx, wy).
                p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, p_xl + e_x, p_yl + e_y)
                self.L_inv[1, 1] = 1.0 / l
                control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                cmd.linear.x = float(control_inputs[0, 0])
                cmd.angular.z = float(control_inputs[1, 0])
                self.get_logger().info(
                    f"Sequence SHOOT_SETUP: dist={dist:.2f}, v={cmd.linear.x:.3f}, w={cmd.angular.z:.3f}",
                    throttle_duration_sec=1.0)
            self.pub_cmd_vel.publish(cmd)

        # Sequence 17: Aim Shot - freeze the shooting direction vector and the swing plan.
        # The blade tip is sitting on the puck right now, so the CURRENT yaw is by
        # definition the contact yaw: the puck-robot-origin line the swing must cross.
        elif self.current_sequence == Sequence.AIM_SHOT:
            self.pub_cmd_vel.publish(Twist())
            geom = self.compute_shot_geometry()
            if geom is None:
                self.get_logger().warn("[Seq 17] Awaiting puck / goal mocap before aiming...", throttle_duration_sec=2.0)
                return

            deg = 180.0 / math.pi
            undershoot = self.get_parameter('undershoot_w').value
            overshoot = self.get_parameter('overshoot_w').value
            spin = geom['spin_sign']
            # Contact is where the blade lines up with the puck, measured, not assumed from
            # the parked yaw - so a slightly crooked finish to the approach still swings true.
            theta_contact = geom['bearing']
            park_err = self.wrap_angle(geom['bearing'] - geom['theta'])

            self.shot_plan = {
                'spin_sign': spin,
                'undershoot': undershoot,
                'overshoot': overshoot,
                'sweep': undershoot + overshoot,
                'theta_contact': theta_contact,
                'theta_windup': self.wrap_angle(theta_contact - spin * undershoot),
                'theta_end': self.wrap_angle(theta_contact + spin * overshoot),
                'shot_heading': geom['shot_heading'],
                'tangent_heading': geom['tangent_heading'],
                'aim_error': geom['aim_error'],
                'radius': geom['radius'],
                'label': geom['label'],
                'target': geom['target'],
            }
            self._shot_puck_pos = geom['puck']

            self.get_logger().info(
                f"[Seq 17] Target {geom['label']} at ({geom['target'][0]:.2f}, {geom['target'][1]:.2f}), "
                f"{geom['shot_dist']:.2f} m from the puck: shooting direction {geom['shot_heading'] * deg:+.1f} deg.")
            self.get_logger().info(
                f"[Seq 17] Swing radius (chassis->puck) {geom['radius']:.2f} m, spin "
                f"{'CCW' if spin > 0 else 'CW'}, predicted puck heading {geom['tangent_heading'] * deg:+.1f} deg "
                f"-> aim error {geom['aim_error'] * deg:+.1f} deg.")
            self.get_logger().info(
                f"[Seq 17] The blade meets the puck wherever it sits {geom['radius']:.2f} m out from the chassis "
                f"- set --l to that point on the blade. Puck is {park_err * deg:+.1f} deg off the parked heading.")
            self.get_logger().info(
                f"[Seq 17] Plan: wind up {undershoot * deg:.0f} deg to yaw {self.shot_plan['theta_windup'] * deg:+.1f}, "
                f"then sweep {self.shot_plan['sweep'] * deg:.0f} deg through contact yaw {theta_contact * deg:+.1f} "
                f"and {overshoot * deg:.0f} deg past it to {self.shot_plan['theta_end'] * deg:+.1f} deg.")

            if abs(geom['aim_error']) > self.get_parameter('max_aim_error').value:
                self.get_logger().error(
                    f"[Seq 17] Aim error {geom['aim_error'] * deg:+.1f} deg is over the limit: the chassis is not "
                    f"square to the shot line, so the puck will leave along {geom['tangent_heading'] * deg:+.1f} deg. "
                    f"Re-run with --shot_setup to approach on the perpendicular.")
                if self.strict_aim:
                    self.get_logger().error("[Seq 17] --strict_aim is set: holding here instead of swinging.")
                    return

            self.windup_stage = 0
            self.state_start_time = None
            self.advance_sequence()

        # Sequence 18: Wind Up - lift the blade over the puck, rotate back by undershoot_w
        # (so the wind-up never drags the puck backwards), set the blade down, settle.
        elif self.current_sequence == Sequence.WIND_UP:
            if self.shot_plan is None:
                self.get_logger().error("[Seq 18] No shot plan; going back to Sequence 17.")
                self.current_sequence = Sequence.AIM_SHOT
                return

            settle = self.get_parameter('arm_settle_time').value
            theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)
            cmd = Twist()

            if self.windup_stage == 0:
                if not self.windup_lift:
                    self.windup_stage = 1
                elif self.state_start_time is None:
                    self.state_start_time = now
                    self.move_arm_using_publisher(0.2, self.get_parameter('windup_lift_z').value)
                    self.get_logger().info("[Seq 18 - Stage 0] Lifting the blade clear of the puck for the wind-up...")
                elif (now - self.state_start_time).nanoseconds / 1e9 >= settle:
                    self.windup_stage = 1
                    self.state_start_time = None
                self.pub_cmd_vel.publish(cmd)
                return

            if self.windup_stage == 1:
                if self.state_start_time is None:
                    self.state_start_time = now
                err = self.wrap_angle(self.shot_plan['theta_windup'] - theta)
                timed_out = (now - self.state_start_time).nanoseconds / 1e9 > 12.0
                if timed_out:
                    self.get_logger().warn(
                        f"[Seq 18 - Stage 1] Wind-up timed out {err * 180.0 / math.pi:+.1f} deg short; "
                        f"swinging from here (contact angle is unchanged, only the run-up is shorter).")
                if abs(err) > 0.03 and not timed_out:
                    w_max = abs(self.get_parameter('windup_speed').value)
                    Kp_w = self.get_parameter('kp_w').value
                    w_cmd = float(np.clip(2.0 * Kp_w * err, -w_max, w_max))
                    # Keep the chassis out of its own motor deadband, or the last degree never closes.
                    if abs(w_cmd) < 0.15:
                        w_cmd = math.copysign(0.15, err)
                    cmd.angular.z = w_cmd
                    self.get_logger().info(
                        f"[Seq 18 - Stage 1] Winding up: {err * 180.0 / math.pi:+.1f} deg to go.",
                        throttle_duration_sec=1.0)
                else:
                    self.windup_stage = 2
                    self.state_start_time = None
                    self.get_logger().info("[Seq 18 - Stage 1] Wind-up angle reached.")
                self.pub_cmd_vel.publish(cmd)
                return

            if self.windup_stage == 2:
                if not self.windup_lift:
                    self.windup_stage = 3
                elif self.state_start_time is None:
                    self.state_start_time = now
                    self.move_arm_using_publisher(0.2, self.get_parameter('swing_arm_z').value)
                    self.get_logger().info("[Seq 18 - Stage 2] Blade back down on the ground...")
                elif (now - self.state_start_time).nanoseconds / 1e9 >= settle:
                    self.windup_stage = 3
                    self.state_start_time = None
                self.pub_cmd_vel.publish(cmd)
                return

            # Stage 3: stand still so the chassis stops rocking before the open-loop swing.
            self.pub_cmd_vel.publish(cmd)
            if self.state_start_time is None:
                self.state_start_time = now
            elif (now - self.state_start_time).nanoseconds / 1e9 >= self.get_parameter('pre_swing_pause').value:
                self.get_logger().info(
                    f"[Seq 18] Wound up at yaw {theta * 180.0 / math.pi:+.1f} deg "
                    f"(planned {self.shot_plan['theta_windup'] * 180.0 / math.pi:+.1f}). Swinging.")
                self.state_start_time = None
                self.swing_started = False
                self.advance_sequence()

        # Sequence 19: Swing - pure open-loop sweep of undershoot_w + overshoot_w.
        # Contact happens once undershoot_w has been swept; the rest is follow-through.
        elif self.current_sequence == Sequence.SWING:
            if self.shot_plan is None:
                self.get_logger().error("[Seq 19] No shot plan; going back to Sequence 17.")
                self.current_sequence = Sequence.AIM_SHOT
                return

            theta = self.get_yaw_from_quaternion(self.robot_pose.orientation)
            deg = 180.0 / math.pi

            if not self.swing_started:
                spin_speed = abs(self.get_parameter('swing_speed').value)
                sweep = self.shot_plan['sweep']
                self._swing_omega = float(self.shot_plan['spin_sign'] * spin_speed)
                self._swing_duration = sweep / spin_speed if spin_speed > 1e-3 else 0.0
                self.swing_started = True
                self.swing_yaw_prev = theta
                self.swing_yaw_accum = 0.0
                self.state_start_time = now
                self._start_swing_timer()
                self.get_logger().info(
                    f"[Seq 19] Open-loop swing: w={self._swing_omega:+.2f} rad/s for {self._swing_duration:.3f} s "
                    f"= {sweep * deg:.0f} deg ({self.shot_plan['undershoot'] * deg:.0f} undershoot to contact "
                    f"+ {self.shot_plan['overshoot'] * deg:.0f} overshoot).")
                return

            # The high-rate swing timer owns cmd_vel until the sweep is finished.
            self.swing_yaw_accum += abs(self.wrap_angle(theta - self.swing_yaw_prev))
            self.swing_yaw_prev = theta
            if not self._swing_done:
                return

            self.pub_cmd_vel.publish(Twist())
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= self._swing_duration + self.get_parameter('swing_stop_time').value:
                self.get_logger().info(
                    f"[Seq 19] Swing finished: commanded {self.shot_plan['sweep'] * deg:.0f} deg, mocap measured "
                    f"{self.swing_yaw_accum * deg:.0f} deg incl. coast (final yaw {theta * deg:+.1f} deg vs "
                    f"planned {self.shot_plan['theta_end'] * deg:+.1f} deg).")
                self.state_start_time = None
                self.advance_sequence()

        # Sequence 20: Shoot Done - hold still, then report where the puck actually went.
        elif self.current_sequence == Sequence.SHOOT_DONE:
            self.pub_cmd_vel.publish(Twist())
            if self.state_start_time is None:
                self.state_start_time = now
                return
            if (now - self.state_start_time).nanoseconds / 1e9 < self.get_parameter('report_delay').value:
                return

            deg = 180.0 / math.pi
            plan = self.shot_plan or {}
            if self.puck_pose is not None and self._shot_puck_pos is not None:
                dx = self.puck_pose.position.x - self._shot_puck_pos[0]
                dy = self.puck_pose.position.y - self._shot_puck_pos[1]
                travel = math.hypot(dx, dy)
                if travel > 0.05:
                    actual = math.atan2(dy, dx)
                    miss = self.wrap_angle(actual - plan.get('shot_heading', actual))
                    self.get_logger().info(
                        f"[Seq 20] Puck travelled {travel:.2f} m on heading {actual * deg:+.1f} deg vs desired "
                        f"{plan.get('shot_heading', 0.0) * deg:+.1f} deg -> miss angle {miss * deg:+.1f} deg.")
                else:
                    self.get_logger().warn(
                        f"[Seq 20] Puck only moved {travel:.2f} m - the blade probably passed over or behind it. "
                        f"Check --undershoot_w / --overshoot_w and the blade height.")
            self.get_logger().info(f"[Seq 20] Shot at {plan.get('label', 'the goal')} complete.")
            self.state_start_time = None
            self.advance_sequence()

        # Sequence 14: Hit Done
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
            offset_stick = np.array([[self.get_parameter('vertical_offset').value],
                                     [self.get_parameter('sideways_offset').value]])
            offset_world = self.get_rotation_matrix(target_theta) @ offset_stick
            target_x = p_xg + float(offset_world[0, 0])
            target_y = p_yg + float(offset_world[1, 0])

            approach_theta = target_theta

            valid_standoff_dist, standoff_x, standoff_y = self.get_valid_standoff_distance(
                target_x, target_y, approach_theta, standoff_dist
            )

            # Convergence telemetry: err tracks the end effector against the pickup
            # point itself, so it is comparable across all four stages; err_stage
            # tracks whichever waypoint this stage is actually driving at.
            wp_x, wp_y = (standoff_x, standoff_y) if self.seq1_stage in (0, 1) else (target_x, target_y)
            self._nav_sample = {
                'sequence': Sequence.MOVE_TO_STICK.name, 'stage': self.seq1_stage,
                'x': x, 'y': y, 'theta': theta, 'p_xl': p_xl, 'p_yl': p_yl,
                'target_x': target_x, 'target_y': target_y,
                'err': float(np.hypot(target_x - p_xl, target_y - p_yl)),
                'err_stage': float(np.hypot(wp_x - p_xl, wp_y - p_yl))}

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
                flipped_target_theta = np.arctan2(np.sin(approach_theta + np.pi), np.cos(approach_theta + np.pi))
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
                self.get_logger().info(f"Distance to stick: {dist}")
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
                # Park farther from the puck by puck_contact_offset so the swept tip grazes the
                # puck EDGE instead of driving through its center (0 -> through center, as before)
                swing_offset = 0.2
                park_dist = swing_offset + self.get_parameter('puck_contact_offset').value
                park_dist = 0
                if self.hit_side is None:
                    c_plus = np.array([p_xg, p_yg]) + park_dist * normal
                    c_minus = np.array([p_xg, p_yg]) - park_dist * normal
                    d_plus = np.hypot(c_plus[0] - x, c_plus[1] - y)
                    d_minus = np.hypot(c_minus[0] - x, c_minus[1] - y)
                    self.hit_side = 1.0 if d_plus <= d_minus else -1.0
                    self.get_logger().info(f"Hit mode: swing side {self.hit_side:+.0f}, aim point ({aim[0]:.2f}, {aim[1]:.2f})")
                p_xg = p_xg + self.hit_side * park_dist * normal[0]
                p_yg = p_yg + self.hit_side * park_dist * normal[1]

            # Convergence telemetry (single-waypoint sequence, so err_stage == err).
            err_puck = float(np.hypot(p_xg - p_xl, p_yg - p_yl))
            self._nav_sample = {
                'sequence': Sequence.MOVE_TO_PUCK.name, 'stage': self.seq4_stage,
                'x': x, 'y': y, 'theta': theta, 'p_xl': p_xl, 'p_yl': p_yl,
                'target_x': p_xg, 'target_y': p_yg, 'err': err_puck, 'err_stage': err_puck}

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
                self.get_logger().info(f"Distance to Puck Target: {distance_to_target}")
                if distance_to_target <= 0.08:
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
                grip_msg.data = not open
                self.pub_gripper_sim.publish(grip_msg)
            self.get_logger().info(f"Mock/sim mode active: {'Opening' if open else 'Closing'} gripper simulated.")
            self.gripper_action_running = False
            self.state_start_time = None
            self.advance_sequence()
            return
        self.get_logger().info("Gripper Operation running...") 
        goal = GripperControl.Goal()
        goal.target_state = 1 if open else 2
        goal.power = 1.0 if not open else 0.5
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
        cmd = Point()
        cmd.x = 0.0
        cmd.z = 0.15 * direction
        cmd.y = 0.0
        self.pub_cmd_arm.publish(cmd)

    def move_arm_using_publisher(self, x, z):
        if self.mock_mode or self.sim_mode:
            self.get_logger().info(f"Mock/sim mode active: Arm move to ({x}, {z}) simulated.")
            return
        
        cmd = Point()
        cmd.x = float(x)
        cmd.y = 0.0
        cmd.z = float(z)
        
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

    def publish_ready_to_receive_puck(self, ready=True):
        msg = Bool()
        msg.data = ready
        self.pub_ready_to_receive_puck.publish(msg)
        self.get_logger().info(f"Published ready_to_receive_pass: {ready}")

    def publish_ready_to_pass_puck(self, ready=True):
        msg = Bool()
        msg.data = ready
        self.pub_ready_to_pass_puck.publish(msg)
        self.get_logger().info(f"Published ready_to_pass_puck: {ready}")

def main(args=None):
    parser = argparse.ArgumentParser(description='Move Robot Node with CLF-CBF Obstacle Avoidance')
    parser.add_argument('--robot_id', type=int, required=True, help='ID of the robot to control')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of ally robot to pass to (0 for goal)')
    parser.add_argument('--hockey_stick_id', type=int, default=1, help='ID tag integer for the hockey stick VRPN tracking topic')
    parser.add_argument('--puck_color', type=str, default='blue', help='Color tag string for the puck VRPN tracking topic')
    parser.add_argument('--mock_mode', action='store_true', help='Enable mock mode for testing without real VRPN data')
    parser.add_argument('--sim_mode', action='store_true', help='Fake gripper/arm actions but use real /vrpn_mocap topics')
    parser.add_argument('--orient_to_stick', action='store_true', help='Enable terminal angle orientation alignment for the hockey stick')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help="Sideways offset for hockey stick pose")
    parser.add_argument('--vertical_offset', type=float, default=0.0, help="Vertical offset for hockey stick pose")
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along the vector field line')
    parser.add_argument('--r_safety', type=float, default=0.35, help='Safety radius for obstacle avoidance')
    parser.add_argument('--l', type=float, default=0.50, help='Look-ahead center to end-effector displacement distance')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Target proximity threshold radius')
    parser.add_argument('--hit_mode', action='store_true', help='Pass/shoot by spinning the carried stick into the puck')
    parser.add_argument('--wait_for_pass', action='store_true', help='Shooter role: park at the goal standoff and wait for the pass')
    parser.add_argument('--swing_offset', type=float, default=0.55, help='Perpendicular park distance from the puck when preparing a hit')
    parser.add_argument('--puck_contact_offset', type=float, default=0.0,
                        help='Distance (m) the stick tip passes from the puck CENTER at contact; 0 = through the center, set to the puck radius (~0.03-0.05) to strike the edge instead')
    parser.add_argument('--wait_radius', type=float, default=3.0, help='Puck arriving within this range of the shooter triggers the shot phase')
    parser.add_argument('--hit_spin_speed', type=float, default=4.0, help='Angular speed (rad/s) of the hit swing')
    parser.add_argument('--hit_swing_angle', type=float, default=4.71, help='Total swing sweep angle (rad)')
    parser.add_argument('--goal_x', type=float, default=0.0, help='Goal mouth center x (m)')
    parser.add_argument('--goal_y', type=float, default=-1.75, help='Goal mouth center y (m)')
    parser.add_argument('--goal_yaw_deg', type=float, default=90.0, help='Goal facing direction (deg)')
    # --- Open-loop goal shooting (Sequences 16-20) ---
    parser.add_argument('--shoot', action='store_true',
                        help='After reaching the puck, aim at the mocap goal and swing the stick open loop')
    parser.add_argument('--goal_id', type=int, default=1, choices=[1, 2],
                        help='Which mocap gate to shoot at: hockey_goal_1 or hockey_goal_2')
    parser.add_argument('--aim_target', type=str, default='goal', choices=['goal', 'ally'],
                        help='goal: shoot at hockey_goal_<id>; ally: shoot at --pass_to_robot instead')
    parser.add_argument('--undershoot_w', type=float, default=60.0,
                        help='DEGREES of wind-up BEFORE the puck-robot-origin line (the swing starts here)')
    parser.add_argument('--overshoot_w', type=float, default=45.0,
                        help='DEGREES of follow-through PAST the puck-robot-origin line (the swing ends here)')
    parser.add_argument('--swing_speed', type=float, default=6.0, help='Open-loop swing angular speed (rad/s)')
    parser.add_argument('--swing_rate', type=float, default=100.0,
                        help='Publish rate (Hz) of the dedicated swing timer; keeps the stop instant crisp')
    parser.add_argument('--windup_speed', type=float, default=1.5, help='Angular speed cap while winding up (rad/s)')
    parser.add_argument('--shot_setup', action='store_true',
                        help='Before approaching the puck, stand on the line through the puck perpendicular to the '
                             'shot, so the swing tangent actually points at the goal (Sequence 16)')
    parser.add_argument('--shot_standoff', type=float, default=1.2,
                        help='Distance (m) from the puck of the Sequence 16 setup point')
    parser.add_argument('--no_windup_lift', action='store_true',
                        help='Do not lift the blade while winding up (default lifts so the wind-up cannot drag the puck)')
    parser.add_argument('--lower_before_aim', action='store_true',
                        help='Put the blade down on arrival, before aiming. Off by default in shoot mode: the blade '
                             'straddles the puck when parked, so lowering it there shoves the puck. Left raised, the '
                             'wind-up sets it down only once it has rotated clear')
    parser.add_argument('--windup_lift_z', type=float, default=0.12, help='Arm z while winding up (m)')
    parser.add_argument('--swing_arm_z', type=float, default=-0.08, help='Arm z for the swing itself (m)')
    parser.add_argument('--arm_settle_time', type=float, default=1.5, help='Seconds allowed for each arm move (s)')
    parser.add_argument('--pre_swing_pause', type=float, default=0.5, help='Hold still this long before swinging (s)')
    parser.add_argument('--swing_stop_time', type=float, default=0.5, help='Braking hold after the sweep (s)')
    parser.add_argument('--report_delay', type=float, default=2.0,
                        help='Seconds to watch the puck after the swing before reporting the miss angle')
    parser.add_argument('--max_aim_error', type=float, default=25.0,
                        help='DEGREES of aim error (predicted vs desired puck heading) that triggers a loud warning')
    parser.add_argument('--strict_aim', action='store_true',
                        help='Refuse to swing when the aim error exceeds --max_aim_error')
    # --- Navigation convergence recording (see sim/analyze_convergence.py) ---
    parser.add_argument('--record', action='store_true',
                        help='Log per-tick navigation error to CSV for the convergence plot and table')
    parser.add_argument('--log_dir', type=str, default='records',
                        help='Directory the --record CSVs are written to')
    parser.add_argument('--trial_id', type=str, default='1',
                        help='Trial label; becomes the trial_<id>_*.csv filename stem')
    parser.add_argument('--start_sequence', type=int, default=0 )
    args, remaining = parser.parse_known_args(args)
    if args.mock_mode and args.sim_mode:
        parser.error('--mock_mode and --sim_mode are mutually exclusive')
    if args.shoot and args.hit_mode:
        parser.error('--shoot and --hit_mode are mutually exclusive (both own the post-puck sequences)')
    if args.swing_speed <= 0.0:
        parser.error('--swing_speed must be positive')
    if args.undershoot_w < 0.0 or args.overshoot_w < 0.0:
        parser.error('--undershoot_w and --overshoot_w are magnitudes in degrees and must be >= 0')
    if args.shoot and args.shot_setup and args.shot_standoff <= args.l:
        parser.error(f'--shot_standoff must exceed --l ({args.l}) so there is room for the straight-in approach')
    valid_starts = [s.value for s in Sequence]
    if args.start_sequence not in valid_starts:
        parser.error('--start_sequence must be one of: ' +
                     ', '.join(f'{s.value}={s.name}' for s in Sequence))
    if args.shoot and args.start_sequence == Sequence.SHOOT_SETUP.value and not args.shot_setup:
        parser.error('--start_sequence 16 (SHOOT_SETUP) also needs --shot_setup, which puts it on the route')

    typo_flags = [tok for tok in remaining if tok != '--ros-args' and tok.startswith('--')]
    if typo_flags:
        parser.error(f"unrecognized arguments: {' '.join(typo_flags)}")

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
        puck_contact_offset=args.puck_contact_offset,
        wait_radius=args.wait_radius,
        hit_spin_speed=args.hit_spin_speed,
        hit_swing_angle=args.hit_swing_angle,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_yaw_deg=args.goal_yaw_deg,
        shoot=args.shoot,
        goal_id=args.goal_id,
        aim_target=args.aim_target,
        undershoot_w_deg=args.undershoot_w,
        overshoot_w_deg=args.overshoot_w,
        swing_speed=args.swing_speed,
        swing_rate=args.swing_rate,
        windup_speed=args.windup_speed,
        shot_setup=args.shot_setup,
        shot_standoff=args.shot_standoff,
        windup_lift=not args.no_windup_lift,
        lower_before_aim=args.lower_before_aim,
        windup_lift_z=args.windup_lift_z,
        swing_arm_z=args.swing_arm_z,
        arm_settle_time=args.arm_settle_time,
        pre_swing_pause=args.pre_swing_pause,
        swing_stop_time=args.swing_stop_time,
        report_delay=args.report_delay,
        max_aim_error_deg=args.max_aim_error,
        strict_aim=args.strict_aim,
        record=args.record,
        log_dir=args.log_dir,
        trial_id=args.trial_id,
        start_sequence=args.start_sequence
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.recorder.close()
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()