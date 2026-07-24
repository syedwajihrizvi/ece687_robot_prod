import argparse
import math
import numpy as np
import rclpy
from enum import Enum
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from turtlesim.msg import Pose
from scipy.optimize import minimize

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

class TurtlesimSequenceController(Node):
    def __init__(self, pass_to_robot, l_default=0.15, tolerance_default=0.20, standoff_distance=2.5, sideways_offset=0.0, vertical_offset=0.0, r_safety=0.35):
        super().__init__('turtlesim_sequence_controller')
        self.pass_to_robot = pass_to_robot

        # Pose Storage structures
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None

        # Obstacle pose storage
        self.obstacle_poses = {}

        # Boundary & Optimization Parameters
        self.declare_parameter('r_safety', r_safety)     
        self.declare_parameter('gamma_cbf', 1.5)         
        self.declare_parameter('gamma_clf', 1.0)         
        self.declare_parameter('clf_penalty', 1e3)       

        self.current_target_pose = None
        self.state_start_time = None

        # Memory variables
        self.chosen_tangent_sign = {}      
        self.filtered_u_p = None           

        # Internal sub-stages trackers
        self.seq1_stage = 0 
        self.seq1_completed = False 

        self.seq4_stage = 0
        self.seq4_completed = False

        # Proportional controller tunings & configurable offsets
        self.declare_parameter('kp_v', 1.2) 
        self.declare_parameter('kp_w', 2.0) 
        self.declare_parameter('v_max', 1.0)  # Maximum workspace velocity cap (m/s)
        self.declare_parameter('l', l_default)
        self.declare_parameter('tolerance', tolerance_default)
        self.declare_parameter('standoff_distance', standoff_distance)
        self.declare_parameter('sideways_offset', sideways_offset)
        self.declare_parameter('vertical_offset', vertical_offset)
        self.declare_parameter('start_sequence', 0)
        
        self.current_sequence = Sequence(self.get_parameter('start_sequence').value)
        self.L_inv = np.array([[1, 0], [0, 1 / self.get_parameter('l').value]])
        
        self.timer = self.create_timer(0.1, self.control_loop)
        qos_pose = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Mocap Stream Subscriptions
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, 10)
        self.create_subscription(PoseStamped, '/vrpn_mocap/puck_1/pose', self.puck_pos_callback, 10)

        # Subscription to obstacles
        for i in range(1, 8):
            topic_name = f'/vrpn_mocap/robot_obstacle_{i}/pose'
            key = f'obstacle_{i}'
            self.create_subscription(PoseStamped, topic_name, self.obstacle_pos_callback(key), 10)
        
        self.create_subscription(Pose, '/turtle1/pose', self.turtlesim_pose_callback, qos_pose)
        self.pub_cmd_vel = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.get_logger().info(f"Turtlesim Sequence State Machine booted up at state: {self.current_sequence.name}")

    def advance_sequence(self):
        """Advances state machine to the next Sequence enum member."""
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

    def turtlesim_pose_callback(self, msg):
        self.robot_pose = msg

    def hockey_stick_pos_callback(self, msg):
        self.hockey_stick_pose = msg.pose

    def obstacle_pos_callback(self, key):
        def callback(msg):
            self.obstacle_poses[key] = msg.pose
        return callback

    def puck_pos_callback(self, msg):
        self.puck_pose = msg.pose

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
                u_nom = u_nom + (1.0 * influence_factor) * tangent

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
            self.get_logger().warn("Waiting for baseline Turtlesim telemetry...", throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()

        # --- SEQUENCE 0: OPEN GRIPPER ---
        if self.current_sequence == Sequence.OPEN_GRIPPER:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 0: Opening gripper mechanism (2s)...")
            
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info("[ACTION] Gripper open complete! Advancing to MOVE_EE_TO_ORIGIN.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 1: MOVE EE TO ORIGIN ---
        elif self.current_sequence == Sequence.MOVE_EE_TO_ORIGIN:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 1: Moving arm EE to origin (0.0, 0.0) (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info("[ACTION] Arm origin pose reached! Advancing to MOVE_EE_TO_REF_POS.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 2: MOVE EE TO REF POS ---
        elif self.current_sequence == Sequence.MOVE_EE_TO_REF_POS:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 2: Moving arm EE to reference position (0.15, 0.15) (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info("[ACTION] Arm reference pose reached! Advancing to MOVE_TO_STICK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 3 & 7: SPATIAL TRACKING WITH OBSTACLE AVOIDANCE ---
        elif self.current_sequence in [Sequence.MOVE_TO_STICK, Sequence.MOVE_TO_PUCK]:
            self.current_target_pose = self.hockey_stick_pose if self.current_sequence == Sequence.MOVE_TO_STICK else self.puck_pose
            if self.current_target_pose is None:
                self.get_logger().warn(f"Sequence {self.current_sequence.name}: Tracking assets missing from stream...", throttle_duration_sec=2.0)
                return

            cmd = Twist()
            v, w = self.nid_kinematics()

            if v == 0.0 and w == 0.0 and (
                (self.current_sequence == Sequence.MOVE_TO_STICK and self.seq1_completed) or
                (self.current_sequence == Sequence.MOVE_TO_PUCK and self.seq4_completed)
            ):
                self.pub_cmd_vel.publish(cmd)
                self.get_logger().info(f"=== Sequence {self.current_sequence.name} Completed! ===")
                self.advance_sequence()
                self.state_start_time = None
                return
                
            cmd.linear.x = v
            cmd.angular.z = w
            self.pub_cmd_vel.publish(cmd)

        # --- SEQUENCE 4: CLOSE GRIPPER ---
        elif self.current_sequence == Sequence.CLOSE_GRIPPER:
            self.pub_cmd_vel.publish(Twist())
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 4: Closing gripper to lock hockey stick tool (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info("[ACTION] Gripper secure lock established! Advancing to LIFT_STICK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 5: LIFT STICK ---
        elif self.current_sequence == Sequence.LIFT_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 5: Lifting stick off platform anchor (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info("[ACTION] Stick clear of platform ground plane! Advancing to MOVE_BACK_ROTATE.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 6: TIMED SAFETY REVERSE DISPLACEMENT ---
        elif self.current_sequence == Sequence.MOVE_BACK_ROTATE:
            cmd = Twist()
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 6: Executing reverse safety clearance step (3s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 3.0:
                cmd.linear.x = -0.2
                self.pub_cmd_vel.publish(cmd)
            else:
                self.pub_cmd_vel.publish(Twist())
                self.get_logger().info("[ACTION] Reverse clearance buffer set! Advancing to MOVE_TO_PUCK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 8: LOWER STICK TO GROUND ---
        elif self.current_sequence == Sequence.LOWER_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 8: Lowering stick onto field ice surface (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info("[ACTION] Stick in strike posture on ground level! Advancing to RELEASE_PUCK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 9: RELEASE / SHOOT PUCK ---
        elif self.current_sequence == Sequence.RELEASE_PUCK:
            dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the field goal"
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info(f"[SHOOT] Sequence 9: Executing kinetic stroke toward {dest} (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 2.0:
                self.get_logger().info(f"[SHOOT] Puck released toward {dest}! Execution sequence finished.")
                self.advance_sequence()
                self.state_start_time = None

        else:
            self.get_logger().info("All sequences completed. Robot is idle.", throttle_duration_sec=3.0)
            self.pub_cmd_vel.publish(Twist())

    def nid_kinematics(self):
        l = self.get_parameter('l').value
        tolerance = self.get_parameter('tolerance').value
        Kp_v = self.get_parameter('kp_v').value
        Kp_w = self.get_parameter('kp_w').value
        v_max = self.get_parameter('v_max').value
        standoff_dist = self.get_parameter('standoff_distance').value
        
        x, y, theta = self.robot_pose.x, self.robot_pose.y, self.robot_pose.theta
        p_xg = self.current_target_pose.position.x
        p_yg = self.current_target_pose.position.y

        siny_cosp = 2 * (self.current_target_pose.orientation.w * self.current_target_pose.orientation.z)
        cosy_cosp = 1 - 2 * (self.current_target_pose.orientation.z ** 2)
        target_theta = math.atan2(siny_cosp, cosy_cosp)
        target_theta = np.arctan2(np.sin(target_theta), np.cos(target_theta))

        p_xl = x + l * math.cos(theta)
        p_yl = y + l * math.sin(theta)

        # --- STAGED APPROACH FOR MOVE_TO_STICK ---
        if self.current_sequence == Sequence.MOVE_TO_STICK:
            target_x = p_xg + self.get_parameter('vertical_offset').value
            target_y = p_yg + self.get_parameter('sideways_offset').value

            valid_standoff_dist, standoff_x, standoff_y = self.get_valid_standoff_distance(
                target_x, target_y, target_theta, standoff_dist
            )

            if self.seq1_stage == 0:
                bearing_to_standoff = np.arctan2(standoff_y - y, standoff_x - x)
                angle_error = np.arctan2(np.sin(bearing_to_standoff - theta), np.cos(bearing_to_standoff - theta))
                
                if abs(angle_error) > 0.02:
                    return 0.0, float(Kp_w * angle_error)
                else:
                    self.seq1_stage = 1
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 3 - Stage 0] Heading aligned to standoff vector. Advancing to Stage 1.")

            elif self.seq1_stage == 1:
                dist = np.sqrt((standoff_x - p_xl)**2 + (standoff_y - p_yl)**2)
                
                if dist <= tolerance:
                    self.seq1_stage = 2
                    self.get_logger().info("[Seq 3 - Stage 1] Arrived at standoff location. Advancing to Stage 2 (Orientation).")
                    return 0.0, 0.0
                else:
                    e_x, e_y = standoff_x - p_xl, standoff_y - p_yl
                    p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y
                    
                    # SATURATE NOMINAL VELOCITY VECTOR TO PREVENT OVER-SATURATION
                    p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                    if p_dot_norm > v_max:
                        p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                        p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max

                    p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, standoff_x, standoff_y)

                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

            elif self.seq1_stage == 2:
                flipped_target_theta = np.arctan2(np.sin(target_theta + np.pi), np.cos(target_theta + np.pi))
                angle_error = np.arctan2(np.sin(flipped_target_theta - theta), np.cos(flipped_target_theta - theta))
                
                if abs(angle_error) > 0.02:
                    return 0.0, float(Kp_w * angle_error)
                else:
                    self.seq1_stage = 3
                    self.filtered_u_p = None
                    self.get_logger().info("[Seq 3 - Stage 2] Alignment complete! Advancing to Stage 3 (Final Move).")

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

        # --- STREAMLINED APPROACH FOR MOVE_TO_PUCK ---
        elif self.current_sequence == Sequence.MOVE_TO_PUCK:
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

            elif self.seq4_stage == 1:
                distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
                if distance_to_target <= tolerance:
                    self.seq4_completed = True
                    self.get_logger().info("[Seq 7 - Stage 1] Arrived at puck location! Sequence complete.")
                    return 0.0, 0.0
                else:
                    e_x, e_y = p_xg - p_xl, p_yg - p_yl
                    p_dot_x_nom, p_dot_y_nom = Kp_v * e_x, Kp_v * e_y

                    # SATURATE NOMINAL VELOCITY VECTOR
                    p_dot_norm = np.hypot(p_dot_x_nom, p_dot_y_nom)
                    if p_dot_norm > v_max:
                        p_dot_x_nom = (p_dot_x_nom / p_dot_norm) * v_max
                        p_dot_y_nom = (p_dot_y_nom / p_dot_norm) * v_max

                    self.get_logger().info(f"[Seq 7 - Stage 1] Driving to puck: distance={distance_to_target:.3f}m, capped_velocity=({p_dot_x_nom:.3f}, {p_dot_y_nom:.3f})", throttle_duration_sec=0.5)

                    p_dot_x, p_dot_y = self.solve_clf_cbf_qp(p_xl, p_yl, p_dot_x_nom, p_dot_y_nom, p_xg, p_yg)

                    self.L_inv[1, 1] = 1.0 / l
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

        return 0.0, 0.0

def main(args=None):
    parser = argparse.ArgumentParser(description='Turtlesim Sequence Controller')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of teammate target')
    parser.add_argument('--l', type=float, default=0.15, help='Look-ahead center distance')
    parser.add_argument('--tolerance', type=float, default=0.20, help='Target proximity threshold radius')
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along vector field line')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help="Sideways offset for hockey stick pose")
    parser.add_argument('--vertical_offset', type=float, default=0.0, help="Vertical offset for hockey stick pose")
    parser.add_argument('--r_safety', type=float, default=0.35, help='Safety radius for obstacle avoidance')
    args, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    
    node = TurtlesimSequenceController(
        pass_to_robot=args.pass_to_robot,
        l_default=args.l,
        tolerance_default=args.tolerance,
        standoff_distance=args.standoff_distance,
        sideways_offset=args.sideways_offset,
        vertical_offset=args.vertical_offset,
        r_safety=args.r_safety
    )
    
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