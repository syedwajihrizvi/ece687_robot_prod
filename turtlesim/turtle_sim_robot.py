import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from turtlesim.msg import Pose

class TurtlesimSequenceController(Node):
    def __init__(self, pass_to_robot, l_default=0.15, tolerance_default=0.15, standoff_distance=2.5):
        super().__init__('turtlesim_sequence_controller')
        self.pass_to_robot = pass_to_robot

        # Pose Storage structures
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None

        self.current_target_pose = None
        self.state_start_time = None
        
        # Internal sub-stages tracker for Sequence 1: 
        # 0: Turn to face standoff, 
        # 1: Drive to standoff, 
        # 2: Rotate to target orientation, 
        # 3: Final translation
        self.seq1_stage = 0 
        self.seq1_completed = False 

        # Internal sub-stages tracker for Sequence 4:
        # 0: Drive directly to target, 
        # 1: Align heading angle with target
        self.seq4_stage = 0
        self.seq4_completed = False

        # Proportional controller tunings & configurable offsets
        self.declare_parameter('kp_v', 0.5)
        self.declare_parameter('kp_w', 1.7) 
        self.declare_parameter('l', l_default)
        self.declare_parameter('tolerance', tolerance_default)
        self.declare_parameter('standoff_distance', standoff_distance) # Linear distance extension along same angle
        
        self.current_sequence = 1

        self.L_inv = np.array([[1, 0], [0, 1/self.get_parameter('l').value]])
        
        self.timer = self.create_timer(0.1, self.control_loop)
        qos_pose = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Mocap Stream Subscriptions
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, 10)
        self.create_subscription(PoseStamped, '/vrpn_mocap/puck_1/pose', self.puck_pos_callback, 10)
        
        self.create_subscription(Pose, '/turtle1/pose', self.turtlesim_pose_callback, qos_pose)
        self.pub_cmd_vel = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.get_logger().info("Turtlesim Sequence State Machine fully booted up at sequence: 1")

    def get_rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    def turtlesim_pose_callback(self, msg):
        self.robot_pose = msg

    def hockey_stick_pos_callback(self, msg):
        self.hockey_stick_pose = msg.pose

    def puck_pos_callback(self, msg):
        self.puck_pose = msg.pose

    def control_loop(self):
        if self.robot_pose is None:
            self.get_logger().warn("Waiting for baseline Turtlesim telemetry...", throttle_duration_sec=2.0)
            return
            
        # --- PHASE 1 & 4: SPATIAL TRACKING OVER DYNAMICS MOCKING ---
        if self.current_sequence in [1, 4]:
            self.current_target_pose = self.hockey_stick_pose if self.current_sequence == 1 else self.puck_pose
            if self.current_target_pose is None:
                self.get_logger().warn(f"Sequence {self.current_sequence}: Tracking assets missing from stream...", throttle_duration_sec=2.0)
                return

            cmd = Twist()
            v, w = self.nid_kinematics()

            # Ensure both sequences check their respective staged completion flags
            if v == 0.0 and w == 0.0 and (
                (self.current_sequence == 1 and self.seq1_completed) or 
                (self.current_sequence == 4 and self.seq4_completed)
            ):
                self.pub_cmd_vel.publish(cmd)
                self.get_logger().info(f"=== Sequence {self.current_sequence} Completed! ===")
                self.current_sequence += 1
                self.state_start_time = None
                return
                
            cmd.linear.x = v
            cmd.angular.z = w
            self.pub_cmd_vel.publish(cmd)

        # --- PHASE 2: GRIPPER ACTION ENGAGEMENT ---
        elif self.current_sequence == 2:
            self.pub_cmd_vel.publish(Twist()) 

            if self.state_start_time is None:
                self.state_start_time = self.get_clock().now()
                self.get_logger().info("[ACTION] Gripper engagement active. Processing manipulator tool lock (5s)...")

            elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed >= 5.0:
                self.get_logger().info("Gripper operational seal established! Advancing to Sequence 3.")
                self.current_sequence += 1
                self.state_start_time = None

        # --- PHASE 3: TIMED SAFETY DISPLACEMENT (BACKWARD REVERSE) ---
        elif self.current_sequence == 3:
            cmd = Twist()
            if self.state_start_time is None:
                self.state_start_time = self.get_clock().now()
                self.get_logger().info("[ACTION] Executing timed step safety clearance backwards (5s)...")

            elapsed = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
            if elapsed < 5.0:
                cmd.linear.x = -0.75
                self.pub_cmd_vel.publish(cmd)
            else:
                self.get_logger().info("Reverse clearance buffer established! Advancing to Sequence 4 (Target: Puck).")
                self.current_sequence += 1
                self.state_start_time = None

        # --- PHASE 5: PUCK STRIKE / PASS TRANSITION ---
        elif self.current_sequence == 5:
            dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the field goal"
            self.get_logger().info(f"[SHOOT] Kinetic impulse executed! Projecting puck toward {dest}.")
            self.current_sequence += 1
        else:
            self.get_logger().info("All sequences completed. Robot is now idle.")
            self.pub_cmd_vel.publish(Twist())

    def nid_kinematics(self):
        l = self.get_parameter('l').value
        tolerance = self.get_parameter('tolerance').value
        Kp_v = self.get_parameter('kp_v').value
        Kp_w = self.get_parameter('kp_w').value
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

        v, w = 0.0, 0.0

        # --- UNIVERSAL MANEUVER STAGING FOR SEQUENCE 1 ---
        if self.current_sequence == 1:
            standoff_x = p_xg + standoff_dist * math.cos(target_theta)
            standoff_y = p_yg + standoff_dist * math.sin(target_theta)

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
                    self.get_logger().info("[Seq 1 - Stage 1] Arrived at offset location. Advancing to Stage 2 (Orientation).")
                else:
                    e_x, e_y = standoff_x - p_xl, standoff_y - p_yl
                    p_dot_x, p_dot_y = Kp_v * e_x, Kp_v * e_y
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
                dist = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
                if dist <= tolerance:
                    self.seq1_completed = True 
                    return 0.0, 0.0  
                else:
                    e_x, e_y = p_xg - p_xl, p_yg - p_yl
                    p_dot_x, p_dot_y = Kp_v * e_x, Kp_v * e_y
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])

        # --- STAGED MANEUVER FOR SEQUENCE 4 ---
        elif self.current_sequence == 4:
            if self.seq4_stage == 0:
                # Stage 4.0: Drive directly to the target point
                distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
                if distance_to_target <= tolerance:
                    self.seq4_stage = 1
                    self.get_logger().info("[Seq 4 - Stage 0] Arrived at target point. Advancing to Stage 1 (Orientation Alignment).")
                    return 0.0, 0.0
                else:
                    e_x, e_y = p_xg - p_xl, p_yg - p_yl
                    p_dot_x, p_dot_y = Kp_v * e_x, Kp_v * e_y
                    control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
                    return float(control_inputs[0, 0]), float(control_inputs[1, 0])
            if self.seq4_stage == 1:
                # Stage 4.1: Align angle with the target orientation
                angle_error = np.arctan2(np.sin(target_theta - theta), np.cos(target_theta - theta))
                if abs(angle_error) > 0.02:
                    w = Kp_w * angle_error
                    return 0.0, float(w)
                else:
                    self.seq4_completed = True
                    self.get_logger().info("[Seq 4 - Stage 1] Orientation alignment complete!")
                    return 0.0, 0.0
                    
        return 0.0, 0.0

def main(args=None):
    parser = argparse.ArgumentParser(description='Turtlesim Sequence Controller')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of teammate target')
    parser.add_argument('--l', type=float, default=0.15, help='Look-ahead center distance')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Target proximity threshold radius')
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along the vector field line')
    
    args, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    
    node = TurtlesimSequenceController(
        pass_to_robot=args.pass_to_robot,
        l_default=args.l,
        tolerance_default=args.tolerance,
        standoff_distance=args.standoff_distance
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