import argparse
import math
import numpy as np
import rclpy
from enum import Enum
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from turtlesim.msg import Pose

"""
Following Sequences:
0: OPEN_GRIPPER (Mock print 2s)
1: MOVE_EE_TO_ORIGIN (Mock print 2s)
2: MOVE_EE_TO_REF_POS (Mock print 2s)
3: MOVE_TO_STICK (3-stage standoff approach)
4: CLOSE_GRIPPER (Mock print 2s)
5: LIFT_STICK (Mock print 2s)
6: MOVE_BACK_ROTATE (Timed reverse 5s)
7: MOVE_TO_PUCK (2-stage approach)
8: LOWER_STICK (Mock print 2s)
9: RELEASE_PUCK (Mock print 2s)
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

class TurtlesimSequenceController(Node):
    def __init__(self, pass_to_robot, l_default=0.15, tolerance_default=0.15, standoff_distance=2.5, sideways_offset=0.0, vertical_offset=0.0):
        super().__init__('turtlesim_sequence_controller')
        self.pass_to_robot = pass_to_robot

        # Pose Storage structures
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None

        self.current_target_pose = None
        self.state_start_time = None
        
        # Internal sub-stages tracker for Sequence.MOVE_TO_STICK
        self.seq1_stage = 0 
        self.seq1_completed = False 

        # Internal sub-stages tracker for Sequence.MOVE_TO_PUCK
        self.seq4_stage = 0
        self.seq4_completed = False

        # Proportional controller tunings & configurable offsets
        self.declare_parameter('kp_v', 0.5)
        self.declare_parameter('kp_w', 1.7) 
        self.declare_parameter('l', l_default)
        self.declare_parameter('tolerance', tolerance_default)
        self.declare_parameter('standoff_distance', standoff_distance)
        self.declare_parameter('sideways_offset', sideways_offset)
        self.declare_parameter('vertical_offset', vertical_offset)
        self.declare_parameter('start_sequence', 0)
        
        # Safely cast parameter to Enum member
        self.current_sequence = Sequence(self.get_parameter('start_sequence').value)

        self.L_inv = np.array([[1, 0], [0, 1 / self.get_parameter('l').value]])
        
        self.timer = self.create_timer(0.1, self.control_loop)
        qos_pose = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Mocap Stream Subscriptions
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, 10)
        self.create_subscription(PoseStamped, '/vrpn_mocap/puck_1/pose', self.puck_pos_callback, 10)
        
        self.create_subscription(Pose, '/turtle1/pose', self.turtlesim_pose_callback, qos_pose)
        self.pub_cmd_vel = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.get_logger().info(f"Turtlesim Sequence State Machine booted up at state: {self.current_sequence.name}")

    def advance_sequence(self):
        """Advances state machine to the next Sequence enum member."""
        next_val = self.current_sequence.value + 1
        if next_val in [s.value for s in Sequence]:
            self.current_sequence = Sequence(next_val)
        else:
            self.current_sequence = None  # Reached end of sequence pipeline

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

        now = self.get_clock().now()

        # --- SEQUENCE 0: OPEN GRIPPER (MOCK PRINT FOR 2s) ---
        if self.current_sequence == Sequence.OPEN_GRIPPER:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 0: Opening gripper mechanism (2s)...")
            
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[ACTION] Gripper opening... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("[ACTION] Gripper open complete! Advancing to MOVE_EE_TO_ORIGIN.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 1: MOVE EE TO ORIGIN (MOCK PRINT FOR 2s) ---
        elif self.current_sequence == Sequence.MOVE_EE_TO_ORIGIN:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 1: Moving arm EE to origin (0.0, 0.0) (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[ACTION] Arm homing to origin... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("[ACTION] Arm origin pose reached! Advancing to MOVE_EE_TO_REF_POS.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 2: MOVE EE TO REF POS (MOCK PRINT FOR 2s) ---
        elif self.current_sequence == Sequence.MOVE_EE_TO_REF_POS:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 2: Moving arm EE to reference position (0.15, 0.15) (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[ACTION] Arm moving to reference... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("[ACTION] Arm reference pose reached! Advancing to MOVE_TO_STICK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 3 & 7: SPATIAL TRACKING ---
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

        # --- SEQUENCE 4: CLOSE GRIPPER (MOCK PRINT FOR 2s) ---
        elif self.current_sequence == Sequence.CLOSE_GRIPPER:
            self.pub_cmd_vel.publish(Twist()) # Lock position
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 4: Closing gripper to lock hockey stick tool (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[ACTION] Gripper closing... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("[ACTION] Gripper secure lock established! Advancing to LIFT_STICK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 5: LIFT STICK (MOCK PRINT FOR 2s) ---
        elif self.current_sequence == Sequence.LIFT_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 5: Lifting stick off platform anchor (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[ACTION] Manipulator elevating stick... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("[ACTION] Stick clear of platform ground plane! Advancing to MOVE_BACK_ROTATE.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 6: TIMED SAFETY REVERSE DISPLACEMENT (5s) ---
        elif self.current_sequence == Sequence.MOVE_BACK_ROTATE:
            cmd = Twist()
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 6: Executing reverse safety clearance step (5s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 5.0:
                cmd.linear.x = -0.15 # Reverse velocity
                self.get_logger().info(f"[ACTION] Backing away... {elapsed:.1f}s", throttle_duration_sec=1.0)
                self.pub_cmd_vel.publish(cmd)
            else:
                self.get_logger().info("[ACTION] Reverse clearance buffer set! Advancing to MOVE_TO_PUCK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 8: LOWER STICK TO GROUND (MOCK PRINT FOR 2s) ---
        elif self.current_sequence == Sequence.LOWER_STICK:
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info("[ACTION] Sequence 8: Lowering stick onto field ice surface (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[ACTION] Lowering tool assembly... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("[ACTION] Stick in strike posture on ground level! Advancing to RELEASE_PUCK.")
                self.advance_sequence()
                self.state_start_time = None

        # --- SEQUENCE 9: RELEASE / SHOOT PUCK (MOCK PRINT FOR 2s) ---
        elif self.current_sequence == Sequence.RELEASE_PUCK:
            dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the field goal"
            if self.state_start_time is None:
                self.state_start_time = now
                self.get_logger().info(f"[SHOOT] Sequence 9: Executing kinetic stroke toward {dest} (2s)...")

            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed < 2.0:
                self.get_logger().info(f"[SHOOT] Kinetic force transfer active... {elapsed:.1f}s", throttle_duration_sec=1.0)
            else:
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

        # --- STAGED APPROACH FOR MOVE_TO_STICK ---
        if self.current_sequence == Sequence.MOVE_TO_STICK:
            # Apply static (x, y) translations
            target_x = p_xg + self.get_parameter('vertical_offset').value
            target_y = p_yg + self.get_parameter('sideways_offset').value

            # Calculate standoff point vector
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
                    self.get_logger().info("[Seq 3 - Stage 0] Heading aligned to standoff vector. Advancing to Stage 1.")

            if self.seq1_stage == 1:
                dist = np.sqrt((standoff_x - p_xl)**2 + (standoff_y - p_yl)**2)
                if dist <= tolerance:
                    self.seq1_stage = 2
                    self.get_logger().info("[Seq 3 - Stage 1] Arrived at standoff location. Advancing to Stage 2 (Orientation).")
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
                    self.get_logger().info("[Seq 3 - Stage 2] Alignment complete! Advancing to Stage 3 (Final Move).")

            if self.seq1_stage == 3:
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

        # --- STAGED APPROACH FOR MOVE_TO_PUCK ---
        elif self.current_sequence == Sequence.MOVE_TO_PUCK:
            if self.seq4_stage == 0:
                distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
                if distance_to_target <= tolerance:
                    self.seq4_stage = 1
                    self.get_logger().info("[Seq 7 - Stage 0] Arrived at puck location. Advancing to Stage 1 (Orientation Alignment).")
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
                    self.get_logger().info("[Seq 7 - Stage 1] Puck orientation alignment complete!")
                    return 0.0, 0.0

        return 0.0, 0.0

def main(args=None):
    parser = argparse.ArgumentParser(description='Turtlesim Sequence Controller')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of teammate target')
    parser.add_argument('--l', type=float, default=0.15, help='Look-ahead center distance')
    parser.add_argument('--tolerance', type=float, default=0.15, help='Target proximity threshold radius')
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Linear projection offset along the vector field line')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help="Sideways offset for hockeystick pose")
    parser.add_argument('--vertical_offset', type=float, default=0.0, help="Vertical offset for hockeystick pose")
    
    args, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    
    node = TurtlesimSequenceController(
        pass_to_robot=args.pass_to_robot,
        l_default=args.l,
        tolerance_default=args.tolerance,
        standoff_distance=args.standoff_distance,
        sideways_offset=args.sideways_offset,
        vertical_offset=args.vertical_offset
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