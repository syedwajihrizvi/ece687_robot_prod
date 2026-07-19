import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from turtlesim.msg import Pose

class TurtlesimSequenceController(Node):
    def __init__(self, pass_to_robot):
        super().__init__('turtlesim_sequence_controller')
        self.pass_to_robot = pass_to_robot

        # Pose Storage structures mapped cleanly for Turtlesim geometry limits
        self.robot_pose = None
        self.hockey_stick_pose = None
        self.puck_pose = None

        self.current_target_pose = None
        self.rotation_phase = False
        self.state_start_time = None

        # Proportional controller tunings
        self.declare_parameter('kp_v', 0.5)
        self.declare_parameter('kp_w', 1.7) # Snappier angular corrections for simulation
        self.declare_parameter('l', 0.15)
        self.declare_parameter('tolerance', 0.15)

        self.L_inv = np.array([[1, 0], [0, 1/self.get_parameter('l').value]])
        
        # Fire loop timer at 10Hz to mimic field environment execution updates
        self.timer = self.create_timer(0.1, self.control_loop)
        
        qos_pose = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Mocap Stream Subscriptions
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, 10)
        self.create_subscription(PoseStamped, '/vrpn_mocap/puck_1/pose', self.puck_pos_callback, 10)
        
        # Connect directly into Turtlesim's master tracking streams
        self.create_subscription(Pose, '/turtle1/pose', self.turtlesim_pose_callback, qos_pose)
        self.pub_cmd_vel = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        self.current_sequence = 1
        self.get_logger().info("Turtlesim Sequence State Machine fully booted up at sequence: 1")

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

            if v == 0.0 and w == 0.0:
                self.pub_cmd_vel.publish(cmd)
                self.get_logger().info(f"=== Sequence {self.current_sequence} Completed! ===")
                self.current_sequence += 1
                self.rotation_phase = False
                return
                
            cmd.linear.x = v
            cmd.angular.z = w
            self.pub_cmd_vel.publish(cmd)

        # --- PHASE 2: GRIPPER ACTION ENGAGEMENT ---
        elif self.current_sequence == 2:
            self.pub_cmd_vel.publish(Twist()) # Freeze position state

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
                cmd.linear.x = -0.15 # Reverse velocity profile
                self.pub_cmd_vel.publish(cmd)
            else:
                self.get_logger().info("Reverse clearance buffer established! Advancing to Sequence 4 (Target: Puck).")
                self.current_sequence += 1
                self.state_start_time = None

        # --- PHASE 5: PUFF STRIKE / PASS TRANSITION ---
        elif self.current_sequence == 5:
            dest = f"Robot {self.pass_to_robot}" if self.pass_to_robot else "the field goal"
            self.get_logger().info(f"[SHOOT] Kinetic impulse executed! Projecting puck toward {dest}.")
            self.current_sequence += 1
            
        else:
            self.get_logger().info("All sequences completed. Robot is now idle.")
            self.pub_cmd_vel.publish(Twist())

    def nid_kinematics(self):
        l = self.get_parameter('l').value
        Kp_v = self.get_parameter('kp_v').value
        Kp_w = self.get_parameter('kp_w').value
        
        # Unpack Turtlesim pose model structure natively
        x, y, theta = self.robot_pose.x, self.robot_pose.y, self.robot_pose.theta
        
        p_xg = self.current_target_pose.position.x
        p_yg = self.current_target_pose.position.y
        
        # Convert orientation structures safely for target calculations
        siny_cosp = 2 * (self.current_target_pose.orientation.w * self.current_target_pose.orientation.z)
        cosy_cosp = 1 - 2 * (self.current_target_pose.orientation.z ** 2)
        target_theta = math.atan2(siny_cosp, cosy_cosp)

        p_xl = x + l * math.cos(theta)
        p_yl = y + l * math.sin(theta)

        p_xg += l * math.cos(target_theta)
        p_yg += l * math.sin(target_theta)

        distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
        angle_error = np.arctan2(np.sin(target_theta - theta), np.cos(target_theta - theta))
        
        self.get_logger().info(f"Target Dist: {distance_to_target:.3f}m | Heading Error: {math.degrees(angle_error):.1f}°", throttle_duration_sec=1.0)
        
        v, w = 0.0, 0.0
        if self.rotation_phase or distance_to_target <= self.get_parameter('tolerance').value:
            self.rotation_phase = True
            if self.current_sequence == 1:
                flipped_target_theta = np.arctan2(np.sin(target_theta + np.pi), np.cos(target_theta + np.pi))
                angle_error = np.arctan2(np.sin(flipped_target_theta - theta), np.cos(flipped_target_theta - theta))
            if abs(angle_error) > 0.02:
                w = Kp_w * angle_error
            else:
                v, w = 0.0, 0.0
        else:
            control_inputs = self.L_inv @ np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]) @ np.array([[Kp_v * (p_xg - p_xl)], [Kp_v * (p_yg - p_yl)]])
            v, w = control_inputs[0, 0], control_inputs[1, 0]
            
        return float(v), float(w)

def main(args=None):
    parser = argparse.ArgumentParser(description='Turtlesim Sequence Controller')
    parser.add_argument('--pass_to_robot', type=int, default=0, help='ID of teammate target')
    args, remaining = parser.parse_known_args(args)
    
    rclpy.init(args=remaining)
    node = TurtlesimSequenceController(pass_to_robot=args.pass_to_robot)
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