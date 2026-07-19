import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, PoseStamped
from turtlesim.msg import Pose
from turtlesim.srv import Spawn

class MoveTurtleSim(Node):
    def __init__(self):
        super().__init__('move_turtle_sim_node')

        self.l = 0.15
        self.L_inv = np.array([[1, 0], [0, 1/self.l]])
        self.K_vp = 0.6
        self.K_wp = 1.0
        self.turtle_pose = None
        self.rotation_phase = False
        
        # 0: target_1, 1: target_2, 2: target_3, 3: target_4, 4: home, 5: finished
        self.target_index = 0 

        # Initialize subscriber variables as None (waiting for data)
        self.target_1 = None
        self.target_2 = None
        self.target_3 = None
        self.target_4 = None
        
        # Pre-populate home pose object
        self.home_pose = self._populate_target_pose(5.544, 5.544, 0.0)
        self.current_target_pose = None

        # Subscribe to all target positions
        self.create_subscription(PoseStamped, '/target_point_1', self.target_point_1_callback, 10)
        self.create_subscription(PoseStamped, '/target_point_2', self.target_point_2_callback, 10)
        self.create_subscription(PoseStamped, '/target_point_3', self.target_point_3_callback, 10)
        self.create_subscription(PoseStamped, '/target_point_4', self.target_point_4_callback, 10)

        # QoS Profiles
        qos_pose = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        qos_cmd_vel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)

        # Create subscriptions and publishers
        self.create_subscription(Pose, '/turtle1/pose', self.turtle1_pose_callback, qos_pose)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', qos_cmd_vel)

        # Fire control loop timer
        self.timer = self.create_timer(0.02, self.control_loop)
        self.action_group = ReentrantCallbackGroup()
        self.get_logger().info("MoveTurtleSimNode initialized. Awaiting topic waypoints...")

    def target_point_1_callback(self, msg):
        if self.target_1 is None:
            x = msg.pose.position.x
            y = msg.pose.position.y
            yaw = self.get_yaw_from_quaternion(msg.pose.orientation)
            self.spawn_target_turtle(x, y, yaw, 'target_turtle_1')
        self.target_1 = msg

    def target_point_2_callback(self, msg):
        if self.target_2 is None:
            x = msg.pose.position.x
            y = msg.pose.position.y
            yaw = self.get_yaw_from_quaternion(msg.pose.orientation)
            self.spawn_target_turtle(x, y, yaw, 'target_turtle_2')
        self.target_2 = msg

    def target_point_3_callback(self, msg):
        if self.target_3 is None:
            x = msg.pose.position.x
            y = msg.pose.position.y
            yaw = self.get_yaw_from_quaternion(msg.pose.orientation)
            self.spawn_target_turtle(x, y, yaw, 'target_turtle_3')
        self.target_3 = msg

    def target_point_4_callback(self, msg):
        if self.target_4 is None:
            x = msg.pose.position.x
            y = msg.pose.position.y
            yaw = self.get_yaw_from_quaternion(msg.pose.orientation)
            self.spawn_target_turtle(x, y, yaw, 'target_turtle_4')
        self.target_4 = msg
    
    def spawn_target_turtle(self, x, y, yaw, name):
        client = self.create_client(Spawn, '/spawn')
        if not client.service_is_ready():
            return # Skip if service isn't ready yet to prevent loop blocking
        request = Spawn.Request()
        request.x = x
        request.y = y
        request.theta = yaw
        request.name = name
        client.call_async(request)

    def _populate_target_pose(self, x, y, yaw):
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'map'
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = 0.0 
        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = np.sin(yaw/2.0)
        target_pose.pose.orientation.w = np.cos(yaw/2.0)
        return target_pose

    def turtle1_pose_callback(self, msg):
        self.turtle_pose = msg

    def control_loop(self):
        if self.turtle_pose is None:
            return
            
        # 1. Complete Shutdown Condition
        if self.target_index >= 5:
            self.cmd_vel_publisher.publish(Twist()) 
            return

        # 2. Dynamic Target Resolution State Machine
        if self.target_index == 0:
            self.current_target_pose = self.target_1
        elif self.target_index == 1:
            self.current_target_pose = self.target_2
        elif self.target_index == 2:
            self.current_target_pose = self.target_3
        elif self.target_index == 3:
            self.current_target_pose = self.target_4
        elif self.target_index == 4:
            self.current_target_pose = self.home_pose

        # 3. Guard: If the current target topic hasn't published data yet, wait patiently
        if self.current_target_pose is None:
            return

        cmd = Twist()
        v, w = self.kinematic_model()
        
        # 4. Target Sequence Handoff
        if v == 0.0 and w == 0.0:
            self.cmd_vel_publisher.publish(Twist()) 
            self.target_index += 1
            self.rotation_phase = False # Reset flag for next leg translation
            
            if self.target_index < 4:
                self.get_logger().info(f"--- Target {self.target_index} completed. Waiting/Moving to Target {self.target_index + 1}... ---")
            elif self.target_index == 4:
                self.get_logger().info("--- All topic waypoints finished! Heading Home... ---")
            else:
                self.get_logger().info("--- Mission Complete! Turtle has returned home safely. ---")
            return

        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_vel_publisher.publish(cmd)

    def get_rotation_matrix(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    def get_yaw_from_quaternion(self, quaternion):
        siny_cosp = 2 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
        cosy_cosp = 1 - 2 * (quaternion.y**2 + quaternion.z**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def kinematic_model(self):
        x, y, theta = self.turtle_pose.x, self.turtle_pose.y, self.turtle_pose.theta
        p_xg, p_yg = self.current_target_pose.pose.position.x, self.current_target_pose.pose.position.y
        target_theta = self.get_yaw_from_quaternion(self.current_target_pose.pose.orientation)
        
        p_xl = x + self.l * np.cos(theta)
        p_yl = y + self.l * np.sin(theta)
        
        # Error Calculation with look-ahead projection
        p_xg += self.l * np.cos(target_theta)
        p_yg += self.l * np.sin(target_theta)
        
        distance_to_target = np.sqrt((p_xg - p_xl)**2 + (p_yg - p_yl)**2)
        angle_error = target_theta - theta
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
        
        v, w = 0.0, 0.0
        
        # --- PHASE CONTROL ---
        if self.rotation_phase or distance_to_target <= 0.15:
            self.rotation_phase = True 
            
            if abs(angle_error) > 0.02:
                v = 0.0
                w = self.K_wp * angle_error
            else:
                v, w = 0.0, 0.0 
        else:
            # --- TRANSLATION PHASE ---
            e_x = p_xg - p_xl
            e_y = p_yg - p_yl
            p_dot_x = self.K_vp * e_x
            p_dot_y = self.K_vp * e_y

            control_inputs = self.L_inv @ self.get_rotation_matrix(theta).transpose() @ np.array([[p_dot_x], [p_dot_y]])
            v, w = control_inputs[0, 0], control_inputs[1, 0]

        return float(v), float(w)

def main(args=None):
    rclpy.init(args=args)
    node = MoveTurtleSim()
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