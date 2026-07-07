import rclpy
import argparse
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped, Point, PoseStamped
from robomaster_msgs.action import MoveArm
from scipy.spatial.transform import Rotation as R
import numpy as np


class EEControllerNode(Node):
    def __init__(self, robot_id):
        super().__init__('ee_controller_node')
        self.robot_name = f"/robot{robot_id}"
        # Declare variables of interest here. Mainly end effector position, and desired position
        self.ee_position = None
        self.hockey_stick_position = None
        self.robot_position = None
        # Declare parameters
        self.declare_parameter('control_frequency', 10.0)

        # Get parameters
        self.control_frequency = self.get_parameter('control_frequency').value

        # Create a timer for control loop
        timer_period = 1.0 / self.control_frequency
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Create a QoS profile for publishers and subscribers
        # TODO: May need to change based on subscription Qos Profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        # Create Subscribers and Publishers with appropriate QoS settings
        # Subscription to the hockey stick wrt to the world frame
        self.create_subscription(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', self.hockey_stick_pos_callback, qos)
        # Subscription to the robot pose wrt to the world frame
        self.create_subscription(PoseStamped, f'/vrpn_mocap/dji_robot_{robot_id}/pose', self.robot_pos_callback, qos)

        # Subscription to end effector state. Possibly arm_position
        # TODO: Get the proper topic name
        self.create_subscription(PointStamped, f'{self.robot_name}/arm_position', self.arm_position_callback, qos)

        # Use move_arm action server since its safer
        # TODO: Get the proper action name
        self.action_group = ReentrantCallbackGroup()
        self.move_arm_client = ActionClient(self, MoveArm, f'{self.robot_name}/move_arm', callback_group=self.action_group)
        # TODO: Once the action server name is confirmed
        # self.move_arm_client.wait_for_server()
        # Logger to show the node is running
        self.target_frame =  'robot8/arm_base_link'
        self.source_frame = 'world'
        self.get_logger().info(f'EEControllerNode initialized for robot ID: {robot_id} with control frequency: {self.control_frequency} Hz @ {self.robot_name}')

    def control_loop(self):
        # Implement the control logic here
        self.get_logger().info('Control loop running...')
        if self.hockey_stick_position is not None:
            self.get_logger().info(f"Hockey Stick coordinates {self.hockey_stick_position.pose.position.x}, {self.hockey_stick_position.pose.position.z}")
        if self.ee_position is not None:
            self.get_logger().info(f"EE coordinates {self.ee_position.point.x}, {self.ee_position.point.z}")
        # Get the matrix conversion
        hm_trans = self.convert_hockey_stick_to_arm_frame()
        self.get_logger().info(f"Homogeneous Matrix: {hm_trans}")
        # Extract the x, z coordinates from the homogenous matrix
        # TODO: Uncomment once action server confirmed
        # goal = MoveArm.Goal()
        # # Calculate x, z fields based on the current end effector position and desired position
        # # Set x, z, and relative field here
        # # Should be relative to the arm base position
        # # Then we convert the target position to the arm base frame as well

        # goal.relative = False
        # future = self.move_arm_client.send_goal_async(goal)
        # future.add_done_callback(self.move_arm_goal_response)
        
    def convert_to_matrix(self, pose_msg: PoseStamped) -> np.ndarray:
        pos = pose_msg.pose.position
        ori = pose_msg.pose.orientation
        translation = np.array([pos.x, pos.y, pos.z])
        quat = [ori.x, ori.y, ori.z, ori.w]
        rotation_matrix = R.from_quat(quat).as_matrix()
        homogeneous_matrix = np.eye(4)
        homogeneous_matrix[:3, :3] = rotation_matrix
        homogeneous_matrix[:3, 3] = translation
        return homogeneous_matrix

    def convert_hockey_stick_to_arm_frame(self):
        robot_to_arm_frame = np.array([[1, 0, 0, -0.14],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0.1],
                                       [0, 0, 0, 1]])
        robot_to_world_frame = self.convert_to_matrix(self.robot_position)
        world_to_robot_frame = np.linalg.inv(robot_to_world_frame)
        hockey_stick_to_world_frame = self.convert_to_matrix(self.hockey_stick_position)
        return robot_to_arm_frame @ world_to_robot_frame @ hockey_stick_to_world_frame

    def move_arm_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Move arm goal rejected')
            return

        self.get_logger().info('Move arm goal accepted')
        goal_handle.get_result_async().add_done_callback(self.move_arm_result_callback)

    def move_arm_result_callback(self, future):
        result = future.result().result

    # For the subscription to the ee position, just log for now
    def arm_position_callback(self, msg):
        self.ee_position = msg
    
    def hockey_stick_pos_callback(self, msg):
        self.hockey_stick_position = msg

    def robot_pos_callback(self, msg):
        self.robot_position = msg
    
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_id', type=int, default=1, help='ID of the robot to control')
    args, remaining = parser.parse_known_args()
    rclpy.init(args=remaining)
    node = EEControllerNode(robot_id=args.robot_id)
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