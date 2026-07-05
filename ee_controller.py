import rclpy
import math
import argparse
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped, Point
from robomaster_msgs.action import MoveArm

class EEControllerNode(Node):
    def __init__(self, robot_id):
        super().__init__('ee_controller_node')
        self.robot_name = f"/robot{robot_id}"

        # Declare variables of interest here. Mainly end effector position, and desired position
        self.ee_position = None
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
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        # Create Subscribers and Publishers with appropriate QoS settings
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
        self.get_logger().info(f'EEControllerNode initialized for robot ID: {robot_id} with control frequency: {self.control_frequency} Hz @ {self.robot_name}')

    def control_loop(self):
        # Implement the control logic here
        self.get_logger().info('Control loop running...')
        # TODO: Uncomment once action server confirmed
        # goal = MoveArm.Goal()
        # # Calculate x, z fields based on the current end effector position and desired position
        # # Set x, z, and relative field here
        # # Should be relative to the arm base position
        # # Then we convert the target position to the arm base frame as well

        # goal.relative = False
        # future = self.move_arm_client.send_goal_async(goal)
        # future.add_done_callback(self.move_arm_goal_response)

    def move_arm_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Move arm goal rejected')
            return

        self.get_logger().info('Move arm goal accepted')
        goal_handle.get_result_async().add_done_callback(self.move_arm_result_callback)

    def move_arm_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Move arm result: {result}')

    # For the subscription to the ee position, just log for now
    def arm_position_callback(self, msg):
        self.ee_position = msg

    

    

    


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