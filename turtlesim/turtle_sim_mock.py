import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from turtlesim.srv import Spawn

class TurtlesimTargetPublisher(Node):
    def __init__(self):
        super().__init__('turtlesim_target_publisher')
        
        # Define field target configurations (x, y, yaw) inside Turtlesim boundaries [0.0 - 11.0]
        self.stick_position = (1.0, 1.0, np.pi/4)
        self.puck_position = (9.0, 10.0, np.pi/2)
        
        # Track if targets have been visually spawned yet
        self.spawned_markers = False
        
        # Set up publishers to match your field tracking interfaces
        self.stick_pub = self.create_publisher(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', 10)
        self.puck_pub = self.create_publisher(PoseStamped, '/vrpn_mocap/puck_1/pose', 10)
        
        # High frequency environment update clock loop (50 Hz)
        self.timer = self.create_timer(0.02, self.environment_tick)
        self.get_logger().info("Turtlesim Target Mock Publisher ready. Spawning asset visual pins...")

    def spawn_marker_turtles(self):
        client = self.create_client(Spawn, '/spawn')
        if not client.service_is_ready():
            return  # Wait until turtlesim window is fully up and initialized
        
        # Pin 1: Hockey Stick position visual anchor
        req1 = Spawn.Request()
        req1.x, req1.y, req1.theta, req1.name = *self.stick_position, 'hockey_stick_pin'
        client.call_async(req1)
        
        # Pin 2: Puck position visual anchor
        req2 = Spawn.Request()
        req2.x, req2.y, req2.theta, req2.name = *self.puck_position, 'puck_pin'
        client.call_async(req2)
        
        self.spawned_markers = True
        self.get_logger().info("Visual scene markers successfully spawned into Turtlesim window!")

    def environment_tick(self):
        if not self.spawned_markers:
            self.spawn_marker_turtles()
            return
            
        now = self.get_clock().now().to_msg()
        
        # Broadcast positions
        stick_msg = self._build_pose_msg(*self.stick_position, now)
        self.stick_pub.publish(stick_msg)
        
        puck_msg = self._build_pose_msg(*self.puck_position, now)
        self.puck_pub.publish(puck_msg)

    def _build_pose_msg(self, x, y, yaw, timestamp):
        msg = PoseStamped()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = np.sin(yaw / 2.0)
        msg.pose.orientation.w = np.cos(yaw / 2.0)
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = TurtlesimTargetPublisher()
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