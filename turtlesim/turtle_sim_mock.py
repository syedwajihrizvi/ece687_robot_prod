import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from turtlesim.srv import Spawn

class TurtlesimTargetPublisher(Node):
    def __init__(self, stick_x, stick_y, stick_denom, puck_x, puck_y, puck_denom):
        super().__init__('turtlesim_target_publisher')
        
        # Calculate orientations dynamically using np.pi divided by your denominator inputs
        # If the denominator passed is 0, default to an angle of 0.0 to prevent zero division
        stick_theta = np.pi / stick_denom if stick_denom != 0 else 0.0
        puck_theta = np.pi / puck_denom if puck_denom != 0 else 0.0
        
        self.get_logger().info("Spawning Turtlesim Target Publisher with the following configurations:")
        # Define field target configurations (x, y, computed_yaw)
        self.stick_position = (stick_x, stick_y, stick_theta)
        self.puck_position = (puck_x, puck_y, puck_theta)
        
        # Track if targets have been visually spawned yet
        self.spawned_markers = False
        
        # Set up publishers to match your field tracking interfaces
        self.stick_pub = self.create_publisher(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', 10)
        self.puck_pub = self.create_publisher(PoseStamped, '/vrpn_mocap/puck_1/pose', 10)
        
        # High frequency environment update clock loop (50 Hz)
        self.timer = self.create_timer(0.02, self.environment_tick)
        
        self.get_logger().info(
            f"Turtlesim Target Mock Publisher ready via Argparse.\n"
            f"-> Hockey Stick Pose: ({stick_x}, {stick_y}) | Yaw: np.pi/{stick_denom} ({math.degrees(stick_theta):.1f}°)\n"
            f"-> Puck Pose: ({puck_x}, {puck_y}) | Yaw: np.pi/{puck_denom} ({math.degrees(puck_theta):.1f}°)"
        )

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

import math # Imported inside script boundary for logger degrees calculation cleanly

def main(args=None):
    parser = argparse.ArgumentParser(description='Turtlesim Target Mock Publisher via Denominator Angles')
    
    # Hockey stick target configs
    parser.add_argument('--stick_x', type=float, default=1.0, help='X position of hockey stick')
    parser.add_argument('--stick_y', type=float, default=1.0, help='Y position of hockey stick')
    parser.add_argument('--stick_denom', type=float, default=2.0, help='Denominator value to divide np.pi for stick angle (e.g., 2 for pi/2)')
    
    # Puck target configs
    parser.add_argument('--puck_x', type=float, default=10.0, help='X position of puck')
    parser.add_argument('--puck_y', type=float, default=5.5, help='Y position of puck')
    parser.add_argument('--puck_denom', type=float, default=0.0, help='Denominator value to divide np.pi for puck angle (use 0 for flat 0.0 rotation)')
    
    # Filter out internal ROS arguments to safely parse your custom flags
    parsed_args, remaining = parser.parse_known_args(args)
    
    rclpy.init(args=remaining)
    node = TurtlesimTargetPublisher(
        stick_x=parsed_args.stick_x,
        stick_y=parsed_args.stick_y,
        stick_denom=parsed_args.stick_denom,
        puck_x=parsed_args.puck_x,
        puck_y=parsed_args.puck_y,
        puck_denom=parsed_args.puck_denom
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