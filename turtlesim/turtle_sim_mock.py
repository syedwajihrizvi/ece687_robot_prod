import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from turtlesim.srv import Spawn

class TurtlesimTargetPublisher(Node):
    def __init__(self, stick_x, stick_y, stick_angle_rad, 
                 puck_x, puck_y, puck_angle_rad,
                 standoff_distance, sideways_offset, vertical_offset,
                 obstacles_user_list):
        super().__init__('turtlesim_target_publisher')
        
        # Target pose configurations: (x, y, theta)
        self.stick_position = (stick_x, stick_y, stick_angle_rad)
        self.puck_position = (puck_x, puck_y, puck_angle_rad)

        # Default Turtlesim starting position for turtle1
        self.start_position = (5.54, 5.54)

        # Kinematic parameters
        self.standoff_distance = standoff_distance
        self.sideways_offset = sideways_offset
        self.vertical_offset = vertical_offset

        # Filter obstacles provided via CLI
        self.obstacles = []
        for ox, oy in obstacles_user_list:
            if ox is not None and oy is not None:
                self.obstacles.append((ox, oy, 0.0))
        
        # Track marker spawning state
        self.spawned_markers = False

        # Target publishers
        self.stick_pub = self.create_publisher(PoseStamped, '/vrpn_mocap/hockey_sticks_1/pose', 10)
        self.puck_pub = self.create_publisher(PoseStamped, '/vrpn_mocap/puck_1/pose', 10)
        
        # Dynamic obstacle publishers
        self.obstacle_pubs = []
        for i in range(1, len(self.obstacles) + 1):
            topic_name = f'/vrpn_mocap/robot_obstacle_{i}/pose'
            pub = self.create_publisher(PoseStamped, topic_name, 10)
            self.obstacle_pubs.append(pub)

        # High frequency environment tick loop (50 Hz)
        self.timer = self.create_timer(0.02, self.environment_tick)
        
        self.get_logger().info(
            f"Turtlesim Target Mock Publisher booted with {len(self.obstacles)} Custom Obstacles.\n"
            f"-> Hockey Stick Pose: ({stick_x}, {stick_y}) @ {math.degrees(stick_angle_rad):.1f} deg (Facing DOWN)\n"
            f"-> Puck Pose: ({puck_x}, {puck_y}) @ {math.degrees(puck_angle_rad):.1f} deg (Facing UP)"
        )

    def spawn_marker_turtles(self):
        client = self.create_client(Spawn, '/spawn')
        if not client.service_is_ready():
            return  # Wait for Turtlesim window initialization
        
        # Pin 1: Hockey Stick position
        req1 = Spawn.Request()
        req1.x, req1.y, req1.theta, req1.name = *self.stick_position, 'hockey_stick_pin'
        client.call_async(req1)
        
        # Pin 2: Puck position
        req2 = Spawn.Request()
        req2.x, req2.y, req2.theta, req2.name = *self.puck_position, 'puck_pin'
        client.call_async(req2)

        # Pins 3+: Explicit CLI obstacles
        for idx, obs_pose in enumerate(self.obstacles, 1):
            req = Spawn.Request()
            req.x, req.y, req.theta, req.name = *obs_pose, f'obstacle_obs_pin_{idx}'
            client.call_async(req)

        self.spawned_markers = True
        self.get_logger().info(f"Visual target markers and {len(self.obstacles)} obstacles spawned in Turtlesim!")

    def environment_tick(self):
        if not self.spawned_markers:
            self.spawn_marker_turtles()
            return
            
        now = self.get_clock().now().to_msg()
        
        # Broadcast target poses
        stick_msg = self._build_pose_msg(*self.stick_position, now)
        self.stick_pub.publish(stick_msg)
        
        puck_msg = self._build_pose_msg(*self.puck_position, now)
        self.puck_pub.publish(puck_msg)

        # Broadcast custom obstacle poses
        for idx, obs_pose in enumerate(self.obstacles):
            obs_msg = self._build_pose_msg(*obs_pose, now)
            self.obstacle_pubs[idx].publish(obs_msg)

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
    parser = argparse.ArgumentParser(description='Turtlesim Target Mock Publisher')
    
    # Hockey stick target configs (Default: Top-Left pointing DOWN = -pi/2)
    parser.add_argument('--stick_x', type=float, default=1.8, help='X position of hockey stick')
    parser.add_argument('--stick_y', type=float, default=9.2, help='Y position of hockey stick')
    parser.add_argument('--stick_angle_rad', type=float, default=-np.pi/2.0, help='Stick facing angle in radians (-pi/2 = DOWN)')
    
    # Puck target configs (Default: Bottom-Right pointing UP = +pi/2)
    parser.add_argument('--puck_x', type=float, default=9.2, help='X position of puck')
    parser.add_argument('--puck_y', type=float, default=1.8, help='Y position of puck')
    parser.add_argument('--puck_angle_rad', type=float, default=np.pi/2.0, help='Puck facing angle in radians (+pi/2 = UP)')

    # Kinematic Offsets
    parser.add_argument('--standoff_distance', type=float, default=2.5, help='Standoff distance')
    parser.add_argument('--sideways_offset', type=float, default=0.0, help='Sideways offset')
    parser.add_argument('--vertical_offset', type=float, default=0.0, help='Vertical offset')

    # Custom Obstacle Coordinates (up to 7)
    parser.add_argument('--obs1_x', type=float, default=None, help='Obstacle 1 X position')
    parser.add_argument('--obs1_y', type=float, default=None, help='Obstacle 1 Y position')
    parser.add_argument('--obs2_x', type=float, default=None, help='Obstacle 2 X position')
    parser.add_argument('--obs2_y', type=float, default=None, help='Obstacle 2 Y position')
    parser.add_argument('--obs3_x', type=float, default=None, help='Obstacle 3 X position')
    parser.add_argument('--obs3_y', type=float, default=None, help='Obstacle 3 Y position')
    parser.add_argument('--obs4_x', type=float, default=None, help='Obstacle 4 X position')
    parser.add_argument('--obs4_y', type=float, default=None, help='Obstacle 4 Y position')
    parser.add_argument('--obs5_x', type=float, default=None, help='Obstacle 5 X position')
    parser.add_argument('--obs5_y', type=float, default=None, help='Obstacle 5 Y position')
    parser.add_argument('--obs6_x', type=float, default=None, help='Obstacle 6 X position')
    parser.add_argument('--obs6_y', type=float, default=None, help='Obstacle 6 Y position')
    parser.add_argument('--obs7_x', type=float, default=None, help='Obstacle 7 X position')
    parser.add_argument('--obs7_y', type=float, default=None, help='Obstacle 7 Y position')

    parsed_args, remaining = parser.parse_known_args(args)
    
    user_obstacles = [
        (parsed_args.obs1_x, parsed_args.obs1_y),
        (parsed_args.obs2_x, parsed_args.obs2_y),
        (parsed_args.obs3_x, parsed_args.obs3_y),
        (parsed_args.obs4_x, parsed_args.obs4_y),
        (parsed_args.obs5_x, parsed_args.obs5_y),
        (parsed_args.obs6_x, parsed_args.obs6_y),
        (parsed_args.obs7_x, parsed_args.obs7_y),
    ]

    rclpy.init(args=remaining)
    node = TurtlesimTargetPublisher(
        stick_x=parsed_args.stick_x,
        stick_y=parsed_args.stick_y,
        stick_angle_rad=parsed_args.stick_angle_rad,
        puck_x=parsed_args.puck_x,
        puck_y=parsed_args.puck_y,
        puck_angle_rad=parsed_args.puck_angle_rad,
        standoff_distance=parsed_args.standoff_distance,
        sideways_offset=parsed_args.sideways_offset,
        vertical_offset=parsed_args.vertical_offset,
        obstacles_user_list=user_obstacles
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