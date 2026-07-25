import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped

class MockVrpnPublisher(Node):
    def __init__(self, robot_id, num_obstacles=10):
        super().__init__('mock_vrpn_publisher')
        
        self.robot_id = robot_id
        
        # 1. Target Configurations (x, y, yaw)
        self.stick_position = (2.0, 3.5, np.pi / 4.0)
        self.puck_position = (6.0, 7.0, 0.0)
        
        # 2. Controlled Robot Initial Pose
        self.robot_x = 1.0
        self.robot_y = 1.0
        self.robot_theta = 0.0
        
        # Track incoming velocities
        self.current_v = 0.0
        self.current_w = 0.0
        
        # 3. Generate static field positions for obstacle robots
        self.obstacle_positions = self._generate_obstacle_field(num_obstacles)
        
        # 4. Create publishers for VRPN Mocap topics
        self.robot_pub = self.create_publisher(
            PoseStamped, 
            f'/mock/vrpn_mocap/dji_robot_{robot_id}/pose', 
            10
        )
        self.stick_pub = self.create_publisher(
            PoseStamped, 
            '/mock/vrpn_mocap/hockey_sticks_1/pose', 
            10
        )
        self.puck_pub = self.create_publisher(
            PoseStamped, 
            '/mock/vrpn_mocap/puck_1/pose', 
            10
        )
        
        # Publishers for all other robots on the field: /mock/vrpn_mocap/dji_robot_{i}/pose
        self.obstacle_pubs = {}
        for i in range(1, num_obstacles + 1):
            if i == self.robot_id:
                continue  # Skip publishing own robot's obstacle topic
            topic_name = f'/mock/vrpn_mocap/dji_robot_{i}/pose'
            self.obstacle_pubs[i] = self.create_publisher(PoseStamped, topic_name, 10)
        
        # 5. Subscribe to velocity commands for active control loop
        self.create_subscription(Twist, f'/robot{robot_id}/cmd_vel', self.cmd_vel_callback, 10)
        
        # Run integration loop at 50 Hz (20ms)
        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.simulation_step)
        
        self.get_logger().info(
            f"Mock VRPN Simulator active for Robot {robot_id}.\n"
            f"-> Broadcasting active robot pose to /mock/vrpn_mocap/dji_robot_{robot_id}/pose\n"
            f"-> Broadcasting {num_obstacles - 1} ally/obstacle poses on /mock/vrpn_mocap/dji_robot_i/pose\n"
            f"-> Listening for control commands on /robot{robot_id}/cmd_vel"
        )

    def _generate_obstacle_field(self, num_obstacles=10):
        obstacles = []
        x0, y0 = self.robot_x, self.robot_y
        x1, y1, _ = self.stick_position
        x2, y2, _ = self.puck_position

        waypoints = [(x0, y0), (x1, y1), (x2, y2)]
        leg_lengths = [
            math.hypot(x1 - x0, y1 - y0),
            math.hypot(x2 - x1, y2 - y1)
        ]
        total_dist = sum(leg_lengths)

        sample_fractions = np.linspace(0.12, 0.92, num_obstacles)
        lateral_offsets = [0.6, -0.7, 0.8, -0.6, 0.7, -0.8, 0.5, -0.6, 0.7, -0.5]

        for idx, f in enumerate(sample_fractions):
            target_dist = f * total_dist
            accumulated_dist = 0.0
            
            for leg_idx in range(len(leg_lengths)):
                leg_len = leg_lengths[leg_idx]
                if accumulated_dist + leg_len >= target_dist:
                    ratio = (target_dist - accumulated_dist) / leg_len
                    p1 = waypoints[leg_idx]
                    p2 = waypoints[leg_idx + 1]

                    cx = p1[0] + ratio * (p2[0] - p1[0])
                    cy = p1[1] + ratio * (p2[1] - p1[1])

                    heading = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                    perp_x = -math.sin(heading)
                    perp_y = math.cos(heading)

                    offset = lateral_offsets[idx % len(lateral_offsets)]
                    ox = cx + offset * perp_x
                    oy = cy + offset * perp_y

                    obstacles.append((ox, oy, heading))
                    break
                accumulated_dist += leg_len

        return obstacles

    def cmd_vel_callback(self, msg):
        self.current_v = msg.linear.x
        self.current_w = msg.angular.z

    def simulation_step(self):
        # 6. Unicycle Integration
        self.robot_x += self.current_v * np.cos(self.robot_theta) * self.dt
        self.robot_y += self.current_v * np.sin(self.robot_theta) * self.dt
        self.robot_theta += self.current_w * self.dt
        
        self.robot_theta = np.arctan2(np.sin(self.robot_theta), np.cos(self.robot_theta))
        
        now = self.get_clock().now().to_msg()
        
        # 7. Broadcast Active Robot Pose
        robot_msg = self._build_pose_msg(self.robot_x, self.robot_y, self.robot_theta, now)
        self.robot_pub.publish(robot_msg)
        
        # 8. Broadcast Target Poses
        stick_msg = self._build_pose_msg(*self.stick_position, now)
        self.stick_pub.publish(stick_msg)
        
        puck_msg = self._build_pose_msg(*self.puck_position, now)
        self.puck_pub.publish(puck_msg)

        # 9. Broadcast Obstacle Robot Poses (Safely skipping self.robot_id)
        for i, obs_pos in enumerate(self.obstacle_positions, 1):
            if i in self.obstacle_pubs:
                obs_msg = self._build_pose_msg(*obs_pos, now)
                self.obstacle_pubs[i].publish(obs_msg)

    def _build_pose_msg(self, x, y, yaw, timestamp):
        msg = PoseStamped()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(np.sin(yaw / 2.0))
        msg.pose.orientation.w = float(np.cos(yaw / 2.0))
        return msg

def main(args=None):
    parser = argparse.ArgumentParser(description='Mock VRPN Publisher for Multi-Robot Field')
    parser.add_argument('--robot_id', type=int, default=1, help='ID of active controlled robot')
    args, remaining = parser.parse_known_args(args)

    rclpy.init(args=remaining)
    node = MockVrpnPublisher(robot_id=args.robot_id, num_obstacles=10)
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