import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

"""
Publish a list of target points for the robot to visit in sequence
It publishes a PoseStamped that the other node subscribes to
"""
class TargetPoints(Node):
    def __init__(self):
        super().__init__('target_points_publisher')
        self.target_points = [
            (10.0, 10.0, np.pi/2),
            (2.2,  3.6,  np.pi/4),
            (2.0,  8.5,  np.pi),     # New Waypoint 3
            (8.5,  2.0,  3*np.pi/2),  # New Waypoint 4
        ]
        # A publisher for each target point, publish them at different topics
        self.point_publishers = []
        for i in range(len(self.target_points)):
            publisher = self.create_publisher(PoseStamped, f'/target_point_{i+1}', 10)
            self.point_publishers.append(publisher)
        self.timer = self.create_timer(1.0, self.publish_target_points)
        self.get_logger().info("TargetPoints node initialized and publishing target points.")

    def publish_target_points(self):
        for i, (x, y, yaw) in enumerate(self.target_points):
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            # Convert yaw to quaternion
            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            self.point_publishers[i].publish(pose_stamped)

def main(args=None):
    rclpy.init(args=args)
    node = TargetPoints()
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