import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import ColorRGBA, Bool, Empty
from math import cos, sin, pi
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class MultiRoboMasterSim(Node):
    def __init__(self, stick_poses=((1.2, 1.2, -pi / 2), (-1.2, -1.2, 0.0)),
                 stick_ids=(1, 2), puck_pose=(-1.2, 1.2), puck_color='blue',
                 goal_pose=(0.0, -1.75, pi / 2), goal_size=(0.9, 0.25),
                 goal_standoff=2.0, spawn_clearance=0.7):
        super().__init__('multi_robomaster_sim')

        # constants
        # robots (robot 1 = passer, robot 6 = shooter, 2-3 act as static obstacles)
        self.ROBOT_IDS = [1, 2, 3, 6]
        self.N = len(self.ROBOT_IDS)
        # time
        self.TIMEOUT_SET_MOBILE_BASE_SPEED = 20 # milliseconds
        self.TIMEOUT_GET_POSES = 10 # milliseconds
        self.TIMEOUT_CHASSIS_SPEED = 500 # milliseconds
        self.DT = (self.TIMEOUT_SET_MOBILE_BASE_SPEED + self.TIMEOUT_GET_POSES) / 1000.
        # robot control
        self.MAX_LINEAR_SPEED = 1.0 # meters / second
        self.MAX_ANGULAR_SPEED = 360 * np.pi / 180 # radians / second
        # dimensions
        self.ENV = [-2., -2., 4., 4.] # (x, y) can vary from (ENV[0], ENV[1]) to (ENV[0]+ENV[2], ENV[1]+ENV[3])
        self.ROBOT_SIZE = [0.24, 0.32] # [w, l]
        self.GRIPPER_SIZE = 0.1
        # hockey props: sticks drawn as a "T" whose leg points along the local +x (facing) axis
        self.STICK_LEG_LENGTH = 0.45
        self.STICK_CROSSBAR_LENGTH = 0.30
        self.STICK_HANDLE_OFFSET = 0.10 # carried: T junction this far ahead of the robot center
        self.STICK_TIP_DIST = self.STICK_HANDLE_OFFSET + self.STICK_LEG_LENGTH # carried: leg tip distance from robot center
        self.GRAB_RADIUS = 0.5 # gripper close within this range of a stick junction -> attach
        self.PUCK_RADIUS = 0.05
        self.PUCK_DAMPING = 0.8 # 1/s exponential speed decay after a hit
        self.PUCK_MAX_SPEED = 2.5 # meters / second
        self.HIT_MARGIN = 0.25 # stick tip within PUCK_RADIUS + this of the puck counts as contact
        self.MIN_HIT_SPEED = 0.5 # tip speeds below this don't launch the puck
        self.SPAWN_CLEARANCE = spawn_clearance # min spawn distance from props/other robots

        # Hockey props state (initial poses kept for in-place game resets)
        self._initial_stick_poses = {sid: np.array(sp, dtype=float) for sid, sp in zip(stick_ids, stick_poses)}
        self._initial_puck_pos = np.array(puck_pose, dtype=float)
        self._goal_standoff = goal_standoff
        self.sticks = {sid: {'pose': p.copy(), 'attached_to': None}
                       for sid, p in self._initial_stick_poses.items()}
        self.puck_pos = self._initial_puck_pos.copy()
        self.puck_vel = np.zeros(2)
        self.goal_pose = np.array(goal_pose, dtype=float) # [x, y, yaw]; the goal frame's +x axis faces the field
        self.goal_size = np.array(goal_size, dtype=float) # [width, depth]
        self.goal_scored = False

        # State: [x, y, theta]
        self.states = {}
        self.leds = {}
        self.velocities = {rid: np.array([0.0, 0.0, 0.0]) for rid in self.ROBOT_IDS}
        self.last_cmd_time = {rid: self.get_clock().now() for rid in self.ROBOT_IDS}
        self.__spawn_robots()

        # Pubs and Subs
        self.pubs = {}
        self.subs_vel = {}
        self.subs_led = {}
        self.subs_grip = {}
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        for rid in self.ROBOT_IDS:
            # Publisher: Mimics VRPN motion capture system
            self.pubs[rid] = self.create_publisher(
                PoseStamped, f'/vrpn_mocap/dji_robot_{rid}/pose', qos)

            # Subscriber: Listen to the controller's cmd_vel
            self.subs_vel[rid] = self.create_subscription(
                Twist, f'/robot{rid}/cmd_vel',
                lambda msg, rid=rid: self.vel_callback(msg, rid), qos)

            # Subscriber: Listen to the controller's leds
            self.subs_led[rid] = self.create_subscription(
                ColorRGBA, f'/robot{rid}/leds/color',
                lambda msg, rid=rid: self.led_callback(msg, rid), qos)

            # Subscriber: sim-mode gripper (reliable QoS — a grab is a single message)
            self.subs_grip[rid] = self.create_subscription(
                Bool, f'/robot{rid}/gripper_sim',
                lambda msg, rid=rid: self.gripper_callback(msg, rid), 10)

        # Subscriber: in-place game reset (also bound to the 'r' key in the plot window)
        self.sub_reset = self.create_subscription(Empty, '/sim/reset', lambda msg: self.reset_game(), 10)

        # Publishers: Mimic VRPN tracking of the hockey sticks and puck
        self.pub_sticks = {sid: self.create_publisher(
            PoseStamped, f'/vrpn_mocap/hockey_sticks_{sid}/pose', qos) for sid in self.sticks}
        self.pub_puck = self.create_publisher(
            PoseStamped, f'/vrpn_mocap/hockey_puck_{puck_color}/pose', qos)

        self.timer = self.create_timer(self.DT, self.update_and_publish)
        self.get_logger().info(f"Simulator started for robots: {self.ROBOT_IDS}")

        # Plots
        self.figure = []
        self.axes = []
        self.patches_robots = {rid: [] for rid in self.ROBOT_IDS}
        self.patches_grippers = {rid: [] for rid in self.ROBOT_IDS}
        self.text_ids = {rid: [] for rid in self.ROBOT_IDS}
        self.stick_lines = {}
        self.stick_texts = {}
        self.trails = {}
        self.__init_plot()
        self.__update_plot()

    def __spawn_robots(self):
        # Place robots randomly, resampling any spawn too close to a stick, the puck,
        # the goal, the shooter's waiting spot, or an already-placed robot — a robot sitting
        # on a target deadlocks the controller's CBF (target inside an obstacle's safety ball)
        goal_forward = np.array([cos(self.goal_pose[2]), sin(self.goal_pose[2])])
        wait_point = self.goal_pose[:2] + self._goal_standoff * goal_forward
        occupied = [s['pose'][:2] for s in self.sticks.values()]
        occupied += [self.puck_pos.copy(), self.goal_pose[:2].copy(), wait_point]
        for rid in self.ROBOT_IDS:
            for _ in range(100):
                x = np.random.uniform(self.ENV[0], self.ENV[0] + self.ENV[2])
                y = np.random.uniform(self.ENV[1], self.ENV[1] + self.ENV[3])
                if all(np.linalg.norm(np.array([x, y]) - q) >= self.SPAWN_CLEARANCE for q in occupied):
                    break
            theta = np.random.random() * 2 * np.pi

            occupied.append(np.array([x, y]))
            self.states[rid] = np.array([x, y, theta])
            self.leds[rid] = np.array([0., 0., 0.])

    def reset_game(self):
        """Restore the initial game state without restarting the process: sticks back on
        their stations, puck at its start, fresh random robot spawns, trails cleared.
        Stop the controllers before resetting, then relaunch them."""
        for sid, s in self.sticks.items():
            s['pose'] = self._initial_stick_poses[sid].copy()
            s['attached_to'] = None
        self.puck_pos = self._initial_puck_pos.copy()
        self.puck_vel = np.zeros(2)
        self.goal_scored = False
        self.__spawn_robots()
        for rid in self.ROBOT_IDS:
            self.velocities[rid] = np.array([0.0, 0.0, 0.0])
        for tr in self.trails.values():
            tr['xs'].clear()
            tr['ys'].clear()
            tr['line'].set_data([], [])
        self.axes.set_title('')
        self.get_logger().info("Game reset: sticks/puck restored, robots respawned, trails cleared.")

    def __on_key(self, event):
        if event.key == 'r':
            self.reset_game()

    def __init_plot(self):
        self.figure, self.axes = plt.subplots()
        self.figure.canvas.mpl_connect('key_press_event', self.__on_key)
        p_env = patches.Rectangle(np.array([self.ENV[0], self.ENV[1]]), self.ENV[2], self.ENV[3], edgecolor=(0, 0, 0, 1), fill=False, linewidth=4)
        self.axes.add_patch(p_env)

        for i, rid in enumerate(self.ROBOT_IDS):
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])], [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            p_robot = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                                     [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                                     [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T),
                                                     facecolor='k')
            p_gripper = patches.Polygon(t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                                       [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T),
                                                       facecolor='k')
            text_id = plt.text(self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0, self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0, s=str(self.ROBOT_IDS[i]), color="red")
            self.patches_robots[rid] = p_robot
            self.patches_grippers[rid] = p_gripper
            self.text_ids[rid] = text_id
            self.axes.add_patch(p_robot)
            self.axes.add_patch(p_gripper)

        # Hockey sticks: "T" markers whose legs point along each stick's facing (+x) direction
        for sid in self.sticks:
            leg_line, = self.axes.plot([], [], color='saddlebrown', linewidth=4)
            bar_line, = self.axes.plot([], [], color='saddlebrown', linewidth=4)
            self.stick_lines[sid] = (leg_line, bar_line)
            self.stick_texts[sid] = plt.text(0, 0, s=f'stick{sid}', color='saddlebrown')

        # Puck: a disc
        self.patch_puck = patches.Circle((self.puck_pos[0], self.puck_pos[1]), self.PUCK_RADIUS, facecolor='tab:blue')
        self.axes.add_patch(self.patch_puck)
        self.text_puck = plt.text(self.puck_pos[0] + 0.1, self.puck_pos[1] + 0.1, s='puck', color='tab:blue')

        # Goal gate: mouth opens along the goal frame's +x axis, box extends behind it
        gyaw = self.goal_pose[2]
        forward = np.array([cos(gyaw), sin(gyaw)])
        side = np.array([-sin(gyaw), cos(gyaw)])
        mouth_a = self.goal_pose[:2] + 0.5 * self.goal_size[0] * side
        mouth_b = self.goal_pose[:2] - 0.5 * self.goal_size[0] * side
        back_a = mouth_a - self.goal_size[1] * forward
        back_b = mouth_b - self.goal_size[1] * forward
        goal_color = (0.1, 0.45, 0.1)
        self.axes.add_patch(patches.Polygon(np.array([mouth_a, mouth_b, back_b, back_a]),
                                            facecolor=(0.75, 0.90, 0.75), edgecolor=goal_color, linewidth=2))
        self.axes.plot([mouth_a[0], mouth_b[0]], [mouth_a[1], mouth_b[1]], color=goal_color, linewidth=4)
        plt.text(self.goal_pose[0] + 0.1, self.goal_pose[1] + 0.1, s='goal', color=goal_color)

        # Motion trails for every movable object
        trail_colors = {1: 'tab:red', 6: 'tab:green'}
        for rid in self.ROBOT_IDS:
            line, = self.axes.plot([], [], color=trail_colors.get(rid, 'tab:gray'), linewidth=1, alpha=0.7)
            self.trails[f'robot{rid}'] = {'xs': [], 'ys': [], 'line': line}
        puck_line, = self.axes.plot([], [], color='tab:blue', linewidth=1, alpha=0.7, linestyle=':')
        self.trails['puck'] = {'xs': [], 'ys': [], 'line': puck_line}

        self.axes.set_xlim(self.ENV[0] - max(self.ROBOT_SIZE), self.ENV[0] + self.ENV[2] + max(self.ROBOT_SIZE))
        self.axes.set_xlim(self.ENV[1] - max(self.ROBOT_SIZE), self.ENV[1] + self.ENV[3] + max(self.ROBOT_SIZE))
        self.axes.grid()
        # self.axes.set_axis_off()
        self.axes.axis('equal')

        plt.ion()
        plt.show()

    def __draw_stick(self, sid):
        sp = self.sticks[sid]['pose']
        c_th, s_th = cos(sp[2]), sin(sp[2])
        R = np.array([[c_th, -s_th], [s_th, c_th]])
        t = sp[:2]
        leg_end = t + R @ np.array([self.STICK_LEG_LENGTH, 0.0])
        bar_a = t + R @ np.array([0.0, self.STICK_CROSSBAR_LENGTH / 2.0])
        bar_b = t + R @ np.array([0.0, -self.STICK_CROSSBAR_LENGTH / 2.0])
        leg_line, bar_line = self.stick_lines[sid]
        leg_line.set_data([t[0], leg_end[0]], [t[1], leg_end[1]])
        bar_line.set_data([bar_a[0], bar_b[0]], [bar_a[1], bar_b[1]])
        self.stick_texts[sid].set_position((t[0] + 0.1, t[1] + 0.1))

    def __append_trail(self, key, p):
        tr = self.trails[key]
        if not tr['xs'] or (p[0] - tr['xs'][-1])**2 + (p[1] - tr['ys'][-1])**2 > 1e-4:
            tr['xs'].append(float(p[0]))
            tr['ys'].append(float(p[1]))
            tr['line'].set_data(tr['xs'], tr['ys'])

    def __update_plot(self):
        for rid in self.ROBOT_IDS:
            R = np.array([[cos(self.states[rid][2]), -sin(self.states[rid][2])], [sin(self.states[rid][2]), cos(self.states[rid][2])]])
            t = np.array([self.states[rid][0], self.states[rid][1]])
            xy_robot = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, self.ROBOT_SIZE[0] / 2.0],
                                      [-self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0],
                                      [self.ROBOT_SIZE[1] / 2.0, -self.ROBOT_SIZE[0] / 2.0]]) @ R.T)
            xy_gripper = t + (np.array([[self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, 0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -0.8 * self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0 + self.GRIPPER_SIZE, -self.GRIPPER_SIZE / 2.0],
                                        [self.ROBOT_SIZE[1] / 2.0, -self.GRIPPER_SIZE / 2.0]]) @ R.T)

            self.patches_robots[rid].xy = xy_robot
            self.patches_grippers[rid].xy = xy_gripper

            self.patches_robots[rid].set_facecolor(self.leds[rid])

            self.text_ids[rid].set_position((self.states[rid][0] + max(self.ROBOT_SIZE) / 2.0, self.states[rid][1] + max(self.ROBOT_SIZE) / 2.0))

            self.__append_trail(f'robot{rid}', self.states[rid][:2])

        for sid in self.sticks:
            self.__draw_stick(sid)

        self.patch_puck.center = (self.puck_pos[0], self.puck_pos[1])
        self.text_puck.set_position((self.puck_pos[0] + 0.1, self.puck_pos[1] + 0.1))
        self.__append_trail('puck', self.puck_pos)

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def __make_pose_msg(self, x, y, theta):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        half_yaw = theta * 0.5
        msg.pose.orientation.z = sin(half_yaw)
        msg.pose.orientation.w = cos(half_yaw)
        return msg

    @staticmethod
    def transform_velocity_local_to_global(robots_speeds, theta):
        # robots_speeds : list of 3
        # theta : scalar
        robots_speeds_global = [0] * 3
        x_dot = robots_speeds[0]
        y_dot = robots_speeds[1]
        th_dot = robots_speeds[2]
        c_th = cos(theta)
        s_th = sin(theta)
        robots_speeds_global[0] = c_th * x_dot - s_th * y_dot
        robots_speeds_global[1] = s_th * x_dot + c_th * y_dot
        robots_speeds_global[2] = robots_speeds[2]
        return robots_speeds_global

    def vel_callback(self, msg, rid):
        # Store commanded velocities
        robot_speeds = MultiRoboMasterSim.transform_velocity_local_to_global([msg.linear.x, msg.linear.y, msg.angular.z], self.states[rid][2])
        self.velocities[rid] = np.array(robot_speeds)
        self.last_cmd_time[rid] = self.get_clock().now() # Update heartbeat

    def led_callback(self, msg, rid):
        # Store commanded velocities
        self.leds[rid] = np.array([msg.r, msg.g, msg.b])

    def gripper_callback(self, msg, rid):
        if msg.data: # gripper closed -> try to grab the nearest free stick
            if any(s['attached_to'] == rid for s in self.sticks.values()):
                return
            robot_p = self.states[rid][:2]
            for sid, s in self.sticks.items():
                if s['attached_to'] is None and np.linalg.norm(s['pose'][:2] - robot_p) <= self.GRAB_RADIUS:
                    s['attached_to'] = rid
                    self.get_logger().info(f"Robot {rid} grabbed stick {sid}")
                    return
            self.get_logger().warn(f"Robot {rid} closed its gripper but no stick is within {self.GRAB_RADIUS} m")
        else: # gripper opened -> drop any carried stick in place
            for sid, s in self.sticks.items():
                if s['attached_to'] == rid:
                    s['attached_to'] = None
                    self.get_logger().info(f"Robot {rid} released stick {sid}")

    def __puck_in_goal(self):
        gyaw = self.goal_pose[2]
        forward = np.array([cos(gyaw), sin(gyaw)])
        side = np.array([-sin(gyaw), cos(gyaw)])
        rel = self.puck_pos - self.goal_pose[:2]
        f_dist = float(np.dot(rel, forward))
        s_dist = float(np.dot(rel, side))
        return -self.goal_size[1] <= f_dist <= 0.05 and abs(s_dist) <= self.goal_size[0] / 2.0

    def __update_props(self, applied):
        # Carried sticks track their robot; a fast-moving stick tip launches the puck
        for sid, s in self.sticks.items():
            rid = s['attached_to']
            if rid is None:
                continue
            th = self.states[rid][2]
            heading = np.array([cos(th), sin(th)])
            s['pose'][:2] = self.states[rid][:2] + self.STICK_HANDLE_OFFSET * heading
            s['pose'][2] = th

            # Hit detection: only a resting puck can be launched (prevents multi-hits per swing)
            if self.goal_scored or np.linalg.norm(self.puck_vel) >= 0.1:
                continue
            tip = self.states[rid][:2] + self.STICK_TIP_DIST * heading
            if np.linalg.norm(tip - self.puck_pos) <= self.PUCK_RADIUS + self.HIT_MARGIN:
                w = applied[rid][2]
                tip_vel = applied[rid][:2] + w * self.STICK_TIP_DIST * np.array([-sin(th), cos(th)])
                speed = float(np.linalg.norm(tip_vel))
                if speed > self.MIN_HIT_SPEED:
                    self.puck_vel = tip_vel / speed * min(speed, self.PUCK_MAX_SPEED)
                    self.get_logger().info(
                        f"Puck hit by robot {rid} (stick {sid}): launched at {np.linalg.norm(self.puck_vel):.2f} m/s")

        # Puck kinematics: integrate with exponential damping, stop at walls, detect goals
        if np.linalg.norm(self.puck_vel) > 1e-3:
            self.puck_pos += self.puck_vel * self.DT
            self.puck_vel *= np.exp(-self.PUCK_DAMPING * self.DT)

            for k in range(2):
                lo = self.ENV[k] + self.PUCK_RADIUS
                hi = self.ENV[k] + self.ENV[k + 2] - self.PUCK_RADIUS
                if self.puck_pos[k] < lo or self.puck_pos[k] > hi:
                    self.puck_pos[k] = min(max(self.puck_pos[k], lo), hi)
                    self.puck_vel[:] = 0.0

            if not self.goal_scored and self.__puck_in_goal():
                self.goal_scored = True
                self.puck_vel[:] = 0.0
                self.get_logger().info("GOAL! Puck crossed the goal mouth.")
                self.axes.set_title('GOAL!', color=(0.1, 0.45, 0.1), fontweight='bold')

    def update_and_publish(self):
        current_time = self.get_clock().now()

        applied = {}
        for rid in self.ROBOT_IDS:
            elapsed_time_since_last_command_received = (current_time - self.last_cmd_time[rid]).nanoseconds / 1e9
            if elapsed_time_since_last_command_received > self.TIMEOUT_CHASSIS_SPEED / 1e3:
                v_cmd = np.array([0.0, 0.0, 0.0])
            else:
                v_cmd = self.velocities[rid]
            applied[rid] = v_cmd

            # Integrate velocity
            # Global X/Y update
            # The controller sends local velocities, which is what the robots are expected to receive.
            # These are then converted to global in the velocity callback in the simulator
            self.states[rid][0] += v_cmd[0] * self.DT
            self.states[rid][1] += v_cmd[1] * self.DT
            self.states[rid][2] += v_cmd[2] * self.DT

            # Create PoseStamped message (Euler to Quaternion, simplified for 2D Z-axis rotation)
            msg = self.__make_pose_msg(self.states[rid][0], self.states[rid][1], self.states[rid][2])
            self.pubs[rid].publish(msg)

        self.__update_props(applied)

        # Publish prop poses every tick so late-joining BEST_EFFORT subscribers get them
        for sid, s in self.sticks.items():
            self.pub_sticks[sid].publish(self.__make_pose_msg(s['pose'][0], s['pose'][1], s['pose'][2]))
        self.pub_puck.publish(self.__make_pose_msg(self.puck_pos[0], self.puck_pos[1], 0.0))

        self.__update_plot()

def main(args=None):
    parser = argparse.ArgumentParser(description='Multi RoboMaster simulator with hockey sticks, puck physics, and a goal gate')
    parser.add_argument('--stick_x', type=float, default=1.2, help='Stick 1 x position (m)')
    parser.add_argument('--stick_y', type=float, default=1.2, help='Stick 1 y position (m)')
    parser.add_argument('--stick_theta_deg', type=float, default=-90.0,
                        help='Stick 1 facing angle (deg); the T leg and the standoff point both lie along this direction — keep it pointed away from the puck')
    parser.add_argument('--stick2_x', type=float, default=-1.2, help='Stick 2 x position (m)')
    parser.add_argument('--stick2_y', type=float, default=-1.2, help='Stick 2 y position (m)')
    parser.add_argument('--stick2_theta_deg', type=float, default=0.0, help='Stick 2 facing angle (deg); keep it pointed away from the puck')
    parser.add_argument('--puck_x', type=float, default=-1.2, help='Puck x position (m)')
    parser.add_argument('--puck_y', type=float, default=1.2, help='Puck y position (m)')
    parser.add_argument('--hockey_stick_id', type=int, default=1, help='ID tag for the stick 1 VRPN topic')
    parser.add_argument('--stick2_id', type=int, default=2, help='ID tag for the stick 2 VRPN topic')
    parser.add_argument('--puck_color', type=str, default='blue', help='Color tag for the puck VRPN topic')
    parser.add_argument('--goal_x', type=float, default=0.0, help='Goal center x (m)')
    parser.add_argument('--goal_y', type=float, default=-1.75, help='Goal center y (m)')
    parser.add_argument('--goal_yaw_deg', type=float, default=90.0,
                        help='Goal facing angle (deg); the mouth opens along this (+x of the goal frame)')
    parser.add_argument('--goal_width', type=float, default=0.9, help='Goal mouth width (m)')
    parser.add_argument('--goal_depth', type=float, default=0.25, help='Goal box depth (m)')
    parser.add_argument('--goal_standoff', type=float, default=2.0,
                        help='Shooter waiting distance from the goal along its facing (spawn clearance only)')
    parser.add_argument('--spawn_clearance', type=float, default=0.7,
                        help='Min spawn distance (m) between a robot and any prop or another robot')
    parsed, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    node = MultiRoboMasterSim(
        stick_poses=((parsed.stick_x, parsed.stick_y, parsed.stick_theta_deg * pi / 180.0),
                     (parsed.stick2_x, parsed.stick2_y, parsed.stick2_theta_deg * pi / 180.0)),
        stick_ids=(parsed.hockey_stick_id, parsed.stick2_id),
        puck_pose=(parsed.puck_x, parsed.puck_y),
        puck_color=parsed.puck_color,
        goal_pose=(parsed.goal_x, parsed.goal_y, parsed.goal_yaw_deg * pi / 180.0),
        goal_size=(parsed.goal_width, parsed.goal_depth),
        goal_standoff=parsed.goal_standoff,
        spawn_clearance=parsed.spawn_clearance)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
