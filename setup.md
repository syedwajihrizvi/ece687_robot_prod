# RobotController Setup Guide (Skeleton)

## 1) Create Package

### Goal
Create a ros2 package where the brains of the robot will be

### Steps
1. Create a package directory:
   ```bash
   source /opt/ros/humble/setup.bash
   source /opt/ros/ws/setup.bash
   cd /ros_ws/src
   ```
2. Add package initializer:
   ```bash
   ros2 pkg create --build-type ament_python robot_controller --dependencies rclpy geometry_msgs sensor_msgs robomaster_msgs
   ```
3. Run build setup
   ```
   cd /ros_ws
   colcon build
   source install/setup.bash
   ```

---

## 2) Modify setup.py

### Goal
Define executable names in setup.py

### Add following to setup.py in control_scripts array
```
    'robot_node = robot_controller.robot:main',
    'mock_node = robot_controller.mock_robot:main'
```

---

## 3) Create Python Files

### Goal
Create project files under robot_controller/robot_controller

### Steps
```
cd robot_controller/robot_controller
touch robot.py
touch mock_robot.py
```

### Paste Content
```
Paste the content of robot.py and mock_robot.py into the docker
```
---

## 4) Build Again

### Goal
Verify scripts execute correctly.

### Typical Commands
```
cd ~/ros_ws
colcon build
source install/setup.bash

```
---

## 5) Open More Docker Containers

### Goal
Open additional docker contains with same structure and shared memory

### Process Skeleton
1. On a new terminal run docker ps
2. Get id of the current docker container
3. Run the following command on the new terminal sudo docker exec -it 3e720c9b5ddf bash
4. Setup Ros
   source /opt/ros/humble/setup.bash
   source /opt/ros/ws/setup.bash

## 6) Run the Node
1. cd ~ros_ws/src
2. Shooting or Passing Puck
For shooting
ros2 run robot_controller robot_node --robot_id 1
For passing
ros2 run robot_controller move_robot_node --robot_id 1 --pass_to_robot 2


### 7) Useful Commands
ros2 run robot_controller robot_node --robot_id 2 --l 0.35 --tolerance 0.15
ros2 run robot_controller robot_node --robot_id 2 --mock_mode --l 0.35 --tolerance 0.15
ros2 run robot_controller mock_node --robot_id 1

ros2 run robot_controller robot_node --robot_id 1 --mock_mode --l 0.35 --tolerance 0.15