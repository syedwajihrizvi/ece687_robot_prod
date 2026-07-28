xhost +

# Persistent container: created on first run, survives simulator restarts.
# Full teardown when you really want it: sudo docker rm -f dji_robomaster_ros_simulator
if [ -z "$(docker ps -q -f name=^dji_robomaster_ros_simulator$)" ]; then
	docker rm -f dji_robomaster_ros_simulator 2>/dev/null
	docker run -itd --rm \
		--network=host --pid=host --ipc=host \
		--volume ./linked_folder:/linked_folder:rw \
		--volume /home/devcontainers/ece687_robot_prod:/ece687_robot_prod:rw \
		--volume /mnt/wslg/.X11-unix:/tmp/.X11-unix:rw \
		--env DISPLAY=:0 \
		--env QT_X11_NO_MITSHM=1 \
		--name="dji_robomaster_ros_simulator" dji_robomaster_ros:1.0 \
		sleep infinity
fi

# (Re)build and launch the simulator. Ctrl+C stops only the simulator, not the container.
# Extra args are forwarded to the simulator node, e.g.:  sudo bash run.sh --stick_x 1.0
docker exec -it dji_robomaster_ros_simulator /bin/bash -c \
	"source /opt/ros/humble/setup.bash && source /opt/ros/ws/setup.bash && cd /linked_folder/ros_ws_sim && colcon build && source install/setup.bash && ros2 run multi_robomaster_ros_sim simulator $*"
