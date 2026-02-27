#!/bin/bash
xhost +
docker exec -u ubuntu -w /home/ubuntu ainex /bin/zsh -c "source /home/ubuntu/ros_ws/.robotrc roslaunch ainex_bringup bringup.launch"
