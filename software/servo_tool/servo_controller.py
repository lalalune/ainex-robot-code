#!/usr/bin/env python3
# encoding: utf-8
import time, os
import threading
from ros_robot_controller_sdk import Board

board = Board()

def get_servo_pulse(servo_id):
    data = board.bus_servo_read_position(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def get_servo_id(servo_id):
    data = board.bus_servo_read_id(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def get_servo_deviation(servo_id):
    data = board.bus_servo_read_offset(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def get_servo_temp_limit(servo_id):
    data = board.bus_servo_read_temp_limit(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def get_servo_angle_limit(servo_id):
    return board.bus_servo_read_angle_limit(servo_id)

def get_servo_vin_limit(servo_id):
    return board.bus_servo_read_vin_limit(servo_id)

def get_servo_vin(servo_id):
    data = board.bus_servo_read_vin(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def get_servo_temp(servo_id):
    data = board.bus_servo_read_temp(servo_id)
    if data is not None:
        return data[0]
    else:
        return None

def set_servo_pulse(servo_id, pulse, use_time):
    board.bus_servo_set_position(use_time, [[servo_id, pulse]])

def set_servo_id(old, new):
    board.bus_servo_set_id(old, new)

def set_servos_pulse(args):
    board.bus_servo_set_position(args[0], args[1:])

def set_servo_deviation(servo_id, dev):
    board.bus_servo_set_offset(servo_id, dev)
    
def save_servo_deviation(servo_id):
    board.bus_servo_save_offset(servo_id)

def set_servo_angle_limit(servo_id, min_angle, max_angle):
    board.bus_servo_set_angle_limit(servo_id, [min_angle, max_angle])

def set_servo_vin_limit(servo_id, min_vin, max_vin):
    board.bus_servo_set_vin_limit(servo_id, [min_vin, max_vin])

def set_servo_temp_limit(servo_id, temp_limit):
    board.bus_servo_set_temp_limit(servo_id, temp_limit)

def unload_servo(servo_id):
    board.bus_servo_enable_torque(servo_id, 1)

def enable_reception(enable=True):
    if enable:
        board.enable_reception(not enable)
        time.sleep(1)
        threading.Thread(target=os.system, args=("/bin/zsh -c \'source $HOME/ros_ws/.robotrc && rostopic pub /ros_robot_controller/enable_reception std_msgs/Bool \"data: true\"\'",), daemon=True).start()
        time.sleep(1)
    else:
        threading.Thread(target=os.system, args=("/bin/zsh -c \'source $HOME/ros_ws/.robotrc && rostopic pub /ros_robot_controller/enable_reception std_msgs/Bool \"data: false\"\'",), daemon=True).start()
        time.sleep(3)
        board.enable_reception(not enable)
        time.sleep(1)

if __name__ == '__main__':
    board.enable_reception(True)
    print(get_servo_id(17))
