#!/usr/bin/env python3
# encoding: utf-8
import time, os
import threading
import action_group_controller
from ros_robot_controller_sdk import Board

board = Board()
controller = action_group_controller.ActionGroupController(board)

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

def set_servo_pulse(servo_id, pulse, use_time):
    board.bus_servo_set_position(float(use_time/1000.0), [[servo_id, pulse]])

def set_servos_pulse(args):
    board.bus_servo_set_position(args[0], args[1:])

def set_servo_deviation(servo_id, dev):
    board.bus_servo_set_offset(servo_id, dev)
    
def save_servo_deviation(servo_id):
    board.bus_servo_save_offset(servo_id)

def unload_servo(servo_id):
    board.bus_servo_enable_torque(servo_id, 1)

def run_action_group(num):
    threading.Thread(target=controller.run_action, args=(num, )).start()    

def stop_action_group():    
    controller.stop_action_group() 

def wait_for_finish():
    while True:
        if controller.action_finish():
            time.sleep(0.01)
        else:
            break

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
