#!/usr/bin/env python3
# encoding: utf-8
import os
import time
import sqlite3 as sql

class ActionGroupController:
    def __init__(self, board=None):
        self.running_action = False
        self.stop_running = False
        self.board = board
        self.action_path = os.path.split(os.path.realpath(__file__))[0]

    def stop_servo(self):
        self.board.stopBusServo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) 

    def stop_action_group(self):
        self.stop_running = True
    
    def action_finish(self):
        return self.running_action

    def run_action(self, actNum):
        '''
        运行动作组
        :param actNum: 动作组名字 ， 字符串类型
        :param times:  运行次数
        :return:
        '''
        if actNum is None:
            return
        actNum = os.path.join(self.action_path, 'ActionGroups', actNum + ".d6a")
        self.stop_running = False
        if os.path.exists(actNum) is True:
            if self.running_action is False:
                self.running_action = True
                ag = sql.connect(actNum)
                cu = ag.cursor()
                cu.execute("select * from ActionGroup")
                while True:
                    act = cu.fetchone()
                    if self.stop_running is True:
                        self.stop_running = False                   
                        break
                    if act is not None:
                        data = []
                        for i in range(0, len(act) - 2, 1):
                            data.extend([[i + 1, act[2 + i]]])
                        self.board.bus_servo_set_position(act[1]/1000.0, data)
                        time.sleep(float(act[1])/1000.0)
                    else:   # 运行完才退出
                        break
                self.running_action = False
                
                cu.close()
                ag.close()
        else:
            self.running_action = False
            print("未能找到动作组文件")
