#!/usr/bin/env python3
# encoding: utf-8
import os
import re
import sys
import copy
import math
import time
import rospy
from ui import Ui_Form
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtWidgets
from std_srvs.srv import Empty
from ainex_interfaces.msg import WalkingOffset
from ainex_interfaces.srv import SetWalkingOffset, GetWalkingOffset, SetWalkingOffsetRequest

class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.set_window_position()
        
        self.message = QMessageBox()
        #################################界面#######################################
        self.pushButton_save.pressed.connect(lambda: self.pushbutton_clicked('save'))
        self.pushButton_set.pressed.connect(lambda: self.pushbutton_clicked('set'))
        self.pushButton_default.pressed.connect(lambda: self.pushbutton_clicked('default'))
        self.pushButton_quit.pressed.connect(lambda: self.pushbutton_clicked('quit'))

        # 窗口居中
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        rospy.wait_for_service('/walking/get_offset')
        
        self.config = rospy.ServiceProxy('/walking/get_offset', GetWalkingOffset)().parameters
        self.doubleSpinBox_high_speed_forward_offset.setValue(self.config.high_speed_forward_offset)
        self.doubleSpinBox_high_speed_backward_offset.setValue(self.config.high_speed_backward_offset)
        self.doubleSpinBox_high_speed_move_left_offset.setValue(self.config.high_speed_move_left_offset)
        self.doubleSpinBox_high_speed_move_right_offset.setValue(self.config.high_speed_move_right_offset)
        self.doubleSpinBox_medium_speed_forward_offset.setValue(self.config.medium_speed_forward_offset)
        self.doubleSpinBox_medium_speed_backward_offset.setValue(self.config.medium_speed_backward_offset)
        self.doubleSpinBox_medium_speed_move_left_offset.setValue(self.config.medium_speed_move_left_offset)
        self.doubleSpinBox_medium_speed_move_right_offset.setValue(self.config.medium_speed_move_right_offset)
        self.doubleSpinBox_low_speed_forward_offset.setValue(self.config.low_speed_forward_offset)
        self.doubleSpinBox_low_speed_backward_offset.setValue(self.config.low_speed_backward_offset)
        self.doubleSpinBox_low_speed_move_left_offset.setValue(self.config.low_speed_move_left_offset)
        self.doubleSpinBox_low_speed_move_right_offset.setValue(self.config.low_speed_move_right_offset)
   
    def set_window_position(self):
        # 窗口居中
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def message_from(self, str):
        try:
            QMessageBox.about(self, '', str)
            time.sleep(0.01)
        except:
            pass
    
    def message_From(self, str):
        self.message_from(str)
   
    # 弹窗提示函数
    def message_delect(self, str):
        messageBox = QMessageBox()
        messageBox.setWindowTitle(' ')
        messageBox.setText(str)
        messageBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        messageBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
        return messageBox.exec_()

    # 窗口退出
    def closeEvent(self, e):        
        result = QMessageBox.question(self,
                                    "关闭窗口提醒",
                                    "exit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            QWidget.closeEvent(self, e)
        else:
            e.ignore()

    def pushbutton_clicked(self, name):
        if name == 'set' or name == 'save':
            self.config.high_speed_forward_offset = self.doubleSpinBox_high_speed_forward_offset.value()
            self.config.high_speed_backward_offset = self.doubleSpinBox_high_speed_backward_offset.value()
            self.config.high_speed_move_left_offset = self.doubleSpinBox_high_speed_move_left_offset.value()
            self.config.high_speed_move_right_offset = self.doubleSpinBox_high_speed_move_right_offset.value()
            self.config.medium_speed_forward_offset = self.doubleSpinBox_medium_speed_forward_offset.value()
            self.config.medium_speed_backward_offset = self.doubleSpinBox_medium_speed_backward_offset.value()
            self.config.medium_speed_move_left_offset = self.doubleSpinBox_medium_speed_move_left_offset.value()
            self.config.medium_speed_move_right_offset = self.doubleSpinBox_medium_speed_move_right_offset.value()
            self.config.low_speed_forward_offset = self.doubleSpinBox_low_speed_forward_offset.value()
            self.config.low_speed_backward_offset = self.doubleSpinBox_low_speed_backward_offset.value()
            self.config.low_speed_move_left_offset = self.doubleSpinBox_low_speed_move_left_offset.value()
            self.config.low_speed_move_right_offset = self.doubleSpinBox_low_speed_move_right_offset.value()
            res = rospy.ServiceProxy('/walking/set_offset', SetWalkingOffset)(SetWalkingOffsetRequest(parameters=self.config))
            if name == 'set':
                if res.result:
                    self.message_From('设置成功')
                else:
                    self.message_From('设置失败')
            elif name == 'save':
                res = rospy.ServiceProxy('/walking/save_offset', Empty)()
                self.message_From('保存成功')
        elif name == 'default':
            self.doubleSpinBox_high_speed_forward_offset.setValue(0.000)
            self.doubleSpinBox_high_speed_backward_offset.setValue(0.000)
            self.doubleSpinBox_high_speed_move_left_offset.setValue(0.000)
            self.doubleSpinBox_high_speed_move_right_offset.setValue(0.000)
            self.doubleSpinBox_medium_speed_forward_offset.setValue(0.000)
            self.doubleSpinBox_medium_speed_backward_offset.setValue(0.000)
            self.doubleSpinBox_medium_speed_move_left_offset.setValue(0.000)
            self.doubleSpinBox_medium_speed_move_right_offset.setValue(0.000)
            self.doubleSpinBox_low_speed_forward_offset.setValue(0.000)
            self.doubleSpinBox_low_speed_backward_offset.setValue(0.000)
            self.doubleSpinBox_low_speed_move_left_offset.setValue(0.000)
            self.doubleSpinBox_low_speed_move_right_offset.setValue(0.000)
        elif name == 'quit':
            sys.exit()

if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
