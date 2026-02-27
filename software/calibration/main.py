#!/usr/bin/env python3
# encoding: utf-8
# Date:2022/10/22
# Author:aiden
import os
import cv2
import sys
import rospy
import queue
import numpy as np
from ainex_sdk import common, pid
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
from ainex_example.pid_track import PIDTrack
from ainex_interfaces.msg import ColorDetect, ColorsDetect, ObjectsInfo
from ainex_kinematics.gait_manager import GaitManager
from ainex_kinematics.motion_manager import MotionManager
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QDesktopWidget

if __name__ == '__main__':
    import ui
else:
    from calibration import ui
CALIB_YAML = '/home/ubuntu/ros_ws/src/ainex_example/config/calib.yaml'
COLOR_TRACK_PID_YAML = '/home/ubuntu/ros_ws/src/ainex_example/config/color_track_pid.yaml'
class MainWindow(QWidget, ui.Ui_Form):
    image_process_size = [160, 120] 
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.set_window_position()
        
        self.center_x_offset = None
        self.rl_offset = None
        self.state = None
        self.start_calib = False
        self.last_data = 0
        self.count = 0
        self.data_queue = queue.Queue(maxsize=1)
        self.image_queue = queue.Queue(maxsize=1)
        self.display_size = (640, 480)

        self.pushButton_save.pressed.connect(lambda: self.button_clicked('save'))
        self.pushButton_exit.pressed.connect(lambda: self.button_clicked('exit'))
        self.pushButton_servo.pressed.connect(lambda: self.button_clicked('servo'))
        self.pushButton_image.pressed.connect(lambda: self.button_clicked('image')) 

        self.config = common.get_yaml_data(COLOR_TRACK_PID_YAML)['color_track']
        self.pid_rl = pid.PID(self.config['pid1_p'], self.config['pid1_i'], self.config['pid1_d'])
        self.pid_ud = pid.PID(self.config['pid2_p'], self.config['pid2_i'], self.config['pid2_d'])
       
        self.head_pan_range = [125, 875]
        self.head_tilt_range = [260, 625]
        self.head_pan_init = 500   # 左右舵机的初始值
        self.head_tilt_init = 260  # 上下舵机的初始值
        self.rl_track = PIDTrack(self.pid_rl, self.head_pan_range, self.head_pan_init)
        self.ud_track = PIDTrack(self.pid_ud, self.head_tilt_range, self.head_tilt_init)

        # 机器人行走的库调用
        self.gait_manager = GaitManager()
        self.motion_manager = MotionManager()

        self.motion_manager.set_servos_position(500, [[23, 500], [24, 260]])
        self.gait_manager.stop()
        self.motion_manager.run_action('walk_ready')
        self.detect_pub = rospy.Publisher("/color_detection/update_detect", ColorsDetect, queue_size=1)
        rospy.ServiceProxy('/color_detection/enter', Empty)()
        rospy.sleep(0.2)
        param = ColorDetect()
        param.color_name = 'blue'
        param.use_name = True
        param.detect_type = 'circle'
        param.image_process_size = self.image_process_size
        param.min_area = 10
        param.max_area = self.image_process_size[0]*self.image_process_size[1]
        self.detect_pub.publish([param])
        rospy.sleep(0.2)
        rospy.ServiceProxy('/color_detection/start', Empty)()

        rospy.Subscriber('/object/pixel_coords', ObjectsInfo, self.get_color_callback)
        self.image_sub = rospy.Subscriber('/color_detection/image_result', Image, self.image_callback, queue_size=1) 

    def set_window_position(self):
        # 窗口居中
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 弹窗提示函数
    def message_from(self, string):
        try:
            QMessageBox.about(self, '', string)
        except:
            pass
    
    # 弹窗提示函数
    def message_delect(self, string):
        messageBox = QMessageBox()
        messageBox.setWindowTitle(' ')
        messageBox.setText(string)
        messageBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        messageBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
        return messageBox.exec_()

    # 窗口退出
    def closeEvent(self, e):    
        result = QMessageBox.question(self,
                                    "Prompt box",
                                    "quit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            # 退出前先把节点退出
            if self.image_sub is not None:
                self.image_sub.unregister()
                self.image_sub = None           
            rospy.ServiceProxy('/color_detection/exit', Empty)()
            QWidget.closeEvent(self, e)
        else:
            e.ignore()

    def button_clicked(self, name):
        if name == 'save':
            if self.center_x_offset and self.rl_offset is not None:
                common.save_yaml_data({'center_x_offset': self.center_x_offset - 320, 'head_pan_offset': int(self.rl_offset) - 500}, CALIB_YAML)
                self.message_from('success')
                os.system('sudo systemctl restart start_app_node')
            else:
                self.message_from('do not detect blue')
        elif name == 'image':
            self.count = 0
            self.last_data = None
            self.motion_manager.set_servos_position(500, [[23, 500], [24, 260]])
            rospy.sleep(0.5)
            self.start_calib = True
            self.state = 'image'
            self.center_x_offset = self.data_queue.get(block=True)
            self.message_from('success')
        elif name == 'servo':
            self.count = 0
            self.last_data = None
            self.start_calib = True
            self.state = 'servo'
            self.rl_offset = self.data_queue.get(block=True)
            self.message_from('success')
        elif name == 'exit':
            if self.image_sub is not None:
                self.image_sub.unregister()
                self.image_sub = None           
            rospy.ServiceProxy('/color_detection/exit', Empty)()
            sys.exit(0)

    def process(self, center):
        if abs(center.x - center.width/2) < 10:
            center.x = center.width/2
        if abs(center.y - center.height/2) < 10:
            center.y = center.height/2
        rl_dis = self.rl_track.track(center.x, center.width/2)
        if self.start_calib:
            if self.last_data is not None:
                if abs(self.last_data - rl_dis) < 0.001:
                    self.count += 1
                else:
                    self.count = 0
                if self.count > 10:
                    self.start_calib = False
                    self.count = 0
                    self.last_data = None
                    self.state = None
                    self.data_queue.put(rl_dis)
        self.last_data = rl_dis
        ud_dis = self.ud_track.track(center.y, center.height/2)

        self.motion_manager.set_servos_position(20, [[23, int(rl_dis)], [24, int(ud_dis)]])

    def get_color_callback(self, msg):
        if msg.data != []:
            if self.state == 'image':
                center_x = msg.data[0].x
                if self.start_calib:
                    if self.last_data is not None:
                        if abs(self.last_data - center_x) < 5:
                            self.count += 1
                        else:
                            self.count = 0
                        if self.count > 5:
                            self.start_calib = False
                            self.count = 0
                            self.last_data = None
                            self.state = None
                            self.data_queue.put(center_x)
                            # self.message_from('success')
                self.last_data = center_x
            elif self.state == 'servo':
                self.process(msg.data[0])

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data) # 原始 RGB 画面
        if not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.image_queue.put_nowait(rgb_image)
        except queue.Full:
            pass
        try:
            image = self.image_queue.get(block=True)
            image_resize = cv2.resize(image, self.display_size)
            qimage = QImage(image_resize.data, image_resize.shape[1], image_resize.shape[0], QImage.Format_RGB888)
            qpix = QPixmap.fromImage(qimage)           
            self.label_display.setPixmap(qpix)
        except BaseException as e:
            print(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    rospy.init_node('calibraion_node')
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
