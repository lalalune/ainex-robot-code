#!/usr/bin/env python3
# encoding: utf-8
# Date:2021/09/28
# Author:aiden
import os
import cv2
import sys
import yaml
import math
import rospy
import logging
import numpy as np
from imp import reload
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from std_srvs.srv import *
from sensor_msgs.msg import Image

if __name__ == '__main__':
    import message_main
    import camera_thread
    from ui import Ui_Form
    from addcolor import Ui_Dialog
else:
    from lab_tool import message_main
    from lab_tool.ui import Ui_Form
    from lab_tool import camera_thread
    from lab_tool.addcolor import Ui_Dialog

ROS_NODE_NAME = 'lab'
class LABWindow(QWidget, Ui_Form):

    def __init__(self):
        super(LABWindow, self).__init__()
        self.setupUi(self)
        self.set_window_position()
        #################################界面#######################################
        self.path = os.path.split(os.path.realpath(__file__))[0]
        self.lab_file = os.path.join(self.path, 'lab_config.yaml')
        self.color = 'red'
        self.L_Min = 0
        self.A_Min = 0
        self.B_Min = 0
        self.L_Max = 255
        self.A_Max = 255
        self.B_Max = 255
        self.kernel_erode = 3
        self.kernel_dilate = 3
        self.color_list = ['red', 'green', 'blue', 'black', 'white', 'yellow']

        self.pushButton_labWrite.pressed.connect(lambda: self.on_pushButton_action_clicked('save'))
        self.pushButton_addcolor.clicked.connect(self.addcolor)
        self.pushButton_deletecolor.clicked.connect(self.deletecolor)
        self.pushButton_camera_change.clicked.connect(self.change_camera)
        self.comboBox_color.currentIndexChanged.connect(self.selectionchange)

        self.slider_num = 6
        self.slider_object = []
        for i in range(self.slider_num):
            object_name = self.findChild(QSlider, 'horizontalSlider' + str(i + 1))
            object_name.valueChanged.connect(self.horizontalSlider_labvaluechange)
            self.slider_object.append(object_name)

        self.label_object = []
        for i in range(self.slider_num):
            object_name = self.findChild(QLabel, 'label' + str(i + 1))
            self.label_object.append(object_name)

        # +
        self.plus_object = []
        for i in range(self.slider_num):
            object_name = self.findChild(QPushButton, 'pushButton_plus' + str(i + 1))
            object_name.pressed.connect(self.button_clicked)
            object_name.released.connect(self.button_released)
            self.plus_object.append(object_name)
        
        # -
        self.reduce_object = []
        for i in range(self.slider_num):
            object_name = self.findChild(QPushButton, 'pushButton_reduce' + str(i + 1))
            object_name.pressed.connect(self.button_clicked)
            object_name.released.connect(self.button_released)
            self.reduce_object.append(object_name)
      
        self.radioButton_zn.toggled.connect(lambda: self.language(self.radioButton_zn))
        self.radioButton_en.toggled.connect(lambda: self.language(self.radioButton_en))        
        self.chinese = True
        try:
            if os.environ['speaker_language'] == 'Chinese':
                self.radioButton_zn.setChecked(True)
            else:
                self.chinese = False
                self.radioButton_en.setChecked(True)
        except:
            self.radioButton_zn.setChecked(True)

        self.size = (640, 480)
        self.display_size = (640, 480)

        self.message = message_main.Message()
        
        self.ros_camera = None
        self.current_camera = 'Mono'
        self.start_time = 0
        self.plus_pressed = False
        self.reduce_pressed = False

        # 长按检测
        self.button_timer = QTimer()
        self.button_timer.timeout.connect(self.button_pressed_check)
        
        self.createConfig()

        # 在rospy.init_node后重新加载logging，不然会失效
        reload(logging)
        logging.basicConfig(level=logging.INFO)       
           
        camera = rospy.get_param('/camera/camera_name', 'usb_cam')
        self.ros_camera = camera_thread.ROS_Camera('/%s/image_raw'%camera)

        self.ros_camera.enter()
        self.ros_camera.start()
        self.ros_camera.raw_data.connect(self.show_image)
        self.pushButton_exit.pressed.connect(self.exit)
        self.enter_func()

    def change_camera(self):
        if self.ros_camera is not None:
            self.ros_camera.exit()

        self.ros_camera.enter()
        self.ros_camera.start()
        self.ros_camera.raw_data.connect(self.show_image)
        self.selectionchange()

    def language(self, name):
        if name.text() == "中文":
            self.chinese = True
            self.label.setText("颜色列表")
            self.pushButton_addcolor.setText("新增颜色")
            self.pushButton_deletecolor.setText("删除颜色")
            self.pushButton_labWrite.setText("保存设置")
            self.pushButton_exit.setText("关闭软件")
            self.textBrowser.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">LAB是由一个亮度通道和两个颜色通道组成的，每个颜色用L、A、B三个数字表示</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- L*代表<span style=\" font-weight:600;\">亮度</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- A*代表从<span style=\" font-weight:600;\">绿色</span>到<span style=\" font-weight:600;\">红色</span>的分量</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- B*代表从<span style=\" font-weight:600;\">蓝色</span>到<span style=\" font-weight:600;\">黄色</span>的分量</p></body></html>")
            if self.pushButton_camera_change.text() == 'Mono':
                self.pushButton_camera_change.setText('单目相机')
            if self.pushButton_camera_change.text() == 'Stereo':
                self.pushButton_camera_change.setText('深度相机')
        elif name.text() == "English":
            self.chinese = False
            self.label.setText("Color list")
            self.pushButton_addcolor.setText("Add")
            self.pushButton_deletecolor.setText("Delete")
            self.pushButton_labWrite.setText("Save")
            self.pushButton_exit.setText("Quit")
            self.textBrowser.setHtml("<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">LAB is composed of one lightness channel and two color channels. And each color is represented by three values, including L, A and B</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- L*refers to <span style=\" font-weight:600;\">lightness</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- A*refers to the components from <span style=\" font-weight:600;\">green </span>to<span style=\" font-weight:600;\"> red</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">- B*refers to the components from <span style=\" font-weight:600;\">blue </span>to<span style=\" font-weight:600;\"> yellow</p></body></html>")
            if self.pushButton_camera_change.text() == '单目相机':
                self.pushButton_camera_change.setText('Mono')
            if self.pushButton_camera_change.text() == '深度相机':
                self.pushButton_camera_change.setText('Stereo')

    def set_window_position(self):
        # 窗口居中
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def log_info(self, msg):
        rospy.loginfo('\033[1;32m%s\033[0m'%msg)

    def exit(self):
        self.exit_func()
        sys.exit(0)

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
                                    "",
                                    "quit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            self.exit_func()
            QWidget.closeEvent(self, e)
        else:
            e.ignore()           

    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    # 开始主程序，显示画面
    def enter_func(self):
        self.button_timer.start(20)
        self.log_info('%s enter'%ROS_NODE_NAME)

    # 停止主程序，关闭画面显示
    def exit_func(self):
        if self.ros_camera is not None:
            self.ros_camera.exit()
        self.button_timer.stop()
        self.log_info('%s exit'%ROS_NODE_NAME)

    def get_yaml_data(self):
        f = open(self.lab_file, 'r', encoding='utf-8')
        file_data = f.read()
        f.close()
        
        data = yaml.load(file_data, Loader=yaml.FullLoader)
        
        return data

    def save_yaml_data(self, data, yaml_file):
        f = open(yaml_file, 'w', encoding='utf-8')
        data['size'] = {'width':[self.size[0]], 'height':[self.size[1]]}
        yaml.dump(data, f)

        f.close()

    def addcolor(self):
        self.qdi = QDialog()
        self.d = Ui_Dialog()
        self.d.setupUi(self.qdi)
        if self.chinese:
            self.d.label_name.setText("名称")
            self.d.pushButton_ok.setText("确定")
            self.d.pushButton_cancel.setText("取消")
        else:
            self.d.label_name.setText("name")
            self.d.pushButton_ok.setText("OK")
            self.d.pushButton_cancel.setText("Cancel")
        self.qdi.show()
        self.d.pushButton_ok.clicked.connect(self.getcolor)
        self.d.pushButton_cancel.pressed.connect(self.closeqdialog)

    def deletecolor(self):
        result = self.message_delect('删除？')
        if not result:
            self.color = self.comboBox_color.currentText()
            del self.current_lab_data['lab'][self.current_camera][self.color]
            self.save_yaml_data(self.current_lab_data, self.lab_file)

            self.comboBox_color.clear()
            self.comboBox_color.addItems(self.current_lab_data['lab'][self.current_camera].keys())

    def getcolor(self):
        color = self.d.lineEdit.text()
        if color != '':
            self.comboBox_color.addItem(color)
        else:
            self.message.info('名称不能为空！')    
        rospy.sleep(0.1)
        self.qdi.accept()
    
    def closeqdialog(self):
        self.qdi.accept()

    def horizontalSlider_labvaluechange(self):
        slider_object = self.sender()
        name = slider_object.objectName()
        value = slider_object.value()
        if name == 'horizontalSlider1': 
            self.current_lab_data['lab'][self.current_camera][self.color]['min'][0] = value
            self.label1.setNum(value)
        if name == 'horizontalSlider3':
            self.current_lab_data['lab'][self.current_camera][self.color]['min'][1] = value
            self.label3.setNum(value)
        if name == 'horizontalSlider5':
            self.current_lab_data['lab'][self.current_camera][self.color]['min'][2] = value
            self.label5.setNum(value)
        if name == 'horizontalSlider2':
            self.current_lab_data['lab'][self.current_camera][self.color]['max'][0] = value
            self.label2.setNum(value)
        if name == 'horizontalSlider4':
            self.current_lab_data['lab'][self.current_camera][self.color]['max'][1] = value
            self.label4.setNum(value)
        if name == 'horizontalSlider6':
            self.current_lab_data['lab'][self.current_camera][self.color]['max'][2] = value
            self.label6.setNum(value)

    def createConfig(self):
        if not os.path.isfile(self.lab_file):          
            data = {'lab':
                    {'Mono':
                    {'red': {'max': [255, 255, 255], 'min': [0, 150, 130]},
                    'green': {'max': [255, 110, 255], 'min': [47, 0, 135]},
                    'blue': {'max': [255, 136, 120], 'min': [0, 0, 0]},
                    'black': {'max': [89, 255, 255], 'min': [0, 0, 0]},
                    'white': {'max': [255, 255, 255], 'min': [193, 0, 0]}},
                    'Stereo':
                    {'red': {'max': [255, 255, 255], 'min': [0, 150, 130]},
                    'green': {'max': [255, 110, 255], 'min': [47, 0, 135]},
                    'blue': {'max': [255, 136, 120], 'min': [0, 0, 0]},
                    'black': {'max': [89, 255, 255], 'min': [0, 0, 0]},
                    'white': {'max': [255, 255, 255], 'min': [193, 0, 0]}}},
                    'size':
                    {'width':[self.size[0]],
                    'height':[self.size[1]]},
                    }
            self.save_yaml_data(data, self.lab_file)
            self.current_lab_data = data

            self.color_list = ['red', 'green', 'blue', 'black', 'white']
        else:
            try:
                self.current_lab_data = self.get_yaml_data()
                self.color_list = self.current_lab_data['lab']['Stereo'].keys()
            except BaseException as e:
                print('error:', e)
                self.message.info('读取保存文件出错！')

        self.comboBox_color.addItems(self.color_list)
        self.comboBox_color.currentIndexChanged.connect(self.selectionchange)
        self.selectionchange()
        
    def getColorValue(self, color):
        if color != '':
            self.current_lab_data = self.get_yaml_data()
            lab_data = self.current_lab_data['lab'][self.current_camera]
            if color in lab_data:
                self.horizontalSlider1.setValue(lab_data[color]['min'][0])
                self.horizontalSlider3.setValue(lab_data[color]['min'][1])
                self.horizontalSlider5.setValue(lab_data[color]['min'][2])
                self.horizontalSlider2.setValue(lab_data[color]['max'][0])
                self.horizontalSlider4.setValue(lab_data[color]['max'][1])
                self.horizontalSlider6.setValue(lab_data[color]['max'][2])
            else:
                self.current_lab_data['lab'][self.current_camera][color] = {'max': [255, 255, 255], 'min': [0, 0, 0]}
                self.save_yaml_data(self.current_lab_data, self.lab_file)
                
                self.horizontalSlider1.setValue(0)
                self.horizontalSlider3.setValue(0)
                self.horizontalSlider5.setValue(0)
                self.horizontalSlider2.setValue(255)
                self.horizontalSlider4.setValue(255)
                self.horizontalSlider6.setValue(255)

    def selectionchange(self):
        self.color = self.comboBox_color.currentText()      
        self.getColorValue(self.color)
        
    def on_pushButton_action_clicked(self, buttonName):
        if buttonName == 'save':
            try:               
                self.save_yaml_data(self.current_lab_data, self.lab_file)
            except Exception as e:
                print(e)
                self.message.info('保存出错！')
                return
            self.message.tips('保存成功！')
   
    def show_image(self, img):
        try:
            image = cv2.resize(img, self.display_size)
            
            frame = image.copy() 
            
            orgFrame_gb = cv2.GaussianBlur(image, (3, 3), 3)
            orgFrame_lab = cv2.cvtColor(orgFrame_gb, cv2.COLOR_RGB2LAB)
                    
            mask = cv2.inRange(orgFrame_lab,
                       (self.current_lab_data['lab'][self.current_camera][self.color]['min'][0], 
                        self.current_lab_data['lab'][self.current_camera][self.color]['min'][1], 
                        self.current_lab_data['lab'][self.current_camera][self.color]['min'][2]),
                       (self.current_lab_data['lab'][self.current_camera][self.color]['max'][0], 
                        self.current_lab_data['lab'][self.current_camera][self.color]['max'][1], 
                        self.current_lab_data['lab'][self.current_camera][self.color]['max'][2]))#对原图像和掩模进行位运算
                 
            eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_erode, self.kernel_erode)))
            dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_dilate, self.kernel_dilate)))
            
            gray_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
            image_hstack = np.hstack((gray_image, frame))
            image_resize = cv2.resize(image_hstack, (self.display_size[0]*2, self.display_size[1]))
            qimage_hstack = QImage(image_resize.data, image_resize.shape[1], image_resize.shape[0], QImage.Format_RGB888)
            qpix = QPixmap.fromImage(qimage_hstack)           
            self.label_orign.setPixmap(qpix)
        except BaseException as e:
            print(e)
    
    def button_clicked(self):
        button_object = self.sender()
        self.button_pressed = button_object
        if button_object in self.plus_object:
            self.plus_pressed = True
            self.start_time = rospy.get_time()
        elif button_object in self.reduce_object:
            self.reduce_pressed = True
            self.start_time = rospy.get_time()

    def button_released(self):
        button_object = self.sender()
        if rospy.get_time() - self.start_time <= 0.2:
            if self.plus_pressed:
                index = self.plus_object.index(self.button_pressed) + 1
                value = int(self.label_object[index - 1].text()) + 1
                self.slider_object[index - 1].setValue(value)
            elif self.reduce_pressed:
                index = self.reduce_object.index(self.button_pressed) + 1
                value = int(self.label_object[index - 1].text()) - 1
                self.slider_object[index - 1].setValue(value)    
        if button_object in self.plus_object:
            self.plus_pressed = False
        elif button_object in self.reduce_object:
            self.reduce_pressed = False

    def button_pressed_check(self):
        if self.plus_pressed:
            if rospy.get_time() - self.start_time > 0.2:
                index = self.plus_object.index(self.button_pressed) + 1
                value = int(self.label_object[index - 1].text()) + 1
                self.slider_object[index - 1].setValue(value)
        elif self.reduce_pressed:
            if rospy.get_time() - self.start_time > 0.2:
                index = self.reduce_object.index(self.button_pressed) + 1
                value = int(self.label_object[index - 1].text()) - 1
                self.slider_object[index - 1].setValue(value)           
        else:
            rospy.sleep(0.01)

if __name__ == "__main__": 
    #import faulthandler
    from loading import LoadingWindow

    #faulthandler.enable() 
    
    app = QApplication(sys.argv)
    #loading = LoadingWindow()
    #loading.show()
    rospy.init_node('%s_node'%ROS_NODE_NAME, log_level=rospy.INFO)
    main_window = LABWindow()
    #loading.main_window = main_window
    main_window.show()
    sys.exit(app.exec_())
