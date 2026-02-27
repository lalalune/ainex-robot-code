#!/usr/bin/env python3
# encoding: utf-8
import os
import sys
import rospy
from ui import Ui_Form
from std_msgs.msg import String
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QToolTip
from PyQt5.QtGui import QFont
from ainex_interfaces.msg import WalkingParam
from ainex_interfaces.srv import SetWalkingParam, GetWalkingParam, SetWalkingCommand

ROS_NODE_NAME = 'walking_controller'
class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton_apply.pressed.connect(lambda: self.button_clicked('apply'))
        self.pushButton_start.pressed.connect(lambda: self.button_clicked('start'))
        self.pushButton_stop.pressed.connect(lambda: self.button_clicked('stop'))
       
        self.radioButton_zh.toggled.connect(lambda: self.language(self.radioButton_zh))
        self.radioButton_en.toggled.connect(lambda: self.language(self.radioButton_en))
        if os.environ['speaker_language'] == 'Chinese': 
            self.radioButton_zh.setChecked(True)
        else:
            self.radioButton_en.setChecked(True)
        self.param_pub = rospy.Publisher('/walking/set_param', WalkingParam, queue_size=1)
        rospy.Service('/walking_module/update_param', GetWalkingParam, self.update_param)
        rospy.wait_for_service('/walking/get_param')
        
        res = rospy.ServiceProxy('/walking/get_param', GetWalkingParam)()
        self.update_param(res)

    def language(self, name):
        QToolTip.setFont(QFont("SansSerif", 10))
        if name.text() == "中文":
            self.chinese = True
            self.label_init_x_offset.setText("init_x_offset(初始姿态在X方向(前后)偏移)")
            self.label_y_offset.setText("init_y_offset(初始姿态在Y方向(左右)偏移)")
            self.label_init_z_offset.setText("init_z_offset(初始姿态在Z方向(上下)偏移)")
            self.label_init_roll_offset.setText("init_roll_offset(初始姿态左右倾斜偏移)")
            self.label_init_pitch_offset.setText("init_pitch_offset(初始姿态前后倾斜偏移)")
            self.label_init_yaw_offset.setText("init_yaw_offset(初始姿态旋转偏移)")
            self.label_period_time.setText("period_time(周期（左右脚各抬一次腿))")
            self.label_dsp_ratio.setText("dsp_ratio(一个周期里双脚着地时间占比)")        
            self.label_step_fb_ratio.setText("step_fb_ratio(左脚和右脚间X方向的距离)")
            self.label_y_swap_amplitude.setText("y_swap_amplitude(左右摆动的距离)")
            self.label_z_swap_amplitude.setText("z_swap_amplitude(上下浮动的距离)")
            self.label_pelvis_offset.setText("pelvis_offset(臀部关节左右摆动的距离)")
            self.label_x_move_amplitude.setText("x_move_amplitude(向X方向移动距离)")
            self.label_y_move_amplitude.setText("y_move_amplitude(向Y方向移动距离)")
            self.label_z_move_amplitude.setText("z_move_amplitude(抬脚高度)")
            self.label_angle_move_amplitude.setText("angle_move_amplitude(转弯度)")
            self.label_arm_swing_gain.setText("arm_swing_gain(手臂摆动幅度)")
            self.label_hip_pitch_offset.setText("hip_pitch_offset(臀部前后倾斜角度)")
        elif name.text() == "English":
            self.chinese = False
            self.label_init_x_offset.setText("<html><head/><body><p>init_x_offset(Initial pose offset in X </p><p>direction (forward/backward))</p></body></html>")
            self.label_y_offset.setText("<html><head/><body><p>init_y_offset(Initial pose offset in Y </p><p>direction (left/right))</p></body></html>")
            self.label_init_z_offset.setText("<html><head/><body><p>init_z_offset(Initial pose offset in Z </p><p>direction (up/down))</p></body></html>")
            self.label_init_roll_offset.setText("<html><head/><body><p>init_roll_offset(Initial pose tilt offset </p><p>left/right)</p></body></html>")
            self.label_init_pitch_offset.setText("<html><head/><body><p>init_pitch_offset(Initial pose tilt offset</p><p>forward/backward)</p></body></html>")
            self.label_init_yaw_offset.setText("<html><head/><body><p>init_yaw_offset(Initial pose rotation </p><p>offset)</p></body></html>")
            self.label_period_time.setText("<html><head/><body><p>period_time(Lift one leg with each foot </p><p>alternately）</p></body></html>")
            self.label_dsp_ratio.setText("<html><head/><body><p>dsp_ratio(Percentage of time both feet </p><p>on the ground in one cycle )</p></body></html>")        
            self.label_step_fb_ratio.setText("<html><head/><body><p>step_fb_ratio(Distance between left and </p><p>right feet in X-direction)</p></body></html>")
            self.label_y_swap_amplitude.setText("<html><head/><body><p>y_swap_amplitude(Horizontal swing </p><p>distance)</p></body></html>")
            self.label_z_swap_amplitude.setText("<html><head/><body><p>z_swap_amplitude(Vertical floating </p><p>distance)</p></body></html>")
            self.label_pelvis_offset.setText("<html><head/><body><p>pelvis_offset(Horizontal swing distance </p><p>of hip joint)</p></body></html>")
            self.label_x_move_amplitude.setText("<html><head/><body><p>x_move_amplitude(Motion distance </p><p>toward X direction)</p></body></html>")
            self.label_y_move_amplitude.setText("<html><head/><body><p>y_move_amplitude(Motion distance </p><p>toward Y direction)</p></body></html>")
            self.label_z_move_amplitude.setText("<html><head/><body><p>z_move_amplitude(Lift height)</p></body></html>")
            self.label_angle_move_amplitude.setText("<html><head/><body><p>angle_move_amplitude(Turning angle)</p></body></html>")
            self.label_arm_swing_gain.setText("<html><head/><body><p>arm_swing_gain(Arm swing range)</p></body></html>")
            self.label_hip_pitch_offset.setText("<html><head/><body><p>hip_pitch_offset(Hip tilt angle </p><p>(forward/backward))</p></body></html>")

    def button_clicked(self, name):
        if name == 'start':
            rospy.ServiceProxy('/walking/command', SetWalkingCommand)('start')
        elif name == 'stop':
            rospy.ServiceProxy('/walking/command', SetWalkingCommand)('stop')
        elif name == 'apply':
            param = WalkingParam()
            param.init_x_offset = float(self.lineEdit_init_x_offset.text())
            param.init_y_offset = float(self.lineEdit_init_y_offset.text())
            param.init_z_offset = float(self.lineEdit_init_z_offset.text())
            param.init_roll_offset = float(self.lineEdit_init_roll_offset.text())
            param.init_pitch_offset = float(self.lineEdit_init_pitch_offset.text())
            param.init_yaw_offset = float(self.lineEdit_init_yaw_offset.text())
            param.period_time = float(self.lineEdit_period_time.text())
            param.dsp_ratio = float(self.lineEdit_dsp_ratio.text())

            param.step_fb_ratio = float(self.lineEdit_step_fb_ratio.text())
            param.x_move_amplitude = float(self.lineEdit_x_move_amplitude.text())
            param.y_move_amplitude = float(self.lineEdit_y_move_amplitude.text())
            param.z_move_amplitude = float(self.lineEdit_z_move_amplitude.text())
            param.angle_move_amplitude = float(self.lineEdit_angle_move_amplitude.text())
            param.move_aim_on = False#(self.lineEdit_move_aim_on.text())
            param.arm_swing_gain = float(self.lineEdit_arm_swing_gain.text())
            param.y_swap_amplitude = float(self.lineEdit_y_swap_amplitude.text())
            param.z_swap_amplitude = float(self.lineEdit_z_swap_amplitude.text())
            param.pelvis_offset = float(self.lineEdit_pelvis_offset.text())
            param.hip_pitch_offset = float(self.lineEdit_hip_pitch_offset.text())
            
            self.param_pub.publish(param)
        
    def update_param(self, msg):
        self.lineEdit_init_x_offset.setText(str(round(msg.parameters.init_x_offset, 4)))
        self.lineEdit_init_y_offset.setText(str(round(msg.parameters.init_y_offset, 4)))
        self.lineEdit_init_z_offset.setText(str(round(msg.parameters.init_z_offset, 4)))
        self.lineEdit_init_roll_offset.setText(str(round(msg.parameters.init_roll_offset, 4)))
        self.lineEdit_init_pitch_offset.setText(str(round(msg.parameters.init_pitch_offset, 4)))
        self.lineEdit_init_yaw_offset.setText(str(round(msg.parameters.init_yaw_offset, 4)))
        self.lineEdit_period_time.setText(str(round(msg.parameters.period_time, 4)))
        self.lineEdit_dsp_ratio.setText(str(round(msg.parameters.dsp_ratio, 4)))
        self.lineEdit_step_fb_ratio.setText(str(round(msg.parameters.step_fb_ratio, 4)))
        self.lineEdit_x_move_amplitude.setText(str(round(msg.parameters.x_move_amplitude, 4)))
        self.lineEdit_y_move_amplitude.setText(str(round(msg.parameters.y_move_amplitude, 4)))
        self.lineEdit_z_move_amplitude.setText(str(round(msg.parameters.z_move_amplitude, 4)))

        self.lineEdit_angle_move_amplitude.setText(str(round(msg.parameters.angle_move_amplitude, 4)))
        self.lineEdit_arm_swing_gain.setText(str(round(msg.parameters.arm_swing_gain, 4)))
        self.lineEdit_y_swap_amplitude.setText(str(round(msg.parameters.y_swap_amplitude, 4)))
        self.lineEdit_z_swap_amplitude.setText(str(round(msg.parameters.z_swap_amplitude, 4)))
        self.lineEdit_pelvis_offset.setText(str(round(msg.parameters.pelvis_offset, 4)))
        self.lineEdit_hip_pitch_offset.setText(str(round(msg.parameters.hip_pitch_offset, 4)))

if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    rospy.init_node('%s_node'%ROS_NODE_NAME, log_level=rospy.INFO)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
