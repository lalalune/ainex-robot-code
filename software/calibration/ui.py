# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(700, 650)
        Form.setMinimumSize(QtCore.QSize(700, 650))
        Form.setMaximumSize(QtCore.QSize(700, 650))
        Form.setSizeIncrement(QtCore.QSize(700, 650))
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(0, 0, 700, 700))
        self.widget.setMinimumSize(QtCore.QSize(700, 700))
        self.widget.setMaximumSize(QtCore.QSize(700, 700))
        self.widget.setStyleSheet("QWidget#widget {\n"
"background-color: #EBEBEB;\n"
"}\n"
"")
        self.widget.setObjectName("widget")
        self.label_display = QtWidgets.QLabel(self.widget)
        self.label_display.setGeometry(QtCore.QRect(30, 50, 640, 480))
        self.label_display.setMinimumSize(QtCore.QSize(640, 480))
        self.label_display.setMaximumSize(QtCore.QSize(640, 480))
        self.label_display.setSizeIncrement(QtCore.QSize(0, 0))
        self.label_display.setStyleSheet("background-color: rgb(50, 50, 60);")
        self.label_display.setText("")
        self.label_display.setObjectName("label_display")
        self.pushButton_exit = QtWidgets.QPushButton(self.widget)
        self.pushButton_exit.setGeometry(QtCore.QRect(600, 560, 70, 71))
        self.pushButton_exit.setStyleSheet("QPushButton{\n"
"background-color: #A2A2A2;\n"
"color:rgb(255, 255, 255)\n"
"}\n"
"QPushButton{border-radius:6px;}\n"
"QPushButton:pressed{\n"
"border:2px solid rgb(126, 188, 89, 0);}")
        self.pushButton_exit.setObjectName("pushButton_exit")
        self.label_camera = QtWidgets.QLabel(self.widget)
        self.label_camera.setGeometry(QtCore.QRect(290, 20, 91, 17))
        self.label_camera.setAlignment(QtCore.Qt.AlignCenter)
        self.label_camera.setObjectName("label_camera")
        self.pushButton_image = QtWidgets.QPushButton(self.widget)
        self.pushButton_image.setGeometry(QtCore.QRect(360, 560, 70, 71))
        self.pushButton_image.setStyleSheet("QPushButton{\n"
"background-color: #FFA500;\n"
"color:rgb(255, 255, 255)\n"
"}\n"
"QPushButton{border-radius:6px;}\n"
"QPushButton:pressed{\n"
"border:2px solid rgb(126, 188, 89, 0);}")
        self.pushButton_image.setObjectName("pushButton_image")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(30, 560, 321, 61))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton_servo = QtWidgets.QPushButton(self.widget)
        self.pushButton_servo.setGeometry(QtCore.QRect(440, 560, 70, 71))
        self.pushButton_servo.setStyleSheet("QPushButton{\n"
"background-color: #FFA500;\n"
"color:rgb(255, 255, 255)\n"
"}\n"
"QPushButton{border-radius:6px;}\n"
"QPushButton:pressed{\n"
"border:2px solid rgb(126, 188, 89, 0);}")
        self.pushButton_servo.setObjectName("pushButton_servo")
        self.pushButton_save = QtWidgets.QPushButton(self.widget)
        self.pushButton_save.setGeometry(QtCore.QRect(520, 560, 70, 71))
        self.pushButton_save.setStyleSheet("QPushButton{\n"
"background-color: #FFA500;\n"
"color:rgb(255, 255, 255)\n"
"}\n"
"QPushButton{border-radius:6px;}\n"
"QPushButton:pressed{\n"
"border:2px solid rgb(126, 188, 89, 0);}")
        self.pushButton_save.setObjectName("pushButton_save")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "calibration 1.0"))
        self.pushButton_exit.setText(_translate("Form", "退出"))
        self.label_camera.setText(_translate("Form", "摄像头画面"))
        self.pushButton_image.setText(_translate("Form", "校准图像"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:11pt;\">将蓝色球放到双脚之间，先点击校准图像，成功</span></p><p><span style=\" font-size:11pt;\">后再点击校准舵机，显示成功即可点击保存生效</span></p></body></html>"))
        self.pushButton_servo.setText(_translate("Form", "校准舵机"))
        self.pushButton_save.setText(_translate("Form", "保存生效"))
