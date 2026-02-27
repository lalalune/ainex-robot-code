#!/usr/bin/env python3
# encoding: utf-8
import os
import re
import sys
import copy
import time
import sqlite3
import ui_ainex
import ui_ainexpro
import threading
import resource_rc
import faulthandler
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from servo_controller import *
from functools import partial
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtSql import QSqlDatabase, QSqlQuery

faulthandler.enable()

class MainWindow(QtWidgets.QWidget, ui_ainex.Ui_Form, ui_ainexpro.Ui_Form):
    def __init__(self):   
        super(MainWindow, self).__init__()
        enable_reception(False)
        self.hand_form = False
        self.servo_special = [17, 18, 21, 22]
        for i in self.servo_special: # 检测形态
            servo_id = get_servo_id(i)
            if servo_id is not None:
                self.hand_form = True
                break
        if not self.hand_form:
            self.setupUi_ainex(self)
            self.widget_id17.hide()
            self.widget_id18.hide()
            self.widget_id21.hide()
            self.widget_id22.hide()
        else:
            self.setupUi_ainexpro(self)        
        
        self.center()
        self.min_time = 50
        self.chinese = True
        self.running = False
        self.start_run = True
        self.save_temp = False
        self.open_action = True
        self.editaction_clicked = False
        self.horizontalSlider_manual = True
        self.horizontalSlider_dev_manual = True
        self.path = '/home/ubuntu/software/ainex_controller'
        self.actdir = os.path.join(self.path, "ActionGroups")    
        self.temp_action_name = 'temp'
        self.temp_path = os.path.join(self.actdir, self.temp_action_name + '.d6a')
        self.setWindowIcon(QIcon(os.path.join(self.path, 'images/AiNexPro.ico')))
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置选中整行，若不设置默认选中单元格
        self.message = QMessageBox()
        self.reflash_action()

        ########################主界面###############################
        self.radioButton_zn.toggled.connect(lambda: self.language(self.radioButton_zn))
        self.radioButton_en.toggled.connect(lambda: self.language(self.radioButton_en))
        language = os.environ.get('speaker_language')
        if language == 'Chinese':
            self.radioButton_zn.setChecked(True)
        else:
            self.radioButton_en.setChecked(True)
        self.joints = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]      
        self.lineEdit_object = {}    
        for i in self.joints:
            self.lineEdit_object[i] = self.findChild(QLineEdit, "lineEdit_id%d" % i)            
        self.validator = QIntValidator(0, 1000)
        for i in self.joints:
            self.lineEdit_object[i].setValidator(self.validator)
            self.lineEdit_object[i].installEventFilter(self)

        self.current_positions = {}
        self.horizontalSlider_pulse_object = {}
        for i in self.joints:
            self.horizontalSlider_pulse_object[i] = self.findChild(QSlider, "horizontalSlider_pulse_id%d" % i)
            # 滑竿同步对应文本框的数值,及滑竿控制相应舵机转动与valuechange函数绑定
            self.horizontalSlider_pulse_object[i].valueChanged.connect(partial(self.pulse_valuechange, i))

            self.current_positions[i] = self.horizontalSlider_pulse_object[i].value()

        self.horizontalSlider_dev_object = {}
        for i in self.joints:
            self.horizontalSlider_dev_object[i] = self.findChild(QSlider, "horizontalSlider_dev_id%d" % i)        
            self.horizontalSlider_dev_object[i].valueChanged.connect(partial(self.dev_valuechange, i))
        
        self.label_object = {}
        for i in self.joints:
            self.label_object[i] = self.findChild(QLabel, "label_id%d" % i) 
        
        self.checkBox_object = {}
        for i in self.joints:
            self.checkBox_object[i] = self.findChild(QCheckBox, "checkBox_id%d" % i)    

        self.checkBox_left_arm.stateChanged.connect(lambda: self.checkBox_valuechange('left_arm'))
        self.checkBox_right_arm.stateChanged.connect(lambda: self.checkBox_valuechange('right_arm'))
        self.checkBox_left_leg.stateChanged.connect(lambda: self.checkBox_valuechange('left_leg'))
        self.checkBox_right_leg.stateChanged.connect(lambda: self.checkBox_valuechange('right_leg'))
        # tableWidget点击获取定位的信号与icon_position函数（添加运行图标）绑定
        self.tableWidget.clicked.connect(self.icon_position)
        self.validator3 = QIntValidator(20, 30000)
        self.lineEdit_time.setValidator(self.validator3)

        # 将编辑动作组的按钮点击时的信号与button_editaction_clicked函数绑定
        self.Button_ServoPowerDown.pressed.connect(lambda: self.button_editaction_clicked('servoPowerDown'))
        self.Button_AngularReadback.pressed.connect(lambda: self.button_editaction_clicked('angularReadback'))
        self.Button_AddAction.pressed.connect(lambda: self.button_editaction_clicked('addAction'))
        self.Button_DelectAction.pressed.connect(lambda: self.button_editaction_clicked('delectAction'))
        
        self.Button_UpdateAction.pressed.connect(lambda: self.button_editaction_clicked('updateAction'))
        self.Button_InsertAction.pressed.connect(lambda: self.button_editaction_clicked('insertAction'))
        self.Button_MoveUpAction.pressed.connect(lambda: self.button_editaction_clicked('moveUpAction'))
        self.Button_MoveDownAction.pressed.connect(lambda: self.button_editaction_clicked('moveDownAction'))        
        self.Button_Exit.clicked.connect(self.exit)
        
        # 将运行及停止运行按钮点击的信号与button_runonline函数绑定
        self.Button_Run.clicked.connect(lambda: self.button_run('run'))
        
        self.icon = QtGui.QIcon()
        self.icon.addPixmap(QtGui.QPixmap(":/images/index.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        
        self.Button_OpenActionGroup.pressed.connect(lambda: self.button_flie_operate('openActionGroup'))
        self.Button_SaveActionGroup.pressed.connect(lambda: self.button_flie_operate('saveActionGroup'))
        self.Button_ReadDeviation.pressed.connect(lambda: self.button_flie_operate('readDeviation'))
        self.Button_DownloadDeviation.pressed.connect(lambda: self.button_flie_operate('downloadDeviation'))
        self.Button_TandemActionGroup.pressed.connect(lambda: self.button_flie_operate('tandemActionGroup'))
        self.Button_ReSetServos.pressed.connect(lambda: self.button_re_clicked('reSetServos'))
        self.Button_RePlace.pressed.connect(lambda: self.button_flie_operate('replace'))
        
        self.validator = QIntValidator(1, 1000)
        self.lineEdit_speed.setValidator(self.validator)
        self.checkBox_speed.stateChanged.connect(self.speed)

        # 将控制动作的按钮点击的信号与action_control_clicked函数绑定
        self.Button_RunAction.pressed.connect(lambda: self.button_controlaction_clicked('runAction'))
        self.Button_StopAction.pressed.connect(lambda: self.button_controlaction_clicked('stopAction'))                
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)  #允许右键产生子菜单
        self.tableWidget.customContextMenuRequested.connect(self.generateMenu)  #右键菜单
        self.Button_run = QtWidgets.QToolButton()
        self.Button_run.setIcon(self.icon)
        self.country_item = QTableWidgetItem(self.icon, '')  
        self.devNew = len(self.joints)*[0]
        self.dev_change = False 
        self.readDevOk = False
        self.totalTime = 0
        self.row = 0
        
        self.origin_speed = []
        self.use_time_list = []
        self.loop = False
        self.update = True
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_position)
        self.update_timer.start(25)

    def update_position(self):
        if not self.update or not self.horizontalSlider_manual:
            return
        for i in self.joints:
            p = self.horizontalSlider_pulse_object[i].value()
            if abs(p - self.current_positions[i]) > 0.01:
                p = self.current_positions[i] * 0.8 + 0.2 * p
                set_servo_pulse(i, int(p), 20)
                self.current_positions[i] = p

    #居中
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def generateMenu(self, pos):
        menu = QMenu()
        if self.chinese:
            item1 = menu.addAction("删除全部动作")
            item2 = menu.addAction("镜像动作")
            item3 = menu.addAction("镜像全部动作")
        else:
            item1 = menu.addAction("Delect all action")
            item2 = menu.addAction("Mirror action")
            item3 = menu.addAction("Mirror all action")
        action = menu.exec_(self.tableWidget.mapToGlobal(pos))
        if action == item1:
            self.button_editaction_clicked("delectAllAction")
        elif action == item2:
            self.button_editaction_clicked("mirrorAction")
        elif action == item3:
            self.button_editaction_clicked("mirrorAllAction")
        else:
            return

    def exit(self):
        sys.exit(0)

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
                                    "",
                                    "Quit?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)
        if result == QMessageBox.Yes:
            enable_reception(True)
            QWidget.closeEvent(self, e)           
        else:
            e.ignore()

    def language(self, name):
        QToolTip.setFont(QFont("SansSerif", 10))
        if name.text() == "中文":
            self.chinese = True
            self.Button_ServoPowerDown.setText("手掰编程")
            self.Button_AngularReadback.setText("角度回读")
            self.Button_AddAction.setText("添加动作")
            self.Button_DelectAction.setText("删除动作")
            self.Button_UpdateAction.setText("更新动作")
            self.Button_InsertAction.setText("插入动作")
            self.Button_MoveUpAction.setText("上移动作")
            self.Button_MoveDownAction.setText("下移动作")        
            self.Button_OpenActionGroup.setText("打开动作文件")
            self.Button_SaveActionGroup.setText("保存动作文件")
            self.Button_ReadDeviation.setText("读取偏差")
            self.Button_DownloadDeviation.setText("下载偏差")
            self.Button_TandemActionGroup.setText("串联动作文件")
            self.Button_ReSetServos.setText("舵机回中")
            self.Button_RunAction.setText("动作运行")
            self.Button_StopAction.setText("动作停止")
            self.Button_Run.setText("运行")
            self.Button_Exit.setText("退出")
            self.checkBox_time.setToolTip("锁定时间")
            self.checkBox.setText("循环")
            self.checkBox_speed.setToolTip('生效')
            self.label_time.setText("时间")
            self.label_speed.setText("速度")
            self.label_total_time.setText("运行总时间")
            self.label_action.setText("动作组")
            self.Button_RePlace.setText("覆盖保存动作")
        elif name.text() == "English":
            self.chinese = False
            self.Button_ServoPowerDown.setText("Manual")
            self.Button_AngularReadback.setText("Read angle")
            self.Button_AddAction.setText("Add action")
            self.Button_DelectAction.setText("Delete action")
            self.Button_UpdateAction.setText("Update action")
            self.Button_InsertAction.setText("Insert action")
            self.Button_MoveUpAction.setText("Action upward")
            self.Button_MoveDownAction.setText("Action down")        
            self.Button_OpenActionGroup.setText("Open action file")
            self.Button_SaveActionGroup.setText("Save action file")
            self.Button_ReadDeviation.setText("Read deviation")
            self.Button_DownloadDeviation.setText("Download deviation")
            self.Button_TandemActionGroup.setText("Integrate file")
            self.Button_ReSetServos.setText("Reset servo")
            self.Button_RunAction.setText("Run action")
            self.Button_StopAction.setText("Stop")
            self.Button_Run.setText("Run")           
            self.Button_Exit.setText("Exit")
            self.checkBox_time.setToolTip("Lock time")
            self.checkBox.setText("Loop")
            self.checkBox_speed.setToolTip('Effect')
            self.label_time.setText("Duration")
            self.label_speed.setText("Speed")
            self.label_total_time.setText("Total duration")  
            self.label_action.setText("Action list")
            self.Button_RePlace.setText("Overwrite act")
    
    def eventFilter(self, widget, event):
        if event.type() == QEvent.KeyPress:
            if event.key()== 16777220 or event.key() == 16777221:
                self.horizontalSlider_manual = False
                for i in self.joints:
                    pulse = int(self.lineEdit_object[i].text())
                    self.horizontalSlider_pulse_object[i].setValue(pulse)
                    self.current_positions[i] = pulse
                    set_servo_pulse(i, pulse, 500)
                self.horizontalSlider_manual = True       
        
        return False

    # 滑竿同步对应文本框的数值,及滑竿控制相应舵机转动
    def pulse_valuechange(self, name):
        if self.horizontalSlider_manual:
            servo_pulse = self.horizontalSlider_pulse_object[name].value()
            self.lineEdit_object[name].setText(str(servo_pulse))   

    def dev_valuechange(self, name):
        if self.horizontalSlider_dev_manual:
            self.devNew[0] = self.horizontalSlider_dev_object[name].value()
            set_servo_deviation(name, self.devNew[0])
            self.label_object[name].setText(str(self.devNew[0]))
            if self.devNew[0] < 0:
                self.devNew[0] = 0xff + self.devNew[0] + 1

    def checkBox_valuechange(self, name):
        if name == 'left_arm':
            for i in self.joints:
                if 13 <= id_ <= 21 and id_%2 != 0:
                    if self.checkBox_left_arm.isChecked():
                        self.checkBox_object[i].setChecked(True)
                    else:
                        self.checkBox_object[i].setChecked(False)
        elif name == 'right_arm':
            for i in self.joints:
                if 14 <= id_ <= 22 and id_%2 == 0:
                    if self.checkBox_right_arm.isChecked():
                        self.checkBox_object[i].setChecked(True)
                    else:
                        self.checkBox_object[i].setChecked(False)
        elif name == 'left_leg':
            for i in self.joints:
                if 1 <= id_ <= 11 and id_%2 != 0:
                    if self.checkBox_left_leg.isChecked():
                        self.checkBox_object[i].setChecked(True)
                    else:
                        self.checkBox_object[i].setChecked(False)
        elif name == 'right_leg':
            for i in self.joints:
                if 2 <= id_ <= 12 and id_%2 == 0:
                    if self.checkBox_right_leg.isChecked():
                        self.checkBox_object[i].setChecked(True)
                    else:
                        self.checkBox_object[i].setChecked(False)

    # 复位按钮点击事件
    def button_re_clicked(self, name):
        self.horizontalSlider_manual = False
        if name == 'reSetServos':
            for i in self.joints:
                self.horizontalSlider_pulse_object[i].setValue(500)
                self.current_positions[i] = 500
                set_servo_pulse(i, 500, 1000)
                self.lineEdit_object[i].setText('500')
            self.horizontalSlider_manual = True

    # 选项卡选择标签状态，获取对应舵机数值
    def tabindex(self, index):       
        return [str(self.horizontalSlider_pulse_object[i].value()) for i in self.joints]
    
    def getIndexData(self, index):
        return [str(self.tableWidget.item(index, j).text()) for j in range(2, self.tableWidget.columnCount())]
    
    # 往tableWidget表格添加一行数据的函数
    def add_line(self, data):
        self.tableWidget.setItem(data[0], 1, QtWidgets.QTableWidgetItem(str(data[0] + 1)))       
        for i in range(2, len(data) + 1):          
            self.tableWidget.setItem(data[0], i, QtWidgets.QTableWidgetItem(data[i - 1]))

    # 在定位行添加运行图标按钮
    def icon_position(self):
        if self.tableWidget.currentColumn() == 0:
            if not self.running:
                self.Button_run = QtWidgets.QToolButton()
                self.Button_run.setIcon(self.icon) 
                item = self.tableWidget.currentRow()
                for i in range(self.tableWidget.rowCount()):
                    if i != item:
                        self.tableWidget.removeCellWidget(i, 0)
                if self.open_action:
                    self.open_action = False
                else:
                    if self.editaction_clicked:
                        self.editaction_clicked = False
                        self.tableWidget.setCellWidget(item, 0, self.Button_run)
                        self.Button_run.clicked.connect(self.icon_position)
                    else:
                        self.tableWidget.setCellWidget(item, 0, self.Button_run)
                        self.Button_run.clicked.connect(self.icon_position)
                        self.action_one()

    def action_one(self):  
        self.horizontalSlider_manual = False
        item = self.tableWidget.currentRow()
        try:
            timer = int(self.tableWidget.item(self.tableWidget.currentRow(), 2).text())
            data = [timer/1000.0]
            if not self.checkBox_time.isChecked():
                self.lineEdit_time.setText(str(timer))
            for ID in self.joints:
                if not self.checkBox_object[ID].isChecked():
                    pulse = int(self.tableWidget.item(item, ID + 2).text())
                    self.current_positions[ID] = pulse
                    self.horizontalSlider_pulse_object[ID].setValue(pulse)
                    self.lineEdit_object[ID].setText(str(pulse))
                    data.extend([[ID, pulse]])
            set_servos_pulse(data)
        except BaseException as e:
            print(e)
            if self.chinese:
                self.message_from('运行出错')
            else:
                self.message_from('Running error')
        self.horizontalSlider_manual = True

    # 编辑动作组按钮点击事件
    def button_editaction_clicked(self, name):        
        self.editaction_clicked = True
        list_data = self.tabindex(self.tableWidget.currentIndex())
        RowCont = self.tableWidget.rowCount()
        item = self.tableWidget.currentRow()
        if name == 'servoPowerDown':
            for ID in self.joints:
                if not self.checkBox_object[ID].isChecked():
                    unload_servo(ID)
            if self.chinese:
                self.message_from('掉电成功')
            else:
                self.message_from('success')
        if name == 'angularReadback':

            use_time = int(self.lineEdit_time.text())
            data = [RowCont, str(use_time)]
            read_error = False
            ID = ''
            for i in self.joints:
                if self.hand_form:
                    pulse = get_servo_pulse(i)                                        
                else:
                    if i in self.servo_special:
                        pulse = 500
                    else:
                        pulse = get_servo_pulse(i)       
                if pulse is None:
                    ID += (' id' + str(i))
                    read_error = True
                else:
                    data.append(str(pulse))    
                        
            if use_time < 20:
                if self.chinese:
                    self.message_from('运行时间必须大于20ms')
                else:
                    self.message_from('Run time must be greater than 20ms')
                return   
            if not read_error:
                self.tableWidget.insertRow(RowCont)    # 增加一行
                self.tableWidget.selectRow(RowCont)    # 定位最后一行为选中行
                self.add_line(data)
                self.totalTime += use_time
                self.label_TotalTime.setText(str((self.totalTime)/1000.0)) 
            else:
                if self.chinese:
                    self.message_from(ID + '号舵机位置读取失败!')
                else:
                    self.message_from('Failed to read the position of' + ID)
        if name == 'addAction':    # 添加动作
            use_time = int(self.lineEdit_time.text())
            data = [RowCont, str(use_time)]            
            if use_time < 20:
                if self.chinese:
                    self.message_from('运行时间必须大于20ms')
                else:
                    self.message_from('Run time must be greater than 20ms')
                return
            self.tableWidget.insertRow(RowCont)    # 增加一行
            self.tableWidget.selectRow(RowCont)    # 定位最后一行为选中行
            data.extend(list_data)
            self.add_line(data)
            self.totalTime += use_time
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'delectAction':    # 删除动作
            if RowCont != 0:
                self.totalTime -= int(self.tableWidget.item(item, 2).text())
                self.tableWidget.removeRow(item)  # 删除选定行                
                self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'delectAllAction':
            if self.chinese:
                result = self.message_delect('此操作会删除列表中的所有动作，是否继续？')
            else:
                result = self.message_delect('This operation will delete all actions in the list. Do you want to continue?')
                
            if result == 0:                              
                for i in range(RowCont):
                    self.tableWidget.removeRow(0)
                self.totalTime = 0
                self.label_TotalTime.setText(str(self.totalTime))
            else:
                pass          
        if name == 'updateAction':    # 更新动作
            use_time = int(self.lineEdit_time.text())
            data = [item, str(use_time)]            
            if use_time < 20:
                if self.chinese:
                    self.message_from('运行时间必须大于20ms')
                else:
                    self.message_from('Run time must be greater than 20ms')
                return
            data.extend(list_data)
            self.add_line(data)
            self.totalTime = 0
            for i in range(RowCont):
                self.totalTime += int(self.tableWidget.item(i,2).text())
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'mirrorAction':
            list_data = self.getIndexData(item)
            use_time  = int(list_data[0])
            list_data = list_data[1:]
            data = [item, str(use_time)]           
            if use_time < 20:
                if self.chinese:
                    self.message_from('运行时间必须大于20ms')
                else:
                    self.message_from('Run time must be greater than 20ms')
                return
            
            odd_id = [1,3,5,7,9,11,13,15,17,19,21]
            for i in odd_id:
                if int(list_data[i-1]) + int(list_data[i]) != 1000:
                    tmp = int(list_data[i-1])
                    list_data[i-1] = str(1000 - int(list_data[i]))
                    list_data[i] = str(1000 - tmp)
                    
            data.extend(list_data)
            self.add_line(data)
        if name == 'mirrorAllAction':
            for item in range(RowCont):
                list_data = self.getIndexData(item)
                use_time = int(list_data[0])
                list_data = list_data[1:]
                data = [item, str(use_time)]           
                if use_time < 20:
                    if self.chinese:
                        self.message_from('运行时间必须大于20ms')
                    else:
                        self.message_from('Run time must be greater than 20ms')
                    return
                
                odd_id = [1,3,5,7,9,11,13,15,17,19,21]
                for i in odd_id:
                    if int(list_data[i-1]) + int(list_data[i]) != 1000:
                        tmp = int(list_data[i-1])
                        list_data[i-1] = str(1000 - int(list_data[i]))
                        list_data[i] = str(1000 - tmp)
                        
                data.extend(list_data)
                self.add_line(data)           
        if name == 'insertAction':    # 插入动作
            if item == -1:
                return
            use_time = int(self.lineEdit_time.text())
            data = [item, str(use_time)]            
            if use_time < 20:
                if self.chinese:
                    self.message_from('运行时间必须大于20ms')
                else:
                    self.message_from('Run time must be greater than 20ms')
                return

            self.tableWidget.insertRow(item)       # 插入一行
            self.tableWidget.selectRow(item)
            data.extend(list_data)
            self.add_line(data)
            self.totalTime += use_time
            self.label_TotalTime.setText(str((self.totalTime)/1000.0))
        if name == 'moveUpAction':
            data_new = [item - 1]
            data = [item]
            if item == 0 or item == -1:
                return
            current_data = self.getIndexData(item)
            uplist_data = self.getIndexData(item - 1)
            data_new.extend(current_data)
            data.extend(uplist_data)
            self.add_line(data_new)           
            self.add_line(data)
            self.tableWidget.selectRow(item - 1) 
        if name == 'moveDownAction':
            data_new = [item + 1]
            data = [item]            
            if item == RowCont - 1:
                return
            current_data = self.getIndexData(item)
            downlist_data = self.getIndexData(item + 1)
            data_new.extend(current_data)
            data.extend(downlist_data)
            self.add_line(data_new)           
            self.add_line(data)            
            self.tableWidget.selectRow(item + 1)
                             
        for i in range(self.tableWidget.rowCount()):    #刷新编号值
            self.tableWidget.item(i , 2).setFlags(self.tableWidget.item(i , 2).flags() & ~Qt.ItemIsEditable)
            self.tableWidget.setItem(i,1,QtWidgets.QTableWidgetItem(str(i + 1)))
        self.icon_position()
    
    def speed(self):
        RowCont = self.tableWidget.rowCount()
        status = self.checkBox_speed.isChecked()
        self.totalTime = 0
        if status:
            speed_value = int(self.lineEdit_speed.text()) / 100.0
            for item in range(RowCont):
                list_data = self.getIndexData(item)
                use_time = int(list_data[0])
                use_time /= speed_value
                if use_time > 30000:
                    use_time = 30000
                if use_time < 20:
                    use_time = 20
                self.tableWidget.setItem(item, 2, QtWidgets.QTableWidgetItem(str(int(round(use_time)))))
                self.totalTime += use_time
            self.label_TotalTime.setText(str(round((self.totalTime)/1000.0, 3)))
        else:
            speed_value = int(self.lineEdit_speed.text())
            self.lineEdit_speed.setText('100')
            speed_value = speed_value / 100.0
            for item in range(RowCont):
                list_data = self.getIndexData(item)
                use_time = int(list_data[0])
                use_time *= speed_value
                if use_time > 30000:
                    use_time = 30000
                if use_time < 20:
                    use_time = 20
                self.tableWidget.setItem(item, 2, QtWidgets.QTableWidgetItem(str(int(round(use_time)))))
                self.totalTime += use_time
            self.label_TotalTime.setText(str(round((self.totalTime)/1000.0, 3)))

    # 在线运行按钮点击事件
    def button_run(self, name):
        self.update = False
        if self.tableWidget.rowCount() == 0:
            if self.chinese:
                self.message_from('请先添加动作!')
            else:
                self.message_from('Please add action first!')
            self.update = True
        else:
            if name == 'run':
                try:
                    if not self.running:
                        self.save_temp = True
                        self.button_flie_operate('saveActionGroup')
                        self.save_temp = False
                        self.running = True
                        if self.Button_Run.text() == 'Run':
                            self.Button_Run.setText('Stop')
                        else:
                            self.Button_Run.setText('停止')                        
                        
                        self.row = self.tableWidget.currentRow()
                        self.label_index.setText(str(self.row))
                        if self.checkBox.isChecked():
                            self.loop = True
                        else:
                            self.loop = False
                        self.time_last = 0
                        self.start_run = True
                        self.timer = QTimer()      
                        self.tableWidget.setUpdatesEnabled(False)
                        self.timer.setTimerType(Qt.PreciseTimer)
                        self.timer.timeout.connect(self.runOline)
                        self.use_time_list = [0]
                        for i in range(self.tableWidget.rowCount() - self.row):
                            use_time = int(self.tableWidget.item(i, 2).text())
                            self.use_time_list.append(use_time)
                        self.use_time_list_ = copy.deepcopy(self.use_time_list)
                        self.timer.start(self.use_time_list_[0])
                        run_action_group(self.temp_action_name)
                    else:
                        stop_action_group()
                        self.tableWidget.setUpdatesEnabled(True)
                        for i in range(self.tableWidget.currentRow()):
                            if i != self.tableWidget.currentRow():
                                self.tableWidget.removeCellWidget(i, 0)
                        self.timer.stop()
                        self.update = True
                        self.running = False
                        if self.Button_Run.text() == 'Stop':
                            self.Button_Run.setText('Run')
                        else:
                            self.Button_Run.setText('运行')
                        if self.chinese:
                            self.message_from('运行结束!')
                        else:
                            self.message_from('Run over!')
                        self.Button_run = QtWidgets.QToolButton()
                        self.Button_run.setIcon(self.icon)
                        self.label_index.setText(str(self.tableWidget.currentRow() + 1))
                        self.tableWidget.setCellWidget(self.tableWidget.currentRow(), 0, self.Button_run)
                except BaseException as e:
                    print(e)   

    def runOline(self):
        self.use_time_list_.remove(self.use_time_list_[0])
        if self.use_time_list_ == []:
            if self.loop:
                self.use_time_list_ = copy.deepcopy(self.use_time_list)
                self.use_time_list_.remove(self.use_time_list_[0])

                self.tableWidget.removeCellWidget(self.tableWidget.currentRow(), 0)
                self.Button_run = QtWidgets.QToolButton()
                self.Button_run.setIcon(self.icon)
                self.tableWidget.selectRow(self.row)
                self.label_index.setText(str(self.row))
                self.tableWidget.setCellWidget(self.row, 0, self.Button_run)
                
                self.timer.start(self.use_time_list_[0])
                wait_for_finish()
                run_action_group(self.temp_action_name)
                self.start_run = False
            else:
                self.tableWidget.setUpdatesEnabled(True)
                self.tableWidget.removeCellWidget(self.tableWidget.currentRow(), 0)
                self.timer.stop()
                self.running = False
                self.update = True
                if self.Button_Run.text() == 'Stop':
                    self.Button_Run.setText('Run')
                else:
                    self.Button_Run.setText('运行')
                if self.chinese:
                    self.message_from('运行结束!')
                else:
                    self.message_from('Run over!')
                self.Button_run = QtWidgets.QToolButton()
                self.Button_run.setIcon(self.icon)
                self.tableWidget.selectRow(self.row)
                self.label_index.setText(str(self.row))
                self.tableWidget.setCellWidget(self.row, 0, self.Button_run)
        else:
            if not self.start_run:
                current_row = self.tableWidget.currentRow() + 1
                
                # self.tableWidget.selectRow(current_row)
                self.tableWidget.setCurrentCell(current_row, 0)
                self.label_index.setText(str(current_row))
            # self.tableWidget.setCellWidget(self.tableWidget.currentRow(), 0, self.Button_run)       
            self.timer.start(self.use_time_list_[0])
            self.start_run = False

    # 文件打开及保存按钮点击事件
    def button_flie_operate(self, name):        
        self.open_action = True
        try:            
            if name == 'openActionGroup':
                self.open_action = True
                dig_o = QFileDialog()
                dig_o.setFileMode(QFileDialog.ExistingFile)
                dig_o.setNameFilter('d6a Flies(*.d6a)')
                openfile = dig_o.getOpenFileName(self, 'OpenFile', '', 'd6a Flies(*.d6a)')
                # 打开单个文件
                # 参数一：设置父组件；参数二：QFileDialog的标题
                # 参数三：默认打开的目录，“.”点表示程序运行目录，/表示当前盘符根目录
                # 参数四：对话框的文件扩展名过滤器Filter，比如使用 Image files(*.jpg *.gif) 表示只能显示扩展名为.jpg或者.gif文件
                # 设置多个文件扩展名过滤，使用双引号隔开；“All Files(*);;PDF Files(*.pdf);;Text Files(*.txt)”
                path = openfile[0]
                try:
                    if path != '':
                        rbt = QSqlDatabase.addDatabase("QSQLITE")
                        rbt.setDatabaseName(path)
                        if rbt.open():
                            actgrp = QSqlQuery()
                            if (actgrp.exec("select * from ActionGroup ")):
                                self.tableWidget.setRowCount(0)
                                self.tableWidget.clearContents()
                                self.totalTime = 0
                                while (actgrp.next()):
                                    count = self.tableWidget.rowCount()
                                    self.tableWidget.setRowCount(count + 1)
                                    for i in range(len(self.joints) + 2):
                                        self.tableWidget.setItem(count, i + 1, QtWidgets.QTableWidgetItem(str(actgrp.value(i))))
                                        if i == 1:
                                            self.totalTime += actgrp.value(i)
                                        self.tableWidget.update()
                                        self.tableWidget.selectRow(count)
                                    self.tableWidget.item(count , 2).setFlags(self.tableWidget.item(count , 2).flags() & ~Qt.ItemIsEditable)                        
                        rbt.close()
                        self.label_TotalTime.setText(str(self.totalTime/1000.0))
                        self.tableWidget.selectRow(0)
                        self.icon_position()                    
                except BaseException as e:
                    print(e)
                    if self.chinese:
                        self.message_from('错误的动作组格式')
                    else:
                        self.message_from('Wrong action group format')
                self.label_action_number.setText(str(path))                                           
            if name == 'saveActionGroup':
                if not self.save_temp:
                    dig_s = QFileDialog()
                    if self.tableWidget.rowCount() == 0:
                        if self.chinese:
                            self.message_from('动作列表是空的哦，没啥要保存的')
                        else:
                            self.message_from('The action list is empty, nothing to save')                      
                        return
                    savefile = dig_s.getSaveFileName(self, 'Savefile', '', 'd6a Flies(*.d6a)')
                    path = savefile[0]
                else:
                    path = self.temp_path
                if os.path.isfile(path):
                    os.system('sudo rm ' + path)
                if path != '':                    
                    if path[-4:] == '.d6a':
                        conn = sqlite3.connect(path)
                    else:
                        conn = sqlite3.connect(path + '.d6a')
                    
                    c = conn.cursor()                    
                    c.execute('''CREATE TABLE ActionGroup([Index] INTEGER PRIMARY KEY AUTOINCREMENT
                    NOT NULL ON CONFLICT FAIL
                    UNIQUE ON CONFLICT ABORT,
                    Time INT,
                    Servo1 INT,
                    Servo2 INT,
                    Servo3 INT,
                    Servo4 INT,
                    Servo5 INT,
                    Servo6 INT,
                    Servo7 INT,
                    Servo8 INT,
                    Servo9 INT,
                    Servo10 INT,
                    Servo11 INT,
                    Servo12 INT,
                    Servo13 INT,
                    Servo14 INT,
                    Servo15 INT,                    
                    Servo16 INT,
                    Servo17 INT,
                    Servo18 INT,
                    Servo19 INT,
                    Servo20 INT,
                    Servo21 INT,
                    Servo22 INT);''')                      
                    for i in range(self.tableWidget.rowCount()):
                        insert_sql = "INSERT INTO ActionGroup(Time,Servo1,Servo2,Servo3,Servo4,Servo5,Servo6,Servo7,Servo8,Servo9,Servo10,Servo11,Servo12,Servo13,Servo14,Servo15,Servo16,Servo17,Servo18,Servo19,Servo20,Servo21,Servo22) VALUES("
                        for j in range(2, self.tableWidget.columnCount()):
                            if j == self.tableWidget.columnCount() - 1:
                                insert_sql += str(self.tableWidget.item(i, j).text())
                            else:
                                insert_sql += str(self.tableWidget.item(i, j).text()) + ','
                        
                        insert_sql += ");"
                        c.execute(insert_sql)
                    
                    conn.commit()
                    conn.close()
                    self.reflash_action()
                    self.save_temp = False
            if name == 'replace':
                path = self.label_action_number.text()
                if os.path.isfile(path):
                    os.system('sudo rm ' + path)
                if path != '':                    
                    if path[-4:] == '.d6a':
                        conn = sqlite3.connect(path)
                    else:
                        conn = sqlite3.connect(path + '.d6a')
                    
                    c = conn.cursor()                    
                    c.execute('''CREATE TABLE ActionGroup([Index] INTEGER PRIMARY KEY AUTOINCREMENT
                    NOT NULL ON CONFLICT FAIL
                    UNIQUE ON CONFLICT ABORT,
                    Time INT,
                    Servo1 INT,
                    Servo2 INT,
                    Servo3 INT,
                    Servo4 INT,
                    Servo5 INT,
                    Servo6 INT,
                    Servo7 INT,
                    Servo8 INT,
                    Servo9 INT,
                    Servo10 INT,
                    Servo11 INT,
                    Servo12 INT,
                    Servo13 INT,
                    Servo14 INT,
                    Servo15 INT,                    
                    Servo16 INT,
                    Servo17 INT,
                    Servo18 INT,
                    Servo19 INT,
                    Servo20 INT,
                    Servo21 INT,
                    Servo22 INT);''')                      
                    for i in range(self.tableWidget.rowCount()):
                        insert_sql = "INSERT INTO ActionGroup(Time,Servo1,Servo2,Servo3,Servo4,Servo5,Servo6,Servo7,Servo8,Servo9,Servo10,Servo11,Servo12,Servo13,Servo14,Servo15,Servo16,Servo17,Servo18,Servo19,Servo20,Servo21,Servo22) VALUES("
                        for j in range(2, self.tableWidget.columnCount()):
                            if j == self.tableWidget.columnCount() - 1:
                                insert_sql += str(self.tableWidget.item(i, j).text())
                            else:
                                insert_sql += str(self.tableWidget.item(i, j).text()) + ','
                        
                        insert_sql += ");"
                        c.execute(insert_sql)
                    
                    conn.commit()
                    conn.close()
                    self.reflash_action()
                    if self.chinese:
                        result = self.message_delect('保存成功!')
                    else:
                        result = self.message_delect('success')  
            if name == 'readDeviation':
                try:
                    self.horizontalSlider_dev_manual = False
                    ID = ''
                    self.readDevOk = True
                    dev_data = []
                    for i in self.joints:
                        if self.hand_form:
                            dev = get_servo_deviation(i)
                        else:
                            if i in self.servo_special:
                                dev = 0
                            else:
                                dev = get_servo_deviation(i)
                        if dev == 999 or dev is None:
                            dev_data.append(0)
                            ID += (' id' + str(i))
                        elif dev > 125:  # 负数
                            dev_data.append(-(0xff - (dev - 1)))                        
                        else:
                            dev_data.append(dev)
                    for i in self.joints:
                        self.horizontalSlider_dev_object[i].setValue(dev_data[i - 1])
                        self.label_object[i].setText(str(dev_data[i - 1]))
                    if ID == '':
                        if self.chinese:
                            self.message_from('读取偏差成功!')
                        else:
                            self.message_from('success!')
                    else:
                        if self.chinese:
                            self.message_from(ID + '号舵机偏差读取失败!')
                        else:
                            self.message_from('Failed to read the deviation of' + ID)
                except BaseException as e:
                    print(e)
                self.horizontalSlider_dev_manual = True
            if name == 'downloadDeviation':
                if self.readDevOk:
                    for ID in self.joints:
                        if self.hand_form:
                            save_servo_deviation(ID)
                        else:
                            if ID in self.servo_special:
                                pass
                            else:
                                save_servo_deviation(ID)
                                
                    if self.chinese:
                        self.message_from('下载偏差成功!')
                    else:
                        self.message_from('success!')
                else:
                    if self.chinese:
                        self.message_from('请先读取偏差！')
                    else:
                        self.message_from('Please read the deviation first！')
            if name == 'tandemActionGroup':
                dig_t = QFileDialog()
                dig_t.setFileMode(QFileDialog.ExistingFile)
                dig_t.setNameFilter('d6a Flies(*.d6a)')
                openfile = dig_t.getOpenFileName(self, 'OpenFile', '', 'd6a Flies(*.d6a)')
                # 打开单个文件
                # 参数一：设置父组件；参数二：QFileDialog的标题
                # 参数三：默认打开的目录，“.”点表示程序运行目录，/表示当前盘符根目录
                # 参数四：对话框的文件扩展名过滤器Filter，比如使用 Image files(*.jpg *.gif) 表示只能显示扩展名为.jpg或者.gif文件
                # 设置多个文件扩展名过滤，使用双引号隔开；“All Files(*);;PDF Files(*.pdf);;Text Files(*.txt)”
                path = openfile[0]
                try:
                    if path != '':
                        tbt = QSqlDatabase.addDatabase("QSQLITE")
                        tbt.setDatabaseName(path)
                        if tbt.open():
                            actgrp = QSqlQuery()
                            if (actgrp.exec("select * from ActionGroup ")):
                                while (actgrp.next()):
                                    count = self.tableWidget.rowCount()
                                    self.tableWidget.setRowCount(count + 1)
                                    for i in range(len(self.joints) + 2):
                                        if i == 0:
                                            self.tableWidget.setItem(count, i + 1, QtWidgets.QTableWidgetItem(str(count + 1)))
                                        else:                      
                                            self.tableWidget.setItem(count, i + 1, QtWidgets.QTableWidgetItem(str(actgrp.value(i))))
                                        if i == 1:
                                            self.totalTime += actgrp.value(i)
                                        self.tableWidget.update()
                                        self.tableWidget.selectRow(count)
                                    self.tableWidget.item(count , 2).setFlags(self.tableWidget.item(count , 2).flags() & ~Qt.ItemIsEditable)
                        self.icon_position()
                        tbt.close()
                        self.tableWidget.selectRow(0)
                        self.label_TotalTime.setText(str(self.totalTime/1000.0))
                except BaseException as e:
                    print(e)
                    if self.chinese:
                        self.message_from('错误的动作组格式')
                    else:
                        self.message_from('Wrong action group format')
        except BaseException as e:
            print(e)

    def listActions(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        pathlist = os.listdir(path)
        actList = []
        
        for f in pathlist:
            if f[0] == '.':
                pass
            else:
                if f[-4:] == '.d6a':
                    f.replace('-', '')
                    if f:
                        actList.append(f[0:-4])
                else:
                    pass
        return actList
    
    def reflash_action(self):
        actList = self.listActions(self.actdir)
        actList.sort()
        
        if len(actList) != 0:        
            self.comboBox_action.clear()
            for i in range(0, len(actList)):
                self.comboBox_action.addItem(actList[i])
        else:
            self.comboBox_action.clear()
    
    # 控制动作组按钮点击事件
    def button_controlaction_clicked(self, name):
        if name == 'delectSingle':
            if self.chinese:
                result = self.message_delect('此操作会删除当前动作组，是否继续？')
            else:
                result = self.message_delect('This operation will delete current action group. Do you want to continue？')               
            if result == 0:                              
                if str(self.comboBox_action.currentText()) != "":
                    os.remove(self.actdir + str(self.comboBox_action.currentText()) + ".d6a")            
                    self.reflash_action()
        if name == 'allDelect':
            if self.chinese:
                result = self.message_delect('此操作会删除所有动作组，是否继续？')
            else:
                result = self.message_delect('This operation will delete all action groups. Do you want to continue？')               
            if result == 0:                              
                actList = self.listActions(self.actdir)
                for d in actList:
                    os.remove(self.actdir + d + '.d6a')
            self.reflash_action()
        if name == 'runAction':   # 动作组运行
            run_action_group(self.comboBox_action.currentText())
        if name == 'stopAction':   # 停止运行
            stop_action_group()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
