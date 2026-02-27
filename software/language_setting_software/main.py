#!/usr/bin/env python3
# encoding:utf-8
import sys, re
from ui import Ui_Form
from PyQt5.QtWidgets import *

class MainWindow(QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        
        self.config_path = '/home/ubuntu/ros_ws/.robotrc'
        self.pushButton_chinese.clicked.connect(lambda: self.button_clicked('chinese'))
        self.pushButton_english.clicked.connect(lambda: self.button_clicked('english'))
        
    def message_confirm(self, string):
        messageBox = QMessageBox()
        messageBox.setWindowTitle(' ')
        messageBox.setText(string)
        messageBox.addButton(QPushButton('OK'), QMessageBox.YesRole)
        messageBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)

        return messageBox.exec_() 

    def button_clicked(self, name):
        result = 1
        if name == 'chinese':
            self.current_language = 'zh'
            result = self.message_confirm('确定?')
        else:
            self.current_language = 'en'
            result = self.message_confirm('sure?')
        
        if result == 0:
            with open(self.config_path, "r") as f:
                data = f.read()
                language = re.findall(r'export speaker_language.*?\n', data)[0].split('=')[1].replace('\n', '')
                if self.current_language == 'zh':
                    data = data.replace("=" + language, "=" + 'Chinese')
                    self.message_confirm('修改成功!')
                else:
                    data = data.replace("=" + language, "=" + 'English')
                    self.message_confirm('Modified successfully!')
                f.close()
            with open(self.config_path, "w") as f:
                f.write(data)
                f.close()
            
if __name__ == "__main__":  
    app = QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
