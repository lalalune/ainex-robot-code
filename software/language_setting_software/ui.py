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
        Form.resize(400, 150)
        self.pushButton_english = QtWidgets.QPushButton(Form)
        self.pushButton_english.setGeometry(QtCore.QRect(240, 60, 99, 30))
        self.pushButton_english.setObjectName("pushButton_english")
        self.pushButton_chinese = QtWidgets.QPushButton(Form)
        self.pushButton_chinese.setGeometry(QtCore.QRect(70, 60, 99, 30))
        self.pushButton_chinese.setObjectName("pushButton_chinese")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "language_setting1.0"))
        self.pushButton_english.setText(_translate("Form", "English"))
        self.pushButton_chinese.setText(_translate("Form", "Chinese"))
