# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/cattin/Lectures/2012HS-MIA_Lab/workspace/ui/mainwindow.ui'
#
# Created: Wed Sep 19 09:45:06 2012
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(489, 600)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralWidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.rateJointsButton = QtGui.QPushButton(self.centralWidget)
        self.rateJointsButton.setObjectName(_fromUtf8("rateJointsButton"))
        self.gridLayout.addWidget(self.rateJointsButton, 1, 2, 1, 1)
        self.LoadXray = QtGui.QPushButton(self.centralWidget)
        self.LoadXray.setObjectName(_fromUtf8("LoadXray"))
        self.gridLayout.addWidget(self.LoadXray, 1, 0, 1, 1)
        self.dectectJointsButton = QtGui.QPushButton(self.centralWidget)
        self.dectectJointsButton.setObjectName(_fromUtf8("dectectJointsButton"))
        self.gridLayout.addWidget(self.dectectJointsButton, 1, 1, 1, 1)
        self.xrayView = QtGui.QGraphicsView(self.centralWidget)
        self.xrayView.setObjectName(_fromUtf8("xrayView"))
        self.gridLayout.addWidget(self.xrayView, 0, 0, 1, 3)
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.rateJointsButton.setText(QtGui.QApplication.translate("MainWindow", "Rate Joints", None, QtGui.QApplication.UnicodeUTF8))
        self.LoadXray.setText(QtGui.QApplication.translate("MainWindow", "Load Xray", None, QtGui.QApplication.UnicodeUTF8))
        self.dectectJointsButton.setText(QtGui.QApplication.translate("MainWindow", "Detect Joints", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

