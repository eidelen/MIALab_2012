# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt4.Qt import *
from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSignature
from PyQt4.QtGui import QApplication, QCursor

from qimage2ndarray import *

import gdcm
import dicom
#import age_determination 
from AgeDetermination import AgeDetermination

import numpy as np
import scipy as sp
import Image

from Ui_mainwindow import Ui_MainWindow

# Global variables
#img = zeros(5)


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    mOrignialXRayImage = Image.new("L", (50, 30)) # dummy image
    mDetectedJoints=None
    
    def __init__(self, parent = None):
        """
        Constructor
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.dectectJointsButton.enabledChange(False)
        self.rateJointsButton.enabledChange(False)
    
    @pyqtSignature("")
    def on_LoadXray_released(self):
        """
        Slot documentation goes here.
        """
        fileName = QFileDialog.getOpenFileName(self, "Open X-ray File", "","DICOM Files (*.dcm)")
        (reader, img) = dicom.open_image(str(fileName))
        self.mOrignialXRayImage = img
        
        self.display_image( self.mOrignialXRayImage )
        self.dectectJointsButton.enabledChange(True)
        print str(fileName)
    
    @pyqtSignature("")
    def on_dectectJointsButton_released(self):
        
        #boneBinaryImage = age_determination.extract_Bones( self.mOrignialXRayImage )
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        aClass = AgeDetermination()
        aClass.setVerbosity( True ) 
        self.mDetectedJoints=aClass.detect_joints_of_interest( self.mOrignialXRayImage )
        #self.display_image( joint_marked_image )
        QApplication.restoreOverrideCursor()
        self.rateJointsButton.enabledChange(True)
        
    @pyqtSignature("")
    def on_rateJointsButton_released(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        if (self.mDetectedJoints!=None):
            scoreTable = np.loadtxt('scores/scores.txt')
            aClass = AgeDetermination()
            aClass.setVerbosity( True ) 
            #[rateSum, okRatings]=aClass.rate_joints(self.mDetectedJoints,scoreTable)
            #print "----- FINAL Score is " + str(rateSum) +" with " + str(okRatings)+" found ratings! ------"
            print "Final prediction: " + str(aClass.rate_joints(self.mDetectedJoints,scoreTable))
        else:
            print "Detect joints first!"
        QApplication.restoreOverrideCursor()
    
    def display_image(self, pilImage ):
        scene = QGraphicsScene()
        qimg = gray2qimage(pilImage, normalize=True) # Convert image to a QImage and normalise it
        scene.addPixmap(QPixmap.fromImage(qimg))
        self.xrayView.setScene(scene);
