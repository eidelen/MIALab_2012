# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt4.Qt import *
from PyQt4.QtGui import *
from PyQt4.QtCore import pyqtSignature

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
    
    def __init__(self, parent = None):
        """
        Constructor
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
    
    @pyqtSignature("")
    def on_LoadXray_released(self):
        """
        Slot documentation goes here.
        """
        fileName = QFileDialog.getOpenFileName(self, "Open X-ray File", "","DICOM Files (*.dcm)")
        (reader, img) = dicom.open_image(str(fileName))
        self.mOrignialXRayImage = img
        
        self.display_image( self.mOrignialXRayImage )
        
    
    @pyqtSignature("")
    def on_dectectJointsButton_released(self):
        #boneBinaryImage = age_determination.extract_Bones( self.mOrignialXRayImage )
        aClass = AgeDetermination()
        boneBinaryImage = aClass.extract_Bones( self.mOrignialXRayImage )
        self.display_image( boneBinaryImage )
    
    @pyqtSignature("")
    def on_rateJointsButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    
    
    def display_image(self, pilImage ):
        scene = QGraphicsScene()
        qimg = gray2qimage(pilImage, normalize=True) # Convert image to a QImage and normalise it
        scene.addPixmap(QPixmap.fromImage(qimg))
        self.xrayView.setScene(scene);
