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

import numpy as np
import scipy as sp
import matplotlib as mp
import matplotlib.pylab

from Ui_mainwindow import Ui_MainWindow

# Global variables
#img = zeros(5)


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
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
        mp.pyplot.imshow(img)
        scene = QGraphicsScene()
        qimg = gray2qimage(img) # Convert image to a QImage and normalise it
        scene.addPixmap(QPixmap.fromImage(qimg))
        self.xrayView.setScene(scene);
        
    
    @pyqtSignature("")
    def on_dectectJointsButton_released(self):
        """
        Slot documentation goes here.
        """
        
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSignature("")
    def on_rateJointsButton_released(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
