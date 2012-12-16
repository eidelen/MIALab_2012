# coding: utf-8

from classify.base import jointClassifier
import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import filter
from skimage.filter import tv_denoise
from skimage.feature import match_template

class templateMatchingEdgeClassifier(jointClassifier):
    
    def __init__(self,windowSize,verbose):
    
        super(templateMatchingEdgeClassifier, self).__init__()
        
        self.windowSize=windowSize
        self.verbose=verbose
        
        return
    
    def classify(self,sample,fingerName,jointName):
        max=sample.max()
        sample=sample/max*255
        sample=sample.astype('uint8')
                 
        cropSamp=sample[self.windowSize[0]:self.windowSize[1],self.windowSize[2]:self.windowSize[3]]
        cropSamp=tv_denoise(cropSamp,weight=0.2)
        
        if self.verbose:
            plt.imshow(cropSamp,cmap=plt.cm.gray)
            plt.show()
        
        cropSamp=filter.hprewitt(cropSamp)

        if self.verbose:
            plt.imshow(cropSamp,cmap=plt.cm.gray)
            plt.show()
        
        #idx=cropSamp<0.08
        #cropSamp[idx]=0    
        
        if self.verbose:
            plt.imshow(cropSamp,cmap=plt.cm.gray)
            plt.show()
        
        scores = []
        for classImage in self.classImages:      
            classImage=tv_denoise(classImage,weight=0.2)
            classImage=filter.hprewitt(classImage)
            
            #idx=classImage<0.08
            #classImage[idx]=0 
            
            if False and self.verbose:
                plt.imshow(classImage,cmap=plt.cm.gray)
                plt.show()
            #do template matching
            score = match_template(classImage, cropSamp,1)
            scores.append(np.max(score))
        
        classification = self.classLabels[np.argmax(scores)]
        
        return classification