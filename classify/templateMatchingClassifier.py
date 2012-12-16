# coding: utf-8

from classify.base import jointClassifier
import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import match_template

class templateMatchingClassifier(jointClassifier):
    
    def __init__(self,windowSize,verbose):
        
                
        super(templateMatchingClassifier, self).__init__()
        
        self.windowSize=windowSize
        self.verbose=verbose
        
        return
    
    def classify(self,sample,fingerName,jointName):
        max=sample.max()
        sample=sample/max*255
        sample=sample.astype('uint8')
        
        cropSamp=sample[self.windowSize[0]:self.windowSize[1],self.windowSize[2]:self.windowSize[3]]
        
        if self.verbose:
            plt.imshow(cropSamp,cmap=plt.cm.gray)
            plt.show()
        
        scores = []
        for classImage in self.classImages:          
            
            score = match_template(classImage, cropSamp,1)
            scores.append(np.max(score))
        
        classification = self.classLabels[np.argmax(scores)]
        
        return classification