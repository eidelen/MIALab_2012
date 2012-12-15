from classify.base import jointClassifier
import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import match_template

class templateMatchingClassifier(jointClassifier):
    
    def __init__(self):
        super(templateMatchingClassifier, self).__init__()
        return
    
    def classify(self,sample,fingerName,jointName):
        
        scores = []
        for classImage in self.classImages:      
            
            #plt.imshow(sample[10:70,20:60],cmap=plt.cm.gray)
            #plt.show()
            
            #plt.imshow(classImage,cmap=plt.cm.gray)
            #plt.show()
            
            #do template matching
            score = match_template(classImage, sample[10:70,20:60],1)
            scores.append(np.max(score))
        
        classification = self.classLabels[np.argmax(scores)]
        
        return classification