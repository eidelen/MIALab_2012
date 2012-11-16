from classify.base import jointClassifier

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import match_template

class templateMatchingClassifier(jointClassifier):
    
    def __init__(self):
        super(templateMatchingClassifier, self).__init__()
        return
    
    def classify(self,sample):
        
        scores = []
        for classImage in self.classImages:
            score = match_template(classImage, sample,1)
            scores.append(np.max(score))
        
        classification = self.classLabels[np.argmax(scores)]
        
        return classification