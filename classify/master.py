import glob
import re
import scipy
import os
import numpy as np

class masterClassifier:
    
    def __init__(self,scoreTable):
        
        self.classifiers = [] # stores all individual classifiers    
        self.classImages = {} # stores all (detected) classImages of the database
        self.classLabels = {} # stores the ground truth for the template database
        self.subjects = {} # stores all subject ids of 'classImages'
        self.allScores = {} # stores the computed scores for each joint
        
        # set the indices of each finger for the score table. Our numbering starts at the distal end.
        #self.scoreTableInds = {'littleFinger':[15, 12, 10], 'middleFinger': [14, 11, 9], 'thumb':[13, 8]}
        self.scoreTableInds = {'littleFinger':[15, 12, 10], 'middleFinger': [14, 11, 9]}
        #self.nJointsForFingers = {'littleFinger': 3, 'middleFinger': 3, 'thumb': 2}
        self.nJointsForFingers = {'littleFinger': 3, 'middleFinger': 3}
        self.jointNames = ['distal', 'middle', 'proximal']
    
        #for finger in ['littleFinger', 'middleFinger', 'thumb']:
        for finger in ['littleFinger', 'middleFinger']:
            
            # load all template images and remember subect ids
            fnames = glob.glob('../extractedJoints/*_' + finger + '_*.png')
            
            self.classImages[finger] = {}
            self.classLabels[finger] = {}
            self.subjects[finger] = {}
            
            for fn in fnames:
                
                # parse subject and joint number from filename
                tokens = re.split('_', os.path.basename(fn))
                subjectId = int(tokens[0])
                jointId = int(re.split('\.',tokens[2])[0])
                
                if not self.jointNames[jointId-1] in self.subjects[finger]: # key does not exist
                    self.subjects[finger][self.jointNames[jointId-1]] = [subjectId]
                    self.classImages[finger][self.jointNames[jointId-1]] = [scipy.misc.imread(fn)]
                    self.classLabels[finger][self.jointNames[jointId-1]] = [int(scoreTable[subjectId-1][self.scoreTableInds[finger][jointId-1]])]
                else: # key already set
                    self.subjects[finger][self.jointNames[jointId-1]].append(subjectId)
                    self.classImages[finger][self.jointNames[jointId-1]].append(scipy.misc.imread(fn))
                    self.classLabels[finger][self.jointNames[jointId-1]].append(int(scoreTable[subjectId-1][self.scoreTableInds[finger][jointId-1]]))
   
    def registerClassifier(self,classifier):
        
        classifier.setClassImages(self.classImages)
        classifier.setClassLabels(self.classLabels)
        self.classifiers.append(classifier)
    
    def classifyHand(self,jointImages):
        
        scores = {}
        
        if len(jointImages['littleFinger'])==self.nJointsForFingers['littleFinger']:
            scores['littleFinger'] = self.classifyFinger(jointImages['littleFinger'],'littleFinger')
        if len(jointImages['middleFinger'])==self.nJointsForFingers['middleFinger']:
            scores['middleFinger'] = self.classifyFinger(jointImages['middleFinger'],'middleFinger')
        #if len(jointImages['thumb'])==self.nJointsForFingers['thumb']:
        #scores['thumb'] = self.classifyFinger(jointImages['thumb'],'thumb')
        
        #Append easy to compare string...
        lf1=scores['littleFinger'][0]
        lf2=scores['littleFinger'][1]
        lf3=scores['littleFinger'][2]
        
        mf1=scores['middleFinger'][0]
        mf2=scores['middleFinger'][1]
        mf3=scores['middleFinger'][2]
        
        scores['resultLine']=  "t3" + "-" + str(mf3) + "-" + str(lf3) + "-" + str(mf2) + "-" + str(lf2) + "-" + "t1" + "-" + str(mf1) + "-" + str(lf1)
        
        
        return scores
        
    def classifyFinger(self,jointImages,fingerName): # classifies each image in joints
        
        scores = []
        
        jointNum=0
        for jointImage in jointImages:
            scores.append(self.classifyJoint(jointImage,fingerName,self.jointNames[jointNum]))
            jointNum=jointNum+1
        
        return scores
                    
    def classifyJoint(self,jointImage,fingerName,jointName):
        
        scores = []
        
        for classifier in self.classifiers:
            #classifier.setClassImages(self.classImages[fingerName][jointName])
            #classifier.setClassLabels(self.classLabels[fingerName][jointName])
            scores.append(classifier.classify(jointImage,fingerName,jointName))
        
        # majority win for the class
        d = [0]*max(scores)
        for s in scores:
            d[s-1] += 1
        classification = np.argmax(d)+1
        
        # get the class label for the winner
        #predictedClass = self.classLabels[fingerName][jointName][score-1]
        
        return classification
    
    def getClassImages(self):
        
        return self.classImages
    
    def getClassLabels(self):
        
        return self.classLabels
    
