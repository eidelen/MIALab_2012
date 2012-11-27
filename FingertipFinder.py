'''
Created on Nov 20, 2012

@author: frank
'''

import matplotlib.pyplot as plt
import dicom
import numpy as np
from skimage.feature import match_template
import cv2
from logilab.constraint import *

class FingertipFinder:
    
    def __init__(self):
        
        self.fileNameRef = '../training/Case1.dcm'
        (reader, self.imgSample) = dicom.open_image(str(self.fileNameRef))
        self.imgSample = self.resize_image(self.imgSample)
        self.patch = self.imgSample[50:100][:,185:270] # only tip
        #patch = imgSample[150:190][:,185:275] # joint
        #patch = imgSample[450:520][:,195:295] # finger start
        #patch = imgSample[80:450][:,310:450] # more of the finger  
        #thumb = imgSample[300:600][:,400:600]      
        #littleFingerPatch = imgSample[180:220][:,80:155] # little finger
      
    def resize_image(self,image):
        
        minwidth = 692; # the minimum width of all images in the training set, determined using normalizeHandplateImages.py
        
        # size is hardcoded, ugly
        newHeight = int(float(minwidth)/(float(image.shape[1]))*image.shape[0])
        img = np.asarray(image,dtype=np.float32)
        image_small = cv2.resize(img, (minwidth, newHeight))
       
        return image_small
      
    def findFingertips(self,image):
                   
        image = image[0:600][:,:]
         
        result = match_template(image, self.patch,1)
               
        removeAreaSize=40
        
        # extract best match candidates
        result_tmp=result.copy()
        xcoos = []
        ycoos = []
        for i in range(0,7):
            px=removeAreaSize
            ij = np.unravel_index(np.argmax(result_tmp), result.shape)
            x, y = ij[::-1]
            result_tmp[y-px:y+px][:,x-px:x+px]=0
            xcoos.append(x)
            ycoos.append(y)
            
        #plt.imshow(result_tmp)
        #plt.show()
        
        # fire up the solver to find the best solution
        variables = ('little','ring','middle','pointer')
        domains = {}
        
        allPoints = []
        for i in range(0,len(xcoos)):
            allPoints.append((xcoos[i],ycoos[i]))
        allPoints = tuple(allPoints)
        for v in variables:
            domains[v] = fd.FiniteDomain(allPoints)
        
        constraints = []
        constraints.append(fd.make_expression(('little','ring'),'%s[0]+%i < %s[0]'%('little',15,'ring')))
        constraints.append(fd.make_expression(('little','middle'),'%s[0]+%i < %s[0]'%('little',15,'middle')))                                         
        constraints.append(fd.make_expression(('little','pointer'),'%s[0]+%i < %s[0]'%('little',15,'pointer')))
        constraints.append(fd.make_expression(('ring','middle'),'%s[0]+%i < %s[0]'%('ring',15,'middle')))
        constraints.append(fd.make_expression(('ring','pointer'),'%s[0]+%i < %s[0]'%('ring',15,'pointer')))
        constraints.append(fd.make_expression(('middle','pointer'),'%s[0]+%i < %s[0]'%('middle',15,'pointer')))

        # no two x must be the same
        constraints.append(fd.make_expression(('little','ring'),'%s[0] != %s[0]'%('little','ring')))
        constraints.append(fd.make_expression(('little','middle'),'%s[0] != %s[0]'%('little','middle')))                                         
        constraints.append(fd.make_expression(('little','pointer'),'%s[0] != %s[0]'%('little','pointer')))
        constraints.append(fd.make_expression(('ring','middle'),'%s[0] != %s[0]'%('ring','middle')))
        constraints.append(fd.make_expression(('ring','pointer'),'%s[0] != %s[0]'%('ring','pointer')))
        constraints.append(fd.make_expression(('middle','pointer'),'%s[0] < %s[0]'%('middle','pointer')))
        
        constraints.append(fd.make_expression(('little','ring'),'%s[1] > %s[1]'%('little','ring')))
        constraints.append(fd.make_expression(('little','middle'),'%s[1] > %s[1]'%('little','middle')))                                         
        constraints.append(fd.make_expression(('little','pointer'),'%s[1] > %s[1]'%('little','pointer')))
        constraints.append(fd.make_expression(('ring','middle'),'%s[1] > %s[1]'%('ring','middle')))
        constraints.append(fd.make_expression(('middle','pointer'),'%s[1] < %s[1]'%('middle','pointer')))
        
        # no two y must be the same
        constraints.append(fd.make_expression(('little','ring'),'%s[1] != %s[1]'%('little','ring')))
        constraints.append(fd.make_expression(('little','middle'),'%s[1] != %s[1]'%('little','middle')))                                         
        constraints.append(fd.make_expression(('little','pointer'),'%s[1] != %s[1]'%('little','pointer')))
        constraints.append(fd.make_expression(('ring','middle'),'%s[1] != %s[1]'%('ring','middle')))
        constraints.append(fd.make_expression(('ring','pointer'),'%s[1] != %s[1]'%('ring','pointer')))
        constraints.append(fd.make_expression(('middle','pointer'),'%s[1] < %s[1]'%('middle','pointer')))

        # distance between ring y and pointer y is smaller than between left y and ring y and between left y and pointer y
        constraints.append(fd.make_expression(('little','ring','pointer'),'%s[1] - %s[1] < %s[1] - %s[1] '%('ring','pointer','little','ring')))
        constraints.append(fd.make_expression(('little','ring','pointer'),'%s[1] - %s[1] < %s[1] - %s[1] '%('ring','pointer','little','pointer')))
        
        # distance between ring y and pointer y is max 100px
        constraints.append(fd.make_expression(('ring','pointer'),'abs(%s[1] - %s[1]) <= 100 '%('ring','pointer')))
        
        r = Repository(variables,domains,constraints)
        solutions = Solver().solve(r)
        
#        for solution in solutions:
#            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))
#            ax1.imshow(self.patch)
#            ax1.set_axis_off()
#            ax1.set_title('template')
#            ax2.imshow(result)
#            ax2.set_axis_off()
#            ax2.set_title('image')
#            ax3.imshow(image)
#            ax3.set_axis_off()
#            ax3.set_title('`match_template`\nresult')
#            # highlight matched region
#            ax3.autoscale(False)
#        
#            ax3.plot(solution['little'][0], solution['little'][1], 'o', markeredgecolor='r', markerfacecolor='y', markersize=10)
#            ax3.plot(solution['ring'][0], solution['ring'][1], 'o', markeredgecolor='r', markerfacecolor='y', markersize=10)
#            ax3.plot(solution['middle'][0], solution['middle'][1], 'o', markeredgecolor='r', markerfacecolor='y', markersize=10)
#            ax3.plot(solution['pointer'][0], solution['pointer'][1], 'o', markeredgecolor='r', markerfacecolor='y', markersize=10)
#        
#            plt.show()
            
        if len(solutions) > 0 :
            return True, solutions[0]
        else :
           return False, []
        

