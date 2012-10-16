# MIA Lab - F.Preiswerk, J.Walti, A.Schneider

# TODO: Check what packets are realy used!

import Image 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches



import numpy as np
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import *
import dicom

from skimage import measure
from skimage import filter
from skimage import data
from skimage.filter import *
#from skimage.segmentation import find
from skimage.morphology import label, closing, square, skeletonize, medial_axis
from skimage.measure import regionprops
from cProfile import label
from scipy.ndimage.measurements import label

import cv2
import peakdet
from regiongrowing import *




class AgeDetermination:
    
    def detect_joints_of_interest(self, numpyImage ):
    
        success, handmask = self.get_hand_mask(numpyImage)
        if not success :
            print "Background Segmentation failed"
            return handmask
        
        xRay_without_background = self.remove_background(numpyImage, handmask)

        #skinMask = self.remove_skin(xRay_without_background, handmask)
        #plt.imshow(skinMask, cmap=cm.Greys_r)
        #plt.show()     
        
    
        self.get_fingers_of_interest(handmask)
        
        return handmask
         
        
    def get_hand_mask(self, numpyImage):
        # Adi
        success = False
        
        thresh = self.get_XRay_BG_Threshold( numpyImage )
        treshMask = numpyImage > thresh
        
        labeled, nr_objects = label( treshMask )
        print "Number of objects found is %d " % nr_objects
        
        label_sizes = sum( treshMask, labeled, range(nr_objects + 1) )
        
        idx_of_biggest_label = np.argmax(label_sizes)
        
        print "size of %d " % label_sizes[idx_of_biggest_label]
        
        # our object should fill at least 20 % of the whole image
        h, w = treshMask.shape
        areaRatio = label_sizes[idx_of_biggest_label] / (h*w)
        if areaRatio > 0.2 :
            success = True
        
        print "Obj. size %d " % label_sizes[idx_of_biggest_label]
        print "Obj. size ratio %f " % areaRatio
                
        treshMask = (labeled == idx_of_biggest_label)
        
        return success, treshMask
    
    
    def remove_background(self, xRay, maskedBG ):
        return xRay * maskedBG;
    
    def remove_skin(self, imgNoBG, bgMask ):
        # kind of growing region :) , which is started only
        # at borders from background to object... what is 
        # actually supposed to skin.
        
        # well -> for tables the height (y or m) is the first value
        imgNoBGCpy = np.copy(imgNoBG)
        
        height, widht = imgNoBGCpy.shape
        
        skinMask = np.zeros((height,widht))
        
        counter = 0
        
        for n in range(0, widht-1):
            for m in range( 0, height ):
                maskVal = bgMask[m,n]
                nextMaskVal = bgMask[m,n+1]
                
                if (maskVal < nextMaskVal) and (counter < 5) :
                    # we are on a edge from background to skin
                    if skinMask[m,n+1] == 0 :
                        # edge pixel not part of previous region grow
                        region = regiongrow(imgNoBGCpy, 30, [m, n+1])
                        # update values
                        skinMask = (skinMask > 0) | (region > 0)
                        imgNoBGCpy = imgNoBGCpy * (~skinMask) # remove parts where is skin
                        
                        counter = counter + 1
                        
                    
                    
        return skinMask
                 
        
        
                    
    def get_fingers_of_interest(self, handmaskImage ):
        
        # Wale
        
        handmaskImage = median_filter(handmaskImage, radius=20, mask=None, percent=70)
        
#        plt.imshow(handmaskImage)
        
        # Compute the medial axis (skeleton) and the distance transform
        skel, distance = medial_axis(handmaskImage, return_distance=True)
        
        # Detect only finger - skeleton
        new_distance = distance <  100
        distance = distance * new_distance
        new_distance = distance > 50 
        
        ## Distance to the background for pixels of the skeleton
        new_distance =  skel * new_distance
        
        # Scans finger I
        
        startX = 170
        startY = 350
        endX = 330
        endY = 800
        
                
        #select the indices for the line on the finger-skeleton (value >0)   
        #  for x in range(startX,endX):
        #    for y in range(startY,endY):

        dist_on_skel = new_distance * distance
        
        plt.figure(figsize=(8, 4))
        plt.subplot(121) 
        plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
        plt.contour(handmaskImage, [0.5], colors='w')
        
        plt.show()
        
        return 0
    
      
    def get_XRay_BG_Threshold(self, pilImage ):
    
        grayhist, bins = np.histogram(pilImage.flatten(),  200 )
   
        avgHistCounts = np.mean(grayhist)
        minimumPeakDiff = avgHistCounts * 0.20
        peaks, valeys = peakdet.peakdet(grayhist, minimumPeakDiff)
    
        # max peak is background
        maxPeakIdx= np.argmax(peaks[:,1])
    
        idx0 = peaks[maxPeakIdx,0]
        idx1 = peaks[maxPeakIdx+1,0]
        val1 = bins[idx0]
        val2 = bins[idx1]

        # set threshold between bg peak and next one
        threshVal = (val1 + val2) / 2.0
    
        #plt.plot(grayhist)
        #plt.show()
    
        return threshVal

    #def __init__(self):
     
    # Frank specific code :)    
    #	(reader, img) = dicom.open_image('../data/Case1.dcm')	
	# self.extract_Bones(img)

	
 

