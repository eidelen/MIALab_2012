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
from skimage.morphology import label, closing, square, skeletonize, medial_axis
from skimage.measure import regionprops
from cProfile import label
from scipy.ndimage.measurements import label

import cv2
import peakdet
from regiongrowing import *




class AgeDetermination:
    
    def detect_joints_of_interest(self, numpyImage ):
        
        # cut lower part of the image 
        imgH, imgW = numpyImage.shape
        numpyImage = numpyImage[0:(imgH-100),:]
        imgH, imgW = numpyImage.shape
        
        #numpyImage = median_filter(numpyImage, radius=2, mask=None, percent=50)
        
        
        success, handmask = self.get_hand_mask(numpyImage)
        if not success :
            print "Background Segmentation failed"
            return handmask
        
        xRay_without_background = self.remove_background(numpyImage, handmask)

        #skinMask = self.remove_skin(xRay_without_background, handmask)
        #plt.imshow(skinMask, cmap=cm.Greys_r)
        #plt.show()     
        
    
        success, littleFingerLine, ringFingerLine, middleFingerLine, pointingFingerLine = self.get_fingers_of_interest( handmask )
        if not success :
            print "Finger Detection Failed"
            return handmask
        
        
        plt.imshow(xRay_without_background, cmap=cm.Greys_r)
        for i in littleFingerLine:
            plt.plot(i[1], i[0], ".r")
        for i in ringFingerLine:
            plt.plot(i[1], i[0], ".r")
        for i in middleFingerLine:
            plt.plot(i[1], i[0], ".r")
        for i in pointingFingerLine:
            plt.plot(i[1], i[0], ".r")
                        
        #plt.ylim([0,imgH])
        #plt.xlim([0,imgW])
        plt.show() 
   
        
        
        
        
        
        return xRay_without_background
         
        
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
        
    def get_fingers_of_interest(self, handmaskImage ):
        
        # identify fingers - adi
        handmaskImage = median_filter(handmaskImage, radius=20, mask=None, percent=70)
        
        maskH, maskW = handmaskImage.shape
         
        # identifying the fingers by diff image of the handmask.
        # each finger has two peaks. that means there are 8 peaks for the 
        # higher 4 fingers. The 'Daumen' is detected downwards, right 
        # of the pointing finger.
        dRowMask = np.diff(handmaskImage,n=1,axis=1)
        dRowMaskIsPeak = (dRowMask > 0)
        dRowSum = np.sum(dRowMask,axis=1)
        interestingRows = (dRowSum == 8)
        
        # the "rowSum == 8" should appear one after another. lets count
        maxLittleFingerCenters = []
        maxRingFingerCenters = []
        maxMiddleFingerCenters = []
        maxPointFingerCenters = []
        curLitleFingerC = []
        curRingFingerC = []
        curMiddleFingerC = []
        curPointFingerC = []
        currentMax8OrderCount = 0
        current8OrderCount = 0
        itRowSize = interestingRows.shape
        for rowIter in range(0, itRowSize[0]):
            if interestingRows[rowIter] :
                current8OrderCount = current8OrderCount + 1
                
                                
                # extract center of litle and middle finger
                currentRowPeaks = dRowMaskIsPeak[rowIter,:]
                currentRowPeaksSize = currentRowPeaks.shape
                cPeakIndex = 0
                startLitleFinger = 0 
                endLitleFinger = 0
                startRingFinger = 0 
                endRingFinger = 0
                startMiddleFinger = 0
                endMiddleFinger = 0
                startPointFinger = 0
                endPointFinger = 0
                for cRowPeakIter in range(0, currentRowPeaksSize[0]) :
                    if currentRowPeaks[cRowPeakIter] :
                        
                        if cPeakIndex == 0 :
                            startLitleFinger = cRowPeakIter
                        
                        if cPeakIndex == 1 :
                            endLitleFinger = cRowPeakIter
                            
                        if cPeakIndex == 2 :
                            startRingFinger = cRowPeakIter
                        
                        if cPeakIndex == 3 :
                            endRingFinger = cRowPeakIter
                            
                        if cPeakIndex == 4 :
                            startMiddleFinger = cRowPeakIter
                        
                        if cPeakIndex == 5 :
                            endMiddleFinger = cRowPeakIter
                            
                        if cPeakIndex == 6 :
                            startPointFinger = cRowPeakIter
                        
                        if cPeakIndex == 7 :
                            endPointFinger = cRowPeakIter
                        
                        cPeakIndex = cPeakIndex + 1
                        
                
                curLitleFingerC.append( [ rowIter ,  (endLitleFinger  + startLitleFinger )/2 ] )
                curRingFingerC.append( [ rowIter ,  (endRingFinger  + startRingFinger )/2 ] )  
                curMiddleFingerC.append( [ rowIter , (endMiddleFinger + startMiddleFinger)/2 ] ) 
                curPointFingerC.append( [ rowIter , (endPointFinger + startPointFinger)/2 ] )
                
            else:
                current8OrderCount = 0
                curLitleFingerC = []
                curRingFingerC= []
                curMiddleFingerC = []
                curPointFingerC = []
                
            if current8OrderCount > currentMax8OrderCount :
                currentMax8OrderCount = current8OrderCount
                maxLittleFingerCenters = curLitleFingerC
                maxRingFingerCenters = curRingFingerC
                maxMiddleFingerCenters = curMiddleFingerC
                maxPointFingerCenters = curPointFingerC
                
            
                
        print "Detected 4 fingers over a distance of %d " % currentMax8OrderCount
        
        detectedFingerRatio = float(currentMax8OrderCount) / float(maskH)
        
        if  detectedFingerRatio < 0.05 :# continous distance needs to be at least 10% of image height
            return False, [], [], [], []
        
        # continous growing
        maxLittleFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxLittleFingerCenters )
        maxRingFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxRingFingerCenters )
        maxMiddleFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxMiddleFingerCenters )
        maxPointFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxPointFingerCenters )
        
        return True, maxLittleFingerCenters, maxRingFingerCenters, maxMiddleFingerCenters, maxPointFingerCenters
        
        
        for i in maxLittleFingerCenters:
            handmaskImage[i[0], i[1]] = 0
            
        for i in maxMiddleFingerCenters:
            handmaskImage[i[0], i[1]] = 0
            
         
        
        plt.imshow(handmaskImage)
        plt.show()
        
        
        #for rowIdx in range(0, h):
        #    cRow = handmaskImage[rowIdx,:]
        #    diff(cRow,n=1,axis=1)
        
        
        
        
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
        
        return True
    
    def continue_central_line(self, peaks, currentCenters ):
        thicknessDiffThreshold = 10
        
        h, w = peaks.shape
        
        # growing downwards
        lastPoint = currentCenters[-1]
        lastCenter = lastPoint[1]     
        for i in range( lastPoint[0] , h) :
            lowIdx, upIdx = self.get_closest_lower_and_upper_true_idx(peaks[i,:], lastCenter )
            
            center = (upIdx + lowIdx)/2   
            diffCenter = center - lastCenter
            
            if np.sqrt(diffCenter*diffCenter) > 5 : # stop the process when thickness difference becomes to big
                break
            
            lastCenter = center
            currentCenters.append( [i, center] )
                   
        # growing upwards
        firstPoint = currentCenters[0]
        firstCenter = firstPoint[1]
        
        for i in range( firstPoint[0] , 0, -1) :
            lowIdx, upIdx = self.get_closest_lower_and_upper_true_idx(peaks[i,:], firstCenter )
            
            center = (upIdx + lowIdx)/2   
            diffCenter = center - firstCenter
            
            if np.sqrt(diffCenter*diffCenter) > 5 : # stop the process when thickness difference becomes to big
                break
            
            firstCenter = center
            
            currentCenters.insert(0, [i,center])
                     
        return currentCenters     
    
    def get_closest_lower_and_upper_true_idx( self, array1D, targetIdx ):
        arrSh = array1D.shape
        lowIdx = 0
        upIdx = 0
        minLowDist = 9999999999
        minUpDist = 9999999999
        for i in range(0, arrSh[0]) :
            if array1D[i] :
                dist = i - targetIdx
                if dist < 0 : # lower idx
                    if (dist*dist) < minLowDist :
                        minLowDist = (dist*dist)
                        lowIdx = i
                if dist > 0 : # uper idx
                    if (dist*dist) < minUpDist :
                        minUpDist = (dist*dist)
                        upIdx = i
        
        return lowIdx, upIdx
                    
      
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
        threshVal = (val2 - val1) * 0.5  + val1
    
        #plt.plot(grayhist)
        #plt.show()
    
        return threshVal
    
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

    #def __init__(self):
     
    # Frank specific code :)    
    #	(reader, img) = dicom.open_image('../data/Case1.dcm')	
	# self.extract_Bones(img)

	
 

