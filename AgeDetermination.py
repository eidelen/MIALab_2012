# MIA Lab - F.Preiswerk, J.Walti, A.Schneider

# TODO: Check what packets are realy used!

import math
import Image 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches



import numpy as np
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import *
import dicom

from FingertipFinder import FingertipFinder

from skimage import measure
from skimage import filter
from skimage import data
from skimage import feature
from skimage.filter import *
from skimage.morphology import label, closing, square, skeletonize, medial_axis
from skimage.measure import regionprops
from cProfile import label
from scipy.ndimage.measurements import label

import peakdet
from regiongrowing import *
from numpy.linalg.linalg import norm
from mpl_toolkits.axisartist.clip_path import atan2
import scipy
from numpy.lib.scimath import sqrt
from IPython.core.display import Math

import glob
import re
import os

import cv2

from classify import masterClassifier
from classify.templateMatchingClassifier import *

# Todo: check img 14, 4

class AgeDetermination:

    verbosity = 0
    
    def detect_joints_of_interest(self, numpyImage ):
        
        # settings
        joint_windows_size = 40
        joint_rect_size = 100

        
        # cut lower part of the image by 100
        imgH, imgW = numpyImage.shape
        numpyImage = numpyImage[0:(imgH-100),:]
        imgH, imgW = numpyImage.shape

        # first resize the image to the same size
        numpyImage = self.resize_image(numpyImage)
                
        #numpyImage = median_filter(numpyImage, radius=2, mask=None, percent=50)

        #success, handmask = self.get_hand_mask(numpyImage)
        #if not success :
        #    print "Background Segmentation failed"
        #    plt.imshow(handmask)
        #    plt.show()
        #    return 
        
        #xRay_without_background = self.remove_background(numpyImage, handmask)

        #skinMask = self.remove_skin(xRay_without_background, handmask)
        #plt.imshow(skinMask, cmap=cm.Greys_r)
        #plt.show()     
        
    
        #success, littleFingerLine, ringFingerLine, middleFingerLine, pointingFingerLine, daumenFingerLine = self.get_fingers_of_interest( handmask )
        success, littleFingerLine, ringFingerLine, middleFingerLine, pointingFingerLine = self.get_fingers_of_interest_franky_approach( numpyImage )
        
        if not success :
            print "Finger Detection Failed"
            return { "littleFinger": [], "middleFinger": [] }
        
        
        ltFingerJointsIdx = self.find_joints_from_intensities( self.read_intensities_of_point_set(littleFingerLine, numpyImage), joint_windows_size, 3 )
        #ringFingerJointsIdx = self.find_joints_from_intensities( self.read_intensities_of_point_set(ringFingerLine, numpyImage), joint_windows_size )
        middleFingerJointsIdx = self.find_joints_from_intensities( self.read_intensities_of_point_set(middleFingerLine, numpyImage) , joint_windows_size, 3)
        #pointingFingerJointsIdx = self.find_joints_from_intensities( self.read_intensities_of_point_set(pointingFingerLine, numpyImage) , joint_windows_size)
        #daumenJointsIdx = self.find_joints_from_intensities( self.read_intensities_of_point_set(daumenFingerLine, numpyImage), joint_windows_size, 2 )
        
        
        
        fingerLineArrays = []
        fingerLineArrays.append(littleFingerLine)
        #fingerLineArrays.append(ringFingerLine)
        fingerLineArrays.append(middleFingerLine)
        #fingerLineArrays.append(pointingFingerLine)
        #fingerLineArrays.append(daumenFingerLine)
        
        jointsArrays = []
        jointsArrays.append(ltFingerJointsIdx)
        #jointsArrays.append(ringFingerJointsIdx)
        jointsArrays.append(middleFingerJointsIdx)
        #jointsArrays.append(pointingFingerJointsIdx)
        #jointsArrays.append(daumenJointsIdx)
        
        croppedJointsLittleFinger = self.crop_joint( numpyImage, littleFingerLine, ltFingerJointsIdx, joint_rect_size)
        croppedJointsMiddleFinger = self.crop_joint( numpyImage, middleFingerLine, middleFingerJointsIdx, joint_rect_size)
        #croppedJointsDaumen = self.crop_joint( xRay_without_background, daumenFingerLine, daumenJointsIdx, joint_rect_size)

        if(self.verbosity > 0):
            self.draw_joints_to_img(numpyImage, fingerLineArrays, jointsArrays, joint_rect_size)
            plotCnt = 1
            for i in range(0,len(croppedJointsLittleFinger)):
                plt.subplot(2,3,plotCnt)
                plt.imshow(croppedJointsLittleFinger[i], cmap=cm.Greys_r)
                plotCnt = plotCnt + 1
            for i in range(0,len(croppedJointsMiddleFinger)):
                plt.subplot(2,3,plotCnt)
                plt.imshow(croppedJointsMiddleFinger[i], cmap=cm.Greys_r)
                plotCnt = plotCnt + 1
            #for i in range(0,len(croppedJointsDaumen)):
            #    plt.subplot(3,3,plotCnt)
            #    plt.imshow(croppedJointsDaumen[i], cmap=cm.Greys_r)
            #    plotCnt = plotCnt + 1
            
            plt.show()
            
        #return xRay_without_background
    #return croppedJointsLittleFinger, croppedJointsMiddleFinger, croppedJointsDaumen
        return { "littleFinger": croppedJointsLittleFinger, "middleFinger": croppedJointsMiddleFinger }
    
    def resize_image(self, image):
        
        minwidth = 692; # the minimum width of all images in the training set, determined using normalizeHandplateImages.py
        
        # size is hardcoded, ugly
        newHeight = int(float(minwidth)/(float(image.shape[1]))*image.shape[0])
        #resized = imresize(img,(newHeight, minwidth))
        #resized = np.asarray(resized)
        img = np.asarray(image,dtype=np.float32)
    
        #img_small = cv2.resize(img, (newHeight, minwidth))
        image_small = cv2.resize(img, (minwidth, newHeight))
        
        return image_small
    
        
    def rate_joints(self, fingers, scoreTable):
        
        # instanciate the master classifier
        classifier = masterClassifier(scoreTable)
        
        # add a template matching classifier
        tmClassifier = templateMatchingClassifier()
        
        classifier.registerClassifier(tmClassifier)
        
        return classifier.classifyHand(fingers)
        
    def rate_joints_wale(self, fingers, scoreTable):
        
        #final score to sum up
        score=0
        
        #count the number of fingers, where we found a rating
        okRatings = 0  
        
        #Name the fingers, which should be treated here ['littleFinger', 'middelFinger','thumb']
        evaluatedFingers = ['littleFinger','middleFinger','thumb']
        
        for fingerName in evaluatedFingers:

            #thumb has only 2 fingers
            if (fingerName=='thumb'):
                totalJoints=2
            else:
                totalJoints=3
            
            if(len(fingers[fingerName])==totalJoints):         #correct num of fingers detected?
                jointNum=1
                for joint in fingers[fingerName]:    #loop through all fingers 
                    template=Image.fromarray((255.0/joint.max()*(joint-joint.min())).astype(np.uint8))
                    newScore=self.get_score_per_template(fingerName,jointNum,template,scoreTable)
                    
                    if (newScore!=False):
                        score+=newScore
                        okRatings+=1
                    
                    jointNum+=1
                        
        return score, okRatings
    
    def get_score_per_template(self,fingerString, jointNr, jointImage, scoreTable):
        
        trainingStudyNr = self.find_matching_study_nr_per_template_matching(fingerString, jointNr, jointImage)  
        
        print 'Hit template from study nr. ' + str(trainingStudyNr)
        
        #Indices for Cattin's magic score.txt file to fit our joint-numbering.
        idxArray={'littleFinger':[15,12,10],'middleFinger':[14,11,9],'thumb':[13,8]}
        
        #extract score from provided score vs. study table
        idx=idxArray[fingerString][jointNr-1]
        #print 'index for score file:' + str(idx)
        
        if (idx>0):    
            score=scoreTable[trainingStudyNr-1][idx]
            print 'Score for joint ' + str(jointNr) + ' in ' + fingerString + ' is ' + str(score) 
            
            return score
        
        return False
    
    def find_matching_study_nr_per_template_matching(self, fingerString, jointNr, jointImage):
        
        template=jointImage
        target=np.asarray(Image.open('extractedJoints/'+ fingerString + str(jointNr)+'.png'))      #load the training-fingers for the actual finger/joint
        
        box=(50, 20, 80, 120)                   #this is crucial
        boxedTemplate=template.crop(box)        #crop the template
        boxedTemplate=np.asarray(boxedTemplate)
        
        #plt.imshow(boxedTemplate,cmap=plt.cm.gray)
        #plt.show()
        
        #do template matching
        match=feature.match_template(target,boxedTemplate, pad_input=True)
        
        #search max respond
        ij = np.unravel_index(np.argmax(match), match.shape)
        x, y = ij[::-1]
        
        #convert to study-nr
        trainingStudyNr = x/140+1
        
        return trainingStudyNr
        
    def get_hand_mask(self, numpyImage):
        # Adi
        success = False
        
        thresh = self.get_XRay_BG_Threshold( numpyImage )
        treshMask = numpyImage > thresh
        
        h, w = treshMask.shape
        
        # set sites and top to zero as well
        border = 10
        treshMask[0:border,:] = 0
        treshMask[:,0:border] = 0
        treshMask[:,(w-border):w] = 0
        
        labeled, nr_objects = label( treshMask )
        print "Number of objects found is %d " % nr_objects
        
        label_sizes = sum( treshMask, labeled, range(nr_objects + 1) )
        
        idx_of_biggest_label = np.argmax(label_sizes)
        
        print "size of %d " % label_sizes[idx_of_biggest_label]
        
        # our object should fill at least 20 % of the whole image
        
        areaRatio = label_sizes[idx_of_biggest_label] / (h*w)
        if areaRatio > 0.2 :
            success = True
        
        print "Obj. size %d " % label_sizes[idx_of_biggest_label]
        print "Obj. size ratio %f " % areaRatio
                
        treshMask = (labeled == idx_of_biggest_label)
        
        return success, treshMask
    
    def remove_background(self, xRay, maskedBG ):
        return xRay * maskedBG;
    
    def get_fingers_of_interest_franky_approach(self, image ):
        finder = FingertipFinder()
        success, fingers = finder.findFingertips( image )
        if not success :
            print "FingertipFinder Failed"
            return False, [], [], [], [] 
        
        
        startLittle = fingers['little'];
        startRing = fingers['ring'];
        startMiddle = fingers['middle'];
        startPoint = fingers['pointer'];
        
        #switch x-y to y-x
        startLittle = [ startLittle[1], startLittle[0] ]
        startRing = [ startRing[1], startRing[0] ]
        startMiddle = [ startMiddle[1], startMiddle[0] ]
        startPoint = [ startPoint[1], startPoint[0] ]
                        
        littleLine = self.continue_central_line_franky_method( image, startLittle )
        RingLine = self.continue_central_line_franky_method( image, startRing )
        middleLine = self.continue_central_line_franky_method( image, startMiddle )
        pointLine = self.continue_central_line_franky_method( image, startPoint )
        
        littleLine = self.interpolate_central_lines(littleLine)
        RingLine = self.interpolate_central_lines(RingLine)
        middleLine = self.interpolate_central_lines(middleLine)
        pointLine = self.interpolate_central_lines(pointLine)
        
        return True, littleLine, RingLine, middleLine, pointLine
        
        
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
        maxPointingFingerWidht = 0
        curPointingFingerWidht = 0
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
                curPointingFingerWidht = endPointFinger - startPointFinger
                
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
                maxPointingFingerWidht = curPointingFingerWidht # used later fro daumen
                
            
                
        print "Detected 4 fingers over a distance of %d " % currentMax8OrderCount
        
        detectedFingerRatio = float(currentMax8OrderCount) / float(maskH)
        
        if  detectedFingerRatio < 0.05 :# continous distance needs to be at least 10% of image height
            return False, [], [], [], [], []
        
        # continous growing
        maxLittleFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxLittleFingerCenters )
        maxRingFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxRingFingerCenters )
        maxMiddleFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxMiddleFingerCenters )
        maxPointFingerCenters = self.continue_central_line( dRowMaskIsPeak, maxPointFingerCenters )
        
                
        # detect Daumen downwards right of pointing finger
        lastPointingFingerPos = maxPointFingerCenters[-1]
        minRightPosition = lastPointingFingerPos[1]
        
        minDaumenThickness = maxPointingFingerWidht*0.2
        maxDaumenThickness = maxPointingFingerWidht*2.5
        
        peakTableSize = dRowMaskIsPeak.shape
        DaumenCenters = []
        for rowIter in range(lastPointingFingerPos[0], peakTableSize[0]):
            currentRowPeaks = dRowMaskIsPeak[rowIter,:]
            currentRowPeaksSize = currentRowPeaks.shape
            endDaumen = 0
            startDaumen = 0
            ctrlPoint = 0
            cPeakIdx = 0
            
            for cRowPeakIter in range( currentRowPeaksSize[0] -1 , 0, -1) : # starting from right
                if currentRowPeaks[cRowPeakIter] :    
                    if cPeakIdx == 0:
                        endDaumen = cRowPeakIter
                        
                    if cPeakIdx == 1:
                        startDaumen = cRowPeakIter
                        
                    if cPeakIdx == 2:
                        ctrlPoint = cRowPeakIter
                        
                    cPeakIdx = cPeakIdx + 1
            
            currentDaumenThickness = endDaumen - startDaumen
            if startDaumen > minRightPosition and currentDaumenThickness < maxDaumenThickness and currentDaumenThickness > minDaumenThickness : # -> end and start daumen are set
                DaumenCenters.append( [ rowIter ,  (endDaumen  + startDaumen )/2 ] )
                break
        
        DaumenCenters = self.continue_central_line( dRowMaskIsPeak, DaumenCenters )
        
        if len(DaumenCenters) == 0 :
            plt.imshow(handmaskImage)
            plt.show()
        
        
        # Central Line Interpolation
        maxLittleFingerCenters = self.interpolate_central_lines(maxLittleFingerCenters)
        maxRingFingerCenters = self.interpolate_central_lines(maxRingFingerCenters)
        maxMiddleFingerCenters = self.interpolate_central_lines(maxMiddleFingerCenters)
        maxPointFingerCenters = self.interpolate_central_lines(maxPointFingerCenters)
        DaumenCenters = self.interpolate_central_lines(DaumenCenters)
                
                
                
        
        return True, maxLittleFingerCenters, maxRingFingerCenters, maxMiddleFingerCenters, maxPointFingerCenters, DaumenCenters
        
        
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
        
        if(self.verbosity>0):
            plt.figure(figsize=(8, 4))
            plt.subplot(121) 
            plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
            plt.contour(handmaskImage, [0.5], colors='w')
        
            plt.show()
        
        return True
    
    def continue_central_line(self, peaks, currentCenters ):
        
        if len(currentCenters) == 0:
            print "continue_central_line invalid argument"
            return []
            
        
        h, w = peaks.shape
        
        # growing downwards
        lastPoint = currentCenters[-1]
        lastCenter = lastPoint[1]     
        lowIdx, upIdx = self.get_closest_lower_and_upper_true_idx(peaks[ lastPoint[0] , :], lastCenter )
        firstThickness = upIdx - lowIdx
        minThickness = firstThickness * 0.5
        maxThickness = firstThickness * 2.0
        
        for i in range( lastPoint[0] , h) :
            lowIdx, upIdx = self.get_closest_lower_and_upper_true_idx(peaks[i,:], lastCenter )
            
            center = (upIdx + lowIdx)/2   
            diffCenter = center - lastCenter
            currentThickness = upIdx - lowIdx
            
            if np.sqrt(diffCenter*diffCenter) > 5 or currentThickness < minThickness or currentThickness > maxThickness: # stop the process when thickness difference becomes to big
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
            currentThickness = upIdx - lowIdx
            
            if np.sqrt(diffCenter*diffCenter) > 5 or currentThickness < minThickness or currentThickness > maxThickness: # stop the process when thickness difference becomes to big
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
     
     
    def interpolate_central_lines(self, currentCenters):
        
        if len(currentCenters) == 0:
            print "interpolate_central_lines - invalid argument"
            return []
        
        #interpolate downwards along direction vector of last third
        centerCount = len(currentCenters)
        
        # y = a*x + b -> a = (p1y-p0y)/(p1x-p0x)
        # b = y-(a*x) = p0y-(a*p0x)
        # interpolate by x = (y-b)/a
        
        p0 = currentCenters[centerCount/3*2]
        p1 = currentCenters[-1]
        
        p0x = float(p0[1])
        p0y = float(p0[0])
        p1x = float(p1[1])
        p1y = float(p1[0])
        
        deltaX = p1x-p0x
        if abs(deltaX) < 0.0001: # in case deltaX is 0
            deltaX = 0.0001
        
        a = (p1y-p0y)/deltaX
        b = p0y-(a*p0x)
        
        for y in range(p1[0], p1[0] + centerCount): #continue line by original length
            x = (y-b)/a
            currentCenters.append([y,x])
            
        return currentCenters
       
    def continue_central_line_franky_method(self, image, initialCenter ):
        
        if len(initialCenter) == 0:
            print "continue_central_line invalid argument"
            return []
        
        currentCenters = []
        currentCenters.append( initialCenter )
        
        smoothed = median_filter(image, radius=2, mask=None, percent=50)
        h, w = smoothed.shape
        
        # compute initial background value
        backgroundLine = smoothed[ initialCenter[0] , range( initialCenter[1], initialCenter[1]+100 ) ]
        minValueOnLine = min( backgroundLine )
        avgValueOnLine = mean(backgroundLine )
        backgroundBorderVal = (minValueOnLine + avgValueOnLine)/2       
        
        # growing downwards
        leftIdx, rightIdx = self.get_left_and_right_border( image[ initialCenter[0] , :], initialCenter[1] , backgroundBorderVal )  
        firstThickness = rightIdx - leftIdx
        lastCenter = (rightIdx + leftIdx)/2   
        minThickness = firstThickness * 0.5
        maxThickness = firstThickness * 2.0
        
        for i in range( initialCenter[0] , h) :
            leftIdx, rightIdx = self.get_left_and_right_border( image[ i , :], lastCenter , backgroundBorderVal )  
            
            center = (rightIdx + leftIdx)/2   
            diffCenter = center - lastCenter
            currentThickness = rightIdx - leftIdx
            
            if np.sqrt(diffCenter*diffCenter) > 5 or currentThickness < minThickness or currentThickness > maxThickness: # stop the process when thickness difference becomes to big
                break
            
            lastCenter = center
            currentCenters.append( [i, center] )
            
        # growing upwards
        leftIdx, rightIdx = self.get_left_and_right_border( image[ initialCenter[0] , :], initialCenter[1] , backgroundBorderVal )  
        lastCenter = (rightIdx + leftIdx)/2   
        
        
        for i in range( initialCenter[0] , 0, -1) :
            leftIdx, rightIdx = self.get_left_and_right_border( image[ i , :], lastCenter , backgroundBorderVal )  
            
            center = (rightIdx + leftIdx)/2   
            diffCenter = center - lastCenter
            currentThickness = rightIdx - leftIdx
            
            if np.sqrt(diffCenter*diffCenter) > 5 or currentThickness < minThickness or currentThickness > maxThickness: # stop the process when thickness difference becomes to big
                break
            
            lastCenter = center
            currentCenters.insert(0, [i,center])
             
        return currentCenters
    
                    
    def get_left_and_right_border( self, array1D, targetIdx, background ):
        arrSh = array1D.shape
        rightIdx = 0
        leftIdx = 0
        
        # look right
        for i in range( targetIdx, arrSh[0] ) :
            if array1D[i] < background :
                rightIdx = i
                break
                
        # look left
        for i in range( targetIdx, 0, -1 ) : 
            if array1D[i] < background :
                leftIdx = i
                break
                
        return leftIdx, rightIdx
    
    
    def read_intensities_of_point_set(self, pointset, img):
        h,w = img.shape
        
        intensities = []
        
        for point in pointset :
            y = point[0]
            x = point[1]
            
            i = 0.0
            
            if x >= 0 and y >= 0 and x < w and y < h :
                i = float( img[y,x] )
            
            intensities.append(i)
        
        return intensities
    
    def find_joints_from_intensities(self, intensities, wSize, maxJoints ):
            
        nI = len(intensities)
        
        wSizeHalf = wSize / 2
            
        sumDiffArr = np.zeros(nI)
        avgDiffArr = 0
        count = 0
        max = 0
        for i in range(wSizeHalf, nI-wSizeHalf):
            w = intensities[i-wSizeHalf:i+wSizeHalf]
            dw = np.diff(w)
            dAccum = 0
            for dwi in dw:
                dAccum = dAccum + abs(dwi)
                
            sumDiffArr[i] = dAccum
            avgDiffArr = avgDiffArr + dAccum
            count = count + 1
            if max < dAccum :
                max = dAccum
            
        
        if count == 0 :
            print "find_joints_from_intensities to small fingerline error"
            return []
        
        avgDiffArr = avgDiffArr/count
        peakThreshold = (max - avgDiffArr) / 3.0
        print "Joint Detection DiffWin Threshold %d " % peakThreshold
        
        modPeaks, valeys = peakdet.peakdet(sumDiffArr[wSizeHalf:nI-wSizeHalf], peakThreshold)
        # correct peak index
        peaks = [];
        for pk in modPeaks:
            peaks.append([ pk[0]+wSizeHalf, pk[1] ])
       
       
        # filter peaks
        
        # remove first peak if too close to beginning
        if len(peaks) > 0 :
            if peaks[0][0] < nI*0.08 :
                peaks.remove(peaks[0])
                
        # remove peaks if too close together
        minDist = nI*0.08
        lastPeakPos = -100
        for pk in peaks :
            distToPrevious = pk[0] - lastPeakPos
            if distToPrevious < minDist :
                peaks.remove(pk)
            else :
                lastPeakPos = pk[0]
          
        # just remove all peaks more than maxJoints :))
        if len(peaks) > maxJoints:  
            peaks = peaks[0:maxJoints]
                                          
        # uncomment if you want to see graphs 
        return peaks
        
        peakTable = np.zeros(nI)
        for pk in peaks:
            peakTable[ int(pk[0]) ] = pk[1] 
                    
        if(self.verbosity>0):
            plt.plot( intensities )
            plt.plot( sumDiffArr )
            plt.plot( peakTable )
            plt.show()
            
        return peaks      
      
    def get_XRay_BG_Threshold(self, pilImage ):
        
        # remove 0 value pixel
        histData = []
        
        h,w = pilImage.shape
        for y in range(0,h):
            for x in range(0,w):
                val = pilImage[y,x]
                if not val == 0:
                    histData.append(val)
            
    
        grayhist, bins = np.histogram( histData, 200 ) #pilImage.flatten(),  200 )
   
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
        threshVal = (val2 + val1) / 2
    
        #plt.plot(grayhist)
        #plt.show()
    
        return threshVal
    
    def compute_rotated_rect(self, directionVector, centerPoint, sideLength):
        
        # working in x-y order
             
        rotMatrix = np.matrix( (( 0, -1), ( 1,  0)) ) # 90 deg rotation
        
        udV1 = directionVector / norm( directionVector )
        udV2 = rotMatrix * udV1  # rotate by 90 degree
        
        centerP = np.array( [[ centerPoint[1] ],[ centerPoint[0] ]] )
        
        p1 = centerP + udV1*sideLength*0.5 - udV2*sideLength*0.5
        p2 = p1 + udV2*sideLength
        p3 = p2 - udV1*sideLength
        p4 = p3 - udV2*sideLength
        
        # convert to y-x space
        rect = []
        rect.append([ p1[1,0], p1[0,0] ])
        rect.append([ p2[1,0], p2[0,0] ])
        rect.append([ p3[1,0], p3[0,0] ])
        rect.append([ p4[1,0], p4[0,0] ])
        
        return rect
    
    def draw_joints_to_img(self, img, fingers, joints, rectSideLength ):
        
        plt.imshow(img, cmap=cm.Greys_r)
           
        jRect = rectSideLength
        
        for idx in range(0,len(joints)) :
            currentFingerJoints = joints[idx]
            currentCenterLine = fingers[idx]
            
            xData = []
            yData = []
            for i in currentCenterLine:
                xData.append(i[1])
                yData.append(i[0])
                
            plt.plot(xData, yData, "r")
            
            for joint in currentFingerJoints:
                arrIdx = int(joint[0])
                coord = currentCenterLine[ arrIdx ]
                                
                plt.plot( coord[1], coord[0], ".b")
                                     
                # get direction vect
                p0V = np.array( [[ currentCenterLine[ arrIdx-10 ][1] ],[ currentCenterLine[ arrIdx-10 ][0] ]] )
                p1V = np.array( [[ currentCenterLine[ arrIdx+10 ][1] ],[ currentCenterLine[ arrIdx+10 ][0] ]] )
                dV = p1V - p0V 
                rect = self.compute_rotated_rect( dV, coord, jRect )
                
                rectX = [ rect[0][1], rect[1][1], rect[2][1], rect[3][1], rect[0][1] ]
                rectY = [ rect[0][0], rect[1][0], rect[2][0], rect[3][0], rect[0][0] ]
                plt.plot(rectX, rectY, "r")
        
                                             
        plt.show() 

    def crop_joint(self, img, finger, fingers, rectSideLength):
           
        cropedJoints = [] 
        for joint in fingers:
            arrIdx = int(joint[0])
            coord = finger[ arrIdx ]
            xc = coord[1]
            yc = coord[0]
                                   
            # compute angle
            p0V = np.array( [[ finger[ arrIdx-10 ][1] ],[ finger[ arrIdx-10 ][0] ]] )
            p1V = np.array( [[ finger[ arrIdx+10 ][1] ],[ finger[ arrIdx+10 ][0] ]] )
            dV = p1V - p0V 
            refV = np.array( [[ 0.0 ],[ 1.0 ]] )
            angle = atan2( refV[0]*dV[1] - refV[1]*dV[0], refV[0]*dV[0] + refV[1]*dV[1] )
            angleDeg = math.degrees(angle)
             
            # crop, rotate and recrop
            cropWidhtH = int( sqrt( rectSideLength * rectSideLength / 2 ) )
            ystart = yc-cropWidhtH
            yend = yc+cropWidhtH
            xstart = xc-cropWidhtH
            xend = xc+cropWidhtH
            crop = img[ ystart:yend, xstart:xend]
            
            cropRot = scipy.ndimage.interpolation.rotate(crop, angleDeg, reshape=False )
            
            #recrop
            cropRot = cropRot[ cropWidhtH-(rectSideLength/2):cropWidhtH+(rectSideLength/2),cropWidhtH-(rectSideLength/2):cropWidhtH+(rectSideLength/2)]
            
            cropedJoints.append(cropRot)
        
        return cropedJoints

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

    def setVerbosity(self,verbosity):

        self.verbosity=verbosity
