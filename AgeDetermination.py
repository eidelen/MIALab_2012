# MIA Lab - F.Preiswerk, J.Walti, A.Schneider
import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import peakdet

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


class AgeDetermination:
    
    def detect_joints_of_interest(self, pilImage ):
        
        handmask = self.get_hand_mask(pilImage)
        
        self.get_fingers_of_interest(handmask)
        
        return handmask
        
        
        
    def get_hand_mask(self, pilImage):
        # Adi
        
        thresh = self.get_XRay_BG_Threshold( pilImage )
        treshMask = pilImage > thresh
        
        labeled, nr_objects = label( treshMask )
        print "Number of objects found is %d " % nr_objects
        
        label_sizes = sum(treshMask, labeled, range(nr_objects + 1))
        
        print label_sizes
        idx_of_biggest_label = np.argmax(label_sizes)
        print "biggest label is %d " % idx_of_biggest_label
        print "size of %d " % label_sizes[idx_of_biggest_label]
        
        treshMask = labeled == idx_of_biggest_label
        
        plt.imshow( treshMask )
        plt.show()
        
        return treshMask
        
        
    def get_fingers_of_interest(self, handmaskImage ):
        # Wale
        handmaskImage = median_filter(handmaskImage, radius=6, mask=None, percent=50)
        
        plt.imshow(handmaskImage)
        
        # Compute the medial axis (skeleton) and the distance transform
        skel, distance = medial_axis(handmaskImage, return_distance=True)
        
        # Distance to the background for pixels of the skeleton
        dist_on_skel = distance * skel
        
        plt.figure(figsize=(8, 4))
        plt.subplot(121) 
        plt.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
        plt.contour(handmaskImage, [0.5], colors='w')
        
        plt.show()
        
        return 0
    
    

    def extract_Bones(self, pilImage ):

        thresh = self.get_XRay_BG_Threshold( pilImage )
        treshMask = pilImage > thresh
        plt.imshow( treshMask )
        plt.title('threshold mask', fontsize=20)
        plt.show()
    
        #get_Fingers(treshMask)
    
        maskedImage = pilImage * treshMask
    
        adaptiveMask = threshold_adaptive(maskedImage, 15, 'mean')
    
        plt.imshow( adaptiveMask )
        plt.title('adaptive mask', fontsize=20)
        plt.show()
    
        maskedImage = pilImage * adaptiveMask
        plt.imshow( maskedImage )
        plt.title('adaptive masked image', fontsize=20)
        plt.show()
    
        cannyS5 = filter.canny(maskedImage, 3)
        plt.imshow( cannyS5, cmap=plt.cm.gray)
        plt.title('tresh - canny sigma 3', fontsize=20)
        plt.show()

        # work on subsampled image to reduce complexity and improve skeletonization result
	   #pilImage_small = pilImage.resize(50)
        pilImage_small = imresize(pilImage, .5 )
        
        smooth = gaussian_filter(pilImage_small, 5)
        thresh = self.get_XRay_BG_Threshold( smooth )
        binaryMask = smooth
        binaryMask[binaryMask<thresh]=0
        binaryMask[binaryMask>=thresh]=1
        plt.imshow(binaryMask)
        plt.title('Binary mask from smoothed image')
        plt.show()
        skel = skeletonize(binaryMask)
        plt.imshow(skel)
        plt.title('Skeletonized image')
        plt.show()
    	overlay = pilImage_small
    	overlay[skel==1]=1
    	plt.imshow(overlay)
    	plt.title('Original image with skeleton overlaid')
    	plt.show()

        return pilImage
    
    
    #def get_Fingers( binaryImage ):
   
    #def remove_first_uniform_layer( mask, grayImage ):
    	# region growing seeded at the masks boarder
      
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

	
 

