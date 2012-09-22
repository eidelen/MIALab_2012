import Image  
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches
import peakdet

import numpy as np


from skimage import measure
from skimage import filter
from skimage import data
from skimage.filter import *
#from skimage.segmentation import find
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
     
def extract_Bones( pilImage ):

    thresh = get_XRay_BG_Threshold( pilImage )
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
    
    return pilImage
    
    
#def get_Fingers( binaryImage ):
   
#def remove_first_uniform_layer( mask, grayImage ):
    # region growing seeded at the masks boarder
      
def get_XRay_BG_Threshold( pilImage ):
    
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

    # set threshold closer to lower peak
    threshVal = (2*val1 + val2) / 3.0
    
    #plt.plot(grayhist)
    #plt.show()
    
    
    #retImage = where(pilImage > threshVal, 1, 0)

    return threshVal
    
