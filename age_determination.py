import Image  
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches
import peakdet

import numpy as np


from skimage import measure
from skimage import filter
from skimage import data
from skimage.filter import threshold_otsu
#from skimage.segmentation import find
from skimage.morphology import label, closing, square
from skimage.measure import regionprops
     
def extract_Bones( pilImage ):
    
    
    #edges = filter.sobel( pilImage )
    
    # Find contours at a constant value of 0.8
    contours = measure.find_contours( pilImage, 5000)
    
    # Display the image and plot all contours found
    #plt.imshow( pilImage, interpolation='nearest')
    
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    plt.title('findcountours', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.show()


    #thresh = threshold_otsu( pilImage )
    thresh = get_Threshold( pilImage )
    treshMask = pilImage > thresh
    plt.imshow( treshMask )
    plt.title('own thresh', fontsize=20)
    plt.show()
    
    maskedImage = pilImage * treshMask
    plt.imshow( maskedImage )
    plt.title('masked Image', fontsize=20)
    plt.show()
    
    cannyS5 = filter.canny(maskedImage, 3)
    plt.imshow( cannyS5, cmap=plt.cm.gray)
    plt.title('tresh - canny sigma 3', fontsize=20)
    plt.show()
    
    
    
    
    image = pilImage

    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))
    
    # remove artifacts connected to image border
    cleared = bw.copy()
    #clear_border(cleared)
    
    # label image regions
    label_image = label(cleared)
    borders = np.logical_xor(bw, cleared)
    label_image[borders] = -1
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(label_image, cmap='jet')
    
    for region in regionprops(label_image, ['Area', 'BoundingBox']):
    
        # skip small images
        if region['Area'] < 100:
            continue
    
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region['BoundingBox']
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    plt.title('regions', fontsize=20)
    plt.show()
    
    return pilImage
    
    
def get_Threshold( pilImage ):
    
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

    threshVal = (val1 + val2) / 2.0
    
    #plt.plot(grayhist)
    #plt.show()
    
    
    #retImage = where(pilImage > threshVal, 1, 0)

    return threshVal
    
