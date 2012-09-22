import Image  
import matplotlib.pyplot as plt  
import peakdet

from numpy import *
     
def extract_Bones( pilImage ):
    
    imgFlattenData = pilImage.flatten()
    grayhist, bins = histogram(imgFlattenData,  200 )
    
    avgHistCounts = mean(grayhist)
    minimumPeakDiff = avgHistCounts * 0.20
    print minimumPeakDiff
    
    peaks, valeys = peakdet.peakdet(grayhist, minimumPeakDiff)
    print peaks
    
    
    plt.plot(grayhist)
    plt.show()
    
    
    # max peaks are background
    maxPeakIdx= argmax(peaks[:,1])
    
    
    skinBinIdx = peaks[maxPeakIdx+1,0]
    boneBinIdx = peaks[maxPeakIdx+2,0]
    skinPeakIntensVal = bins[skinBinIdx]
    bonePeakIntensVal = bins[boneBinIdx]

    threshVal = (bonePeakIntensVal + skinPeakIntensVal) / 2.0
    
    plt.plot(grayhist)
    plt.show()
    
    
    retImage = where(pilImage > threshVal, 1, 0)

    return retImage
    
