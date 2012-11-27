#> ipython --pylab --deep-reload

import os
import numpy as np
import dicom
from AgeDetermination import AgeDetermination
from scipy.misc import imsave

#scores = np.loadtxt('../training/scores.txt')
aClass = AgeDetermination()

directory = '../extractedJoints'
if not os.path.exists(directory):
	os.makedirs(directory)
	
#Add numbers of the working studies here:	
okImg=[1,2,3,4,6,7,9,10,11,12,13,15,17,19,20,21,22,23,26,27,28,29,30,33]

for i in okImg:

	(reader, img) = dicom.open_image("../training/Case" + str(i) + ".dcm")
	fingers = aClass.detect_joints_of_interest(img)
	
	if(len(fingers)==2):
		if(len(fingers['littleFinger'])==3):
			jointNum=1
			for joint in fingers['littleFinger']:
				imsave(directory + '/' + str(i) + "_littleFinger_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1
				
		if(len(fingers['middleFinger'])==3):
			jointNum = 1
			for joint in fingers['middleFinger']:
				imsave(directory + '/' + str(i) + "_middleFinger_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1

