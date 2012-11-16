#> ipython --pylab --deep-reload

import os
import numpy as np
import dicom
from AgeDetermination import AgeDetermination
from scipy.misc import imsave

scores = np.loadtxt('../training/scores.txt')
aClass = AgeDetermination()

directory = '../extractedJoints'
if not os.path.exists(directory):
	os.makedirs(directory)
	
#Add numbers of the working studies here:	
okImg=[1,3,10,23,28,29,30];

for i in okImg:

	(reader, img) = dicom.open_image("../training/Case" + str(i) + ".dcm")
	joints = aClass.detect_joints_of_interest(img)
	
	if(len(joints)==3):
		if(len(joints['littleFinger'])==3):
			jointNum=1
			for joint in joints['littleFinger']:
				imsave(directory + '/' + str(i) + "_littleFinger_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1
				
		if(len(joints['middleFinger'])==3):
			jointNum = 1
			for joint in joints['middleFinger']:
				imsave(directory + '/' + str(i) + "_middleFinger_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1

		if(len(joints['thumb'])==2):
			jointNum=1
			for joint in joints['thumb']:
				imsave(directory + '/' + str(i) + "_thumb_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1
