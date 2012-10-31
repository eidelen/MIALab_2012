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

for i in range(0,len(scores)-1):
	if(i+1==4 or i+1==7 or i+1 ==12):
		# for all datasets that make the program fail (Aedu, fix if you have time)
		continue

	(reader, img) = dicom.open_image("../training/Case" + str(i+1) + ".dcm")
	joints = aClass.detect_joints_of_interest(img)
	
	if(len(joints)==3):
		if(len(joints['littleFinger'])==3):
			jointNum=1
			for joint in joints['littleFinger']:
				imsave(directory + '/' + str(i+1) + "_lf_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1

		if(len(joints['middleFinger'])==3):
                        jointNum=1
                        for joint in joints['middleFinger']:
                                imsave(directory + '/' + str(i+1) + "_mf_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1

		if(len(joints['thumb'])==3):
                        jointNum=1
                        for joint in joints['thumb']:
                                imsave(directory + '/' + str(i+1) + "_th_" + str(jointNum) + ".png", joint)
				jointNum = jointNum+1

		



