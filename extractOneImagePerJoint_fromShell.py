#> ipython --pylab --deep-reload

import os
import numpy as np
import dicom
from AgeDetermination import AgeDetermination
from scipy.misc import imsave
import Image

scores = np.loadtxt('../training/scores.txt')
aClass = AgeDetermination()

directory = '../extractedJoints'
if not os.path.exists(directory):
	os.makedirs(directory)
	
#Add numbers of the working studies here:	
okImg=[1,3,10,23,28,29,30];

size=140*30,140

imgLF1=Image.new('L',size)
imgLF2=Image.new('L',size)
imgLF3=Image.new('L',size)

imagesLF=list()
imagesLF.append(imgLF1)
imagesLF.append(imgLF2)
imagesLF.append(imgLF3)

for i in okImg:

	(reader, img) = dicom.open_image("../training/Case" + str(i) + ".dcm")
	joints = aClass.detect_joints_of_interest(img)
	
	if(len(joints)==3):
		if(len(joints['littleFinger'])==3):
			jointNum=1
			for joint in joints['littleFinger']:
				#imagesLF[jointNum-1].paste(Image.fromarray((255.0*255.0/joint.max()*(joint-joint.min())).astype(np.uint16)),((i-1)*140,0))
				imagesLF[jointNum-1].paste(Image.fromarray((255.0/joint.max()*(joint-joint.min())).astype(np.uint8)),((i-1)*140,0))
				#imagesLF[jointNum-1].paste(Image.fromarray(joint,'L'),((i-1)*140,0))
				jointNum = jointNum+1
jointNum=1
for img in imagesLF:
	img.save(directory+"/LittleFinger" + str(jointNum) + ".png")
	jointNum=jointNum+1