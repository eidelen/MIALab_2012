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
okImg=[1,3,10,23,28,29,30]

evaluatedFingers = ['littleFinger','middleFinger','thumb']

size=140*33,140

fingerImagesPool=dict()

for finger in evaluatedFingers:
	images=list()
	fingerImagesPool[finger]=images
	
	imgLF1=Image.new('L',size)
	images.append(imgLF1)
	
	imgLF2=Image.new('L',size)
	images.append(imgLF2)
	
	if (images!='thumb'):
		imgLF3=Image.new('L',size)
		images.append(imgLF3)

for i in okImg:
	
	(reader, img) = dicom.open_image("../training/Case" + str(i) + ".dcm")
		
	fingers = aClass.detect_joints_of_interest(img)
	
	for fingerName in evaluatedFingers:
				
		if(len(fingers)==3):
			if(len(fingers[fingerName])>=2):
				jointNum=1
				for joint in fingers[fingerName]:
					#imagesLF[jointNum-1].paste(Image.fromarray((255.0*255.0/joint.max()*(joint-joint.min())).astype(np.uint16)),((i-1)*140,0))
					fingerImagesPool[fingerName][jointNum-1].paste(Image.fromarray((255.0/joint.max()*(joint-joint.min())).astype(np.uint8)),((i-1)*140,0))
					#imagesLF[jointNum-1].paste(Image.fromarray(joint,'L'),((i-1)*140,0))
					jointNum = jointNum+1

#write down images
for finger in fingerImagesPool:
	jointNum=1
	for img in fingerImagesPool[finger]:
		img.save(directory+"/" + finger + str(jointNum) + ".png")
		jointNum=jointNum+1