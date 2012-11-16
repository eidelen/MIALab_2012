class jointClassifier(object):

	def __init__(self):
		
		self.classImages = []
		self.classLabels = []
		
		if self.__class__ is jointClassifier:
			raise NotImplementedError
	
	def setClassImages(self,images):
		self.classImages = images
		
	def setClassLabels(self,labels):
		self.classLabels = labels