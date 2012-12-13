from classify.base import jointClassifier

from math import sqrt
import numpy as np
from sklearn import decomposition

class PCAClassifier(jointClassifier):
    
    def __init__(self):
        super(PCAClassifier, self).__init__()
        
        self.PCA_models = {}
        self.classes = {}
        self.classMeans = {}
        
        self.firstRun=1;
    
    def buildPCAModels(self):
        
        for finger in ['littleFinger', 'middleFinger']:
            
            self.thisFinger_models = {};
            self.thisFinger_classes = {};
            self.thisFinger_classMeans = {};
            
            for joint in ['proximal','middle','distal']:
                
                images = self.classImages[finger][joint]
                classes = self.classLabels[finger][joint]
                
                img = images[1]
                nDims = img.shape[0]*img.shape[1]
                
                X = np.zeros((len(images),nDims))
                
                for i in range(1,len(images)):
                    img = images[i]
                    vec = np.reshape(img,(1,nDims))
                    X[i,:] = vec
                
                pca = decomposition.PCA(n_components=6)
                pca.fit(X)
                X_transformed = pca.transform(X)
                
                classes_unique = np.unique(np.array(classes))
                classMeans = np.zeros((len(classes_unique),pca.n_components))
                for i in range(0,len(classes_unique)):
                    c = classes[i]
                    indices = np.where(np.array(classes)==c)[0]
                    samples = X_transformed[indices,:]
                    classMeans[i,:] = np.mean(samples, 0)
                    
                self.thisFinger_models[joint] = pca;
                self.thisFinger_classes[joint] = classes_unique;
                self.thisFinger_classMeans[joint] = classMeans;
            
            self.PCA_models[finger] = self.thisFinger_models;
            self.classes[finger] = self.thisFinger_classes;
            self.classMeans[finger] = self.thisFinger_classMeans;
        
        self.firstRun=0;
        
        return
    
    def classify(self,sample,fingerName,jointName):
        
        if(self.firstRun):
            self.buildPCAModels()
            
        # project sample into PCA space
        pca = self.PCA_models[fingerName][jointName]
        nDims = sample.shape[0]*sample.shape[1]
        vec = np.reshape(sample,(1,nDims))
        coeff = pca.transform(vec)[0]

        # compute mahalanobis distance (simply the L2-norm in PCA space)
        classMeans = self.classMeans[fingerName][jointName]
        classLabels = self.classLabels[fingerName][jointName]
        
        dists = np.zeros(len(classMeans))
        for c in range(0,len(classMeans)):
            # compute distance
            mean = classMeans[c]
            for i in np.arange(0,len(mean)):
                dists[c] = dists[c] + sqrt(np.abs(coeff[i]*mean[i]))
        
        return classLabels[np.argmin(dists)]
    
    
