import IModelClassification
import numpy as np
import sys
sys.path.append(r'C:\Users\adines\Downloads\caffe\python')
import caffe
import os
import inspect

class CaffeModelClassification(IModelClassification.IModelClassification):

    def loadModel(self,model):
        path=inspect.stack()[0][1]
        pos=path.rfind(os.sep)
        path=path[:pos+1]
        pathModelo='Classification'+os.sep+'model'+os.sep+model+'.prototxt'
        pathModelo=path+pathModelo
        pathWeights='Classification'+os.sep+'weights'+os.sep+model+'.caffemodel'
        pathWeights=path+pathWeights
        net=caffe.Classifier(pathModelo, pathWeights,
                       mean=np.load('C:/Users/adines/Downloads/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(255 , 255))
        return net

    def predict(self,image,model):
        modelo=self.loadModel(model)
        preprocessImage=self.preprocess(image)
        prediction = modelo.predict([preprocessImage])
        return self.postprocess(prediction)

    def preprocess(self,im):
        pass

    def postprocess(self,preds):
        pass
