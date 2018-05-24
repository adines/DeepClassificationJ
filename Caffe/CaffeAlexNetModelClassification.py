import numpy as np
import inspect
import os
import sys
sys.path.append(r'C:\Users\adines\Downloads\caffe\python')
import caffe
from Caffe import CaffeModelClassification


class CaffeAlexNetModelClassification(CaffeModelClassification.CaffeModelClassification):
    def preprocess(self,im):
        input_image = caffe.io.load_image(im)
        return input_image

    def postprocess(self,preds):
        path=inspect.stack()[0][1]
        pos=path.rfind(os.sep)
        path=path[:pos+1]
        labels = np.loadtxt(path+"synset_words.txt", str, delimiter='\n')
        return labels[preds[0].argmax()]
