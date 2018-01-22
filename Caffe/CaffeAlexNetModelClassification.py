import numpy as np
import inspect

from Caffe import CaffeModelClassification


class CaffeAlexNetModelClassification(CaffeModelClassification.CaffeModelClassification):
    def postprocess(self,preds):
        path=inspect.stack()[0][1]
        pos=path.rfind('\\')
        path=path[:pos+1]
        labels = np.loadtxt(path+"synset_words.txt", str, delimiter='\n')
        return labels[preds[0].argmax()]
