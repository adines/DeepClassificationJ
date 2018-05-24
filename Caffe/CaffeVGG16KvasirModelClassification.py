import sys
sys.path.append(r'C:\Users\adines\Downloads\caffe\python')
import caffe
from Caffe import CaffeModelClassification


class CaffeVGG16KvasirModelClassification(CaffeModelClassification.CaffeModelClassification):
    def preprocess(self,im):
        input_image = caffe.io.load_image(im)
        return input_image

    def postprocess(self,preds):
        labels=["dyed-lifted-polyps","dyed-resection-margins","esophagitis","normal-cecum","normal-pylorus","normal-z-line","polyps","ulcerative-colitis"]
        return labels[preds[0].argmax()]
