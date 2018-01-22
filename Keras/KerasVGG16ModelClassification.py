import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from Keras import KerasModelClassification


class KerasVGG16ModelClassification(KerasModelClassification.KerasModelClassification):

    def loadModel(self,model):
        return VGG16()

    def preprocess(self,im):
        img = image.load_img(im,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def postprocess(self,preds):
        prediction=decode_predictions(preds, top=1)[0]
        return prediction[0][1]
