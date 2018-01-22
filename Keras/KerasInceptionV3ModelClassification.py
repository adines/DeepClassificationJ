import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from Keras import KerasModelClassification


class KerasInceptionV3ModelClassification(KerasModelClassification.KerasModelClassification):

    def loadModel(self,model):
        return InceptionV3()

    def preprocess(self,im):
        img = image.load_img(im,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def postprocess(self,preds):
        prediction=decode_predictions(preds, top=1)[0]
        return prediction[0][1]
