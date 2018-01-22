import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

from Keras import KerasModelClassification


class KerasResNet50ModelClassification(KerasModelClassification.KerasModelClassification):

    def loadModel(self,model):
        return ResNet50()

    def preprocess(self,im):
        img = image.load_img(im,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    def postprocess(self,preds):
        prediction=decode_predictions(preds, top=1)[0]
        return prediction[0][1]
