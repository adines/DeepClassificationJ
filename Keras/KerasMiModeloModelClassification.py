import numpy as np
from keras.preprocessing import image

from Keras import KerasModelClassification


class KerasMiModeloModelClassification(KerasModelClassification.KerasModelClassification):
    def preprocess(self,im):
        img = image.load_img(im,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x
