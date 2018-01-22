import numpy as np
from keras.preprocessing import image

from Keras import KerasModelClassification


class KerasResNetISICModelClassification(KerasModelClassification.KerasModelClassification):
    def preprocess(self,im):
        img = image.load_img(im,target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x

    def postprocess(self,preds):
        max=np.argmax(preds)
        classes=["MSK-1","MSK-2","MSK-3","MSK-4","MSK-5","SONIC","UDA-1"]
        return classes[max]
