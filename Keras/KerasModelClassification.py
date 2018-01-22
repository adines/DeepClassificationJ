import IModelClassification
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import inspect

class KerasModelClassification(IModelClassification.IModelClassification):

    def loadModel(self,model):
        # load json and create model
        path=inspect.stack()[0][1]
        pos=path.rfind('\\')
        pathModelo=path[:pos+1]+'Classification/model/'+model+'.json'
        json_file = open(pathModelo, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        pathWeights=path[:pos+1]+'Classification/weights/'+model+'.h5'
        loaded_model.load_weights(pathWeights)
        return loaded_model

    def predict(self,image,model):
        modelo=self.loadModel(model)
        preprocessImage=self.preprocess(image)
        y_pred=modelo.predict(preprocessImage)
        return self.postprocess(y_pred)

    def preprocess(self,im):
        img = image.load_img(im)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return x

    def postprocess(self,preds):
        return preds
