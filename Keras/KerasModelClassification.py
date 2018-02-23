import IModelClassification
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import inspect
import os

class KerasModelClassification(IModelClassification.IModelClassification):

    def loadModel(self,model):
        path=inspect.stack()[0][1]
        pos=path.rfind(os.sep)
        pathModelo=path[:pos+1]+'Classification'+os.sep+'model'+os.sep+model+'.json'
        json_file = open(pathModelo, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        pathWeights=path[:pos+1]+'Classification'+os.sep+'weights'+os.sep+model+'.h5'
        loaded_model.load_weights(pathWeights)
        return loaded_model

    def predict(self,image,model):
        modelo=self.loadModel(model)
        preprocessImage=self.preprocess(image)
        y_pred=modelo.predict(preprocessImage)
        return self.postprocess(y_pred)

    def preprocess(self,im):
        pass

    def postprocess(self,preds):
        return preds
