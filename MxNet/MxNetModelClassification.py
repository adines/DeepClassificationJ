import IModelClassification
import mxnet as mx
from PIL import Image
import numpy as np
import inspect
import os
from collections import namedtuple

class MxNetModelClassification(IModelClassification.IModelClassification):
    def loadModel(self,model):
        path = inspect.stack()[0][1]
        pos = path.rfind(os.sep)
        pathModelo = path[:pos + 1] + 'Classification/model/' + model + '.json'
        symbol = mx.sym.load(pathModelo)
        pathWeights = path[:pos + 1] + 'Classification/weights/' + model + '.params'
        save_dict = mx.nd.load(pathWeights)
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        mod = mx.mod.Module(symbol=symbol, context=mx.cpu(), label_names=None)
        mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))],
                 label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params, allow_missing=True)
        return mod

    def predict(self,image,model):
        modelo=self.loadModel(model)
        preprocessImage = self.preprocess(image)
        Batch = namedtuple('Batch', ['data'])
        modelo.forward(Batch([mx.nd.array(preprocessImage)]))
        return self.postprocess(modelo.get_outputs()[0].asnumpy())

    def preprocess(self,image):
        pass

    def postprocess(self,preds):
        pass