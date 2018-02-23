import IModelClassification
from torch.autograd import Variable
import torch
from PIL import Image
from torchvision import transforms
import inspect
import importlib
import os

class PyTorchModelClassification(IModelClassification.IModelClassification):
    def loadModel(self,model):
        # Definir una clase con la estructura del modelo
        # Cargar la estructura del modelo a través del constructor de la clase creada
        # Cargar los pesos de la red con model.load_state_dict(torch.load(PATH))
        # Devolver el modelo
        framework="PyTorch"
        class_name =model
        packageName=framework+".Classification.model"
        class_ = getattr(importlib.import_module(packageName + "." + class_name), class_name)
        modelo=class_()


        path = inspect.stack()[0][1]
        pos = path.rfind(os.sep)
        path = path[:pos + 1]
        pathWeights = 'Classification'+os.sep+'weights'+os.sep + model + '.pth'
        pathWeights = path + pathWeights
        modelo.load_state_dict(torch.load(pathWeights))
        return modelo

    def predict(self,image,model):
        modelo=self.loadModel(model)
        preprocessImage = self.preprocess(image)
        y_pred = modelo(preprocessImage)
        return self.postprocess(y_pred)

    def preprocess(self,image):
        img_pil=Image.open(image)
        prep = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        img_prep=prep(img_pil)
        img_var=Variable(img_prep)
        img_var=img_var.unsqueeze(0)
        return img_var

    def postprocess(self,preds):
        pass