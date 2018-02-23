from PyTorch import PyTorchModelClassification
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import inspect
import os

class PyTorchResNet18ModelClassification(PyTorchModelClassification.PyTorchModelClassification):
    def loadModel(self,model):
        return models.resnet18(pretrained=True)

    def preprocess(self,image):
        img_pil = Image.open(image)
        prep = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        img_prep = prep(img_pil)
        img_var = Variable(img_prep)
        img_var=img_var.unsqueeze(0)
        return img_var

    def postprocess(self,preds):
        path = inspect.stack()[0][1]
        pos = path.rfind(os.sep)
        path = path[:pos + 1]
        labels = np.loadtxt(path + "synset_words.txt", str, delimiter='\n')
        return labels[preds.data.numpy().argmax()]
