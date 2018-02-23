import numpy as np
from MxNet import MxNetModelClassification
from PIL import Image
import inspect
import os


class MxNetResNet101ModelClassification(MxNetModelClassification.MxNetModelClassification):
    def preprocess(self,image):
        img = Image.open(image)
        img=img.resize((224,224), Image.NEAREST)
        img_arr = np.array(img.getdata()).astype(np.float32).reshape((img.size[0], img.size[1], 3))
        img_arr = np.swapaxes(img_arr, 0, 2)
        img_arr = np.swapaxes(img_arr, 1, 2)
        img_arr = img_arr[np.newaxis, :]
        return img_arr

    def postprocess(self,preds):
        preds = np.squeeze(preds)
        preds = np.argsort(preds)[::-1]
        path = inspect.stack()[0][1]
        pos = path.rfind(os.sep)
        path = path[:pos + 1]
        labels = np.loadtxt(path + "synset_words.txt", str, delimiter='\n')
        return labels[preds[0]]
