import IModelClassification
from subprocess import *
import os
import inspect


class DL4JModelClassification(IModelClassification.IModelClassification):

    def jarWrapper(self,*args):
        process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
        result = []

        for line in process.stdout:
            result.append(line.decode('utf-8').rstrip('\n').rstrip('\r'))
        return result[0]

    def loadModel(self,model):
        pass

    def predict(self,image,model):
        path=inspect.stack()[0][1]
        pos=path.rfind(os.sep)
        path=path[:pos+1]+'PredictDL4JMaven-1.0-SNAPSHOT.jar'

        args = [path, image, model]
        result = self.jarWrapper(*args)
        return result

    def preprocess(self,im):
        pass

    def postprocess(self,preds):
        pass
