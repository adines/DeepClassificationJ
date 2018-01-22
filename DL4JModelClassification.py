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
            # print(line.decode('utf-8'))
        # errcode = process.returncode
        # for line in result:
        #     print(line)


        # while process.poll() is None:
        #     line = process.stdout.readline()
        #     ret.append(line[:-1])
        # stdout, stderr = process.communicate()
        # ret += stdout.split('\n')
        # if stderr != '':
        #     ret += stderr.split('\n')
        # ret.remove('')
        return result[0]

    def loadModel(self,model):
        pass

    def predict(self,image,model):
        # Cambiar para que no haya que poner la ruta absoluta
        # path=inspect.stack()[0][1]+'PredictDL4JMaven-1.0-SNAPSHOT.jar'
        path=inspect.stack()[0][1]
        pos=path.rfind('\\')
        path=path[:pos+1]+'PredictDL4JMaven-1.0-SNAPSHOT.jar'

        # print(path)
        args = [path, image, model]
        result = self.jarWrapper(*args)
        return result

    def preprocess(self,im):
        pass

    def postprocess(self,preds):
        pass
