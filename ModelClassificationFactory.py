import DL4JModelClassification
import importlib

class ModelClassificationFactory:
    @staticmethod
    def getClassificationModel(framework,model):
        if framework=="DL4J":
            return DL4JModelClassification.DL4JModelClassification()
        else:
            class_name = framework+model+"ModelClassification"
            class_ = getattr(importlib.import_module(framework+"."+class_name), class_name)
            return class_()
        # return eval(framework+model+"ModelClassification."+framework+model+"ModelClassification()")
