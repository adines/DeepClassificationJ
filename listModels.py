import inspect
import os
import argparse

def listModels(framework):
    models=[]
    path = inspect.stack()[0][1]
    pos = path.rfind(os.sep)
    path = path[:pos + 1]
    if len(path)==0:
        path=framework
    else:
        path=path+os.sep+framework
    files=os.listdir(path=path)
    for file in files:
        if file.endswith("ModelClassification.py") and file.startswith(framework) and len(file)!=len(framework+"ModelClassification.py"):
            pos=file.find("ModelClassification.py")
            models.append(file[len(framework):pos])
    return models

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--framework", required=True, help="Framework to obtain the models")
args=vars(parser.parse_args())
print(listModels(args["framework"]))