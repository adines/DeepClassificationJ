import inspect
import os
import argparse
from subprocess import *
import json


def jarWrapper( *args):
    process = Popen(['java', '-jar'] + list(args), stdout=PIPE, stderr=PIPE)
    result = []
    for line in process.stdout:
        result.append(line.decode('utf-8').rstrip('\n').rstrip('\r'))
    return result[0]

def listModels(framework):
    models=[]
    path = inspect.stack()[0][1]
    pos = path.rfind(os.sep)
    path = path[:pos]
    if pos==-1:
        path=framework
    else:
        path=path+os.sep+framework
    if framework!='DL4J':
        files=os.listdir(path=path)
        for file in files:
            if file.endswith("ModelClassification.py") and file.startswith(framework) and len(file)!=len(framework+"ModelClassification.py"):
                pos=file.find("ModelClassification.py")
                models.append(file[len(framework):pos])
    else:
        path = path + os.sep+'PredictDL4JMaven-1.0-SNAPSHOT.jar'
        args = [path]
        result = jarWrapper(*args)
        models=result.split(",")
    return models

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--framework", required=True, help="Framework to obtain its models")
    args=vars(parser.parse_args())

    data={}
    data['type']='models'
    data['framework']=args["framework"]
    data['models']=listModels(args["framework"])
    with open('data.json','w') as f:
        json.dump(data,f)
