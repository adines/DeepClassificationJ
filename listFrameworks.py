import inspect
import os
import json

def listFrameworks():
    path=inspect.stack()[0][1]
    pos=path.rfind(os.sep)
    path=path[:pos]
    frameworks=[]
    if(pos==-1):
        path='.'
    dirs=os.listdir(path=path)

    for d in dirs:
        if "." not in d and not d.startswith("__"):
            frameworks.append(d)
    return frameworks

if __name__=="__main__":
    data={}
    data['type']='frameworks'
    data['frameworks']=listFrameworks()
    with open('data.json','w') as f:
        json.dump(data,f)
