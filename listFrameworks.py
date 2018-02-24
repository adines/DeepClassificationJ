import inspect
import os

def listFrameworks():
    path=inspect.stack()[0][1]
    pos=path.rfind(os.sep)
    path=path[:pos+1]
    frameworks=[]
    if(len(path)==0):
        path='.'
    dirs=os.listdir(path=path)

    for d in dirs:
        if os.path.isdir(d) and not d.startswith("__") and not d.startswith("."):
            frameworks.append(d)
    return frameworks

print(listFrameworks())