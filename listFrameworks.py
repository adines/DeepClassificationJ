import inspect
import os

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

print(listFrameworks())