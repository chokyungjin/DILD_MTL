
import os

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

    return path



def boolean(v):
    if isinstance(v,bool):
        return v

    if v.lower() in ['true']:
        return True
    elif v.lower() in ['false']:
        return False