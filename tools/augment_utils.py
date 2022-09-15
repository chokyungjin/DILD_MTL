
import numpy as np
import cv2
import torch

class Compose:
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,image,label):
        for t in self.transforms:
            image,label  = t(image,label)
        return image,label

class HFlip:
    def __init__(self, p=0.5):
        self.p=p
    def __call__(self,image,label):
        if np.random.rand() < self.p:
            
           image = cv2.flip(image,1)
           label = cv2.flip(label,1)

        return image,label

    
class VFlip:
    def __init__(self, p=0.5):
        self.p=p
    def __call__(self,image,label):
       
        if np.random.rand() < self.p:

           image = cv2.flip(image,-1)
           label = cv2.flip(label,-1)
        return image,label


class Resize_segmentation:
    def __init__(self,size):
        self.size = (size,size)
    def __call__(self,image,label):
        image = cv2.resize(image,self.size,interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label,self.size,interpolation=cv2.INTER_NEAREST)
      
        return image,label
        
class Resize_classification:
    def __init__(self,size):
        self.size = (size,size)
        
    def __call__(self,image,label):
        image = cv2.resize(image,self.size,interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label,self.size,interpolation=cv2.INTER_NEAREST)
      
        return image,label


class Normalize:
    def __init__(self,mean,std):
        self.mean=mean
        self.std = std
    def __call__(self,image,label):        
        image = image.astype(np.float32) / 255
        label = label.astype(np.float32) / 255

        image = (image - self.mean)/self.std #(340,340,3)
        label = label[...,np.newaxis] # (340,340,1)

        image = image.transpose((2,0,1)) # (3,340,340)
        label = label.transpose((2,0,1)) # (1,340,340)

        return torch.from_numpy(image), torch.from_numpy(label)



class Normalize_For_NII:
    def __init__(self,mean,std):
        self.mean=mean
        self.std = std
    def __call__(self,image):        
        image = image.astype(np.float32) / 255
        image = (image - self.mean)/self.std #(340,340,3)
        image = image.transpose((2,0,1)) # (3,340,340)

        return torch.from_numpy(image)




class Resize_For_NII:
    def __init__(self,size):
        self.size = (size,size)
    def __call__(self,image):
        image = cv2.resize(image,self.size,interpolation=cv2.INTER_CUBIC)

        return image

        

class Compose_For_Video:
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,image):
        for t in self.transforms:
            image = t(image)
        return image

class Resize_For_Video:
    def __init__(self,size):
        self.size = (size,size)
    def __call__(self,image):
        image = cv2.resize(image,self.size,interpolation=cv2.INTER_CUBIC)
    
        return image

class Normalize_For_Video:
    def __init__(self,mean,std):
        self.mean=mean
        self.std = std
    def __call__(self,image):        
        image = image.astype(np.float32) / 255

        image = (image - self.mean)/self.std #(340,340,3)

        image = image.transpose((2,0,1)) # (3,340,340)

        return torch.from_numpy(image)


