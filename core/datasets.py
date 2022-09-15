
import numpy as np
import os
import cv2
from numpy.core.defchararray import add, translate
import glob
import SimpleITK as sitk
from numpy.core.shape_base import stack
from tools import augment_utils
import json 

from torch.nn import functional as F

import torch



class Infarct_Dataset:

    def __init__(self, root_dir, domain):
        
        self.file_paths=glob.glob(root_dir +domain +'/ADC/*')
         
                  
    def __len__(self):
        return len(self.file_paths) 

    def __getitem__(self,index):
        adc_path = self.file_paths[index]            
        
                   
        b1000_path = adc_path.replace('/ADC','/b1000')
        flair_path = adc_path.replace('/ADC','/flair')
        gt_path = adc_path.replace('/ADC','/infarct_mask')
       

        adc_image = cv2.imread(adc_path,cv2.IMREAD_GRAYSCALE)
        b1000_image = cv2.imread(b1000_path,cv2.IMREAD_GRAYSCALE)
        flair_image = cv2.imread(flair_path,cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)

        return adc_image, b1000_image,flair_image, gt_image



class Infarct_Dataset_For_MultiTask(Infarct_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir,domain)
        
        self.transform = transform

    def __getitem__(self, index):
        adc_image, b1000_image, flair_image, gt_image =  super().__getitem__(index)

        image = np.stack([
            adc_image,b1000_image,flair_image                         
        ], axis=2)
                    
        whole_gt = np.sum(gt_image>0)/(gt_image.shape[0]*gt_image.shape[1])
        
        if whole_gt > 0.05:
            cls_label = np.asarray([1],dtype=np.float32)
        else:
            cls_label = np.asarray([0],dtype=np.float32)
            
        # data augmentation

        if self.transform is not None:
            image,gt_image = self.transform(image,gt_image)        

        return image,gt_image,cls_label


class Infarct_Dataset_From_NII_v2:
    def __init__(self,root_Dir):
        self.adc_paths = glob.glob(root_Dir + '*/ADC.nii')

    def __len__(self):
        return len(self.adc_paths)

    def normalize(self,image,eps=0):
        image = (image-image.min()) / (image.max()-image.min()+eps)
        image = image*255
        return image.astype(np.uint8)


    def __getitem__(self,index):
        adc_path = self.adc_paths[index]
        b1000_path = adc_path.replace('ADC.nii','b1000.nii')
        
        if not os.path.isfile(b1000_path):
            b1000_path=adc_path.replace('ADC.nii','B1000.nii')


        brain_path = adc_path.replace('ADC.nii','brain.nii')

        if not os.path.isfile(brain_path):                      
            brain_path = adc_path.replace('ADC.nii','brain_ADC.nii')
                    
        adc_images = sitk.GetArrayFromImage(sitk.ReadImage(adc_path))
        b1000_images = sitk.GetArrayFromImage(sitk.ReadImage(b1000_path))
        brain_images = sitk.GetArrayFromImage(sitk.ReadImage(brain_path))
        gt_images = None

        adc = np.zeros_like(adc_images,dtype=np.uint8) 
        b1000 = np.zeros_like(adc_images,dtype=np.uint8) 
        brain = np.zeros_like(adc_images,dtype=np.uint8) 
        gt = np.zeros_like(adc_images,dtype=np.uint8) 

        for index in range(adc_images.shape[0]):
            adc[index] =self.normalize(adc_images[index])
            b1000[index] =self.normalize(b1000_images[index])

            if np.max(brain_images[index])==1:
                brain[index]=(brain_images[index]*255).astype(np.uint8)
            else:
                brain[index] = self.normalize(brain_images[index])

            if gt_images is not None:
                gt[index] = gt_images[index].astype(np.uint8)*255
            

        return '', adc, b1000, brain, gt


    def get_bbox(self,mask):

        h,w=mask.shape

        xmin,ymin = w,h
        xmax,ymax = 0,0

        for y in range(h):
            for x in range(w):

                if mask[y,x]>0:
                    xmin = min(xmin,x)
                    ymin = min(ymin,y)
                    xmax = max(xmax,x)
                    ymax = max(ymax,y)

        return xmin,ymin,xmax,ymax

    def masking(self,data,mask):
        data = data.astype(np.float32)
        mask = mask.astype(np.float32)

        data = data * mask/255
        return data.astype(np.uint8)


class Onset_Dataset:
    def __init__(self,root_dir,domain):            

        npy_dir = root_dir + f'{domain}/NPY/'        
        self.npy_paths = glob.glob(npy_dir+'*.npy')

        self.domain = domain
        self.data_dict = json.load(open(root_dir  + 'HeLP_Dataset.json'))
        
        if self.domain == 'validation':
            self.domain = 'train'

    def __len__(self):
        return len(self.npy_paths)
    
    def __getitem__(self,index):
        #1. feature_maps
        #2. oneset onehot
        
        npy_path = self.npy_paths[index]
        feature_maps = np.load(npy_path,allow_pickle=True)
        

        PID = os.path.basename(npy_path).replace('.npy','')
        onset = self.data_dict[self.domain][PID]

        return feature_maps,[onset]


import pickle

class DILD_Dataset:
    def __init__(self,root_dir,domain,additional_domain=''):            

        pic_dir = '{}'.format(domain)     
        self.pic_paths = glob.glob(root_dir + pic_dir+'/*.pickle')
        
        if additional_domain is not '':

            pic_dir2 ='./{}'.format(additional_domain)     
            self.pic_paths += glob.glob(root_dir + pic_dir2+'/*.pickle')

        
    def __len__(self):
        return len(self.pic_paths)
    
    def __getitem__(self,index):
       
        pic_paths = self.pic_paths[index]
        pid = os.path.basename(pic_paths).replace('.pickle','')
                        
        with open(pic_paths,"rb") as fr:
            pic_data = pickle.load(fr)
            
            input_data = pic_data['input']
            labels = pic_data['label']
            
            

        return input_data,labels,pid


class DILD_Dataset_resize:
    def __init__(self,root_dir,domain,transform):            

        pic_dir = './{}'.format(domain)     
        self.pic_paths = glob.glob(root_dir + pic_dir+'/*.pickle')

        self.transform = transform
        print(self.transform)

    def __len__(self):
        return len(self.pic_paths)
    
    def __getitem__(self,index):
        #1. feature_maps
        #2. oneset onehot
        
        pic_paths = self.pic_paths[index]

                
        with open(pic_paths,"rb") as fr:
            pic_data = pickle.load(fr)
        
            input_data = pic_data['input']
            labels = pic_data['label']
            print(input_data.shape)
            print(labels.shape)
            
        # data augmentation
       # if self.transform is not None:
        image,label = self.transform(input_data,labels)
       
        return image, label
       

class DILD_Dataset_lung_mask:
    
    def __init__(self,root_dir,domain,additional_domain=''):            

        pic_dir = './{}'.format(domain)     
        self.pic_paths = glob.glob(root_dir + pic_dir+'/*.pickle')
        
        if additional_domain is not '':

            pic_dir2 ='./{}'.format(additional_domain)     
            self.pic_paths += glob.glob(root_dir + pic_dir2+'/*.pickle')


    def __len__(self):
        return len(self.pic_paths)
    
    def __getitem__(self,index):
        #1. feature_maps
        #2. oneset onehot
        
        pic_paths = self.pic_paths[index]
        pic_lung_paths = self.pic_paths[index].replace('/pickle_norm','/pickle_lung_mask')
        


        with open(pic_lung_paths,"rb") as fr:
            pic_lung_data = pickle.load(fr)

            lung_data = pic_lung_data['lung_mask']

            if len(lung_data.shape)==4:                
                lung_data = lung_data[:,:,:,0]

            lung_data = np.expand_dims(lung_data,axis=1)
            lung_data = (lung_data.astype(np.float32)/255)
           


        with open(pic_paths,"rb") as fr:
            pic_data = pickle.load(fr)
        
            input_data = pic_data['input']
            labels = pic_data['label']

        input_data = torch.from_numpy(input_data)
        lung_data = torch.from_numpy(lung_data)  
        
        lung_data=F.interpolate(lung_data, (input_data.shape[2],input_data.shape[3]),mode='nearest')

        input_data = lung_data*input_data

        return input_data, labels
       
class DILD_CROP_lung_mask:
        
    def __init__(self,root_dir,domain,additional_domain=''):            

        pic_dir = './{}'.format(domain)     
        self.pic_paths = glob.glob(root_dir + pic_dir+'/*.pickle')
        
        if additional_domain is not '':

            pic_dir2 ='./{}'.format(additional_domain)     
            self.pic_paths += glob.glob(root_dir + pic_dir2+'/*.pickle')

    def __len__(self):
        return len(self.pic_paths)
    
    def __getitem__(self,index):
        #1. feature_maps
        #2. oneset onehot
        
        pic_paths = self.pic_paths[index]
        print(pic_paths)
        input()

        pic_lung_paths = self.pic_paths[index].replace('/pickle_norm','/pickle_lung_mask')
        
    

        with open(pic_lung_paths,"rb") as fr:
            pic_lung_data = pickle.load(fr)

            lung_data = pic_lung_data['lung_mask']

            if len(lung_data.shape)==4:                
                lung_data = lung_data[:,:,:,0]

            lung_data = np.expand_dims(lung_data,axis=1)
            lung_data = (lung_data.astype(np.float32)/255)
           
            max_lung_mask = np.max(lung_data[:,0,:,:],axis=0)
            box = self.get_bbox(max_lung_mask)



        with open(pic_paths,"rb") as fr:
            pic_data = pickle.load(fr)
        
            input_data = pic_data['input']

            labels = pic_data['label']

        input_data = torch.from_numpy(input_data)
        lung_data = torch.from_numpy(lung_data)  
        
        lung_data=F.interpolate(lung_data, (input_data.shape[2],input_data.shape[3]),mode='nearest')
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3] 

        lung_data = lung_data[:,:,ymin:ymax,xmin:xmax]
        input_data = input_data[:,:,ymin:ymax,xmin:xmax]
        
        return input_data, labels
       

    def get_bbox(self,mask):

        h,w=mask.shape

        xmin,ymin = w,h
        xmax,ymax = 0,0

        for y in range(h):
            for x in range(w):

                if mask[y,x]>0:
                    xmin = min(xmin,x)
                    ymin = min(ymin,y)
                    xmax = max(xmax,x)
                    ymax = max(ymax,y)

        return xmin,ymin,xmax,ymax 
