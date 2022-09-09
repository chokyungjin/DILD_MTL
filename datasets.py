import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import pydicom
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
    
class DiseaseDataset_Segmentation(Dataset):
    def __init__(self, input_path, mode, image_size, naive_lung_mul , args):
        self.mode = mode
        self.args = args
        self.naive_lung_mul = naive_lung_mul
        self.image_size = image_size
        
        with open(input_path, "r") as f:
            self.samples = json.load(f)
        
        if mode == 'train':
            if args.aug == 'True':
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    A.ShiftScaleRotate(shift_limit=0.0625, 
                                       scale_limit=0.2, 
                                       rotate_limit=10, p=0.2),

                    ToTensorV2(transpose_mask=True),
                    ])
            else:
                self.transform = A.Compose([
                    A.Resize(self.image_size, self.image_size),
                    ToTensorV2(transpose_mask=True),
                    ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                ToTensorV2(transpose_mask=True),
                ])
        

    def __getitem__(self, idx):
        
        imgs = self.preprocessing_dcm(self.samples['imgs'][idx] , 
                                        self.samples['lung_mask'][idx])
        
        masks = self.preprocessing_mask(self.samples['mask'][idx])
        
        transformed = self.transform(image=imgs , mask=masks)
        
        imgs = transformed['image']
        masks = transformed['mask']
        labels = self.samples['labels'][idx]

        labels = [0]*6
        for i in range(6):
            if torch.sum(masks[:,:,i]) != 0:
                labels[i] = 1
                
        lung_mask = np.load(self.samples['lung_mask'][idx])
        lung_slice = False
        if np.sum(lung_mask) != 0:
            lung_slice = True
        else:
            lung_slice = False
            
        name = self.samples['imgs'][idx].split('/')[-1].split('.dcm')[0]
        if name.split('_')[0] == 'NSIP':
            patient_id = str(name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2] + '_' + name.split('_')[3])
        else:
            patient_id = str(name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2])
        return imgs, masks , torch.Tensor(np.array(labels)) , patient_id, lung_slice , self.samples['imgs'][idx]
            
    def __len__(self):
        return len(self.samples['labels'])
    
    def preprocessing_dcm(self, path, lung_path):

        dcm = pydicom.dcmread(path, force=True)
        try:
            img_out = dcm.pixel_array.astype('float32')
        except:
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            img_out = dcm.pixel_array.astype('float32')
            
        intercept = dcm[0x28,0x1052].value
        slope = dcm[0x28,0x1053].value
        img_out = slope*img_out + intercept
        
        if self.naive_lung_mul:
            lung_path = lung_path.replace('/mask/' , '/seg_mask/')
            lung_mask = np.load(lung_path) * 255.0
            if lung_mask.shape[-1] == 3:
                lung_mask = lung_mask[:,:,0]

        img_out[img_out < -1024] = -1024
        img_out[img_out > 3072] = 3072
        
        center = -400
        width = 1500
        low = center - width // 2
        high = center + width // 2
        img_out = (img_out-low) / (high-low)
        img_out[img_out < 0.] = 0
        img_out[img_out > 1.] = 1

        if self.naive_lung_mul:
            img_out[lung_mask == 0] = 0

        return img_out
    
    def preprocessing_mask(self,path):
        
        mask_dummy = np.load(path)[:,:,0]
        mask_out = np.zeros((512, 512, 6))
        mask_out[:,:,0][mask_dummy == 101] = 1. # normal
        mask_out[:,:,1][mask_dummy == 102] = 1. # honey comb 
        mask_out[:,:,2][mask_dummy == 103] = 1. # GGO 
        mask_out[:,:,3][mask_dummy == 104] = 1. # consolidation
        mask_out[:,:,4][mask_dummy == 105] = 1. # emphysema
        mask_out[:,:,5][mask_dummy == 106] = 1. # reticular opacity
        return mask_out
                
