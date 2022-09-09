import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from config import parse_arguments
import json
import volumentations as Vol
from volumentations import *

## 6 channel dataset Segmentation Aug success
class DiseaseDataset_Segmentation(Dataset):
    def __init__(self, input_path, mode, image_size, aug, transform=None):
        self.mode = mode # Unused variable. However, it will be used for transform
        self.image_size = image_size
        self.aug = aug
        with open(input_path, "r") as f:
            self.samples = json.load(f)
        
    def __getitem__(self, idx):

        image = np.load(self.samples['imgs'][idx])
        image = np.transpose(image,(1,2,0))
        mask  = np.load(self.samples['mask'][idx])
        mask = np.transpose(mask,(3,1,2,0))

        image = np.expand_dims(image, axis=0)
        img_patch_size=(image.shape[0] , self.image_size, self.image_size ,  image.shape[-1])
        mask_patch_size=(mask.shape[0] , self.image_size, self.image_size ,  mask.shape[-1])
        
        result_image = Vol.Compose([
                    Resize(img_patch_size, interpolation=1, always_apply=True, p=1.0),
        ])(image=image)
        result_mask = Vol.Compose([
                    Resize(mask_patch_size, interpolation=1, always_apply=True, p=1.0),
        ])(mask=mask)
        
        imgs  = result_image['image']
        masks  = result_mask['mask']
        
        if self.mode == 'train':
            if self.aug == 'True':
                result = Vol.Compose([
                    Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
                    Flip(0, p=0.5),
                    GaussianNoise(var_limit=(0, 5), p=0.2),
                    RandomGamma(gamma_limit=(0.5, 1.5), p=0.2),
                ])(image=imgs,mask=masks)
            else:
                result = Vol.Compose([
                ])(image=imgs,mask=masks)
        else:
            result = Vol.Compose([                
            ])(image=imgs,mask=masks)

        imgs = result['image']
        masks  = result['mask']
        
        labels = self.samples['labels'][idx]
        name = self.samples['imgs'][idx].split('/')[-1].split('.npy')[0]
        
        if name.split('_')[0] == 'NSIP':
            patient_id = str(name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2] + '_' + name.split('_')[3])
        else:
            patient_id = str(name.split('_')[0] + '_' + name.split('_')[1] + '_' + name.split('_')[2])
        
        return imgs , masks, labels , patient_id
            
    def __len__(self):
        return len(self.samples['labels'])

# For test
if __name__ == '__main__':
    dataset = DiseaseDataset_Segmentation('.json', 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1)

    for imgs, labels in iter(dataloader):
        print(imgs.shape, labels.shape)