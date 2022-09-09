import os
import sys
import random
import math

import numpy as np
from config import parse_arguments
from models.unet import UNet

import torch.nn.functional as F
import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import time
import pathlib
from datetime import datetime

import warnings
from utils_folder.eval_metric import get_metric, get_mertrix
from utils_folder.loss import dice

from datasets import DiseaseDataset_Segmentation
from sklearn.metrics import matthews_corrcoef

fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_thresh = lambda x, thresh :  1.0 * (x > thresh)  

def test(loader, model):
    
    res_dict = {'patient':[], 'logits':[], 'labels' : [] ,'preds' : []}
    
    model.eval()
    correct = 0
    total = 0
    cnt = 0
    patient_id_list = []
    mean_preds = 0
    normal_dice = 0
    ggo_dice = 0
    reticular_opacity_dice = 0
    honey_comb_dice = 0
    emphysema_dice = 0
    consolidation_dice = 0
    
    for iter_, (imgs, mask, labels, patient_id, _,_ ) in tqdm.tqdm(enumerate(iter(loader))):
        
        cnt +=1
        imgs = imgs.cuda()
        mask = mask.cuda()
        labels = labels.cuda()
        patient_id = patient_id[0]
        outputs, pred_labels = model(imgs)
        
        pred_labels = torch.sigmoid(pred_labels)
        _, preds = pred_labels.max(1)
        mean_preds += pred_labels.clone().cpu().detach().numpy()
        total += labels.size(0)
        correct += torch.sum(preds == labels).item()
        
        for i in range(6):
            if i==0:
                normal_dice += dice(outputs[0,i,:,:], 
                                    mask[0,i,:,:]).cpu().detach().numpy()
            elif i==1:
                honey_comb_dice += dice(outputs[0,i,:,:], 
                                        mask[0,i,:,:]).cpu().detach().numpy()
            elif i==2:
                ggo_dice += dice(outputs[0,i,:,:], 
                                 mask[0,i,:,:]).cpu().detach().numpy()
            elif i==3:
                consolidation_dice += dice(outputs[0,i,:,:], 
                                           mask[0,i,:,:]).cpu().detach().numpy()
            elif i==4:
                emphysema_dice += dice(outputs[0,i,:,:], 
                                       mask[0,i,:,:]).cpu().detach().numpy()
            else:
                reticular_opacity_dice += dice(outputs[0,i,:,:], 
                                               mask[0,i,:,:]).cpu().detach().numpy()

        if patient_id_list != patient_id:            
            if not patient_id_list :
                patient_id_list = patient_id
                mean_dice = np.asarray([normal_dice , ggo_dice, reticular_opacity_dice, 
                                        honey_comb_dice, emphysema_dice, consolidation_dice ])
                res_dict['patient'].append(patient_id_list)
                res_dict['preds'].append(np.argmax(np.array(mean_preds[0]) / cnt))
                res_dict['logits'].append(((np.array(mean_preds[0]) / cnt)).tolist())
                res_dict['labels'].append(int(labels.clone().cpu().detach().numpy()[0]))
                res_dict['mean_dice'].append(((mean_dice) / cnt).tolist())                          
                
                mean_preds = 0
                normal_dice = 0
                ggo_dice = 0
                reticular_opacity_dice = 0
                honey_comb_dice = 0
                emphysema_dice = 0
                consolidation_dice = 0
                cnt = 0
            else:
                patient_id_list = patient_id
                mean_dice = np.asarray([normal_dice , ggo_dice, reticular_opacity_dice, 
                                        honey_comb_dice, emphysema_dice, consolidation_dice ])
                res_dict['patient'].append(patient_id_list)
                res_dict['preds'].append(np.argmax(np.array(mean_preds[0]) / cnt))
                res_dict['logits'].append(((np.array(mean_preds[0]) / cnt)).tolist())
                res_dict['labels'].append(int(labels.clone().cpu().detach().numpy()[0]))
                res_dict['mean_dice'].append(((mean_dice) / cnt).tolist())
                
                mean_preds = 0
                normal_dice = 0
                ggo_dice = 0
                reticular_opacity_dice = 0
                honey_comb_dice = 0
                emphysema_dice = 0
                consolidation_dice = 0
                cnt = 0
                

        if iter_ == len(loader)-1:
            return res_dict
                
    return res_dict
def main(args):
    ##### Initial Settings
    warnings.filterwarnings('ignore')
    split_path = args.train_path.split('_')
    args.class_list = [split_path[1], split_path[2]]
    
    log_dir = './logs'
    
    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)
    
    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    # select network
    print('[*] build network... backbone:' , args.backbone)
    if args.backbone == 'unet':
        model = UNet(num_classes=args.num_class)
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')
    
    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    
    model = model.cuda()
    
    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')    
    test_datasets = DiseaseDataset_Segmentation(args.test_path, 'test', 
                                                args.img_size, 
                                                args.naive_lung_mul, 
                                                args)

    test_loader = torch.utils.data.DataLoader(test_datasets, 
                                            batch_size=args.batch_size, 
                                            num_workers=args.w, 
                                            pin_memory=True, 
                                            drop_last=True)

    result_dict = test(test_loader, model)
    
    get_mertrix(result_dict['labels'], 
                result_dict['preds'],
                result_dict['logits'], 
                log_dir, 
                args.class_list)
    
    get_metric(result_dict['labels'], 
               result_dict['preds'], 
               result_dict['logits'], 
               log_dir, 
               args.class_list)
    
    print("MCC:" , matthews_corrcoef(result_dict['labels'], 
                                     result_dict['preds']))

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)