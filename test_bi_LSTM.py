import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import json
import argparse
import numpy as np

from core import datasets
from core import networks
from core import losses
from tools import eval_utils
from tools import augment_utils
from tools import io_utils

def collate(batch):
    features = []
    labels = []
    PIDs = []
    for feature, label,pid in batch:        
        
        feature = feature.transpose(1,0,2,3)
        label = np.asarray(label,dtype=np.float32)
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).float()
        features.append(feature)
        labels.append(label)
        PIDs.append(pid)

    return torch.stack(features), torch.stack(labels), PIDs


parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--root_dir',default='F:/DILD/external/external_3D/',type=str)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--tag',default='lstm_384_oversampling_cop',type=str)
parser.add_argument('--amp', default=False, type=io_utils.boolean)
parser.add_argument('--domain', default='external', type=str)


if __name__ == '__main__':
    args = parser.parse_args()

    # 1. make folders   
    model_dir = io_utils.create_directory('./experiments/models/')
    tensorboard_dir = io_utils.create_directory('./experiments/tensorboards/{}/'.format(args.tag))
    model_path = model_dir + args.tag +'.pth'

    # 2. transform, dataset, dataloader
    test_dataset = datasets.DILD_Dataset(args.root_dir, args.domain)
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False, collate_fn=collate
    )
        
    
    # 3. build networks
    model = networks.DILD_Classifier(3)
    weights = model.state_dict()
    model_path = './ex.pth'
    torch.save(weights, model_path)

    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    model.cpu()

    # multi-gpus
    try:
        gpu_option = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        gpu_option = '0'

    num_gpus = len(gpu_option.split(','))
    if num_gpus > 1:
        print('# number of gpus : {}'.format(num_gpus))
        model = nn.DataParallel(model)

    eval_dict = {}
       

    test_length = len(test_loader)
    all_th_valid_acc=[]

    evaluator = eval_utils.Single_Evaluator_2(3)
     
    pred_dict = {}
    for i, (features, labels, PIDs )in enumerate(test_loader):
        sys.stdout.write('\r[{}/{}]'.format(i+1,test_length))
        sys.stdout.flush()


        images = features.cpu().float()
        labels = labels.cpu().float()
        
        with torch.no_grad():               

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits= model(features)

        pred_cls = torch.softmax(logits,dim=1)[0].cpu().detach().numpy()
        PID = PIDs[0] 
        pred_class = []  
        for c in pred_cls:
            pred_class.append(float(c))

        
        pred_dict[PID]={}
        pred_dict[PID]['prediction']=list(pred_class)        
        pred_dict[PID]['gt']=float(labels[0].item())


    F = open('pred_cls_internal_test_no_oversample.json','w')
    json.dump(pred_dict,F,indent='\t')
    F.close()
