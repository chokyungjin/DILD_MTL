import os
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np


from core import datasets
from core import networks
from core import losses
from tools import eval_utils
from tools import augment_utils
from tools import io_utils
from core import convLSTM

def collate(batch):
    features = []
    labels = []

    for feature, label in batch:        
        
        feature = torch.from_numpy(feature).float()
        label = np.asarray(label,dtype=np.float32)
        label = torch.from_numpy(label).float()
        features.append(feature)
        labels.append(label)

    return torch.stack(features), torch.stack(labels)


parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--root_dir',default='F:/DILD/211227_oversampling/pickle_bridge/',type=str)

parser.add_argument('--additional_domain', default='train_cop',type=str)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--max_epochs', default=300, type=int)
parser.add_argument('--class_loss', default='binary-cross-entropy', type=str)

parser.add_argument('--optim', default='adamw', type=str)
parser.add_argument('--lr',default=1e-4,type=float)
parser.add_argument('--wd',default=4e-5,type=float)
 
parser.add_argument('--tag',default='CNN_LSTM_NEW@Oversample',type=str)

parser.add_argument('--layer',default=3,type=int)
parser.add_argument('--amp', default=False, type=io_utils.boolean)

if __name__ == '__main__':
    args = parser.parse_args()

    # 1. make folders   
    model_dir = io_utils.create_directory('./experiments/models/')
    tensorboard_dir = io_utils.create_directory('./experiments/tensorboards/{}/'.format(args.tag))
    model_path = model_dir + args.tag +'.pth'

    # 2. transform, dataset, dataloader

    train_dataset = datasets.DILD_Dataset(args.root_dir,'train', args.additional_domain)    
    valid_dataset = datasets.DILD_Dataset(args.root_dir, 'valid')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=True, collate_fn=collate)
    


    print('train_loader')
    print(len(train_loader))
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers // 4,
        shuffle=False, drop_last=False,collate_fn=collate)


    print('val_loader')
    print(len(valid_loader))

    # 3. build networks
    
    model = networks.DILD_LSTM(3,args.layer)
   

    print('model')
    model.cuda()
    model.train()

    # multi-gpus
    try:
        # 0,1
        gpu_option = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        gpu_option = '0'

    num_gpus = len(gpu_option.split(','))
    if num_gpus > 1:
        print('# number of gpus : {}'.format(num_gpus))
        model = nn.DataParallel(model)


    # 4. build losses

     # sigmoid base
    class_loss_fn = nn.CrossEntropyLoss().cuda()
   
    # 5. build optimizer and scheduler
    if args.optim =='sgd':            
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr, weight_decay=args.wd, 
            momentum=0.9, nesterov=True
        )
        
    elif args.optim =='adamw':        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, weight_decay=args.wd, 
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.max_epochs * 0.5), int(args.max_epochs * 0.75)], gamma=0.1)

    # 6. build tensorboard
    writer = SummaryWriter(tensorboard_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    length = len(train_loader)
    valid_length = len(valid_loader)

    best_acc = 0.0
    evaluator = eval_utils.Evaluator_multi(3)
    
    for epoch in range(args.max_epochs):
        optimizer.zero_grad()
        loss_list = []
        
        for iter, (features,labels) in enumerate(train_loader):
            sys.stdout.write('\r# Epoch = {} [{}/{}]'.format(epoch + 1, iter + 1, length))
            sys.stdout.flush()

            features = features.cuda().float()
            labels = labels.cuda().float()
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(features)            
          
            loss = class_loss_fn(logits, labels.long())
            optimizer.zero_grad()
           
            loss.backward() # zero grad -> step update // zero grad 5 / 1 update
            optimizer.step()
            
            loss_list.append(loss.item())
    
        print()
        
        scheduler.step()

        loss = np.mean(loss_list)

        print('# Epoch = {}, loss = {:.4f}'.format(epoch + 1, loss))
        writer.add_scalar('Training/Loss', loss, epoch)
        
        model.eval()
        for i, (features, labels )in enumerate(valid_loader):
            sys.stdout.write('\r[{}/{}]'.format(i+1,valid_length))
            sys.stdout.flush()

            images = features.cuda().float() 
            labels = labels.cuda().long() #Long 

            with torch.no_grad():
                    
                with torch.cuda.amp.autocast(enabled=args.amp):                    
                    seg_logits= model(images)
               
            pred_cls = torch.softmax(seg_logits,dim=1) #soft

            for i in range(pred_cls.size(0)):
                label = labels[i].cpu().detach().numpy()
                pred_class = pred_cls[i].cpu().detach().numpy()
                
                evaluator.add(pred_class,label) # acc
       
        model.train()

        print()    
        valid_acc = evaluator.get()

        valid_acc = np.mean(valid_acc)

                
        if num_gpus > 1:
            weights = model.module.state_dict()
        else:
            weights = model.state_dict()
            

        if valid_acc >= best_acc:
            best_acc = valid_acc
            torch.save(weights, model_path)
            
       
        print('# Epoch = {},  valid_acc = {:.2f}%, best_valid_acc={:.2f}%'.format(epoch + 1, valid_acc,best_acc))


        writer.add_scalar('Evaluation/acc', valid_acc, epoch)
        writer.add_scalar('Evaluation/Best_acc', best_acc, epoch)