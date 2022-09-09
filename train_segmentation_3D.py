import os
import sys
sys.path.append('./MONAI')
from monai.networks.nets import *
from monai.transforms import *

import random
import numpy as np
from config import parse_arguments
from datasets_3D import DiseaseDataset_Segmentation

from utils_folder.utils import AverageMeter, ProgressMeter, save_model, calculate_parameters, pad_collate_fn
from utils_folder.loss import DiceLoss, FocalLoss
from evaluation import evaluate

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

import time
import pathlib
from datetime import datetime
import warnings
    
def train(args, epoch, loader, val_loader, model, device, optimizer, writer ,scheduler):
    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    cls_losses = AverageMeter('Cls Loss', ':.4f')
    dice_coff = AverageMeter('Dice Coff', ':.4f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses , dice_coff],
        prefix='Epoch: [{}]'.format(epoch))

    running_loss = 0
    dice_arr = 0
    correct = 0
    total = 0
    end = time.time()

    for iter_, (imgs, mask, labels,_ ) in enumerate(iter(loader)):
        
        # imgs, masks , labels , lung_mask
        imgs = imgs.to(device)
        mask = mask.to(device)
        labels = labels.to(device, dtype=torch.long)
        
        outputs, pred_labels = model(imgs)
        
        _, preds = pred_labels.max(1)
        total += labels.size(0)
        correct += torch.sum(preds == labels).item()

        if args.per_class_dice_loss == True:
            seg_criterion = DiceLoss()
            for b in range(imgs.shape[0]):
                for i in range(6):
                    if i==0:
                        normal_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                    elif i==1:
                        ggo_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                    elif i==2:
                        reticular_opacity_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                    elif i==3:
                        honey_comb_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                    elif i==4:
                        emphysema_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                    else:
                        consolidation_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
            
            seg_loss = normal_dice + ggo_dice + reticular_opacity_dice + honey_comb_dice + \
                emphysema_dice + consolidation_dice

        elif args.seg_loss == 'dice':
            seg_criterion = DiceLoss()
            seg_loss = seg_criterion(outputs, mask)
        else:
            seg_criterion = nn.BCEWithLogitsLoss()

        if args.focal == True:
            cls_criterion = FocalLoss()
        else:
            cls_criterion = nn.CrossEntropyLoss()

        seg_loss = seg_criterion(outputs, mask)
        cls_loss = cls_criterion(pred_labels, labels)
        train_dice = dice(outputs, mask)

        overall_loss = args.lambda_seg * seg_loss + args.lambda_cls * cls_loss
        
        losses.update(overall_loss.item(), imgs[0].size(0))
        dice_coff.update(train_dice.item(), imgs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()
        
        running_loss += overall_loss.item()
        dice_arr += train_dice.item()

        if (iter_ % args.print_freq == 0)& (iter_ != 0):
            progress.display(iter_)
            writer.add_scalar('overall_train_loss', running_loss/iter_, (epoch*len(loader))+iter_)
            writer.add_scalar('train_dice', dice_arr/iter_, (epoch*len(loader))+iter_)
            writer.add_scalar('train_acc', 100.*correct/total, (epoch*len(loader))+iter_)
            
    
    val_batch_time = AverageMeter('Time', ':6.3f')
    val_losses = AverageMeter('Loss', ':.4f')
    val_dice_coff = AverageMeter('Dice Coff ', ':.4f')

    progress = ProgressMeter(
        len(val_loader),
        [val_batch_time, val_losses,val_dice_coff],
        prefix='Epoch: [{}]'.format(epoch))
    
    end = time.time()
    val_dice_arr = 0
    val_running_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        model.eval()
        print('[*] Validation Phase' )
        for iter_, (imgs, mask, labels,_ ) in enumerate(iter(val_loader)):
            imgs = imgs.to(device)
            mask = mask.to(device)
            labels = labels.to(device, dtype=torch.long)

            outputs, pred_labels = model(imgs)
            _, preds = pred_labels.max(1)
            
            val_total += labels.size(0)
            val_correct += torch.sum(preds == labels).item()

            if args.per_class_dice_loss == True:
                seg_criterion = DiceLoss()
                for b in range(imgs.shape[0]):
                    for i in range(6):
                        if i==0:
                            normal_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                        elif i==1:
                            ggo_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                        elif i==2:
                            reticular_opacity_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                        elif i==3:
                            honey_comb_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                        elif i==4:
                            emphysema_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
                        else:
                            consolidation_dice = seg_criterion(outputs[b,i,:,:], mask[b,i,:,:])
            
                seg_loss = normal_dice + ggo_dice + reticular_opacity_dice + honey_comb_dice + \
                    emphysema_dice + consolidation_dice

            elif args.seg_loss == 'dice':
                seg_criterion = DiceLoss()
                seg_loss = seg_criterion(outputs, mask)
            else:
                seg_criterion = nn.BCEWithLogitsLoss()
                seg_loss = seg_criterion(outputs, mask)

            if args.focal == True:
                cls_criterion = FocalLoss()
            else:
                cls_criterion = nn.CrossEntropyLoss()

            cls_loss = cls_criterion(pred_labels, labels)
            val_dice = dice(outputs, mask)

            val_overall_loss = args.lambda_seg * seg_loss + args.lambda_cls * cls_loss
            
            
            val_losses.update(val_overall_loss.item(), imgs[0].size(0))
            val_dice_coff.update(val_dice.item(), imgs[0].size(0))
            
            val_batch_time.update(time.time() - end)
            end = time.time()
            
            val_running_loss += val_overall_loss.item()
            val_dice_arr += val_dice.item()

            if (iter_ % args.print_freq == 0)& (iter_ != 0):
                progress.display(iter_)
                writer.add_scalar('overall_val_loss', val_running_loss/iter_, (epoch*len(val_loader))+iter_)
                writer.add_scalar('val_dice', val_dice_arr/iter_, (epoch*len(val_loader))+iter_)
                writer.add_scalar('val_acc', 100.*val_correct/val_total, (epoch*len(val_loader))+iter_)
                
        scheduler.step(np.mean(val_running_loss))
    model.train()

def test(args, epoch, loader, model, device, writer):
    print('[*] Test Phase')
    
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for iter_, (imgs, mask, labels,_ ) in enumerate(iter(loader)):
            imgs = imgs.to(device)
            mask = mask.to(device)
            labels = labels.to(device, dtype=torch.long)

            _, pred_labels = model(imgs)
            pred_labels = F.softmax(pred_labels)
            _, preds = pred_labels.max(1)

            total += labels.size(0)
            correct += torch.sum(preds == labels).item()

    test_acc = 100.*correct/total
    print('[*] Test Acc: {:5f}'.format(test_acc))
    writer.add_scalar('Test acc', test_acc, iter_)
    model.train()
    return test_acc

def test_dice(args, epoch, loader, model, writer):
    print('[*] Test Phase')
    
    test_dice_arr = 0

    with torch.no_grad():
        model.eval()
        for iter_, (imgs, mask, labels ) in enumerate(iter(loader)):

            imgs = imgs.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs)
            test_dice = dice(outputs, mask)

            test_dice_arr += test_dice.item()

    print('[*] Test Dice: {:5f}'.format(np.mean(test_dice_arr)))
    writer.add_scalar('Test Dice', np.mean(test_dice_arr), iter_)
    model.train()
    return np.mean(test_dice_arr)


def main(args):
    ##### Initial Settings
    warnings.filterwarnings('ignore')

    split_path = args.train_path.split('_')
    args.class_list = [split_path[1], split_path[2]]
    print('[*] using {} bit images'.format(args.bit))

    # device check & pararrel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)

    # path setting
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M%S'))
    folder_name = '{}_{}'.format(today, args.message)
    
    args.log_dir = os.path.join(args.log_dir, folder_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # for log
    f = open(os.path.join(args.log_dir,'arguments.txt'), 'w')
    f.write(str(args))
    f.close()
    print('[*] log directory: {} '.format(args.log_dir))
    print('[*] checkpoint directory: {} '.format(args.checkpoint_dir))
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # select network
    print('[*] build network... backbone: {}'.format(args.backbone))
    if args.backbone == 'unet':
        model = BasicUNet_Shallow_Aux(dimensions=3, 
                in_channels = 1,
                num_classes = args.num_class,
                out_channels= args.out_channels,
                features=(32, 32, 64, 128, 256, 32))
    else:
        ValueError('Have to set the backbone network in [resnet, vgg, densenet]')

    print(('[i] Total Params: %.2fM'%(calculate_parameters(model))))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    writer = SummaryWriter(args.log_dir)    

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=0.9,
                                weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                         mode='min',
                                         factor=0.5,
                                         patience=10, )

    if args.resume is True:
        checkpoint = torch.load(args.pretrained)
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    
    ##### Dataset & Dataloader
    print('[*] prepare datasets & dataloader...')
    train_datasets = DiseaseDataset_Segmentation(args.train_path, 'train', 
                                                 args.img_size, 
                                                 args.aug , args)
    val_datasets = DiseaseDataset_Segmentation(args.val_path, 'val', 
                                               args.img_size, 
                                               args.aug , args)
    test_datasets = DiseaseDataset_Segmentation(args.test_path, 'test', 
                                                args.img_size, 
                                                args.aug , args)

    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, 
                                num_workers=args.w, pin_memory=True, 
                                shuffle=True, drop_last=True,
                                collate_fn=pad_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=args.batch_size, 
                                num_workers=args.w, pin_memory=True, drop_last=True,
                                collate_fn=pad_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=args.batch_size, 
                                num_workers=args.w, 
                                pin_memory=True, drop_last=True,
                                collate_fn=pad_collate_fn)
    
    ##### Train & Test
    print('[*] start a train & test loop')
    best_model_path = os.path.join(args.checkpoint_dir,'best.pth.tar')

    for epoch in range(args.start_epoch, args.epochs):
        train(args, epoch, train_loader, val_loader, model, device, optimizer, writer ,scheduler)
        acc = test(args, epoch, test_loader, model, device, writer )
        
        save_name = '{}.pth.tar'.format(epoch)
        save_name = os.path.join(args.checkpoint_dir, save_name)
        save_model(save_name, epoch, model, optimizer ,scheduler)

        if epoch == 0:
            best_acc = acc
            save_model(best_model_path, epoch, model, optimizer , scheduler)

        else:
            if best_acc < acc:
                best_acc = acc
                save_model(best_model_path, epoch, model, optimizer , scheduler)

    ##### Evaluation (with best model)
    print("[*] Best Model's Results")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)