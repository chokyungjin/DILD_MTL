
from pickle import BINBYTES
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MemoryEfficientSwish, SeparableConv2d, BiFPN
from .backbones import efficientnet

class EfficientDet(nn.Module):
    def __init__(self,backbone,num_classes,pretrained=False,norm='bn'):

        super().__init__()
        self.num_classes = num_classes
        self.encoder = efficientnet.build_efficientnet(backbone,pretrained)
        
        if backbone == 'efficientnet-b0':
            self.feature_sizes = [24,40,112,1280]
            self.fpn_cell_repeats = 3
            self.fpn_feature_size = 64

        elif backbone=='efficientnet-b4':
            self.fpn_feature_size = 224
            self.fpn_cell_repeats = 7
            self.feature_sizes=[32,56,160,1792]


        elif backbone=='efficientnet-b7':
            self.fpn_feature_size = 384
            self.fpn_cell_repeats = 8
            self.feature_sizes=[48,80,224,2560]


        self.p2_base_layer = nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[0],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)

            ]
            
        )
        self.p3_base_layer=nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[1],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)
            
            ]
        )
        self.p4_base_layer=nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[2],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)
            
            ]
        )
        self.p5_base_layer=nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[3],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)
            
            ]
        )

        self.p2_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)
        self.p3_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)
        self.p4_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)
        self.p5_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)

        self.bifpn = nn.Sequential(*[BiFPN(self.fpn_feature_size) for _ in range(self.fpn_cell_repeats)])

        self.seg_classifier = SeparableConv2d(self.fpn_feature_size,self.num_classes,norm=None,activation=None)
        self.activation = MemoryEfficientSwish()

        # Classification

    def forward(self, x):

        _,_,h,w = x.size()
        C2, C3, C4, C5 = self.encoder(x)

        P2 = self.p2_base_layer(C2)
        P3 = self.p3_base_layer(C3)
        P4 = self.p4_base_layer(C4)
        P5 = self.p5_base_layer(C5)
              
        P2 = self.p2_layer(P2)
        P3 = self.p3_layer(P3)
        P4 = self.p4_layer(P4)
        P5 = self.p5_layer(P5)

        # c5-> classification
        # c2,3,4,5-> segmentation

        outputs = self.bifpn([P2,P3,P4,P5])
             
        P2 =self.activation(outputs[0])         
        seg_logits =  self.seg_classifier(P2) 
        seg_logits =F.interpolate(seg_logits,(h,w),mode='bilinear',align_corners=False) 
       
        return seg_logits




class EfficientDet_MultiTask(nn.Module):
    def __init__(self,backbone,num_classes,pretrained=False,norm='bn'):

        super().__init__()
        self.num_classes = num_classes

        self.encoder = efficientnet.build_efficientnet(backbone,pretrained)
        
        if backbone == 'efficientnet-b0':
            self.feature_sizes = [24,40,112,1280]
            self.fpn_cell_repeats = 3
            self.fpn_feature_size = 64

        elif backbone=='efficientnet-b4':
            self.fpn_feature_size = 224
            self.fpn_cell_repeats = 7
            self.feature_sizes=[32,56,160,1792]


        elif backbone=='efficientnet-b7':
            self.fpn_feature_size = 384
            self.fpn_cell_repeats = 8
            self.feature_sizes=[48,80,224,2560]


        self.p2_base_layer = nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[0],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)

            ]
            
        )
        self.p3_base_layer=nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[1],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)
            
            ]
        )
        self.p4_base_layer=nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[2],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)
            
            ]
        )
        self.p5_base_layer=nn.Sequential(
            *[
                SeparableConv2d(self.feature_sizes[3],self.fpn_feature_size,norm),
                SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm)
            
            ]
        )

        self.p2_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)
        self.p3_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)
        self.p4_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)
        self.p5_layer = SeparableConv2d(self.fpn_feature_size,self.fpn_feature_size,norm,activation=None)

        self.bifpn = nn.Sequential(*[BiFPN(self.fpn_feature_size) for _ in range(self.fpn_cell_repeats)])

        self.seg_classifier = SeparableConv2d(self.fpn_feature_size,self.num_classes,norm=None,activation=None)
        self.activation = MemoryEfficientSwish()


        # Classification
        self.classifier = SeparableConv2d(self.feature_sizes[-1],self.num_classes,norm=None,activation=None)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):

        _,_,h,w = x.size()
        C2, C3, C4, C5 = self.encoder(x)

        P2 = self.p2_base_layer(C2)
        P3 = self.p3_base_layer(C3)
        P4 = self.p4_base_layer(C4)
        P5 = self.p5_base_layer(C5)
              
        P2 = self.p2_layer(P2)
        P3 = self.p3_layer(P3)
        P4 = self.p4_layer(P4)
        P5 = self.p5_layer(P5)

        # c5-> classification
        # c2,3,4,5-> segmentation

        outputs = self.bifpn([P2,P3,P4,P5])
                  
        P2 =self.activation(outputs[0])         
        seg_logits =  self.seg_classifier(P2) 
        seg_logits =F.interpolate(seg_logits,(h,w),mode='bilinear',align_corners=False) 
       
        
        f=self.classifier(C5)
        cls_logits = self.gap(f).view(f.size(0),self.num_classes)


        return cls_logits,seg_logits,f
  
    def features(self,x):

        _,_,_, C5 = self.encoder(x)
        
        return C5

class Onset_Classifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv3d(1280,640,(3,3,3),1,padding=True)
        self.gn1 = nn.GroupNorm(4,640)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(640,640,(3,3,3),1,padding=True)
        self.gn2 = nn.GroupNorm(4, 640)
        self.act2 = nn.ReLU()

        self.pool1 = nn.MaxPool3d((2,2,2),2)

        self.conv3 = nn.Conv3d(640,320,(3,3,3),1,padding=True)
        self.gn3 = nn.GroupNorm(4, 320)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv3d(320,320,(3,3,3),1,padding=True)
        self.gn4 = nn.GroupNorm(4, 320)
        self.act4 = nn.ReLU()
        
        self.classifier = nn.Conv3d(320,num_classes,(3,3,3),1,padding=True)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self,x): # 연결
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)
        
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = self.act4(x)

        f = self.classifier(x)
       
        logits = self.gap(f)[:,:,0,0,0] # batch, class
        return logits


class DILD_Classifier(nn.Module): #vgg
    def __init__(self,num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv3d(6,128,(3,3,3),1,padding=True)
        self.gn1 = nn.GroupNorm(4,128)
        # self.bn1 = nn.BatchNorm3d(128) # output 
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(128,128,(3,3,3),1,padding=True)
        self.gn2 = nn.GroupNorm(4, 128)
        # self.bn2 = nn.BatchNorm3d(128)
        self.act2 = nn.ReLU()

        self.pool1 = nn.MaxPool3d((2,2,2),2)

        self.conv3 = nn.Conv3d(128,256,(3,3,3),1,padding=True)
        self.gn3 = nn.GroupNorm(4, 256)
        # self.bn3 = nn.BatchNorm3d(256)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv3d(256,256,(3,3,3),1,padding=True)
        self.gn4 = nn.GroupNorm(4, 256)
        # self.bn4 = nn.BatchNorm3d(256)
        self.act4 = nn.ReLU()
        
        
        self.conv5 = nn.Conv3d(256,512,(3,3,3),1,padding=True)
        self.gn5 = nn.GroupNorm(4, 512)
        # self.bn5 = nn.BatchNorm3d(512)
        self.act5 = nn.ReLU()

        self.conv6 = nn.Conv3d(512,512,(3,3,3),1,padding=True)
        self.gn6 = nn.GroupNorm(4, 512)
        # self.bn6 = nn.BatchNorm3d(512)
        self.act6 = nn.ReLU()

        self.pool2 = nn.MaxPool3d((2,2,2),2)

        self.classifier = nn.Conv3d(512,num_classes,(1,1,1),1,padding=False)
        self.gap = nn.AdaptiveAvgPool3d((1,1,1)) 

    def forward(self,x): 

        x=F.interpolate(x,scale_factor=[1.0, 0.5, 0.5],  mode='nearest')

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)
        
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.gn4(x)
        x = self.act4(x)

        x = self.pool2(x)

        x = self.conv5(x)
        x = self.gn5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.gn6(x)
        x = self.act6(x)

        f = self.classifier(x)
       
        logits = self.gap(f)[:,:,0,0,0] # batch, class
        return logits


