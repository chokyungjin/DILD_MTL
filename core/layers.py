
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels=None,norm='bn',activation='swish'):
        
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        depthwise_dict = {
            "in_channels" : in_channels,
            "out_channels" : in_channels,
            "kernel_size" : 3,
            "padding" : 1,
            "stride" :1,
            "groups":in_channels,
            "bias":False

        }

        self.depthwise_conv=nn.Conv2d(**depthwise_dict)
        pointwise_dict = {
            "in_channels":in_channels,
            "out_channels":out_channels,
            "kernel_size" :1,
            "stride":1
        }
        self.pointwise_conv = nn.Conv2d(**pointwise_dict)

        if norm == 'bn':
            self.norm =nn.BatchNorm2d(out_channels,momentum=0.01,eps=1e-3)
        elif norm =='gn':
            self.norm=nn.GroupNorm(4,out_channels,eps=1e-3)
        else:
            self.norm = None


        if activation == 'swish':
            self.activation = MemoryEfficientSwish()
        elif activation=='relu':
            self.activation=nn.ReLU()
        else:
            self.activation = None

    def forward(self,x):
        x=self.depthwise_conv(x)
        x=self.pointwise_conv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class BiFPN(nn.Module):
    def __init__(self,fpn_feature_size, norm='bn',activation='swish',eps=1e-3,mode='nearest'):
        super().__init__()

        self.eps =eps

        if activation == 'swish': # model ??????????????? sigmoid ?????? ?????? ?????? ??????????????? ???????????? 
            self.activation = MemoryEfficientSwish()
        elif activation =='relu':
            self.activation = nn.ReLU()
            
        self.p5_up = nn.Upsample(scale_factor=2,mode=mode)
        self.p4_up = nn.Upsample(scale_factor=2,mode=mode)
        self.p3_up = nn.Upsample(scale_factor=2,mode=mode)

        self.p3_down = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # 1/2
        self.p4_down = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # 1/2
        self.p5_down = nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # 1/2

        self.p4_1_conv = SeparableConv2d(fpn_feature_size,norm=norm,activation=None)
        self.p3_1_conv = SeparableConv2d(fpn_feature_size,norm=norm,activation=None)
        
        self.p2_2_conv = SeparableConv2d(fpn_feature_size,norm=norm,activation=None)        
        self.p3_2_conv = SeparableConv2d(fpn_feature_size,norm=norm,activation=None)
        self.p4_2_conv = SeparableConv2d(fpn_feature_size,norm=norm,activation=None)
        self.p5_2_conv = SeparableConv2d(fpn_feature_size,norm=norm,activation=None)
        
        self.p4_1_weights = nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)
        self.p3_1_weights = nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)
        
        self.p2_2_weights = nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)       
        self.p3_2_weights = nn.Parameter(torch.ones(3,dtype=torch.float32),requires_grad=True)
        self.p4_2_weights = nn.Parameter(torch.ones(3,dtype=torch.float32),requires_grad=True)
        self.p5_2_weights = nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)

    def get_weights(self, weights):
        weights = F.relu(weights)
        weights = weights/(torch.sum(weights, dim = 0) + self.eps)
        return weights

    def forward(self, inputs):

        """
        illustration of a minimal bifpn units
            P5_0 -------------------------> P5_2 -------->
               |-------------|                ???
                             ???                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------??? ???
                             ???                |
            P3_0 ---------> P3_1 ---------> P3_2 -------->
               |-------------|--------------??? ???
                             |--------------??? |
            P2_0 -------------------------> P2_2 -------->
        """
        # P2_0 = torch.Size([1, 64, 256, 256])
        # P3_0 = torch.Size([1, 64, 128, 128])
        # P4_0 = torch.Size([1, 64, 64, 64])
        # P5_0 = torch.Size([1, 64, 32, 32])


        P2_0,P3_0,P4_0,P5_0 = inputs
        
        
        # ?????? = ?????? ??????, ???????????? ????????????, weight??? ????????? ????????? ??? ?????? ?????????
        # weights ?????? ????????? ?????? 1??? ??????
        # relu?????????, ????????? ????????? ??????
        # sum ??? ????????? - normalize
        # ??? ????????? ?????? ???????????? ?????????
        # 0??? ????????? ??????, 1??? ????????? ??????
        # ????????? ?????? ????????? - ?????? ?????? ?????? ????????? ???

        weights = self.get_weights(self.p4_1_weights)
        P4_1 =  self.p4_1_conv(self.activation(weights[0]*P4_0+weights[1]*self.p5_up(P5_0)))

        weights = self.get_weights(self.p3_1_weights)
        P3_1 = self.p3_1_conv(self.activation(weights[0]*P3_0+weights[1]*self.p4_up(P4_1)))

        weights = self.get_weights(self.p2_2_weights)
        P2_2 = self.p2_2_conv(self.activation(weights[0]*P2_0+weights[1]*self.p3_up(P3_1)))

        weights = self.get_weights(self.p3_2_weights)
        P3_2 = self.p3_2_conv(self.activation(weights[0]*P3_0+weights[1]* P3_1+weights[2]*self.p3_down(P2_2)))

        weights = self.get_weights(self.p4_2_weights)
        P4_2 = self.p4_2_conv(self.activation(weights[0]*P4_0+weights[1]*P4_1+weights[2]*self.p4_down(P3_2)))

        weights = self.get_weights(self.p5_2_weights)
        P5_2 = self.p5_2_conv(self.activation(weights[0] * P5_0 + weights[1] * self.p5_down(P4_2)))                
        
        return P2_2, P3_2,P4_2,P5_2