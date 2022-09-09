import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import *

class UNet(nn.Module):
    def __init__(self ,num_classes=None):
        super(UNet, self).__init__()
        self.num_classes = num_classes
    
        def CBR2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels = input_channels, 
                                 out_channels = output_channels, 
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding=padding, bias = bias)]
            layers += [nn.BatchNorm2d(num_features=output_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr
        
        def start_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
        
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=output_channels)]    
            
            start = nn.Sequential(*layers)
            return start
        
        def encoder_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            
            layers += [nn.MaxPool2d(kernel_size=kernel_size)]
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=output_channels)]
            
            
            enc = nn.Sequential(*layers)
            return enc
        
        def decoder_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
                        
           
            layers += [CBR2d(input_channels=input_channels*2, output_channels=output_channels)]
            layers += [CBR2d(input_channels=input_channels, output_channels=int(output_channels/2))]
            layers += [nn.ConvTranspose2d(in_channels=int(input_channels/2), out_channels=int(output_channels/2), 
                                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            
            dec = nn.Sequential(*layers)
            return dec
        
        def bridge(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            layers += [nn.MaxPool2d(kernel_size=2)]
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=input_channels)]

            bridge = nn.Sequential(*layers)
            return bridge
        
        def end_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            layers += [CBR2d(input_channels=input_channels*2, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=input_channels)]
            
            end = nn.Sequential(*layers)
            return end
        
        self.encoder1 = start_block(1, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)
        
        self.bridge = bridge(512, 1024)
        self.upconv = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                         kernel_size=2, stride=2, padding=0, bias=True)
            
        self.decoder4 = decoder_block(512, 512)
        self.decoder3 = decoder_block(256, 256)
        self.decoder2 = decoder_block(128, 128)
        self.decoder1 = end_block(64, 64)
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1, padding=0, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(512,self.num_classes)
        
    #data propagation
    def forward(self, x):
        encoder1 = self.encoder1(x)
        
        encoder2 = self.encoder2(encoder1)
        
        encoder3 = self.encoder3(encoder2)
        
        encoder4 = self.encoder4(encoder3)
        
        bridge = self.bridge(encoder4)
    
        y = self.avgpool(bridge)
        y = torch.flatten(y,1)
        y = self.fc2(y)
        
        upconv = self.upconv(bridge)
    
        cat4 = torch.cat((upconv, encoder4), dim=1)
        decoder4 = self.decoder4(cat4)
        
        cat3 = torch.cat((decoder4, encoder3), dim=1)
        decoder3 = self.decoder3(cat3)
        
        cat2 = torch.cat((decoder3, encoder2), dim=1)
        decoder2 = self.decoder2(cat2)
        
        cat1 = torch.cat((decoder2, encoder1), dim=1)
        decoder1 = self.decoder1(cat1)
        
        
        x = self.fc(decoder1)
        
        return x ,y

class UNet_early(nn.Module):
    def __init__(self ,num_classes=None):
        super(UNet_early, self).__init__()
        self.num_classes = num_classes
    
        def CBR2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels = input_channels, 
                                 out_channels = output_channels, 
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding=padding, bias = bias)]
            layers += [nn.BatchNorm2d(num_features=output_channels)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr
        
        def start_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
        
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=output_channels)]    
            
            start = nn.Sequential(*layers)
            return start
        
        def encoder_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            
            layers += [nn.MaxPool2d(kernel_size=kernel_size)]
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=output_channels)]
            
            
            enc = nn.Sequential(*layers)
            return enc
        
        def decoder_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
                        
           
            layers += [CBR2d(input_channels=input_channels*2, output_channels=output_channels)]
            layers += [CBR2d(input_channels=input_channels, output_channels=int(output_channels/2))]
            layers += [nn.ConvTranspose2d(in_channels=int(input_channels/2), out_channels=int(output_channels/2), 
                                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            
            dec = nn.Sequential(*layers)
            return dec
        
        def bridge(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            layers += [nn.MaxPool2d(kernel_size=2)]
            layers += [CBR2d(input_channels=input_channels, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=input_channels)]

            bridge = nn.Sequential(*layers)
            return bridge
        
        def end_block(input_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=True):
            layers = []
            layers += [CBR2d(input_channels=input_channels*2, output_channels=output_channels)]
            layers += [CBR2d(input_channels=output_channels, output_channels=input_channels)]
            
            end = nn.Sequential(*layers)
            return end
        
        self.encoder1 = start_block(1, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)
        
        self.bridge = bridge(512, 1024)
        self.upconv = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                         kernel_size=2, stride=2, padding=0, bias=True)
            
        self.decoder4 = decoder_block(512, 512)
        self.decoder3 = decoder_block(256, 256)
        self.decoder2 = decoder_block(128, 128)
        self.decoder1 = end_block(64, 64)
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=1, stride=1, padding=0, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(512,self.num_classes)
        
    #data propagation
    def forward(self, x):
        encoder1 = self.encoder1(x)
        
        encoder2 = self.encoder2(encoder1)
        
        encoder3 = self.encoder3(encoder2)
        
        encoder4 = self.encoder4(encoder3)

        y = self.avgpool(encoder4)
        y = torch.flatten(y,1)
        y = self.fc2(y)
        
        bridge = self.bridge(encoder4)
        
        upconv = self.upconv(bridge)
    
        cat4 = torch.cat((upconv, encoder4), dim=1)
        decoder4 = self.decoder4(cat4)
        
        cat3 = torch.cat((decoder4, encoder3), dim=1)
        decoder3 = self.decoder3(cat3)
        
        cat2 = torch.cat((decoder3, encoder2), dim=1)
        decoder2 = self.decoder2(cat2)
        
        cat1 = torch.cat((decoder2, encoder1), dim=1)
        decoder1 = self.decoder1(cat1)
        
        
        x = self.fc(decoder1)
        
        return x ,y

class aux_U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=3 , num_classes=3):
        super(aux_U_Net,self).__init__()
        self.num_classes = num_classes
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        # --- aux path ---
        self.aux = aux_conv(ch_in=512, ch_out=1024)
        #  --- decoding path ---
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc2 = nn.Linear(512,self.num_classes)

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # aux path
        y = self.avgpool(x5)
        y = torch.flatten(y,1)
        y = self.fc2(y)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1, y

class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """
    def __init__(self ,num_classes=None):
        super(R2AttU_Net, self).__init__()
        self.num_classes = num_classes
        in_ch = 1
        out_ch=6
        t=2 
        base_dim=32
        n1 = base_dim
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32, n1 * 64]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)
        self.RRCNN6 = RRCNN_block(filters[4], filters[5], t=t)
        self.RRCNN7 = RRCNN_block(filters[5], filters[6], t=t)

        self.Up7 = up_conv(filters[6], filters[5])
        self.Att7 = Attention_block(F_g=filters[5], F_l=filters[5], F_int=filters[4])
        self.Up_RRCNN7 = RRCNN_block(filters[6], filters[5], t=t)
        
        self.Up6 = up_conv(filters[5], filters[4])
        self.Att6 = Attention_block(F_g=filters[4], F_l=filters[4], F_int=filters[3])
        self.Up_RRCNN6 = RRCNN_block(filters[5], filters[4], t=t)
        
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc2 = nn.Linear(2048, self.num_classes)

       # self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)
        
        e6 = self.Maxpool5(e5)
        e6 = self.RRCNN6(e6)
        
        e7 = self.Maxpool6(e6)
        e7 = self.RRCNN7(e7)
        
        y = self.avgpool(e7)
        y = torch.flatten(y,1)
        y = self.fc2(y)

        d7 = self.Up7(e7)
        e6 = self.Att7(g=d7, x=e6)
        d7 = torch.cat((e6, d7), dim=1)
        d7 = self.Up_RRCNN7(d7)
        
        d6 = self.Up6(e6)
        e5 = self.Att6(g=d6, x=e5)
        d6 = torch.cat((e5, d6), dim=1)
        d6 = self.Up_RRCNN6(d6)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out , y

