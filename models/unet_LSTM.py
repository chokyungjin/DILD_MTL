import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np

########### New version ##########
class Patient_Cls_model(torch.nn.Module):
    def __init__(self, encoder_model, hidden_dim,  num_classes=None, end2end=False, backbone_name='unet'):
        super(Patient_Cls_model, self).__init__()        
        self.end2end = end2end
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        # bottleNeck encoder
        self.encoder_model = torch.nn.Sequential(*list(encoder_model.children()))[:5] 
        # Slicewise Feature Extract
        self.encoder   = self.encoder_model
        self.pool      = torch.nn.AdaptiveAvgPool2d(1)
        # LSTM
        if backbone_name == 'unet':
            self.LSTM    = nn.LSTM(input_size=self.hidden_dim, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True)
        else :
            self.LSTM    = nn.LSTM(input_size=640, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(512*2, 512, True)
        self.relu1   = nn.ReLU(True)
        self.drop1   = nn.Dropout(p=0.5)
        # Head
        self.fc      = nn.Linear(512, self.num_classes, True)       

    def forward(self, x, x_lens):
        cnn_embed_seq = []
        for i in range(x.shape[-1]):
            if self.end2end:
                out = self.encoder(x[..., i])
                #out = self.pool(out)
                out = out.view(out.shape[0], -1)
                cnn_embed_seq.append(out)   
            
            else:
                self.encoder.eval()
                with torch.no_grad():    
                    out = self.encoder(x[..., i]) # 512,32,32
                    out = self.pool(out) # 512,1,1
                    out = out.view(out.shape[0], -1) # 512, 1
                    cnn_embed_seq.append(out)

        stacked_feat = torch.stack(cnn_embed_seq, dim=1) # 512, 16 , 1
        self.LSTM.flatten_parameters()  # For Multi GPU  
        x_packed = pack_padded_sequence(stacked_feat, x_lens, 
        batch_first=True, enforce_sorted=False)
        RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    
        # input shape = batch, seq, feature
        
        fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
        x = self.linear1(fc_input)
        x = self.relu1(x)  
        x = self.drop1(x)  
        x = self.fc(x)     

        return x  

########### New version ##########
class Patient_Cls_Seg_model(torch.nn.Module):
    def __init__(self, encoder_model, hidden_dim,  num_classes=None, end2end=False, backbone_name='unet'):
        super(Patient_Cls_Seg_model, self).__init__()        
        self.end2end = end2end
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.unet = encoder_model
        # bottleNeck encoder
        self.only_encoder_model = torch.nn.Sequential(*list(encoder_model.children()))[:5] 
        # Slicewise Feature Extract
        self.encoder   = self.only_encoder_model
        self.pool      = torch.nn.AdaptiveAvgPool2d(1)
        # LSTM
        if backbone_name == 'unet':
            self.LSTM    = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True)
        else :
            self.LSTM    = nn.LSTM(input_size=640, hidden_size=512, num_layers=3, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(512*2, 512, True)
        self.relu1   = nn.ReLU(True)
        self.drop1   = nn.Dropout(p=0.5)
        # Head
        self.fc      = nn.Linear(512, self.num_classes, True)       


    def forward_imp(self,x):    
        cnn_seg_whole = []
        for idx in range(x.shape[0]):
            seg_array = []
            for i in range(x.shape[-1]):
                if self.end2end:
                    seg_array.append(self.unet(x[idx,:,:,:, i].unsqueeze(0)))
                else:
                    self.encoder.eval()
                    with torch.no_grad():
                        seg_array.append(self.unet(x[idx,:,:,:, i].unsqueeze(0)))
            cnn_seg_whole.append(torch.stack(seg_array, dim=-1))
        seg_stacked_feat = torch.cat(cnn_seg_whole , 0)

        return seg_stacked_feat

    def forward(self, x, x_lens):
        cnn_embed_seq = []
        for i in range(x.shape[-1]):
            if self.end2end:
                out = self.encoder(x[..., i])[-1]
                #out = self.pool(out)
                out = out.view(out.shape[0], -1)
                cnn_embed_seq.append(out)
            
            else:
                self.encoder.eval()
                with torch.no_grad():    
                    out = self.encoder(x[..., i])[-1]
                    #out = self.pool(out)
                    out = out.view(out.shape[0], -1)
                    cnn_embed_seq.append(out)

        out = self.forward_imp(x)
        stacked_feat = torch.stack(cnn_embed_seq, dim=1)
        self.LSTM.flatten_parameters()  # For Multi GPU  
        x_packed = pack_padded_sequence(stacked_feat, x_lens, 
        batch_first=True, enforce_sorted=False)
        RNN_out, (h_n, h_c) = self.LSTM(x_packed, None)    
        # input shape = batch, seq, feature
        
        fc_input = torch.cat([h_n[-1], h_n[-2]], dim=-1) # Due to the Bi-directional
        x = self.linear1(fc_input)
        x = self.relu1(x)  
        x = self.drop1(x)  
        x = self.fc(x)

        
        return out, x 

