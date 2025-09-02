import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class graphattention(nn.Module):
    def __init__(self,c_in,c_out,dropout,d=16, emb_length=0, aptonly=False, noapt=False):
        super(graphattention,self).__init__()
        self.d = d
        self.aptonly = aptonly
        self.noapt = noapt
        self.mlp = linear(c_in*2,c_out)
        self.dropout = dropout
        self.emb_length = emb_length
        if aptonly:
            self.qm = FC(self.emb_length, d) 
            self.km = FC(self.emb_length, d)  
        elif noapt:
            self.qm = FC(c_in, d) 
            self.km = FC(c_in, d) 
        else:
            self.qm = FC(c_in + self.emb_length, d) 
            self.km = FC(c_in + self.emb_length, d)  

    def forward(self,x,embedding):
      
        out = [x]

        embedding = embedding.repeat((x.shape[0], x.shape[-1], 1, 1)) 
        embedding = embedding.permute(0,2,3,1) 

        if self.aptonly:
            x_embedding = embedding
            query = self.qm(x_embedding).permute(0, 3, 2, 1)
            key = self.km(x_embedding).permute(0, 3, 2, 1)  #
            
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  
            
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        elif self.noapt:
            x_embedding = x
            query = self.qm(x_embedding).permute(0, 3, 2, 1)  # 
            key = self.km(x_embedding).permute(0, 3, 2, 1)  #
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  #
            
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        else:
            x_embedding = torch.cat([x,embedding], axis=1) 
            query = self.qm(x_embedding).permute(0,3,2,1) 
            key = self.km(x_embedding).permute(0,3,2,1) 
           
            attention = torch.matmul(query, key.permute(0,1,3,2)) 
            
            attention /= (self.d**0.5)
            attention = F.softmax(attention, dim=-1)

        x = torch.matmul(x.permute(0,3,1,2), attention).permute(0,2,3,1)
        out.append(x)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h, 0


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class FC(nn.Module):
    def __init__(self,c_in,c_out):
        super(FC,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class seirdst(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gat_bool=True, addaptadj=True, aptonly=False, noapt=False, aptinit=None, in_dim=8,out_dim=2,residual_channels=8,dilation_channels=8,skip_channels=32,end_channels=64,kernel_size=2,blocks=1,layers=2,emb_length=8):
        super(seirdst, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gat_bool = gat_bool
        self.aptonly = aptonly
        self.noapt = noapt
        self.addaptadj = addaptadj
        self.emb_length = emb_length
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gat = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0

        if gat_bool and addaptadj:
            self.embedding = nn.Parameter(torch.randn(self.emb_length, num_nodes).to(device), requires_grad=True).to(device)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolution
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gat_bool:
                    self.gat.append(graphattention(dilation_channels,residual_channels,dropout, emb_length=emb_length, aptonly=aptonly, noapt=noapt))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        attentions = []
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate


            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gat_bool:
                if self.addaptadj:
                    x, att = self.gat[i](x, self.embedding)
                    

            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        
        return x, 0


# 2025.1.24


