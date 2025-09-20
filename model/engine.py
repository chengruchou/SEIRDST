import torch.optim as optim
import torch
import torch.nn as nn
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout,
                 lrate, wdecay, device, supports,
                 gat_bool=True, addaptadj=True, aptonly=False, noapt=False, aptinit=None):
        self.model = seirdst(device, num_nodes, dropout,
            supports=supports,
            gat_bool=gat_bool,
            addaptadj=addaptadj,
            aptonly=aptonly,
            noapt=noapt,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=1,    
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16)
        self.model.to(device)
        self.pad_t = max(0, self.model.receptive_field - 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (self.pad_t, 0, 0, 0))
        output, _ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = real_val.permute(0, 2, 1).unsqueeze(-1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (self.pad_t, 0, 0, 0))
        output, _ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = real_val.permute(0, 2, 1).unsqueeze(-1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
