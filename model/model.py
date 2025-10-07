import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchdiffeq import odeint
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
        self.in_dim = in_dim
        self.out_dim = out_dim
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
                                                   kernel_size=(1,kernel_size),dilation=(1, new_dilation)))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1, new_dilation)))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
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
        if input.size(1) != self.in_dim: 
            input = input.permute(0, 3, 2, 1).contiguous()
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
        x = F.softplus(x)
        return x, 0


# 2025.1.24

class seir_ode(nn.Module):
    """
    SEIR ODE model
    使用方法:
            seir_sol = odeint(
                self.ODE_FUNC,  # 微分方程邏輯
                y0,             # 初始條件
                self.t_points,  # 求解的時間點
                method='rk4'    # 求解方法，可選
            )
    """
    def __init__(self ,N):
        super(seir_ode, self).__init__()
        self.N = N

        if not isinstance(N, torch.Tensor):
            N = torch.tensor(N, dtype=torch.float32)
        self.register_buffer('N_placeholder', N)

        self.I0 = 1.0
        self.R0 = 0.0
        self.E0 = 0.0
        self.S0 = N - self.I0 - self.R0 - self.E0
        self.log_beta = nn.Parameter(torch.tensor([-1.20], dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.tensor([-1.65], dtype=torch.float32))
        self.log_gamma = nn.Parameter(torch.tensor([-2.30], dtype=torch.float32))

    def forward(self, t, y): # y: [Num_Nodes, 4] -> [S, E, I, R]
        beta = torch.exp(self.log_beta)
        sigma = torch.exp(self.log_sigma)
        gamma = torch.exp(self.log_gamma)

        S, E, I, R = y.unbind(dim=-1)

        dSdt = -beta * S * I / self.N_placeholder
        dEdt = beta * S * I / self.N_placeholder - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I

        return torch.stack([dSdt, dEdt, dIdt, dRdt], dim=-1)
    
class BatchODE(nn.Module):
    """
    ODE 函數的包裝器，用於在 odeint 內部處理 Batch x Node 的人口 N。
    """
    def __init__(self, ode_func: seir_ode):
        super().__init__()
        # 註冊 seir_ode 的可學習參數，使其參與梯度計算
        self.log_beta = ode_func.log_beta
        self.log_sigma = ode_func.log_sigma
        self.log_gamma = ode_func.log_gamma
        # 創建一個 buffer 來存放當前批次的人口 N_ode
        self.register_buffer('N_ode', torch.tensor([1.0], dtype=torch.float32))

    def set_N(self, N_ode_tensor):
        """設置當前批次的 N 向量 (Batch * Node)"""
        self.N_ode = N_ode_tensor

    def forward(self, t, y):
        # y 的形狀: [Batch*Node, 4]
        beta = torch.exp(self.log_beta)
        sigma = torch.exp(self.log_sigma)
        gamma = torch.exp(self.log_gamma)

        S, E, I, R = y.unbind(dim=-1) # S, E, I, R 的形狀: [Batch*Node]
        
        # 使用 set_N 傳入的 N_ode 進行除法，確保正確的節點廣播
        N_safe = self.N_ode.to(S.device)
        
        # SEIR 微分方程 (所有運算都具備自動微分能力)
        dSdt = -beta * S * I / N_safe
        dEdt = beta * S * I / N_safe - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I

        return torch.stack([dSdt, dEdt, dIdt, dRdt], dim=-1)
    
class hybrid_seir(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gat_bool=True, addaptadj=True, aptonly=False, noapt=False, aptinit=None, in_dim=8,out_dim=2,residual_channels=8,dilation_channels=8,skip_channels=32,end_channels=64,kernel_size=2,blocks=1,layers=2,emb_length=8):
        super(hybrid_seir, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
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
                                                   kernel_size=(1,kernel_size),dilation=(1, new_dilation)))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=(1, new_dilation)))

                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
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
        
        # --- SEIR 整合部分 ---
        # F=8: 0:cases, 1:egg_pos_rate, 2:egg_total, 3:pop, 4:area, 5:pop_density, 6:week_sin, 7:week_cos
        # F=0 (cases) 將被替換為 SEIR Infected (I)
        # F=5 (pop_density) 將被替換為 SEIR Exposed (E)

        N_dummy = 1.0 
        self.seir_ode_func = seir_ode(N=N_dummy)
        self.batch_ode_func = BatchODE(self.seir_ode_func)


    def forward(self, input):
        # 確保 input 具有正確的形狀 [B, F, V, T_in]
        if input.size(1) != self.in_dim: 
            input = input.permute(0, 3, 2, 1).contiguous()
        
        B, C, V, T_in = input.shape # B: batch size, C: feature dim, V: num nodes, T_in: input time length
        
        # ----------------------------------------------------
        # STAGE 1: SEIR ODE 求解
        # ----------------------------------------------------
        # 1. 時間點 T_points: 與輸入長度 T_in 匹配
        T_points = torch.arange(0, T_in, dtype=torch.float32).to(input.device)

        # 2. 提取初始條件 (y0) 和人口 (N)
        # 這裡我們使用**標準化後**的數值進行 SEIR 模擬
        
        # C=0 (cases) at t=0 -> I0
        I_init = input[:, 0, :, 0] 
        # C=3 (pop) at t=0 -> N_norm
        N_norm = input[:, 3, :, 0] 

        # 近似初始狀態 S0, E0, R0
        R_init = torch.zeros_like(I_init)
        E_init = I_init.clone() 
        S_init = N_norm - E_init - I_init - R_init
        
        # y0: [B, V, 4] -> Flatten to [B*V, 4]
        y0 = torch.stack([S_init, E_init, I_init, R_init], dim=-1).reshape(-1, 4)
        
        # N_ode: 每個節點的人口 N_norm (Flatten to [B*V])
        N_ode_tensor = N_norm.reshape(-1)
        
        # 3. 設置 N 並求解 ODE
        self.batch_ode_func.set_N(N_ode_tensor)
        
        # seir_sol_flat: [T, B*V, 4]
        seir_sol_flat = odeint(
            self.batch_ode_func,  
            y0,             
            T_points,  
            method='rk4'    
        )
        
        # seir_sol: [T, B, V, 4] -> Permute to [B, 4, V, T]
        seir_sol = seir_sol_flat.view(T_in, B, V, 4)
        seir_sol_permuted = seir_sol.permute(1, 3, 2, 0) # [B, 4, V, T_in]

        # ----------------------------------------------------
        # STAGE 2: 特徵替換 (Feature Replacement)
        # ----------------------------------------------------
        # 複製原始輸入，並替換核心特徵
        modified_input = input.clone() 
        
        # 替換 C=0 (cases) 為 SEIR Infected (I)
        I_ode = seir_sol_permuted[:, 2:3, :, :] # [B, 1, V, T_in]
        modified_input[:, 0:1, :, :] = I_ode
        
        # 替換 C=5 (pop_density) 為 SEIR Exposed (E) (作為輔助物理特徵)
        E_ode = seir_sol_permuted[:, 1:2, :, :] # [B, 1, V, T_in]
        modified_input[:, 5:6, :, :] = E_ode
        
        # ----------------------------------------------------
        # STAGE 3: GNN 處理 (與原來邏輯一致)
        # ----------------------------------------------------
        # 使用 modified_input 繼續後續的 GNN 流程
        
        in_len = modified_input.size(3)
        if in_len<self.receptive_field:
            x = F.pad(modified_input,(self.receptive_field-in_len,0,0,0))
        else:
            x = modified_input
            
        x = self.start_conv(x)
        skip = None
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
                skip = torch.zeros_like(s)
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
        x = F.softplus(x)
        
        # 回傳結果形狀與原來一致: [B, out_dim, V, T']
        return x, 0