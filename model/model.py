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
        self.register_buffer('N', N)

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

        dSdt = -beta * S * I / self.N
        dEdt = beta * S * I / self.N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I

        return torch.stack([dSdt, dEdt, dIdt, dRdt], dim=-1)
    
class Hybrid_SEIR_GNN(nn.Module):
    
    def __init__(self, device, num_nodes, N_total, t_points, GNN_params):
        super(Hybrid_SEIR_GNN, self).__init__()
        
        # ODE_Function (SEIR 核心)
        self.ode_func = seir_ode(N=N_total)
        self.register_buffer('t_points', t_points) 
        self.num_nodes = num_nodes
        self.device = device
        
        # seirdst (下游 GNN 模型)
        self.gnn_model = seirdst(device=device, num_nodes=num_nodes, **GNN_params)
        
        # 輔助特徵生成器 (將 SEIR 輸出 [T, N, 4] 轉換為 GNN 輸入 [1, 8, N, T])
        # GNN 的 in_dim=8，SEIR 只有 4 維，我們需要額外 4 維輔助特徵 X_aux
        self.aux_feature_generator = nn.Sequential(
             # 這裡可以是一個複雜的網路，但我們用一個簡單的線性層來示範如何處理維度
             nn.Linear(4, GNN_params['in_dim'] - 4)
        )
        
    # forward 函式現在接收: 
    # y0: 初始 SEIR 狀態 [Num_Nodes, 4] (S0, E0, I0, R0)
    # X_aux_input: 額外特徵，形狀: [T, Num_Nodes, C_aux]
    def forward(self, y0, X_aux_input):
        
        # ----------------------------------------------------
        # 階段一: 物理模型 (SEIR) 求解
        # ----------------------------------------------------
        # y0 的形狀: [Num_Nodes, 4]
        # odeint 求解結果: [T, Num_Nodes, 4] (時間, 節點, S E I R)
        seir_sol = odeint(
            self.ode_func,  
            y0,             
            self.t_points,  
            method='rk4'    
        ) 
        
        # ----------------------------------------------------
        # 階段二: 數據預處理 (構建 GNN 輸入張量)
        # ----------------------------------------------------
        # 1. 確保輔助特徵和 SEIR 求解結果的時間維度 T 一致
        if seir_sol.shape[0] != X_aux_input.shape[0]:
             raise ValueError("SEIR time steps must match auxiliary feature time steps.")

        # 2. SEIR 輸出作為特徵的一部分: [T, N, 4]
        seir_features = seir_sol
        
        # 3. 構建 GNN 的 in_dim=8 輸入張量
        # 合併 SEIR 輸出 [T, N, 4] 和 輔助特徵 [T, N, 4]
        # 我們將使用 X_aux_input 作為額外的 4 維特徵
        combined_features = torch.cat([seir_features, X_aux_input], dim=-1) # 形狀: [T, N, 8]
        
        # 4. 轉換為 seirdst 要求的輸入形狀: [B, C_in, N, T_in]
        # [T, N, C] -> [1, C, N, T] (假設 Batch Size = 1)
        # B=1, C=8, N=Num_Nodes, T=T_points
        gnn_input = combined_features.permute(2, 0, 1).unsqueeze(0) 
        # permute(1, 2, 0).unsqueeze(0) => [1, N, C, T] -> [1, C, N, T] (假設 seirdst 的 C 維度在第二位)
        # 這裡的維度轉換是 GNN 模型整合的常見難點，需要嚴格匹配 seirdst 內部期望的順序。
        # 根據 seirdst forward 的邏輯，它期望 C 維度在第二位：[B, C, N, T]
        gnn_input = combined_features.permute(2, 1, 0).unsqueeze(0) 
        # [T, N, C] -> permute(2, 1, 0) -> [C, N, T] -> unsqueeze(0) -> [1, C, N, T]

        # ----------------------------------------------------
        # 階段三: 後續模型 (seirdst GNN) 運算
        # ----------------------------------------------------
        final_output, _ = self.gnn_model(gnn_input) 
        
        # final_output 的形狀: [1, out_dim, N, T']
        return final_output, seir_sol # 同時回傳最終預測和 SEIR 軌跡 (用於分析)

# =========================================================
# 模擬使用
# =========================================================
# # 假設參數
# DEVICE = 'cpu'
# NUM_NODES = 10
# N_TOTAL = 1000 # 全局人口
# T_STEPS = 160
# T_POINTS = torch.linspace(0.0, 160.0, T_STEPS, dtype=torch.float32)

# # 模擬初始條件 (10 個節點)
# # S0, E0, I0, R0. 假設每個節點初始 I0=1
# I0_nodes = torch.ones(NUM_NODES, dtype=torch.float32) * 1.0
# R0_nodes = torch.zeros(NUM_NODES, dtype=torch.float32)
# E0_nodes = torch.zeros(NUM_NODES, dtype=torch.float32)
# S0_nodes = N_TOTAL - I0_nodes - R0_nodes - E0_nodes
# Y0_TENSOR = torch.stack([S0_nodes, E0_nodes, I0_nodes, R0_nodes], dim=-1) # [10, 4]

# # 模擬額外輔助特徵 X_aux (例如：天氣、節日等 4 個特徵)
# # 必須與 T_STEPS 和 NUM_NODES 維度匹配。
# X_AUX_INPUT = torch.randn(T_STEPS, NUM_NODES, 4) # 形狀 [T, N, 4]

# # 傳遞給 seirdst 模型的參數
# GNN_PARAMS = {
#     'in_dim': 8,
#     'out_dim': 2,
#     'residual_channels': 8,
#     'dilation_channels': 8,
#     'skip_channels': 32,
#     'end_channels': 64,
#     'kernel_size': 2,
#     'blocks': 1,
#     'layers': 2,
#     'emb_length': 8
# }

# # 實例化混合模型
# model = Hybrid_SEIR_GNN(DEVICE, NUM_NODES, N_TOTAL, T_POINTS, GNN_PARAMS)

# # 進行一次 Forward Pass
# final_pred, seir_results = model(Y0_TENSOR, X_AUX_INPUT)

# print(f"SEIR 求解軌跡形狀 (T, N, 4): {seir_results.shape}")
# print(f"GNN 最終預測形狀 (1, out_dim, N, T'): {final_pred.shape}")