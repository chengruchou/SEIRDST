import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import util
from model import seirdst

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default=r'test_dataset\METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str, default=r'test_dataset\adj_mx.pkl', help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')

    # 與 train.py 命名一致的開關
    parser.add_argument('--gat_bool', action='store_true', help='whether to add graph attention/convolution branch')
    parser.add_argument('--aptonly', action='store_true', help='use only adaptive adjacency (ignore external supports)')
    parser.add_argument('--addaptadj', action='store_true', help='enable adaptive adjacency')
    parser.add_argument('--randomadj', action='store_true', help='randomly initialize adaptive adjacency')

    # 模型/資料形狀參數
    parser.add_argument('--seq_length', type=int, default=12, help='prediction horizon')
    parser.add_argument('--nhid', type=int, default=32, help='hidden channels base')
    parser.add_argument('--in_dim', type=int, default=2, help='input feature channels')
    parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # 測試批次
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for test')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to .pth checkpoint')

    # 視覺化
    parser.add_argument('--plotheatmap', type=str, default='False', help='plot learned adjacency heatmap if True')

    return parser.parse_args()

def build_model(args, device):
    # 讀圖結構並準備 supports
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    # 建立 SEIRDST（長期正解：單通道輸出）
    model = seirdst(
        device=device,
        num_nodes=args.num_nodes,
        dropout=args.dropout,
        supports=supports,
        gat_bool=args.gat_bool,
        addaptadj=args.addaptadj,
        aptonly=args.aptonly,
        noapt=False,
        aptinit=adjinit,
        in_dim=args.in_dim,
        out_dim=1,  # 關鍵：與訓練保持單通道輸出
        residual_channels=args.nhid,
        dilation_channels=args.nhid,
        skip_channels=args.nhid * 8,
        end_channels=args.nhid * 16
    ).to(device)
    return model

def main():
    args = parse_args()
    device = torch.device(args.device)
    print(args)

    # 構建模型並載入權重
    model = build_model(args, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print('model load successfully')

    # 計算測試時需要的左側 padding（使輸出 T == seq_length）
    pad_t = max(0, model.receptive_field - 1)

    # 載入資料與 scaler
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # 準備真值：[N, V, T]
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # (N, V, T)

    # 推論
    outputs = []
    with torch.no_grad():
        for x, y in dataloader['test_loader'].get_iterator():
            testx = torch.Tensor(x).to(device)           # (N, T, V, F)
            testx = testx.transpose(1, 3)                # -> (N, F, V, T)
            tx = F.pad(testx, (pad_t, 0, 0, 0))          # 左側 padding，對齊訓練/驗證
            out, _ = model(tx)                           # (N, 1, V, T)
            preds = out.transpose(1, 3).squeeze(-1).permute(0, 2, 1)  # -> (N, V, T)
            outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # 對齊樣本數
    for i in range(1, args.seq_length):
        delta = (yhat[:, :, i] - yhat[:, :, i-1]).abs().mean().item()
        print(f'Delta between horizon {i} and {i+1}: {delta:.6f}')
    # 逐 horizon 評估
    amae, amape, armse = [], [], []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])  # (N, V)
        real = realy[:, :, i]                           # (N, V)
        mae, mape, rmse = util.metric(pred, real)
        print(f'Evaluate best model on test data for horizon {i+1}, '
              f'Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test RMSE: {rmse:.4f}')
        amae.append(mae); amape.append(mape); armse.append(rmse)

    print(f'On average over {args.seq_length} horizons, '
          f'Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}')

    # 可選：視覺化自適應圖（用 embedding 構成靜態相似度矩陣）
    if args.plotheatmap == 'True' and args.addaptadj and hasattr(model, 'embedding'):
        E = model.embedding.detach().to('cpu')          # (emb, V)
        A = torch.mm(E.t(), E)                          # (V, V) 內積相似
        A = F.softmax(F.relu(A), dim=1).numpy()         # 參考 GWNet 式處理
        A = A * (1 / np.max(A))                         # normalize 到 [0,1]
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = pd.DataFrame(A)
        sns.heatmap(df, cmap='RdYlBu')
        plt.savefig('./seirdst_embedding_adj.pdf')
        print('Saved heatmap to ./seirdst_embedding_adj.pdf')

if __name__ == '__main__':
    main()
