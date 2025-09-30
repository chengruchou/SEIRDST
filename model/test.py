# -*- coding: utf-8 -*-
import torch
import argparse
import numpy as np
import util
import matplotlib.pyplot as plt
from engine import trainer
from model import seirdst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--adjdata', type=str, required=True)
    parser.add_argument('--adjtype', type=str, default='doubletransition')
    parser.add_argument('--gat_bool', action='store_true')
    parser.add_argument('--aptonly', action='store_true')
    parser.add_argument('--addaptadj', action='store_true')
    parser.add_argument('--randomadj', action='store_true')
    parser.add_argument('--seq_length', type=int, default=12)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--in_dim', type=int, default=8)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    # 資料 & scaler（y 的 scaler）
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # 載入圖
    supports = None
    if args.gat_bool and args.addaptadj:
        adj_mx = util.load_adj(args.adjdata, args.adjtype)
        supports = [torch.tensor(i).to(device) for i in adj_mx]

    # 用 engine 來拿 model 與 pad_t（保持與訓練一致）
    engine = trainer(
        scaler, in_dim=args.in_dim, seq_length=args.seq_length,
        num_nodes=args.num_nodes, nhid=args.nhid,
        dropout=args.dropout, lrate=0.01, wdecay=0.0001,
        device=device, supports=supports, gat_bool=args.gat_bool,
        addaptadj=args.addaptadj, aptinit=None,
    )

    print("loading model:", args.checkpoint)
    engine.model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    engine.model.eval()

    # 取出真值 (N, T, V)
    realy = torch.tensor(dataloader['y_test'], device=device)          # (N, Ty, V, F)
    realy = realy.transpose(1, 3)[:, 0, :, :].permute(0, 2, 1).cpu()   # -> (N, T, V)

    N = dataloader['y_test'].shape[0]
    B = max(1, min(args.batch_size, N))
    # 推論：補左側 padding；輸出轉成 (B, T', V)
    outputs = []
    for start in range(0, N, B):
        end = min(start + B, N)
        x = dataloader['x_test'][start:end]  # (b, Tx, V, F)
        testx = torch.tensor(x, device=device).transpose(1, 3)  # (b, F, V, T)
        with torch.no_grad():
            tx = torch.nn.functional.pad(testx, (engine.pad_t, 0, 0, 0))
            out, _ = engine.model(tx)                            # (b, 1, V, T')
            preds = out.permute(0, 3, 2, 1).squeeze(-1).cpu()    # -> (b, T', V)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)                                    # (N, T', V)
    yhat = yhat[:realy.shape[0], :, :] 
    # 對齊 horizon
    min_h = min(yhat.shape[1], realy.shape[1])
    yhat = yhat[:, :min_h, :]
    realy = realy[:, :min_h, :]

    print("Prediction shape:", tuple(yhat.shape), "Real shape:", tuple(realy.shape))

    # === cap（避免極端值） ===
    ytr_cases = torch.tensor(dataloader['y_train'][..., 0])
    cap = int(torch.quantile(ytr_cases, 0.995).item())
    cap = max(cap, 40)

    # === 指標（每一個 horizon）===
    amae, amape, armse = [], [], []
    for i in range(min_h):
        pred = scaler.inverse_transform(yhat[:, i, :])  # (N, V) tensor
        pred = torch.round(pred)
        pred = torch.clamp(pred, min=0, max=cap)
        real = realy[:, i, :]

        m1, m2, m3 = util.metric(pred, real)
        print(f"Horizon {i+1:2d}: Test MAE {m1:.4f}, MAPE {m2:.4f}, RMSE {m3:.4f}")
        amae.append(float(m1)); amape.append(float(m2)); armse.append(float(m3))

    print("On average over horizons, MAE {:.4f}, MAPE {:.4f}, RMSE {:.4f}"
          .format(np.mean(amae), np.mean(amape), np.mean(armse)))

    # === Exact / ±1 ===
    pred_all = []
    for i in range(min_h):
        pred_i = scaler.inverse_transform(yhat[:, i, :])  # (N, V) tensor
        pred_i = torch.round(pred_i)
        pred_i = torch.clamp(pred_i, min=0, max=cap)
        pred_all.append(pred_i.to(torch.int32).cpu().numpy())
    pred_all = np.stack(pred_all, axis=1)                 # (N, T, V)
    real_all = realy.to(torch.int32).cpu().numpy()

    exact_acc, within1_acc = [], []
    for i in range(min_h):
        p = pred_all[:, i, :].reshape(-1)
        r = real_all[:, i, :].reshape(-1)
        total = p.size
        exact = (p == r).sum() / total
        within1 = (np.abs(p - r) <= 1).sum() / total
        exact_acc.append(exact); within1_acc.append(within1)
        print(f"[Exact] Horizon {i+1:2d}: {exact*100:.2f}%  [±1] {within1*100:.2f}%")

    # 圖表
    plt.figure(figsize=(10,5))
    x = np.arange(1, min_h+1); w=0.4
    plt.bar(x - w/2, np.array(exact_acc)*100, width=w, label="Exact match (%)")
    plt.bar(x + w/2, np.array(within1_acc)*100, width=w, label="Within ±1 (%)")
    plt.xticks(x); plt.xlabel("Horizon (weeks ahead)"); plt.ylabel("Accuracy (%)")
    plt.title("Exact / Within-1 Accuracy by Horizon"); plt.legend(); plt.tight_layout()
    plt.savefig("test_exact_match_accuracy.png")

    # 隨機一個里、四個 horizon（自動去重）
    horizons_to_plot = sorted(set([0, min(2, min_h-1), min(5, min_h-1), min(11, min_h-1)]))
    vid = np.random.randint(0, yhat.shape[2])
    plt.figure(figsize=(12,8))
    for idx, h in enumerate(horizons_to_plot, 1):
        pred_h = scaler.inverse_transform(yhat[:, h, :])[:, vid]
        pred_h = torch.round(pred_h)
        pred_h = torch.clamp(pred_h, min=0, max=cap).numpy()
        real_h = realy[:, h, vid].numpy()
        plt.subplot(2,2,idx)
        plt.plot(real_h, label="Real cases", marker="o")
        plt.plot(pred_h, label="Pred cases", marker="x")
        plt.title(f"Village {vid}, Horizon={h+1}")
        plt.xlabel("Test sample index"); plt.ylabel("Weekly cases"); plt.legend()
    plt.tight_layout(); plt.savefig("test_pred_vs_real.png")


if __name__ == "__main__":
    main()
