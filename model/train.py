import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default=r'test_dataset\METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default=r'test_dataset\adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gat_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=1000, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save', type=str, default='result', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

args = parser.parse_args()


def main():
    # set seed (optional)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    engine = trainer(
        scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
        args.learning_rate, args.weight_decay, device, supports,
        args.gat_bool, args.addaptadj, adjinit
    )

    print("scaler.mean:", engine.scaler.mean, "scaler.std:", engine.scaler.std)
    print("start training...", flush=True)

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')  # <── track 最佳驗證 loss

    for i in range(1, args.epochs + 1):
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.train(trainx, trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.eval(testx, testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss, mtrain_mape, mtrain_rmse = map(np.mean, [train_loss, train_mape, train_rmse])
        mvalid_loss, mvalid_mape, mvalid_rmse = map(np.mean, [valid_loss, valid_mape, valid_rmse])
        his_loss.append(mvalid_loss)

        log = ('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, '
               'Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch')
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                         mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        # === 只保留 best.pth ===
        if mvalid_loss < best_val_loss:
            best_val_loss = mvalid_loss
            torch.save(engine.model.state_dict(), args.save + "_best.pth")
            print(f"Best model updated at epoch {i}, val_loss={mvalid_loss:.4f}")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    engine.model.load_state_dict(torch.load(args.save + "_best.pth"))
    print("x_feature_mean:", dataloader.get('x_feature_mean'))
    print("x_feature_std :", dataloader.get('x_feature_std'))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)    # (N, T, V, F)
    realy = realy.transpose(1, 3)[:, 0, :, :]                # -> (N, V, T)

    # 逐批推論，確保左側 padding 與訓練一致
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            tx = F.pad(testx, (engine.pad_t, 0, 0, 0))       # 對齊 train/eval 的左側 padding
            out, _ = engine.model(tx)                        # (B, 1, V, T')
            preds = out.transpose(1, 3).squeeze(-1).permute(0, 2, 1)  # -> (B, V, T')
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)                         # (N, V, T')
    yhat = yhat[:realy.size(0), ...]

    # —— 對齊 horizon 長度，避免超界 ——
    min_h = min(yhat.shape[2], realy.shape[2])
    yhat = yhat[:, :, :min_h]                                # (N, V, H)
    realy = realy[:, :, :min_h]                              # (N, V, H)

    print("Training finished")
    print("The valid loss on best model is", str(round(best_val_loss, 4)))
    print("Aligned horizons:", min_h)

    amae, amape, armse = [], [], []
    for i in range(min_h):
        pred = scaler.inverse_transform(yhat[:, :, i])  # torch.Tensor, (N, V)
        pred = torch.round(pred)                        # ← 變成整數
        pred = torch.clamp(pred, min=0)                 # ← 不要負數
        real = realy[:, :, i]

        n = min(pred.shape[0], real.shape[0])
        pred = pred[:n]
        real = real[:n]
        # DEBUG（若要列印 min/max）
        pred = pred.to(dtype=torch.float32, device=real.device)
        real = real.to(dtype=torch.float32, device=real.device)

        pred_np = pred.detach().cpu().numpy()
        real_np = real.detach().cpu().numpy()
        print(f"DEBUG horizon {i}: pred min={pred_np.min():.2f}, max={pred_np.max():.2f}, "
            f"real min={real_np.min():.2f}, max={real_np.max():.2f}")

        # 指標（util.metric 用 torch，所以維持 tensor）
        m1, m2, m3 = util.metric(pred, real)
        print(f"Evaluate best model on test data for horizon {i+1}, "
            f"Test MAE: {m1:.4f}, Test MAPE: {m2:.4f}, Test RMSE: {m3:.4f}")
    amae.append(float(m1)); amape.append(float(m2)); armse.append(float(m3))


    print(f"On average over {min_h} horizons, Test MAE: {np.mean(amae):.4f}, "
          f"Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
