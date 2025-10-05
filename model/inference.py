# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import util
from engine import trainer
from data_prepare import make_dataset, load_villages, week_start_range  # 前處理/週期工具

# 說明：
# 1) 先用 make_dataset() 依你現有規則把原始資料整理為週×里×特徵，並切出 x/y（seq_x/seq_y）
# 2) 載入訓練好的 best.pth，取「最後一筆 x 視窗」做未來 T 期預測
# 3) 反標準化 → 四捨五入 → 非負/上限裁切（cap）→ 存 CSV（里層級 + 區層級 共四份）

def run_inference(
    year: int,
    raw_dir: str,
    out_base: str,
    checkpoint: str,
    adjdata: str,
    num_nodes: int,
    in_dim: int = 8,
    seq_x: int = 4,
    horizon_T: int = 2,
    nhid: int = 32,
    dropout: float = 0.3,
    device_str: str = "cuda:0",
    gat_bool: bool = False,
    aptonly: bool = False,
    addaptadj: bool = False
):
    device = torch.device(device_str)

    # 1) 先「建資料集」到 training_data/dataset_TN_{year}_weekly_ext
    ds_dir = make_dataset(year, raw_dir, out_base, seq_x=seq_x, seq_y=horizon_T, y_start=1)
    ds_dir = Path(ds_dir)  # e.g. training_data/dataset_TN_2023_weekly_ext

    # 2) 載入整理後的資料 + scaler（y 只看病例的 mean/std）
    #    load_dataset 會：
    #    - 對 x 的每個通道做 per-feature 標準化
    #    - 以 y_train[...,0]（病例）做 scaler
    data = util.load_dataset(str(ds_dir), batch_size=1, valid_batch_size=1, test_batch_size=1)

    scaler = data["scaler"]
    # 用訓練集病例做分位數上限，抑制極端值
    ytr_cases = torch.tensor(data["y_train"][..., 0])
    cap = int(torch.quantile(ytr_cases, 0.995).item())
    cap = max(cap, 40)  # 至少不低於 40（依你的資料合理上限，可調）

    # 3) 建 model/engine（與訓練一致），拿到 pad_t
    supports = None
    if gat_bool and addaptadj:
        # 與訓練相同的圖設定
        _, _, adj = util.load_adj(adjdata, "doubletransition")
        supports = [torch.tensor(i, device=device) for i in adj]

    eng = trainer(
        scaler=scaler, in_dim=in_dim, seq_length=seq_x,
        num_nodes=num_nodes, nhid=nhid, dropout=dropout,
        lrate=1e-2, wdecay=1e-4, device=device,
        supports=supports, gat_bool=gat_bool, addaptadj=addaptadj, aptonly=aptonly, aptinit=None
    )

    print(f"loading model: {checkpoint}")
    eng.model.load_state_dict(torch.load(checkpoint, map_location=device))
    eng.model.eval()

    # 4) 取「最後一筆」觀測視窗作未來 T 期推論
    #    這裡直接抓 test.npz 的最後一個 x（已標準化），形狀 (1, Tx, V, F)
    x_last = data["x_test"][-1:]     # numpy
    V = x_last.shape[2]
    testx = torch.tensor(x_last, device=device).transpose(1, 3)   # (1, F, V, Tx)
    with torch.no_grad():
        tx = F.pad(testx, (eng.pad_t, 0, 0, 0))
        out, _ = eng.model(tx)                                     # (1, 1, V, T')
        yhat = out.permute(0, 3, 2, 1).squeeze(-1).cpu()           # -> (1, T', V)

    # 取所需的 horizon 數
    Tprime = yhat.shape[1]
    H = min(horizon_T, Tprime)
    yhat = yhat[:, :H, :]  # (1, H, V)

    # 5) 反標準化 → 四捨五入 → 非負/上限裁切 → 轉 int numpy
    pred_list = []
    for h in range(H):
        ph = scaler.inverse_transform(yhat[:, h, :])     # tensor, (1, V)
        ph = torch.round(ph)
        ph = torch.clamp(ph, min=0, max=cap)
        pred_list.append(ph.to(torch.int32).cpu().numpy()[0])  # (V,)
    pred_mat = np.stack(pred_list, axis=0)  # (H, V), 整數病例

    # 6) 生出未來 H 週的日期與欄位（里）
    villages = load_villages(str(Path(raw_dir) / "MosIndex_Tainan.csv"))
    vids = villages["VillageID"].tolist()
    # 由 週起始日生成完整週序列，取最後一週後面的 H 週
    weeks_all = week_start_range(year)
    future_weeks = pd.date_range(weeks_all[-1] + pd.Timedelta(days=7), periods=H, freq="W-MON")

    # 如果資料裡的里順序與 vids 不同，以資料為準重新對齊欄位
    if len(vids) != V:
        # 嘗試使用 ds 內部節點順序（graph_sensor_ids.txt）
        graph_ids_txt = Path(out_base) / f"out_graph_{year}" / "graph_sensor_ids.txt"
        if graph_ids_txt.exists():
            vids = graph_ids_txt.read_text(encoding="utf-8").split(",")
        else:
            vids = [str(i) for i in range(V)]  # 後備

    # 7) 輸出目錄
    out_dir = Path(out_base) / f"inference_TN_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 里層級輸出（兩份）
    # =========================
    # (a) 寬表：index=未來週，columns=VillageID
    df_wide = pd.DataFrame(pred_mat, index=future_weeks, columns=vids)
    df_wide.index.name = "week_start"
    # 確保整數輸出
    df_wide = df_wide.astype("Int64")
    wide_path = out_dir / f"forecast_T{H}_wide.csv"
    df_wide.to_csv(wide_path, encoding="utf-8-sig")

    # (b) 長表：week_start, VillageID, pred_cases
    df_long = (
        df_wide.reset_index()
               .melt(id_vars=["week_start"], var_name="VillageID", value_name="pred_cases")
    )
    # 確保整數輸出
    df_long["pred_cases"] = df_long["pred_cases"].astype("Int64")
    long_path = out_dir / f"forecast_T{H}_long.csv"
    df_long.to_csv(long_path, index=False, encoding="utf-8-sig")

    # =========================
    # 區層級輸出（兩份）
    # =========================
    # 里 → 區 對照
    vmap = villages.set_index("VillageID")[["TownCode"]].copy()
    # 檢查欄位對照完整性
    missing = [c for c in df_wide.columns if c not in vmap.index]
    if len(missing) > 0:
        raise ValueError(f"下列 VillageID 缺少 TownCode 對照：{missing}")

    # ---- 寬表（區）----
    # 將欄位 VillageID 依其 TownCode 做加總
    # 用轉置 groupby 再轉回，避免 pandas 對 axis=1 的未來棄用警告
    col_to_towncode = vmap["TownCode"].reindex(df_wide.columns)
    df_wide_district = (
        df_wide.T
               .groupby(col_to_towncode, sort=True)
               .sum()
               .T
               .astype("Int64")
    )
    df_wide_district.index.name = "week_start"
    wide_dist_path = out_dir / f"forecast_T{H}_wide_district.csv"
    df_wide_district.to_csv(wide_dist_path, encoding="utf-8-sig")

    # ---- 長表（區）----
    df_long_dist = (
        df_long.merge(vmap["TownCode"], left_on="VillageID", right_index=True, how="left")
               .rename(columns={"TownCode": "DistrictCode"})
               .groupby(["week_start", "DistrictCode"], as_index=False)["pred_cases"]
               .sum()
    )
    df_long_dist["pred_cases"] = df_long_dist["pred_cases"].astype("Int64")
    long_dist_path = out_dir / f"forecast_T{H}_long_district.csv"
    df_long_dist.to_csv(long_dist_path, index=False, encoding="utf-8-sig")

    # 8) meta
    meta = {
        "year": year,
        "seq_x_used": seq_x,
        "horizon_requested": horizon_T,
        "horizon_returned": int(H),
        "num_villages": int(V),
        "cap": int(cap),
        "dataset_dir": str(ds_dir),
        "checkpoint": checkpoint,
        "outputs": {
            "wide_csv": str(wide_path),
            "long_csv": str(long_path),
            "wide_csv_district": str(wide_dist_path),
            "long_csv_district": str(long_dist_path)
        }
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True, help="資料年度，例如 2023")
    ap.add_argument("--raw_dir", type=str, required=True, help="原始資料資料夾（包含 MosIndex_Tainan.csv / 病例 / 蚊卵 / 人口/面積）")
    ap.add_argument("--out_base", type=str, default="training_data", help="輸出資料根目錄（會在底下建立 dataset_TN_{year}_weekly_ext、inference_TN_{year}）")
    ap.add_argument("--checkpoint", type=str, required=True, help="訓練好的權重（best.pth）")
    ap.add_argument("--adjdata", type=str, required=True, help="圖的 adj_mat.pkl（若訓練有用圖）")
    ap.add_argument("--num_nodes", type=int, required=True, help="節點數（里數）")
    ap.add_argument("--in_dim", type=int, default=8)
    ap.add_argument("--seq_x", type=int, default=4, help="輸入時間窗長度")
    ap.add_argument("--horizon_T", type=int, default=2, help="要預測的未來時間步數")
    ap.add_argument("--nhid", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--gat_bool", action="store_true")
    ap.add_argument("--aptonly", action="store_true")
    ap.add_argument("--addaptadj", action="store_true")
    args = ap.parse_args()

    run_inference(
        year=args.year,
        raw_dir=args.raw_dir,
        out_base=args.out_base,
        checkpoint=args.checkpoint,
        adjdata=args.adjdata,
        num_nodes=args.num_nodes,
        in_dim=args.in_dim,
        seq_x=args.seq_x,
        horizon_T=args.horizon_T,
        nhid=args.nhid,
        dropout=args.dropout,
        device_str=args.device,
        gat_bool=args.gat_bool,
        aptonly=args.aptonly,
        addaptadj=args.addaptadj
    )
