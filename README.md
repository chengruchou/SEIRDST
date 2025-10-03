# SEIR-DST Graph Neural Network for Dengue Weekly Forecasting

本專案使用 **時空圖神經網路 (SEIR-DST)** 模型，搭配真實的區里級登革熱病例數、蚊卵監測數據、人口與面積資訊，進行 **未來 T 個週期的病例數預測**。  
整體流程：**資料整理 → 建立圖結構 → 訓練 → 測試與視覺化 → 推論輸出**。

---

## 📂 專案結構

```
.
├── dataset_find/     # 原始數據 (病例、蚊卵、人口、面積、氣象資料等)
├── model/            # 核心程式碼 (模型、訓練、測試、推論等)
│   ├── data_prepare.py
│   ├── engine.py
│   ├── inference.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   └── util.py
├── README.md
└── LICENSE
```

---

## ⚙️ 環境需求

- Python 3.9+
- PyTorch 2.0+
- 其他套件：
  ```bash
  pip install numpy pandas scipy matplotlib torchdiffeq h5py tables
  ```

---

## 📂 目錄說明

### dataset_find
此資料夾存放原始數據，主要包含：
- `2015_dengue.csv`, `2023_dengue.csv`：每週各里登革熱病例數
- `egg_and_positive_2023.csv`：各里蚊卵監測數據（陽性卵數與總卵數）
- `112年現住人口統計表/*.csv`：各里人口數資料
- `臺南市112年所轄面積暨行政區里鄰數.csv`：各里面積
- `MosIndex_Tainan.csv`：各里的地理資訊（ID、經緯度）
- `Grid_ObsRain_臺南市.csv`：逐時網格化降雨觀測資料
- `Grid_ObsTempAvg_臺南市.csv`：逐時網格化平均氣溫資料

> ⚠️ 注意：目前程式中主要使用病例數、蚊卵、人口、面積與 MosIndex，氣象資料雖有保留但目前數據僅統計至2022年，因此尚未整合進 2023 模型特徵。

---

### model
此資料夾為專案核心程式碼，包含整個建模、訓練與推論流程：

- `data_prepare.py`  
  整合原始資料，產生 **週別 × 里 × 特徵** 的矩陣，並切割成 `train/val/test.npz`。

- `engine.py`  
  訓練引擎：包含 Loss 計算、反向傳播、標準化處理與評估函式。

- `model.py`  
  模型結構（SEIR-DST Graph Neural Network），支援 GAT 與適應性鄰接矩陣。

- `train.py`  
  讀入 `.npz` 與圖結構檔 (`adj_mat.pkl`)，進行模型訓練，並儲存最佳模型 checkpoint。

- `test.py`  
  載入 checkpoint，計算評估指標（MAE, MAPE, RMSE, Exact/±1 Accuracy），並輸出圖表。

- `inference.py`  
  對最新資料進行推論，產生未來 T 期病例數的預測結果，輸出 CSV。

- `util.py`  
  工具函式：`DataLoader`、`StandardScaler`、metrics、鄰接矩陣處理等。

---

## 📊 資料格式

`data_prepare.py` 輸出的 dataset 結構：

- `train.npz`, `val.npz`, `test.npz`  
  - `x`: (N, Tx, V, F)  
  - `y`: (N, Ty, V, F)  
  - `x_offsets`: 時間窗切片  
  - `y_offsets`: 預測 horizon  

其中：
- `N`: 樣本數
- `Tx`: 輸入時間長度
- `Ty`: 輸出時間長度
- `V`: 節點數（里數）
- `F`: 特徵數（預設 8）

---

## 📂 資料集來源與前處理

程式會整合上述資料，並依週進行對齊與特徵組合，最終生成：

- 每週 × 每里 × 特徵矩陣 `data`  
- 分割為 `train/val/test` 三組 `.npz` 檔案  
- 對應的特徵順序如下：
  ```
  0: cases (週病例數)
  1: egg_positive (陽性卵數)
  2: egg_total (總卵數)
  3: population (人口數)
  4: area (面積)
  5: pop_density (人口密度)
  6: week_sin (週期性 sine)
  7: week_cos (週期性 cosine)
  ```
- 輸出資料夾：`dataset_TN_{year}_weekly_ext/`，內含：
  - `train.npz`, `val.npz`, `test.npz`（模型訓練用）
  - `tn_dengue_weekly_{year}.h5`（完整數據保存）

---

## 🖥️ data_prepare.py 使用方式

### 產生 Dataset
```bash
python model/data_prepare.py 
```

### 主要參數 (程式碼最後一行中設定)
- `--year`：資料年份（例如 `2023`）
- `--base_in`：原始資料目錄（需包含病例、蚊卵、人口、面積、MosIndex）
- `--base_out`：輸出目錄（會自動建立 dataset 資料夾）
- `--seq_x`：輸入序列長度（預設 4，模型要拿多少週之前資料）
- `--seq_y`：輸出序列長度（預設 2，模型要預測多少週之後）
- `--y_start`：預測起始位移（預設 1）

---

## 🚀 模型訓練

```bash
python model/train.py \
--device cuda:0 \
--data training_data/dataset_TN_2023_weekly_ext \
--adjdata training_data/out_graph_2023/adj_mat.pkl \
--adjtype doubletransition \
--num_nodes 751 \
--in_dim 8 \
--seq_length 4 \
--nhid 32 \
--dropout 0.3 \
--batch_size 64 \
--learning_rate 0.001 \
--epochs 200 \
--save outputs/exp1
```

訓練完成後會輸出：`outputs/exp1_best.pth`

---

## 📈 模型測試與視覺化

```bash
python model/test.py \
--device cuda:0 \
--data training_data/dataset_TN_2023_weekly_ext \
--adjdata training_data/out_graph_2023/adj_mat.pkl \
--adjtype doubletransition \
--num_nodes 751 \
--in_dim 8 \
--seq_length 4 \
--nhid 32 \
--dropout 0.3 \
--checkpoint outputs/exp1_best.pth
```

輸出：
- `test_exact_match_accuracy.png`（各 horizon 的精準率與 ±1 精準率）
- `test_pred_vs_real.png`（隨機里別的折線圖）

---

## 🔮 推論（未來病例數預測）

```bash
python model/inference.py \
--year 2023 \
--raw_dir dataset_find \
--out_base outputs \  
--checkpoint outputs/exp1_best.pth \  
--adjdata training_data/out_graph_2023/adj_mat.pkl \
--num_nodes 751 \
--in_dim 8 \
--seq_x 4 \
--horizon_T 2 \
--nhid 32 \
--dropout 0.3 \
--device cuda:0 \
--gat_bool \
--addaptadj
```
---

### 產出檔案與路徑
推論完成後會在 `outputs/inference_TN_2023/` 產出三個檔案：

1. `forecast_T2_wide.csv`（寬表）  
2. `forecast_T2_long.csv`（長表）  
3. `meta.json`（本次推論的設定摘要）  

以下為 `meta.json` 的重點欄位（實際內容以檔案為準）：
- `year`: 2023
- `seq_x_used`: 6（推論時實際採用的視窗長度）
- `horizon_requested`: 2（要求預測步數）
- `horizon_returned`: 2（實際輸出步數）
- `num_villages`: 751（里數）
- `cap`: 40（輸出截斷上限）
- `dataset_dir`: training_data/dataset_TN_2023_weekly_ext
- `outputs.wide_csv`: training_data/inference_TN_2023/forecast_T2_wide.csv
- `outputs.long_csv`: training_data/inference_TN_2023/forecast_T2_long.csv

---

### 檔案格式說明

#### 1) `forecast_T{T}_wide.csv`（寬表）
- **索引**：`week_start`（該預測對應的起始週日期）
- **欄位**：各 `VillageID`（共 751 欄）
- **儲存值**：四捨五入且經過區間裁切的整數病例數（見「後處理規則」）

> 適合用於：整體熱度圖、一次性檢視全里預測矩陣、與地圖/網格對位。

#### 2) `forecast_T{T}_long.csv`（長表）
- **欄位**：`week_start, VillageID, pred_cases`
- **每列**：某一週 × 某一里 × 預測值（整數）
- **特性**：更易於 groupby、串接到資料庫或與其他維度（行政區、群聚）做彙整分析。

> 適合用於：時間序列折線圖、分區聚合（例如：區級/都會區級）。

#### 3) `meta.json`（設定摘要）
- **用途**：記錄此次推論的關鍵配置與輸出路徑，方便追溯與重現。  
- **差異對齊**：`seq_x_used` 反映推論腳本最後採用的視窗長度；`horizon_returned` 反映模型實際可輸出的步數（可能因 checkpoint 設定不同而與要求值略有不同）。

---

## ⚠️ 注意事項

1. **`--in_dim` 必須與資料特徵數一致**（使用 8 個特徵時設為 8）。
2. **`--num_nodes` 必須與 `adj_mat.pkl` 節點數一致**。
3. 訓練與推論皆會先在 **標準化空間** 計算 loss，最後再還原、四捨五入、截斷到 `[0, cap]`。
4. 模型輸出已正規化為 `(B, H, V, 1)`，避免維度錯誤。
5. 建圖時請確保村里順序與 dataset 對齊。

---

## 📌 常見錯誤排除

- **`in_dim mismatch`** → 請確認 `--in_dim` 與 dataset 特徵數相同。
- **`num_nodes mismatch`** → 確認 `--num_nodes` 與 `adj_mat.pkl` 節點數一致。
- **`CUDA OOM`** → 調低 batch size 或 nhid。
- **`ModuleNotFoundError: torchdiffeq`** → 安裝：`pip install torchdiffeq`。

---

## 📜 授權

- 資料來源需遵循原單位授權。
- 程式碼授權依專案設定。
