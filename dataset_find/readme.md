# 📘 Tainan Dengue & Weather Data  (112 & 104 Year)
本專案蒐集並整理臺南市登革熱病例數、病媒蚊監測資料與網格化氣象資料，提供研究人員進行分析、建模與可視化使用。

---

## 📂 資料來源

### 1. 🌦️ 網格化氣象資料  
| 資料集 | 說明 | 下載連結 |
|--------|----------------|-------------------------------|
| **Grid_ObsRain_臺南市** | 臺南市逐時網格化降雨觀測資料 | [前往下載](https://data.gov.tw/en/datasets/130309) |
| **Grid_ObsTempAvg_臺南市** | 臺南市逐時網格化平均氣溫資料 | [前往下載](https://data.gov.tw/en/datasets/130307) |

> **格式**：CSV  
> **時間範圍**：依資料集更新狀況  
> **主要欄位**：  
> - **LocationID**：網格代號  
> - **ObsTime**：觀測時間  
> - **Rainfall**：降雨量 (mm)  
> - **Temperature**：平均氣溫 (°C)

---

### 2. 🦟 登革熱病例數  
| 資料集 | 說明 | 下載連結 |
|--------|----------------|-------------------------------|
| **臺南市本土登革熱病例數** | 臺南市各行政區每週登革熱病例數統計 | [前往下載](https://data.tainan.gov.tw/DataSet/Detail/6489b738-811c-49cd-82be-94e29ef8ddfb) |

> **格式**：CSV  
> **時間範圍**：依官方資料更新  
> **主要欄位**：  
> - **County**：縣市  
> - **District**：行政區  
> - **CaseCount**：病例數  
> - **ReportDate**：通報日期

---

### 3. 🐜 病媒蚊監測資料  
| 資料集 | 說明 | 下載連結 |
|--------|----------------|-------------------------------|
| **MosIndex_Tainan** | 臺南市病媒蚊調查資料 | [前往下載](https://data.gov.tw/dataset/24159) |
| **egg_and_positive_2023** | 臺南市登革熱誘卵桶監測資訊 (Ovitrap data) | [前往下載](https://data.gov.tw/dataset/128442) |

> **格式**：CSV  
> **時間範圍**：依官方資料更新  
> **主要欄位** (依資料集內容而定)：  
> - **District**：行政區  
> - **MosquitoIndex**：病媒蚊指數  
> - **EggCount**：誘卵桶卵數  
> - **PositiveRate**：陽性率  
> - **ObsTime**：觀測時間  

---
<!-- 

## 📥 資料下載方式

你可以直接在上方連結下載 CSV 檔案，或使用 `wget` / `curl` 取得資料，例如：

```bash
# 下載臺南市降雨資料
wget https://data.gov.tw/dataset/130309

# 下載臺南市平均氣溫資料
wget https://data.gov.tw/dataset/130307

# 下載臺南市本土登革熱病例數
wget https://data.tainan.gov.tw/DataSet/Detail/6489b738-811c-49cd-82be-94e29ef8ddfb

# 下載臺南市病媒蚊調查資料
wget https://data.gov.tw/dataset/24159

# 下載臺南市登革熱誘卵桶監測資訊
wget https://data.gov.tw/dataset/128442 -->
