# 📘 Tainan Dengue & Weather Data  112 & 104 Y
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
> - **Rainfall**：降雨量 (mm)（僅限降雨資料集）  
> - **Temperature**：平均氣溫 (°C)（僅限氣溫資料集）

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
> **主要欄位**：  
> - **County**：縣市  
> - **Town**：行政區  
> - **Village**：里別  
> - **MosqIndex**：病媒蚊指數（僅限 MosIndex_Tainan）  
> - **EggCount**：誘卵桶卵數（僅限 Ovitrap data）  
> - **Positive**：陽性桶數（僅限 Ovitrap data）  
> - **PositiveRate**：陽性率（僅限 Ovitrap data）  
> - **Date / Week**：觀測日期或週次  

---


## 請gpt排版
112年現住人口統計表 https://data.tainan.gov.tw/DataSet/Detail/8e6e59c5-6b8a-4baf-b525-fb87112d9d1b

