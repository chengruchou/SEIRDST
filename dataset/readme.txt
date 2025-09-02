1. 登革熱相關公開資料 (台南市政府) 
https://data.tainan.gov.tw/DataSet?org=&sercat=&tag=&format=&keyword=%E7%99%BB%E9%9D%A9%E7%86%B1&sortingKey=updatedDate_true

2. 網格化氣象資料 (政府資料開放平台)
https://data.gov.tw/en/datasets/130309

如果要自己爬氣象資料:
台南市天氣資料：每小時的溫度、濕度、降雨量、能見度，每日的溫差、晴雨天
爬蟲程式可參考：https://github.com/JackyWeng526/Taiwan_Weather_Data
資料來源：https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp
站點位置：https://e-service.cwb.gov.tw/wdps/obs/state.htm
資料初步處理：https://e-service.cwb.gov.tw/HistoryDataQuery/downloads/Readme.pdf
雨量小於 0.1 則改為 0.05
紀錄錯誤改為 0
風向不定改為 0
無觀測改為 -1
累計於後改為 0
（只有無觀測才是缺值，其他都不是缺值）
資料缺值處理：
缺值可以由附近站點取值，或是取前後時間的值去計算(漸進過渡)。

p.s. 台南市里鄰界線有調整過，若空間單位有以里為單位須注意歷史資料之差異
可參考village_list

