# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import math, re, json, pickle

# ========= 工具 =========
def normalize_name(s:str):
    if pd.isna(s): return s
    s = str(s).strip().replace("台","臺")
    s = re.sub(r"\s+","", s).replace("　","")
    return s

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1,lat1,lon2,lat2])
    dlon, dlat = lon2-lon1, lat2-lat1
    a = (math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(a))

def parse_date_any(x):
    try:
        return datetime.strptime(str(int(x)), "%Y%m%d")
    except Exception:
        return pd.to_datetime(x, errors="coerce")

def egg_code_to_mos_code(egg_code_str:str)->str:
    s = str(egg_code_str).zfill(8)
    return s[:3] + s[-3:] + "0"   # 例如 67000320 → 6703200

def week_start_range(year:int):
    first_monday = pd.date_range(f"{year}-01-01", f"{year}-01-07", freq="D")
    first_monday = first_monday[first_monday.weekday==0][0]
    last_monday  = pd.date_range(f"{year}-12-25", f"{year}-12-31", freq="D")
    last_monday  = last_monday[last_monday.weekday==0][-1]
    return pd.date_range(first_monday, last_monday, freq="W-MON")

# ========= 里中心點（MosIndex） =========
def load_villages(mos_csv:str):
    use = ["County","Town","Village","VillageID","VillageLon","VillageLat"]
    df = pd.read_csv(mos_csv, encoding="utf-8-sig", usecols=use).drop_duplicates()
    df["County"]=df["County"].map(normalize_name)
    df["Town"]=df["Town"].map(normalize_name)
    df["Village"]=df["Village"].map(normalize_name)
    df = df[df["County"]=="臺南市"].copy()
    df["VillageLon"] = pd.to_numeric(df["VillageLon"], errors="coerce")
    df["VillageLat"] = pd.to_numeric(df["VillageLat"], errors="coerce")
    df = df.dropna(subset=["VillageLon","VillageLat"])
    df = df.drop_duplicates(subset=["VillageID"]).sort_values("VillageID").reset_index(drop=True)
    df["TownCode"] = df["VillageID"].str.extract(r"^(\d{7})-")
    return df

# ========= 病例 → 週×里 =========
def weekly_cases(dengue_csv:str, villages:pd.DataFrame, year:int):
    cas = pd.read_csv(dengue_csv, encoding="utf-8-sig", low_memory=False)
    date_col = next((c for c in ["確診日","發病日","日期","通報日","date","Date"] if c in cas.columns), None)
    cas["date"] = cas[date_col].apply(parse_date_any)
    cas = cas[(cas["date"]>=pd.Timestamp(f"{year}-01-01")) & (cas["date"]<=pd.Timestamp(f"{year}-12-31"))]
    cas["lon"] = pd.to_numeric(cas.get("經度", cas.get("lon")), errors="coerce")
    cas["lat"] = pd.to_numeric(cas.get("緯度", cas.get("lat")), errors="coerce")
    cas = cas.dropna(subset=["lon","lat","date"]).copy()

    vill_lons = villages["VillageLon"].to_numpy()
    vill_lats = villages["VillageLat"].to_numpy()
    vids = villages["VillageID"].tolist()

    nearest = []
    for _,r in cas.iterrows():
        dcoarse = np.sqrt((np.cos(np.radians(r["lat"]))*(r["lon"]-vill_lons))**2 + (r["lat"]-vill_lats)**2)
        top = np.argsort(dcoarse)[:5]
        dists = [haversine_km(r["lon"], r["lat"], vill_lons[i], vill_lats[i]) for i in top]
        nearest.append(vids[top[int(np.argmin(dists))]])
    cas["VillageID"] = nearest

    cas["week_start"] = cas["date"] - cas["date"].dt.weekday * np.timedelta64(1,"D")
    weeks = week_start_range(year)
    grid = pd.MultiIndex.from_product([weeks, vids], names=["week_start","VillageID"])
    g = cas.groupby(["week_start","VillageID"]).size().rename("cases").reset_index()
    df = (pd.DataFrame(index=grid).reset_index()
            .merge(g, on=["week_start","VillageID"], how="left")
            .fillna({"cases":0}).astype({"cases":int}))
    pivot = df.pivot(index="week_start", columns="VillageID", values="cases").reindex(index=weeks, columns=vids)
    return pivot

# ========= 蚊卵 → 週×里 =========
def weekly_egg(egg_csv:str, villages:pd.DataFrame, year:int):
    egg = pd.read_csv(egg_csv, encoding="utf-8-sig")
    egg["行政區域代碼"] = egg["行政區域代碼"].astype(str)
    def parse_period(s):
        m = re.search(r"(\d+)\D*第(\d+)\s*週", str(s))
        if not m: return pd.NaT
        y = 1911 + int(m.group(1)); w = int(m.group(2))
        return pd.to_datetime(f"{y}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
    egg["week_start"] = egg["監測週期"].map(parse_period)
    egg = egg[egg["week_start"].dt.year==year].copy()

    mos = villages.drop_duplicates(subset=["Town","TownCode"])[["Town","TownCode"]]
    egg["MosTownCode"] = egg["行政區域代碼"].map(egg_code_to_mos_code)
    egg = egg.merge(mos, left_on="MosTownCode", right_on="TownCode", how="left")
    egg["陽性率"] = egg["陽性率"].astype(str).str.replace("%","", regex=False)
    egg["陽性率"] = pd.to_numeric(egg["陽性率"], errors="coerce")/100.0
    egg.rename(columns={"總卵粒數 ":"總卵粒數"}, inplace=True)
    egg["總卵粒數"] = pd.to_numeric(egg["總卵粒數"], errors="coerce")

    e_week_town = egg.groupby(["Town","week_start"]).agg(
        egg_pos_rate=("陽性率","mean"),
        egg_total=("總卵粒數","mean")
    ).reset_index()

    vill = villages[["VillageID","Town"]].copy()
    weeks = week_start_range(year)
    grid = pd.MultiIndex.from_product([weeks, vill["VillageID"]], names=["week_start","VillageID"])
    base = pd.DataFrame(index=grid).reset_index().merge(vill, on="VillageID", how="left")
    out = base.merge(e_week_town, on=["Town","week_start"], how="left")
    out["egg_pos_rate"] = out.groupby("week_start")["egg_pos_rate"].transform(lambda s: s.fillna(s.median()))
    out["egg_total"]    = out.groupby("week_start")["egg_total"].transform(lambda s: s.fillna(s.median()))
    pivot1 = out.pivot(index="week_start", columns="VillageID", values="egg_pos_rate").reindex(index=weeks)
    pivot2 = out.pivot(index="week_start", columns="VillageID", values="egg_total").reindex(index=weeks)
    return pivot1, pivot2

# ========= 動態人口/面積 =========
def dynamic_pop_area(pop_csvs: dict, area_csv: str, villages: pd.DataFrame, weeks: pd.Index):
    area = pd.read_csv(area_csv, encoding="utf-8-sig")
    area["行政區別"] = area["行政區別"].map(normalize_name)
    area_map = area.set_index("行政區別")["面積（平方公里）"].to_dict()

    V = len(villages); T = len(weeks)
    mat_pop = np.zeros((T,V))
    mat_area = np.zeros((T,V))
    mat_density = np.zeros((T,V))

    pop_dfs = {}
    for m, path in pop_csvs.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["區域別"] = df["區域別"].map(normalize_name)
        pop_dfs[m] = df.set_index("區域別")["人口數總計"].to_dict()

    for ti, w in enumerate(weeks):
        m = w.month
        pop_map = pop_dfs.get(m, {})
        for vi, row in villages.iterrows():
            town = row["Town"]
            p = pop_map.get(town, np.nan)
            a = area_map.get(town, np.nan)
            mat_pop[ti,vi] = p
            mat_area[ti,vi] = a
            mat_density[ti,vi] = p/a if (p and a) else np.nan

    idx = pd.Index(weeks, name="week_start")
    cols = pd.Index(villages["VillageID"], name="VillageID")
    return (
        pd.DataFrame(mat_pop, index=idx, columns=cols),
        pd.DataFrame(mat_area, index=idx, columns=cols),
        pd.DataFrame(mat_density, index=idx, columns=cols),
    )

# ========= 組裝成 (T,V,F) & 切 NPZ =========
def make_dataset(year:int, base_in:str, base_out:str, seq_x:int=4, seq_y:int=2, y_start:int=1):
    base_in = Path(base_in); base_out = Path(base_out)
    out_dir = base_out / f"dataset_TN_{year}_weekly_ext"
    out_dir.mkdir(parents=True, exist_ok=True)

    villages = load_villages(str(base_in/"MosIndex_Tainan.csv"))
    df_cases = weekly_cases(str(base_in/f"{year}_dengue.csv"), villages, year)
    df_cases.to_hdf(out_dir / f"tn_dengue_weekly_{year}.h5", key="df")

    df_ep, df_eggs = weekly_egg(str(base_in/"egg_and_positive_2023.csv"), villages, year)

    # 動態人口
    pop_csvs = {m: str(base_in/"112年現住人口統計表"/f"112年{m}月現住人口統計表.csv") for m in range(1,13)}
    df_pop, df_area, df_density = dynamic_pop_area(
        pop_csvs, str(base_in/"臺南市112年所轄面積暨行政區里鄰數"/"臺南市所轄面積暨行政區里鄰數-112年1月.csv"), villages, df_cases.index
    )

    weeks = df_cases.index
    vids  = df_cases.columns
    V, T = len(vids), len(weeks)

    week_num = pd.Series(range(1, T+1), index=weeks)
    week_sin = np.sin(2*np.pi*week_num/52.0).to_numpy()[:,None].repeat(V, axis=1)
    week_cos = np.cos(2*np.pi*week_num/52.0).to_numpy()[:,None].repeat(V, axis=1)

    mats = [
        df_cases.values,                                     # 0 cases
        df_ep.reindex(index=weeks, columns=vids).values,     # 1 egg_pos_rate
        df_eggs.reindex(index=weeks, columns=vids).values,   # 2 egg_total
        df_pop.reindex(index=weeks, columns=vids).values,    # 3 pop
        df_area.reindex(index=weeks, columns=vids).values,   # 4 area
        df_density.reindex(index=weeks, columns=vids).values,# 5 pop_density
        week_sin,                                            # 6 week_sin
        week_cos,                                            # 7 week_cos
    ]
    data = np.stack(mats, axis=-1)
    if np.isnan(data).any():
        print("WARNING: NaN detected in data, filling with 0")
        data = np.nan_to_num(data, nan=0.0) 

    F = data.shape[-1]
    x_offsets = np.sort(np.arange(-(seq_x-1), 1, 1))
    y_offsets = np.sort(np.arange(y_start, seq_y+1, 1))
    x, y = [], []
    min_t = abs(min(x_offsets)); max_t = abs(T - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x.append(data[t + x_offsets, ...]); y.append(data[t + y_offsets, ...])
    x, y = np.stack(x, 0), np.stack(y, 0)

    N = x.shape[0]
    n_test = round(N*0.2); n_train = round(N*0.7); n_val = N-n_test-n_train

    np.savez_compressed(out_dir/"train.npz", x=x[:n_train], y=y[:n_train],
                        x_offsets=x_offsets.reshape(-1,1), y_offsets=y_offsets.reshape(-1,1))
    np.savez_compressed(out_dir/"val.npz", x=x[n_train:n_train+n_val], y=y[n_train:n_train+n_val],
                        x_offsets=x_offsets.reshape(-1,1), y_offsets=y_offsets.reshape(-1,1))
    np.savez_compressed(out_dir/"test.npz", x=x[-n_test:], y=y[-n_test:],
                        x_offsets=x_offsets.reshape(-1,1), y_offsets=y_offsets.reshape(-1,1))

    report = {
        "year": year, "T": int(T), "V": int(V), "F": int(F),
        "x_shape": list(x.shape), "y_shape": list(y.shape),
        "samples": {"train":int(n_train), "val":int(n_val), "test":int(n_test)},
        "dataset_dir": str(out_dir)
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return out_dir

# ========= 建圖 =========
def build_graph_from_villages(villages:pd.DataFrame, out_dir:Path, k=8, thresh=0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    sids = villages["VillageID"].astype(str).tolist()
    lons = villages["VillageLon"].to_numpy(); lats = villages["VillageLat"].to_numpy()
    n = len(sids); dist = np.full((n,n), np.inf, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i==j: continue
            dist[i,j] = haversine_km(lons[i], lats[i], lons[j], lats[j])
    edges=set()
    for i in range(n):
        idx = np.argsort(dist[i,:])
        for j in [j for j in idx if j!=i][:k]:
            edges.add((i,j)); edges.add((j,i))
    pd.DataFrame([(sids[i],sids[j],float(dist[i,j])) for i,j in edges],
                 columns=["from","to","distance"]).to_csv(out_dir/"distances.csv", index=False, encoding="utf-8-sig")
    (out_dir/"graph_sensor_ids.txt").write_text(",".join(sids), encoding="utf-8")
    sigma = dist[np.isfinite(dist)].std() or 1.0
    adj = np.exp(-np.square(dist/sigma)); adj[adj<thresh]=0.0
    with open(out_dir/"adj_mat.pkl","wb") as f:
        pickle.dump([sids, {s:i for i,s in enumerate(sids)}, adj], f, protocol=2)
    return str(out_dir)

# ========= 一鍵執行 =========
def run_all(year:int=2023, base_in:str="dataset_find", base_out:str="training_data", make_graph=False):
    out_dir = make_dataset(year, base_in, base_out)
    if make_graph:
        villages = load_villages(str(Path(base_in)/"MosIndex_Tainan.csv"))
        build_graph_from_villages(villages, Path(base_out)/f"out_graph_{year}")
    print("Done:", out_dir)

# 範例
run_all(2023, base_in="dataset_find", base_out="training_data", make_graph=True)
