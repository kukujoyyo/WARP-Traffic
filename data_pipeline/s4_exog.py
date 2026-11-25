"""S4: sensor-level exogenous variables, event labels and weather-adjusted capacity factors.

Outputs per district (data_proc/regions/D*):
  wx_hourly.npy      [H, N, F] standardised causal (as-of) model inputs
  wx_hourly_raw.npy  [H, N, R] raw values aligned to the true hour (for labels and analysis)
  events.npy         [T, N] uint8 bitmask (heat, rain, smoke, incident, lightrain, heat95, holiday)
  cap_factor_hourly.npy [H, N] weather capacity adjustment factor f(w) known at the hour
"""
import os, re, glob, json, zipfile, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
PROC = os.path.join(ROOT, "data_proc")
RAW = os.path.join(ROOT, "data_raw")
THR = os.path.join(ROOT, "data_raw", "firststreet", "2_Extreme_Threshold_ZIP.csv")
BITS = {"heat": 0, "rain": 1, "smoke": 2, "incident": 3, "lightrain": 4, "heat95": 5, "holiday": 6}
WX_NAMES = ["dT_thr", "T_C", "RH", "log_precip", "vis", "wind", "log_pm25", "smoke_hms"]
RAW_NAMES = ["T_F", "T_C", "RH", "precip_mm", "vis_mi", "wind_kt", "pm25", "smoke_hms", "T_thr_F"]
# HCM-style capacity adjustment factors (rain intensity in mm/h; visibility in miles)
CAF = dict(light=0.97, moderate=0.93, heavy=0.86, lowvis=0.90)


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def idw_weights(nlat, nlon, slat, slon, k=3, maxkm=60):
    D = haversine(nlat[:, None], nlon[:, None], slat[None], slon[None])
    W = np.zeros_like(D)
    order = np.argsort(D, 1)[:, :k]
    for i in range(len(nlat)):
        for j in order[i]:
            if D[i, j] <= maxkm or j == order[i, 0]:
                W[i, j] = 1.0 / max(D[i, j], 1.0) ** 2
    return W


def idw_apply(A, W):
    """A [H, S] with NaN; W [N, S] -> [H, N] missing-aware IDW (falls back to all stations)."""
    ok = ~np.isnan(A)
    num = np.nan_to_num(A) @ W.T
    den = ok.astype(np.float64) @ W.T
    out = num / np.where(den > 0, den, np.nan)
    if np.isnan(out).any():   # fall back to the region-wide mean at that hour
        fb = np.nanmean(A, 1, keepdims=True)
        out = np.where(np.isnan(out), fb, out)
    return out.astype(np.float32)


def load_thresholds(meta):
    thr = pd.read_csv(THR)
    thr = thr[thr.STATE == "California"]
    gz = None
    for fn in glob.glob(os.path.join(RAW, "census", "*.zip")):
        with zipfile.ZipFile(fn) as z:
            gz = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str})
    gz.columns = [c.strip() for c in gz.columns]
    gz["ZIP"] = gz.GEOID.astype(int)
    thr = thr.merge(gz[["ZIP", "INTPTLAT", "INTPTLONG"]], on="ZIP")
    D = haversine(meta.Lat.values[:, None], meta.Lng.values[:, None], thr.INTPTLAT.values[None], thr.INTPTLONG.values[None])
    j = D.argmin(1)
    return thr.HEAT_THRESHOLD_F.values[j].astype(np.float32), thr.ZIP.values[j], D.min(1)


def hms_density(meta, days):
    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(meta.Lng, meta.Lat)], crs="EPSG:4326")
    out = np.zeros((len(days), len(meta)), np.float32)
    lvl = {"Light": 1, "Medium": 2, "Heavy": 3}
    for i, d in enumerate(days):
        fn = os.path.join(RAW, "hms", f"hms_smoke{d:%Y%m%d}.zip")
        if not os.path.exists(fn) or os.path.getsize(fn) < 500:
            continue
        try:
            g = gpd.read_file(f"zip://{fn}")
        except Exception:
            continue
        if len(g) == 0 or "Density" not in g:
            continue
        g = g.set_crs("EPSG:4326", allow_override=True)
        g["lv"] = g.Density.map(lvl).fillna(0)
        bb = g.cx[meta.Lng.min() - 1:meta.Lng.max() + 1, meta.Lat.min() - 1:meta.Lat.max() + 1]
        if len(bb) == 0:
            continue
        j = gpd.sjoin(pts, bb[["lv", "geometry"]], predicate="within")
        if len(j):
            m = j.groupby(level=0).lv.max()
            out[i, m.index.values] = m.values
    return out


def incident_mask(meta, time):
    """[T, N] bool: crash on the same freeway & direction 0-1 mi downstream / 0-0.5 mi upstream, 0-60 min after start."""
    fn = os.path.join(PROC, "accidents_CA_2017_2021.parquet")
    acc = pd.read_parquet(fn)
    T, N = len(time), len(meta)
    mask = np.zeros((T, N), bool)
    t0 = time[0]
    route = meta.Fwy.str.extract(r"(\d+)")[0].astype(int).values
    dirn = meta.Fwy.str[-1].values
    vec = {"N": (1, 0), "S": (-1, 0), "E": (0, 1), "W": (0, -1)}
    lat0 = np.deg2rad(meta.Lat.mean())
    acc = acc[(acc.Start_Lat.between(meta.Lat.min() - 0.05, meta.Lat.max() + 0.05)) &
              (acc.Start_Lng.between(meta.Lng.min() - 0.05, meta.Lng.max() + 0.05))]
    n_match = 0
    for _, a in acc.iterrows():
        cand = route == a.route
        if a.dir in vec:
            cand &= dirn == a.dir
        if not cand.any():
            continue
        idx = np.where(cand)[0]
        dy = (a.Start_Lat - meta.Lat.values[idx]) * 69.0
        dx = (a.Start_Lng - meta.Lng.values[idx]) * 69.0 * np.cos(lat0)
        hit = []
        for ii, i in enumerate(idx):
            v = vec[dirn[i]]
            along = dy[ii] * v[0] + dx[ii] * v[1]          # >0: crash downstream of the sensor
            lat = abs(dy[ii] * v[1] - dx[ii] * v[0])
            if -0.5 <= along <= 1.0 and lat <= 0.3:
                hit.append(i)
        if not hit:
            continue
        s = int((a.Start_Time - t0) / pd.Timedelta("5min"))
        if s < 0 or s >= T:
            continue
        mask[s:min(T, s + 13), hit] = True
        n_match += 1
    return mask, n_match


def prep_accidents():
    out = os.path.join(PROC, "accidents_CA_2017_2021.parquet")
    if os.path.exists(out):
        return
    cols = ["ID", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng", "Street", "State", "Description"]
    fr = []
    for ch in pd.read_csv(os.path.join(RAW, "accidents", "US_Accidents_Dec21_updated.csv"), usecols=cols, chunksize=500_000):
        fr.append(ch[ch.State == "CA"])
    a = pd.concat(fr)
    a["Start_Time"] = pd.to_datetime(a.Start_Time.str[:19], errors="coerce")
    a = a[(a.Start_Time >= "2017-01-01") & (a.Start_Time < "2022-01-01")]
    st = a.Street.fillna("").str.upper()
    m = st.str.extract(r"(?:^|\s)(I|US|CA|SR|INTERSTATE|HIGHWAY|HWY|STATE ROUTE)[\s-]*(\d{1,3})\s*([NSEW])?(?:\s|$|B)")
    a["route"] = pd.to_numeric(m[1], errors="coerce")
    a["dir"] = m[2].fillna("")
    a = a[a.route.notna()]
    a["route"] = a.route.astype(int)
    a.to_parquet(out)
    print("accidents CA freeway", len(a))


def main(region):
    rd = os.path.join(PROC, "regions", region)
    meta = pd.read_csv(os.path.join(rd, "meta.csv"))
    time = pd.DatetimeIndex(np.load(os.path.join(rd, "time.npy")))
    W = os.path.join(PROC, "weather")
    hours = pd.DatetimeIndex(np.load(os.path.join(W, "grid_hours.npy")))
    H, N = len(hours), len(meta)
    st = pd.read_csv(os.path.join(W, "asos_stations.csv"))
    A = np.load(os.path.join(W, "asos_hourly.npy"))         # [H,S,7]: T_C,Td_C,RH,precip_mm,vis_mi,wind_kt,T_F
    Wn = idw_weights(meta.Lat.values, meta.Lng.values, st.lat.values, st.lon.values)
    feat = {}
    for f, k in [("T_C", 0), ("RH", 2), ("precip_mm", 3), ("vis_mi", 4), ("wind_kt", 5), ("T_F", 6)]:
        feat[f] = idw_apply(A[:, :, k], Wn)
    pm = np.load(os.path.join(W, "pm25_hourly.npy"))
    ps = pd.read_csv(os.path.join(W, "pm25_sites.csv"))
    Wp = idw_weights(meta.Lat.values, meta.Lng.values, ps.Latitude.values, ps.Longitude.values, k=3, maxkm=50)
    feat["pm25"] = idw_apply(pm, Wp)
    days = pd.date_range(hours[0].normalize(), hours[-1].normalize(), freq="D")
    smoke_d = hms_density(meta, days)                         # [D, N]
    day_of_hour = (hours.normalize() - days[0]).days.values
    feat["smoke_hms"] = smoke_d[day_of_hour]
    thr, zips, zd = load_thresholds(meta)
    meta["T_thr_F"], meta["zip"], meta["zip_km"] = thr, zips, zd
    meta.to_csv(os.path.join(rd, "meta.csv"), index=False)

    # Model inputs: as-of features (ASOS grid is already causal). EPA/HMS: PM2.5 shifted at S2; HMS daily
    # product is issued during the day, so the model sees the previous day's analysis.
    smoke_prev = np.concatenate([np.zeros((24, N), np.float32), feat["smoke_hms"][:-24]])
    X = np.stack([feat["T_F"] - thr[None], feat["T_C"], feat["RH"], np.log1p(np.clip(feat["precip_mm"], 0, None)),
                  np.clip(feat["vis_mi"], 0, 10), feat["wind_kt"], np.log1p(np.clip(feat["pm25"], 0, None)), smoke_prev], -1)
    X = np.nan_to_num(X, nan=np.nanmean(X))
    trh = hours < pd.Timestamp("2020-03-15")
    mu, sd = X[trh].mean((0, 1)), X[trh].std((0, 1)) + 1e-6
    np.save(os.path.join(rd, "wx_hourly.npy"), ((X - mu) / sd).astype(np.float16))
    json.dump(WX_NAMES, open(os.path.join(rd, "wx_names.json"), "w"))
    json.dump(dict(mean=mu.tolist(), std=sd.tolist()), open(os.path.join(rd, "wx_norm.json"), "w"))

    # True-time raw values for labels: ASOS as-of value at H+1 describes hour [H, H+1); PM2.5 unshift.
    def lead(a, k=1):
        return np.concatenate([a[k:], a[-k:]])
    pm_true = idw_apply(np.concatenate([pm[1:], pm[-1:]]), Wp)
    R = np.stack([lead(feat["T_F"]), lead(feat["T_C"]), lead(feat["RH"]), lead(feat["precip_mm"]), lead(feat["vis_mi"]),
                  lead(feat["wind_kt"]), pm_true, feat["smoke_hms"], np.broadcast_to(thr[None], (H, N))], -1).astype(np.float32)
    np.save(os.path.join(rd, "wx_hourly_raw.npy"), R)

    # ---------------- event labels at 5-min resolution
    hi = np.asarray((time.floor("h") - hours[0]) // pd.Timedelta("1h"))
    ev = np.zeros((len(time), N), np.uint8)
    Tday = pd.DataFrame(R[:, :, 0], index=hours).resample("D").max()          # daily Tmax F per node
    heat_day = (Tday.values > thr[None]).astype(np.uint8)
    summer = (Tday.index.month >= 6) & (Tday.index.month <= 9) & (Tday.index < pd.Timestamp("2020-03-15"))
    p95 = np.nanpercentile(Tday.values[summer], 95, axis=0)
    heat95 = (Tday.values > p95[None]).astype(np.uint8)
    di = np.asarray((time.normalize() - Tday.index[0]).days)
    ev |= heat_day[di] << BITS["heat"]
    ev |= heat95[di] << BITS["heat95"]
    pr = R[:, :, 3][hi]
    ev |= (pr >= 2.5).astype(np.uint8) << BITS["rain"]
    ev |= ((pr >= 0.25) & (pr < 2.5)).astype(np.uint8) << BITS["lightrain"]
    smk = (R[:, :, 6][hi] >= 55.5) | ((R[:, :, 7][hi] >= 2) & (R[:, :, 6][hi] >= 35.5))
    ev |= smk.astype(np.uint8) << BITS["smoke"]
    inc, n_inc = incident_mask(meta, time)
    ev |= inc.astype(np.uint8) << BITS["incident"]
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(time[0], time[-1])
    ev |= (np.isin(time.normalize(), hol).astype(np.uint8)[:, None] << BITS["holiday"])
    np.save(os.path.join(rd, "events.npy"), ev)

    # ---------------- capacity factor f(w), known at the forecast origin (as-of precip / visibility)
    p_asof, v_asof = feat["precip_mm"], feat["vis_mi"]
    f = np.ones((H, N), np.float32)
    f = np.where(p_asof >= 6.4, CAF["heavy"], np.where(p_asof >= 2.5, CAF["moderate"], np.where(p_asof >= 0.25, CAF["light"], 1.0)))
    f = f * np.where(v_asof < 0.5, CAF["lowvis"], 1.0)
    np.save(os.path.join(rd, "cap_factor_hourly.npy"), f.astype(np.float32))

    rep = {k: float(((ev >> b) & 1).mean()) for k, b in BITS.items()}
    rep["incident_matches"] = int(n_inc)
    rep["thr_F_median"] = float(np.median(thr))
    json.dump(rep, open(os.path.join(rd, "event_report.json"), "w"), indent=1)
    print(region, rep, flush=True)


if __name__ == "__main__":
    import sys
    prep_accidents()
    for r in sys.argv[1:]:
        main(r)
