"""S2: station-level hourly weather grids (local Pacific wall-clock time, 2017-2021).

ASOS: temperature, dew point, RH, hourly precipitation, visibility, wind speed.
EPA AQS: hourly PM2.5 (parameters 88101 + 88502, GMT stamps).
HMS: daily smoke-density polygons are handled in s4 (sensor-level spatial join).

Values on the hourly grid are causal "as-of" values: the grid value at hour H is
the latest observation made at or before H:00 (within 2 h), so nothing after the
forecast origin leaks into the inputs.
"""
import os, glob, zipfile, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RAW = os.path.join(ROOT, "data_raw")
OUT = os.path.join(ROOT, "data_proc", "weather")
os.makedirs(OUT, exist_ok=True)
GRID = pd.date_range("2017-01-01 00:00", "2021-12-31 23:00", freq="h")
TZ = "America/Los_Angeles"


def to_local_naive(utc_series):
    return pd.to_datetime(utc_series, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)


def asof_grid(df, cols, tol="2h"):
    df = df.sort_values("t").drop_duplicates("t", keep="last")
    g = pd.DataFrame({"t": GRID})
    return pd.merge_asof(g, df[["t"] + cols], on="t", direction="backward", tolerance=pd.Timedelta(tol))


def asos():
    st = pd.read_csv(os.path.join(RAW, "asos", "stations_sel.csv"))
    feats = ["tmpf", "dwpf", "relh", "p01i", "vsby", "sknt"]
    arrs, keep = [], []
    for sid in st.sid:
        fn = os.path.join(RAW, "asos", f"{sid}.csv")
        if not os.path.exists(fn) or os.path.getsize(fn) < 1000:
            continue
        d = pd.read_csv(fn, na_values=["M"], low_memory=False)
        if len(d) < 5000:
            continue
        d["t"] = to_local_naive(d["valid"])
        for c in feats:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        # p01i is the accumulation since the last routine (:5x) report. Hourly total =
        # the routine report's value; specials only update the running total.
        d["p01i"] = d["p01i"].where(d["p01i"] >= 0)
        g = asof_grid(d, [c for c in feats if c != "p01i"])
        routine = d[d["t"].dt.minute >= 45]
        g["p01i"] = asof_grid(routine, ["p01i"], tol="70min")["p01i"].to_numpy()
        cov = g["tmpf"].notna().mean()
        if cov < 0.6:
            print("drop", sid, cov)
            continue
        a = g[feats].to_numpy(np.float32)
        a[:, 0] = (a[:, 0] - 32) / 1.8          # degC (keep F copy for heat threshold below)
        a[:, 1] = (a[:, 1] - 32) / 1.8
        a[:, 3] = a[:, 3] * 25.4                # mm/h
        arrs.append(np.concatenate([a, g[["tmpf"]].to_numpy(np.float32)], 1))
        keep.append(sid)
        print(sid, f"coverage={cov:.3f}", flush=True)
    A = np.stack(arrs, 1)  # [H, S, 7]
    np.save(os.path.join(OUT, "asos_hourly.npy"), A)
    st[st.sid.isin(keep)].set_index("sid").loc[keep].reset_index().to_csv(os.path.join(OUT, "asos_stations.csv"), index=False)
    print("ASOS", A.shape, "features: T_C, Td_C, RH, precip_mm, vis_mi, wind_kt, T_F")


def pm25():
    frames = []
    for fn in sorted(glob.glob(os.path.join(RAW, "epa", "hourly_*.zip"))):
        with zipfile.ZipFile(fn) as z:
            name = z.namelist()[0]
            it = pd.read_csv(z.open(name), usecols=["State Code", "County Code", "Site Num", "Latitude", "Longitude",
                                                     "Date GMT", "Time GMT", "Sample Measurement"], chunksize=2_000_000)
            for ch in it:
                ch = ch[ch["State Code"].astype(str).isin(["6", "06"])]
                frames.append(ch)
        print(fn, flush=True)
    d = pd.concat(frames)
    d["site"] = d["County Code"].astype(int).astype(str) + "-" + d["Site Num"].astype(int).astype(str)
    d["t"] = to_local_naive(pd.to_datetime(d["Date GMT"] + " " + d["Time GMT"]).dt.tz_localize("UTC"))
    d = d[d["Sample Measurement"] > -5]
    sites = d.groupby("site")[["Latitude", "Longitude"]].first()
    piv = d.groupby(["t", "site"])["Sample Measurement"].mean().unstack()
    # EPA hourly values are period-beginning; the hour [H, H+1) is known at H+1.
    piv.index = piv.index + pd.Timedelta("1h")
    piv = piv.reindex(GRID)
    cov = piv.notna().mean()
    piv = piv.loc[:, cov > 0.3]
    piv = piv.ffill(limit=2)
    np.save(os.path.join(OUT, "pm25_hourly.npy"), piv.to_numpy(np.float32))
    sites.loc[piv.columns].reset_index().to_csv(os.path.join(OUT, "pm25_sites.csv"), index=False)
    print("PM2.5", piv.shape)


if __name__ == "__main__":
    import sys
    np.save(os.path.join(OUT, "grid_hours.npy"), GRID.values.astype("datetime64[m]"))
    if "asos" in sys.argv:
        asos()
    if "pm25" in sys.argv:
        pm25()
