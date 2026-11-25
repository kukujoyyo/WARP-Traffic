"""E1: empirical distribution shift of freeway flow under extreme heat, heavy rain, wildfire smoke and crashes.

A. matched-day design (event day vs same-weekday non-event days within +/-4 weeks), Wasserstein/KS
   against a normal-vs-normal placebo; volume, peak and volatility changes.
B. fixed-effects regression of hourly log-flow on heat anomaly, rain intensity and PM2.5 bins.
C. discharge-capacity proxy under rain / heat / smoke (high quantiles of peak-period flow).
D. speed-flow (FD) shift under rain on PEMS-BAY sensors (paired LargeST flow + PEMS-BAY speed, Jan-Jun 2017).
"""
import os, json, numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
PROC = os.path.join(ROOT, "data_proc", "regions")
OUT = os.path.join(ROOT, "results", "E1")
os.makedirs(OUT, exist_ok=True)
LOCK = (pd.Timestamp("2020-03-15"), pd.Timestamp("2020-06-30"))
B = {"heat": 0, "rain": 1, "smoke": 2, "incident": 3, "lightrain": 4, "holiday": 6}


def load(region):
    d = os.path.join(PROC, region)
    flow = np.load(os.path.join(d, "flow.npy"))
    time = pd.DatetimeIndex(np.load(os.path.join(d, "time.npy")))
    ev = np.load(os.path.join(d, "events.npy"))
    raw = np.load(os.path.join(d, "wx_hourly_raw.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(d, "meta.csv"))
    return flow, time, ev, raw, meta


def day_tables(flow, time, ev):
    nd = len(time) // 288
    F = flow[:nd * 288].reshape(nd, 288, -1)
    E = ev[:nd * 288].reshape(nd, 288, -1)
    days = time[::288][:nd]
    bit = lambda k: ((E >> B[k]) & 1)
    share = {k: bit(k).mean(1).mean(1) for k in ["heat", "smoke", "rain", "lightrain", "incident"]}   # sensor-time share
    hol = bit("holiday")[:, 0, 0] > 0
    return F, days, share, hol, E


def matched_design(region, flow, time, ev):
    F, days, share, hol, E = day_tables(flow, time, ev)
    lock = (days >= LOCK[0]) & (days <= LOCK[1])
    any_rain = (((E >> B["rain"]) & 1) | ((E >> B["lightrain"]) & 1)).mean(2).max(1) >= 0.1   # >=10% sensors wet in some 5-min
    heat_sensor_share = ((E >> B["heat"]) & 1)[:, 0].mean(1)
    smoke_hours = ((E >> B["smoke"]) & 1).mean(2)                                               # [D,288] share of sensors
    smoke_day = (smoke_hours >= 0.5).sum(1) >= 72                                              # >= 6 h on half the sensors
    heat_day = heat_sensor_share >= 0.5
    normal = ~heat_day & ~smoke_day & ~any_rain & ~hol & ~lock & (heat_sensor_share == 0) & ((smoke_hours >= 0.2).sum(1) < 12)
    events = {"heat": heat_day & ~hol & ~lock & ~any_rain, "smoke": smoke_day & ~hol & ~lock & ~heat_day & ~any_rain}
    rows, profiles = [], {}
    for et, mask in events.items():
        for i in np.where(mask)[0]:
            cand = [j for j in range(max(0, i - 28), min(len(days), i + 29)) if normal[j] and days[j].dayofweek == days[i].dayofweek and j != i]
            if len(cand) < 3:
                continue
            cand = cand[:5] if len(cand) <= 5 else sorted(cand, key=lambda j: abs(j - i))[:5]
            Fe, Fn = F[i], F[cand]                                       # [288,N], [k,288,N]
            ok = ~np.isnan(Fe).any(0) & ~np.isnan(Fn).any((0, 1))
            if ok.sum() < 10:
                continue
            Fe, Fn = Fe[:, ok], Fn[:, :, ok]
            base = Fn.mean(0)
            vol = (Fe.sum(0) - base.sum(0)) / base.sum(0)
            am = (Fe[72:108].sum(0) - base[72:108].sum(0)) / base[72:108].sum(0)
            mid = (Fe[132:180].sum(0) - base[132:180].sum(0)) / base[132:180].sum(0)
            pm = (Fe[180:228].sum(0) - base[180:228].sum(0)) / base[180:228].sum(0)
            vola = lambda x: np.nanstd(np.diff(x, axis=-2), axis=-2) / np.nanmean(x, axis=-2)
            dv = (vola(Fe) - vola(Fn).mean(0)) / vola(Fn).mean(0)
            w_e = np.array([stats.wasserstein_distance(Fe[:, s], Fn[:, :, s].ravel()) for s in range(Fe.shape[1])]) / base.mean(0)
            ks_e = np.array([stats.ks_2samp(Fe[:, s], Fn[:, :, s].ravel()).statistic for s in range(Fe.shape[1])])
            # placebo: one matched normal day against the remaining matched days
            w_p = np.array([stats.wasserstein_distance(Fn[0, :, s], Fn[1:, :, s].ravel()) for s in range(Fe.shape[1])]) / base.mean(0)
            ks_p = np.array([stats.ks_2samp(Fn[0, :, s], Fn[1:, :, s].ravel()).statistic for s in range(Fe.shape[1])])
            rows.append(dict(region=region, event=et, date=str(days[i].date()), n_match=len(cand), n_sensors=int(ok.sum()),
                             dvol=np.median(vol), dam=np.median(am), dmid=np.median(mid), dpm=np.median(pm), dvola=np.median(dv),
                             W1=np.median(w_e), W1_placebo=np.median(w_p), KS=np.median(ks_e), KS_placebo=np.median(ks_p)))
            profiles.setdefault(et, []).append((Fe.mean(1), base.mean(1)))
    df = pd.DataFrame(rows)
    # hour-level design for heavy rain and crashes (sensor-time cells vs same sensor, same time-of-week, dry/no-crash)
    E5 = ev
    H = []
    for et in ["rain", "incident"]:
        cells = np.argwhere(((E5 >> B[et]) & 1) > 0)
        if len(cells) > 60000:
            cells = cells[np.random.default_rng(0).choice(len(cells), 60000, replace=False)]
        d = []
        for t, s in cells:
            lags = [t + k * 2016 for k in (-4, -3, -2, -1, 1, 2, 3, 4) if 0 <= t + k * 2016 < len(time)]
            lags = [l for l in lags if ev[l, s] & ((1 << B["rain"]) | (1 << B["lightrain"]) | (1 << B["incident"]) | (1 << B["holiday"])) == 0
                    and not (LOCK[0] <= time[l] <= LOCK[1])]
            if len(lags) < 2 or np.isnan(flow[t, s]):
                continue
            ref = np.nanmean(flow[lags, s])
            if ref > 20:
                d.append(((flow[t, s] - ref) / ref, time[t].hour, abs(flow[t, s] - ref) / ref))
        d = np.array(d)
        H.append(dict(region=region, event=et, n_cells=len(d), dflow_median=float(np.median(d[:, 0])), dflow_mean=float(np.mean(d[:, 0])),
                      abs_dev_median=float(np.median(d[:, 2])),
                      dflow_peak=float(np.median(d[(d[:, 1] >= 6) & (d[:, 1] < 10) | (d[:, 1] >= 15) & (d[:, 1] < 19), 0]))))
    # normal-normal placebo for hour-level design
    rng = np.random.default_rng(1)
    d = []
    tt = rng.integers(8 * 2016, len(time) - 8 * 2016, 60000)
    ss = rng.integers(0, flow.shape[1], 60000)
    for t, s in zip(tt, ss):
        if ev[t, s] != 0 or LOCK[0] <= time[t] <= LOCK[1]:
            continue
        lags = [t + k * 2016 for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        lags = [l for l in lags if ev[l, s] == 0 and not (LOCK[0] <= time[l] <= LOCK[1])]
        if len(lags) < 2 or np.isnan(flow[t, s]):
            continue
        ref = np.nanmean(flow[lags, s])
        if ref > 20:
            d.append(((flow[t, s] - ref) / ref, abs(flow[t, s] - ref) / ref))
    d = np.array(d)
    H.append(dict(region=region, event="placebo", n_cells=len(d), dflow_median=float(np.median(d[:, 0])), dflow_mean=float(np.mean(d[:, 0])),
                  abs_dev_median=float(np.median(d[:, 1])), dflow_peak=np.nan))
    return df, pd.DataFrame(H), profiles


def fe_regression(region, flow, time, ev, raw, n_sensors=120, seed=0):
    """log hourly flow ~ weather bins | sensor x hour-of-week x year FE + week-of-year FE; cluster SE by date."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    S = rng.choice(flow.shape[1], min(n_sensors, flow.shape[1]), replace=False)
    nh = len(time) // 12
    q = np.nansum(flow[:nh * 12, S].reshape(nh, 12, -1), 1)
    okc = (~np.isnan(flow[:nh * 12, S])).reshape(nh, 12, -1).all(1)
    hours = time[::12][:nh]
    R = np.asarray(raw[:nh])[:, S]                                        # true-hour raw values
    dT = R[:, :, 0] - R[:, :, 8]
    p = R[:, :, 3]
    pm = R[:, :, 6]
    df = pd.DataFrame(dict(lq=np.log(np.clip(q, 1, None)).ravel(), ok=okc.ravel() & (q.ravel() > 50),
                           sensor=np.repeat(S[None], nh, 0).ravel(), how=np.repeat((hours.dayofweek * 24 + hours.hour).values[:, None], len(S), 1).ravel(),
                           year=np.repeat(hours.year.values[:, None], len(S), 1).ravel(),
                           woy=np.repeat(hours.isocalendar().week.values.astype(int)[:, None], len(S), 1).ravel(),
                           date=np.repeat(hours.normalize().values[:, None], len(S), 1).ravel(),
                           dT=dT.ravel(), p=p.ravel(), pm=pm.ravel()))
    lock = (df.date >= LOCK[0]) & (df.date <= LOCK[1])
    hol = np.repeat((((ev[:nh * 12:12, 0] >> B["holiday"]) & 1) > 0)[:, None], len(S), 1).ravel()
    df = df[df.ok & ~lock & ~hol].dropna()
    bins = {"heat_0_5F": (df.dT > 0) & (df.dT <= 5), "heat_gt5F": df.dT > 5, "rain_0.25_2.5": (df.p >= 0.25) & (df.p < 2.5),
            "rain_2.5_7.6": (df.p >= 2.5) & (df.p < 7.6), "rain_ge7.6": df.p >= 7.6,
            "pm_35_55": (df.pm >= 35.5) & (df.pm < 55.5), "pm_55_150": (df.pm >= 55.5) & (df.pm < 150.5), "pm_ge150": df.pm >= 150.5}
    X = pd.DataFrame({k: v.astype(float) for k, v in bins.items()})
    # absorb FE by sequential demeaning (alternating projections)
    y = df.lq.copy()
    g1 = df.sensor.astype(str) + "_" + df.how.astype(str) + "_" + df.year.astype(str)
    g2 = df.woy.astype(str) + "_" + df.year.astype(str)
    Z = pd.concat([y.rename("y"), X], axis=1)
    for _ in range(8):
        Z = Z - Z.groupby(g1.values).transform("mean")
        Z = Z - Z.groupby(g2.values).transform("mean")
    res = sm.OLS(Z["y"], Z[list(bins)]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.date)[0]})
    out = pd.DataFrame(dict(coef=res.params, se=res.bse, p=res.pvalues, n_obs_bin=X.sum()))
    out["pct_effect"] = 100 * (np.exp(out.coef) - 1)
    out["region"] = region
    return out.reset_index().rename(columns={"index": "bin"})


def capacity_proxy(region, flow, time, ev, meta):
    """99th pct of weekday peak-period 5-min flow under each condition vs dry/no-event, per sensor (>=30 cells)."""
    pk = ((time.hour >= 6) & (time.hour < 10)) | ((time.hour >= 15) & (time.hour < 19))
    wk = time.dayofweek < 5
    lock = (time >= LOCK[0]) & (time <= LOCK[1])
    base_sel = pk & wk & ~lock
    rows = []
    for et in ["rain", "lightrain", "heat", "smoke"]:
        r = []
        for s in range(flow.shape[1]):
            e = ev[:, s]
            on = base_sel & (((e >> B[et]) & 1) > 0)
            off = base_sel & (e == 0)
            if on.sum() < 30 or off.sum() < 1000:
                continue
            a, b = np.nanpercentile(flow[on, s], 99), np.nanpercentile(flow[off, s], 99)
            if b > 50:
                r.append(a / b)
        r = np.array(r)
        if len(r) == 0:
            continue
        rows.append(dict(region=region, event=et, n_sensors=len(r), ratio_median=float(np.median(r)),
                         ratio_iqr_lo=float(np.percentile(r, 25)), ratio_iqr_hi=float(np.percentile(r, 75)),
                         reduction_pct=float(100 * (1 - np.median(r))), wilcoxon_p=float(stats.wilcoxon(r - 1).pvalue) if len(r) > 10 else np.nan))
    return pd.DataFrame(rows)


def pemsbay_fd():
    """Speed-flow relation under rain vs dry for PEMS-BAY sensors (2017-01-01..2017-06-30)."""
    pb = os.path.join(ROOT, "data_raw", "pemsbay")
    sp = pd.concat([pd.read_parquet(os.path.join(pb, f), columns=["node_id", "t0_timestamp", "x_t+0_d0"]) for f in ["train.parquet", "test.parquet"]])
    loc = pd.read_csv(os.path.join(pb, "sensor_locations.csv"))
    sp["t"] = pd.to_datetime(sp.t0_timestamp)
    V = sp.pivot_table(index="t", columns="node_id", values="x_t+0_d0")
    V.columns = loc.sensor_id.values[V.columns]
    meta = pd.read_csv(os.path.join(PROC, "D4", "meta.csv"))
    flow = np.load(os.path.join(PROC, "D4", "flow.npy"), mmap_mode="r")
    time = pd.DatetimeIndex(np.load(os.path.join(PROC, "D4", "time.npy")))
    ev = np.load(os.path.join(PROC, "D4", "events.npy"), mmap_mode="r")
    raw = np.load(os.path.join(PROC, "D4", "wx_hourly_raw.npy"), mmap_mode="r")
    ids = [i for i in V.columns if i in set(meta.ID)]
    col = {i: int(np.where(meta.ID.values == i)[0][0]) for i in ids}
    tsel = (time >= V.index.min()) & (time <= V.index.max())
    ti = np.where(tsel)[0]
    Q = pd.DataFrame(np.asarray(flow[ti][:, [col[i] for i in ids]]), index=time[ti], columns=ids)
    E = pd.DataFrame(np.asarray(ev[ti][:, [col[i] for i in ids]]), index=time[ti], columns=ids)
    V = V.reindex(index=Q.index)[ids]
    rows, pairs = [], {}
    for i in ids:
        q, v, e = Q[i].values, V[i].values, E[i].values
        lanes = max(int(meta.Lanes.values[col[i]]), 1)
        ok = ~np.isnan(q) & ~np.isnan(v) & (v > 0)
        rain = ok & ((((e >> B["rain"]) & 1) > 0) | (((e >> B["lightrain"]) & 1) > 0))
        heavy = ok & (((e >> B["rain"]) & 1) > 0)
        dry = ok & (e & ((1 << B["rain"]) | (1 << B["lightrain"]) | (1 << B["incident"]))) == 0
        if rain.sum() < 100:
            continue
        qh = q * 12 / lanes                                               # veh/h/lane
        k = qh / v                                                        # veh/mi/lane
        def fd(m):
            ff = m & (k < 15)                                             # uncongested branch
            vf = np.nanpercentile(v[ff], 50) if ff.sum() > 20 else np.nan
            cap = np.nanpercentile(qh[m], 99)
            kc = cap / np.nanpercentile(v[m & (qh >= 0.9 * cap)], 50) if (m & (qh >= 0.9 * cap)).sum() > 5 else np.nan
            cong = m & (v < 45)
            return vf, cap, kc, cong.mean()
        d, r = fd(dry), fd(rain)
        h = fd(heavy) if heavy.sum() >= 30 else (np.nan,) * 4
        rows.append(dict(sensor=i, lanes=lanes, n_rain=int(rain.sum()), n_heavy=int(heavy.sum()), vf_dry=d[0], vf_rain=r[0], vf_heavy=h[0],
                         cap_dry=d[1], cap_rain=r[1], cap_heavy=h[1], kc_dry=d[2], kc_rain=r[2], cong_dry=d[3], cong_rain=r[3], cong_heavy=h[3]))
        pairs[i] = (qh[dry], v[dry], qh[rain], v[rain])
    return pd.DataFrame(rows), pairs


if __name__ == "__main__":
    import sys
    regions = sys.argv[1:] or ["D3", "D4", "D7"]
    A, Hh, Reg, Cap, prof = [], [], [], [], {}
    for r in regions:
        flow, time, ev, raw, meta = load(r)
        df, H, profiles = matched_design(r, flow, time, ev)
        A.append(df); Hh.append(H); prof[r] = profiles
        Cap.append(capacity_proxy(r, flow, time, ev, meta))
        Reg.append(fe_regression(r, flow, time, ev, raw))
        print(r, "done", flush=True)
    pd.concat(A).to_csv(os.path.join(OUT, "matched_days.csv"), index=False)
    pd.concat(Hh).to_csv(os.path.join(OUT, "hour_level_shift.csv"), index=False)
    pd.concat(Reg).to_csv(os.path.join(OUT, "fe_regression.csv"), index=False)
    pd.concat(Cap).to_csv(os.path.join(OUT, "capacity_proxy.csv"), index=False)
    np.save(os.path.join(OUT, "profiles.npy"), {r: {k: [(a.tolist(), b.tolist()) for a, b in v] for k, v in p.items()} for r, p in prof.items()}, allow_pickle=True)
    fd, pairs = pemsbay_fd()
    fd.to_csv(os.path.join(OUT, "pemsbay_fd.csv"), index=False)
    np.save(os.path.join(OUT, "pemsbay_pairs.npy"), {k: [x.tolist() for x in v] for k, v in list(pairs.items())[:12]}, allow_pickle=True)
    # summary
    m = pd.concat(A)
    summ = m.groupby(["region", "event"]).agg(n_days=("date", "count"), dvol=("dvol", "median"), dam=("dam", "median"), dmid=("dmid", "median"),
                                              dpm=("dpm", "median"), dvola=("dvola", "median"), W1=("W1", "median"),
                                              W1_placebo=("W1_placebo", "median"), KS=("KS", "median"), KS_placebo=("KS_placebo", "median"))
    pv = m.groupby(["region", "event"]).apply(lambda g: stats.wilcoxon(g.W1 - g.W1_placebo).pvalue if len(g) > 5 else np.nan)
    summ["W1_vs_placebo_p"] = pv
    summ.to_csv(os.path.join(OUT, "matched_summary.csv"))
    print(summ.to_string())
    print(pd.concat(Hh).to_string())
    print(pd.concat(Cap).to_string())
    print(pd.concat(Reg).to_string())
    print(fd.describe().T[["mean", "50%"]].to_string())
