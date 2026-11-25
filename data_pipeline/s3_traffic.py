"""S3: traffic quality control, gap filling, graph and capacity estimates per district."""
import os, json, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RAW = os.path.join(ROOT, "data_raw", "largest")
TR = os.path.join(ROOT, "data_proc", "traffic")
OUT = os.path.join(ROOT, "data_proc", "regions")
YEARS = [2017, 2018, 2019, 2020, 2021]


def interp_short_gaps(x, max_gap=6):
    """Linear interpolation of NaN runs <= max_gap steps, column-wise."""
    df = pd.DataFrame(x)
    filled = df.interpolate(limit=max_gap, limit_area="inside")
    # undo fills inside runs longer than max_gap (pandas fills the first `limit` of a long run)
    isn = df.isna()
    out = filled.to_numpy(np.float32).copy()
    for j in range(x.shape[1]):
        m = isn[j].to_numpy()
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(s, e):
            if b - a > max_gap:
                out[a:b, j] = np.nan
    return out


def main():
    meta = pd.read_csv(os.path.join(RAW, "metadata.csv"))
    adj_all = np.load(os.path.join(RAW, "adjacency.npy"), mmap_mode="r")
    time = np.concatenate([np.load(os.path.join(TR, f"time_{y}.npy")) for y in YEARS])
    tidx = pd.DatetimeIndex(time)
    assert (np.diff(time).astype(int) == 5).all()
    report = {}
    for d in [3, 4, 7]:
        cols = np.load(os.path.join(TR, f"D{d}_cols.npy"))
        x = np.concatenate([np.load(os.path.join(TR, f"D{d}_{y}.npy")) for y in YEARS])
        n0 = x.shape[1]
        nan_all = np.isnan(x).mean(0)
        nan_test = np.isnan(x[tidx.year == 2021]).mean(0)
        nan_yr = np.stack([np.isnan(x[tidx.year == y]).mean(0) for y in YEARS])
        zero = (x == 0).mean(0)
        same = np.zeros(n0)
        for j in range(n0):          # share of steps inside >= 1 h constant runs (stuck detectors)
            v = x[:, j]
            eq = np.concatenate([[False], v[1:] == v[:-1]])
            dd = np.diff(np.concatenate([[0], eq.astype(np.int8), [0]]))
            s, e = np.where(dd == 1)[0], np.where(dd == -1)[0]
            L = e - s
            same[j] = L[L >= 12].sum() / len(v)
        lanes = meta.Lanes.values[cols]
        keep = (nan_all < 0.10) & (nan_yr.max(0) < 0.30) & (nan_test < 0.10) & (zero < 0.05) & (same < 0.02) & (lanes > 0)
        k = np.where(keep)[0]
        x = interp_short_gaps(x[:, k])
        gcols = cols[k]
        adj = np.asarray(adj_all[np.ix_(gcols, gcols)], dtype=np.float32)
        # capacity proxy: 99.9th pct of 5-min flow in the training period
        tr = (tidx < pd.Timestamp("2020-03-15"))
        cap = np.nanpercentile(x[tr], 99.9, axis=0).astype(np.float32)
        od = os.path.join(OUT, f"D{d}")
        os.makedirs(od, exist_ok=True)
        np.save(os.path.join(od, "flow.npy"), x)
        np.save(os.path.join(od, "time.npy"), time)
        np.save(os.path.join(od, "adj.npy"), adj)
        np.save(os.path.join(od, "capacity.npy"), cap)
        m = meta.iloc[gcols].reset_index(drop=True)
        m["col"] = gcols
        m["capacity_5min"] = cap
        m["cap_vphpl"] = cap * 12 / m.Lanes.clip(lower=1)
        m.to_csv(os.path.join(od, "meta.csv"), index=False)
        report[f"D{d}"] = dict(raw=int(n0), kept=int(len(k)), missing_after=float(np.isnan(x).mean()),
                               adj_density=float((adj > 0).mean()), cap_vphpl_median=float(m.cap_vphpl.median()))
        print(d, report[f"D{d}"], flush=True)
    json.dump(report, open(os.path.join(OUT, "qc_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
