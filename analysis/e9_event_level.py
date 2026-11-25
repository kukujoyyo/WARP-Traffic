"""E9: event-level inference and calibration by event type and horizon (reviewer items #3 and #4).

Event level (#4). Cells within one storm or heat wave are strongly dependent, so cell-level tests overstate the
evidence. Origins (every 30 min) at which a stratum is active anywhere in the network are grouped into episodes;
active origins less than 3 h apart belong to the same episode. For each episode we sum CRPS and cell counts of
WARP-Traffic and GWNet-Gaussian (summed over the seeds both have), and report
  - the aggregate CRPS reduction, with a 95% bootstrap interval that resamples episodes (5,000 draws),
  - the share of episodes in which WARP-Traffic is better and a Wilcoxon signed-rank test on per-episode differences,
  - the reduction after removing the largest episode (most cells), e.g. the October 2021 atmospheric river.

Calibration (#3). Coverage of the central 90% and 50% intervals and the mean 90% width by stratum and lead time
(15/30/60 min), with 95% intervals from a bootstrap that resamples whole days (moving-block, block = 1 day).

usage: python analysis/e9_event_level.py           -> results/E9/*.csv and paper/tab_event_level.tex, tab_calib.tex
"""
import os, json, numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data_proc", "regions")
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(RES, "E9")
STRATA = ["heat", "rain", "smoke", "incident"]
LABEL = {"heat": "Heat", "rain": "Heavy rain", "smoke": "Smoke", "incident": "Crash", "anyevent": "Any event", "all": "All", "normal": "Normal"}
SEEDS = {"D3": [0, 1, 2], "D4": [0, 1], "D7": [0, 1]}
HZ = {"15min": 2, "30min": 5, "60min": 11}
GAP = 6          # origins (30 min apart): 3 h
B = 5000
rng = np.random.default_rng(0)


def load(region, run):
    z = np.load(os.path.join(RES, region, run, "series.npz"))
    return {k: z[k] for k in z.files}


def episodes(active):
    idx = np.where(active)[0]
    lab = np.full(len(active), -1)
    if len(idx) == 0:
        return lab, 0
    e = np.concatenate([[0], np.cumsum(np.diff(idx) > GAP)])
    lab[idx] = e
    return lab, int(e[-1]) + 1


def event_level(region):
    rows = []
    a = [load(region, f"wremindpool_s{s}") for s in SEEDS[region]]
    g = [load(region, f"gwnet_gauss_s{s}") for s in SEEDS[region]]
    t = pd.DatetimeIndex(np.load(os.path.join(PROC, region, "time.npy")))[a[0]["t0"]]
    for k in STRATA + ["anyevent"]:
        n = a[0][k][..., 2].sum(1)                                   # cells per origin (same for every model)
        lab, E = episodes(n > 0)
        if E < 3:
            continue
        ca = sum(x[k][..., 0].sum(1) for x in a)                     # CRPS sum over horizons and seeds
        cg = sum(x[k][..., 0].sum(1) for x in g)
        m = lab >= 0
        A = np.bincount(lab[m], ca[m], E); G = np.bincount(lab[m], cg[m], E); N = np.bincount(lab[m], n[m], E)
        gain = 1 - A.sum() / G.sum()
        bs = np.empty(B)
        for i in range(B):
            j = rng.integers(0, E, E)
            bs[i] = 1 - A[j].sum() / G[j].sum()
        d = (A - G) / np.maximum(N, 1)                               # per-episode CRPS difference per cell
        w = stats.wilcoxon(d) if np.any(d != 0) else None
        big = int(np.argmax(N))
        keep = np.arange(E) != big
        start = t[np.where(lab == big)[0][0]]
        rows.append(dict(region=region, stratum=k, episodes=E, cells=int(N.sum()), gain=100 * gain,
                         lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                         share_better=float((d < 0).mean()), p_wilcoxon=float(w.pvalue) if w else np.nan,
                         largest_start=str(start), largest_cells_share=float(N[big] / N.sum()),
                         gain_wo_largest=100 * (1 - A[keep].sum() / G[keep].sum())))
    return pd.DataFrame(rows)


def calibration(region="D3", models=("wremindpool", "gwnet_gauss")):
    rows = []
    for mdl in models:
        zs = [load(region, f"{mdl}_s{s}") for s in SEEDS[region]]
        t = pd.DatetimeIndex(np.load(os.path.join(PROC, region, "time.npy")))[zs[0]["t0"]]
        day = np.asarray((t.normalize() - t.normalize()[0]).days)
        D = day.max() + 1
        for k in ["all", "normal"] + STRATA:
            for hz, h in HZ.items():
                n = sum(np.bincount(day, z[k][:, h, 2], D) for z in zs)
                c90 = sum(np.bincount(day, z[k + "_cal"][:, h, 0], D) for z in zs)
                wid = sum(np.bincount(day, z[k + "_cal"][:, h, 1], D) for z in zs)
                c50 = sum(np.bincount(day, z[k + "_cal"][:, h, 2], D) for z in zs)
                ok = n > 0
                n, c90, wid, c50 = n[ok], c90[ok], wid[ok], c50[ok]
                L = len(n)
                bs = np.empty((B // 5, 2))
                for i in range(len(bs)):
                    j = rng.integers(0, L, L)
                    bs[i] = c90[j].sum() / n[j].sum(), c50[j].sum() / n[j].sum()
                rows.append(dict(region=region, model=mdl, stratum=k, horizon=hz, days=L, cells=n.sum(),
                                 cov90=c90.sum() / n.sum(), cov90_lo=np.percentile(bs[:, 0], 2.5), cov90_hi=np.percentile(bs[:, 0], 97.5),
                                 cov50=c50.sum() / n.sum(), cov50_lo=np.percentile(bs[:, 1], 2.5), cov50_hi=np.percentile(bs[:, 1], 97.5),
                                 width90=wid.sum() / n.sum()))
    return pd.DataFrame(rows)


def tex_event(df):
    L = []
    for _, r in df.iterrows():
        p = r.p_wilcoxon
        ps = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        L.append(f"{r.region} & {LABEL[r.stratum]} & {r.episodes} & {r.gain:.1f} & [{r.lo:.1f}, {r.hi:.1f}] & "
                 f"{100 * r.share_better:.0f} & {ps} & {r.gain_wo_largest:.1f}\\\\")
    return "\n".join(L) + "\n"


def tex_calib(df):
    L = []
    for k in ["all", "normal"] + STRATA:
        cells = []
        for mdl in ["wremindpool", "gwnet_gauss"]:
            for hz in HZ:
                r = df[(df.model == mdl) & (df.stratum == k) & (df.horizon == hz)].iloc[0]
                hw = max(r.cov90 - r.cov90_lo, r.cov90_hi - r.cov90)
                cells.append(f"{r.cov90:.3f}\\,{{\\scriptsize$\\pm${hw:.3f}}}")
        L.append(f"{LABEL[k]} & " + " & ".join(cells) + "\\\\")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    ev = pd.concat([event_level(r) for r in SEEDS])
    ev.to_csv(os.path.join(OUT, "event_level.csv"), index=False)
    print(ev.round(3).to_string())
    open(os.path.join(PAPER, "tab_event_level.tex"), "w").write(tex_event(ev))
    cal = calibration("D3")
    cal.to_csv(os.path.join(OUT, "calibration_D3.csv"), index=False)
    print(cal.round(3).to_string())
    open(os.path.join(PAPER, "tab_calib.tex"), "w").write(tex_calib(cal))


if __name__ == "__main__":
    main()
