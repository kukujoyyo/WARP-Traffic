"""Aggregate summaries over seeds; event-stratified tables; Diebold-Mariano tests (Holm-corrected)."""
import os, re, json, glob, numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
NAMES = {"ha": "HA", "snaive": "Seasonal naive", "climatology": "Climatology", "gwnet": "GWNet", "stid": "STID",
         "staeformer": "STAEformer", "gwnet+wx": "GWNet+W", "staeformer+wx": "STAEformer+W", "stid+wx": "STID+W",
         "gwnet_gauss": "GWNet-Gaussian", "gwnet_quantile": "GWNet-Quantile", "gwnet_mc": "GWNet-MCDropout",
         "staeformer_quantile": "STAEformer-Quantile", "gwnet_gauss+wx": "GWNet-Gaussian+W", "deepar": "DeepAR",
         "staeformer_conformal": "STAEformer-Conformal", "gwnet_conformal": "GWNet-Conformal",
         "remind": "REMIND", "remind_wx": "REMIND+W (concat)", "wremind": "WARP-Traffic", "remind_orig": "REMIND (orig. encoder)", "wremind_orig": "WARP-Traffic (orig. encoder)",
         "wremindpool": "WARP-Traffic", "remindpool": r"REMIND $\oplus$ GWNet-G", "gwens": "GWNet-G deep ensemble",
         "gwwxpool": r"GWNet-G $\oplus$ GWNet-G+W"}
NAMES["wremind"] = "WARP-Traffic memory expert"
STRATA = ["all", "normal", "heat", "rain", "smoke", "incident"]


def collect(region, suffix=""):
    rows = []
    for d in sorted(glob.glob(os.path.join(RES, region, "*_s[0-9]"))):
        base = os.path.basename(d)
        m = re.match(r"(.+)_s(\d+)$", base)
        model, seed = m.group(1), int(m.group(2))
        for kind, fn in [("", f"summary{suffix}.json"), ("conformal", f"summary_conformal{suffix}.json")]:
            f = os.path.join(d, fn)
            if not os.path.exists(f):
                continue
            s = json.load(open(f))
            name = model if not kind else model.split("+")[0] + "_conformal" + ("+wx" if "+wx" in model else "")
            for st in STRATA + ["anyevent", "lightrain", "heat95"]:
                if st not in s:
                    continue
                for hz, r in s[st].items():
                    if not isinstance(r, dict):
                        continue
                    rows.append(dict(region=region, model=name, seed=seed, stratum=st, horizon=hz, ECE=s[st].get("ECE", np.nan), **r))
            if not kind:
                rows.append(dict(region=region, model=name, seed=seed, stratum="ES", horizon="avg", CRPS=s["ES"]["all"], MAE=s["ES"]["regionevent"]))
    return pd.DataFrame(rows)


def dm_test(a, b, lag=12):
    """Two-sided DM test on loss differentials d=a-b (Newey-West variance)."""
    d = a - b
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def origin_loss(region, model, stratum, metric=0, seeds=None, conformal=False):
    fs = sorted(glob.glob(os.path.join(RES, region, f"{model}_s[0-9]", "series_conformal.npz" if conformal else "series.npz")))
    if seeds is not None:
        fs = [f for f in fs if int(f.split("_s")[-1][0]) in seeds]
    if not fs:
        return None
    L = []
    for f in fs:
        z = np.load(f)
        s = z[stratum]
        num, cnt = s[..., metric].sum(1), s[..., 2].sum(1)
        L.append(np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan))
    return np.nanmean(np.stack(L), 0)


def cell_mean(region, model, stratum, seeds, metric=0):
    """Cell-weighted mean score, the same aggregation as the main tables."""
    fs = sorted(glob.glob(os.path.join(RES, region, f"{model}_s[0-9]", "series.npz")))
    fs = [f for f in fs if int(f.split("_s")[-1][0]) in seeds]
    num = cnt = 0.0
    for f in fs:
        s = np.load(f)[stratum]
        num += s[..., metric].sum()
        cnt += s[..., 2].sum()
    return num / cnt


def run_seeds(region, model):
    return {int(f.split("_s")[-1][0]) for f in glob.glob(os.path.join(RES, region, f"{model}_s[0-9]", "series.npz"))}


def dm_table(region, ref="wremindpool", others=("gwnet_gauss", "gwnet_gauss+wx", "gwens", "gwwxpool", "remindpool", "remind", "wremind",
                                                 "staeformer_quantile", "deepar")):
    rows = []
    for st in ["all", "normal", "anyevent", "heat", "rain", "smoke", "incident"]:
        if origin_loss(region, ref, st) is None:
            continue
        for o in others:
            # compare on the seeds both models have, so multi-seed and seed-0-only runs are tested like for like
            sd = run_seeds(region, ref) & run_seeds(region, o)
            if not sd:
                continue
            a, b = origin_loss(region, ref, st, seeds=sd), origin_loss(region, o, st, seeds=sd)
            stat, p = dm_test(a, b)
            rows.append(dict(region=region, stratum=st, vs=o, dm=stat, p=p, rel_crps=cell_mean(region, ref, st, sd) / cell_mean(region, o, st, sd) - 1,
                             rel_crps_origin=np.nanmean(a) / np.nanmean(b) - 1))
    df = pd.DataFrame(rows)
    if len(df):
        # Holm correction within region
        pv = df.p.fillna(1).values
        order = np.argsort(pv)
        adj = np.empty_like(pv)
        run = 0
        for k, i in enumerate(order):
            run = max(run, (len(pv) - k) * pv[i])
            adj[i] = min(1, run)
        df["p_holm"] = adj
    return df


def table(df, metric="CRPS", horizon="avg", strata=STRATA):
    g = df[(df.horizon == horizon) & df.stratum.isin(strata)].groupby(["model", "stratum"])[metric]
    m, s, n = g.mean().unstack(), g.std().unstack(), g.count().unstack()
    return m[strata], s[strata], n[strata]


if __name__ == "__main__":
    import sys
    for region in sys.argv[1:] or ["D3"]:
        df = collect(region)
        df.to_csv(os.path.join(RES, f"all_{region}.csv"), index=False)
        for met in ["CRPS", "MAE", "PICP90", "Winkler90", "Violation"]:
            if met not in df:
                continue
            m, s, n = table(df, met)
            print(f"\n== {region} {met} (avg over horizons) ==")
            print(m.round(3).to_string())
        dm = dm_table(region)
        dm.to_csv(os.path.join(RES, f"dm_{region}.csv"), index=False)
        print(dm.round(4).to_string())
