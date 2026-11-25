"""E5: does the regime-aware gate explain *when* and *why* memory is trusted less?
- regime posterior pi vs weak labels over the forecast window (AUC, best-F1, crash F1 vs a deviation-only detector)
- mean trust weight g by stratum, for REMIND and WARP-Traffic
"""
import os, sys, json, glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data_proc", "regions")
REG = ["heat", "rain", "smoke", "incident"]
BITS = [0, 1, 2, 3]


def labels(region, t0, subset=None):
    ev = np.load(os.path.join(PROC, region, "events.npy"), mmap_mode="r")
    idx = t0[:, None] + np.arange(1, 13)[None]
    e = np.asarray(ev[idx.ravel()]).reshape(len(t0), 12, -1)
    if subset is not None:
        e = e[..., subset]
    return np.stack([((e >> b) & 1).max(1) for b in BITS], -1)          # [O, N, 4]


def best_f1(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return float(f.max())


def main(region, subset=None):
    sub = np.load(subset) if subset else None
    out = {"regime": [], "gate": []}
    for d in sorted(glob.glob(os.path.join(RES, region, "*_s[0-9]"))):
        f = os.path.join(d, "extra.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        if "gate" not in z:
            continue
        model = os.path.basename(d)
        t0 = z["t0"].astype(int)
        Y = labels(region, t0, sub)
        g = z["gate"].astype(np.float32)                                     # [O,N]
        anyev = Y.max(-1) > 0
        row = dict(model=model, g_normal=float(g[~anyev].mean()))
        for k, r in enumerate(REG):
            row[f"g_{r}"] = float(g[Y[..., k] > 0].mean()) if (Y[..., k] > 0).any() else np.nan
        if "gate_h" in z:                                                    # [O,3,N]: horizons 15/30/60 min
            gh = z["gate_h"].astype(np.float32)
            for j, h in enumerate(["15", "30", "60"]):
                row[f"gh{h}_normal"] = float(gh[:, j][~anyev].mean())
                row[f"gh{h}_event"] = float(gh[:, j][anyev].mean())
                for k, r in enumerate(REG):
                    row[f"gh{h}_{r}"] = float(gh[:, j][Y[..., k] > 0].mean()) if (Y[..., k] > 0).any() else np.nan
        out["gate"].append(row)
        if "pi" in z:
            P = z["pi"].astype(np.float32)
            for k, r in enumerate(REG):
                y, p = Y[..., k].ravel(), P[..., k].ravel()
                if y.sum() < 20:
                    continue
                rng = np.random.default_rng(0)
                sel = rng.choice(len(y), min(len(y), 3_000_000), replace=False)
                y, p = y[sel], p[sel]
                out["regime"].append(dict(model=model, regime=r, prevalence=float(y.mean()), auc=float(roc_auc_score(y, p)),
                                          ap=float(average_precision_score(y, p)), best_f1=best_f1(y, p)))
    for k, v in out.items():
        df = pd.DataFrame(v)
        df.to_csv(os.path.join(RES, f"E5_{k}_{region}.csv"), index=False)
        print(df.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
