"""E8: decision value of probabilistic forecasts in a cost-loss alerting setting (Richardson, 2000).

Two operational alerts per sensor and lead time (15/30/60 min):
  near-capacity : q >= 0.9 * C_i f(w)  -> pre-emptive metering / advisory deployment
  demand drop   : q <= 0.6 * HA(i, time-of-week), 06:00-21:00 -> disruption response (incident, closure, smoke)
For cost-loss ratio r = C/L, an alert is issued when P(event) > r. Value score V = (E_clim - E) / (E_clim - E_perfect).
"""
import os, sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data_proc", "regions")
RATIOS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def value_curve(p, o, ratios=RATIOS):
    ob = o.mean()
    out = []
    for r in ratios:
        act = p > r
        E = (act * r + (~act) * o).mean()
        Ec = min(r, ob)
        Ep = ob * r
        out.append((Ec - E) / (Ec - Ep) if Ec > Ep else np.nan)
    return np.array(out)


def load_events(region, t0, subset=None):
    ev = np.load(os.path.join(PROC, region, "events.npy"), mmap_mode="r")
    hs = np.array([3, 6, 12])
    idx = t0[:, None] + hs[None]
    e = np.asarray(ev[idx.ravel()]).reshape(len(t0), 3, -1)
    if subset is not None:
        e = e[..., subset]
    anyev = ((e & 0b1111) > 0)
    return anyev


def main(region, models, subset=None):
    rows, curves = [], {}
    sub = np.load(subset) if subset else None
    for m in models:
        f = os.path.join(RES, region, m, "extra.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ev = load_events(region, z["t0"].astype(int), sub)
        for alert in ["near", "drop"]:
            p, o, v = z[f"p{alert}"].astype(np.float32), z[f"o{alert}"], z["valid"]
            for hi, h in enumerate(["15min", "30min", "60min"]):
                for stratum, sm in [("all", np.ones_like(ev[:, hi], bool)), ("event", ev[:, hi]), ("normal", ~ev[:, hi])]:
                    mk = v[:, hi] & sm
                    pp, oo = p[:, hi][mk], o[:, hi][mk].astype(bool)
                    if oo.sum() < 20:
                        continue
                    vc = value_curve(pp, oo)
                    brier = float(((pp - oo) ** 2).mean())
                    try:
                        auc = float(roc_auc_score(oo, pp)) if len(np.unique(pp)) > 1 else 0.5
                    except ValueError:
                        auc = np.nan
                    rows.append(dict(region=region, model=m, alert=alert, horizon=h, stratum=stratum, base_rate=float(oo.mean()),
                                     brier=brier, auc=auc, V_mean=float(np.nanmean(np.clip(vc, -1, 1))), V_max=float(np.nanmax(vc)),
                                     V_r02=float(vc[3]), V_r05=float(vc[9]), n=int(mk.sum())))
                    curves[(m, alert, h, stratum)] = vc.tolist()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RES, "E8"), exist_ok=True)
    df.to_csv(os.path.join(RES, "E8", f"value_{region}.csv"), index=False)
    json.dump({"|".join(k): v for k, v in curves.items()}, open(os.path.join(RES, "E8", f"curves_{region}.json"), "w"))
    return df


if __name__ == "__main__":
    region = sys.argv[1]
    subset = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".npy") else None
    models = [d for d in os.listdir(os.path.join(RES, region)) if os.path.exists(os.path.join(RES, region, d, "extra.npz"))]
    df = main(region, models, subset)
    print(df[(df.horizon == "30min")].pivot_table(index="model", columns=["alert", "stratum"], values=["V_mean", "brier"]).round(3).to_string())
