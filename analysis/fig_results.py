"""Result figures: stratified CRPS skill (fig05), horizon curves (fig06), gate interpretability (fig07),
robustness (fig08), decision value (fig09), motivation/case (fig01)."""
import os, sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from style import setup, save, plt, MODEL_COL, top_legend
from matplotlib.patches import Patch
from aggregate import collect

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
setup()
SHOW = [("climatology", "Climatology"), ("gwnet_gauss", "GWNet-G"), ("staeformer_quantile", "STAEformer-Q"), ("deepar", "DeepAR"),
        ("gwnet_gauss+wx", "GWNet-G+W"), ("gwens", "GWNet-G ens."), ("remindpool", "REMIND$\\oplus$GWNet-G"),
        ("wremind", "WARP-Traffic mem. expert"), ("wremindpool", "WARP-Traffic")]
COLS = ["#aab7b8", "#7f8c8d", "#16a085", "#d68910", "#566573", "#bfc9ca", "#2471a3", "#f1948a", "#c0392b"]
STR = [("normal", "Normal"), ("heat", "Heat"), ("rain", "Heavy rain"), ("smoke", "Smoke"), ("incident", "Crash")]


def fig05(region="D3", ref="gwnet_gauss"):
    """CRPS change relative to the strongest parametric baseline, by stratum (negative = better)."""
    df = collect(region)
    df = df[df.horizon == "avg"]
    d = df.groupby(["model", "stratum"]).CRPS.mean()
    seeds = df.groupby("model").seed.agg(set)

    def rel(k, s):   # change vs. ref on the seeds both models share
        c = seeds[k] & seeds[ref]
        x = df[df.seed.isin(c) & (df.stratum == s)].groupby("model").CRPS.mean()
        return 100 * (x[k] / x[ref] - 1)

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    strata = [("all", "All")] + STR
    ms = [(k, l, c) for (k, l), c in zip(SHOW, COLS) if (k, "all") in d.index and k not in ("climatology", ref)]
    w = 0.8 / len(ms)
    for j, (k, l, c) in enumerate(ms):
        v = [rel(k, s) for s, _ in strata]
        ax.bar(np.arange(len(strata)) + (j - len(ms) / 2 + 0.5) * w, v, w, color=c, label=l, edgecolor="none")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(strata))); ax.set_xticklabels([l for _, l in strata])
    ax.set_ylabel("CRPS change vs. GWNet-G (%)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=6.5, handlelength=1)
    save(fig, f"fig05_strata_{region}")


def fig06(region="D3"):
    df = collect(region)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.35))
    for (k, l), c in zip(SHOW, COLS):
        if k == "climatology":
            continue
        for ax, (st, met) in zip(axs, [("all", "CRPS"), ("anyevent", "CRPS"), ("all", "PICP90")]):
            s = df[(df.model == k) & (df.stratum == st) & (df.horizon != "avg")].groupby("horizon")[met].mean()
            if len(s) == 0:
                continue
            s = s.reindex(["15min", "30min", "60min"])
            ax.plot([15, 30, 60], s.values, "-o", ms=2.5, color=c, label=l, lw=1.4 if k == "wremindpool" else 0.9)
    for ax, t in zip(axs, ["(a) CRPS, all cells", "(b) CRPS, event cells", "(c) 90% PI coverage"]):
        ax.set_title(t, loc="left"); ax.set_xlabel("Horizon (min)"); ax.set_xticks([15, 30, 60])
    axs[2].axhline(0.9, color="k", lw=0.5, ls=":")
    axs[0].set_ylabel("CRPS (veh/5 min)")
    top_legend(fig, axs, 4)
    save(fig, f"fig06_horizon_{region}")


def fig07(region="D3"):
    """Gate interpretability: (a) mean g by stratum REMIND vs WARP-Traffic; (b) regime ROC AUC; (c) case-day time series."""
    from sklearn.metrics import roc_curve
    cases = os.path.join(RES, "cases", f"cases_{region}.npz")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 1.5]))
    g = pd.read_csv(os.path.join(RES, f"E5_gate_{region}.csv"))
    ax = axs[0]
    cols = ["g_normal", "g_heat", "g_rain", "g_smoke", "g_incident"]
    wr = g[g.model.str.match(r"wremind_s\d")]
    bars = [("remind_s", "#2471a3", "REMIND (all $h$)", cols)]
    if "gh15_normal" in wr:
        bars += [("wremind_s", c, f"WARP-Traffic, {h} min", [f"gh{h}_{x[2:]}" for x in cols])
                 for h, c in [("15", "#f5b7b1"), ("30", "#e74c3c"), ("60", "#922b21")]]
    else:
        bars += [("wremind_s", "#c0392b", "WARP-Traffic", cols)]
    w = 0.8 / len(bars)
    for j, (m, c, l, cc) in enumerate(bars):
        r = g[g.model.str.match(m + r"\d")][cc].mean()
        ax.bar(np.arange(5) + (j - len(bars) / 2 + 0.5) * w, r.values, w, color=c, label=l)
    ax.set_xticks(range(5)); ax.set_xticklabels(["Normal", "Heat", "Rain", "Smoke", "Crash"], rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean trust in memory $g$"); ax.set_title("(a) Trust by condition", loc="left")
    ax.set_ylim(0.0, 1.0)
    ax = axs[1]
    rg = pd.read_csv(os.path.join(RES, f"E5_regime_{region}.csv"))
    rg = rg[rg.model.str.startswith("wremind_s")].groupby("regime")[["auc", "ap", "prevalence"]].mean()
    ax.barh(range(len(rg)), rg.auc.values, color="#c0392b", height=0.55)
    for i, (a, p) in enumerate(zip(rg.auc, rg.ap)):
        ax.text(a + 0.01, i, f"{a:.2f} (AP {p:.2f})", va="center", fontsize=6)
    ax.set_yticks(range(len(rg))); ax.set_yticklabels([s.capitalize().replace("Incident", "Crash") for s in rg.index])
    ax.set_xlim(0.5, 1.15); ax.set_xlabel("ROC AUC of $\\pi$"); ax.set_title("(b) Regime posterior", loc="left")
    ax = axs[2]
    if os.path.exists(cases):
        z = np.load(cases, allow_pickle=True)
        sel = z["day_key"] == "rain_ar"
        t = np.arange(sel.sum()) * 5 / 60 + 4
        node = int(np.nanargmax((z["ev"][sel] >> 1 & 1).sum(0)))                  # sensor with the most heavy-rain cells
        for key, c, l in [("remind_s0_gate", "#2471a3", "REMIND $g$"), ("wremind_s0_gate", "#c0392b", "WARP-Traffic $g$")]:
            if key in z:
                ax.plot(t, z[key][sel][:, node].astype(float), color=c, label=l)
        if "wremind_s0_pi" in z:
            ax.plot(t, z["wremind_s0_pi"][sel][:, node, 1].astype(float), color="#c0392b", ls="--", lw=0.8, label="WARP-Traffic $\\pi_{rain}$")
        rain = (z["ev"][sel][:, node] >> 1 & 1).astype(bool)
        ax.fill_between(t, 0, 1, where=rain, color="#85c1e9", alpha=0.35, lw=0, label="Heavy rain")
        ax.set_ylim(0, 1); ax.set_xlabel("Hour, 24 Oct 2021"); ax.set_title("(c) Atmospheric river, 30-min gate", loc="left")
    top_legend(fig, [axs[0]], 2, over=[axs[0]], fontsize=6.3)
    top_legend(fig, [axs[2]], 2, over=[axs[2]], fontsize=6.3)
    save(fig, f"fig07_gate_{region}")


def fig01(region="D3"):
    """Motivation: predictive 90% bands on the atmospheric-river day, REMIND vs WARP-Traffic vs GWNet-G."""
    z = np.load(os.path.join(RES, "cases", f"cases_{region}.npz"), allow_pickle=True)
    sel = z["day_key"] == "rain_ar"
    ev = z["ev"][sel]
    rainy = (ev >> 1 & 1).mean(0) >= 0.5
    node = int(np.nanargmax(np.where(rainy, np.nanmean(z["y"][sel], 0), -1)))    # busiest sensor with rain most of the day
    t = np.arange(sel.sum()) * 5 / 60 + 4
    y = z["y"][sel][:, node]; up = z["upper"][sel][:, node]
    runs = [(k, l, c) for k, l, c in [("gwnet_gauss_s0", "GWNet-Gaussian", "#7f8c8d"), ("remind_s0", "REMIND", "#2471a3"),
                                      ("wremindpool_s0", "WARP-Traffic", "#c0392b")] if f"{k}_q" in z]
    fig, axs = plt.subplots(1, len(runs), figsize=(7.2, 2.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    axs = np.atleast_1d(axs)
    rain = (ev[:, node] >> 1 & 1).astype(bool)
    for ax, (k, l, c) in zip(axs, runs):
        q = z[f"{k}_q"][sel][:, :, node]
        ytop = 1.15 * max(np.nanmax(up), np.nanmax(y))
        ax.fill_between(t, 0, 1, where=rain, color="#d6eaf8", lw=0, transform=ax.get_xaxis_transform())
        ax.fill_between(t, q[:, 0], q[:, 4], color=c, alpha=0.25, lw=0, label="90% PI")
        ax.plot(t, q[:, 2], color=c, lw=0.9, label="Median")
        ax.plot(t, y, color="k", lw=0.7, label="Observed")
        ax.plot(t, up, color="k", lw=0.6, ls=":", label="$C_i f(\\mathbf{w})$")
        cov = np.mean((y >= q[:, 0]) & (y <= q[:, 4]))
        crps_like = np.nanmean(np.abs(q[:, 2] - y))
        ax.set_title(f"{l}\ncov. {100 * cov:.0f}%, MAE {crps_like:.1f}", loc="left")
        ax.set_xlabel("Hour, 24 Oct 2021")
    axs[0].set_ylabel("Flow (veh/5 min), 30-min ahead")
    h = [Patch(color="#7f8c8d", alpha=0.3, lw=0), plt.Line2D([], [], color="#7f8c8d", lw=0.9), plt.Line2D([], [], color="k", lw=0.7),
         plt.Line2D([], [], color="k", lw=0.6, ls=":"), Patch(color="#d6eaf8", lw=0)]
    top_legend(fig, axs, 5, handles=h, labels=["90% PI", "Median", "Observed", "$C_i f(\\mathbf{w})$", "Heavy rain"])
    save(fig, f"fig01_motivation_{region}")


def fig08():
    """Sensitivity and robustness: pool weight, sensor outages, weather noise / dropout."""
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.35))
    models = [("remindpool", "REMIND$\\oplus$GWNet-G", "#2471a3"), ("gwwxpool", "GWNet-G$\\oplus$GWNet-G+W", "#5dade2"),
              ("wremindpool", "WARP-Traffic", "#c0392b"), ("gwnet_gauss", "GWNet-G", "#7f8c8d"), ("gwnet_gauss+wx", "GWNet-G+W", "#566573")]

    def get(m, sfx, st="all", seed=0, tag=""):
        f = os.path.join(RES, "D3", f"{m}{tag}_s{seed}", f"summary{sfx}.json")
        return json.load(open(f))[st]["avg"]["CRPS"] if os.path.exists(f) else np.nan
    ax = axs[0]
    ws = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    runs = ["gwnet_gauss", "wremindpool_w3", "wremindpool_w4", "wremindpool", "wremindpool_w6", "wremindpool_w7", "wremind"]
    for st, ls, l in [("all", "-", "all cells"), ("anyevent", "--", "event cells")]:
        ax.plot(ws, [get(r, "", st) for r in runs], ls, marker="o", ms=2.5, color="#c0392b", label=l)
    ax.set_xlabel("Pool weight $\\lambda$ of memory expert"); ax.set_ylabel("CRPS"); ax.set_title("(a) Pool weight", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2)
    ax.set_ylim(None, ax.get_ylim()[1] + 0.12)
    ax = axs[1]
    for m, l, c in [x for x in models if x[0] != "gwwxpool"]:
        fr = [1.0, 0.8, 0.6]
        v = [get(m, "") if f == 1 else get(m, f"_nf{f}_wn0.0_wd0") for f in fr]
        ax.plot([100 * (1 - f) for f in fr], v, "-o", ms=2.5, color=c, label=l)
    ax.set_xlabel("Sensors with failed input (%)"); ax.set_title("(b) Detector outages", loc="left")
    ax = axs[2]
    for m, l, c in [x for x in models if x[0] in ("wremindpool", "gwnet_gauss+wx")]:
        v = [get(m, "", "anyevent"), get(m, "_nf1.0_wn0.5_wd0", "anyevent"), get(m, "_nf1.0_wn1.0_wd0", "anyevent"),
             get(m, "_nf1.0_wn0.0_wd1", "anyevent")]
        ax.plot(range(4), v, "-o", ms=2.5, color=c, label=l)
    for m, l, c in [x for x in models if x[0] in ("remindpool", "gwnet_gauss")]:
        ax.axhline(get(m, "", "anyevent"), color=c, ls=":", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["clean", "σ=0.5", "σ=1.0", "missing"])
    ax.set_xlabel("Weather input"); ax.set_title("(c) Weather errors, event cells", loc="left")
    top_legend(fig, axs[1:], 4, over=axs[1:])
    save(fig, "fig08_robustness")


def fig09(region="D3"):
    cur = json.load(open(os.path.join(RES, "E8", f"curves_{region}.json")))
    from e8_decision import RATIOS
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True, gridspec_kw=dict(wspace=0.16))
    show = [("gwnet_s0", "GWNet (det.)", "#bfc9ca"), ("gwnet_gauss_s0", "GWNet-G", "#7f8c8d"), ("staeformer_quantile_s0", "STAEformer-Q", "#16a085"),
            ("deepar_s0", "DeepAR", "#d68910"), ("remindpool_s0", "REMIND$\\oplus$GWNet-G", "#2471a3"), ("wremindpool_s0", "WARP-Traffic", "#c0392b")]
    for ax, (al, hz, t) in zip(axs, [("near", "30min", "(a) Near capacity\n30 min ahead"), ("near", "60min", "(b) Near capacity\n60 min ahead"),
                                     ("drop", "30min", "(c) Demand drop\n30 min ahead"), ("drop", "60min", "(d) Demand drop\n60 min ahead")]):
        for k, l, c in show:
            key = f"{k}|{al}|{hz}|all"
            if key in cur:
                ax.plot(RATIOS, np.clip(cur[key], -0.2, 1), color=c, label=l, lw=1.4 if k.startswith("wremindpool") else 0.9)
        ax.set_title(t, loc="left"); ax.set_xlabel("Cost-loss ratio $C/L$"); ax.set_ylim(-0.2, 1); ax.axhline(0, color="k", lw=0.4)
    axs[0].set_ylabel("Value score $V$")
    for ax in axs:
        ax.set_xticks([0.25, 0.5, 0.75]); ax.set_xlim(0, 1)
    top_legend(fig, axs, 6)
    save(fig, f"fig09_value_{region}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
