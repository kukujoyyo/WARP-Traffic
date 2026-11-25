"""E10: when does memory help? Conditions x lead time x district, a pooling decomposition, and alert operating points.

Part A (series.npz, all shared seeds). For each district, stratum and lead time:
  - gain of WARP-Traffic (pool) over GWNet-Gaussian, 1 - P/G,
  - gain of the memory expert alone, 1 - M/G,
  - decomposition of the pool gain with the equal-weight identity P = (M+G)/2 - A:
        1 - P/G = (G-M)/(2G)  [accuracy term: how good the memory expert is]
                + A/G         [diversity term: how differently the two experts err]
    A is computed as the residual (M+G)/2 - P, which equals lambda(1-lambda) times the Cramer distance without truncation.
  95% intervals: bootstrap over whole days (2,000 draws).

Part B (extra.npz, seeds with stored alert probabilities). Operating points of the two alerts of Section 5.5:
  - probability of detection (POD) at a fixed false-alarm rate (FPR = 1% of non-event cells),
  - share of missed events avoided relative to GWNet-Gaussian at that budget,
  - false alarms avoided at a fixed POD of 0.8,
  - concentration: share of the additional detections that come from the five best days, and share of days on which
    WARP-Traffic detects at least as many events.
  Probabilities are shares of 100 samples, so ROC curves are built from per-day histograms over 101 levels and the
  day bootstrap is exact and fast.

usage: python analysis/e10_conditions.py   -> results/E10/*.csv, paper/tab_alert.tex, paper/figures/fig10_conditions.*
"""
import os, sys, numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from style import setup, save
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
PROC = os.path.join(ROOT, "data_proc", "regions")
PAPER = os.path.join(ROOT, "paper")
OUT = os.path.join(RES, "E10")
SEEDS = {"D3": [0, 1, 2], "D4": [0, 1], "D7": [0, 1]}
STRATA = ["normal", "heat", "rain", "smoke", "incident", "anyevent"]
LABEL = {"normal": "Normal", "heat": "Heat", "rain": "Heavy rain", "smoke": "Smoke", "incident": "Crash",
         "anyevent": "Any event", "all": "All"}
HZ = {"15 min": [2], "30 min": [5], "60 min": [11], "1-60 min": list(range(12))}
B = 2000
rng = np.random.default_rng(0)


def days_of(region, t0):
    t = pd.DatetimeIndex(np.load(os.path.join(PROC, region, "time.npy")))[t0]
    return np.asarray((t.normalize() - t.normalize()[0]).days)


def ser(region, run):
    z = np.load(os.path.join(RES, region, run, "series.npz"))
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- Part A
def part_a():
    rows = []
    for region, seeds in SEEDS.items():
        P = [ser(region, f"wremindpool_s{s}") for s in seeds]
        M = [ser(region, f"wremind_s{s}") for s in seeds]
        G = [ser(region, f"gwnet_gauss_s{s}") for s in seeds]
        day = days_of(region, P[0]["t0"])
        D = day.max() + 1
        for k in ["all"] + STRATA:
            for hz, hs in HZ.items():
                agg = lambda Z: sum(np.bincount(day, z[k][:, hs, 0].sum(1), D) for z in Z)
                p, m, g = agg(P), agg(M), agg(G)
                n = np.bincount(day, P[0][k][:, hs, 2].sum(1), D)
                ok = n > 0
                p, m, g = p[ok], m[ok], g[ok]
                L = len(p)

                def stats_(j):
                    Ps, Ms, Gs = p[j].sum(), m[j].sum(), g[j].sum()
                    return np.array([1 - Ps / Gs, 1 - Ms / Gs, (Gs - Ms) / (2 * Gs), ((Ms + Gs) / 2 - Ps) / Gs])

                est = stats_(np.arange(L))
                bs = np.array([stats_(rng.integers(0, L, L)) for _ in range(B)])
                lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
                r = dict(region=region, stratum=k, horizon=hz, days=L, cells=int(n.sum() * len(seeds)))
                for i, nm in enumerate(["pool", "mem", "acc", "div"]):
                    r[nm], r[nm + "_lo"], r[nm + "_hi"] = 100 * est[i], 100 * lo[i], 100 * hi[i]
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Part B
def load_events(region, t0):
    ev = np.load(os.path.join(PROC, region, "events.npy"), mmap_mode="r")
    idx = t0[:, None] + np.array([3, 6, 12])[None]
    e = np.asarray(ev[idx.ravel()]).reshape(len(t0), 3, -1)
    sub = os.path.join(PROC, region, "subset_core.npy")
    if region != "D3" and os.path.exists(sub):
        e = e[..., np.load(sub)]
    return (e & 0b1111) > 0


def hist(region, run, alert, hi, stratum):
    """Per-day counts of positives and negatives at each of 101 probability levels."""
    z = np.load(os.path.join(RES, region, run, "extra.npz"))
    t0 = z["t0"].astype(int)
    day = days_of(region, t0)
    D = day.max() + 1
    p = np.rint(z[f"p{alert}"][:, hi].astype(np.float32) * 100).astype(np.int64).clip(0, 100)
    o = z[f"o{alert}"][:, hi]
    v = z["valid"][:, hi]
    if stratum == "event":
        v = v & load_events(region, t0)[:, hi]
    dd = np.broadcast_to(day[:, None], p.shape)
    key = (dd * 101 + p)[v]
    oo = o[v]
    pos = np.bincount(key[oo], minlength=D * 101).reshape(D, 101)
    neg = np.bincount(key[~oo], minlength=D * 101).reshape(D, 101)
    return pos, neg


def roc(pos, neg):
    """Alert when p >= level. Returns FPR and POD for levels 100..0 with a leading (0,0)."""
    tp = np.concatenate([[0], np.cumsum(pos[::-1])])
    fp = np.concatenate([[0], np.cumsum(neg[::-1])])
    return fp / max(fp[-1], 1), tp / max(tp[-1], 1)


def pod_at(pos, neg, fpr0):
    f, t = roc(pos, neg)
    return np.interp(fpr0, f, t)


def fpr_at(pos, neg, pod0):
    f, t = roc(pos, neg)
    i = np.searchsorted(t, pod0)                       # t is non-decreasing
    i = min(max(i, 1), len(t) - 1)
    w = (pod0 - t[i - 1]) / max(t[i] - t[i - 1], 1e-12)
    return f[i - 1] + w * (f[i] - f[i - 1])


def hits_by_day(pos, neg, fpr0):
    """Expected detections per day at the level (with randomisation at ties) where the overall FPR equals fpr0."""
    N = neg.sum()
    cn = np.concatenate([[0], np.cumsum(neg.sum(0)[::-1])])
    i = min(max(np.searchsorted(cn, fpr0 * N), 1), len(cn) - 1)
    w = (fpr0 * N - cn[i - 1]) / max(cn[i] - cn[i - 1], 1)  # fraction of the tied level that is alerted
    rp = pos[:, ::-1]
    return rp[:, :i - 1].sum(1) + w * rp[:, i - 1]


def part_b(fpr0=0.01, pod0=0.8):
    rows = []
    hzs = {"15 min": 0, "30 min": 1, "60 min": 2}
    for region, seeds in SEEDS.items():
        seeds = [s for s in seeds if os.path.exists(os.path.join(RES, region, f"wremindpool_s{s}", "extra.npz"))]
        for alert in ["near", "drop"]:
            for stratum in ["all", "event"]:
                for hz, hi in hzs.items():
                    H = {m: [hist(region, f"{m}_s{s}", alert, hi, stratum) for s in seeds] for m in ["gwnet_gauss", "wremindpool"]}
                    pg = sum(h[0] for h in H["gwnet_gauss"]); ng = sum(h[1] for h in H["gwnet_gauss"])
                    pw = sum(h[0] for h in H["wremindpool"]); nw = sum(h[1] for h in H["wremindpool"])
                    keep = (pg.sum(1) + ng.sum(1)) > 0
                    pg, ng, pw, nw = pg[keep], ng[keep], pw[keep], nw[keep]
                    L = len(pg)

                    def st(j):
                        a, b = pod_at(pg[j].sum(0), ng[j].sum(0), fpr0), pod_at(pw[j].sum(0), nw[j].sum(0), fpr0)
                        c, d = fpr_at(pg[j].sum(0), ng[j].sum(0), pod0), fpr_at(pw[j].sum(0), nw[j].sum(0), pod0)
                        return np.array([a, b, 1 - (1 - b) / (1 - a), c, d, 1 - d / c])

                    est = st(np.arange(L))
                    bs = np.array([st(rng.integers(0, L, L)) for _ in range(B // 2)])
                    lo, hi_ = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
                    hg = hits_by_day(pg, ng, fpr0); hw = hits_by_day(pw, nw, fpr0)
                    diff = hw - hg
                    top5 = np.sort(diff)[::-1][:5].sum() / max(diff.sum(), 1e-9)
                    act = pg.sum(1) > 0                          # days with at least one event
                    rows.append(dict(region=region, seeds=len(seeds), alert=alert, stratum=stratum, horizon=hz,
                                     base_rate=float(pg.sum() / (pg.sum() + ng.sum())), n_pos=int(pg.sum() / len(seeds)),
                                     pod_gw=est[0], pod_wr=est[1], miss_avoided=100 * est[2],
                                     miss_avoided_lo=100 * lo[2], miss_avoided_hi=100 * hi_[2],
                                     fpr_gw=100 * est[3], fpr_wr=100 * est[4], fa_avoided=100 * est[5],
                                     fa_avoided_lo=100 * lo[5], fa_avoided_hi=100 * hi_[5],
                                     extra_hits=float(diff.sum() / len(seeds)), top5_share=100 * top5,
                                     days_better=100 * float((diff[act] >= 0).mean()), days=int(act.sum())))
                    print(rows[-1]["region"], alert, stratum, hz, round(est[2] * 100, 1), round(est[5] * 100, 1), flush=True)
    return pd.DataFrame(rows)


def level_for(hist, target):
    """(i, w): alert on the i-1 highest levels and a fraction w of level i so that the cumulative share equals target."""
    c = np.concatenate([[0], np.cumsum(hist[::-1])]) / max(hist.sum(), 1)
    i = min(max(np.searchsorted(c, target), 1), len(c) - 1)
    w = (target - c[i - 1]) / max(c[i] - c[i - 1], 1e-12)
    return i, w


def apply_level(hist, iw):
    i, w = iw
    r = hist[::-1]
    return (r[:i - 1].sum() + w * r[i - 1]) / max(hist.sum(), 1)


def part_c(cal_days=90, fpr0=0.01, pod0=0.8):
    """Frozen thresholds: operating points set on Jan-Mar 2021, applied unchanged to Apr-Dec 2021."""
    rows = []
    for region, seeds in SEEDS.items():
        seeds = [s for s in seeds if os.path.exists(os.path.join(RES, region, f"wremindpool_s{s}", "extra.npz"))]
        for alert in ["near", "drop"]:
            for hz, hi in {"30 min": 1, "60 min": 2}.items():
                r = dict(region=region, alert=alert, horizon=hz)
                for m, tag in [("gwnet_gauss", "g"), ("wremindpool", "w")]:
                    H = [hist(region, f"{m}_s{s}", alert, hi, "all") for s in seeds]
                    pos, neg = sum(h[0] for h in H), sum(h[1] for h in H)
                    cp, cn = pos[:cal_days].sum(0), neg[:cal_days].sum(0)
                    ep, en = pos[cal_days:].sum(0), neg[cal_days:].sum(0)
                    a = level_for(cn, fpr0)                     # fixed false-alarm budget
                    r[f"fpr_{tag}"], r[f"pod_{tag}"] = 100 * apply_level(en, a), apply_level(ep, a)
                    b = level_for(cp, pod0)                     # fixed detection target
                    r[f"pod80_{tag}"], r[f"fpr80_{tag}"] = apply_level(ep, b), 100 * apply_level(en, b)
                r["miss_avoided"] = 100 * (1 - (1 - r["pod_w"]) / (1 - r["pod_g"]))
                r["fa_avoided"] = 100 * (1 - r["fpr80_w"] / r["fpr80_g"])
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- outputs
def figure(a):
    setup()
    regs, hz3 = ["D3", "D4", "D7"], ["15 min", "30 min", "60 min"]
    rows = ["normal", "heat", "rain", "smoke", "incident"]
    fig = plt.figure(figsize=(7.1, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], hspace=0.55, wspace=0.22)
    cmap = plt.get_cmap("RdBu")
    for pi, (col, title) in enumerate([("pool", "(a) WARP-Traffic vs GWNet-Gaussian"), ("mem", "(b) Memory expert alone vs GWNet-Gaussian")]):
        ax = fig.add_subplot(gs[0, pi])
        Z = np.full((len(rows), 9), np.nan); S = np.zeros_like(Z, bool)
        for ri, k in enumerate(rows):
            for gi, r in enumerate(regs):
                for hi, h in enumerate(hz3):
                    q = a[(a.region == r) & (a.stratum == k) & (a.horizon == h)].iloc[0]
                    Z[ri, gi * 3 + hi] = q[col]
                    S[ri, gi * 3 + hi] = (q[col + "_lo"] > 0) or (q[col + "_hi"] < 0)
        im = ax.imshow(Z, cmap=cmap, vmin=-15, vmax=15, aspect="auto")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                txt = f"{Z[i, j]:.1f}" if S[i, j] else f"({Z[i, j]:.1f})"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                        color="white" if abs(Z[i, j]) > 9 else "black")
        for x in [2.5, 5.5]:
            ax.axvline(x, color="white", lw=2)
        ax.set_xticks(range(9)); ax.set_xticklabels(["15", "30", "60"] * 3)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([LABEL[k] for k in rows] if pi == 0 else [])
        for gi, r in enumerate(regs):
            ax.text(gi * 3 + 1, -0.6, r, ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_xlabel("Lead time (min)")
        ax.set_title(title, loc="left", pad=19)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_axes([0.92, 0.585, 0.012, 0.29])
    cb = fig.colorbar(im, cax=cax); cb.set_label("CRPS reduction (%)", fontsize=7); cb.ax.tick_params(labelsize=6)
    # (c) decomposition, all lead times
    ax = fig.add_subplot(gs[1, :])
    cats = ["normal", "heat", "rain", "smoke", "incident"]
    x0 = 0; xt, xl = [], []
    for r in regs:
        for k in cats:
            q = a[(a.region == r) & (a.stratum == k) & (a.horizon == "1-60 min")].iloc[0]
            ax.bar(x0, q.acc, 0.38, color="#7f8c8d", edgecolor="none")
            ax.bar(x0 + 0.4, q["div"], 0.38, color="#c0392b", edgecolor="none")
            ax.plot([x0 - 0.2, x0 + 0.6], [q.pool, q.pool], color="black", lw=1.1)
            xt.append(x0 + 0.2); xl.append(LABEL[k].replace("Heavy rain", "Rain"))
            x0 += 1.1
        x0 += 0.6
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=0, fontsize=6.2)
    for gi, r in enumerate(regs):
        ax.text(np.mean(xt[gi * 5:(gi + 1) * 5]), ax.get_ylim()[1] * 1.02, r, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Contribution to\nCRPS reduction (%)")
    ax.set_title("(c) Decomposition of the pool gain, 1-60 min", loc="left", pad=14)
    ax.legend([Patch(color="#7f8c8d"), Patch(color="#c0392b"), plt.Line2D([], [], color="black", lw=1.1)],
              ["Accuracy term (G-M)/2G", "Diversity term A/G", "Pool gain 1-P/G"],
              frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=6.6)
    save(fig, "fig10_conditions")


def tex_alert(b):
    L = []
    for r in ["D3", "D4", "D7"]:
        for alert, an in [("near", "Near-capacity"), ("drop", "Demand drop")]:
            cells = []
            for hz in ["30 min", "60 min"]:
                q = b[(b.region == r) & (b.alert == alert) & (b.stratum == "all") & (b.horizon == hz)].iloc[0]
                cells += [f"{q.pod_gw:.3f}", f"{q.pod_wr:.3f}",
                          f"{q.miss_avoided:.1f} [{q.miss_avoided_lo:.1f}, {q.miss_avoided_hi:.1f}]",
                          f"{q.fa_avoided:.1f} [{q.fa_avoided_lo:.1f}, {q.fa_avoided_hi:.1f}]"]
            L.append(f"{r if alert == 'near' else ''} & {an} & " + " & ".join(cells) + "\\\\")
        if r != "D7":
            L.append("\\addlinespace")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    a = part_a()
    a.to_csv(os.path.join(OUT, "conditions.csv"), index=False)
    print(a[a.horizon == "1-60 min"].round(2).to_string())
    figure(a)
    b = part_b()
    b.to_csv(os.path.join(OUT, "alerts.csv"), index=False)
    print(b.round(3).to_string())
    open(os.path.join(PAPER, "tab_alert.tex"), "w").write(tex_alert(b))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "frozen":
        c = part_c()
        c.to_csv(os.path.join(OUT, "alerts_frozen.csv"), index=False)
        print(c.round(3).to_string())
    elif len(sys.argv) > 1 and sys.argv[1] == "fig":
        figure(pd.read_csv(os.path.join(OUT, "conditions.csv")))
    else:
        main()
