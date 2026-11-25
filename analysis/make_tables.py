"""Build LaTeX tables (main D3, multi-region, ablation) and a macro file from results/*/summary*.json."""
import os, sys, json, glob, re, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from aggregate import collect, dm_table, NAMES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
GROUPS = [("Statistical", ["ha", "snaive", "climatology"]),
          ("Deterministic + split conformal", ["gwnet_conformal", "stid_conformal", "staeformer_conformal"]),
          ("Probabilistic deep", ["gwnet_gauss", "gwnet_quantile", "gwnet_mc", "staeformer_quantile", "deepar"]),
          ("Weather-aware", ["gwnet_conformal+wx", "staeformer_conformal+wx", "gwnet_gauss+wx"]),
          ("Pooled controls", ["gwens", "gwwxpool"]),
          ("Retrieval-guided", ["remind_orig", "remind", "remind_wx", "remindpool", "wremind", "wremindpool"])]
NAMES = dict(NAMES, **{"stid_conformal": "STID-Conformal", "gwnet_conformal+wx": "GWNet+W-Conformal",
                       "staeformer_conformal+wx": "STAEformer+W-Conformal"})
EV = ["normal", "heat", "rain", "smoke", "incident"]


def agg(df):
    d = df[df.horizon == "avg"]
    m = d.groupby(["model", "stratum"]).mean(numeric_only=True)
    s = d.groupby(["model", "stratum"]).std(numeric_only=True)
    n = d.groupby(["model", "stratum"]).seed.nunique()
    return m, s, n


def fmt(v, best, d=2, lower=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    t = f"{v:.{d}f}"
    return r"\textbf{" + t + "}" if best else t


def main_table(region="D3"):
    df = collect(region)
    m, s, n = agg(df)
    models = [x for _, g in GROUPS for x in g if (x, "all") in m.index]
    cols = [("all", "MAE", 2), ("all", "RMSE", 2), ("all", "CRPS", 2), ("all", "PICP90", 3), ("all", "Winkler90", 1),
            ("all", "ECE", 3), ("all", "Violation", 2)] + [(e, "CRPS", 2) for e in EV]
    vals = {}
    for c in cols:
        st, met, _ = c
        col = {}
        for x in models:
            try:
                v = m.loc[(x, st), met]
            except KeyError:
                v = np.nan
            if met == "Violation":
                v = 100 * v
            col[x] = v
        vals[c] = col
    best = {}
    for c, col in vals.items():
        st, met, _ = c
        arr = pd.Series(col).dropna()
        if met == "PICP90":
            best[c] = (arr - 0.9).abs().idxmin()
        elif met in ("Violation",):
            best[c] = None
        else:
            best[c] = arr.idxmin()
    lines = []
    for gname, g in GROUPS:
        g = [x for x in g if x in models]
        if not g:
            continue
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}}\\" % (len(cols) + 1, gname))
        for x in g:
            row = [NAMES.get(x, x)]
            for c in cols:
                v = vals[c][x]
                if c[1] in ("PICP90", "ECE", "Violation", "Winkler90") and x in ("ha", "snaive") or \
                   (c[1] in ("ECE",) and "conformal" in x and False):
                    row.append("--")
                    continue
                row.append(fmt(v, best[c] == x, c[2]))
            if n.get((x, "all"), 1) > 1:
                row[0] += r"$^{\dagger}$"
            lines.append(" & ".join(row) + r"\\")
        lines.append(r"\midrule")
    # seed-0 rows of the multi-seed models, so that single-seed rows are compared on the same seed
    m0 = agg(df[df.seed == 0])[0]
    ref0 = [x for x in models if n.get((x, "all"), 1) > 1 and NAMES.get(x, x) in ("GWNet-Gaussian", "WARP-Traffic")]
    if not ref0:
        open(os.path.join(PAPER, f"tab_main_{region}.tex"), "w").write("\n".join(lines[:-1]) + "\n")
        return m, s, n
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{%d}{l}{\textit{Seed 0 only (same-seed reference for unmarked rows)}}\\" % (len(cols) + 1))
    for x in [x for x in models if n.get((x, "all"), 1) > 1 and NAMES.get(x, x) in ("GWNet-Gaussian", "WARP-Traffic")]:
        row = [NAMES.get(x, x) + " (seed 0)"]
        for c in cols:
            try:
                v = m0.loc[(x, c[0]), c[1]]
            except KeyError:
                v = np.nan
            row.append(fmt(100 * v if c[1] == "Violation" else v, False, c[2]))
        lines.append(" & ".join(row) + r"\\")
    open(os.path.join(PAPER, f"tab_main_{region}.tex"), "w").write("\n".join(lines) + "\n")
    return m, s, n


def region_table(regions=("D3", "D4", "D7"), models=("climatology", "gwnet_conformal", "gwnet_gauss", "gwnet_gauss+wx", "gwwxpool",
                                                      "staeformer_quantile", "deepar", "remind", "remindpool", "wremind", "wremindpool")):
    lines = []
    blocks = {}
    for r in regions:
        try:
            blocks[r] = agg(collect(r))[0]
        except Exception:
            pass
    for x in models:
        row = [NAMES.get(x, x)]
        for r in regions:
            m = blocks.get(r)
            for st, met in [("all", "MAE"), ("all", "CRPS"), ("anyevent", "CRPS"), ("all", "PICP90")]:
                try:
                    v = m.loc[(x, st), met]
                    row.append(f"{v:.3f}" if met == "PICP90" else f"{v:.2f}")
                except Exception:
                    row.append("--")
        lines.append(" & ".join(row) + r"\\")
    # bold best per column (min except PICP closest to .9)
    cells = [l[:-2].split(" & ") for l in lines]
    for j in range(1, len(cells[0])):
        vs = []
        for i, c in enumerate(cells):
            try:
                vs.append((i, float(c[j])))
            except ValueError:
                pass
        if not vs:
            continue
        picp = (j - 1) % 4 == 3
        i0 = min(vs, key=lambda t: abs(t[1] - 0.9) if picp else t[1])[0]
        cells[i0][j] = r"\textbf{" + cells[i0][j] + "}"
    open(os.path.join(PAPER, "tab_regions.tex"), "w").write("\n".join(" & ".join(c) + r"\\" for c in cells) + "\n")


ABL = [("wremind", "wremindpool", "WARP-Traffic (full)"),
       ("wremind_noCal", "pool_wremind_noCal", "M1: no calendar-consistent retrieval"),
       ("wremind_wxret", "pool_wremind_wxret", "M1: weather-weighted retrieval ($\\alpha=0.6$)"),
       ("wremind_noM2", "pool_wremind_noM2", "M2: REMIND scalar similarity gate"),
       ("wremind_noH", "pool_wremind_noH", "M2: one gate for all horizons"),
       ("wremind_noPi", "pool_wremind_noPi", "M2: no regime supervision"),
       ("wremind_nll", "pool_wremind_nll", "M2: likelihood instead of CRPS training"),
       ("wremind_noM3", "pool_wremind_noM3", "M3: no capacity penalty / truncation"),
       ("wremind_noM4", "pool_wremind_noM4", "M4: no rare-event training"),
       ("remind", "remindpool", "REMIND (same backbone)"),
       ("gwnet_gauss+wx", "gwwxpool", "Parametric partner instead of memory")]


def ablation_table(region="D3"):
    df = collect(region)
    m, s, n = agg(df[df.seed == 0])   # ablations exist for seed 0 only; compare like for like
    rows = []
    ref = m.loc[("wremindpool", "all"), "CRPS"] if ("wremindpool", "all") in m.index else np.nan
    for k, kp, lab in ABL:
        if (kp, "all") not in m.index:
            continue
        r = [lab]
        r.append(f"{m.loc[(k, 'all'), 'CRPS']:.2f}" if (k, "all") in m.index else "--")
        for st in ["all", "normal", "anyevent", "rain", "incident"]:
            r.append(f"{m.loc[(kp, st), 'CRPS']:.2f}")
        r.append(f"{m.loc[(kp, 'all'), 'PICP90']:.3f}")
        r.append(f"{100 * m.loc[(kp, 'all'), 'Violation']:.2f}")
        r.append(f"{100 * (m.loc[(kp, 'all'), 'CRPS'] / ref - 1):+.1f}")
        rows.append(" & ".join(r) + r"\\")
    open(os.path.join(PAPER, f"tab_ablation_{region}.tex"), "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    for r in sys.argv[1:] or ["D3"]:
        main_table(r)
    ablation_table("D3")
    region_table()


def dm_tex(region="D3"):
    """Appendix table: DM statistics of WARP-Traffic vs. each competitor by stratum (negative favours WARP-Traffic)."""
    df = dm_table(region)
    st = ["all", "normal", "anyevent", "heat", "rain", "smoke", "incident"]
    lines = []
    for o, g in df.groupby("vs", sort=False):
        row = [NAMES.get(o, o)]
        for s in st:
            r = g[g.stratum == s]
            if len(r) == 0:
                row.append("--")
                continue
            r = r.iloc[0]
            star = "***" if r.p_holm < 0.001 else "**" if r.p_holm < 0.01 else "*" if r.p_holm < 0.05 else ""
            row.append(f"{100 * r.rel_crps:+.1f} ({r.dm:.1f}){star}")
        lines.append(" & ".join(row) + r"\\")
    open(os.path.join(PAPER, f"tab_dm_{region}.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    dm_tex("D3")
