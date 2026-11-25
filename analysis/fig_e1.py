"""Fig. 3: event-induced shift. (a) FE regression effects; (b) peak discharge-capacity proxy;
(c) PEMS-BAY speed-flow under dry vs heavy rain; (d) hour-level |deviation| vs placebo."""
import numpy as np, pandas as pd
from style import setup, save, plt, COL, top_legend

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
E1 = os.path.join(ROOT, "results", "E1")
setup()
reg = pd.read_csv(f"{E1}/fe_regression.csv")
cap = pd.read_csv(f"{E1}/capacity_proxy.csv")
hl = pd.read_csv(f"{E1}/hour_level_shift.csv")
fd = pd.read_csv(f"{E1}/pemsbay_fd.csv")
pairs = np.load(f"{E1}/pemsbay_pairs.npy", allow_pickle=True).item()

fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw=dict(wspace=0.3, hspace=0.62))
axs = axs.ravel()
ax = axs[0]
bins = ["heat_0_5F", "heat_gt5F", "rain_0.25_2.5", "rain_2.5_7.6", "rain_ge7.6", "pm_35_55", "pm_55_150", "pm_ge150"]
lab = ["Heat 0–5°F", "Heat >5°F", "Rain 0.25–2.5", "Rain 2.5–7.6", "Rain ≥7.6", "PM 35–55", "PM 55–150", "PM ≥150"]
x = np.arange(len(bins))
for k, d in enumerate(["D3", "D4", "D7"]):
    r = reg[reg.region == d].set_index("bin").loc[bins]
    eff = 100 * (np.exp(r.coef) - 1); se = 100 * r.se
    ax.errorbar(x + (k - 1) * 0.22, eff, yerr=1.96 * se, fmt="o", ms=2.5, color=COL[d], lw=0.8, capsize=1.2, label=d)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=55, ha="right")
ax.set_ylabel("Effect on hourly flow (%)"); ax.set_title("(a) Fixed-effects estimates", loc="left")

ax = axs[1]
ev = ["lightrain", "rain", "heat", "smoke"]; evl = ["Light rain", "Heavy rain", "Heat", "Smoke"]
for k, d in enumerate(["D3", "D4", "D7"]):
    c = cap[cap.region == d].set_index("event").reindex(ev)
    ax.bar(np.arange(4) + (k - 1) * 0.26, 100 * (1 - c.ratio_median), 0.26, color=COL[d], label=d)
for y, t in [(3, "HCM light"), (7, "HCM moderate"), (14, "HCM heavy")]:
    ax.axhline(y, color="gray", lw=0.5, ls=":")
    ax.text(1.01, y, t.replace("HCM ", "HCM\n"), fontsize=5.5, va="center", ha="left", color="gray",
            transform=ax.get_yaxis_transform(), clip_on=False, linespacing=0.9)
ax.set_xticks(range(4)); ax.set_xticklabels(evl, rotation=30, ha="right")
ax.set_ylabel("Peak-flow reduction (%)"); ax.set_title("(b) Discharge-capacity proxy", loc="left")

ax = axs[2]
k = max(pairs, key=lambda s: len(pairs[s][2]))
qd, vd, qr, vr = [np.asarray(a) for a in pairs[k]]
sub = np.random.default_rng(0).choice(len(qd), min(4000, len(qd)), replace=False)
ax.scatter(qd[sub], vd[sub], s=0.6, c="#95a5a6", lw=0, label="Dry", rasterized=True)
ax.scatter(qr, vr, s=0.8, c="#2471a3", lw=0, label="Rain", rasterized=True)
ax.set_xlabel("Flow (veh/h/lane)"); ax.set_ylabel("Speed (mph)")
ax.set_title(f"(c) Speed–flow, sensor {int(k)}", loc="left")
ax.legend(frameon=False, markerscale=6, loc="center left", bbox_to_anchor=(0.0, 0.33), handletextpad=0.1)   # empty low-flow, mid-speed area

ax = axs[3]
for j, e in enumerate(["placebo", "incident", "rain"]):
    v = [hl[(hl.region == d) & (hl.event == e)].abs_dev_median.values[0] * 100 for d in ["D3", "D4", "D7"]]
    ax.bar(np.arange(3) + (j - 1) * 0.26, v, 0.26, color=["#bdc3c7", "#d68910", "#2471a3"][j], label=["Normal (placebo)", "Crash", "Heavy rain"][j])
ax.set_xticks(range(3)); ax.set_xticklabels(["D3", "D4", "D7"])
ax.set_ylabel("Median |deviation| from\nsame time-of-week (%)"); ax.set_title("(d) Hour-level shift", loc="left")
ax.set_ylim(0, 19.5)
ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.0, fontsize=6.5)
h = [plt.Line2D([], [], marker="s", ls="", ms=5, color=COL[d]) for d in ["D3", "D4", "D7"]]
top_legend(fig, axs[:2], 3, handles=h, labels=["D3 Sacramento", "D4 Bay Area", "D7 Los Angeles"], fontsize=7)
save(fig, "fig03_e1_shift")
print("ok")
