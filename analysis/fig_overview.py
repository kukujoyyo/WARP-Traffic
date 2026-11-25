"""Overall architecture of WARP-Traffic: setting and challenges | proposed model | training and evaluation.
Insets are crops of Fig. 2 (study area) and Fig. 1 (WARP-Traffic forecast on the atmospheric-river day)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Circle
from style import save, FIG

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "savefig.dpi": 400,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
W, H = 7.2, 4.9
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()

GREEN_H, GREEN = "#6f8f72", "#8db85a"
ORANGE_H, ORANGE = "#f4a259", "#ee7d14"
PINK_H, RED, DASHRED = "#e2788f", "#c0504d", "#a0522d"
TEAL_H, TEAL = "#4f9db5", "#2b8299"
INK, EDGE = "#1a1a1a", "#3a3a3a"


def panel(x0, x1, y0, y1, title, color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="white", ec=EDGE, lw=1.1, zorder=1))
    ax.add_patch(FancyBboxPatch((x0, y1 - 0.32), x1 - x0, 0.32, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=color, ec=EDGE, lw=1.1, zorder=2))
    ax.add_patch(plt.Rectangle((x0 + 0.006, y1 - 0.32), x1 - x0 - 0.012, 0.14, fc=color, ec="none", zorder=2.1))
    ax.plot([x0, x1], [y1 - 0.32, y1 - 0.32], color=EDGE, lw=1.1, zorder=2.2)
    ax.text((x0 + x1) / 2, y1 - 0.16, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=3)


def pill(cx, cy, w, text, color, h=0.24, fs=7.6, tc="white"):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h / 2}",
                                fc=color, ec="none", zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc, zorder=4)


def dbox(cx, cy, w, h, text, color, fs=7.2, ls=(0, (4, 2)), lw=0.9, tc=None, fc="white"):
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fc, ec=color, lw=lw, ls=ls, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc or color, zorder=4, linespacing=1.15)


def arrow(p, q, color=INK, lw=0.9, ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=8, shrinkA=0, shrinkB=0))


def seg(pts, color=INK, lw=0.9, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2.5, solid_capstyle="butt")


def inset(path, crop, x0, y0, w, h):
    img = mpimg.imread(path)
    (c0, c1), (r0, r1) = crop
    a = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    a.imshow(img[r0:r1, c0:c1], interpolation="lanczos"); a.set_axis_off()
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fc="none", ec="#9e9e9e", lw=0.5, zorder=3))


YB, YT = 0.45, 4.78   # panel extent
# ---------------- left: setting and challenges ----------------
panel(0.12, 2.36, YB, YT, "Freeway Forecasting Setting", GREEN_H)
inset(f"{FIG}/fig02_study_area.png", ((40, 660), (235, 935)), 0.24, 3.45, 0.86, 0.97)
inset(f"{FIG}/fig02_study_area.png", ((732, 1418), (170, 940)), 1.2, 3.45, 0.98, 0.97)
ax.text(0.67, 3.37, "California", ha="center", va="top", fontsize=6.5, style="italic", color=INK)
ax.text(1.69, 3.37, "D3 Sacramento, 388 sensors", ha="center", va="top", fontsize=6.5, style="italic", color=INK)
pill(1.24, 3.0, 1.95, "Probabilistic Flow Forecasting, 5–60 min", GREEN)
pill(1.24, 2.7, 1.95, "Event-aware Traffic Operations", GREEN)
seg([(0.12, 2.5), (2.36, 2.5)], color=EDGE, lw=0.9, ls=(0, (5, 2)))
ax.add_patch(plt.Rectangle((0.12, 2.17), 2.24, 0.28, fc=ORANGE_H, ec=EDGE, lw=1.1, zorder=2))
ax.text(1.24, 2.31, "Challenges", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=3)
pill(1.24, 1.93, 1.55, "Extreme Weather & Events", ORANGE)
ax.text(1.24, 1.75, "heat · heavy rain · wildfire smoke · crashes", ha="center", va="center", fontsize=6.8, style="italic", color=INK)
ax.text(1.24, 1.6, "+", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
pill(1.24, 1.44, 1.55, "Capacity Reduction", ORANGE)
ax.text(1.24, 1.24, "$q_{i,t}\\leq C_i\\,f(\\mathbf{w}_t),\\quad f(\\mathbf{w})<1$", ha="center", va="center", fontsize=7.5, color=INK)
ax.text(1.24, 1.08, "+", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
pill(1.24, 0.92, 1.55, "Calendar-blind Analogs", ORANGE)
ax.text(1.24, 0.74, "morning ramp-up matched to evening ramp-down", ha="center", va="center", fontsize=6.5, style="italic", color=INK)
ax.add_patch(plt.Rectangle((0.3, 0.5), 1.88, 0.17, fc="#fdf2e9", ec=ORANGE_H, lw=1.6, zorder=2.8))
dbox(1.24, 0.585, 1.82, 0.12, "Event regimes are rare in training data", "#2e4a8a", fs=6.8, lw=0.8, tc=INK)

# ---------------- middle: proposed model ----------------
X0, X1 = 2.56, 4.92
panel(X0, X1, YB, YT, "Proposed WARP-Traffic", PINK_H)
pill(3.74, 4.2, 2.0, "Weather-annotated Memory Bank", RED)
xa, xw = 3.2, 4.3
ax.text(xa, 3.98, "Traffic History", ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
ax.text(xw, 3.98, "Weather Evidence", ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
for x, t in [(xa, "$\\mathbf{X}_t$"), (xw, "$\\mathbf{W}_t$")]:
    ax.add_patch(Circle((x, 3.76), 0.1, fc="#4bacc6", ec=EDGE, lw=0.8, zorder=4))
    ax.text(x, 3.76, t, ha="center", va="center", fontsize=7.5, zorder=5, color=INK)
pill(3.74, 3.42, 2.0, "M1  Calendar-consistent Retrieval", RED)
arrow((xa, 3.66), (xa, 3.54)); arrow((xw, 3.66), (xw, 3.54))
for x, t in [(3.2, "Memory-calibrated\nBranch"), (4.3, "Deviation-aware\nBranch")]:
    ax.add_patch(plt.Rectangle((x - 0.47, 2.86), 0.94, 0.32, fc="white", ec=EDGE, lw=1.0, zorder=3))
    ax.text(x, 3.02, t, ha="center", va="center", fontsize=7, fontweight="bold", color=INK, zorder=4, linespacing=1.1)
    arrow((x, 3.3), (x, 3.18))
ax.text(4.3, 2.81, "frozen GWNet backbone", ha="center", va="top", fontsize=6.2, style="italic", color=INK)
for x in (3.2, 4.3):
    arrow((x, 2.86 if x == 3.2 else 2.7), (x, 2.6))
pill(3.74, 2.48, 2.0, "M2  Regime & Horizon Trust Gate", RED)
for x, t in [(3.2, "Regime Posterior $\\boldsymbol{\\pi}_i$\nheat, rain, smoke, crash"), (4.3, "Horizon Trust $g_{i,h}$\n12 lead times")]:
    arrow((x, 2.36), (x, 2.24), color=DASHRED)
    dbox(x, 2.04, 1.06, 0.38, t, DASHRED, fs=6.7)
seg([(3.2, 1.85), (3.2, 1.76), (4.3, 1.76), (4.3, 1.85)])
pill(3.42, 1.55, 1.5, "M3  Capacity Bound $C_i f(\\mathbf{w})$", RED, fs=7.2)
arrow((3.74, 1.76), (3.74, 1.67))
arrow((4.17, 1.55), (4.32, 1.55))
ax.add_patch(Circle((4.45, 1.55), 0.12, fc="#b4a7d6", ec=EDGE, lw=0.8, zorder=4))
ax.text(4.45, 1.55, "$F^{\\mathrm{mem}}$", ha="center", va="center", fontsize=6.6, zorder=5)
# weather evidence also sets the capacity bound

# two experts (actor/critic style)
ex = [(3.15, "Parametric Expert", "#c0392b", "#7f8c8d"), (4.33, "Memory Expert", "#f1c40f", "#c0504d")]
for cx, lab, ec, lc in ex:
    ax.add_patch(plt.Rectangle((cx - 0.5, 0.58), 1.0, 0.6, fc="white", ec=ec, lw=1.2, zorder=3))
    ax.add_patch(plt.Rectangle((cx - 0.42, 1.21), 0.84, 0.15, fc="white", ec="#7b2cbf", lw=0.8, zorder=3))
    ax.text(cx, 1.285, lab, ha="center", va="center", fontsize=6.8, fontweight="bold", style="italic", color="#1f3a93", zorder=4)
    z = np.linspace(-3, 3, 200)
    y = np.exp(-z ** 2 / 1.3) if "Param" in lab else 0.6 * np.exp(-(z + 1.1) ** 2 / 0.35) + 0.9 * np.exp(-(z - 0.7) ** 2 / 0.5)
    y = y / y.max()
    ax.fill_between(cx + z * 0.14, 0.74, 0.74 + 0.36 * y, color=lc, alpha=0.25, lw=0, zorder=3.5)
    ax.plot(cx + z * 0.14, 0.74 + 0.36 * y, color=lc, lw=1.0, zorder=3.6)
ax.text(3.15, 0.6, "GWNet-Gaussian", ha="center", va="bottom", fontsize=6.0, color=INK, zorder=4)
ax.text(4.33, 0.6, "GMM from analogs", ha="center", va="bottom", fontsize=6.0, color=INK, zorder=4)
arrow((4.45, 1.43), (4.45, 1.36))

# ---------------- right: training and evaluation ----------------
R0, R1 = 5.12, 7.08
panel(R0, R1, YB, YT, "Training & Evaluation", TEAL_H)
pill(6.1, 4.2, 1.6, "M5  Linear Pooling", TEAL)
ax.text(6.1, 3.96, "$F=\\lambda F^{\\mathrm{mem}}+(1-\\lambda)\\,F^{\\mathrm{par}},\\ \\lambda=0.5$", ha="center", va="center", fontsize=7.2, color=INK)
ax.text(6.1, 3.8, "truncated to $[0,\\,U_{i,t}]$", ha="center", va="center", fontsize=7.2, color=INK)
dbox(5.62, 3.36, 0.86, 0.36, "M4 Rare-event\nOversampling", TEAL, fs=7.0)
dbox(6.58, 3.36, 0.86, 0.36, "CRPS-based\nTraining", TEAL, fs=7.0)
seg([(5.62, 3.18), (5.62, 2.98)]); seg([(6.58, 3.18), (6.58, 2.98)]); seg([(5.62, 2.98), (6.58, 2.98)])
arrow((6.1, 2.98), (6.1, 2.86))
pill(6.1, 2.74, 1.6, "Predictive Distribution", TEAL)
arrow((6.1, 2.62), (6.1, 2.5))
dbox(6.1, 2.37, 1.5, 0.22, "Quantiles · Trust · Regimes", TEAL, fs=7.0)
arrow((6.1, 2.26), (6.1, 2.14))
pill(6.1, 2.02, 1.6, "Event-stratified Evaluation", TEAL)
inset(f"{FIG}/fig01_motivation_D3.png", ((1795, 2470), (262, 950)), 5.52, 0.58, 1.16, 1.16)
ax.text(6.1, 1.83, "24 Oct 2021 atmospheric river, 30 min ahead", ha="center", va="center", fontsize=6.2, style="italic", color=INK)

# experts -> M5 (bracket under the experts, then up between panels)
seg([(3.15, 0.58), (3.15, 0.52), (4.33, 0.52), (4.33, 0.58)])
seg([(4.33, 0.52), (5.02, 0.52), (5.02, 4.2)])
arrow((5.02, 4.2), (5.3, 4.2))

# setting -> model: dotted from the challenge box to the traffic input, which also feeds the parametric expert
seg([(2.18, 0.585), (2.46, 0.585), (2.46, 3.76)], ls=(0, (1.5, 1.5)))
seg([(2.46, 3.76), (2.62, 3.76)])
arrow((2.62, 3.76), (xa - 0.1, 3.76))
ax.plot(2.62, 3.76, "o", ms=2.5, color=INK, zorder=6)
seg([(2.62, 3.76), (2.62, 0.88)])
arrow((2.62, 0.88), (2.65, 0.88))

# bottom loop: same design, held-out districts
seg([(6.1, YB), (6.1, 0.17), (0.04, 0.17), (0.04, 4.0)])
arrow((0.04, 4.0), (0.12, 4.0))
ax.text(3.07, 0.255, "Same Design Run Unchanged in Held-out Districts: D4 Bay Area & D7 Los Angeles",
        ha="center", va="center", fontsize=8.3, fontweight="bold", color=INK)

save(fig, "fig00_overview")
print("ok")
