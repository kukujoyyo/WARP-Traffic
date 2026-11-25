"""Fig. 5: memory expert of WARP-Traffic at tensor level (retrieval M1, branches, gate M2, mixture and bound M3, losses).
Drawn in the block-and-tensor style of the overview (Fig. 2); pooling (M5) is shown there, not here."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
from style import save

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "savefig.dpi": 400,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
W, H = 7.6, 4.55
FS, VS = 1.3, 1.22        # font scale and vertical stretch: text >= 6 pt once scaled to the text width
fig = plt.figure(figsize=(W, H * VS))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()

INK, EDGE = "#1a1a1a", "#3a3a3a"
DRED, PINK, LRED = "#bdd7ee", "#f4b6c2", "#c00000"   # modules light blue, losses light pink
TEALC, GREENC, ORANGEC, YELC, GREYC = "#2fa4c4", "#7cc242", "#f79646", "#fff7c2", "#bfbfbf"
BLUE = "#2e5c9a"
CW, CH = 0.105, 0.1          # tensor cell size


def module(cx, cy, w, h, text, fc=DRED, fs=6.5, tc=INK, italic_first=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=fc, ec=EDGE, lw=0.9, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=4, linespacing=1.2,
            style="italic" if italic_first else "normal")
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def cells(x, y, colors, dots=None):
    for k, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((x + k * CW, y), CW, CH, fc=c, ec=EDGE, lw=0.5, zorder=3))
        if dots is not None and k == dots:
            ax.text(x + (k + 0.5) * CW, y + CH * 0.55, "···", ha="center", va="center", fontsize=6, zorder=4)
    return x + len(colors) * CW


def stack(x, y, colors, dots, rows=3, dx=0.05, dy=-0.135):
    """three offset rows of cells, top row at (x, y)"""
    for r in range(rows):
        cells(x + r * dx, y + r * dy, colors, dots)
    ax.text(x + dx + len(colors) * CW / 2, y + dy + CH + 0.005, "···", ha="center", va="bottom", fontsize=5, zorder=4)
    return x, y + (rows - 1) * dy, x + (rows - 1) * dx + len(colors) * CW, y + CH


def brace_top(x0, x1, y, text):
    ax.plot([x0, x0, x1, x1], [y + 0.01, y + 0.05, y + 0.05, y + 0.01], color=INK, lw=0.6)
    ax.plot([(x0 + x1) / 2] * 2, [y + 0.05, y + 0.08], color=INK, lw=0.6)
    ax.text((x0 + x1) / 2, y + 0.09, text, ha="center", va="bottom", fontsize=6.6, color=INK)


def brace_left(x, y0, y1, text):
    ax.text(x, (y0 + y1) / 2, "{", ha="center", va="center", fontsize=17, color=INK, family="DejaVu Sans")
    ax.text(x - 0.06, (y0 + y1) / 2, text, ha="right", va="center", fontsize=6.6, color=INK)


def arrow(p, q, color=INK, lw=0.8, ls="-"):
    ax.annotate("", xy=q, xytext=p, zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=7, shrinkA=0, shrinkB=0))


def seg(pts, color=INK, lw=0.8, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2.5, solid_capstyle="butt")


def label(x, y, text, fs=6.7, **kw):
    ax.text(x, y, text, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), fontsize=fs, color=kw.pop("color", INK), zorder=6, **kw)


# ---------------- inputs ----------------
xq = stack(0.42, 3.62, [TEALC] * 2 + [TEALC] + [TEALC] * 2, dots=2)
brace_top(0.42, 0.42 + 5 * CW, 3.75, "$T_h=12$")
brace_left(0.33, xq[1], xq[3], "$N$")
label(0.74, 3.22, "Query History $\\mathbf{X}_t$\n$12\\times N$", va="top")

xm = stack(0.42, 2.56, [TEALC] * 2 + [GREYC] + [GREENC] * 2, dots=2)
brace_top(0.42, 0.42 + 5 * CW, 2.69, "history $|$ future")
brace_left(0.33, xm[1], xm[3], "$|\\mathcal{M}|$")
label(0.74, 2.16, "Memory Bank $\\mathcal{M}$\n124,016 windows\nPCA-64 keys", va="top")

xw = stack(0.42, 1.06, [ORANGEC] * 2 + [ORANGEC] + [ORANGEC] * 2, dots=2)
brace_top(0.42, 0.42 + 5 * CW, 1.19, "$F=8$ variables")
brace_left(0.33, xw[1], xw[3], "$N$")
label(0.74, 0.66, "Weather $\\mathbf{W}_t$\n$12\\times N\\times 8$", va="top")

# ---------------- M1 retrieval ----------------
c1 = module(2.08, 3.42, 1.26, 0.62, "M1 Coarse Retrieval\nsame day type,\n±30 min, top $M=30$")
c2 = module(2.08, 2.45, 1.26, 0.62, "Fine Retrieval\nsensor and $J=4$\nneighbours, top $K=8$")
arrow((xq[2] + 0.02, 3.5), (c1[0], 3.5))
seg([(xm[2] + 0.02, 2.42), (1.3, 2.42), (1.3, 3.32)]); arrow((1.3, 3.32), (c1[0], 3.32))
arrow((2.05, c1[1]), (2.05, c2[3]))

# analog tensor
xa0 = 2.9
xa = stack(xa0, 2.72, [TEALC] * 3 + [GREENC] * 3, dots=None)
ax.plot([xa0 + 3 * CW] * 2, [xa[1] - 0.01, 2.72 + CH + 0.01], color=INK, lw=0.9, zorder=4)
brace_top(xa0, xa0 + 6 * CW, 2.85, "$\\tilde{\\mathbf{x}}$ (12) $|$ $\\tilde{\\mathbf{y}}$ (12)")
ax.text(xa[2] + 0.05, (xa[1] + xa[3]) / 2, "}", ha="center", va="center", fontsize=17, color=INK, family="DejaVu Sans")
label(xa[2] + 0.1, (xa[1] + xa[3]) / 2, "$K$", ha="left", fs=6.8)
label((xa0 + xa[2]) / 2, xa[1] - 0.08, "Retrieved Analogs", va="top", fontweight="bold")
arrow((c2[2], 2.5), (xa0 - 0.02, 2.5))

# weather annotation of analogs (evidence, not ranking)
wa = (3.3, 1.72)
ax.add_patch(plt.Rectangle((wa[0] - 0.65, wa[1] - 0.17), 1.3, 0.34, fc="white", ec=BLUE, lw=0.9, ls=(0, (4, 2)), zorder=3))
label(wa[0], wa[1], "Weather Annotation\n$s^{\\mathrm{wx}}$, event flags $\\bar{\\mathbf{r}}_i$", fs=6.6, color=BLUE, fontweight="bold")
arrow((wa[0], xa[1] - 0.24), (wa[0], wa[1] + 0.17), color=BLUE)
seg([(xw[2] + 0.02, 0.95), (1.55, 0.95), (1.55, wa[1])], color=BLUE)
arrow((1.55, wa[1]), (wa[0] - 0.58, wa[1]), color=BLUE)
label(1.62, 1.78, "not used for ranking", fs=6.2, color=BLUE, style="italic", ha="left")

# ---------------- branches ----------------
BX = 4.88
b1 = module(BX, 3.42, 1.3, 0.62, "Memory-calibrated\nBranch: attention\nover $K$ analogs\n4 Gaussian components")
b2 = module(BX, 2.42, 1.3, 0.56, "Deviation-aware Branch\n$\\boldsymbol{\\mu}^{\\mathrm{GW}}+\\mathrm{Lin}(\\mathbf{h})+\\gamma\\,\\mathrm{Lin}(\\mathbf{d})$\n4 Gaussian components")
jx = 4.08
seg([(xa[2] + 0.22, 2.55), (jx, 2.55), (jx, 3.42)]); arrow((jx, 3.42), (b1[0], 3.42))
arrow((jx, 2.55), (b2[0], 2.55)); ax.plot(jx, 2.55, "o", ms=2.2, color=INK, zorder=6)
# frozen backbone feeds the deviation-aware branch; gradient stopped
bb = module(BX, 1.68, 1.3, 0.5, "GWNet-G+W\nBackbone (frozen)\non $\\mathbf{X}_t$, $\\mathbf{W}_t$, calendar", fc="#d9d9d9", fs=6.2)
arrow((BX, bb[3]), (BX, b2[1]))
for dy in (0, 0.04):
    ax.plot([BX - 0.08, BX + 0.08], [1.97 + dy, 2.03 + dy], color=LRED, lw=0.9, zorder=6)
label(BX + 0.12, 2.03, "Stop Gradient", fs=6.3, color=LRED, ha="left")

# ---------------- M2 gate ----------------
gx = 2.78
gcols = [TEALC] * 4 + [ORANGEC] * 2 + [GREENC] * 3
for r in range(2):
    cells(gx, 1.02 - r * 0.15, gcols)
label(gx + 4.5 * CW, 1.2, "Gate Features (concatenation)", fs=6.7, va="bottom", fontweight="bold")
label(gx + 4.5 * CW, 0.78, "$[\\bar s^{\\mathrm{tr}}_i, s^{\\mathrm{tr}}_{i,1}, c^{\\mathrm{mem}}_i, \\|\\mathbf{d}_i\\|$ $|$ $\\bar s^{\\mathrm{wx}}_i, s^{\\mathrm{wx}}_{i,1}$ $|$ $\\mathbf{e}_i, \\bar{\\mathbf{r}}_i, \\mathbf{u}_i]$", fs=6.4, va="top")
arrow((wa[0], wa[1] - 0.17), (wa[0], 1.33), color=BLUE)
seg([(xw[2] + 0.02, 0.86), (2.55, 0.86), (2.55, 0.95)])
arrow((2.55, 0.95), (gx, 0.95))
m2 = module(BX, 0.95, 1.0, 0.44, "M2 Gate MLP\nhidden = 64", fs=6.8)
arrow((gx + 9 * CW + 0.02, 0.95), (m2[0], 0.95))

# heads
hx = 5.68
YP = 1.24                 # regime-posterior row
cells(hx, YP - 0.05, [YELC] * 4)
label(hx - 0.02, YP + 0.08, "Regime Posterior $\\boldsymbol{\\pi}_i$", fs=6.7, va="bottom", ha="left", fontweight="bold")
label(hx - 0.02, YP - 0.08, "4 Sigmoid:\nheat, rain, smoke,\ncrash", fs=6.0, va="top", ha="left", style="italic")
cells(hx, 0.52, [YELC] * 2 + [YELC] + [YELC] * 3, dots=2)
label(hx - 0.02, 0.66, "Horizon Trust $g_{i,h}$", fs=6.7, va="bottom", ha="left", fontweight="bold")
label(hx - 0.02, 0.46, "12 Sigmoid on $[\\mathbf{h}_i;\\boldsymbol{\\pi}_i]$", fs=6.0, va="top", ha="left", style="italic")
seg([(m2[2], 1.05), (5.55, 1.05), (5.55, YP)]); arrow((5.55, YP), (hx, YP))
seg([(m2[2], 0.85), (5.55, 0.85), (5.55, 0.57)]); arrow((5.55, 0.57), (hx, 0.57))

# ---------------- mixture and M3 ----------------
MX = 6.28
mx = module(MX, 2.42, 0.98, 0.72, "Horizon-wise\nMixture\n$g_{i,h}\\,$GMM$^{\\mathrm{mem}}$\n$+(1-g_{i,h})\\,$GMM$^{\\mathrm{rob}}$", fs=6.1)
m3 = module(MX, 3.5, 0.98, 0.7, "M3 Capacity\nBound: truncate\nto $[0,U_{i,t}]$,\n$U_{i,t}=C_i\\,f(\\mathbf{w}_{i,t})$", fs=6.4)
seg([(b1[2], 3.42), (5.64, 3.42), (5.64, 2.55)]); arrow((5.64, 2.55), (mx[0], 2.55))
arrow((b2[2], 2.32), (mx[0], 2.32))
seg([(hx + 6 * CW + 0.02, 0.57), (6.69, 0.57)]); arrow((6.69, 0.57), (6.69, mx[1]))
arrow((MX, mx[3]), (MX, m3[1]))
label(MX + 0.05, 2.97, "inference", fs=6.0, style="italic", ha="left")
ax.add_patch(Ellipse((MX, 4.07), 0.32, 0.32 / VS, fc="#b4a7d6", ec=EDGE, lw=0.8, zorder=4))
label(MX, 4.07, "$F^{\\mathrm{mem}}$", fs=5.8)
arrow((MX, m3[3]), (MX, 3.94))
label(MX - 0.17, 4.07, "to M5 (Fig. 2)", fs=6.2, ha="right", style="italic")

# ---------------- losses ----------------
lx = 7.13
l_crps = module(lx, 2.62, 0.62, 0.42, "CRPS\n(closed\nform)", fc=PINK, fs=6.3)
l_cap = module(lx, 2.12, 0.62, 0.42, "$\\mathcal{L}_{\\mathrm{cap}}$\nmass\noutside", fc=PINK, fs=6.3)
l_gate = module(lx, YP, 0.62, 0.42, "$\\mathcal{L}_{\\mathrm{gate}}$\nweighted\nBCE", fc=PINK, fs=6.3)
lab = module(lx, 0.58, 0.66, 0.56, "Event\nLabels $\\tilde y_{i,r}$\nheat, rain,\nsmoke, crash", fc="#9dc3e6", fs=5.8)
arrow((mx[2], 2.62), (l_crps[0], 2.62)); arrow((mx[2], 2.18), (l_cap[0], 2.18))
arrow((hx + 4 * CW + 0.02, YP), (l_gate[0], YP))
arrow((lx, lab[3]), (lx, l_gate[1]))
bus = 7.52
for b in (l_crps, l_cap, l_gate):
    seg([(b[2], (b[1] + b[3]) / 2), (bus, (b[1] + b[3]) / 2)])
    ax.plot(bus, (b[1] + b[3]) / 2, "o", ms=2.2, color=INK, zorder=6)
YT = 4.3
bp = module(3.4, YT, 1.5, 0.36, "Backpropagation:\nBranches & Gate", fc=PINK, fs=6.6)
seg([(bus, YP), (bus, YT), (bp[2] + 0.1, YT)])
arrow((bp[2] + 0.1, YT), (bp[2], YT))
arrow((BX, YT), (BX, b1[3])); ax.plot(BX, YT, "o", ms=2.2, color=INK, zorder=6)
label(7.5, YT + 0.04, "Total Loss $\\mathcal{L}=30\\,\\overline{\\mathrm{CRPS}}+0.05\\,\\mathcal{L}_{\\mathrm{NLL}}+0.2\\,\\mathcal{L}_{\\mathrm{gate}}+0.5\\,\\mathcal{L}_{\\mathrm{cap}}$",
      fs=6.7, va="bottom", ha="right", fontweight="bold", style="italic")

for t in fig.findobj(plt.Text):
    t.set_fontsize(t.get_fontsize() * FS)
save(fig, "fig04_framework")
print("ok")
