import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
FIG = os.path.join(ROOT, "paper", "figures")
COL = {"D3": "#c0392b", "D4": "#2471a3", "D7": "#7d3c98"}
MODEL_COL = {"WARP-Traffic": "#c0392b", "REMIND": "#2471a3", "REMIND+W": "#5dade2", "GWNet-G": "#7f8c8d",
             "STAEformer-Q": "#16a085", "DeepAR": "#d68910", "Climatology": "#aab7b8"}
REGION_NAME = {"D3": "Sacramento (D3)", "D4": "Bay Area core (D4)", "D7": "Los Angeles core (D7)"}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
                         "axes.linewidth": 0.6, "lines.linewidth": 1.0, "pdf.fonttype": 42})


def top_legend(fig, axs, ncol, handles=None, labels=None, over=None, pad=0.012, **kw):
    """Legend above the panels (never over data): placed just above the highest axes title, centred on `over` axes."""
    axs = list(axs)
    if handles is None:
        seen = {}
        for ax in axs:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if not l.startswith("_") and l not in seen:
                    seen[l] = h
        handles, labels = list(seen.values()), list(seen)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    tops = []
    for a in axs:   # highest drawn title (left, centre or right) or the axes top
        t = [x for x in (a._left_title, a.title, a._right_title) if x.get_text()]
        tops += [inv.transform(x.get_window_extent())[1, 1] for x in t] or [a.get_position().y1]
    top = max(tops)
    span = over if over is not None else axs
    x0, x1 = min(a.get_position().x0 for a in span), max(a.get_position().x1 for a in span)
    kw = {**dict(frameon=False, handlelength=1.6, columnspacing=1.4, handletextpad=0.5, fontsize=6.8), **kw}
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=((x0 + x1) / 2, top + pad), ncol=ncol, **kw)


def save(fig, name):
    import os
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
