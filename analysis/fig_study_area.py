"""Fig. 2: study districts, sensors (coloured by local extreme-heat threshold) and weather stations."""
import numpy as np, pandas as pd, geopandas as gpd
from style import setup, save, plt, REGION_NAME

import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
P = os.path.join(ROOT, "data_proc")
setup()
st = gpd.read_file("zip://" + os.path.join(ROOT, "data_raw", "census", "cb_2018_us_state_20m.zip"))
co = gpd.read_file("zip://" + os.path.join(ROOT, "data_raw", "census", "cb_2018_us_county_20m.zip"))
ca, cac = st[st.STUSPS == "CA"], co[co.STATEFP == "06"]
ws = pd.read_csv(f"{P}/weather/asos_stations.csv")
pm = pd.read_csv(f"{P}/weather/pm25_sites.csv")
fig = plt.figure(figsize=(7.2, 2.9))
ax0 = fig.add_axes([0.0, 0.05, 0.22, 0.9])
ca.plot(ax=ax0, color="#f4f6f6", edgecolor="#7f8c8d", lw=0.5)
boxes = {}
metas = {}
for d in ["D3", "D4", "D7"]:
    m = pd.read_csv(f"{P}/regions/{d}/meta.csv")
    if d != "D3":
        m = m.iloc[np.load(f"{P}/regions/{d}/subset_core.npy")]
    metas[d] = m
    b = (m.Lng.min() - 0.08, m.Lat.min() - 0.08, m.Lng.max() + 0.08, m.Lat.max() + 0.08)
    boxes[d] = b
    ax0.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="k", lw=0.8))
    ax0.text(b[2] + 0.15, b[3], d, fontsize=7, va="top")
ax0.set_axis_off(); ax0.set_xlabel(""); ax0.set_ylabel("")
ax0.set_title("(a) California", loc="left")
vmin = min(m.T_thr_F.min() for m in metas.values()); vmax = max(m.T_thr_F.max() for m in metas.values())
for i, d in enumerate(["D3", "D4", "D7"]):
    ax = fig.add_axes([0.24 + i * 0.255, 0.2, 0.24, 0.66])
    b, m = boxes[d], metas[d]
    cac.plot(ax=ax, color="#fbfcfc", edgecolor="#d5d8dc", lw=0.4)
    sc = ax.scatter(m.Lng, m.Lat, c=m.T_thr_F, s=3, cmap="YlOrRd", vmin=vmin, vmax=vmax, lw=0)
    w = ws[(ws.lon.between(b[0] - 0.3, b[2] + 0.3)) & (ws.lat.between(b[1] - 0.3, b[3] + 0.3))]
    ax.scatter(w.lon, w.lat, marker="^", s=14, c="#1f618d", ec="w", lw=0.3, label="ASOS station")
    p = pm[(pm.Longitude.between(b[0] - 0.3, b[2] + 0.3)) & (pm.Latitude.between(b[1] - 0.3, b[3] + 0.3))]
    ax.scatter(p.Longitude, p.Latitude, marker="s", s=8, c="#27ae60", ec="w", lw=0.3, label="PM$_{2.5}$ monitor")
    pad = 0.25
    ax.set_xlim(b[0] - pad, b[2] + pad); ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad(m.Lat.mean())), adjustable="datalim")   # equal frames; the map extent adapts
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(True); s.set_lw(0.5)
    ax.set_title(f"({'bcd'[i]}) {REGION_NAME[d]}" + "\n" + f"{len(m)} sensors", loc="left")
    if i == 0:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.03), ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.0)
cax = fig.add_axes([0.62, 0.115, 0.25, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_label("Local extreme-heat threshold (°F)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
save(fig, "fig02_study_area")
print("ok")
