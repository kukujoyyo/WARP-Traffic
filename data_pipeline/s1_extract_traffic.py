"""S1: extract District 3/4/7 mainline flow from LargeST CA yearly h5 files (5-min, 2017-2021)."""
import sys, os, h5py, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RAW = os.path.join(ROOT, "data_raw", "largest")
OUT = os.path.join(ROOT, "data_proc", "traffic")
os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(os.path.join(RAW, "metadata.csv"))
years = [int(y) for y in sys.argv[1:]] or [2017, 2018, 2019, 2020, 2021]
for y in years:
    f = h5py.File(os.path.join(RAW, f"{y}.h5"), "r")
    ids = np.array([int(s) for s in f["t/block0_items"][:]])
    assert (ids == meta.ID.values).all()
    ts = pd.to_datetime(f["t/axis1"][:])
    vals = f["t/block0_values"][:]  # (T, 8600) float64
    for d in [3, 4, 7]:
        cols = np.where((meta.District == d) & (meta.Type == "Mainline"))[0]
        np.save(os.path.join(OUT, f"D{d}_{y}.npy"), vals[:, cols].astype(np.float32))
        np.save(os.path.join(OUT, f"D{d}_cols.npy"), cols)
    np.save(os.path.join(OUT, f"time_{y}.npy"), ts.values.astype("datetime64[m]"))
    print(y, vals.shape, ts[0], ts[-1], np.isnan(vals).mean(), flush=True)
    del vals
