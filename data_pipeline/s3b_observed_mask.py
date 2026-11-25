"""Save, per region, which flow values were actually observed (before the short-gap interpolation of s3_traffic).

observed.npy  [T, N] bool, same rows/columns as flow.npy. Interpolated values (gaps <= 30 min) are False.
Used at evaluation time so that (i) model inputs are filled causally and (ii) only observed targets are scored.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
TR = os.path.join(ROOT, "data_proc", "traffic")
OUT = os.path.join(ROOT, "data_proc", "regions")
YEARS = [2017, 2018, 2019, 2020, 2021]


def main():
    rep = {}
    for d in [3, 4, 7]:
        od = os.path.join(OUT, f"D{d}")
        cols = np.load(os.path.join(TR, f"D{d}_cols.npy"))
        gcols = pd.read_csv(os.path.join(od, "meta.csv"))["col"].to_numpy()
        pos = {c: i for i, c in enumerate(cols)}
        k = np.array([pos[c] for c in gcols])
        raw = np.concatenate([np.load(os.path.join(TR, f"D{d}_{y}.npy"), mmap_mode="r")[:, k] for y in YEARS])
        obs = ~np.isnan(raw)
        flow = np.load(os.path.join(od, "flow.npy"), mmap_mode="r")
        assert flow.shape == obs.shape, (flow.shape, obs.shape)
        valid = ~np.isnan(flow)
        assert not (obs & ~valid).any(), "observed value missing after interpolation"
        same = np.allclose(np.asarray(flow)[obs], raw[obs], equal_nan=False)
        np.save(os.path.join(od, "observed.npy"), obs)
        rep[f"D{d}"] = dict(valid=float(valid.mean()), observed=float(obs.mean()),
                            interpolated_share_of_valid=float((valid & ~obs).sum() / valid.sum()), values_match=bool(same))
        print(d, rep[f"D{d}"], flush=True)


if __name__ == "__main__":
    main()
