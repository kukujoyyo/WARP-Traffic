"""Non-learned references: historical average (HA), weekly seasonal naive (SN), and
climatology (empirical time-of-week quantiles; the reference forecast for the Value Score)."""
import os, json, argparse, numpy as np, torch
from wremind.data import Region
from wremind.metrics import Accumulator
from wremind.train import historical_tables, RES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="D3")
    p.add_argument("--subset", default="")
    p.add_argument("--test_stride", type=int, default=6)
    p.add_argument("--reeval", type=int, default=0)
    a = p.parse_args()
    R = Region(a.region, node_subset=np.load(a.subset) if a.subset else None)
    HA, CL = historical_tables(R)
    te = R.origins("test", a.test_stride)
    for name in ["ha", "snaive", "climatology"]:
        acc = Accumulator(len(te))
        extra = {k: [] for k in ["pnear", "pdrop", "onear", "odrop", "valid", "t0"]}
        gen = torch.Generator(device="cuda").manual_seed(0)
        for s in range(0, len(te), 32):
            b = R.batch(te[s:s + 32])
            fut = b["t0"][:, None] + torch.arange(1, 13, device="cuda")
            y = b["y"] * R.std + R.mean
            smp = None
            if name == "ha":
                pt = HA[R.dow_idx[fut], R.tod_idx[fut]]
            elif name == "snaive":
                pt = R.z[fut - 2016] * R.std + R.mean
            else:
                we = (R.dow_idx[fut] >= 5).long()
                q = CL[we, R.tod_idx[fut]]                                  # [B,T,N,Q]
                smp = q.permute(0, 3, 1, 2)                                 # quantile draws as samples
                pt = q[..., q.size(-1) // 2]
            up = R.cap_max[None] * R.cap_factor[R.hour_idx[b["t0"]]]
            up_near = R.cap[None] * R.cap_factor[R.hour_idx[b["t0"]]]
            acc.update(smp, pt, y, b["ymask"], b["ev_f"], torch.zeros_like(up), up)
            ha = HA[R.dow_idx[fut], R.tod_idx[fut]]
            day = ((R.tod_idx[fut] >= 72) & (R.tod_idx[fut] < 252))[..., None]
            hs = [2, 5, 11]
            thr_near, thr_drop = 0.9 * up_near[:, None, :], 0.6 * ha
            if smp is not None:
                pn = (smp[:, :, hs] >= thr_near[:, None]).float().mean(1)
                pdp = (smp[:, :, hs] <= thr_drop[:, None, hs]).float().mean(1)
            else:
                pn = (pt[:, hs] >= thr_near).float(); pdp = (pt[:, hs] <= thr_drop[:, hs]).float()
            extra["pnear"].append(pn.half().cpu()); extra["pdrop"].append(pdp.half().cpu())
            extra["onear"].append((y[:, hs] >= thr_near).cpu()); extra["odrop"].append(((y[:, hs] <= thr_drop[:, hs]) & day[:, hs]).cpu())
            extra["valid"].append((b["ymask"][:, hs] & day[:, hs]).cpu()); extra["t0"].append(b["t0"].cpu())
        od = os.path.join(RES, a.region, f"{name}_s0")
        os.makedirs(od, exist_ok=True)
        json.dump(acc.summary(), open(os.path.join(od, "summary.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(od, "series.npz"), t0=te, **acc.series)
        np.savez_compressed(os.path.join(od, "extra.npz"), **{k: torch.cat(v).numpy() for k, v in extra.items()})
        s = acc.summary()["all"]["avg"]
        print(a.region, name, {k: round(v, 3) for k, v in s.items()})


if __name__ == "__main__":
    main()
