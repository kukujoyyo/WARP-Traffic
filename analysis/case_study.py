"""Case studies: 5-min-stride forecasts on selected event days for selected sensors.
Saves per-model predictive quantiles (30-min horizon), observations, and (REMIND family) trust gate / regime posterior.
usage: python -m analysis.case_study D3 gwnet_gauss_s0 remind_s0 wremind_s0 wremindpool_s0=wremind_s0+gwnet_gauss_s0 ...
"""
import os, sys, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wremind.data import Region, Retriever
from wremind import train as T
from wremind.models import sample_mixture

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
OUT = os.path.join(ROOT, "results", "cases")
DAYS = {"rain_ar": "2021-10-24", "heat": "2021-06-17", "smoke": "2021-08-27", "normal": "2021-10-13"}
H = 5  # 30-min horizon index


def main():
    region, runs = sys.argv[1], sys.argv[2:]
    os.makedirs(OUT, exist_ok=True)
    R = Region(region)
    tstamp = pd.to_datetime(np.load(os.path.join(ROOT, "data_proc", "regions", region, "time.npy")))
    origins = {}
    for k, d in DAYS.items():
        idx = np.where((tstamp >= pd.Timestamp(d) + pd.Timedelta("4h")) & (tstamp < pd.Timestamp(d) + pd.Timedelta("23h")))[0]
        origins[k] = idx - H - 1
    allo = np.concatenate(list(origins.values()))
    res = {"t0": allo, "y": None}
    import wremind.pool as P
    P.R_GLOBAL = R
    for spec in runs:
        # "name=a+b" is an equal-weight linear pool of runs a and b (M5); otherwise a single trained run
        run, parts = (spec.split("=")[0], spec.split("=")[1].split("+")) if "=" in spec else (spec, [spec])
        experts = [P.expert(p, region, R) for p in parts]
        gen = torch.Generator(device="cuda").manual_seed(2025)
        Q, G, PI, Y, U = [], [], [], [], []
        S = 200
        with torch.no_grad():
            for s in range(0, len(allo), 16):
                b = R.batch(allo[s:s + 16])
                if len(experts) == 2:      # exact truncated equal-weight pool (as in wremind.pool)
                    sa, out, qa = P.draw(experts[0], b, S, gen, bounded=True)
                    sb, _, qb = P.draw(experts[1], b, S, gen, bounded=True)
                    wa, wb = 0.5 * qa, 0.5 * qb
                    pa = torch.where(wa + wb > 1e-12, wa / (wa + wb).clamp_min(1e-12), torch.full_like(wa, 0.5))
                    uc = torch.rand(sa.shape[0], S, 1, sa.shape[-1], device="cuda", generator=gen)
                    smp = torch.where(uc < pa[:, None], sa, sb)
                else:
                    smp, out, _ = P.draw(experts[0], b, S, gen)
                if out is not None:
                    gr = out["gate_raw"].float()
                    G.append((gr if gr.dim() == 2 else gr[..., H]).cpu())
                    if "regime_logits" in out:
                        PI.append(torch.sigmoid(out["regime_logits"]).cpu())
                smn = smp[:, :, H] * R.std + R.mean                                  # [B,S,N]
                Q.append(torch.quantile(smn.float(), torch.tensor([.05, .25, .5, .75, .95], device="cuda"), dim=1).permute(1, 0, 2).cpu())
                Y.append((b["y"][:, H] * R.std + R.mean).cpu())
                U.append((R.cap_max[None] * R.cap_factor[R.hour_idx[b["t0"]]]).cpu())
        res[f"{run}_q"] = torch.cat(Q).numpy().astype(np.float32)
        if G:
            res[f"{run}_gate"] = torch.cat(G).numpy().astype(np.float16)
        if PI:
            res[f"{run}_pi"] = torch.cat(PI).numpy().astype(np.float16)
        res["y"] = torch.cat(Y).numpy().astype(np.float32)
        res["upper"] = torch.cat(U).numpy().astype(np.float32)
        print(run, "done", flush=True)
        del experts
        torch.cuda.empty_cache()
    res["day_key"] = np.concatenate([[k] * len(v) for k, v in origins.items()])
    res["ev"] = np.asarray(R.events[torch.as_tensor(allo + H + 1, device=R.events.device)].cpu())
    np.savez_compressed(os.path.join(OUT, f"cases_{region}.npz"), **res)


if __name__ == "__main__":
    main()
