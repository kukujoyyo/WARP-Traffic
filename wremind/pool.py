"""Forecaster-level trust pooling (final WARP-Traffic forecast).

The memory expert (retrieval-guided, regime-aware) and a parametric probabilistic STGNN are combined as a linear
pool p = w * p_mem + (1 - w) * p_par, sampled by drawing round(w*S) trajectories from the memory expert and the rest
from the parametric expert. With --truncate 1 the pool is truncated to the admissible interval (M3): each expert is
sampled from its truncated distribution and the pool weight is reweighted by the experts' in-bound probability masses
(w q_a / (w q_a + (1-w) q_b), per node and lead time), i.e. exact sampling from the truncated linear pool. Outputs follow
train.evaluate (summary.json, series.npz, extra.npz) so that all analysis scripts apply unchanged.

usage: python -m wremind.pool --region D3 --a wremind_v3_s0 --b gwnet_gauss_s0 --name wremindpool --seed 0
"""
import os, sys, json, argparse, time, numpy as np, torch
from easydict import EasyDict as edict
from wremind.data import Region, Retriever
from wremind.metrics import Accumulator
import torch.nn.functional as F
from wremind.models import sample_mixture, mixture_weights, truncated_gauss
from wremind import train as T


def expert(run, region, R):
    """Rebuild a trained model from the argument record in its train.log."""
    d = os.path.join(T.RES, region, run)
    rec = [l for l in open(os.path.join(d, "train.log")) if l.startswith("{")][-1]
    rec = json.loads(rec[:rec.rindex("}") + 1])
    sys_argv = sys.argv
    sys.argv = ["x"]
    a = T.get_args()
    sys.argv = sys_argv
    # options introduced during development take the value that applied when an older run was trained
    for k, v in dict(hgate=0, loss="nll", tod_window=0, backbone="none", nll_weight=0.05, bb_freeze=1).items():
        setattr(a, k, v)
    for k, v in rec.items():
        setattr(a, k, v)
    a.bb_init = ""                                        # weights come from model.pt
    m, head, cfg = T.build(a, R)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt")))
    m.eval()
    ret = None
    if head == "remind":
        ret = Retriever(R, M=a.M, K=a.K, joint=cfg.joint, alpha=a.alpha, tau_w=a.tau_w, seed=a.seed,
                        tod_window=getattr(a, "tod_window", 0) if cfg.joint else 0,
                        bank_start=None)
    return edict(m=m, head=head, cfg=cfg, ret=ret, args=a)


@torch.no_grad()
def draw(e, b, S, gen, bounded=False):
    """-> (samples [B,S,T,N], remind outputs or None, in-bound probability mass [B,T,N] or None)."""
    if e.head == "remind":
        bb = dict(b)
        bb.update(e.ret(b["t0"], b["x"], "test"))
        out = T.fwd(e.m, bb)
        if not bounded:
            return (sample_mixture(out, S, gen, b["lower"], b["upper"]) if e.cfg.truncate else sample_mixture(out, S, gen)), out, None
        mu = torch.cat([out["mem_mu"], out["rob_mu"]], 1)
        sg = torch.cat([out["mem_sigma"], out["rob_sigma"]], 1).clamp_min(1e-6)
        w = mixture_weights(out, out["gate_raw"])
        q = (torch.special.ndtr((b["upper"][:, None, None] - mu) / sg) - torch.special.ndtr((b["lower"][:, None, None] - mu) / sg))
        q = (w * q.clamp_min(0)).sum(1) / w.sum(1)
        return sample_mixture(out, S, gen, b["lower"], b["upper"]), out, q
    if bounded and e.head == "gauss":
        o = T.forward_base(e.m, e.head, e.cfg, b, R_GLOBAL)
        mu, sg = o[:, 0], F.softplus(o[:, 1]) + 1e-3
        q = (torch.special.ndtr((b["upper"][:, None] - mu) / sg) - torch.special.ndtr((b["lower"][:, None] - mu) / sg)).clamp_min(0)
        return truncated_gauss(mu, sg, S, gen, b["lower"], b["upper"]), None, q
    smp, pt = T.base_predict(e.m, e.head, e.cfg, b, R_GLOBAL, S, gen)
    if smp is None:
        smp = pt[:, None]
    if bounded:                                         # sample-based experts: in-bound share, then clip
        lo, up = b["lower"][:, None, None], b["upper"][:, None, None]
        q = ((smp >= lo) & (smp <= up)).float().mean(1)
        return torch.minimum(torch.maximum(smp, lo), up), None, q
    return smp, None, None


def main():
    global R_GLOBAL
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="D3")
    p.add_argument("--a", required=True, help="memory expert run")
    p.add_argument("--b", required=True, help="parametric expert run")
    p.add_argument("--w", type=float, default=0.5, help="pool weight of expert a")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--truncate", type=int, default=1)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--test_stride", type=int, default=6)
    p.add_argument("--save_extra", action="store_true")
    p.add_argument("--node_frac", type=float, default=1.0)
    p.add_argument("--wx_noise", type=float, default=0.0)
    p.add_argument("--wx_drop", action="store_true")
    p.add_argument("--subset", default="")
    a = p.parse_args()
    R = Region(a.region, node_subset=np.load(a.subset) if a.subset else None)
    R_GLOBAL = R
    ea, eb = expert(a.a, a.region, R), expert(a.b, a.region, R)
    te = R.origins("test", a.test_stride)
    out_dir = os.path.join(T.RES, a.region, f"{a.name}_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    acc = Accumulator(len(te), probabilistic=True)
    HA, _ = T.historical_tables(R) if a.save_extra else (None, None)
    extra = {k: [] for k in ["gate", "gate_h", "pi", "pnear", "pdrop", "onear", "odrop", "valid", "t0"]}
    gen = torch.Generator(device="cuda").manual_seed(2025)
    g2 = torch.Generator(device="cuda").manual_seed(7)
    k = int(round(a.w * a.samples))
    cap_obs = R.cap_factor.clone()
    R.perturb_weather(a.wx_noise, a.wx_drop, seed=7)
    t_inf = 0.0
    for s in range(0, len(te), 16):
        b = R.batch(te[s:s + 16])
        if a.node_frac < 1:
            keep = torch.rand(b["x"].shape[0], 1, R.N, device="cuda", generator=g2) < a.node_frac
            hist = b["t0"][:, None] + torch.arange(-11, 1, device="cuda") - 2016
            b["x"] = torch.where(keep, b["x"], R.z_in[hist])
        torch.cuda.synchronize(); t1 = time.time()
        if a.truncate and 0 < k < a.samples:
            sa, oa, qa = draw(ea, b, a.samples, gen, bounded=True)
            sb, _, qb = draw(eb, b, a.samples, gen, bounded=True)
            wa, wb = a.w * qa, (1 - a.w) * qb
            pa = torch.where(wa + wb > 1e-12, wa / (wa + wb).clamp_min(1e-12), torch.full_like(wa, a.w))   # [B,T,N]
            uc = torch.rand(sa.shape[0], a.samples, 1, sa.shape[-1], device="cuda", generator=gen)        # one per trajectory
            smp = torch.where(uc < pa[:, None], sa, sb)
        else:
            sa, oa, _ = draw(ea, b, k, gen) if k > 0 else (None, None, None)
            sb, _, _ = draw(eb, b, a.samples - k, gen) if k < a.samples else (None, None, None)
            smp = torch.cat([x for x in (sa, sb) if x is not None], 1)
            if a.truncate:
                smp = torch.minimum(torch.maximum(smp, b["lower"][:, None, None]), b["upper"][:, None, None])
        torch.cuda.synchronize(); t_inf += time.time() - t1
        y = b["y"] * R.std + R.mean
        smn = smp * R.std + R.mean
        ptn = smn.median(1).values
        up = R.cap_max[None] * cap_obs[R.hour_idx[b["t0"]]]
        acc.update(smn, ptn, y, b["ymask"], b["ev_f"], torch.zeros_like(up), up)
        if a.save_extra:
            up_near = R.cap[None] * cap_obs[R.hour_idx[b["t0"]]]
            fut = b["t0"][:, None] + torch.arange(1, 13, device="cuda")
            ha = HA[R.dow_idx[fut], R.tod_idx[fut]]
            day = ((R.tod_idx[fut] >= 72) & (R.tod_idx[fut] < 252))[..., None]
            hs = [2, 5, 11]
            thr_near, thr_drop = 0.9 * up_near[:, None, :], 0.6 * ha
            extra["pnear"].append((smn[:, :, hs] >= thr_near[:, None]).float().mean(1).half().cpu())
            extra["pdrop"].append((smn[:, :, hs] <= thr_drop[:, None, hs]).float().mean(1).half().cpu())
            extra["onear"].append((y[:, hs] >= thr_near).cpu())
            extra["odrop"].append(((y[:, hs] <= thr_drop[:, hs]) & day[:, hs]).cpu())
            extra["valid"].append((b["ymask"][:, hs] & day[:, hs]).cpu())
            extra["t0"].append(b["t0"].cpu())
            if oa is not None:
                gr = oa["gate_raw"]
                extra["gate"].append((gr if gr.dim() == 2 else gr.mean(-1)).half().cpu())
                if gr.dim() == 3:
                    extra["gate_h"].append(gr[..., hs].permute(0, 2, 1).half().cpu())
                if "regime_logits" in oa:
                    extra["pi"].append(torch.sigmoid(oa["regime_logits"]).half().cpu())
    summ = acc.summary()
    summ["inference_s_per_window"] = t_inf / len(te)
    summ["pool"] = dict(a=a.a, b=a.b, w=a.w, truncate=a.truncate)
    summ["train"] = {}
    for e in (ea, eb):
        f = os.path.join(T.RES, a.region, a.a if e is ea else a.b, "done_train")
        if os.path.exists(f):
            summ["train"]["a" if e is ea else "b"] = json.load(open(f))
    summ["params"] = sum(p.numel() for p in ea.m.parameters()) + sum(p.numel() for p in eb.m.parameters())
    sfx = "" if a.node_frac == 1 and a.wx_noise == 0 and not a.wx_drop else f"_nf{a.node_frac}_wn{a.wx_noise}_wd{int(a.wx_drop)}"
    json.dump(summ, open(os.path.join(out_dir, f"summary{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, f"series{sfx}.npz"), t0=te, **acc.series)
    if a.save_extra:
        np.savez_compressed(os.path.join(out_dir, f"extra{sfx}.npz"), **{k: torch.cat(v).numpy() for k, v in extra.items() if v})
    s0 = summ["all"]["avg"]; e0 = summ["anyevent"]["avg"]
    print(f"POOL {a.name}: all CRPS {s0['CRPS']:.3f} MAE {s0['MAE']:.3f} | events CRPS {e0['CRPS']:.3f} | ES {summ['ES']}", flush=True)
    open(os.path.join(out_dir, "train.log"), "a").write(json.dumps(vars(a)) + "\n" + f"POOL all CRPS {s0['CRPS']:.3f}\n")


if __name__ == "__main__":
    main()
