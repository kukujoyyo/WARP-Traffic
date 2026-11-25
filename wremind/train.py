"""Train / evaluate one model on one district.

python -m wremind.train --region D3 --model wremind --seed 0
"""
import os, sys, json, time, math, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
from easydict import EasyDict as edict

from wremind.data import Region, Retriever, REGIMES, EVENT_BITS
from wremind.models import WREMIND, sample_mixture
from wremind.baselines import GWNet, STID, STAEformer, DeepAR, pinball, QUANTILES
from wremind.metrics import Accumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
RES = os.path.join(ROOT, "results")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REMIND_FAMILY = {
    # name: (joint retrieval, wx_feat, regime_gate, regime_sup, lambda_fd, truncate, event_aug)
    "remind": (False, False, False, False, 0.0, False, False),
    "remind_wx": (False, True, False, False, 0.0, False, False),            # REMIND + weather concat
    "wremind": (True, True, True, True, 0.5, True, True),
    "wremind_noM1": (False, True, True, True, 0.5, True, True),
    "wremind_noM1_nowx": (False, False, True, True, 0.5, True, True),
    "wremind_noM2": (True, True, False, False, 0.5, True, True),
    "wremind_noPi": (True, True, True, False, 0.5, True, True),
    "wremind_noM3": (True, True, True, True, 0.0, False, True),
    "wremind_M3pen": (True, True, True, True, 0.5, False, True),             # penalty only
    "wremind_M3trunc": (True, True, True, True, 0.0, True, True),            # truncation only
    "wremind_noM4": (True, True, True, True, 0.5, True, False),
    "wremind_jointonly": (True, False, False, False, 0.0, False, False),     # REMIND + M1 only
    "remind_trunc": (False, False, False, False, 0.0, True, False),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="D3")
    p.add_argument("--model", default="wremind")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--n_epoch", type=int, default=16000, help="training windows per epoch")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--M", type=int, default=30)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau_w", type=float, default=1.0)
    p.add_argument("--lambda_fd", type=float, default=None)
    p.add_argument("--lambda_gate", type=float, default=0.2)
    p.add_argument("--event_frac", type=float, default=0.3)
    p.add_argument("--event_perturb", type=float, default=1.0)
    p.add_argument("--test_stride", type=int, default=6)
    p.add_argument("--val_stride", type=int, default=12)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--bank_months", type=int, default=0, help="restrict memory to the last k training months (0=all)")
    p.add_argument("--node_frac", type=float, default=1.0, help="random sensor dropout at test (robustness)")
    p.add_argument("--wx_noise", type=float, default=0.0, help="std of noise added to weather inputs at test")
    p.add_argument("--wx_drop", action="store_true", help="weather unavailable at test (inputs set to climatology)")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--save_extra", action="store_true")
    p.add_argument("--bb_init", default="auto", help="trained gwnet_gauss(+wx) checkpoint for the backbone ('' = random init)")
    p.add_argument("--bb_freeze", type=int, default=1)
    p.add_argument("--loss", default="auto", help="auto (WARP-Traffic: crps, REMIND: nll) | nll | crps")
    p.add_argument("--nll_weight", type=float, default=0.05)
    p.add_argument("--hgate", type=int, default=1, help="horizon-specific trust gate (WARP-Traffic)")
    p.add_argument("--final", type=int, default=0, help="marker: run with the final code version (no effect)")
    p.add_argument("--reeval", type=int, default=0, help="marker: re-evaluate an existing run (no effect)")
    p.add_argument("--tod_window", type=int, default=6, help="M1 calendar-consistent retrieval window (steps); joint mode only")
    p.add_argument("--backbone", default="gwnet", help="deviation-aware branch backbone for the REMIND family: gwnet | none (original encoder)")
    p.add_argument("--subset", default="", help="file with node indices to use")
    return p.parse_args()


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- statistics shared by several models
def historical_tables(R):
    """HA [7,288,N] and climatology quantiles [2,288,N,Q] in original units from the training period."""
    tr = R.origins("train", 1)
    z = R.z[tr].float() * R.std + R.mean
    v = R.valid[tr]
    dow, tod = R.dow_idx[tr], R.tod_idx[tr]
    HA = torch.zeros(7, 288, R.N, device=z.device)
    for d in range(7):
        for t0 in range(0, 288, 24):
            sel = (dow == d) & (tod >= t0) & (tod < t0 + 24)
            zz, vv, tt = z[sel], v[sel], tod[sel]
            for t in range(t0, t0 + 24):
                m = tt == t
                HA[d, t] = (zz[m] * vv[m]).sum(0) / vv[m].sum(0).clamp_min(1)
    Q = torch.linspace(0.01, 0.99, 50, device=z.device)
    CL = torch.zeros(2, 288, R.N, len(Q), device=z.device)
    we = (dow >= 5)
    for w in range(2):
        for t in range(288):
            sel = (we == bool(w)) & (tod == t)
            zz = z[sel].masked_fill(~v[sel], float("nan"))
            CL[w, t] = torch.nanquantile(zz, Q, dim=0).T
    return HA, torch.nan_to_num(CL)


# ---------------------------------------------------------------- model construction
def build(args, R):
    name = args.model
    base = name.split("+")[0]
    wx = "+wx" in name
    N = R.N
    adj = torch.as_tensor(R.adj, device="cuda")
    if base in REMIND_FAMILY:
        j, wf, rg, rs, lfd, tr, ea = REMIND_FAMILY[base]
        cfg = edict(hidden_dim=args.hidden, dropout=0.1, pred_len=12, memory_mixtures=4, robust_mixtures=4,
                    memory_sigma_floor=1e-3, robust_sigma_floor=0.1 if args.backbone == "none" else 1e-3, deviation_residual_scale=0.3,
                    similarity_alignment_weight=0.5, dispersion_alignment_weight=0.5, mae_weight=30.0,
                    memory_dropout=0.3, wx_feat=wf, regime_gate=rg, regime_sup=rs,
                    lambda_fd=lfd if args.lambda_fd is None else (args.lambda_fd if lfd > 0 else 0.0),
                    lambda_gate=args.lambda_gate, event_aug=ea, event_perturb=args.event_perturb,
                    regime_pos_weight=[5.0, 10.0, 10.0, 20.0], joint=j, truncate=tr, backbone=args.backbone,
                    loss_type=("crps" if rg else "nll") if args.loss == "auto" else args.loss, crps_weight=30.0, nll_weight=args.nll_weight,
                    horizon_gate=bool(args.hgate) and rg)
        m = WREMIND(cfg, R.adj, R.F, 5, len(REGIMES)).cuda()
        return m, "remind", cfg
    in_dim = 1 + 2 + (R.F if wx else 0)
    head = "point"
    for h in ["gauss", "quantile", "mc"]:
        if base.endswith("_" + h):
            head = h
    arch = base.split("_")[0]
    oh = "gauss" if head == "mc" else head
    if arch == "gwnet":
        a = adj / adj.sum(1, keepdim=True).clamp_min(1e-6)
        m = GWNet(N, [a, a.T], in_dim, head=oh)
    elif arch == "stid":
        m = STID(N, in_dim, head=oh)
    elif arch == "staeformer":
        m = STAEformer(N, in_dim, head=oh)
    elif arch == "deepar":
        m = DeepAR(N, 2 + (R.F if wx else 0))
        head = "deepar"
    else:
        raise ValueError(name)
    return m.cuda(), head, edict(arch=arch, wx=wx, head=head)


def base_inputs(b, R, wx):
    B, T, N = b["x"].shape
    t0 = b["t0"]
    hist = t0[:, None] + torch.arange(-T + 1, 1, device=t0.device)
    tod = (R.tod_idx[hist].float() / 288)[..., None, None].expand(-1, -1, N, 1)
    dow = (R.dow_idx[hist].float() / 6)[..., None, None].expand(-1, -1, N, 1)
    parts = [b["x"].unsqueeze(-1), tod, dow] + ([b["wx"]] if wx else [])
    return torch.cat(parts, -1), hist


def forward_base(m, head, cfg, b, R, mc=False):
    inp, hist = base_inputs(b, R, cfg.wx)
    if cfg.arch == "gwnet":
        o = m(inp, mc=mc)
    elif cfg.arch == "stid":
        o = m(inp, R.tod_idx[b["t0"]], R.dow_idx[b["t0"]], mc=mc)
    else:
        o = m(inp, R.tod_idx[hist], R.dow_idx[hist], mc=mc)
    return o


def deepar_cov(b, R, wx):
    inp, hist = base_inputs(b, R, wx)
    cov_h = inp[..., 1:]
    t0 = b["t0"]
    fut = t0[:, None] + torch.arange(1, 13, device=t0.device)
    N = R.N
    tod = (R.tod_idx[fut].float() / 288)[..., None, None].expand(-1, -1, N, 1)
    dow = (R.dow_idx[fut].float() / 6)[..., None, None].expand(-1, -1, N, 1)
    cf = [tod, dow] + ([b["wx"][:, -1:].expand(-1, 12, -1, -1)] if wx else [])   # weather held at origin (causal)
    return cov_h, torch.cat(cf, -1)


def base_loss(m, head, cfg, b, R):
    y, mk = b["y"], b["ymask"].float()
    if head == "deepar":
        ch, cf = deepar_cov(b, R, cfg.wx)
        mu, sg = m(b["x"], y, ch, cf)
        return ((0.5 * ((y - mu) / sg) ** 2 + sg.log()) * mk).sum() / mk.sum()
    o = forward_base(m, head, cfg, b, R)
    if head == "point":
        return ((o[:, 0] - y).abs() * mk).sum() / mk.sum()
    if head in ("gauss", "mc"):
        mu, sg = o[:, 0], F.softplus(o[:, 1]) + 1e-3
        return ((0.5 * ((y - mu) / sg) ** 2 + sg.log()) * mk).sum() / mk.sum()
    if head == "quantile":
        q, _ = torch.sort(o, 1)
        return pinball(q, y, mk)


@torch.no_grad()
def base_predict(m, head, cfg, b, R, S, gen):
    """-> (samples [B,S,T,N] or None, point [B,T,N]) in standardised units."""
    if head == "deepar":
        ch, cf = deepar_cov(b, R, cfg.wx)
        s = m.sample(b["x"], ch, cf, S, gen)
        return s, s.median(1).values
    if head == "mc":
        m.eval()
        outs = []
        P = 20
        for _ in range(P):
            o = forward_base(m, head, cfg, b, R, mc=True)
            mu, sg = o[:, 0], F.softplus(o[:, 1]) + 1e-3
            eps = torch.randn((mu.size(0), S // P) + mu.shape[1:], device=mu.device, generator=gen)
            outs.append(mu[:, None] + sg[:, None] * eps)
        s = torch.cat(outs, 1)
        return s, s.median(1).values
    o = forward_base(m, head, cfg, b, R)
    if head == "point":
        return None, o[:, 0]
    if head == "gauss":
        mu, sg = o[:, 0], F.softplus(o[:, 1]) + 1e-3
        eps = torch.randn((mu.size(0), S) + mu.shape[1:], device=mu.device, generator=gen)
        return mu[:, None] + sg[:, None] * eps, mu
    if head == "quantile":
        q, _ = torch.sort(o, 1)                                   # [B,Q,T,N]
        lv = QUANTILES.to(q.device)
        u = (torch.arange(S, device=q.device) + 0.5) / S
        pos = ((u - lv[0]) / (lv[1] - lv[0]))                      # fractional index, may extrapolate
        lo = pos.floor().clamp(0, len(lv) - 2).long()
        fr = (pos - lo.float())
        a, c = q[:, lo], q[:, lo + 1]
        s = a + fr.view(1, -1, 1, 1) * (c - a)
        return s, q[:, len(lv) // 2]


# ---------------------------------------------------------------- loops
def fwd(model, b):
    """bf16 autocast for the REMIND-family encoders; likelihood terms are computed in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b)
    return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}


def make_batch(R, ret, t0, split, args, test_aug=None):
    b = R.batch(t0)
    if test_aug is not None:
        test_aug(b)
    if ret is not None:
        b.update(ret(b["t0"], b["x"], split))
    return b


def event_origin_flags(R, origins):
    """Region-level weather event (heat/rain/smoke on >=10% of sensors) or any incident within the target window."""
    ev = R.events
    flags = np.zeros(len(origins), bool)
    for i in range(0, len(origins), 4096):
        o = torch.as_tensor(origins[i:i + 4096], device=ev.device)
        e = ev[o[:, None] + torch.arange(1, 13, device=ev.device)]       # [B,T,N]
        w = (((e >> 0) & 1) | ((e >> 1) & 1) | ((e >> 2) & 1)).float().mean(-1).amax(1)
        inc = ((e >> 3) & 1).float().amax(1).sum(-1)
        flags[i:i + 4096] = ((w >= 0.1) | (inc >= max(1, 0.01 * R.N))).cpu().numpy()
    return flags


def run(args):
    set_seed(args.seed)
    subset = np.load(args.subset) if args.subset else None
    R = Region(args.region, node_subset=subset)
    model, head, cfg = build(args, R)
    is_remind = head == "remind"
    if is_remind and args.backbone == "gwnet" and args.bb_init:
        src = args.bb_init
        if src == "auto":
            src = os.path.join(RES, args.region, ("gwnet_gauss+wx" if cfg.wx_feat else "gwnet_gauss") + f"_s{args.seed}", "model.pt")
        model.init_backbone(torch.load(src), freeze=bool(args.bb_freeze))
    lr = args.lr or (5e-4 if is_remind else 1e-3)
    run_name = args.model + (f"_{args.tag}" if args.tag else "") + f"_s{args.seed}"
    out_dir = os.path.join(RES, args.region, run_name)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "train.log"), "a")
    P = lambda *a: (print(*a, flush=True), log.write(" ".join(map(str, a)) + "\n"), log.flush())
    P(json.dumps(vars(args)), f"N={R.N} lr={lr}")

    ret = None
    if is_remind:
        bank_start = None
        if args.bank_months > 0:
            bank_start = str(pd.Timestamp("2020-10-31") - pd.DateOffset(months=args.bank_months))
        ret = Retriever(R, M=args.M, K=args.K, joint=cfg.joint, alpha=args.alpha, tau_w=args.tau_w,
                        bank_start=bank_start, seed=args.seed, tod_window=args.tod_window if cfg.joint else 0)
    tr = R.origins("train", 1)
    va = R.origins("val", args.val_stride)
    te = R.origins("test", args.test_stride)
    probs = np.ones(len(tr)) / len(tr)
    use_aug = is_remind and cfg.event_aug
    if use_aug:
        fl = event_origin_flags(R, tr)
        pe = fl.mean()
        probs = np.where(fl, args.event_frac / max(pe, 1e-6), (1 - args.event_frac) / max(1 - pe, 1e-6))
        probs = probs / probs.sum()
        P(f"event windows {fl.mean():.3f} of train -> sampled at {args.event_frac}")
    iw_all = (1.0 / len(tr)) / probs
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4 if not is_remind else 0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, bad, ck = float("inf"), 0, os.path.join(out_dir, "model.pt")
    rng = np.random.default_rng(args.seed)
    t_train = time.time()
    epoch_times = []
    if not os.path.exists(os.path.join(out_dir, "done_train")):
        for ep in range(args.epochs):
            model.train()
            te0 = time.time()
            idx = rng.choice(len(tr), size=args.n_epoch, replace=False, p=probs)
            tot, n = 0.0, 0
            for s in range(0, len(idx), args.bs):
                bi = idx[s:s + args.bs]
                b = make_batch(R, ret, tr[bi], "train", args)
                if is_remind:
                    w = torch.as_tensor(iw_all[bi], device="cuda", dtype=torch.float32)
                    out = fwd(model, b)
                    loss, logs = model.loss(b, out, weight=w / w.mean() if use_aug else None)
                else:
                    loss = base_loss(model, head, cfg, b, R)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item(); n += 1
                if args.max_steps and n >= args.max_steps:
                    break
            sched.step()
            # validation loss
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for s in range(0, len(va), 32):
                    b = make_batch(R, ret, va[s:s + 32], "val", args)
                    if is_remind:
                        out = fwd(model, b)
                        l, _ = model.loss(b, out)
                    else:
                        l = base_loss(model, head, cfg, b, R)
                    vt += l.item(); vn += 1
            vl = vt / vn
            epoch_times.append(time.time() - te0)
            P(f"ep {ep} train {tot / n:.4f} val {vl:.4f} time {epoch_times[-1]:.0f}s")
            if vl < best - 1e-4:
                best, bad = vl, 0
                torch.save(model.state_dict(), ck)
            else:
                bad += 1
                if bad >= args.patience:
                    break
        open(os.path.join(out_dir, "done_train"), "w").write(json.dumps(dict(best_val=best, epochs=len(epoch_times),
                                                                             train_time=time.time() - t_train,
                                                                             epoch_time=float(np.mean(epoch_times)))))
    model.load_state_dict(torch.load(ck))
    model.eval()
    evaluate(args, R, model, head, cfg, ret, te, out_dir, P)


def conformal_tables(args, R, model, head, cfg):
    va = R.origins("val", 3)
    res = []
    with torch.no_grad():
        for s in range(0, len(va), 32):
            b = R.batch(va[s:s + 32])
            _, pt = base_predict(model, head, cfg, b, R, 2, None)
            r = (b["y"] - pt).masked_fill(~b["ymask"], float("nan"))
            res.append(r)
    r = torch.cat(res)                                        # [V,T,N]
    lv = (torch.arange(99, device=r.device) + 1) / 100
    return torch.nan_to_num(torch.nanquantile(r, lv, dim=0))  # [99,T,N]


@torch.no_grad()
def evaluate(args, R, model, head, cfg, ret, te, out_dir, P, tag=""):
    gen = torch.Generator(device="cuda").manual_seed(2025)
    is_remind = head == "remind"
    acc = Accumulator(len(te), probabilistic=True)
    acc_conf = Accumulator(len(te)) if head == "point" else None
    conf_q = conformal_tables(args, R, model, head, cfg) if head == "point" else None
    HA, _ = historical_tables(R) if args.save_extra else (None, None)
    extra = {k: [] for k in ["gate", "gate_h", "pi", "pnear", "pdrop", "onear", "odrop", "valid", "ws", "t0"]}
    cap_obs = R.cap_factor.clone()                                       # observed envelope, used for the violation metric
    R.perturb_weather(args.wx_noise, args.wx_drop, seed=7)               # weather robustness: reaches every weather route
    test_aug = None
    if args.node_frac < 1:
        g2 = torch.Generator(device="cuda").manual_seed(7)
        def test_aug(b):             # detector outages: inputs of dropped sensors replaced by last-week values
            keep = torch.rand(b["x"].shape[0], 1, R.N, device="cuda", generator=g2) < args.node_frac
            hist = b["t0"][:, None] + torch.arange(-11, 1, device="cuda") - 2016
            b["x"] = torch.where(keep, b["x"], R.z_in[hist])
    t_inf = 0.0
    for s in range(0, len(te), 16):
        b = R.batch(te[s:s + 16])
        if test_aug is not None:
            test_aug(b)
        torch.cuda.synchronize(); t1 = time.time()
        if ret is not None:
            b.update(ret(b["t0"], b["x"], "test"))
        if is_remind:
            out = fwd(model, b)
            if cfg.truncate:
                smp = sample_mixture(out, args.samples, gen, b["lower"], b["upper"])
            else:
                smp = sample_mixture(out, args.samples, gen)
            pt = smp.median(1).values
        else:
            smp, pt = base_predict(model, head, cfg, b, R, args.samples, gen)
        torch.cuda.synchronize(); t_inf += time.time() - t1
        y = b["y"] * R.std + R.mean
        ptn = pt * R.std + R.mean
        smn = smp * R.std + R.mean if smp is not None else None
        lo = torch.zeros_like(b["upper"])
        up = R.cap_max[None] * cap_obs[R.hour_idx[b["t0"]]]               # admissible bound (M3, violation metric)
        up_near = R.cap[None] * cap_obs[R.hour_idx[b["t0"]]]         # near-capacity alert reference
        acc.update(smn, ptn, y, b["ymask"], b["ev_f"], lo, up)
        if acc_conf is not None:
            smc = (pt[:, None] + conf_q[None]) * R.std + R.mean
            acc_conf.update(smc, ptn, y, b["ymask"], b["ev_f"], lo, up)
        if args.save_extra:
            fut = b["t0"][:, None] + torch.arange(1, 13, device="cuda")
            ha = HA[R.dow_idx[fut], R.tod_idx[fut]]                       # [B,T,N]
            day = ((R.tod_idx[fut] >= 72) & (R.tod_idx[fut] < 252))[..., None]
            thr_near = 0.9 * up_near[:, None, :]
            thr_drop = 0.6 * ha
            hs = [2, 5, 11]
            if smn is not None:
                pn = (smn[:, :, hs] >= thr_near[:, None, :]).float().mean(1)
                pdp = (smn[:, :, hs] <= thr_drop[:, None, hs]).float().mean(1)
            else:
                pn = (ptn[:, hs] >= thr_near).float()
                pdp = (ptn[:, hs] <= thr_drop[:, hs]).float()
            extra["pnear"].append(pn.half().cpu()); extra["pdrop"].append(pdp.half().cpu())
            extra["onear"].append((y[:, hs] >= thr_near).cpu()); extra["odrop"].append(((y[:, hs] <= thr_drop[:, hs]) & day[:, hs]).cpu())
            extra["valid"].append((b["ymask"][:, hs] & day[:, hs]).cpu())
            extra["t0"].append(b["t0"].cpu())
            if is_remind:
                gr = out["gate_raw"]
                extra["gate"].append((gr if gr.dim() == 2 else gr.mean(-1)).half().cpu())
                if gr.dim() == 3:
                    extra["gate_h"].append(gr[..., hs].permute(0, 2, 1).half().cpu())
                extra["ws"].append(b["ret_ws"].mean(-1).half().cpu())
                if "regime_logits" in out:
                    extra["pi"].append(torch.sigmoid(out["regime_logits"]).half().cpu())
    summ = acc.summary()
    summ["inference_s_per_window"] = t_inf / len(te)
    tr_info = json.load(open(os.path.join(out_dir, "done_train")))
    summ["train"] = tr_info
    summ["params"] = sum(p.numel() for p in model.parameters())
    sfx = tag or ("" if args.node_frac == 1 and args.wx_noise == 0 and not args.wx_drop else
                  f"_nf{args.node_frac}_wn{args.wx_noise}_wd{int(args.wx_drop)}")
    json.dump(summ, open(os.path.join(out_dir, f"summary{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, f"series{sfx}.npz"), t0=te, **{k: v for k, v in acc.series.items()})
    if acc_conf is not None:
        sc = acc_conf.summary()
        json.dump(sc, open(os.path.join(out_dir, f"summary_conformal{sfx}.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(out_dir, f"series_conformal{sfx}.npz"), t0=te, **acc_conf.series)
    if args.save_extra:
        np.savez_compressed(os.path.join(out_dir, f"extra{sfx}.npz"), **{k: torch.cat(v).numpy() for k, v in extra.items() if v})
    a = summ["all"]["avg"]; e = summ["anyevent"]["avg"]
    P(f"TEST {args.model}: all CRPS {a['CRPS']:.3f} MAE {a['MAE']:.3f} | events CRPS {e['CRPS']:.3f} MAE {e['MAE']:.3f} | ES {summ['ES']}")


if __name__ == "__main__":
    run(get_args())
