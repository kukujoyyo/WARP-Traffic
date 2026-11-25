"""REMIND and WARP-Traffic.

REMIND components (encoder, memory branch, robust branch, trust gate, gated mixture)
are imported unchanged from the official repository (remind_core, commit d0e7342).
WARP-Traffic adds:
  M1 joint weather-calendar retrieval (data.Retriever(joint=True));
  M2 regime-aware trust gate (RegimeGate) with weak supervision of regime posteriors;
  M3 capacity-consistent mixture (tail-mass penalty + truncated predictive sampling);
  M4 rare-event training (event-stratified sampling, event-scaled memory perturbation).
"""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "remind_core"))
from algorithm.encoder import SpatioTemporalEncoder  # noqa: E402
from algorithm.memory_branch import MemoryBranch     # noqa: E402
from algorithm.robust_branch import RobustBranch     # noqa: E402
from algorithm.trust_gate import TrustGate           # noqa: E402
from wremind.baselines import GWNet                  # noqa: E402

LOG_2PI = math.log(2 * math.pi)


def normalize_adjacency(adj):
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    d = adj.sum(-1).clamp_min(1.0).pow(-0.5)
    return d[:, None] * adj * d[None, :]


def masked_mdn_log_prob(y, mask, logits, mu, sigma):
    """Trajectory-level mixture log-likelihood per node (REMIND convention), with missing targets masked.
    y, mask [B,T,N]; logits [B,N,M]; mu/sigma [B,M,T,N] -> [B,N]."""
    sigma = sigma.clamp_min(1e-6)
    ld = -0.5 * (((y.unsqueeze(1) - mu) / sigma) ** 2 + 2 * torch.log(sigma) + LOG_2PI)
    ld = (ld * mask.unsqueeze(1)).sum(2).permute(0, 2, 1)       # [B, N, M]
    return torch.logsumexp(torch.log_softmax(logits, -1) + ld, -1)


def mixture_tail_mass(outputs, lower, upper):
    """Probability mass the gated mixture puts outside [lower, upper] (per B,T,N)."""
    wt = mixture_weights(outputs, outputs["gate_raw"])               # [B,M,T|1,N]
    mu = torch.cat([outputs["mem_mu"], outputs["rob_mu"]], 1)      # [B,M,T,N]
    sg = torch.cat([outputs["mem_sigma"], outputs["rob_sigma"]], 1).clamp_min(1e-6)
    lo = lower[:, None, None, :]
    up = upper[:, None, None, :]
    p_lo = torch.special.ndtr((lo - mu) / sg)
    p_up = 1 - torch.special.ndtr((up - mu) / sg)
    return (wt * p_lo).sum(1), (wt * p_up).sum(1)


def mixture_weights(outputs, gate):
    """Component weights [B,M,T,N]; gate is [B,N] (REMIND, one weight per trajectory) or [B,N,T] (horizon-specific)."""
    g = gate[:, None, :] if gate.dim() == 2 else gate.permute(0, 2, 1)             # [B,1|T,N]
    pm = outputs["mem_logits"].softmax(-1).permute(0, 2, 1)[:, :, None, :]           # [B,Mm,1,N]
    pr = outputs["rob_logits"].softmax(-1).permute(0, 2, 1)[:, :, None, :]
    return torch.cat([g[:, None] * pm, (1 - g[:, None]) * pr], 1)


def _abs_moment(m, s):
    """E|Z| for Z ~ N(m, s^2)."""
    z = m / s
    return m * (2 * torch.special.ndtr(z) - 1) + 2 * s * torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def mixture_crps(y, gate, outputs):
    """Closed-form CRPS of the gated Gaussian mixture, per (B,T,N) marginal (Grimit et al., 2006)."""
    w = mixture_weights(outputs, gate)                                               # [B,M,T|1,N]
    mu = torch.cat([outputs["mem_mu"], outputs["rob_mu"]], 1)                         # [B,M,T,N]
    sg = torch.cat([outputs["mem_sigma"], outputs["rob_sigma"]], 1).clamp_min(1e-4)
    t1 = (w * _abs_moment(y.unsqueeze(1) - mu, sg)).sum(1)
    dm = mu.unsqueeze(1) - mu.unsqueeze(2)                                           # [B,M,M,T,N]
    ds = torch.sqrt(sg.unsqueeze(1) ** 2 + sg.unsqueeze(2) ** 2)
    t2 = (w.unsqueeze(1) * w.unsqueeze(2) * _abs_moment(dm, ds)).sum((1, 2))
    return t1 - 0.5 * t2                                                             # [B,T,N]


class WeatherContext(nn.Module):
    """Summarises the input-window weather/calendar into a node context (used by M1-concat and M2)."""

    def __init__(self, n_wx, n_cal, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * n_wx + n_cal, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, wx, cal):
        # wx [B,T,N,F]; cal [B,C]
        last, mean = wx[:, -1], wx.mean(1)
        c = cal[:, None, :].expand(-1, wx.size(2), -1)
        return self.net(torch.cat([last, mean, c], -1))              # [B,N,H]


class RegimeGate(nn.Module):
    """M2: trust gate conditioned on retrieval quality, weather and traffic deviation signals.
    Outputs the fusion weight g and multi-label regime posteriors pi (heat, rain, smoke, incident)."""

    def __init__(self, hidden, n_wx, n_regimes, use_regime=True, n_horizons=1):
        super().__init__()
        self.n_horizons = n_horizons
        n_in = 4 + 2 + 3 + n_regimes   # REMIND stats + weather-sim stats + deviation stats + analog event rates
        self.use_regime = use_regime
        self.feat = nn.Sequential(nn.Linear(n_in + hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.regime_head = nn.Linear(hidden, n_regimes)
        self.gate_head = nn.Linear(hidden + (n_regimes if use_regime else 0), n_horizons)

    def forward(self, ret_s, ret_ws, c_mem, d_node, dev, ret_regime, wctx):
        s1, s_bar = ret_s[..., 0], ret_s.mean(-1)
        stats = torch.stack([s_bar, s1, c_mem, d_node.norm(dim=-1), ret_ws.mean(-1), ret_ws[..., 0]], -1)
        h = self.feat(torch.cat([stats, dev, ret_regime, wctx], -1))
        logits = self.regime_head(h)
        gin = torch.cat([h, torch.sigmoid(logits)], -1) if self.use_regime else h
        gate = torch.sigmoid(self.gate_head(gin))
        gate = gate.squeeze(-1) if self.n_horizons == 1 else gate              # [B,N] or [B,N,T]
        return dict(gate=gate, s_bar=s_bar, s1=s1, d_norm=d_node.norm(dim=-1), regime_logits=logits)


class WREMIND(nn.Module):
    """flags: wx_feat (weather context injected into both branches), regime_gate (M2),
    capacity (M3 penalty weight > 0), event_aug (M4). All off == REMIND."""

    def __init__(self, cfg, adj, n_wx, n_cal, n_regimes=4):
        super().__init__()
        self.cfg = cfg
        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer("adj_norm", normalize_adjacency(adj))
        D, N = cfg.hidden_dim, adj.size(0)
        self.memory_branch = MemoryBranch(SpatioTemporalEncoder(1, D, cfg.dropout, num_nodes=N), D, cfg.pred_len,
                                          cfg.memory_mixtures, cfg.memory_sigma_floor)
        self.robust_branch = RobustBranch(SpatioTemporalEncoder(1, D, cfg.dropout, num_nodes=N), D, cfg.pred_len,
                                          cfg.robust_mixtures, cfg.robust_sigma_floor, cfg.deviation_residual_scale)
        self.use_bb = cfg.get("backbone", "none") == "gwnet"
        if self.use_bb:
            # deviation-aware branch on a Graph WaveNet backbone with calendar (and, if wx_feat, weather) inputs
            a = adj / adj.sum(1, keepdim=True).clamp_min(1e-6)
            self.register_buffer("sup_f", a)
            self.register_buffer("sup_b", a.T.contiguous())
            self.bb = GWNet(N, [a, a.T], 1 + 2 + (n_wx if cfg.wx_feat else 0), head="gauss")
        need_ctx = cfg.wx_feat or cfg.regime_gate
        self.wctx = WeatherContext(n_wx, n_cal, D) if need_ctx else None
        if cfg.wx_feat:
            # weather context enters the memory-calibration and deviation pathways additively
            self.mem_film = nn.Linear(D, 2 * D + 1)
            self.rob_film = nn.Linear(D, 2 * D)
        if cfg.regime_gate:
            self.gate = RegimeGate(D, n_wx, n_regimes, use_regime=cfg.regime_sup,
                                   n_horizons=cfg.pred_len if cfg.get("horizon_gate", False) else 1)
        else:
            self.gate = TrustGate(D)

    def _mem(self, x, ret_x, ret_y, ret_s, ctx):
        out = self.memory_branch(x, ret_x, ret_y, ret_s, self.adj_norm)
        if ctx is None:
            return out
        mb = self.memory_branch
        mc = out["mem_context"] + self.mem_film(ctx)
        B, N = mc.shape[:2]
        sigma_node = F.softplus(mb.sigma_head(mc)) + mb.sigma_floor
        logits = mb.mix_logits_head(mc)
        dmu = mb.mix_delta_head(mc).reshape(B, N, mb.n_mixtures, mb.pred_len)
        msg = mb.mix_sigma_head(mc).reshape(B, N, mb.n_mixtures, mb.pred_len)
        out["mem_logits"] = logits
        out["mem_mu"] = (out["mu_mem"].permute(0, 2, 1).unsqueeze(2) + dmu).permute(0, 2, 3, 1)
        out["mem_sigma"] = (F.softplus(msg) + sigma_node.unsqueeze(2) + mb.sigma_floor).permute(0, 2, 3, 1)
        return out

    def init_backbone(self, state_dict, freeze=True):
        """Start the deviation-aware branch from a trained probabilistic GWNet: load it, zero the additive heads so the
        branch initially reproduces the backbone, and (optionally) freeze it."""
        self.bb.load_state_dict(state_dict)
        rb = self.robust_branch
        with torch.no_grad():
            for lin in [rb.base_head, rb.dev_head, rb.mix_delta_head]:
                lin.weight.zero_(); lin.bias.zero_()
            for lin in [rb.inflation_head, rb.mix_sigma_head]:
                lin.weight.mul_(0.1); lin.bias.fill_(-4.0)
            if self.cfg.wx_feat:
                self.rob_film.weight.zero_(); self.rob_film.bias.zero_()
        self.bb_frozen = freeze
        if freeze:
            for p in self.bb.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "bb_frozen", False):
            self.bb.eval()
        return self

    def _backbone(self, b):
        x = b["x"]
        B, T, N = x.shape
        parts = [x.unsqueeze(-1), b["tcal"][:, :, None, :].expand(-1, -1, N, -1)]
        if self.cfg.wx_feat:
            parts.append(b["wx"])
        self.bb.supports = [self.sup_f, self.sup_b]
        o = self.bb(torch.cat(parts, -1))                           # [B,2,T,N]
        return o[:, 0], o[:, 1]

    def _rob(self, x, ret_x, ret_s, ctx, b=None):
        out = self.robust_branch(x, ret_x, ret_s, self.adj_norm)
        if ctx is None and not self.use_bb:
            return out
        rb = self.robust_branch
        c = out["rob_context"] + (self.rob_film(ctx) if ctx is not None else 0)
        B, N = c.shape[:2]
        base_c, d_node = c[..., :c.size(-1) // 2], c[..., c.size(-1) // 2:]
        mu_rob = (rb.base_head(base_c) + rb.residual_scale * rb.dev_head(d_node)).permute(0, 2, 1)
        sigma_rob = (F.softplus(rb.base_sigma_head(base_c)) + F.softplus(rb.inflation_head(d_node)) + rb.sigma_floor).permute(0, 2, 1)
        if self.use_bb:
            mu_bb, s_bb = self._backbone(b)                         # [B,T,N]
            mu_rob = mu_bb + mu_rob
            sigma_rob = (F.softplus(s_bb) + F.softplus(rb.inflation_head(d_node)).permute(0, 2, 1) + rb.sigma_floor)
        dmu = rb.mix_delta_head(c).reshape(B, N, rb.n_mixtures, rb.pred_len)
        msg = rb.mix_sigma_head(c).reshape(B, N, rb.n_mixtures, rb.pred_len)
        out["mu_rob"], out["sigma_rob"] = mu_rob, sigma_rob
        out["rob_logits"] = rb.mix_logits_head(c)
        out["rob_mu"] = (mu_rob.permute(0, 2, 1).unsqueeze(2) + dmu).permute(0, 2, 3, 1)
        out["rob_sigma"] = (F.softplus(msg) + sigma_rob.permute(0, 2, 1).unsqueeze(2) + rb.sigma_floor).permute(0, 2, 3, 1)
        out["d_node"] = d_node
        return out

    def forward(self, b):
        cfg = self.cfg
        x = b["x"].unsqueeze(-1)
        ret_x, ret_y, ret_s = b["ret_x"].unsqueeze(-1), b["ret_y"].unsqueeze(-1), b["ret_s"]
        rho = torch.ones_like(ret_s[..., 0])
        if self.training and cfg.memory_dropout > 0:
            strength = cfg.memory_dropout * torch.ones_like(rho)
            if cfg.event_aug:  # M4: stronger memory perturbation where the window is an event
                sev = b["regime"].amax(-1)
                strength = (strength * (1 + cfg.event_perturb * sev)).clamp(max=0.9)
            rho = 1 - strength * torch.rand_like(rho)
        ctx = self.wctx(b["wx"], b["cal"]) if self.wctx is not None else None
        fctx = ctx if cfg.wx_feat else None
        mem = self._mem(x, ret_x, ret_y, ret_s, fctx)
        rob = self._rob(x, ret_x, ret_s, fctx, b)
        if cfg.regime_gate:
            proto = rob["proto_x"].squeeze(-1)                      # [B,T,N]
            e = (b["x"] - proto).abs()
            e_now, e_mean = e[:, -3:].mean(1), e.mean(1)            # recent and window deviation
            e_nb = e_now @ self.adj_norm.T                          # neighbourhood deviation
            dev = torch.stack([e_now, e_mean, e_now - e_nb], -1)
            g = self.gate(ret_s, b["ret_ws"], mem["c_mem"], rob["d_node"], dev, b["ret_regime"], ctx)
        else:
            g = self.gate(ret_s, mem["c_mem"], mem["c_mem_ref"], rob["d_node"])
        gate = g["gate"]
        r_ = rho if gate.dim() == 2 else rho.unsqueeze(-1)
        out = {**mem, **rob, **g, "gate_raw": gate, "gate_mix": gate * r_ if self.training else gate, "rho": rho}
        gt = gate.unsqueeze(1) if gate.dim() == 2 else gate.permute(0, 2, 1)            # [B,1|T,N]
        out["pred_mean"] = gt * mem["mu_mem"] + (1 - gt) * rob["mu_rob"]
        return out

    def loss(self, b, out, weight=None):
        cfg = self.cfg
        y, m = b["y"], b["ymask"].float()
        gm = out["gate_mix"].clamp(1e-6, 1 - 1e-6)
        cnt = m.sum(1).clamp_min(1)
        if gm.dim() == 2:   # REMIND: trajectory-level mixture likelihood
            lp_mem = masked_mdn_log_prob(y, m, out["mem_logits"], out["mem_mu"], out["mem_sigma"])
            lp_rob = masked_mdn_log_prob(y, m, out["rob_logits"], out["rob_mu"], out["rob_sigma"])
            lp = torch.logsumexp(torch.stack([torch.log(gm) + lp_mem, torch.log1p(-gm) + lp_rob], -1), -1)  # [B,N]
        else:               # horizon-specific gate: per-step marginal mixture likelihood
            wts = mixture_weights(out, gm).clamp_min(1e-9)
            mu_ = torch.cat([out["mem_mu"], out["rob_mu"]], 1)
            sg_ = torch.cat([out["mem_sigma"], out["rob_sigma"]], 1).clamp_min(1e-6)
            ld = -0.5 * (((y.unsqueeze(1) - mu_) / sg_) ** 2 + 2 * torch.log(sg_) + LOG_2PI)
            lp = (torch.logsumexp(torch.log(wts) + ld, 1) * m).sum(1)                   # [B,N]
        w = weight[:, None] if weight is not None else 1.0
        nll = -(w * lp / cnt).mean()
        mae = (w * ((out["pred_mean"] - y).abs() * m).sum(1) / cnt).mean()
        logs = dict(nll=nll.item(), mae=mae.item())
        if cfg.get("loss_type", "nll") == "crps":
            # proper scoring rule matched to evaluation: the gate is fitted on the CRPS of the blended forecast
            crps = (w * (mixture_crps(y, out["gate_mix"], out) * m).sum(1) / cnt).mean()
            loss = cfg.crps_weight * crps + cfg.nll_weight * nll
            logs["crps"] = crps.item()
        else:
            loss = nll + cfg.mae_weight * mae
        if cfg.similarity_alignment_weight > 0 and not cfg.regime_gate:
            gr = out["gate_raw"].clamp(1e-6, 1 - 1e-6)
            st = out["s_bar"].detach().clamp(0, 1)
            vt = out["c_mem_ref"].detach()
            vt = (vt - vt.amin()) / (vt.amax() - vt.amin() + 1e-6)
            l_sim = -(st * gr.log() + (1 - st) * (1 - gr).log()).mean()
            l_var = -(vt * (1 - gr).log() + (1 - vt) * gr.log()).mean()
            loss = loss + cfg.similarity_alignment_weight * l_sim + cfg.dispersion_alignment_weight * l_var
        if cfg.regime_gate and cfg.regime_sup and cfg.lambda_gate > 0:
            pw = torch.as_tensor(cfg.regime_pos_weight, device=y.device)
            l_gate = F.binary_cross_entropy_with_logits(out["regime_logits"], b["regime"], pos_weight=pw)
            loss = loss + cfg.lambda_gate * l_gate
            logs["gate"] = l_gate.item()
        if cfg.lambda_fd > 0:
            p_lo, p_up = mixture_tail_mass(out, b["lower"], b["upper"])
            l_fd = (p_lo + p_up).mean()
            loss = loss + cfg.lambda_fd * l_fd
            logs["fd"] = l_fd.item()
        return loss, logs


@torch.no_grad()
def sample_mixture(out, S, gen, lower=None, upper=None):
    """Draw S trajectories per node from the gated mixture (REMIND sampling: one component per
    node trajectory). With bounds (M3) the mixture is truncated to [lower, upper]: each component's
    weight is multiplied by its probability mass inside the bounds and renormalised, and the selected
    component is sampled from its truncated distribution by inverse-CDF sampling (Algorithm 1)."""
    mu = torch.cat([out["mem_mu"], out["rob_mu"]], 1)
    sg = torch.cat([out["mem_sigma"], out["rob_sigma"]], 1).clamp_min(1e-6)
    B, Mx, T, N = mu.shape
    w = mixture_weights(out, out["gate_raw"]).expand(-1, -1, T, -1)                 # [B,M,T,N]
    if lower is not None:
        clo = torch.special.ndtr((lower[:, None, None, :] - mu) / sg)               # [B,M,T,N]
        chi = torch.special.ndtr((upper[:, None, None, :] - mu) / sg)
        q = (chi - clo).clamp_min(0)
        wq = w * q
        ok = wq.sum(1, keepdim=True) > 1e-12                                        # all mass outside: keep prior weights
        w = torch.where(ok, wq, w)
    # one uniform per trajectory, inverted through each step's component CDF: with a scalar gate this is
    # REMIND's one-component-per-trajectory sampling; with a horizon gate it keeps trajectories coherent
    uc = torch.rand(B, S, 1, N, device=mu.device, generator=gen)
    cdf = w.cumsum(1)                                                               # [B,M,T,N]
    idx = (cdf.unsqueeze(1) < uc.unsqueeze(2) * cdf[:, -1:].unsqueeze(1)).sum(2).clamp(max=Mx - 1)  # [B,S,T,N]
    m, s = mu.gather(1, idx), sg.gather(1, idx)                                     # [B,S,T,N]
    u = torch.rand(m.shape, device=m.device, generator=gen)
    if lower is None:
        return m + s * torch.special.ndtri(u.clamp(1e-6, 1 - 1e-6))
    lo, hi = clo.gather(1, idx), chi.gather(1, idx)
    hi = torch.maximum(hi, lo + 1e-6)
    p = (lo + u * (hi - lo)).clamp(1e-6, 1 - 1e-6)
    z = m + s * torch.special.ndtri(p)
    # numerical guard only (ndtri saturates far in the tails); the draw is already from the truncated law
    return torch.minimum(torch.maximum(z, lower[:, None, None, :]), upper[:, None, None, :])


def truncated_gauss(mu, sg, S, gen, lower, upper):
    """S draws from N(mu, sg^2) truncated to [lower, upper] (inverse CDF). mu, sg: [B,T,N]; bounds [B,N]."""
    lo = torch.special.ndtr((lower[:, None, :] - mu) / sg)
    hi = torch.special.ndtr((upper[:, None, :] - mu) / sg)
    hi = torch.maximum(hi, lo + 1e-6)
    u = torch.rand((mu.size(0), S) + mu.shape[1:], device=mu.device, generator=gen)
    p = (lo[:, None] + u * (hi - lo)[:, None]).clamp(1e-6, 1 - 1e-6)
    z = mu[:, None] + sg[:, None] * torch.special.ndtri(p)
    return torch.minimum(torch.maximum(z, lower[:, None, None, :]), upper[:, None, None, :])
