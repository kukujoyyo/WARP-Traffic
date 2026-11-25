"""Region data, chronological splits, window batching and GPU two-stage retrieval.

Retrieval follows REMIND (coarse sample-level search, then node-level search over the
retrieved samples and each node's graph neighbours; similarity exp(-L1/tau)).
The original repository pre-computes dense retrieval caches on disk, which does not
scale to multi-year, thousand-sensor data, so both stages run on the GPU per batch.
With ``joint=True`` the search is WARP-Traffic's M1 (weather-calendar re-ranking).
"""
import os, json, numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
PROC = os.path.join(ROOT, "data_proc")
EVENT_BITS = {"heat": 0, "rain": 1, "smoke": 2, "incident": 3, "lightrain": 4, "heat95": 5, "holiday": 6}
REGIMES = ["heat", "rain", "smoke", "incident"]
TH, TP = 12, 12

SPLITS = {
    "train": [("2017-01-01", "2020-03-14 23:55"), ("2020-07-01", "2020-10-31 23:55")],
    "val": [("2020-11-01", "2020-12-31 23:55")],
    "test": [("2021-01-01", "2021-12-31 23:55")],
}


class Region:
    """All arrays of one district kept on the GPU (flow is standardised with a global scalar)."""

    def __init__(self, name, device="cuda", node_subset=None):
        d = os.path.join(PROC, "regions", name)
        self.name, self.device, self.dir = name, device, d
        flow = np.load(os.path.join(d, "flow.npy"))            # [T, N] veh/5min, NaN = missing
        self.time = pd.DatetimeIndex(np.load(os.path.join(d, "time.npy")))
        self.meta = pd.read_csv(os.path.join(d, "meta.csv"))
        adj = np.load(os.path.join(d, "adj.npy"))
        wx = np.load(os.path.join(d, "wx_hourly.npy"))           # [H, N, F] standardised weather
        self.wx_raw = np.load(os.path.join(d, "wx_hourly_raw.npy"), mmap_mode="r")
        ev = np.load(os.path.join(d, "events.npy"))              # [T, N] uint8 bitmask
        cap = np.load(os.path.join(d, "capacity.npy"))            # [N] veh/5min
        self.cap_factor_hourly = np.load(os.path.join(d, "cap_factor_hourly.npy"))  # [H, N]
        if node_subset is not None:
            flow, adj, wx, ev, cap = flow[:, node_subset], adj[np.ix_(node_subset, node_subset)], wx[:, node_subset], ev[:, node_subset], cap[node_subset]
            self.cap_factor_hourly = self.cap_factor_hourly[:, node_subset]
            self.meta = self.meta.iloc[node_subset].reset_index(drop=True)
        self.N = flow.shape[1]
        self.T = flow.shape[0]
        self.wx_names = json.load(open(os.path.join(d, "wx_names.json")))
        self.F = wx.shape[2]

        train_mask = self.split_mask("train")
        valid = ~np.isnan(flow)
        self.mean = float(np.nanmean(flow[train_mask]))
        self.std = float(np.nanstd(flow[train_mask]))
        z = np.where(valid, (flow - self.mean) / self.std, 0.0).astype(np.float32)

        # Causal evaluation (default on; WREMIND_CAUSAL=0 restores the original behaviour). flow.npy contains linear
        # interpolation of gaps <= 30 min, which uses values after the gap. With observed.npy available, model inputs
        # use the last observed value instead (z_in), and only observed targets are scored (ymask).
        obs_f = os.path.join(d, "observed.npy")
        self.causal = os.environ.get("WREMIND_CAUSAL", "1") == "1" and os.path.exists(obs_f)
        z_in = z
        if self.causal:
            obs = np.load(obs_f, mmap_mode="r")
            obs = np.asarray(obs[:, node_subset] if node_subset is not None else obs)
            fill = valid & ~obs
            z_in = z.copy()
            for ti, j in zip(*np.nonzero(fill)):             # few cells (<0.03%): carry the last observed value forward
                k = ti - 1
                while k >= 0 and fill[k, j]:
                    k -= 1
                z_in[ti, j] = z[k, j] if k >= 0 and valid[k, j] else 0.0
            self.n_filled = int(fill.sum())

        t = torch.device(device)
        self.z = torch.from_numpy(z).to(t)
        self.valid = torch.from_numpy(valid).to(t)
        self.z_in = torch.from_numpy(z_in).to(t) if self.causal else self.z
        self.scored = torch.from_numpy(obs.copy()).to(t) if self.causal else self.valid
        self.events = torch.from_numpy(ev).to(t)
        self.wx = torch.from_numpy(wx.astype(np.float32)).to(t)
        self.cap = torch.from_numpy(cap.astype(np.float32)).to(t)          # 99.9th pct (near-capacity alerts)
        # M3 admissible bound: maximum discharge observed at the detector in the training period
        self.cap_max = torch.from_numpy(np.nanmax(flow[train_mask], 0).astype(np.float32)).to(t)
        self.cap_factor = torch.from_numpy(self.cap_factor_hourly.astype(np.float32)).to(t)
        self.adj = adj.astype(np.float32)
        hours = (self.time - pd.Timestamp("2017-01-01")) // pd.Timedelta("1h")
        self.hour_idx = torch.from_numpy(np.asarray(hours, dtype=np.int64)).to(t)
        tod = (self.time.hour * 60 + self.time.minute).values / 1440.0
        dow = self.time.dayofweek.values
        hol = (ev[:, 0] >> EVENT_BITS["holiday"]) & 1 if ev.ndim == 2 else np.zeros(self.T)
        cal = np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), (dow >= 5) | (hol > 0),
                        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], 1).astype(np.float32)
        self.cal = torch.from_numpy(cal).to(t)
        self.tod_idx = torch.from_numpy((self.time.hour * 12 + self.time.minute // 5).values.astype(np.int64)).to(t)
        self.dow_idx = torch.from_numpy(dow.astype(np.int64)).to(t)

        # neighbour table for node-level retrieval: self + J strongest adjacency links
        J = 4
        a = adj.copy(); np.fill_diagonal(a, -1)
        nb = np.argsort(-a, 1)[:, :J]
        nb = np.where(np.take_along_axis(a, nb, 1) > 0, nb, np.arange(self.N)[:, None])
        self.nbr = torch.from_numpy(np.concatenate([np.arange(self.N)[:, None], nb], 1)).to(t)

    def perturb_weather(self, noise=0.0, drop=False, seed=7):
        """Robustness test: perturb (noise) or remove (drop) the standardised weather of the test period. Because every
        route reads self.wx / self.cap_factor, this reaches the model inputs, the weather context of the gate, the
        weather similarity of retrieved analogs (query side; the memory bank keeps its observed weather) and the
        capacity factor of M3, which is recomputed from the perturbed precipitation and visibility."""
        if noise <= 0 and not drop:
            return
        h0 = int(self.hour_idx[int(np.where(self.split_mask("test"))[0][0])])
        w = self.wx[h0:]
        if drop:
            w = torch.zeros_like(w)
        else:
            g = torch.Generator(device=w.device).manual_seed(seed)
            w = w + noise * torch.randn(w.shape, device=w.device, generator=g)
        self.wx[h0:] = w
        nm = json.load(open(os.path.join(self.dir, "wx_norm.json")))
        mu, sd = np.asarray(nm["mean"]), np.asarray(nm["std"])
        ip, iv = self.wx_names.index("log_precip"), self.wx_names.index("vis")
        p = torch.expm1(w[..., ip] * float(sd[ip]) + float(mu[ip])).clamp_min(0)
        v = w[..., iv] * float(sd[iv]) + float(mu[iv])
        f = torch.where(p >= 6.4, 0.86, torch.where(p >= 2.5, 0.93, torch.where(p >= 0.25, 0.97, 1.0)))
        f = f * torch.where(v < 0.5, 0.90, 1.0)
        self.cap_factor[h0:] = f.to(self.cap_factor.dtype)

    def split_mask(self, split):
        m = np.zeros(self.T, bool)
        for a, b in SPLITS[split]:
            m |= (self.time >= pd.Timestamp(a)) & (self.time <= pd.Timestamp(b))
        return m

    def origins(self, split, stride=1):
        """Indices t0 of the last input step; the whole window [t0-11, t0+12] must lie in the split."""
        m = self.split_mask(split)
        ok = np.zeros(self.T, bool)
        idx = np.arange(TH - 1, self.T - TP)
        inside = m[idx - TH + 1] & m[idx + TP]
        # the window must also be contiguous in time (no split boundary gap)
        contiguous = (self.time[idx + TP] - self.time[idx - TH + 1]) == pd.Timedelta(minutes=5 * (TH + TP - 1))
        ok[idx[inside & contiguous]] = True
        o = np.where(ok)[0]
        return o[::stride]

    # ------------------------------------------------------------------ batching
    def window(self, t0, offs):
        """t0: [B] long tensor -> indices [B, L]."""
        return t0[:, None] + offs[None, :]

    def batch(self, t0):
        dev = self.z.device
        t0 = torch.as_tensor(t0, device=dev, dtype=torch.long)
        hist = self.window(t0, torch.arange(-TH + 1, 1, device=dev))
        fut = self.window(t0, torch.arange(1, TP + 1, device=dev))
        x = self.z_in[hist]                                # [B, TH, N] (causally filled inputs)
        y = self.z[fut]                                    # [B, TP, N]
        ym = self.scored[fut]                              # observed targets only
        wx_h = self.wx[self.hour_idx[hist]]                # [B, TH, N, F] (as-of hourly, causal)
        ev_f = self.events[fut]                            # [B, TP, N]
        regime = torch.stack([((ev_f >> EVENT_BITS[r]) & 1).amax(1) for r in REGIMES], -1).float()  # [B, N, R]
        capf = self.cap_factor[self.hour_idx[t0]]          # [B, N] factor known at origin
        upper = (self.cap_max[None, :] * capf - self.mean) / self.std  # [B, N] standardised capacity bound
        lower = torch.full_like(upper, -self.mean / self.std)
        tcal = torch.stack([self.tod_idx[hist].float() / 288, self.dow_idx[hist].float() / 6], -1)   # [B, TH, 2]
        return dict(t0=t0, x=x, y=y, ymask=ym, xmask=self.valid[hist], wx=wx_h, cal=self.cal[t0], ev_f=ev_f,
                    regime=regime, upper=upper, lower=lower, tcal=tcal)


class Retriever:
    """GPU two-stage retrieval over a memory bank of training windows."""

    def __init__(self, region, bank_stride=3, pca_dim=64, M=30, K=8, tau=1.0, joint=False, alpha=0.6,
                 tau_w=1.0, M_pre=90, exclude_days=1, bank_end=None, bank_start=None, seed=0, tod_window=0):
        self.r, self.M, self.K, self.tau, self.joint = region, M, K, tau, joint
        self.alpha, self.tau_w, self.M_pre = alpha, tau_w, M_pre
        self.tod_window = tod_window
        dev = region.z.device
        bank = region.origins("train", stride=bank_stride)
        if bank_end is not None:
            bank = bank[region.time[bank] <= pd.Timestamp(bank_end)]
        if bank_start is not None:
            bank = bank[region.time[bank] >= pd.Timestamp(bank_start)]
        self.bank = torch.as_tensor(bank, device=dev)
        self.ex = exclude_days * 288
        self.hoffs = torch.arange(-TH + 1, 1, device=dev)
        self.foffs = torch.arange(1, TP + 1, device=dev)
        # PCA keys of flattened standardised windows (coarse stage)
        g = torch.Generator(device="cpu").manual_seed(seed)
        sub = self.bank[torch.randperm(len(self.bank), generator=g)[:20000].to(dev)]
        X = region.z[region.window(sub, self.hoffs)].reshape(len(sub), -1)
        self.mu = X.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(X - self.mu, q=pca_dim, center=False)
        self.P = V                                          # [TH*N, d]
        keys = []
        for i in range(0, len(self.bank), 4096):
            b = self.bank[i:i + 4096]
            keys.append((region.z[region.window(b, self.hoffs)].reshape(len(b), -1) - self.mu) @ self.P)
        self.keys = torch.cat(keys)                         # [Nb, d]
        self.knorm = (self.keys ** 2).sum(1)
        # region-level weather/calendar context of bank windows for M1 re-ranking
        self.bank_ctx = self.context(self.bank)

    def context(self, t0):
        r = self.r
        w = r.wx[r.hour_idx[t0]].mean(1)                    # [B, F] region-mean weather at origin
        return torch.cat([w, r.cal[t0][:, :3] * 1.0], -1)

    def node_wx(self, t0):
        return self.r.wx[self.r.hour_idx[t0]]               # [B, N, F]

    @torch.no_grad()
    def __call__(self, t0, x, split):
        r = self.r
        B, N = x.shape[0], r.N
        q = (x.reshape(B, -1) - self.mu) @ self.P
        d2 = (q ** 2).sum(1, keepdim=True) + self.knorm[None] - 2 * q @ self.keys.T   # [B, Nb]
        if split == "train":  # exclude windows near the query (no overlap, no same-day memory)
            near = (self.bank[None, :] - t0[:, None]).abs() < self.ex
            d2 = d2.masked_fill(near, float("inf"))
        if self.tod_window > 0:  # calendar-consistent memory: same day type, time of day within +/- tod_window steps
            tq, tb = r.tod_idx[t0][:, None], r.tod_idx[self.bank][None, :]
            dt = (tq - tb).abs(); dt = torch.minimum(dt, 288 - dt)
            wk = lambda t: (r.dow_idx[t] >= 5)
            ok = (dt <= self.tod_window) & (wk(t0)[:, None] == wk(self.bank)[None, :])
            d2 = d2.masked_fill(~ok, float("inf"))
        Mc = self.M_pre if self.joint else self.M
        dist, idx = torch.topk(d2, Mc, dim=1, largest=False)                           # [B, Mc]
        if self.joint:
            ctx_q = self.context(t0)                                                    # [B, C]
            ctx_c = self.bank_ctx[idx]                                                  # [B, Mc, C]
            wd = (ctx_c - ctx_q[:, None]).norm(dim=-1)
            ts = torch.exp(-(dist.clamp_min(0).sqrt() / (TH * N) ** 0.5) / self.tau)
            score = self.alpha * ts + (1 - self.alpha) * torch.exp(-wd / self.tau_w)
            sel = torch.topk(score, self.M, dim=1).indices
            idx = idx.gather(1, sel)
        cand_t = self.bank[idx]                                                         # [B, M]
        # node-level stage: candidates = M samples x (self + J neighbours)
        nb = r.nbr                                                                      # [N, J+1]
        hw = r.window(cand_t.reshape(-1), self.hoffs)                                   # [B*M, TH]
        ch = r.z[hw][:, :, nb]                                                          # [B*M, TH, N, J+1]
        ch = ch.reshape(B, self.M, TH, N, -1)
        l1 = (ch - x[:, None, :, :, None]).abs().sum(2)                                # [B, M, N, J+1]
        l1 = l1.permute(0, 2, 1, 3).reshape(B, N, -1)                                   # [B, N, M*(J+1)]
        s_tr = torch.exp(-l1 / self.tau)
        if self.joint:
            wq = self.node_wx(t0)                                                       # [B, N, F]
            wc = r.wx[r.hour_idx[cand_t]]                                               # [B, M, N, F]
            wc = wc[:, :, nb]                                                           # [B, M, N, J+1, F]
            wd = (wc - wq[:, None, :, None]).norm(dim=-1).permute(0, 2, 1, 3).reshape(B, N, -1)
            s_w = torch.exp(-wd / self.tau_w)
            score = self.alpha * s_tr + (1 - self.alpha) * s_w
        else:
            s_w = None
            score = s_tr
        top = torch.topk(score, self.K, dim=2).indices                                   # [B, N, K]
        J1 = nb.shape[1]
        m_sel, j_sel = top // J1, top % J1
        node_sel = nb[torch.arange(N, device=x.device)[None, :, None].expand_as(j_sel), j_sel]  # [B, N, K]
        t_sel = cand_t.gather(1, m_sel.reshape(B, -1)).reshape(B, N, self.K)            # [B, N, K]
        hist_idx = t_sel[..., None] + self.hoffs                                        # [B, N, K, TH]
        fut_idx = t_sel[..., None] + self.foffs
        node_e = node_sel[..., None].expand_as(hist_idx)
        ret_x = r.z[hist_idx, node_e].permute(0, 2, 3, 1)                               # [B, K, TH, N]
        ret_y = r.z[fut_idx, node_e].permute(0, 2, 3, 1)                                # [B, K, TP, N]
        ret_s = s_tr.gather(2, top)                                                     # [B, N, K]
        out = dict(ret_x=ret_x, ret_y=ret_y, ret_s=ret_s, ret_t=t_sel, ret_node=node_sel)
        out["ret_ws"] = s_w.gather(2, top) if s_w is not None else torch.ones_like(ret_s)
        # analog event flags (were the retrieved memories themselves events?)
        evm = r.events[fut_idx, node_e]                                                 # [B, N, K, TP]
        out["ret_regime"] = torch.stack([((evm >> EVENT_BITS[k]) & 1).amax(-1).float().mean(-1) for k in REGIMES], -1)
        return out
