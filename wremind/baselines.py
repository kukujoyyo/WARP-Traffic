"""Baselines re-implemented in one framework: Graph WaveNet, STID, STAEformer, DeepAR,
with point / Gaussian / quantile / MC-dropout heads and optional weather inputs."""
import math, torch, torch.nn as nn, torch.nn.functional as F

QUANTILES = torch.arange(0.05, 0.96, 0.05)


def out_dim(head):
    return {"point": 1, "gauss": 2, "quantile": len(QUANTILES)}[head]


# ---------------------------------------------------------------- Graph WaveNet (Wu et al., 2019)
class GWNet(nn.Module):
    def __init__(self, N, supports, in_dim, out_len=12, head="point", C=32, skip=128, end=256, blocks=4, layers=2, dropout=0.3):
        super().__init__()
        self.head, self.out_len, self.dropout = head, out_len, dropout
        self.supports = [s for s in supports]
        self.nodevec1 = nn.Parameter(torch.randn(N, 10) * 0.1)
        self.nodevec2 = nn.Parameter(torch.randn(10, N) * 0.1)
        self.start = nn.Conv2d(in_dim, C, 1)
        n_sup = len(supports) + 1
        self.filt, self.gate, self.res, self.skipc, self.bn, self.gconv = [nn.ModuleList() for _ in range(6)]
        self.dil = []
        rf = 1
        for _ in range(blocks):
            d, add = 1, 1
            for _ in range(layers):
                self.filt.append(nn.Conv2d(C, C, (1, 2), dilation=d))
                self.gate.append(nn.Conv2d(C, C, (1, 2), dilation=d))
                self.skipc.append(nn.Conv2d(C, skip, 1))
                self.bn.append(nn.BatchNorm2d(C))
                self.gconv.append(nn.Conv2d(C * (n_sup * 2 + 1), C, 1))
                self.dil.append(d)
                rf += add; add *= 2; d *= 2
        self.rf = rf
        self.end1 = nn.Conv2d(skip, end, 1)
        self.end2 = nn.Conv2d(end, out_len * out_dim(head), 1)

    def nconv(self, x, A):
        return torch.einsum("bcnt,nm->bcmt", x, A)

    def forward(self, inp, mc=False):
        # inp [B, T, N, C] -> [B, out, T_out, N]
        x = inp.permute(0, 3, 2, 1)
        if x.size(3) < self.rf:
            x = F.pad(x, (self.rf - x.size(3), 0))
        x = self.start(x)
        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), 1)
        sup = self.supports + [adp]
        skip = 0
        for i in range(len(self.filt)):
            r = x
            h = torch.tanh(self.filt[i](x)) * torch.sigmoid(self.gate[i](x))
            s = self.skipc[i](h)
            skip = s + (skip[..., -s.size(3):] if torch.is_tensor(skip) else 0)
            outs = [h]
            for A in sup:
                x1 = self.nconv(h, A); outs.append(x1)
                outs.append(self.nconv(x1, A))
            h = self.gconv[i](torch.cat(outs, 1))
            h = F.dropout(h, self.dropout, training=self.training or mc)
            x = self.bn[i](h + r[..., -h.size(3):])
        y = self.end2(F.relu(self.end1(F.relu(skip))))       # [B, out*D, N, 1]
        B, _, N, _ = y.shape
        return y.reshape(B, self.out_len, -1, N).permute(0, 2, 1, 3)   # [B, D, T, N]


# ---------------------------------------------------------------- STID (Shao et al., 2022)
class STID(nn.Module):
    def __init__(self, N, in_dim, in_len=12, out_len=12, head="point", D=32, layers=3, extra=0, dropout=0.15):
        super().__init__()
        self.head, self.out_len, self.dropout = head, out_len, dropout
        self.node_emb = nn.Parameter(torch.empty(N, D)); nn.init.xavier_uniform_(self.node_emb)
        self.tod_emb = nn.Parameter(torch.empty(288, D)); nn.init.xavier_uniform_(self.tod_emb)
        self.dow_emb = nn.Parameter(torch.empty(7, D)); nn.init.xavier_uniform_(self.dow_emb)
        self.ts_emb = nn.Conv2d(in_dim * in_len, D, 1)
        hid = D * 4
        self.enc = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 1), nn.ReLU(), nn.Dropout(dropout), nn.Conv2d(hid, hid, 1)) for _ in range(layers)])
        self.reg = nn.Conv2d(hid, out_len * out_dim(head), 1)

    def forward(self, inp, tod, dow, mc=False):
        B, T, N, C = inp.shape
        x = inp.permute(0, 1, 3, 2).reshape(B, T * C, N, 1)
        h = torch.cat([self.ts_emb(x), self.node_emb.T[None, :, :, None].expand(B, -1, -1, -1),
                       self.tod_emb[tod][:, :, None, None].expand(-1, -1, N, -1),
                       self.dow_emb[dow][:, :, None, None].expand(-1, -1, N, -1)], 1)
        for l in self.enc:
            if mc:
                l.train()
            h = h + l(h)
        y = self.reg(h)
        return y.reshape(B, self.out_len, -1, N).permute(0, 2, 1, 3)


# ---------------------------------------------------------------- STAEformer (Liu et al., 2023)
class AttnLayer(nn.Module):
    def __init__(self, D, heads=4, ff=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, ff), nn.ReLU(), nn.Linear(ff, D))
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.dp(self.attn(x, x, x, need_weights=False)[0]))
        return self.n2(x + self.dp(self.ff(x)))


class STAEformer(nn.Module):
    def __init__(self, N, in_dim, in_len=12, out_len=12, head="point", d_in=24, d_tod=24, d_dow=24, d_ad=80, layers=3, dropout=0.1):
        super().__init__()
        self.head, self.out_len, self.in_len = head, out_len, in_len
        self.inp = nn.Linear(in_dim, d_in)
        self.tod = nn.Embedding(288, d_tod)
        self.dow = nn.Embedding(7, d_dow)
        self.ad = nn.Parameter(torch.empty(in_len, N, d_ad)); nn.init.xavier_uniform_(self.ad)
        D = d_in + d_tod + d_dow + d_ad
        self.t_layers = nn.ModuleList([AttnLayer(D, dropout=dropout) for _ in range(layers)])
        self.s_layers = nn.ModuleList([AttnLayer(D, dropout=dropout) for _ in range(layers)])
        self.out = nn.Linear(in_len * D, out_len * out_dim(head))

    def forward(self, inp, tod_seq, dow_seq, mc=False):
        B, T, N, _ = inp.shape
        h = torch.cat([self.inp(inp), self.tod(tod_seq)[:, :, None].expand(-1, -1, N, -1),
                       self.dow(dow_seq)[:, :, None].expand(-1, -1, N, -1), self.ad[None].expand(B, -1, -1, -1)], -1)
        D = h.size(-1)
        for l in self.t_layers:
            h = l(h.permute(0, 2, 1, 3).reshape(B * N, T, D)).reshape(B, N, T, D).permute(0, 2, 1, 3)
        for l in self.s_layers:
            h = l(h.reshape(B * T, N, D)).reshape(B, T, N, D)
        y = self.out(h.permute(0, 2, 1, 3).reshape(B, N, T * D))
        return y.reshape(B, N, self.out_len, -1).permute(0, 3, 2, 1)


# ---------------------------------------------------------------- DeepAR (Salinas et al., 2020)
class DeepAR(nn.Module):
    """Per-sensor autoregressive GRU with Gaussian likelihood (node embedding + calendar covariates)."""

    def __init__(self, N, cov_dim, H=64, layers=2, emb=16):
        super().__init__()
        self.emb = nn.Embedding(N, emb)
        self.rnn = nn.GRU(1 + cov_dim + emb, H, layers, batch_first=True, dropout=0.1)
        self.mu, self.sg = nn.Linear(H, 1), nn.Linear(H, 1)

    def _inp(self, y_prev, cov, e):
        return torch.cat([y_prev, cov, e], -1)

    def forward(self, x, y, cov_h, cov_f):
        # x [B,T,N], y [B,T,N]; cov_* [B,T,N,C]  -> teacher forced (mu, sigma) over horizon
        B, T, N = x.shape
        e = self.emb.weight[None, None].expand(B, T + y.size(1), -1, -1)
        seq_y = torch.cat([x, y], 1)
        prev = torch.cat([torch.zeros_like(seq_y[:, :1]), seq_y[:, :-1]], 1)
        cov = torch.cat([cov_h, cov_f], 1)
        inp = self._inp(prev.unsqueeze(-1), cov, e).permute(0, 2, 1, 3).reshape(B * N, -1, cov.size(-1) + 1 + e.size(-1))
        h, _ = self.rnn(inp)
        h = h[:, T:].reshape(B, N, -1, h.size(-1)).permute(0, 2, 1, 3)
        return self.mu(h).squeeze(-1), F.softplus(self.sg(h)).squeeze(-1) + 1e-3

    @torch.no_grad()
    def sample(self, x, cov_h, cov_f, S, gen):
        B, T, N = x.shape
        Tp = cov_f.size(1)
        e = self.emb.weight[None].expand(B, -1, -1)
        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        inp = self._inp(prev.unsqueeze(-1), cov_h, e[:, None].expand(-1, T, -1, -1)).permute(0, 2, 1, 3).reshape(B * N, T, -1)
        _, hs = self.rnn(inp)
        hs = hs.repeat_interleave(S, 1)                                     # [L, B*N*S, H]
        last = x[:, -1].reshape(B * N, 1).repeat_interleave(S, 0)
        outs = []
        for k in range(Tp):
            c = cov_f[:, k].reshape(B * N, -1).repeat_interleave(S, 0)
            ek = e.reshape(B * N, -1).repeat_interleave(S, 0)
            o, hs = self.rnn(torch.cat([last, c, ek], -1)[:, None], hs)
            mu, sg = self.mu(o[:, 0]), F.softplus(self.sg(o[:, 0])) + 1e-3
            last = mu + sg * torch.randn(mu.shape, device=mu.device, generator=gen)
            outs.append(last)
        s = torch.cat(outs, 1).reshape(B, N, S, Tp)
        return s.permute(0, 2, 3, 1)                                       # [B,S,T,N]


def pinball(qpred, y, mask):
    # qpred [B,Q,T,N]
    q = QUANTILES.to(y.device)[None, :, None, None]
    d = y.unsqueeze(1) - qpred
    l = torch.maximum(q * d, (q - 1) * d)
    return (l * mask.unsqueeze(1)).sum() / (mask.sum() * len(QUANTILES)).clamp_min(1)
