"""Streaming probabilistic metrics, stratified by event type at the target time."""
import numpy as np, torch

STRATA = ["all", "normal", "heat", "rain", "smoke", "incident", "lightrain", "heat95", "anyevent"]
HORIZONS = [3, 6, 12]          # 15, 30, 60 min
PIT_BINS = 20


def strata_masks(ev, valid):
    """ev [B,T,N] uint8 -> dict of bool masks."""
    b = lambda k: ((ev >> k) & 1).bool()
    heat, rain, smoke, inc, lrain, h95, hol = b(0), b(1), b(2), b(3), b(4), b(5), b(6)
    anyev = heat | rain | smoke | inc
    m = {"all": valid, "normal": valid & ~anyev & ~lrain & ~hol, "heat": valid & heat, "rain": valid & rain,
         "smoke": valid & smoke, "incident": valid & inc, "lightrain": valid & lrain, "heat95": valid & h95,
         "anyevent": valid & anyev}
    return m


def crps_from_sorted(xs, y):
    """xs [B,S,T,N] sorted along S, y [B,T,N] -> CRPS [B,T,N] (energy form, unbiased-free empirical)."""
    S = xs.size(1)
    t1 = (xs - y.unsqueeze(1)).abs().mean(1)
    w = (2 * torch.arange(1, S + 1, device=xs.device, dtype=xs.dtype) - S - 1).view(1, S, 1, 1)
    t2 = (w * xs).sum(1) / (S * S)
    return t1 - t2


class Accumulator:
    def __init__(self, n_origins, T=12, probabilistic=True):
        self.prob = probabilistic
        self.tsum = {}
        self.T = T
        self.series = {k: np.zeros((n_origins, T, 3), np.float32) for k in ["all", "normal", "anyevent", "heat", "rain", "smoke", "incident"]}  # crps, mae, count
        # per-origin calibration sums (for block-bootstrap calibration analysis): 90% coverage, 90% width, 50% coverage
        self.cal_keys = list(self.series) if probabilistic else []
        for k in self.cal_keys:
            self.series[k + "_cal"] = np.zeros((n_origins, T, 3), np.float32)
        self.pit = {k: np.zeros(PIT_BINS, np.int64) for k in STRATA}
        self.es = {"all": [0.0, 0], "regionevent": [0.0, 0]}
        self.ptr = 0

    @torch.no_grad()
    def update(self, samples, point, y, valid, ev, lower=None, upper=None, std=1.0, mean=0.0):
        """samples [B,S,T,N] (original units) or None; point [B,T,N]; y [B,T,N] original units."""
        B = y.size(0)
        masks = strata_masks(ev, valid)
        err = point - y
        if samples is not None:
            xs, _ = torch.sort(samples, 1)
            crps = crps_from_sorted(xs, y)
            S = xs.size(1)
            q05, q95 = xs[:, int(round(0.05 * (S - 1)))], xs[:, int(round(0.95 * (S - 1)))]
            cover = ((y >= q05) & (y <= q95)).float()
            q25, q75 = xs[:, int(round(0.25 * (S - 1)))], xs[:, int(round(0.75 * (S - 1)))]
            cover50 = ((y >= q25) & (y <= q75)).float()
            width = q95 - q05
            wink = width + (2 / 0.1) * ((q05 - y).clamp_min(0) + (y - q95).clamp_min(0))
            pit = (xs <= y.unsqueeze(1)).float().mean(1)
            viol = None
            if lower is not None:
                viol = ((samples < lower[:, None, None, :] - 1e-3) | (samples > upper[:, None, None, :] + 1e-3)).float().mean(1)
                neg = (samples < -1e-3).float().mean(1)
            pit_bin = (pit * PIT_BINS).clamp(max=PIT_BINS - 1).long()
        else:
            crps = err.abs()
        M = torch.stack([masks[k] for k in STRATA]).float()          # [K,B,T,N]
        big = (y >= 20).float()
        mets = {"n": torch.ones_like(y), "ae": err.abs(), "se": err ** 2,
                "ape": err.abs() / y.clamp_min(1) * big, "n_ape": big, "crps": crps}
        if samples is not None:
            mets.update(cov=cover, wid=width, wink=wink)
            if viol is not None:
                mets.update(viol=viol, neg=neg)
        for name, v in mets.items():
            tot = (M * v[None]).sum((1, 3))                               # [K,T]
            if name not in self.tsum:
                self.tsum[name] = torch.zeros_like(tot, dtype=torch.float64)
            self.tsum[name] += tot.double()
        if samples is not None:
            for i, k in enumerate(STRATA):
                self.pit[k] += torch.bincount(pit_bin[masks[k]], minlength=PIT_BINS).cpu().numpy()
        for k in self.series:
            if k.endswith("_cal"):
                continue
            mf = masks[k].float()
            s = self.series[k][self.ptr:self.ptr + B]
            s[..., 0] = (crps * mf).sum(-1).cpu().numpy()
            s[..., 1] = (err.abs() * mf).sum(-1).cpu().numpy()
            s[..., 2] = mf.sum(-1).cpu().numpy()
            if samples is not None and k in self.cal_keys:
                c = self.series[k + "_cal"][self.ptr:self.ptr + B]
                c[..., 0] = (cover * mf).sum(-1).cpu().numpy()
                c[..., 1] = (width * mf).sum(-1).cpu().numpy()
                c[..., 2] = (cover50 * mf).sum(-1).cpu().numpy()
        if samples is not None:   # spatial energy score at 15/30/60 min (sensor vector), 50 draws
            sub = samples[:, :50]
            regev = (masks["anyevent"].float().mean(-1) > 0.2)          # [B,T]
            for h in [2, 5, 11]:
                xh, yh = sub[:, :, h], y[:, h]
                vm = valid[:, h].float()
                xh, yh = xh * vm[:, None], yh * vm
                t1 = (xh - yh[:, None]).norm(dim=-1).mean(1)
                t2 = torch.cdist(xh, xh).mean((1, 2)) / 2
                es = (t1 - t2) / vm.sum(-1).clamp_min(1).sqrt()
                self.es["all"][0] += es.sum().item(); self.es["all"][1] += B
                r = regev[:, h]
                self.es["regionevent"][0] += es[r].sum().item(); self.es["regionevent"][1] += int(r.sum())
        self.ptr += B

    def summary(self):
        out = {}
        tot = {n: v.cpu().numpy() for n, v in self.tsum.items()}
        for i, k in enumerate(STRATA):
            out[k] = {}
            for key, sel in [("15min", [2]), ("30min", [5]), ("60min", [11]), ("avg", list(range(self.T)))]:
                agg = {n: float(v[i, sel].sum()) for n, v in tot.items()}
                n = agg.get("n", 0)
                if n == 0:
                    continue
                r = dict(n=n, MAE=agg["ae"] / n, RMSE=(agg["se"] / n) ** 0.5,
                         MAPE=100 * agg["ape"] / max(agg["n_ape"], 1), CRPS=agg["crps"] / n)
                if "cov" in agg:
                    r.update(PICP90=agg["cov"] / n, MPIW90=agg["wid"] / n, Winkler90=agg["wink"] / n)
                if "viol" in agg:
                    r.update(Violation=agg["viol"] / n, NegMass=agg["neg"] / n)
                out[k][key] = r
            if self.prob and self.pit[k].sum() > 0:
                p = self.pit[k] / self.pit[k].sum()
                cdf = np.concatenate([[0], np.cumsum(p)])
                lv = np.arange(0.1, 1.0, 0.1)
                cov = [np.interp(0.5 + a / 2, np.linspace(0, 1, PIT_BINS + 1), cdf) - np.interp(0.5 - a / 2, np.linspace(0, 1, PIT_BINS + 1), cdf) for a in lv]
                out[k]["ECE"] = float(np.mean(np.abs(np.array(cov) - lv)))
                out[k]["PIT"] = p.tolist()
        out["ES"] = {k: v[0] / max(v[1], 1) for k, v in self.es.items()}
        return out
