"""Re-evaluate a trained run (inference only) with the current evaluation code.

Rebuilds the model from the argument record in its train.log and the saved model.pt, then calls train.evaluate,
which overwrites summary*.json / series*.npz / extra*.npz of that run. No training.

usage: python -m wremind.reeval --region D3 --run wremind_s0 [--save_extra] [--node_frac 0.8] [--wx_noise 0.5] [--wx_drop]
"""
import os, argparse, numpy as np, pandas as pd, torch
from wremind.data import Region, Retriever
from wremind import train as T
from wremind.pool import expert


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="D3")
    p.add_argument("--run", required=True)
    p.add_argument("--save_extra", action="store_true")
    p.add_argument("--node_frac", type=float, default=1.0)
    p.add_argument("--wx_noise", type=float, default=0.0)
    p.add_argument("--wx_drop", action="store_true")
    a = p.parse_args()
    d = os.path.join(T.RES, a.region, a.run)
    rec = [l for l in open(os.path.join(d, "train.log")) if l.startswith("{")][-1]
    import json
    rec = json.loads(rec[:rec.rindex("}") + 1])
    subset = np.load(rec["subset"]) if rec.get("subset") else None
    R = Region(a.region, node_subset=subset)
    e = expert(a.run, a.region, R)
    args = e.args
    args.save_extra, args.node_frac, args.wx_noise, args.wx_drop = a.save_extra, a.node_frac, a.wx_noise, a.wx_drop
    T.set_seed(args.seed)
    ret = None
    if e.head == "remind":
        bank_start = None
        if getattr(args, "bank_months", 0) > 0:
            bank_start = str(pd.Timestamp("2020-10-31") - pd.DateOffset(months=args.bank_months))
        ret = Retriever(R, M=args.M, K=args.K, joint=e.cfg.joint, alpha=args.alpha, tau_w=args.tau_w,
                        bank_start=bank_start, seed=args.seed, tod_window=args.tod_window if e.cfg.joint else 0)
    te = R.origins("test", args.test_stride)
    log = open(os.path.join(d, "train.log"), "a")
    P = lambda *x: (print(*x, flush=True), log.write("REEVAL " + " ".join(map(str, x)) + "\n"), log.flush())
    T.evaluate(args, R, e.m, e.head, e.cfg, ret, te, d, P)


if __name__ == "__main__":
    main()
