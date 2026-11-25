"""E7: scalability of REMIND vs WARP-Traffic (retrieval + forward + sampling) with network size."""
import sys, os, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from easydict import EasyDict as edict
from wremind.data import Region, Retriever
from wremind.models import WREMIND, sample_mixture
from wremind.train import fwd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # holds data_raw, data_proc, results, paper
OUT = os.path.join(ROOT, "results", "E7")
os.makedirs(OUT, exist_ok=True)


def cfg(joint):
    return edict(hidden_dim=64, dropout=0.1, pred_len=12, memory_mixtures=4, robust_mixtures=4, memory_sigma_floor=1e-3,
                 robust_sigma_floor=0.1, deviation_residual_scale=0.3, similarity_alignment_weight=0.5, dispersion_alignment_weight=0.5,
                 mae_weight=30.0, memory_dropout=0.3, wx_feat=joint, regime_gate=joint, regime_sup=joint, lambda_fd=0.5 if joint else 0,
                 lambda_gate=0.2, event_aug=joint, event_perturb=1.0, regime_pos_weight=[5., 10., 10., 20.], joint=joint, truncate=joint, backbone="gwnet")


rows = []
for region, sub in [("D3", None), ("D4", "subset_core.npy"), ("D7", "subset_core.npy"), ("D7", None), ("D4", None)]:
    s = np.load(os.path.join(ROOT, "data_proc", "regions", region, sub)) if sub else None
    R = Region(region, node_subset=s)
    te = R.origins("test", 6)[:64]
    for joint in [False, True]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        c = cfg(joint)
        t0 = time.time(); ret = Retriever(R, M=30, K=8, joint=joint); t_build = time.time() - t0
        model = WREMIND(c, R.adj, R.F, 5).cuda().eval()
        gen = torch.Generator(device="cuda").manual_seed(0)
        bs = 4
        tr, tm, ts = [], [], []
        with torch.no_grad():
            for i in range(0, len(te), bs):
                b = R.batch(te[i:i + bs])
                torch.cuda.synchronize(); a = time.time()
                b.update(ret(b["t0"], b["x"], "test")); torch.cuda.synchronize(); bb = time.time()
                out = fwd(model, b); torch.cuda.synchronize(); cc = time.time()
                smp = sample_mixture(out, 100, gen, b["lower"], b["upper"]) if joint else sample_mixture(out, 100, gen)
                torch.cuda.synchronize(); dd = time.time()
                if i > 0:
                    tr.append((bb - a) / bs); tm.append((cc - bb) / bs); ts.append((dd - cc) / bs)
        rows.append(dict(region=region, N=R.N, model="WARP-Traffic" if joint else "REMIND", bank=len(ret.bank), index_build_s=t_build,
                         retrieval_ms=1000 * np.mean(tr), forward_ms=1000 * np.mean(tm), sampling_ms=1000 * np.mean(ts),
                         total_ms=1000 * (np.mean(tr) + np.mean(tm) + np.mean(ts)), peak_gb=torch.cuda.max_memory_allocated() / 1e9))
        print(rows[-1], flush=True)
        del model, ret
    del R
    torch.cuda.empty_cache()
json.dump(rows, open(os.path.join(OUT, "scale.json"), "w"), indent=1)
