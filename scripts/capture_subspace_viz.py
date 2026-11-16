#!/usr/bin/env python
"""Capture answer-position hidden states projected onto the belief subspace, for visualization.

    .venv/bin/python scripts/capture_subspace_viz.py --config configs/main.yaml --model olmo_instruct

For each counterfactual trial (cf_trials.parquet, the belief-stated trials used by the corrected RQ3),
run a forward pass with the answer prefix, grab the answer-position hidden state at the model's erase
layer, and project it onto the belief subspace bases found by the corrected RQ3:
  - ans_U2 (2 x d): the 2-D answer-position belief subspace  -> coords u2_0, u2_1
  - ans_U4 (4 x d): the 4-D version; we keep the first 3 dims -> coords u4_0, u4_1, u4_2
Bases are read from directions_v2.npz (co-located with cf_trials in the run dir). Projection is the plain
inner product h . U_k (rows of U are the orthonormal basis vectors), matching how the subspace coordinates
are defined for the erase intervention.

Writes results/rq3_v2/<model>_<fmt>/subspace_proj.parquet with one row per cf trial:
  trial_id, belief_dir, kappa, x, topic, dimension, group, u2_0, u2_1, u4_0, u4_1, u4_2
All cf trials here are belief-stated (belief_dir in {-1,+1}); there are no no-belief rows.

GPU only (needs the model). Small: one forward pass over cf_trials, no generation, one captured layer.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from idea.pipeline import _backend, _cues, _topics, run_dir, load_config  # noqa: E402
from idea.runner import make_request  # noqa: E402
from idea.mech.rq3v2 import forward_states  # noqa: E402

# erase-answer layer per model (the layer whose answer-position subspace the corrected RQ3 erases)
ERASE_LAYER = {"olmo_instruct": 24, "llama": 16, "tulu_final": 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", default="chat")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rd = run_dir(cfg)
    layer = ERASE_LAYER.get(a.model, 24)
    out = rd / "results" / "rq3_v2" / f"{a.model}_{a.fmt}"
    cf = pd.read_parquet(out / "cf_trials.parquet")
    dirs = np.load(out / "directions_v2.npz")
    U2 = dirs[f"L{layer}/ans_U2"].astype(np.float64)        # [2, d]
    U4 = dirs[f"L{layer}/ans_U4"].astype(np.float64)        # [4, d]
    print(f"[subviz] {a.model}: {len(cf)} cf trials, erase layer {layer}, "
          f"U2 {U2.shape}, U4 {U4.shape}", flush=True)

    lib, topics = _cues(cfg), _topics(cfg)
    tmap = {t.topic_id: t for t in topics}
    backend = _backend(cfg, a.model)
    reqs = [make_request(r, tmap, lib, a.fmt, "post") for r in cf.itertuples(index=False)]
    _, last, _ = forward_states(backend, reqs, [layer], prefill=True)
    H = last[layer].astype(np.float64)                      # [n, d] answer-position hidden states

    p2 = H @ U2.T                                           # [n, 2] coords in the 2-D belief subspace
    p4 = H @ U4[:3].T                                       # [n, 3] first 3 coords of the 4-D subspace
    res = pd.DataFrame(dict(
        trial_id=cf.trial_id.to_numpy(),
        belief_dir=cf.belief_dir.to_numpy(),
        kappa=cf.kappa.to_numpy(),
        x=cf.x.to_numpy(),
        topic=cf.topic.to_numpy(),
        dimension=cf.dimension.to_numpy(),
        group=cf.group.to_numpy(),
        u2_0=p2[:, 0], u2_1=p2[:, 1],
        u4_0=p4[:, 0], u4_1=p4[:, 1], u4_2=p4[:, 2],
    ))
    res.to_parquet(out / "subspace_proj.parquet", index=False)
    # quick sanity: separation of belief_dir along each U2 axis (Cohen's d)
    bd = res.belief_dir.to_numpy()
    for name, col in [("u2_0", res.u2_0), ("u2_1", res.u2_1)]:
        pos, neg = col[bd > 0].to_numpy(), col[bd < 0].to_numpy()
        sp = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2) or 1.0
        print(f"[subviz]   {name}: mean(+)={pos.mean():+.3f} mean(-)={neg.mean():+.3f} d={(pos.mean()-neg.mean())/sp:+.3f}", flush=True)
    print(f"[subviz] wrote {out/'subspace_proj.parquet'}", flush=True)


if __name__ == "__main__":
    main()
