#!/usr/bin/env python
"""Capture per-trial belief "dose" for the dose-response analysis (RQ3 add-on).

    .venv/bin/idea-py scripts/capture_doses.py --config configs/main.yaml --model olmo_instruct

For each counterfactual trial (cf_trials.parquet: the belief-stated and matched no-belief versions used by
the corrected RQ3), run a forward pass with the answer prefix, grab the answer-position hidden state at the
model's erase layer, and project it onto the empirical belief-content direction at that layer. The projection,
signed toward the stated belief, is the per-trial DOSE (how strongly the belief is represented on that trial).

Writes results/rq3_v2/<model>_<fmt>/doses.parquet with one row per cf trial:
  trial_id, b, belief_dir, kappa, x, dose_raw (h . u), dose (signed toward the stated belief = belief_dir*dose_raw).
The belief-content unit direction u is the normalized difference of mean answer-position hidden states between
belief-in-H1 and belief-in-H0 trials at the erase layer (computed from the captured states themselves, so it
needs no extra files). Behavioral response is computed later, on CPU, from cf_trials (analysis/13).

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
    print(f"[dose] {a.model}: {len(cf)} cf trials, erase layer {layer}", flush=True)

    lib, topics = _cues(cfg), _topics(cfg)
    tmap = {t.topic_id: t for t in topics}
    backend = _backend(cfg, a.model)
    reqs = [make_request(r, tmap, lib, a.fmt, "post") for r in cf.itertuples(index=False)]
    _, last, _ = forward_states(backend, reqs, [layer], prefill=True)
    H = last[layer].astype(np.float64)                       # [n, d] answer-position hidden states

    b = cf.b.to_numpy()                                      # stated belief about H1 claim direction: +1/-1/0
    bdir = cf.belief_dir.to_numpy()                          # b * h_dir: which claim the user asserts
    # belief-content axis: mean(belief toward the asserted claim) - mean(opposite), using stated-belief rows only
    pos = H[bdir > 0].mean(0) if (bdir > 0).any() else np.zeros(H.shape[1])
    neg = H[bdir < 0].mean(0) if (bdir < 0).any() else np.zeros(H.shape[1])
    u = pos - neg
    nrm = np.linalg.norm(u)
    u = u / nrm if nrm > 0 else u
    dose_raw = H @ u                                         # projection onto the belief axis
    dose = np.where(bdir != 0, np.sign(bdir) * dose_raw, dose_raw)   # signed toward the stated belief
    res = pd.DataFrame(dict(trial_id=cf.trial_id.to_numpy(), b=b, belief_dir=bdir,
                            kappa=cf.kappa.to_numpy(), x=cf.x.to_numpy(),
                            dose_raw=dose_raw, dose=dose))
    res.to_parquet(out / "doses.parquet", index=False)
    print(f"[dose] wrote {out/'doses.parquet'}  (|u|={nrm:.3f}; "
          f"mean dose belief-stated={dose[bdir!=0].mean():.3f}, no-belief={dose[bdir==0].mean():.3f})", flush=True)


if __name__ == "__main__":
    main()
