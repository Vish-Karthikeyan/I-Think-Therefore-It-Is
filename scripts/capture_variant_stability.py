#!/usr/bin/env python
"""Stability of the belief subspaces under prompt rewording (reviewer request, 2026-11-16; geometry only).

    .venv/bin/python scripts/capture_variant_stability.py --config configs/main.yaml --model olmo_instruct

Re-renders the corrected RQ3's counterfactual trials (cf: 1500 training-topic trials; held: 300 trials from unseen
Tier 2 topics), each as believes-H1 / believes-H0 / no-belief, under the original wording (variant 0, a self-check
that must reproduce the saved subspaces) and the two reworded templates of P6 (variants 1 and 2: reworded hypothesis
header and question; the belief sentence is unchanged). Captures answer-position states at layers 12-24, rebuilds
ans_U2 / ans_U4 with the pipeline's recipe for each wording from the cf trials, and compares them with the saved
original-wording subspaces (directions_v2.npz): principal-angle cosines and subspace overlap (mean cos^2). Geometric
retention: the share of each wording's held-out belief displacement (pos - none, neg - none) captured by the ORIGINAL
ans_U4, against the wording's own ans_U4.

Writes results/rq3_v2/<model>_chat/variant_stability.csv and variant_bases.npz. GPU; one forward pass per trial,
condition and wording, no generation.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from idea.pipeline import _backend, _cues, _topics, run_dir, load_config  # noqa: E402
from idea.runner import make_request  # noqa: E402
from idea.mech.rq3v2 import forward_states, _orth  # noqa: E402

LAYERS = [12, 16, 20, 24]
CONDS = ("pos", "neg", "none")


def build(S):
    """Pipeline recipe (mech/rq3v2.run): U2 = orth(mean(pos-none), mean(neg-none)); U4 = top-4 right singular vectors."""
    dp, dn = S["pos"] - S["none"], S["neg"] - S["none"]
    U2 = _orth(np.stack([dp.mean(0), dn.mean(0)])).astype(np.float64)
    _, _, Vt = np.linalg.svd(np.concatenate([dp, dn]), full_matrices=False)
    return U2, Vt[:4].astype(np.float64)


def pcos(A, B):
    return np.linalg.svd(A @ B.T, compute_uv=False)


def shares(U, S):
    """Share of the belief displacement captured by subspace U: of the mean shift, and of per-trial energy."""
    X = np.concatenate([S["pos"] - S["none"], S["neg"] - S["none"]])
    m = np.concatenate([(S["pos"] - S["none"]).mean(0, keepdims=True), (S["neg"] - S["none"]).mean(0, keepdims=True)])
    def sh(Y):
        return float(((Y @ U.T) ** 2).sum() / (Y ** 2).sum())
    return sh(m), sh(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", default="chat"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = run_dir(cfg) / "results" / "rq3_v2" / f"{a.model}_{a.fmt}"
    D = np.load(out / "directions_v2.npz")
    lib, topics = _cues(cfg), _topics(cfg)
    tmap = {t.topic_id: t for t in topics}
    frames = {"cf": pd.read_parquet(out / "cf_trials.parquet"), "held": pd.read_parquet(out / "held_trials.parquet")}
    if a.limit:
        frames = {k: v.head(a.limit) for k, v in frames.items()}
    backend = _backend(cfg, a.model)
    layers = [l for l in LAYERS if l < len(backend.layers()) and f"L{l}/ans_U2" in D.files]

    def reqs(frame, cond, variant):
        rows = []
        for r in frame.itertuples(index=False):
            bd = r.h_dir if cond == "pos" else (-r.h_dir if cond == "neg" else 0)
            rows.append(make_request(r._replace(belief_dir=int(bd), b=int(bd * r.h_dir)), tmap, lib, a.fmt, "post",
                                     variant=variant))
        return rows

    recs, bases = [], {}
    for v in (0, 1, 2):
        S = {st: {c: None for c in CONDS} for st in frames}
        for st, fr in frames.items():
            if v == 0 and st == "held":
                continue   # the self-check needs only the cf rebuild
            for c in CONDS:
                _, la, _ = forward_states(backend, reqs(fr, c, v), layers)
                S[st][c] = {l: la[l].astype(np.float64) for l in layers}
            print(f"[variant] {a.model} wording {v}: {st} captured", flush=True)
        for l in layers:
            Ucf = {c: S["cf"][c][l] for c in CONDS}
            U2v, U4v = build(Ucf)
            U2o, U4o = D[f"L{l}/ans_U2"].astype(np.float64), D[f"L{l}/ans_U4"].astype(np.float64)
            c2, c4 = pcos(U2v, U2o), pcos(U4v, U4o)
            rec = dict(model=a.model, wording=v, layer=l, n_cf=len(frames["cf"]),
                       U2_cos_min=float(c2.min()), U2_cos=";".join(f"{x:.4f}" for x in c2),
                       U4_cos=";".join(f"{x:.4f}" for x in c4), U4_overlap=float((c4 ** 2).mean()))
            if v > 0:
                H = {c: S["held"][c][l] for c in CONDS}
                rec["held_mean_share_origU4"], rec["held_trial_share_origU4"] = shares(U4o, H)
                rec["held_mean_share_ownU4"], rec["held_trial_share_ownU4"] = shares(U4v, H)
            recs.append(rec)
            bases[f"v{v}/L{l}/ans_U2"], bases[f"v{v}/L{l}/ans_U4"] = U2v.astype(np.float32), U4v.astype(np.float32)
            print(f"[variant] {a.model} wording {v} L{l}: U2 cos_min={c2.min():.4f} U4 overlap={(c4 ** 2).mean():.3f}"
                  + (f" | held mean shift in orig U4 {rec['held_mean_share_origU4']:.3f} (own {rec['held_mean_share_ownU4']:.3f})"
                     if v > 0 else ""), flush=True)
    pd.DataFrame(recs).to_csv(out / "variant_stability.csv", index=False)
    np.savez(out / "variant_bases.npz", **bases)
    print(f"[variant] wrote {out/'variant_stability.csv'}", flush=True)


if __name__ == "__main__":
    main()
