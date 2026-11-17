#!/usr/bin/env python
"""Layer-by-layer and erasure views of the belief subspace (RQ3 visualization add-on).

    .venv/bin/python scripts/capture_subspace_layers.py --config configs/main.yaml --model olmo_instruct

Part A (no model, CPU): the corrected RQ3 run already saved, in capture.npz, the hidden states of every
counterfactual trial in three versions -- believes H1 ("pos"), believes H0 ("neg"), no belief ("none") -- at
layers 4-24, at the answer position ("last") and averaged over the belief-sentence tokens ("span"), for the
1500 training-topic trials ("cf") and 300 held-out validation-topic trials ("held"). Part A projects them:
  answer position     -> the layer's own 2-D answer subspace ans_U2 (directions_v2.npz)
  belief-sentence span -> the layer's own 2-D span subspace span_U2 (pos/neg only; "none" has no belief sentence)
and writes subspace_layers.parquet (long: set, cond, idx, layer, pos, c0, c1).

Part B (GPU): erasure. For the held-out trials (pos/neg/none), run the model unerased and with the RQ3
erasure at the erase layer (ans_U4 coordinates set to the no-belief value ans_t4, answer position), and
capture the answer-position states at the layers AFTER the erase layer (the state at the erase layer itself
is captured before the patch; its post-erasure coordinates follow algebraically and are computed in
analysis/16). Downstream layers beyond the saved ones (> 24) have no saved basis, so the 2-D answer subspace is
rebuilt there from the training cf trials with the pipeline's exact recipe (orth of mean(pos-none),
mean(neg-none)); for downstream layers that do have a saved basis the rebuilt one is compared with it as a
check (principal-angle cosines). Writes subspace_erasure.parquet (set, cond, run, idx, layer, c0, c1),
erasure_answers.parquet (cond, run, idx, p) and bases_downstream.npz.

Run on the cluster (capture.npz is large and only there). --part A runs only Part A.
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

ERASE_LAYER = {"olmo_instruct": 24, "llama": 16, "tulu_final": 24}
CONDS = ("pos", "neg", "none")


def part_a(out: Path, D) -> None:
    C = np.load(out / "capture.npz")
    layers = sorted(int(k.split("/")[0][1:]) for k in D.files if k.endswith("/ans_U2"))
    rows = []
    for st in ("cf", "held"):
        for cond in CONDS:
            for l in layers:
                for pos, ukey in (("last", "ans_U2"), ("span", "span_U2")):
                    key = f"{st}_{cond}_{pos}_L{l}"
                    if key not in C.files or (pos == "span" and cond == "none"):
                        continue
                    H = C[key].astype(np.float64)
                    U = D[f"L{l}/{ukey}"].astype(np.float64)
                    P = H @ U.T
                    rows.append(pd.DataFrame(dict(set=st, cond=cond, idx=np.arange(len(H)), layer=l, pos=pos,
                                                  c0=P[:, 0], c1=P[:, 1])))
    res = pd.concat(rows, ignore_index=True)
    res.to_parquet(out / "subspace_layers.parquet", index=False)
    print(f"[layers] wrote {out/'subspace_layers.parquet'} ({len(res)} rows; layers {layers})", flush=True)


def part_b(cfg, model: str, fmt: str, out: Path, D) -> None:
    lib, topics = _cues(cfg), _topics(cfg)
    tmap = {t.topic_id: t for t in topics}
    backend = _backend(cfg, model)
    nl = len(backend.layers())
    E = ERASE_LAYER[model]
    down = sorted({l for l in range(E + 4, nl, 4)} | {nl - 1})
    print(f"[erase] {model}: {nl} layers, erase L{E}, downstream capture {down}", flush=True)

    def req(row, bdir):
        row = row._replace(belief_dir=int(bdir), b=int(bdir * row.h_dir))
        return make_request(row, tmap, lib, fmt, "post")

    def reqs_for(frame, cond):
        rs = list(frame.itertuples(index=False))
        return [req(r, r.h_dir if cond == "pos" else (-r.h_dir if cond == "neg" else 0)) for r in rs]

    cf = pd.read_parquet(out / "cf_trials.parquet")
    held = pd.read_parquet(out / "held_trials.parquet")

    # 1. downstream bases from the training cf trials (unerased), pipeline recipe
    S = {}
    for cond in CONDS:
        _, la, _ = forward_states(backend, reqs_for(cf, cond), down)
        S[cond] = {l: la[l].astype(np.float32) for l in down}
        print(f"[erase] cf {cond} captured", flush=True)
    B = {}
    for l in down:
        dp = (S["pos"][l] - S["none"][l]).mean(0); dn = (S["neg"][l] - S["none"][l]).mean(0)
        B[f"L{l}/ans_U2"] = _orth(np.stack([dp, dn]))
        if f"L{l}/ans_U2" in D.files:                      # check against the saved basis
            s = np.linalg.svd(B[f"L{l}/ans_U2"].astype(np.float64) @ D[f"L{l}/ans_U2"].astype(np.float64).T,
                              compute_uv=False)
            print(f"[erase] check L{l}: principal-angle cosines rebuilt vs saved = {np.round(s, 4)}", flush=True)
    np.savez(out / "bases_downstream.npz", **B)
    del S

    # 2. held-out trials, unerased and erased
    U4 = D[f"L{E}/ans_U4"].astype(np.float32); t4 = D[f"L{E}/ans_t4"].astype(np.float32)
    rows, ans = [], []
    for cond in CONDS:
        rq = reqs_for(held, cond)
        for run, patch in (("unerased", None),
                           ("erased", dict(layer=E, U=U4, targets=np.tile(t4, (len(rq), 1)), pos="last"))):
            p, la, _ = forward_states(backend, rq, down, patch=patch)
            ans.append(pd.DataFrame(dict(cond=cond, run=run, idx=np.arange(len(rq)), p=p)))
            for l in down:
                P = la[l].astype(np.float64) @ B[f"L{l}/ans_U2"].astype(np.float64).T
                rows.append(pd.DataFrame(dict(set="held", cond=cond, run=run, idx=np.arange(len(rq)), layer=l,
                                              c0=P[:, 0], c1=P[:, 1])))
            print(f"[erase] held {cond} {run}: mean p={np.nanmean(p):.3f}", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(out / "subspace_erasure.parquet", index=False)
    pd.concat(ans, ignore_index=True).to_parquet(out / "erasure_answers.parquet", index=False)
    print(f"[erase] wrote subspace_erasure.parquet, erasure_answers.parquet, bases_downstream.npz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fmt", default="chat")
    ap.add_argument("--part", default="AB", choices=["A", "B", "AB"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = run_dir(cfg) / "results" / "rq3_v2" / f"{a.model}_{a.fmt}"
    D = np.load(out / "directions_v2.npz")
    if "A" in a.part:
        part_a(out, D)
    if "B" in a.part:
        part_b(cfg, a.model, a.fmt, out, D)


if __name__ == "__main__":
    main()
