"""Step 10: exploratory prediction P4 - does the belief/contradiction signal appear in OLMo's
activations at DPO? (Runs where the saved activations are: FarmShare, CPU only.)

    python analysis/10_p4_stage_probes.py [--layers 8 16 24] [--max-n 4000]

Logged prediction (2026-11-11, before the data existed): "the belief/contradiction signal in the
residual stream (probe accuracy, and cosine to the Instruct direction) is weak at SFT and appears
at DPO, mirroring P0."

Data: last-prompt-token residual-stream activations of the posterior prompt, saved during
elicitation, for OLMo 2 7B SFT and DPO (exploratory run, chat format, explicit labels) and OLMo 2 7B
Instruct (main run, chat; restricted to labels here so the three stages see the same kind of cue).
Layers 8-26. Tier 2 trials with a stated belief.

Two measures per stage and layer, both from the logged prediction:
  1. Probe accuracy. A logistic-regression probe reads, from the activations, whether the table
     contradicts the user's stated belief (contra = 1) or supports it (support = 0). Trained on
     70% of topics, tested on the held-out 30% (same topic split for every stage). Reported:
     held-out accuracy and AUC, against the majority-class rate.
  2. Cosine to the Instruct direction. The bias direction d_bias (mean activation where the model
     caved to the belief minus where it resisted, matched within strata, top-8 topic components
     removed: exactly the RQ3 construction, hat_k8) is built for each stage from its own trials
     and compared with OLMo Instruct's saved RQ3 direction at the same layer. SFT, DPO and Instruct
     are successive fine-tunes of one base model, so their residual streams share a basis and the
     cosine is interpretable; a random direction gives |cosine| of about 0.016 (1/sqrt(4096)).

Output: results/p4/p4_stage_probes.csv (one row per stage x layer) and a printed summary.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from idea.mech import probe, steer  # noqa: E402
from idea.mech.capture import ActStore  # noqa: E402
from idea.pipeline import load_responses, run_dir, load_config  # noqa: E402

STAGES = [("olmo_sft", "ext"), ("olmo_dpo", "ext"), ("olmo_instruct", "main")]


def stage_frame(model: str, cfg_name: str):
    cfg = load_config(ROOT / "configs" / f"{cfg_name}.yaml")
    rd = run_dir(cfg)
    df = load_responses(rd)
    d = df[(df.model == model) & (df.fmt == "chat") & df.key_post.notna() & (df.cue_type == "label")]
    return cfg, rd, d.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="*", type=int, default=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    ap.add_argument("--max-n", type=int, default=None, help="quick test: subsample trials")
    ap.add_argument("--out", default=str(ROOT / "results" / "p4" / "p4_stage_probes.csv"))
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    main_cfg = load_config(ROOT / "configs" / "main.yaml")
    inst_dirs = steer.load_directions(run_dir(main_cfg) / "results" / "rq3" / "olmo_instruct_chat" / "directions")
    seed = main_cfg.get("seed", 0)
    rows = []
    for model, cfg_name in STAGES:
        cfg, rd, d = stage_frame(model, cfg_name)
        if a.max_n and len(d) > a.max_n:
            d = d.sample(a.max_n, random_state=0).reset_index(drop=True)
        store = ActStore(rd / "acts" / f"{model}_chat")
        # same topic split as RQ3 (train / test), so all stages are compared on identical topics
        train_t, _, test_t = probe.split_topics(d.topic.unique(), 0.3, 0.0, seed + 1)
        sub = d[(d.tier == 2) & ~d.excluded & d.e.isin(["contra", "support"])].reset_index(drop=True)
        dtrain = d[d.topic.isin(train_t)].reset_index(drop=True)
        print(f"{model}: {len(d)} label trials with activations; probe set {len(sub)}")
        for l in a.layers:
            H = store.get(list(sub.key_post), l)
            ok = np.isfinite(H).all(1)
            y = (sub.e == "contra").astype(int).values
            tr = ok & sub.topic.isin(train_t).values
            te = ok & sub.topic.isin(test_t).values
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
            clf.fit(H[tr], y[tr])
            p = clf.predict_proba(H[te])[:, 1]
            acc = float(((p > 0.5).astype(int) == y[te]).mean())
            auc = float(roc_auc_score(y[te], p)) if len(set(y[te])) == 2 else np.nan
            maj = float(max(y[te].mean(), 1 - y[te].mean()))
            # stage's own bias direction (RQ3 construction) vs Instruct's saved direction
            Htr = store.get(list(dtrain.key_post), l)
            dirs = steer.build_directions(dtrain, lambda _l, H=Htr: H, [l], ks=(8,), seed=seed)
            cos_hat = cos_raw = np.nan
            if l in dirs and l in inst_dirs:
                def cos(u, v):
                    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                cos_hat = cos(dirs[l]["hat_k8"], inst_dirs[l]["hat_k8"])
                cos_raw = cos(dirs[l]["raw"], inst_dirs[l]["raw"])
            rows.append(dict(stage=model, layer=l, n_train=int(tr.sum()), n_test=int(te.sum()), probe_acc=acc,
                             probe_auc=auc, majority=maj, cos_hat_k8_to_instruct=cos_hat, cos_raw_to_instruct=cos_raw))
            print(f"  layer {l}: acc {acc:.3f} (majority {maj:.3f}), AUC {auc:.3f}, cos(hat_k8) {cos_hat:.3f}, cos(raw) {cos_raw:.3f}")
        store._cache.clear()
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.groupby("stage")[["probe_acc", "probe_auc", "cos_hat_k8_to_instruct"]].agg(["mean", "max"]).round(3))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
