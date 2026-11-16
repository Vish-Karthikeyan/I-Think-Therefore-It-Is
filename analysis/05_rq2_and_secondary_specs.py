"""Step 5: RQ2 (prediction), RQ2b (behavioral signatures) and the secondary model specifications.

    python analysis/05_rq2_and_secondary_specs.py [--study main] [--models gpt llama ...]
                                                  [--max-folds N] [--n-boot 1000]

These analyses use the `idea` package (pip install -e . from the repository root), because the
RQ2 predictor (a low-rank model fitted with PyTorch) and the RQ2b bootstrap are long; the code
called here is exactly the code the study ran. The script feeds it the released dataset.

RQ2 (src/idea/analysis/rq2.py). Can identity information predict how much a group's evidence
slope shifts, for topics or identity dimensions the predictor never saw?
  * Targets: per-cell slope shifts relative to the no-belief slope, one unpooled regression per
    cell (cell = group x cue type x source x belief x evidence direction).
  * Predictors: the model's perceived stance g, survey gaps v_k, source, belief, evidence
    direction; a rank-2 interaction term W (the "full" model) vs no W ("no_W") vs cell means.
  * Two splits: 4 folds of held-out topics; 7 folds holding out one identity dimension.
  * Success (preregistered): on a split, the full model has lower RMSE than both baselines AND
    higher Spearman correlation than both.
RQ2b (src/idea/analysis/rq2b.py), exploratory and gated: run only for models whose RQ2
succeeded on the held-out-topic split.
Secondary specifications (src/idea/analysis/secondary.py), per model x format:
  * spline: is the evidence slope nonlinear in x? (F test of a 4-df spline vs linear)
  * mixture: two-component EM, "updated" vs "ignored the evidence"; who ignores it
  * latent_prior: instrumental-variable version of Eq. 3 that treats lp_before as a noisy
    measure of the true prior (the paraphrased prior lp_before_rep is the instrument)

Outputs: analysis/output/rq2/, analysis/output/secondary/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common import ROOT, load_dataset, model_formats, out_dir, save

sys.path.insert(0, str(ROOT / "src"))
from idea.analysis import rq2, rq2b, secondary  # noqa: E402


N_STARTS = 10


def mixture_multistart(df: pd.DataFrame, model: str, fmt: str):
    """The two-component mixture (updated vs ignored the evidence) is fitted by EM from random
    starting weights, and the likelihood has several local maxima: when the evidence slope is near
    0, the two components are hard to tell apart, and one start can label them the other way round
    from another. So: fit from N_STARTS seeds, keep the start with the highest log-likelihood, and
    report every start (if starts with near-equal likelihood disagree, the split is not identified
    and should not be interpreted). The original pipeline used a single start (seed 0)."""
    from idea.analysis.rq1 import analysis_frame as pkg_frame
    d = pkg_frame(df[df.fmt == fmt], 2, model)
    nz = d[d.x.abs() > 0.05].reset_index(drop=True)
    fits = [secondary.mixture_em(nz, seed=s) for s in range(N_STARTS)]
    starts = pd.DataFrame([dict(seed=s, loglik=f["loglik"], iterations=f["iterations"],
                                mean_p_ignore=f["mean_p_ignore"], **{f"updated_{k}": v for k, v in f["updated"].items()})
                           for s, f in enumerate(fits)])
    best = fits[int(starts.loglik.idxmax())]
    near = starts[starts.loglik >= starts.loglik.max() - 1.0]     # starts within 1 log-lik unit of the best
    stable = bool(near.mean_p_ignore.max() - near.mean_p_ignore.min() < 0.05)
    return best, starts, stable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="main", choices=["main", "exploratory", "pilot"])
    ap.add_argument("--models", nargs="*", help="limit to these model keys")
    ap.add_argument("--max-folds", type=int, default=None, help="quick check: first N folds per split")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--skip-secondary", action="store_true")
    a = ap.parse_args()
    df = load_dataset("pilot" if a.study == "pilot" else "main")
    df = df[df.study == a.study]
    pairs = [(m, f) for m, f in model_formats(df) if not a.models or m in a.models]

    rq2_rows = []
    for model, fmt in pairs:
        tag = f"{model}_{fmt}_{a.study}"
        if not a.skip_secondary:
            # ---- secondary specifications (every model x format)
            try:
                sec = secondary.run_secondary(df, model, fmt)      # spline + latent prior (+ 1-start mixture)
                mix, starts, stable = mixture_multistart(df, model, fmt)
                od = out_dir("secondary")
                starts.to_csv(od / f"{tag}_mixture_starts.csv", index=False)
                rows = [dict(spec="spline", f_stat=sec.get("spline", {}).get("f_stat"),
                             f_p=sec.get("spline", {}).get("f_p"))]
                rows.append(dict(spec="mixture", **{f"updated_{k}": v for k, v in mix["updated"].items()},
                                 mean_p_ignore=mix["mean_p_ignore"], loglik=mix["loglik"],
                                 n_starts=N_STARTS, stable_across_starts=stable))
                pd.DataFrame(rows).to_csv(od / f"{tag}_summary.csv", index=False)
                mix["p_ignore_coefs"].to_csv(od / f"{tag}_mixture_p_ignore.csv")
                lp = sec["latent_prior"]
                pd.DataFrame(dict(est=lp["params"], se=lp["se"])).to_csv(od / f"{tag}_latent_prior.csv")
                if "spline" in sec:
                    sec["spline"]["curves"].to_csv(od / f"{tag}_spline_curves.csv", index=False)
                print(f"secondary specs done: {tag}")
            except Exception as e:  # noqa: BLE001 - report and continue with the other models
                print(f"secondary specs failed for {tag}: {e!r}")
        if fmt != "chat":          # RQ2 is preregistered for chat format only
            continue
        # ---- RQ2
        r2 = rq2.evaluate(df, model, fmt, max_folds=a.max_folds)
        if r2["folds"].empty:
            print(f"RQ2 {tag}: {r2.get('note')}")
            continue
        save(r2["folds"], "rq2", f"{tag}_folds", show=False)
        save(r2["summary"], "rq2", f"{tag}_summary")
        rq2_rows += [dict(model=model, fmt=fmt, split=k, success=v) for k, v in r2["success"].items()]
        # ---- RQ2b (gated on RQ2's held-out-topic success)
        b = rq2b.run(df, model, fmt, r2["success"], n_boot=a.n_boot)
        if b.get("gated_out") or b.get("empty"):
            print(f"RQ2b {tag}: not run ({'gated out: RQ2 did not beat both baselines on held-out topics' if b.get('gated_out') else b.get('note')})")
            continue
        od = out_dir("rq2b")
        b["contrasts"].to_csv(od / f"{tag}_contrasts.csv", index=False)
        b["table"].to_csv(od / f"{tag}_table.csv")
        b["cv"].to_csv(od / f"{tag}_cv_rank.csv", index=False)
    if rq2_rows:
        save(pd.DataFrame(rq2_rows), "rq2", f"success_{a.study}")


if __name__ == "__main__":
    main()
