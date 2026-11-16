"""Step 6: RQ3 - activation steering and the capability audit (OLMo 2 7B Instruct, Llama 3.1 8B).

    python analysis/06_rq3_steering_audit.py

Steering needs the model weights and a GPU, so it ran on FarmShare (`idea steer`, `idea audit`);
its outputs are released in results/rq3/<model>_chat/. This script reads them and builds the
paper's tables. The steering vectors themselves are in steering_vectors/.

What was done (src/idea/pipeline.py: stage_steer; src/idea/mech/):
  1. Masking probe: a ridge probe trained on belief-neutral trials reads the Bayesian posterior
     logit (lp_before + x) from the last-token hidden state; on contradicted trials, masking index =
     median((probe read - stated) / (Bayes - stated)): about 1 = the hidden state is at the Bayesian
     answer while the stated answer is not (masking); about 0 = the hidden state is as distorted as
     the output; below 0 = further from Bayes than the stated answer.
  2. Directions (training topics only): d_bias = mean activation on contradicted trials where the
     model caved minus where it resisted, matched within strata; topic variance removed by
     projecting out the top k principal components of belief-neutral activations (hat_k4/8/16).
     Controls: random direction of equal norm, topic-only component, raw d_bias, a
     prompt-contrast direction, and per-dimension directions.
  3. Validation sweep (validation topics): layers {12,16,20,24} x strength multiples
     {0.5,1,2,4,8} x mode {all tokens, last token}; chosen = the setting that brings the
     contradicted-evidence gap closest to zero.
  4. One held-out test (test topics + Tier 1): preregistered success = contradicted slope moves
     toward the unsteered neutral slope, AND neutral and Tier 1 slopes change by < 0.05, in every
     identity dimension. Also the reversed direction (amplify) as a check that it is causal.
  5. Audit: MMLU-Pro (14 subjects, up to 500 items each), BBQ (500 items) and SycEval (300 MMLU
     items) unsteered, steered, and with the random direction.

Outputs (analysis/output/rq3/): steering_test.csv, chosen_settings.csv, sweep_best.csv,
masking.csv, bias_angles.csv, audit_summary.csv, audit_tests.csv
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT, save

sys.path.insert(0, str(ROOT / "src"))
from idea.audit import merge_parts  # noqa: E402  (sample-weighted merge of benchmark pieces)

RQ3 = ROOT / "results" / "rq3"
MODELS = {"olmo_instruct": "OLMo 2 7B Instruct", "llama": "Llama 3.1 8B Instruct"}


def steering_tables():
    tests, chosen, sweeps, masks, angles = [], [], [], [], []
    for key, label in MODELS.items():
        d = RQ3 / f"{key}_chat"
        if not (d / "steering_test.csv").exists():
            print(f"{key}: no steering results yet")
            continue
        t = pd.read_csv(d / "steering_test.csv")
        keep = [c for c in ["direction", "layer", "psi", "mode", "success", "t2_contra_gap_before",
                            "t2_contra_gap_after", "t2_contra_moved_toward_neutral", "t2_neutral_change",
                            "tier1_neutral_change", "tier1_contra_change", "tier1_support_change"] if c in t]
        t = t[keep].assign(model=key)
        # share of the contradicted-evidence gap removed (1 = gap closed, < 0 = gap widened)
        t["gap_closed_frac"] = 1 - t.t2_contra_gap_after / t.t2_contra_gap_before
        tests.append(t)
        s = json.loads((d / "summary.json").read_text())
        chosen.append(dict(model=key, **s))
        sw = pd.read_csv(d / "sweep_validation.csv")
        # best setting per layer on validation topics (smallest |gap after|)
        sw["abs_gap"] = sw.t2_contra_gap_after.abs()
        sweeps.append(sw.loc[sw.groupby("layer").abs_gap.idxmin()].assign(model=key))
        m = pd.read_csv(d / "masking_overall.csv").assign(dimension="overall")
        if (d / "masking_by_dimension.csv").exists():
            m = pd.concat([m, pd.read_csv(d / "masking_by_dimension.csv")])
        masks.append(m.assign(model=key))
        if (d / "bias_angles.csv").exists():
            angles.append(pd.read_csv(d / "bias_angles.csv").assign(model=key))
    if tests:
        save(pd.concat(tests, ignore_index=True), "rq3", "steering_test")
        save(pd.DataFrame(chosen), "rq3", "chosen_settings")
        save(pd.concat(sweeps, ignore_index=True), "rq3", "sweep_best", show=False)
        save(pd.concat(masks, ignore_index=True), "rq3", "masking")
        save(pd.concat(angles, ignore_index=True), "rq3", "bias_angles")


def two_prop_z(p1, n1, p2, n2):
    """Two-sample test of proportions (pooled SE); returns difference, SE, z, p."""
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    return p1 - p2, se, z, 2 * stats.norm.sf(abs(z))


def audit_tables():
    summ, tests = [], []
    for key in MODELS:
        pdir = RQ3 / f"{key}_chat" / "audit_parts"
        files = sorted(pdir.glob("*.csv")) if pdir.exists() else []
        if not files:
            continue
        merged = merge_parts(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
        conds = set(merged.condition)
        complete = merged.get("mmlu_pro_subjects_found", pd.Series(0)).eq(14).all() and \
            {"unsteered", "steered", "random"} <= conds
        cols = {"mmlu_pro/exact_match,custom-extract": "mmlu_pro_acc", "mmlu_pro/n": "mmlu_pro_n",
                "mmlu_pro_subjects_found": "mmlu_pro_subjects", "bbq/acc,none": "bbq_acc",
                "bbq/accuracy_amb,none": "bbq_acc_ambiguous", "bbq/accuracy_disamb,none": "bbq_acc_disambiguated",
                "bbq/amb_bias_score,none": "bbq_bias_ambiguous", "bbq/disamb_bias_score,none": "bbq_bias_disambiguated",
                "bbq/n": "bbq_n", "syceval_initial_accuracy": "syceval_initial_acc",
                "syceval_regressive_flip_rate": "syceval_regressive_flip", "syceval_progressive_flip_rate": "syceval_progressive_flip",
                "syceval_n": "syceval_n"}
        s = merged[["condition"] + [c for c in cols if c in merged]].rename(columns=cols)
        s.insert(0, "model", key)
        s["complete"] = complete
        summ.append(s)
        # steered and random vs unsteered: two-proportion tests on the accuracy measures.
        # MMLU-Pro is compared only on the subjects BOTH conditions have finished (subjects differ in
        # difficulty, so a partial set cannot be compared with the full one).
        if "unsteered" in conds:
            u = s[s.condition == "unsteered"].iloc[0]
            mu = merged[merged.condition == "unsteered"].iloc[0]
            for cond in [c for c in ("steered", "random") if c in conds]:
                r = s[s.condition == cond].iloc[0]
                mr = merged[merged.condition == cond].iloc[0]
                acc_col = "exact_match,custom-extract"
                subj = [k[:-2] for k in merged.columns if k.startswith("mmlu_pro_") and k.endswith("/n")
                        and pd.notna(mr.get(k)) and pd.notna(mu.get(k))]
                if subj:
                    n_r = sum(mr[f"{x}/n"] for x in subj)
                    n_u = sum(mu[f"{x}/n"] for x in subj)
                    a_r = sum(mr[f"{x}/{acc_col}"] * mr[f"{x}/n"] for x in subj) / n_r
                    a_u = sum(mu[f"{x}/{acc_col}"] * mu[f"{x}/n"] for x in subj) / n_u
                    diff, se, z, p = two_prop_z(a_r, n_r, a_u, n_u)
                    tests.append(dict(model=key, condition=cond, measure=f"mmlu_pro_acc ({len(subj)} shared subjects)",
                                      value=a_r, unsteered=a_u, diff=diff, se=se, z=z, p=p, n=n_r, complete=complete))
                for acc, n in (("bbq_acc", "bbq_n"), ("syceval_initial_acc", "syceval_n")):
                    if acc in s and pd.notna(r.get(acc)) and pd.notna(u.get(acc)):
                        diff, se, z, p = two_prop_z(r[acc], r[n], u[acc], u[n])
                        tests.append(dict(model=key, condition=cond, measure=acc, value=r[acc], unsteered=u[acc],
                                          diff=diff, se=se, z=z, p=p, n=r[n], complete=complete))
    if summ:
        save(pd.concat(summ, ignore_index=True), "rq3", "audit_summary")
    if tests:
        save(pd.DataFrame(tests), "rq3", "audit_tests")
    print("\nNote: SycEval reads one letter right after the user names a letter, so flip rates sit near 1 "
          "(see CODEBOOK/README limitations); BBQ with limit=500 covers the age category only.")


if __name__ == "__main__":
    steering_tables()
    audit_tables()
