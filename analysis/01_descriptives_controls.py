"""Step 1: sample description and instrument controls (working notes Section 4.5).

    python analysis/01_descriptives_controls.py [--study main|exploratory|pilot]

Tables (analysis/output/controls/<name>_<study>.csv):
  sample               trials, usable trials and exclusion rate per model x format x tier x cue type
  coverage             mean probability mass on valid integer answers (before / after), and how
                       often the API fallback re-prompt was needed
  retest               test-retest reliability: correlation between the two paraphrased BEFORE
                       questions (lp_before vs lp_before_rep), over unique prior prompts
  null_evidence        tables with no evidence (D = 0): the Bayesian update is exactly 0, so mean
                       |y| is the noise floor and b*y is drift toward the stated belief
  complement_sums      pilot only: P(H1) + P(H0) asked separately (should be about 1)
  prior_distribution   distribution of the BEFORE probability by model, tier and evidence type
  readout_calibration  the same trials read two ways on open models: full next-token
                       distribution (primary) vs 10 sampled answers; agreement and slopes
  evidence_slopes      raw descriptive slopes of y on x for no-belief / supported / contradicted
                       trials (no controls), a quick picture of the main effect
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import load_dataset, save


def logit(p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)   # same clipping as the pipeline
    return np.log(p / (1 - p))


def sample_table(df):
    g = df.groupby(["model", "fmt", "tier", "cue_type"])
    return g.agg(trials=("trial_id", "size"), usable=("excluded", lambda v: int((~v).sum())),
                 excluded_frac=("excluded", "mean")).reset_index()


def coverage(df):
    g = df.groupby(["model", "fmt", "tier", "cue_type"])
    return g.agg(n=("trial_id", "size"), excluded_frac=("excluded", "mean"), cov_prior=("cov_prior", "mean"),
                 cov_post=("cov_post", "mean"), reprompt_frac=("att_post", "mean")).reset_index()


def retest(df):
    # one row per distinct BEFORE prompt (the prior does not depend on the evidence table)
    # Gemini and Claude asked the paraphrase on a seeded subset only (retest == True); the other
    # models always have it, so this filter changes nothing for them.
    d = df[~df.excluded & df.lp_before_rep.notna()].drop_duplicates(["model", "fmt", "cue_id", "topic", "h_dir", "b", "source", "q",
                                          "treated_first"])
    rows = []
    for (m, f), g in d.groupby(["model", "fmt"]):
        rows.append(dict(model=m, fmt=f, r=g.lp_before.corr(g.lp_before_rep), n=len(g),
                         mean_abs_diff=(g.lp_before - g.lp_before_rep).abs().mean()))
    return pd.DataFrame(rows)


def null_evidence(df):
    d = df[(df.D == 0) & ~df.excluded].copy()
    d["e_null"] = np.where(d.b == 0, "no_belief", "belief")
    d["toward_belief"] = d.b * d.y
    return d.groupby(["model", "fmt", "tier", "e_null"]).agg(
        n=("y", "size"), mean_abs_y=("y", lambda v: v.abs().mean()), mean_y=("y", "mean"),
        drift_toward_belief=("toward_belief", "mean")).reset_index()


def complement_sums(df):
    if "sum_before" not in df or "complement" not in df:
        return pd.DataFrame()
    d = df[df.complement.astype(str).str.lower().eq("true") & df.sum_before.notna()]
    return d.groupby(["model", "fmt"])[["sum_before", "sum_after"]].describe().round(3).reset_index()


def prior_distribution(df):
    d = df[~df.excluded]
    return d.groupby(["model", "fmt", "tier", "e"]).p_prior.describe(percentiles=[0.05, 0.5, 0.95]).reset_index()


def readout_calibration(df):
    if "p_prior_s" not in df:
        return pd.DataFrame()
    d = df[df.calibration & df.p_prior_s.notna() & df.p_post_s.notna() & ~df.excluded].copy()
    d["lp_before_s"], d["lp_after_s"] = logit(d.p_prior_s), logit(d.p_post_s)
    rows = []
    for (m, f), g in d.groupby(["model", "fmt"]):
        rec = dict(model=m, fmt=f, n=len(g), r_prior=g.lp_before.corr(g.lp_before_s),
                   r_post=g.lp_after.corr(g.lp_after_s))
        for tag, a, b in (("dist", "lp_after", "lp_before"), ("sample", "lp_after_s", "lp_before_s")):
            fit = smf.ols(f"{a} ~ {b} + x + x:contra", g).fit()
            rec[f"beta_{tag}"], rec[f"delta_b_{tag}"] = fit.params["x"], fit.params.get("x:contra", np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def evidence_slopes(df):
    """Unadjusted: regress y on x separately for each evidence type (Tier 2, usable, non-null)."""
    d = df[~df.excluded & (df.e != "null") & (df.tier == 2)]
    rows = []
    for (m, f, e), g in d.groupby(["model", "fmt", "e"]):
        fit = smf.ols("y ~ x", g).fit()
        rows.append(dict(model=m, fmt=f, e=e, n=len(g), slope=fit.params["x"], se=fit.bse["x"],
                         mean_y=g.y.mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="main", choices=["main", "exploratory", "pilot"])
    a = ap.parse_args()
    df = load_dataset("pilot" if a.study == "pilot" else "main")
    df = df[df.study == a.study]
    tables = dict(sample=sample_table(df), coverage=coverage(df), retest=retest(df),
                  null_evidence=null_evidence(df), complement_sums=complement_sums(df),
                  prior_distribution=prior_distribution(df), readout_calibration=readout_calibration(df),
                  evidence_slopes=evidence_slopes(df))
    for name, tab in tables.items():
        if not tab.empty:
            save(tab, "controls", f"{name}_{a.study}", show=name in ("retest", "readout_calibration", "evidence_slopes"))


if __name__ == "__main__":
    main()
