"""Shared helpers for the IDEA analysis scripts.

Every analysis script in this folder reads the released trial-level datasets
(`datasets/main_trials.csv.gz`, `datasets/pilot_trials.csv.gz`) and uses only pandas,
numpy, scipy and statsmodels, except where a script says otherwise. This file holds the
pieces they share:

* `load_dataset`   - read a released dataset with the right column types
* `analysis_frame` - the preregistered analysis sample (exclusions, tier, model)
* `fit_ols`        - the primary regression (Eq. 3) with topic-clustered standard errors
* `holm`           - Holm correction for the two co-primary tests
* `estimate`       - pull one coefficient (estimate, SE, p, 95% CI) out of a fit

The regression and sample definitions here are the same as in the `idea` package
(`src/idea/analysis/rq1.py`); `tests/test_release_analysis.py` checks that they give
identical numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # repository root
DATA = ROOT / "datasets"                            # released CSVs
OUT = ROOT / "analysis" / "output"                  # every script writes its tables here

UNSPECIFIED = "unspecified"   # the control "identity" (no persona); delta_j is relative to it

# Identity dimensions and their groups (22 groups + the Unspecified control); identical to
# idea.cues.DIMENSIONS (checked by tests/test_release_analysis.py).
DIMENSIONS = {
    "political": ["democrat", "republican", "independent"],
    "race": ["white", "black", "hispanic", "asian_american"],
    "sexuality": ["straight", "gay_lesbian", "bisexual"],
    "gender": ["man", "woman", "nonbinary"],
    "religion": ["evangelical", "catholic", "jewish", "muslim", "atheist"],
    "geography": ["rural", "urban"],
    "age": ["age_20s", "age_60s"],
}

# Primary regression (Eq. 3 of the working notes):
#   lp_after = alpha + rho*lp_before + beta*x + Delta_b*(x*contra) + s*(x*support)
#              + theta_cong*(x*contra*m) + (x*support*m) + controls
# Delta_b    = coefficient on x:contra   (how much flatter the evidence slope is when the
#                                         evidence contradicts the user's stated belief)
# theta_cong = coefficient on x:contra:m (how that discount changes with identity-belief
#                                         congruence m = b * g)
BASE_TERMS = ("lp_before + x + x:contra + x:support + x:contra:m + x:support:m"
              " + x:dpos + x:C(cue_type) + x:C(source)")

# Orderings used in tables.
OLMO7_STAGES = ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
OLMO13_STAGES = ["olmo13_base", "olmo13_sft", "olmo13_dpo", "olmo13_instruct"]
TULU_STAGES = ["tulu_base", "tulu_sft", "tulu_dpo", "tulu_final"]


def load_dataset(name: str = "main") -> pd.DataFrame:
    """Read datasets/<name>_trials.csv.gz (name = 'main' or 'pilot')."""
    path = DATA / f"{name}_trials.csv.gz"
    # Only empty cells are missing. pandas would otherwise read the evidence label e = "null"
    # (a table with no evidence, x = 0) as a missing value and let those trials into the analysis.
    df = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])
    assert (df.e == "null").any(), "evidence label 'null' was lost on reading"
    for c in ("excluded", "excluded_cov08", "tier1_low_g", "matched_pair", "calibration", "heuristic_conflict"):
        if c in df:
            df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df


def out_dir(sub: str) -> Path:
    p = OUT / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def analysis_frame(df: pd.DataFrame, tier: int | None = 2, model: str | None = None,
                   fmt: str | None = None, cue_type: str | None = None, m_col: str = "m",
                   excluded_col: str = "excluded") -> pd.DataFrame:
    """The analysis sample.

    1. drop excluded trials (unusable answers: coverage below the cutoff or a missing
       before/after answer; `excluded_col` lets robustness checks swap the rule),
    2. drop null-evidence tables (x = 0; they carry no evidence direction),
    3. keep one tier (2 = socially meaningful topics, the primary analysis; None = both),
    4. optionally one model / format / cue type,
    5. `m_col` picks the congruence measure: m (model-judged stance, primary), m_svy
       (Pew survey stance) or m_beh (behavioral stance) - robustness checks.
    """
    d = df[~df[excluded_col] & (df.e != "null")].copy()
    if tier is not None:
        d = d[d.tier == tier]
    if model is not None:
        d = d[d.model == model]
    if fmt is not None:
        d = d[d.fmt == fmt]
    if cue_type is not None:
        d = d[d.cue_type == cue_type]
    if m_col != "m":
        d["m"] = d[m_col].fillna(0.0)
    # Unspecified first, so every group effect is relative to the no-persona control
    d["group"] = pd.Categorical(d.group, categories=[UNSPECIFIED] + sorted(set(d.group) - {UNSPECIFIED}))
    d["d"] = d.d.astype(int)
    d["dpos"] = (d.d > 0).astype(int)   # separate slope for evidence favoring H1 vs H0
    return d.reset_index(drop=True)


def formula(terms: str, d: pd.DataFrame) -> str:
    """Drop control terms that are constant in this sample (e.g. labels-only models)."""
    f = f"lp_after ~ {terms}"
    if d.cue_type.nunique() < 2:
        f = f.replace(" + x:C(cue_type)", "")
    if d.source.nunique() < 2:
        f = f.replace(" + x:C(source)", "")
    return f


def fit_ols(d: pd.DataFrame, terms: str = BASE_TERMS):
    """OLS with standard errors clustered by topic (trials on one topic are not independent)."""
    groups = d.topic.astype("category").cat.codes
    return smf.ols(formula(terms, d), d).fit(cov_type="cluster", cov_kwds={"groups": groups})


def estimate(fit, name: str) -> dict:
    """est, se, p and 95% CI of one coefficient (NaN if the term is not in the model)."""
    if name not in fit.params:
        return dict(est=np.nan, se=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    est, se = float(fit.params[name]), float(fit.bse[name])
    return dict(est=est, se=se, p=float(fit.pvalues[name]), lo=est - 1.96 * se, hi=est + 1.96 * se)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: sort p-values, compare the i-th smallest with alpha/(m-i); stop at the
    first failure (later tests also fail)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m, ok = {}, len(items), True
    for i, (k, p) in enumerate(items):
        ok = ok and (p < alpha / (m - i))
        out[k] = bool(ok)
    return out


def model_formats(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(model, format) pairs present in the data, in a stable order."""
    return sorted(df.groupby(["model", "fmt"]).size().index.tolist())


def save(tab: pd.DataFrame, sub: str, name: str, show: bool = True) -> Path:
    p = out_dir(sub) / f"{name}.csv"
    tab.to_csv(p, index=False)
    if show:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(f"\n== {sub}/{name}.csv")
            print(tab.round(4).to_string(index=False))
    return p
