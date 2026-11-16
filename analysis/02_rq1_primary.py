"""Step 2: RQ1 primary analysis (preregistered).

    python analysis/02_rq1_primary.py [--study main|exploratory|pilot] [--engine ols|mixed]

For every model x prompt format, on Tier 2 trials (socially meaningful topics), fit Eq. 3:

    lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m
               + x:dpos + x:C(cue_type) + x:C(source)

  lp_before, lp_after  log-odds the model gives H1 before / after seeing the evidence table
  x                    exact log-likelihood ratio of the table for H1 (the evidence strength)
  contra / support     1 if the user's stated belief is contradicted / supported by the table
  m = b * g            identity-belief congruence: stated belief (b = +1/-1, 0 if none) times the
                       model's perceived stance of the user's group on H1 (g)

Co-primary estimands, Holm-corrected together:
  Delta_b    (x:contra)    evidence slope when the evidence contradicts the user's belief, minus
                           the slope with no stated belief; negative = contradicting evidence is
                           discounted
  theta_cong (x:contra:m)  change in that discount per unit of congruence; negative = the
                           discount is larger when the belief fits the user's identity

Also reported: beta (x, the no-belief evidence slope; 1 = Bayesian), rho (lp_before), and the
support shift (x:support). Standard errors are clustered by topic.

Outputs (analysis/output/rq1/):
  per_model_<study>.csv        one row per model x format
  joint_<study>_<fmt>.csv      do the models differ? (Wald test that all model interactions
                               on x:contra and x:contra:m are zero)
  stage_contrasts_<study>.csv  OLMo base -> SFT -> DPO -> Instruct in completion format (main)
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import (BASE_TERMS, OLMO7_STAGES, analysis_frame, estimate, fit_ols, formula, holm,
                    load_dataset, model_formats, save)


def fit_mixed(d: pd.DataFrame):
    """Mixed-model version (preregistered alternative engine): random intercepts per item and
    random slopes on x per topic. Slow on the full data (minutes per model)."""
    d = d.assign(_one=1)
    md = smf.mixedlm(formula(BASE_TERMS, d), d, groups="_one",
                     vc_formula={"item": "0 + C(item_id)", "topic_slope": "0 + C(topic):x"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return md.fit(method="lbfgs", reml=True)


def primary(d: pd.DataFrame, engine: str = "ols") -> dict:
    """Fit Eq. 3 on one analysis sample and return the estimands."""
    fit = fit_ols(d) if engine == "ols" else fit_mixed(d)
    db, th = estimate(fit, "x:contra"), estimate(fit, "x:contra:m")
    sig = holm({"delta_b": db["p"], "theta_cong": th["p"]})       # Holm over the two co-primaries
    return dict(n=int(fit.nobs), beta_neutral=estimate(fit, "x")["est"], rho=estimate(fit, "lp_before")["est"],
                delta_b=db["est"], delta_b_se=db["se"], delta_b_p=db["p"], delta_b_lo=db["lo"], delta_b_hi=db["hi"],
                theta_cong=th["est"], theta_se=th["se"], theta_p=th["p"], theta_lo=th["lo"], theta_hi=th["hi"],
                holm_delta_b=sig["delta_b"], holm_theta=sig["theta_cong"],
                support_shift=estimate(fit, "x:support")["est"], support_shift_se=estimate(fit, "x:support")["se"],
                support_shift_p=estimate(fit, "x:support")["p"])


def per_model(df: pd.DataFrame, engine: str = "ols", m_col: str = "m", excluded_col: str = "excluded",
              tier: int = 2) -> pd.DataFrame:
    rows = []
    for model, fmt in model_formats(df):
        d = analysis_frame(df, tier, model, fmt, m_col=m_col, excluded_col=excluded_col)
        if len(d) < 50:            # too few usable trials to fit (reported as missing)
            rows.append(dict(model=model, fmt=fmt, n=len(d), note="fewer than 50 usable trials"))
            continue
        rows.append(dict(model=model, fmt=fmt, congruence=m_col, **primary(d, engine)))
    return pd.DataFrame(rows)


def joint_test(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Stack all models of one format and test whether Delta_b and theta_cong differ by model
    (each model gets its own intercept, rho and beta; the Wald test is on the model
    interactions with x:contra and x:contra:m)."""
    d = analysis_frame(df, 2, fmt=fmt)
    if d.model.nunique() < 2:
        return pd.DataFrame()
    terms = BASE_TERMS + " + C(model) + lp_before:C(model) + x:C(model) + x:contra:C(model) + x:contra:m:C(model)"
    fit = fit_ols(d, terms)
    names = [n for n in fit.params.index if n.startswith(("x:contra:C(model)", "x:contra:m:C(model)"))]
    R = np.zeros((len(names), len(fit.params)))
    for i, n in enumerate(names):
        R[i, list(fit.params.index).index(n)] = 1
    out = pd.DataFrame([dict(term=n, est=fit.params[n], se=fit.bse[n]) for n in names])
    # A topic-clustered covariance has rank at most (number of topics - 1), so a Wald test of
    # more restrictions than that is undefined (statsmodels then returns a meaningless value,
    # e.g. a negative statistic). This happens for the exploratory set (32+ restrictions,
    # 17 topics); the main run has 8. Report it as not estimable instead.
    n_clusters = d.topic.nunique()
    if len(names) > n_clusters - 1:
        return out.assign(wald_stat=np.nan, wald_p=np.nan, n_restrictions=len(names),
                          note=f"not estimable: {len(names)} restrictions > {n_clusters} topic clusters - 1")
    w = fit.wald_test(R, scalar=True)
    return out.assign(wald_stat=float(w.statistic), wald_p=float(w.pvalue), n_restrictions=len(names))


def stage_contrasts(res: pd.DataFrame, stages: list[str], fmt: str) -> pd.DataFrame:
    """Order a training ladder (base -> SFT -> DPO -> final) and show the change at each step."""
    r = res[(res.fmt == fmt) & res.model.isin(stages)].copy()
    if r.empty:
        return r
    r["stage_order"] = r.model.map({m: i for i, m in enumerate(stages)})
    r = r.sort_values("stage_order")
    for col in ("beta_neutral", "delta_b", "theta_cong"):
        r[f"{col}_change_from_prev"] = r[col].diff()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="main", choices=["main", "exploratory", "pilot"])
    ap.add_argument("--engine", default="ols", choices=["ols", "mixed"])
    a = ap.parse_args()
    df = load_dataset("pilot" if a.study == "pilot" else "main")
    df = df[df.study == a.study]
    tag = a.study if a.engine == "ols" else f"{a.study}_mixed"

    res = per_model(df, a.engine)
    save(res, "rq1", f"per_model_{tag}")
    if a.engine == "ols":
        for fmt in sorted(df.fmt.unique()):
            j = joint_test(df, fmt)
            if not j.empty:
                save(j, "rq1", f"joint_{tag}_{fmt}")
        save(stage_contrasts(res, OLMO7_STAGES, "completion"), "rq1", f"stage_contrasts_{tag}_completion")
        save(stage_contrasts(res, OLMO7_STAGES, "chat"), "rq1", f"stage_contrasts_{tag}_chat")


if __name__ == "__main__":
    main()
