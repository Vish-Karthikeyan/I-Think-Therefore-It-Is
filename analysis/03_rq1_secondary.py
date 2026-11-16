"""Step 3: RQ1 secondary analyses (preregistered as secondary).

    python analysis/03_rq1_secondary.py [--study main|exploratory|pilot]

For every model x format with at least 200 usable Tier 2 trials:

  delta_j            unequal deference (Eq. 6): for each identity group j, how much more (or less)
                     group j's contradicted-evidence slope drops than the Unspecified control's.
                     Fitted in one regression with group interactions on x, x:contra, x:support.
  delta_j_equality   Wald test, within each identity dimension, that all its groups' delta_j are
                     equal (e.g. Democrat = Republican = Independent).
  theta_by_dimension theta_cong estimated separately for each identity dimension.
  cue_type           Delta_b and theta_cong for explicit labels vs vignettes, and their difference.
  tier               Delta_b on Tier 1 (topics without social meaning) and the Tier 2 - Tier 1
                     difference ("social amplification").
  delta_j_tier1      delta_j on Tier 1 topics.
  simple_difference  a check that forces rho = 1: regress y = lp_after - lp_before on x terms.
  prior_shift        gamma_c: how much the model's BEFORE answer moves with the user's identity
                     and stated belief, relative to Unspecified users with no stated belief.

Outputs: analysis/output/rq1_secondary/<name>_<study>.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import (BASE_TERMS, DIMENSIONS, UNSPECIFIED, analysis_frame, estimate, fit_ols, load_dataset,
                    model_formats, save)

PRIMARY = __import__("02_rq1_primary")


def unequal_deference(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta_j = coefficient on x:contra:C(group)[j] (Unspecified is the reference level)."""
    terms = ("lp_before + x + x:C(group) + x:contra + x:contra:C(group) + x:support + x:support:C(group)"
             " + x:dpos + x:C(cue_type) + x:C(source)")
    fit = fit_ols(d, terms)
    rows = []
    for g in d.group.cat.categories:
        if g == UNSPECIFIED:
            continue
        e = estimate(fit, f"x:contra:C(group)[T.{g}]")
        dim = d.loc[d.group == g, "dimension"].iloc[0] if (d.group == g).any() else None
        rows.append(dict(group=g, dimension=dim, delta=e["est"], se=e["se"], p=e["p"], lo=e["lo"], hi=e["hi"]))
    # Equality within each dimension: H0 delta_g1 = delta_g2 = ... (successive differences = 0)
    tests = []
    names_all = list(fit.params.index)
    for k, gs in DIMENSIONS.items():
        names = [f"x:contra:C(group)[T.{g}]" for g in gs if f"x:contra:C(group)[T.{g}]" in fit.params]
        if len(names) < 2:
            continue
        R = np.zeros((len(names) - 1, len(fit.params)))
        idx = [names_all.index(n) for n in names]
        for i in range(len(names) - 1):
            R[i, idx[i]], R[i, idx[i + 1]] = 1, -1
        w = fit.wald_test(R, scalar=True)
        tests.append(dict(dimension=k, stat=float(w.statistic), df=len(names) - 1, p=float(w.pvalue)))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def theta_by_dimension(d: pd.DataFrame) -> pd.DataFrame:
    """Replace the single x:contra:m term by one term per dimension: x*contra*m*[dimension = k]."""
    dd = d.copy()
    for k in DIMENSIONS:
        dd[f"xcm_{k}"] = dd.x * dd.contra * dd.m * (dd.dimension == k)
    terms = BASE_TERMS.replace(" + x:contra:m", "") + "".join(
        f" + xcm_{k}" for k in DIMENSIONS if (dd[f"xcm_{k}"] != 0).any())
    fit = fit_ols(dd, terms)
    return pd.DataFrame([dict(dimension=k, **estimate(fit, f"xcm_{k}")) for k in DIMENSIONS])


def contrasts(df: pd.DataFrame, model: str, fmt: str) -> dict[str, pd.DataFrame]:
    out = {}
    # (a) cue type: separate fits for labels and vignettes, then the difference in one model
    rows = []
    for ct in ("label", "vignette"):
        d = analysis_frame(df, 2, model, fmt, cue_type=ct)
        if len(d) > 50:
            r = PRIMARY.primary(d)
            rows.append(dict(cue_type=ct, delta_b=r["delta_b"], delta_b_se=r["delta_b_se"],
                             theta=r["theta_cong"], theta_se=r["theta_se"]))
    d2 = analysis_frame(df, 2, model, fmt)
    if d2.cue_type.nunique() == 2:
        f = fit_ols(d2, BASE_TERMS + " + x:contra:C(cue_type) + x:contra:m:C(cue_type)")
        rows.append(dict(cue_type="vignette_minus_label",
                         delta_b=f.params.get("x:contra:C(cue_type)[T.vignette]", np.nan),
                         delta_b_se=f.bse.get("x:contra:C(cue_type)[T.vignette]", np.nan),
                         theta=f.params.get("x:contra:m:C(cue_type)[T.vignette]", np.nan),
                         theta_se=f.bse.get("x:contra:m:C(cue_type)[T.vignette]", np.nan)))
    out["cue_type"] = pd.DataFrame(rows)
    # (b) tier: does the belief discount grow on socially meaningful topics?
    dall = analysis_frame(df, None, model, fmt)
    if dall.tier.nunique() == 2:
        dall["tier2"] = (dall.tier == 2).astype(int)
        f = fit_ols(dall, "lp_before + x + x:tier2 + x:contra + x:contra:tier2 + x:support + x:support:tier2"
                          " + x:dpos + x:C(source)")
        t1, amp = estimate(f, "x:contra"), estimate(f, "x:contra:tier2")
        out["tier"] = pd.DataFrame([dict(term="delta_b_tier1", **t1),
                                    dict(term="amplification_tier2_minus_tier1", **amp)])
        d1 = analysis_frame(df, 1, model, fmt)
        if len(d1) > 200:
            out["delta_j_tier1"], _ = unequal_deference(d1)
    # (c) simple after-minus-before regression (rho forced to 1), as a check
    f = smf.ols("y ~ x + x:contra + x:support + x:contra:m", d2).fit(
        cov_type="cluster", cov_kwds={"groups": d2.topic.astype("category").cat.codes})
    out["simple_difference"] = pd.DataFrame([dict(term=k, est=f.params[k], se=f.bse[k]) for k in f.params.index])
    return out


def prior_shift(df: pd.DataFrame) -> pd.DataFrame:
    """gamma = mean BEFORE log-odds for (group, cue type, belief b) minus the mean for
    Unspecified users with no belief, within model, format, tier and cue type."""
    d = df[~df.excluded]
    keys = ["model", "fmt", "tier", "cue_type"]
    ref = d[(d.group == UNSPECIFIED) & (d.b == 0)].groupby(keys).lp_before.mean().rename("ref")
    g = d.groupby(keys + ["group", "b"]).lp_before.agg(["mean", "sem", "size"]).reset_index()
    g = g.join(ref, on=keys)
    g["gamma"] = g["mean"] - g["ref"]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="main", choices=["main", "exploratory", "pilot"])
    a = ap.parse_args()
    df = load_dataset("pilot" if a.study == "pilot" else "main")
    df = df[df.study == a.study]
    out: dict[str, list] = {}
    for model, fmt in model_formats(df):
        d = analysis_frame(df, 2, model, fmt)
        if len(d) < 200:
            continue
        dj, t = unequal_deference(d)
        out.setdefault("delta_j", []).append(dj.assign(model=model, fmt=fmt))
        out.setdefault("delta_j_equality", []).append(t.assign(model=model, fmt=fmt))
        out.setdefault("theta_by_dimension", []).append(theta_by_dimension(d).assign(model=model, fmt=fmt))
        for name, tab in contrasts(df, model, fmt).items():
            out.setdefault(name, []).append(tab.assign(model=model, fmt=fmt))
    for name, tabs in out.items():
        save(pd.concat(tabs, ignore_index=True), "rq1_secondary", f"{name}_{a.study}", show=name != "delta_j")
    save(prior_shift(df), "rq1_secondary", f"prior_shift_{a.study}", show=False)


if __name__ == "__main__":
    main()
