"""The RQ1 regressions that RQ3 needs (copied from the study's RQ1 code so that `idea.mech` is self-contained).

Primary regression, per model (Tier 2), OLS with standard errors clustered by topic:
    lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m
               + x:dpos + x:C(cue_type) + x:C(source)
* Delta_b    = coefficient on x:contra    (contradicted slope minus no-belief slope)
* theta_cong = coefficient on x:contra:m  (change in the contradicted slope per unit of perceived stance m)
* delta_j    = x:contra:C(group)[j]       (group j's drop minus Unspecified's drop)

RQ3 uses them for: the "is there anything to fix" gate (primary), the identity dimensions with a credible
theta (theta_by_dimension, for per-dimension directions), and the spread of delta_j within dimensions, one
part of the preregistered steering success criterion (steer.delta_j_spread).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..cues import DIMENSIONS, UNSPECIFIED

BASE_TERMS = "lp_before + x + x:contra + x:support + x:contra:m + x:support:m + x:dpos + x:C(cue_type) + x:C(source)"


def analysis_frame(df: pd.DataFrame, tier: int | None = 2, model: str | None = None) -> pd.DataFrame:
    """Usable trials (not excluded, evidence present) of one tier, with Unspecified as the reference group."""
    d = df[~df.excluded & (df.e != "null")].copy()
    if tier is not None:
        d = d[d.tier == tier]
    if model is not None:
        d = d[d.model == model]
    d["group"] = pd.Categorical(d.group, categories=[UNSPECIFIED] + sorted(set(d.group) - {UNSPECIFIED}))
    d["d"] = d.d.astype(int)
    d["dpos"] = (d.d > 0).astype(int)   # piecewise slope for evidence favoring H1
    return d.reset_index(drop=True)


def _formula(terms: str, d: pd.DataFrame) -> str:
    f = f"lp_after ~ {terms}"
    if d.cue_type.nunique() < 2:
        f = f.replace(" + x:C(cue_type)", "")
    if d.source.nunique() < 2:
        f = f.replace(" + x:C(source)", "")
    return f


def fit_ols(d: pd.DataFrame, terms: str = BASE_TERMS):
    groups = d.topic.astype("category").cat.codes
    return smf.ols(_formula(terms, d), d).fit(cov_type="cluster", cov_kwds={"groups": groups})


@dataclass
class Est:
    name: str
    est: float
    se: float
    p: float

    @property
    def ci(self):
        return self.est - 1.96 * self.se, self.est + 1.96 * self.se


def _get(fit, name: str) -> Est:
    if name not in fit.params:
        return Est(name, np.nan, np.nan, np.nan)
    return Est(name, float(fit.params[name]), float(fit.bse[name]), float(fit.pvalues[name]))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m, ok = {}, len(items), True
    for i, (k, p) in enumerate(items):
        ok = ok and (p < alpha / (m - i))
        out[k] = bool(ok)
    return out


def primary(d: pd.DataFrame) -> dict:
    fit = fit_ols(d)
    db, th = _get(fit, "x:contra"), _get(fit, "x:contra:m")
    return dict(fit=fit, delta_b=db, theta_cong=th, holm=holm({"delta_b": db.p, "theta_cong": th.p}), n=int(fit.nobs))


def theta_by_dimension(d: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension theta_cong (fixed-effect analogue of theta + tau_k)."""
    dd = d.copy()
    for k in DIMENSIONS:
        dd[f"xcm_{k}"] = dd.x * dd.contra * dd.m * (dd.dimension == k)
    terms = BASE_TERMS.replace(" + x:contra:m", "") + "".join(f" + xcm_{k}" for k in DIMENSIONS
                                                            if (dd[f"xcm_{k}"] != 0).any())
    fit = fit_ols(dd, terms)
    rows = [dict(dimension=k, **_get(fit, f"xcm_{k}").__dict__) for k in DIMENSIONS]
    return pd.DataFrame(rows).drop(columns="name")


def unequal_deference(d: pd.DataFrame) -> pd.DataFrame:
    """delta_j for every group relative to Unspecified (Eq. 6)."""
    terms = ("lp_before + x + x:C(group) + x:contra + x:contra:C(group) + x:support + x:support:C(group)"
             " + x:dpos + x:C(cue_type) + x:C(source)")
    fit = fit_ols(d, terms)
    rows = []
    for g in d.group.cat.categories:
        if g == UNSPECIFIED:
            continue
        e = _get(fit, f"x:contra:C(group)[T.{g}]")
        rows.append(dict(group=g, dimension=d.loc[d.group == g, "dimension"].iloc[0] if (d.group == g).any() else None,
                         delta=e.est, se=e.se, p=e.p))
    return pd.DataFrame(rows)
