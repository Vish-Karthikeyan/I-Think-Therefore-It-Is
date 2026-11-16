"""RQ1 estimands (Section 4.7, Eqs. 3-6).

Primary regression, per model (Tier 2):
    lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m
               + x:dpos + x:C(cue_type) + x:C(source)  [+ (1|item) + (0+x|topic)]
* Delta_b      = coefficient on x:contra        (contradicted slope - no-belief slope)
* theta_cong   = coefficient on x:contra:m      (change in contradicted slope per unit m)
* delta_j      = x:contra:C(group)[j]           (group j's drop minus Unspecified's drop)
Co-primary tests are Holm-corrected. Three fitting engines are provided:
`ols` (cluster-robust by topic; fast pipeline check), `mixed` (statsmodels MixedLM with
item intercepts and topic slopes), and `bambi` (full Bayesian model incl. tau_k).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from ..cues import DIMENSIONS, UNSPECIFIED

BASE_TERMS = "lp_before + x + x:contra + x:support + x:contra:m + x:support:m + x:dpos + x:C(cue_type) + x:C(source)"


def analysis_frame(df: pd.DataFrame, tier: int | None = 2, model: str | None = None,
                   cue_type: str | None = None, m_col: str = "m") -> pd.DataFrame:
    d = df[~df.excluded & (df.e != "null")].copy()
    if tier is not None:
        d = d[d.tier == tier]
    if model is not None:
        d = d[d.model == model]
    if cue_type is not None:
        d = d[d.cue_type == cue_type]
    if m_col != "m":
        d["m"] = d[m_col].fillna(0.0)
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


def fit_mixed(d: pd.DataFrame, terms: str = BASE_TERMS):
    """Random item intercepts and random topic slopes on x (crossed, via variance
    components inside a single group)."""
    d = d.assign(_one=1)
    md = smf.mixedlm(_formula(terms, d), d, groups="_one",
                     vc_formula={"item": "0 + C(item_id)", "topic_slope": "0 + C(topic):x"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return md.fit(method="lbfgs", reml=True)


def fit_bambi(d: pd.DataFrame, draws: int = 1000, chains: int = 4, **kw):
    """Full hierarchical model: (1|item) + (0+x|topic) + (0+x:contra:m|dimension)."""
    import bambi as bmb

    d = d.copy()
    d["xcm"] = d.x * d.contra * d.m
    f = (_formula(BASE_TERMS, d) + " + (1|item_id) + (0 + x|topic) + (0 + xcm|dimension)")
    model = bmb.Model(f, d)
    return model, model.fit(draws=draws, chains=chains, **kw)


def brms_script(csv_path: str, out_rds: str = "rq1_fit.rds") -> str:
    """R/brms equivalent, for users who prefer R."""
    return f"""library(brms)
d <- read.csv("{csv_path}")
d$xcm <- d$x * d$contra * d$m
fit <- brm(lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m +
             x:factor(d) + x:cue_type + x:source +
             (1 | item_id) + (0 + x | topic) + (0 + xcm | dimension),
           data = d, chains = 4, cores = 4, iter = 2000, seed = 1)
saveRDS(fit, "{out_rds}")
print(summary(fit))
print(ranef(fit)$dimension)  # tau_k: per-dimension deviations of theta_cong
"""


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
    params, bse, pv = fit.params, fit.bse, fit.pvalues
    if name not in params:
        return Est(name, np.nan, np.nan, np.nan)
    return Est(name, float(params[name]), float(bse[name]), float(pv[name]))


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, m, ok = {}, len(items), True
    for i, (k, p) in enumerate(items):
        ok = ok and (p < alpha / (m - i))
        out[k] = bool(ok)
    return out


def primary(d: pd.DataFrame, engine: str = "ols") -> dict:
    fit = {"ols": fit_ols, "mixed": fit_mixed}[engine](d)
    db, th = _get(fit, "x:contra"), _get(fit, "x:contra:m")
    sig = holm({"delta_b": db.p, "theta_cong": th.p})
    return dict(fit=fit, delta_b=db, theta_cong=th, holm=sig,
                beta_neutral=_get(fit, "x"), support_shift=_get(fit, "x:support"),
                support_m=_get(fit, "x:support:m"), rho=_get(fit, "lp_before"),
                alpha=_get(fit, "Intercept"), n=int(fit.nobs))


def theta_by_dimension(d: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension theta_cong (fixed-effect analogue of theta + tau_k; secondary)."""
    dd = d.copy()
    for k in DIMENSIONS:
        dd[f"xcm_{k}"] = dd.x * dd.contra * dd.m * (dd.dimension == k)
    terms = BASE_TERMS.replace(" + x:contra:m", "") + "".join(f" + xcm_{k}" for k in DIMENSIONS
                                                            if (dd[f"xcm_{k}"] != 0).any())
    fit = fit_ols(dd, terms)
    rows = [dict(dimension=k, **_get(fit, f"xcm_{k}").__dict__) for k in DIMENSIONS]
    return pd.DataFrame(rows).drop(columns="name")


def unequal_deference(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta_j for every group relative to Unspecified (Eq. 6), plus a Wald test of
    equality within each dimension."""
    terms = ("lp_before + x + x:C(group) + x:contra + x:contra:C(group) + x:support + x:support:C(group)"
             " + x:dpos + x:C(cue_type) + x:C(source)")
    fit = fit_ols(d, terms)
    rows = []
    for g in d.group.cat.categories:
        if g == UNSPECIFIED:
            continue
        e = _get(fit, f"x:contra:C(group)[T.{g}]")
        lo, hi = e.ci
        rows.append(dict(group=g, dimension=d.loc[d.group == g, "dimension"].iloc[0] if (d.group == g).any() else None,
                         delta=e.est, se=e.se, p=e.p, lo=lo, hi=hi))
    deltas = pd.DataFrame(rows)
    tests = []
    for k, gs in DIMENSIONS.items():
        names = [f"x:contra:C(group)[T.{g}]" for g in gs if f"x:contra:C(group)[T.{g}]" in fit.params]
        if len(names) < 2:
            continue
        R = np.zeros((len(names) - 1, len(fit.params)))
        idx = [list(fit.params.index).index(n) for n in names]
        for i in range(len(names) - 1):
            R[i, idx[i]], R[i, idx[i + 1]] = 1, -1
        w = fit.wald_test(R, scalar=True)
        tests.append(dict(dimension=k, stat=float(w.statistic), df=len(names) - 1, p=float(w.pvalue)))
    return deltas, pd.DataFrame(tests)


def prior_shift(df: pd.DataFrame) -> pd.DataFrame:
    """gamma_c = E[l | c] - E[l | Unspecified, b=0], by model, tier, group, cue type, b."""
    d = df[~df.excluded]
    ref = d[(d.group == UNSPECIFIED) & (d.b == 0)].groupby(["model", "tier", "cue_type"]).lp_before.mean().rename("ref")
    g = d.groupby(["model", "tier", "group", "cue_type", "b"]).lp_before.agg(["mean", "sem", "size"]).reset_index()
    g = g.join(ref, on=["model", "tier", "cue_type"])
    g["gamma"] = g["mean"] - g["ref"]
    return g


def rho_corrected(rho: float, r_retest: float) -> float:
    return rho / r_retest if r_retest and r_retest > 0 else np.nan


def contrasts(df: pd.DataFrame, model: str) -> dict[str, pd.DataFrame]:
    """Secondary contrasts: cue type, tier (social amplification), simple-difference check."""
    out = {}
    rows = []
    for ct in ("label", "vignette"):
        d = analysis_frame(df, 2, model, ct)
        if len(d) > 50:
            r = primary(d)
            rows.append(dict(cue_type=ct, delta_b=r["delta_b"].est, delta_b_se=r["delta_b"].se,
                             theta=r["theta_cong"].est, theta_se=r["theta_cong"].se))
    d2 = analysis_frame(df, 2, model)
    if d2.cue_type.nunique() == 2:
        f = fit_ols(d2, BASE_TERMS + " + x:contra:C(cue_type) + x:contra:m:C(cue_type)")
        rows.append(dict(cue_type="vignette_minus_label",
                         delta_b=f.params.get("x:contra:C(cue_type)[T.vignette]", np.nan),
                         delta_b_se=f.bse.get("x:contra:C(cue_type)[T.vignette]", np.nan),
                         theta=f.params.get("x:contra:m:C(cue_type)[T.vignette]", np.nan),
                         theta_se=f.bse.get("x:contra:m:C(cue_type)[T.vignette]", np.nan)))
    out["cue_type"] = pd.DataFrame(rows)

    dall = analysis_frame(df, None, model)
    if dall.tier.nunique() == 2:
        dall["tier2"] = (dall.tier == 2).astype(int)
        f = fit_ols(dall, "lp_before + x + x:tier2 + x:contra + x:contra:tier2 + x:support + x:support:tier2"
                          " + x:dpos + x:C(source)")
        t1 = _get(f, "x:contra")
        amp = _get(f, "x:contra:tier2")
        out["tier"] = pd.DataFrame([dict(term="delta_b_tier1", est=t1.est, se=t1.se, p=t1.p),
                                    dict(term="amplification_tier2_minus_tier1", est=amp.est, se=amp.se, p=amp.p)])
        # delta_j in Tier 1 (deference to the person on topics without social meaning)
        d1 = analysis_frame(df, 1, model)
        if len(d1) > 200:
            out["delta_j_tier1"], _ = unequal_deference(d1)
    # simple after-minus-before analysis (forces rho = 1), reported as a check
    f = smf.ols("y ~ x + x:contra + x:support + x:contra:m", d2).fit(
        cov_type="cluster", cov_kwds={"groups": d2.topic.astype("category").cat.codes})
    out["simple_difference"] = pd.DataFrame([dict(term=k, est=f.params[k], se=f.bse[k]) for k in f.params.index])
    return out


def per_model(df: pd.DataFrame, engine: str = "ols", m_cols=("m",)) -> pd.DataFrame:
    rows = []
    for model in sorted(df.model.unique()):
        for fmt in sorted(df[df.model == model].fmt.unique()):
            for m_col in m_cols:
                if m_col not in df:
                    continue
                d = analysis_frame(df[df.fmt == fmt], 2, model, m_col=m_col)
                if len(d) < 50:
                    continue
                r = primary(d, engine)
                rows.append(dict(model=model, fmt=fmt, congruence=m_col, n=r["n"],
                                 beta_neutral=r["beta_neutral"].est, rho=r["rho"].est,
                                 delta_b=r["delta_b"].est, delta_b_se=r["delta_b"].se, delta_b_p=r["delta_b"].p,
                                 theta_cong=r["theta_cong"].est, theta_se=r["theta_cong"].se,
                                 theta_p=r["theta_cong"].p, holm_delta_b=r["holm"]["delta_b"],
                                 holm_theta=r["holm"]["theta_cong"],
                                 support_shift=r["support_shift"].est, support_shift_se=r["support_shift"].se))
    return pd.DataFrame(rows)


def joint_models(df: pd.DataFrame, models: list[str] | None = None, fmt: str | None = None) -> dict:
    """Stack models; test x:contra:C(model) and x:contra:m:C(model) jointly zero."""
    d = df if fmt is None else df[df.fmt == fmt]
    if models:
        d = d[d.model.isin(models)]
    d = analysis_frame(d, 2)
    if d.model.nunique() < 2:
        return {}
    terms = (BASE_TERMS + " + C(model) + lp_before:C(model) + x:C(model) + x:contra:C(model) + x:contra:m:C(model)")
    fit = fit_ols(d, terms)
    names = [n for n in fit.params.index if n.startswith("x:contra:C(model)") or n.startswith("x:contra:m:C(model)")]
    R = np.zeros((len(names), len(fit.params)))
    for i, n in enumerate(names):
        R[i, list(fit.params.index).index(n)] = 1
    w = fit.wald_test(R, scalar=True)
    return dict(fit=fit, wald_stat=float(w.statistic), wald_p=float(w.pvalue), n_restrictions=len(names),
                interactions=pd.DataFrame([dict(term=n, est=fit.params[n], se=fit.bse[n]) for n in names]))


def stage_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """OLMo base vs SFT vs DPO vs Instruct, all in completion format."""
    from ..models import OLMO_STAGES
    d = df[(df.fmt == "completion") & df.model.isin(OLMO_STAGES)]
    res = per_model(d)
    if res.empty:
        return res
    res["stage_order"] = res.model.map({m: i for i, m in enumerate(OLMO_STAGES)})
    res = res.sort_values("stage_order")
    for col in ("delta_b", "theta_cong"):
        res[f"{col}_change_from_prev"] = res[col].diff()
    return res


def run_all(df: pd.DataFrame, engine: str = "ols") -> dict[str, pd.DataFrame]:
    out = {"per_model": per_model(df, engine, m_cols=tuple(c for c in ("m", "m_svy", "m_beh") if c in df)),
           "prior_shift": prior_shift(df), "stage_contrasts": stage_contrasts(df)}
    deltas, tests = [], []
    for model in sorted(df.model.unique()):
        for fmt in sorted(df[df.model == model].fmt.unique()):
            d = analysis_frame(df[df.fmt == fmt], 2, model)
            if len(d) < 200:
                continue
            dj, t = unequal_deference(d)
            deltas.append(dj.assign(model=model, fmt=fmt)); tests.append(t.assign(model=model, fmt=fmt))
            th = theta_by_dimension(d).assign(model=model, fmt=fmt)
            out.setdefault("_theta_dim", []).append(th)
            for name, tab in contrasts(df[df.fmt == fmt], model).items():
                out.setdefault(f"_{name}", []).append(tab.assign(model=model, fmt=fmt))
    if deltas:
        out["delta_j"] = pd.concat(deltas, ignore_index=True)
        out["delta_j_equality"] = pd.concat(tests, ignore_index=True)
    for k in [k for k in out if k.startswith("_")]:
        out[k[1:]] = pd.concat(out.pop(k), ignore_index=True)
    for fmt in df.fmt.unique():
        j = joint_models(df, fmt=fmt)
        if j:
            out[f"joint_{fmt}"] = j["interactions"].assign(wald_stat=j["wald_stat"], wald_p=j["wald_p"])
    return out
