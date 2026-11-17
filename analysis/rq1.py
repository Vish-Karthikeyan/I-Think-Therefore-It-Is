"""RQ1: does a stated user belief change how much a model's answer moves with the evidence?

    python analysis/rq1.py                     # every default analysis, results printed
    python analysis/rq1.py --list              # the analyses and what each one reports
    python analysis/rq1.py --only primary,tier --models gpt,llama --save out/rq1

Reads only `datasets/main_trials.csv.gz` and needs only pandas, numpy, scipy and statsmodels.

The primary regression, fitted per model and prompt format on the usable Tier 2
trials that show evidence (x != 0), with standard errors clustered by topic:

    lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m
               + x:dpos + x:C(cue_type) + x:C(source)

  lp_before, lp_after  log-odds the model gives H1 before / after seeing the evidence table
  x                    exact log-likelihood ratio of the table for H1 (the Bayesian update)
  contra, support      1 if the table contradicts / supports the user's stated belief
  m = b * g            identity-belief congruence: stated belief (b = +1/-1, 0 if none) times the
                       model's perceived stance of the user's group on H1 (g)
  dpos                 1 if the evidence favors H1 (a separate slope for each evidence direction)

  beta        (x)            no-belief evidence slope (1 = Bayesian, 0 = ignores the evidence)
  Delta_b     (x:contra)     belief-following: change in that slope when the evidence contradicts the
                             belief (negative = contradicting evidence moves the answer less)
  theta_cong  (x:contra:m)   change in Delta_b per unit of congruence (preregistered hypothesis: < 0)
  s           (x:support)    change in the slope when the evidence supports the belief
  Delta_b and theta_cong are co-primary and Holm-corrected within each model and format.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# --------------------------------------------------------------------------------------------------
# Analyses (set False to skip; --only / --skip override this on the command line)
# --------------------------------------------------------------------------------------------------
ANALYSES = {
    "instrument_checks": True,     # Results, instrument checks: exclusions, test-retest, null-evidence drift, readout calibration
    "primary": True,               # Results, RQ1: beta, Delta_b, theta_cong, s per model x format, Holm
    "joint_tests": True,           # Results, RQ1: do the models differ? Wald test per study x format
    "robustness": True,            # Appendix, robustness: g_svy, g_beh, weakest proxies dropped, 0.8 cutoff, Tier 1 low |g|
    "mixed_model": False,          # Appendix, robustness: mixed model (item intercepts, topic slopes); SLOW, ~1.5 min per 8,500 trials
    "unequal_deference": True,     # Results, unequal deference: delta_j per group, equality within each dimension
    "cue_type": True,              # Results, cue type: labels vs vignettes
    "tier": True,                  # Results, tier: Tier 1 Delta_b, Tier 2 - Tier 1, delta_j on Tier 1
    "theta_by_dimension": True,    # Results, dimension: theta_cong per identity dimension
    "simple_difference": True,     # Preregistered secondary check (not tabulated): rho forced to 1
    "prior_shift": True,           # Preregistered secondary check (not tabulated): identity/belief shift of the prior
    "spline": True,                # Secondary specifications: spline in x, F test against a linear slope (main run)
    "mixture": True,               # Secondary specifications: two-component mixture, 10 EM starts (main run); about 1 min
    "latent_prior": True,          # Secondary specifications: latent-prior (errors-in-variables) model (main run)
    "training_stages": True,       # Results, training stages (P0, P2, P5): stage ladders and same-trial step tests
    "system_prompt": True,         # Results, system prompt (P3): debiasing instruction vs the original model
    "prompt_wording": True,        # Results, robustness to wording (P6): rewordings, uniformity tests, tier comparison
    "predictions": True,           # Results, exploratory predictions: verdicts for P0, P2, P3, P5, P6
    "local_slopes": True,          # Results, shape of the belief effect: Delta_b by evidence-strength bin
    "additivity_anchoring": True,  # Results, shape of the belief effect: matched shift and anchoring checks
    "heuristic_conflict": True,    # Results, numeracy demands: slopes on heuristic-conflicting vs consistent items
}

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "datasets" / "main_trials.csv.gz"
UNSPECIFIED = "unspecified"   # the no-persona control; group effects are relative to it
DIMENSIONS = {
    "political": ["democrat", "republican", "independent"],
    "race": ["white", "black", "hispanic", "asian_american"],
    "sexuality": ["straight", "gay_lesbian", "bisexual"],
    "gender": ["man", "woman", "nonbinary"],
    "religion": ["evangelical", "catholic", "jewish", "muslim", "atheist"],
    "geography": ["rural", "urban"],
    "age": ["age_20s", "age_60s"],
}
BASE_TERMS = ("lp_before + x + x:contra + x:support + x:contra:m + x:support:m"
              " + x:dpos + x:C(cue_type) + x:C(source)")
TERMS = [t.strip() for t in BASE_TERMS.split("+")]

# Training ladders (base -> SFT -> DPO -> final)
OLMO7_STAGES = ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
OLMO13_STAGES = ["olmo13_base", "olmo13_sft", "olmo13_dpo", "olmo13_instruct"]
TULU_STAGES = ["tulu_base", "tulu_sft", "tulu_dpo", "tulu_final"]
# Chat models used by the analyses of the shape of the belief effect (one per model family and size)
HEADLINE_CHAT = ["gpt", "claude", "gemini", "phi4", "llama", "olmo_instruct", "olmo13_instruct", "tulu_final"]
# The same plus the OLMo 2 7B SFT and DPO stages (heuristic conflict)
CHAT10 = ["gpt", "llama", "olmo_sft", "olmo_dpo", "olmo_instruct", "gemini", "claude", "phi4",
          "olmo13_instruct", "tulu_final"]
# Models with two prompt rewordings (P6) and with the debiasing system prompt (P3)
REWORDED = ["olmo_instruct", "llama", "olmo13_instruct", "phi4"]
SYSTEM_PROMPT = ["olmo_instruct", "llama"]
# Two Tier 2 topics whose Pew item is the weakest proxy (two more weak proxies are not Tier 2 topics)
WEAK_PROXY_TOPICS = ["religious_exemptions", "concealed_carry", "faith_reentry", "public_transit"]

warnings.filterwarnings("ignore")   # statsmodels/pandas notices (e.g. clustered-covariance rank) clutter the output


# ==================================================================================================
# Data and the primary model
# ==================================================================================================
def load(path: Path) -> pd.DataFrame:
    """Read the trial data as the codebook says: only empty cells are missing (the evidence label
    e = "null" must survive), booleans cast back from text."""
    df = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])
    assert (df.e == "null").any(), "evidence label 'null' was lost on reading"
    for c in ("excluded", "excluded_cov08", "tier1_low_g", "calibration", "heuristic_conflict", "retest"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df


def frame(df: pd.DataFrame, tier: int | None = 2, model: str | None = None, fmt: str | None = None,
          cue_type: str | None = None, m_col: str = "m", excluded_col: str = "excluded") -> pd.DataFrame:
    """The analysis sample: usable trials (`excluded_col` False) with evidence (e != "null"), one tier
    (None = both), optionally one model / format / cue type. `m_col` swaps the congruence measure
    (m_svy, m_beh; missing = 0). The group factor has Unspecified as its reference level."""
    d = df[~df[excluded_col] & (df.e != "null")].copy()
    for col, val in (("tier", tier), ("model", model), ("fmt", fmt), ("cue_type", cue_type)):
        if val is not None:
            d = d[d[col] == val]
    if m_col != "m":
        d["m"] = d[m_col].fillna(0.0)
    d["group"] = pd.Categorical(d.group, categories=[UNSPECIFIED] + sorted(set(d.group) - {UNSPECIFIED}))
    d["d"] = d.d.astype(int)
    d["dpos"] = (d.d > 0).astype(int)
    return d.reset_index(drop=True)


def formula(terms: str, d: pd.DataFrame) -> str:
    """Drop control slopes that are constant in the sample (labels-only models, a single source)."""
    f = f"lp_after ~ {terms}"
    if d.cue_type.nunique() < 2:
        f = f.replace(" + x:C(cue_type)", "")
    if d.source.nunique() < 2:
        f = f.replace(" + x:C(source)", "")
    return f


def clustered_ols(f: str, d: pd.DataFrame):
    """OLS with standard errors clustered by topic (trials on one topic are not independent)."""
    return smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.topic.astype("category").cat.codes})


def fit_ols(d: pd.DataFrame, terms: str = BASE_TERMS):
    return clustered_ols(formula(terms, d), d)


def fit_mixed(d: pd.DataFrame):
    """Mixed-model engine: random intercepts per item and random slopes on x per topic (crossed,
    as variance components within one group)."""
    d = d.assign(_one=1)
    md = smf.mixedlm(formula(BASE_TERMS, d), d, groups="_one",
                     vc_formula={"item": "0 + C(item_id)", "topic_slope": "0 + C(topic):x"})
    return md.fit(method="lbfgs", reml=True)


def coef(fit, name: str) -> dict:
    """Estimate, SE, p and 95% CI of one coefficient (NaN if the term is not in the model)."""
    if name not in fit.params:
        return dict(est=np.nan, se=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    est, se = float(fit.params[name]), float(fit.bse[name])
    return dict(est=est, se=se, p=float(fit.pvalues[name]), lo=est - 1.96 * se, hi=est + 1.96 * se)


def lincomb(fit, weights: dict[str, float]) -> tuple[float, float, float]:
    """Estimate, SE and two-sided p of a linear combination of coefficients."""
    names = list(fit.params.index)
    c = np.zeros(len(names))
    for k, w in weights.items():
        c[names.index(k)] += w
    est, se = float(c @ fit.params.values), float(np.sqrt(c @ fit.cov_params().values @ c))
    return est, se, float(2 * stats.norm.sf(abs(est / se))) if se > 0 else np.nan


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Holm step-down: the i-th smallest p is compared with alpha / (m - i); after the first
    failure every later test fails too."""
    out, m, ok = {}, len(pvals), True
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        ok = ok and (p < alpha / (m - i))
        out[k] = bool(ok)
    return out


def model_formats(df: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(df.groupby(["model", "fmt"]).size().index.tolist())


def primary_fit(d: pd.DataFrame, engine: str = "ols") -> dict:
    """Fit the primary model on one sample and return its estimands."""
    fit = fit_ols(d) if engine == "ols" else fit_mixed(d)
    db, th, sh = coef(fit, "x:contra"), coef(fit, "x:contra:m"), coef(fit, "x:support")
    sig = holm({"delta_b": db["p"], "theta_cong": th["p"]})
    return dict(n=int(fit.nobs), beta_neutral=coef(fit, "x")["est"], rho=coef(fit, "lp_before")["est"],
                delta_b=db["est"], delta_b_se=db["se"], delta_b_p=db["p"], delta_b_lo=db["lo"], delta_b_hi=db["hi"],
                theta_cong=th["est"], theta_se=th["se"], theta_p=th["p"], theta_lo=th["lo"], theta_hi=th["hi"],
                holm_delta_b=sig["delta_b"], holm_theta=sig["theta_cong"],
                support_shift=sh["est"], support_shift_se=sh["se"], support_shift_p=sh["p"])


def per_model(df: pd.DataFrame, engine: str = "ols", m_col: str = "m", excluded_col: str = "excluded") -> pd.DataFrame:
    """The primary model for every model x format in `df` (Tier 2)."""
    rows = []
    for model, fmt in model_formats(df):
        d = frame(df, 2, model, fmt, m_col=m_col, excluded_col=excluded_col)
        study = df.loc[df.model == model, "study"].iloc[0]
        if len(d) < 50:
            rows.append(dict(study=study, model=model, fmt=fmt, n=len(d), note="fewer than 50 usable trials"))
            continue
        rows.append(dict(study=study, model=model, fmt=fmt, congruence=m_col, **primary_fit(d, engine)))
    return pd.DataFrame(rows)


_CACHE: dict = {}


def primary_all(df: pd.DataFrame) -> pd.DataFrame:
    """The primary estimates of every model x format (computed once, shared by several analyses)."""
    if "primary" not in _CACHE:
        _CACHE["primary"] = per_model(df)
    return _CACHE["primary"]


def stacked_variant_test(df: pd.DataFrame, base: str, variant: str, trial_ids=None) -> dict:
    """Two models on the same trials (chat, Tier 2), every term of the primary model interacted with a
    variant dummy; the variant's change in Delta_b is the coefficient on x:contra:var."""
    key = ("stacked", base, variant, None if trial_ids is None else len(trial_ids))
    if key in _CACHE:
        return _CACHE[key]
    d = df[df.model.isin([base, variant]) & (df.fmt == "chat")]
    if trial_ids is not None:
        d = d[d.trial_id.isin(trial_ids)]
    d = frame(d, 2)
    d["var"] = (d.model == variant).astype(int)
    terms = [t for t in TERMS if d.cue_type.nunique() > 1 or "cue_type" not in t]
    fit = fit_ols(d, " + ".join(terms + ["var"] + [f"{t}:var" for t in terms]))
    out = {"n": int(fit.nobs)}
    for name, term in (("delta_b_change", "x:contra:var"), ("support_shift_change", "x:support:var"),
                       ("beta_change", "x:var"), ("theta_change", "x:contra:m:var")):
        e = coef(fit, term)
        out.update({name: e["est"], f"{name}_se": e["se"], f"{name}_p": e["p"]})
    _CACHE[key] = out
    return out


def has(df: pd.DataFrame, *models: str) -> bool:
    return all((df.model == m).any() for m in models)


def wald_rows(fit, rows: list[dict]) -> tuple[float, float, int]:
    """Wald chi2 test of R b = 0; each row of R is given as {coefficient name: weight}."""
    names = list(fit.params.index)
    R = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for n, w in r.items():
            R[i, names.index(n)] += w
    rb = R @ fit.params.values
    stat = float(rb @ np.linalg.solve(R @ fit.cov_params().values @ R.T, rb))
    return stat, float(stats.chi2.sf(stat, len(rows))), len(rows)


def logit(p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)   # the same clipping as the dataset's log-odds
    return np.log(p / (1 - p))


# ==================================================================================================
# Instrument checks
# ==================================================================================================
def instrument_checks(df: pd.DataFrame) -> dict:
    keys = ["model", "fmt", "tier", "cue_type"]
    g = df.groupby(keys)
    sample = g.agg(trials=("trial_id", "size"), usable=("excluded", lambda v: int((~v).sum())),
                   excluded_frac=("excluded", "mean")).reset_index()
    coverage = g.agg(n=("trial_id", "size"), excluded_frac=("excluded", "mean"), cov_prior=("cov_prior", "mean"),
                     cov_post=("cov_post", "mean"), reprompt_frac=("att_post", "mean")).reset_index()

    # Test-retest: the two paraphrased BEFORE questions, one row per distinct prior prompt. Gemini and
    # Claude asked the paraphrase on a seeded subset only (lp_before_rep missing elsewhere).
    d = df[~df.excluded & df.lp_before_rep.notna()].drop_duplicates(
        ["model", "fmt", "cue_id", "topic", "h_dir", "b", "source", "q", "treated_first"])
    retest = pd.DataFrame([dict(model=m, fmt=f, r=x.lp_before.corr(x.lp_before_rep), n=len(x),
                                mean_abs_diff=(x.lp_before - x.lp_before_rep).abs().mean())
                           for (m, f), x in d.groupby(["model", "fmt"])])

    # Null-evidence tables (D = 0): the Bayesian update is 0, so mean |y| is the noise floor and
    # b * y is drift toward the stated belief.
    d = df[(df.D == 0) & ~df.excluded].copy()
    d["e_null"] = np.where(d.b == 0, "no_belief", "belief")
    d["toward_belief"] = d.b * d.y
    null = d.groupby(["model", "fmt", "tier", "e_null"]).agg(
        n=("y", "size"), mean_abs_y=("y", lambda v: v.abs().mean()), mean_y=("y", "mean"),
        drift_toward_belief=("toward_belief", "mean")).reset_index()

    prior = df[~df.excluded].groupby(["model", "fmt", "tier", "e"]).p_prior.describe(
        percentiles=[0.05, 0.5, 0.95]).reset_index()

    # Readout calibration (open models): calibration trials also read by 10 sampled answers
    d = df[df.calibration & df.p_prior_s.notna() & df.p_post_s.notna() & ~df.excluded].copy()
    d["lp_before_s"], d["lp_after_s"] = logit(d.p_prior_s), logit(d.p_post_s)
    rows = []
    for (m, f), x in d.groupby(["model", "fmt"]):
        rec = dict(model=m, fmt=f, n=len(x), r_prior=x.lp_before.corr(x.lp_before_s),
                   r_post=x.lp_after.corr(x.lp_after_s))
        for tag, a, b in (("dist", "lp_after", "lp_before"), ("sample", "lp_after_s", "lp_before_s")):
            fit = smf.ols(f"{a} ~ {b} + x + x:contra", x).fit()
            rec[f"beta_{tag}"], rec[f"delta_b_{tag}"] = fit.params["x"], fit.params.get("x:contra", np.nan)
        rows.append(rec)
    calib = pd.DataFrame(rows)

    # Raw slopes of y on x per evidence type (Tier 2, no controls): a quick picture of the main effect
    d = df[~df.excluded & (df.e != "null") & (df.tier == 2)]
    rows = []
    for (m, f, e), x in d.groupby(["model", "fmt", "e"]):
        fit = smf.ols("y ~ x", x).fit()
        rows.append(dict(model=m, fmt=f, e=e, n=len(x), slope=fit.params["x"], se=fit.bse["x"], mean_y=x.y.mean()))
    slopes = pd.DataFrame(rows)

    # Summary per model x format: share excluded, retest r, null-table noise (both tiers) and drift
    rows = []
    for r in retest.itertuples():
        s = sample[(sample.model == r.model) & (sample.fmt == r.fmt)]
        n = null[(null.model == r.model) & (null.fmt == r.fmt)]
        nb = n[n.e_null == "belief"]
        rows.append(dict(model=r.model, fmt=r.fmt, trials=int(s.trials.sum()),
                         excluded_frac=1 - s.usable.sum() / s.trials.sum(), retest_r=r.r,
                         null_abs_y=np.average(n.mean_abs_y, weights=n.n) if len(n) else np.nan,
                         null_drift=np.average(nb.drift_toward_belief, weights=nb.n) if len(nb) else np.nan))
    return {"summary": pd.DataFrame(rows), "sample": sample, "coverage": coverage, "retest": retest,
            "null_evidence": null, "prior_distribution": prior, "readout_calibration": calib,
            "evidence_slopes": slopes}


# ==================================================================================================
# Primary analysis and joint tests
# ==================================================================================================
def primary(df: pd.DataFrame) -> dict:
    return {"per_model": primary_all(df)}


def joint_tests(df: pd.DataFrame) -> dict:
    """Within one study and format, stack all models and test whether Delta_b and theta_cong differ by
    model (each model keeps its own intercept, rho and beta; Wald test on the model interactions)."""
    out = []
    for study in sorted(df.study.unique()):
        for fmt in sorted(df.fmt.unique()):
            d = frame(df[df.study == study], 2, fmt=fmt)
            if d.model.nunique() < 2:
                continue
            terms = BASE_TERMS + (" + C(model) + lp_before:C(model) + x:C(model) + x:contra:C(model)"
                                  " + x:contra:m:C(model)")
            fit = fit_ols(d, terms)
            names = [n for n in fit.params.index if n.startswith(("x:contra:C(model)", "x:contra:m:C(model)"))]
            t = pd.DataFrame([dict(study=study, fmt=fmt, term=n, est=fit.params[n], se=fit.bse[n]) for n in names])
            # A topic-clustered covariance has rank at most (topics - 1): with more restrictions than
            # that (the exploratory set, 36 restrictions on 17 topics) the Wald test is undefined.
            k = d.topic.nunique()
            if len(names) > k - 1:
                t = t.assign(wald_stat=np.nan, wald_p=np.nan, n_restrictions=len(names),
                             note=f"not estimable: {len(names)} restrictions > {k} topic clusters - 1")
            else:
                R = np.zeros((len(names), len(fit.params)))
                for i, n in enumerate(names):
                    R[i, list(fit.params.index).index(n)] = 1
                w = fit.wald_test(R, scalar=True)
                t = t.assign(wald_stat=float(w.statistic), wald_p=float(w.pvalue), n_restrictions=len(names))
            out.append(t)
    return {"joint": pd.concat(out, ignore_index=True)} if out else {}


# ==================================================================================================
# Secondary analyses: unequal deference, cue type, tier, dimension, simple difference, prior shift
# ==================================================================================================
def secondary_runs(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Model x format runs with at least 200 usable Tier 2 trials (the secondary analyses' threshold)."""
    runs = []
    for model, fmt in model_formats(df):
        d = frame(df, 2, model, fmt)
        if len(d) >= 200:
            runs.append((model, fmt, d))
    return runs


def deference_fit(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta_j = coefficient on x:contra:C(group)[j], Unspecified as reference; Wald test within each
    dimension that all its groups' delta_j are equal (successive differences = 0)."""
    terms = ("lp_before + x + x:C(group) + x:contra + x:contra:C(group) + x:support + x:support:C(group)"
             " + x:dpos + x:C(cue_type) + x:C(source)")
    fit = fit_ols(d, terms)
    rows = []
    for g in d.group.cat.categories:
        if g == UNSPECIFIED:
            continue
        e = coef(fit, f"x:contra:C(group)[T.{g}]")
        dim = d.loc[d.group == g, "dimension"].iloc[0] if (d.group == g).any() else None
        rows.append(dict(group=g, dimension=dim, delta=e["est"], se=e["se"], p=e["p"], lo=e["lo"], hi=e["hi"]))
    tests, names_all = [], list(fit.params.index)
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


def unequal_deference(df: pd.DataFrame) -> dict:
    dj, eq = [], []
    for model, fmt, d in secondary_runs(df):
        a, b = deference_fit(d)
        study = d.study.iloc[0]
        dj.append(a.assign(study=study, model=model, fmt=fmt))
        eq.append(b.assign(study=study, model=model, fmt=fmt))
    eq = pd.concat(eq, ignore_index=True)
    # Holm across every equality test of a study (for the main run: 7 dimensions x 9 runs = 63 tests)
    eq["holm_within_study"] = False
    for study, t in eq.groupby("study"):
        sig = holm(dict(zip(t.index, t.p)))
        eq.loc[t.index, "holm_within_study"] = [sig[i] for i in t.index]
    return {"delta_j": pd.concat(dj, ignore_index=True), "equality": eq}


def cue_type(df: pd.DataFrame) -> dict:
    """Separate fits for explicit labels and vignettes, and their difference in one joint fit."""
    rows = []
    for model, fmt, d2 in secondary_runs(df):
        for ct in ("label", "vignette"):
            d = frame(df, 2, model, fmt, cue_type=ct)
            if len(d) > 50:
                r = primary_fit(d)
                rows.append(dict(model=model, fmt=fmt, cue_type=ct, delta_b=r["delta_b"], delta_b_se=r["delta_b_se"],
                                 theta=r["theta_cong"], theta_se=r["theta_se"]))
        if d2.cue_type.nunique() == 2:
            f = fit_ols(d2, BASE_TERMS + " + x:contra:C(cue_type) + x:contra:m:C(cue_type)")
            db, th = coef(f, "x:contra:C(cue_type)[T.vignette]"), coef(f, "x:contra:m:C(cue_type)[T.vignette]")
            rows.append(dict(model=model, fmt=fmt, cue_type="vignette_minus_label", delta_b=db["est"],
                             delta_b_se=db["se"], theta=th["est"], theta_se=th["se"]))
    return {"cue_type": pd.DataFrame(rows)}


def tier_contrast(df: pd.DataFrame, model: str, fmt: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Delta_b on Tier 1 (topics without social meaning), the Tier 2 - Tier 1 difference, and delta_j
    on Tier 1."""
    dall = frame(df, None, model, fmt)
    if dall.tier.nunique() < 2:
        return None, None
    dall["tier2"] = (dall.tier == 2).astype(int)
    f = fit_ols(dall, "lp_before + x + x:tier2 + x:contra + x:contra:tier2 + x:support + x:support:tier2"
                      " + x:dpos + x:C(source)")
    tier = pd.DataFrame([dict(model=model, fmt=fmt, term="delta_b_tier1", **coef(f, "x:contra")),
                         dict(model=model, fmt=fmt, term="amplification_tier2_minus_tier1", **coef(f, "x:contra:tier2"))])
    d1 = frame(df, 1, model, fmt)
    dj1 = deference_fit(d1)[0].assign(model=model, fmt=fmt) if len(d1) > 200 else None
    return tier, dj1


def tier(df: pd.DataFrame) -> dict:
    tiers, dj1 = [], []
    for model, fmt, _ in secondary_runs(df):
        t, j = tier_contrast(df, model, fmt)
        tiers += [t] if t is not None else []
        dj1 += [j] if j is not None else []
    out = {"tier_contrast": pd.concat(tiers, ignore_index=True)} if tiers else {}
    if dj1:
        out["delta_j_tier1"] = pd.concat(dj1, ignore_index=True)
    return out


def theta_by_dimension(df: pd.DataFrame) -> dict:
    """theta_cong per dimension: x:contra:m replaced by one term x*contra*m*[dimension = k] per dimension."""
    out = []
    for model, fmt, d in secondary_runs(df):
        dd = d.copy()
        for k in DIMENSIONS:
            dd[f"xcm_{k}"] = dd.x * dd.contra * dd.m * (dd.dimension == k)
        terms = BASE_TERMS.replace(" + x:contra:m", "") + "".join(
            f" + xcm_{k}" for k in DIMENSIONS if (dd[f"xcm_{k}"] != 0).any())
        fit = fit_ols(dd, terms)
        out.append(pd.DataFrame([dict(model=model, fmt=fmt, dimension=k, **coef(fit, f"xcm_{k}")) for k in DIMENSIONS]))
    return {"theta_by_dimension": pd.concat(out, ignore_index=True)}


def simple_difference(df: pd.DataFrame) -> dict:
    """Regress the update y = lp_after - lp_before on the evidence terms (forces rho = 1)."""
    out = []
    for model, fmt, d in secondary_runs(df):
        f = clustered_ols("y ~ x + x:contra + x:support + x:contra:m", d)
        out.append(pd.DataFrame([dict(model=model, fmt=fmt, term=k, est=f.params[k], se=f.bse[k]) for k in f.params.index]))
    return {"simple_difference": pd.concat(out, ignore_index=True)}


def prior_shift(df: pd.DataFrame) -> dict:
    """gamma: mean BEFORE log-odds of (group, belief b) minus that of Unspecified users with no belief,
    within model, format, tier and cue type."""
    d = df[~df.excluded]
    keys = ["model", "fmt", "tier", "cue_type"]
    ref = d[(d.group == UNSPECIFIED) & (d.b == 0)].groupby(keys).lp_before.mean().rename("ref")
    g = d.groupby(keys + ["group", "b"]).lp_before.agg(["mean", "sem", "size"]).reset_index().join(ref, on=keys)
    g["gamma"] = g["mean"] - g["ref"]
    return {"prior_shift": g}


# ==================================================================================================
# Robustness (preregistered; main run, where g_svy and the Tier 1 low-|g| flag exist)
# ==================================================================================================
def robustness(df: pd.DataFrame) -> dict:
    main = df[df.study == "main"]
    checks = {
        "primary": per_model(main),
        "congruence_svy": per_model(main, m_col="m_svy"),               # Pew survey stance instead of g
        "congruence_beh": per_model(main, m_col="m_beh"),               # behavioral stance instead of g
        "svy_weak_dropped": per_model(main[~main.topic.isin(WEAK_PROXY_TOPICS)], m_col="m_svy"),
        "completion_cov08": per_model(main[main.fmt == "completion"], excluded_col="excluded_cov08"),
    }
    keep = ["model", "fmt", "n", "delta_b", "delta_b_se", "holm_delta_b", "theta_cong", "theta_se", "holm_theta"]
    summary = pd.concat([t.reindex(columns=keep).assign(check=k) for k, t in checks.items()], ignore_index=True)
    out = {"summary": summary[["check"] + keep].sort_values(["model", "fmt", "check"]).reset_index(drop=True)}
    out.update({k: v for k, v in checks.items() if k != "primary"})
    # Tier 1 restricted to (topic, cue) pairs with mean |g| < 0.25 in the pilot (Tier 2 unchanged)
    sub = main[(main.tier == 2) | main.tier1_low_g]
    tiers, dj1 = [], []
    for model, fmt in model_formats(sub):
        t, j = tier_contrast(sub, model, fmt)
        tiers += [t] if t is not None else []
        dj1 += [j] if j is not None else []
    out["tier1_low_g_tier_contrast"] = pd.concat(tiers, ignore_index=True)
    out["tier1_low_g_delta_j_tier1"] = pd.concat(dj1, ignore_index=True)
    return out


def mixed_model(df: pd.DataFrame) -> dict:
    return {"per_model": per_model(df[df.study == "main"], engine="mixed")}


# ==================================================================================================
# Secondary specifications (preregistered; main run)
# ==================================================================================================
def spec_runs(df: pd.DataFrame):
    for model, fmt in model_formats(df[df.study == "main"]):
        yield model, fmt, frame(df, 2, model, fmt)


def spline(df: pd.DataFrame) -> dict:
    """Spline (4 df) in x for the no-belief, contradicting and supporting slopes; F test of the spline
    model against the linear one. The fitted curves are evaluated at 25 points of x (prior 0, m 0)."""
    tests, curves = [], []
    for model, fmt, d in spec_runs(df):
        lin = smf.ols("lp_after ~ lp_before + x + x:contra + x:support + x:contra:m", d).fit()
        spl = smf.ols("lp_after ~ lp_before + bs(x, df=4) + bs(x, df=4):contra + bs(x, df=4):support + x:contra:m",
                      d).fit()
        f_stat, f_p, _ = spl.compare_f_test(lin)
        tests.append(dict(model=model, fmt=fmt, f_stat=f_stat, f_p=f_p))
        grid = np.linspace(d.x.min(), d.x.max(), 25)
        for e in ("neutral", "contra", "support"):
            nd = pd.DataFrame(dict(x=grid, lp_before=0.0, contra=int(e == "contra"), support=int(e == "support"), m=0.0))
            curves.append(pd.DataFrame(dict(model=model, fmt=fmt, e=e, x=grid, response=spl.predict(nd))))
    return {"f_test": pd.DataFrame(tests), "curves": pd.concat(curves, ignore_index=True)}


def mixture_em(d: pd.DataFrame, seed: int, covariates: str = "contra + support + contra:m + C(dimension)",
               n_iter: int = 200, tol: float = 1e-6) -> dict:
    """Two components fitted by EM:
        updated:  lp_after = a1 + r1*lp_before + b1*x + e,  e ~ N(0, s1)
        ignored:  lp_after = a0 + r0*lp_before      + e,  e ~ N(0, s0)
    P(ignored) = logistic(Z gamma), Z from `covariates`: weighted least squares for the components,
    a fractional-response logistic GLM for the mixing weights. Starts from random weights."""
    rng = np.random.default_rng(seed)
    y = d.lp_after.values
    X1 = np.column_stack([np.ones(len(d)), d.lp_before.values, d.x.values])
    X0 = X1[:, :2]
    Zdf = patsy.dmatrix(covariates, d, return_type="dataframe")
    Z = np.asarray(Zdf)
    w = np.clip(rng.uniform(0.2, 0.8, len(d)), 1e-3, 1 - 1e-3)        # P(ignored | data)
    ll_old = -np.inf
    for it in range(n_iter):
        b1 = np.linalg.lstsq(X1 * np.sqrt(1 - w)[:, None], y * np.sqrt(1 - w), rcond=None)[0]
        b0 = np.linalg.lstsq(X0 * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        s1 = np.sqrt(np.sum((1 - w) * (y - X1 @ b1) ** 2) / np.sum(1 - w))
        s0 = np.sqrt(np.sum(w * (y - X0 @ b0) ** 2) / np.sum(w))
        gam = sm.GLM(w, Z, family=sm.families.Binomial()).fit().params
        pi = 1 / (1 + np.exp(-(Z @ gam)))
        f1 = stats.norm.pdf(y, X1 @ b1, s1) * (1 - pi)
        f0 = stats.norm.pdf(y, X0 @ b0, s0) * pi
        tot = f1 + f0 + 1e-300
        ll = np.sum(np.log(tot))
        w = np.clip(f0 / tot, 1e-6, 1 - 1e-6)
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
    return dict(updated=dict(alpha=b1[0], rho=b1[1], beta=b1[2], sigma=s1),
                p_ignore_coefs=pd.Series(gam, index=Zdf.columns), loglik=ll, iterations=it + 1,
                mean_p_ignore=float(pi.mean()))


def mixture(df: pd.DataFrame, n_starts: int = 10) -> dict:
    """The mixture likelihood has several local maxima (with an evidence slope near 0 the components are
    hard to tell apart), so EM runs from 10 seeds and the best start is kept. The split counts as
    stable if the starts within 1 log-likelihood unit of the best agree on mean P(ignore) within 0.05.
    Trials with |x| <= 0.05 are left out."""
    best_rows, starts_all, coefs = [], [], []
    for model, fmt, d in spec_runs(df):
        nz = d[d.x.abs() > 0.05].reset_index(drop=True)
        fits = [mixture_em(nz, seed=s) for s in range(n_starts)]
        starts = pd.DataFrame([dict(model=model, fmt=fmt, seed=s, loglik=f["loglik"], iterations=f["iterations"],
                                    mean_p_ignore=f["mean_p_ignore"], **{f"updated_{k}": v for k, v in f["updated"].items()})
                               for s, f in enumerate(fits)])
        b = int(starts.loglik.idxmax())
        near = starts[starts.loglik >= starts.loglik.max() - 1.0]
        stable = bool(near.mean_p_ignore.max() - near.mean_p_ignore.min() < 0.05)
        best_rows.append(dict(model=model, fmt=fmt, **{f"updated_{k}": v for k, v in fits[b]["updated"].items()},
                              mean_p_ignore=fits[b]["mean_p_ignore"], loglik=fits[b]["loglik"], n_starts=n_starts,
                              stable_across_starts=stable))
        starts_all.append(starts)
        coefs.append(fits[b]["p_ignore_coefs"].rename("est").rename_axis("term").reset_index().assign(model=model, fmt=fmt))
    return {"best": pd.DataFrame(best_rows), "starts": pd.concat(starts_all, ignore_index=True),
            "p_ignore_coefs": pd.concat(coefs, ignore_index=True)}


def latent_prior(df: pd.DataFrame) -> dict:
    """lp_before is a noisy measure of the model's prior; the paraphrased prior lp_before_rep measures
    the same prior independently, so it instruments lp_before (2SLS). Gives rho corrected for
    measurement error and the slopes conditional on the latent prior."""
    out = []
    for model, fmt, d in spec_runs(df):
        X_ex = patsy.dmatrix("x + x:contra + x:support + x:contra:m", d, return_type="dataframe")
        endog, inst = d.lp_before.values, d.lp_before_rep.values
        Zfull = np.column_stack([X_ex.values, inst])
        endog_hat = Zfull @ np.linalg.lstsq(Zfull, endog, rcond=None)[0]           # first stage
        X2 = np.column_stack([X_ex.values, endog_hat])
        beta = np.linalg.lstsq(X2, d.lp_after.values, rcond=None)[0]
        resid = d.lp_after.values - np.column_stack([X_ex.values, endog]) @ beta  # residuals use the observed prior
        cov = resid @ resid / (len(d) - X2.shape[1]) * np.linalg.inv(X2.T @ X2)
        r2 = 1 - np.sum((endog - endog_hat) ** 2) / np.sum((endog - endog.mean()) ** 2)
        names = list(X_ex.columns) + ["lp_before(latent)"]
        out.append(pd.DataFrame(dict(model=model, fmt=fmt, term=names, est=beta, se=np.sqrt(np.diag(cov)),
                                     first_stage_r2=r2)))
    return {"latent_prior": pd.concat(out, ignore_index=True)}


# ==================================================================================================
# Exploratory model comparisons: training stages (P0, P2, P5), system prompt (P3), wording (P6)
# ==================================================================================================
def ladder(res: pd.DataFrame, stages: list[str], fmt: str) -> pd.DataFrame:
    """One training ladder in order, with the change from the previous stage."""
    r = res[(res.fmt == fmt) & res.model.isin(stages)].copy()
    r["stage_order"] = r.model.map({m: i for i, m in enumerate(stages)})
    r = r.sort_values("stage_order")
    for col in ("beta_neutral", "delta_b", "theta_cong"):
        r[f"{col}_change_from_prev"] = r[col].diff()
    return r


def stage_steps(df: pd.DataFrame) -> pd.DataFrame:
    steps = [("tulu", "sft->dpo", "tulu_sft", "tulu_dpo"), ("tulu", "dpo->final", "tulu_dpo", "tulu_final"),
             ("olmo7", "sft->dpo", "olmo_sft", "olmo_dpo"), ("olmo7", "dpo->instruct", "olmo_dpo", "olmo_instruct")]
    return pd.DataFrame([dict(ladder=l, step=s, **stacked_variant_test(df, a, b))
                         for l, s, a, b in steps if has(df, a, b)])


def training_stages(df: pd.DataFrame) -> dict:
    pm = primary_all(df)
    lad = pd.concat([ladder(pm, st, f).assign(ladder=name) for name, st in
                     (("olmo7", OLMO7_STAGES), ("olmo13", OLMO13_STAGES), ("tulu", TULU_STAGES))
                     for f in ("chat", "completion")], ignore_index=True)
    out = {"ladders": lad, "steps": stage_steps(df)}
    out["scale_13_14b"] = pm[pm.model.isin(["olmo13_instruct", "phi4"]) & (pm.fmt == "chat")]   # P2
    return out


def system_prompt(df: pd.DataFrame) -> dict:
    pm = primary_all(df).set_index(["model", "fmt"])
    rows = []
    for base in SYSTEM_PROMPT:
        if not has(df, base, f"{base}_sp"):
            continue
        b0, b1 = pm.loc[(base, "chat")], pm.loc[(f"{base}_sp", "chat")]
        rows.append(dict(model=base, delta_b_original=b0.delta_b, delta_b_sp=b1.delta_b, delta_b_sp_p=b1.delta_b_p,
                         beta_original=b0.beta_neutral, beta_sp=b1.beta_neutral,
                         **stacked_variant_test(df, base, f"{base}_sp")))
    return {"system_prompt": pd.DataFrame(rows)} if rows else {}


def p6_table(df: pd.DataFrame) -> pd.DataFrame:
    """P6: each rewording against the original model on the same third of trials."""
    rows = []
    for base in REWORDED:
        for v in (1, 2):
            var = f"{base}_pv{v}"
            if not has(df, base, var):
                continue
            ids = set(df[df.model == var].trial_id)
            o = per_model(df[(df.model == base) & (df.fmt == "chat") & df.trial_id.isin(ids)]).iloc[0]
            n = per_model(df[df.model == var]).iloc[0]
            if pd.isna(n.get("delta_b")):
                rows.append(dict(model=base, variant=var, note=n.get("note")))
                continue
            rows.append(dict(model=base, variant=var, n_original=o.n, n_variant=n.n,
                             delta_b_original=o.delta_b, delta_b_lo=o.delta_b_lo, delta_b_hi=o.delta_b_hi,
                             delta_b_variant=n.delta_b, support_original=o.support_shift, support_variant=n.support_shift,
                             same_sign_delta_b=np.sign(o.delta_b) == np.sign(n.delta_b),
                             same_sign_support=np.sign(o.support_shift) == np.sign(n.support_shift),
                             within_ci=o.delta_b_lo <= n.delta_b <= o.delta_b_hi,
                             **stacked_variant_test(df, base, var, ids)))
    return pd.DataFrame(rows)


def prompt_wording(df: pd.DataFrame) -> dict:
    """P6 and its extension to both estimands and both tiers. Arms: original, wording 1, wording 2 of
    each reworded model, on the trials run under the rewordings (the same ids for both)."""
    bases = [b for b in REWORDED if has(df, b, f"{b}_pv1", f"{b}_pv2")]
    if not bases:
        return {}
    arm = {0: "original", 1: "wording 1", 2: "wording 2"}
    name = lambda b, v: b if v == 0 else f"{b}_pv{v}"  # noqa: E731
    est_rows, chg_rows, tests = [], [], []

    def change(d0, d1):
        d = pd.concat([d0.assign(var=0), d1.assign(var=1)], ignore_index=True)
        terms = [t for t in TERMS if d.cue_type.nunique() > 1 or "cue_type" not in t]
        fit = fit_ols(d, " + ".join(terms + ["var"] + [f"{t}:var" for t in terms]))
        out = {"n_stacked": int(fit.nobs)}
        for nm, term in (("delta_b_change", "x:contra:var"), ("theta_change", "x:contra:m:var")):
            e = coef(fit, term)
            out.update({nm: e["est"], f"{nm}_se": e["se"], f"{nm}_p": e["p"]})
        return out

    for tier_ in (2, 1):
        frames = {}
        for b in bases:
            ids = set(df[df.model == name(b, 1)].trial_id)
            assert ids == set(df[df.model == name(b, 2)].trial_id), b
            for v in (0, 1, 2):
                d = frame(df[(df.model == name(b, v)) & (df.fmt == "chat") & df.trial_id.isin(ids)], tier_)
                frames[(b, v)] = d
                fit = fit_ols(d)
                db, th = coef(fit, "x:contra"), coef(fit, "x:contra:m")
                mx = float(d.loc[d.contra == 1, "x"].abs().mean())
                est_rows.append(dict(tier=tier_, model=b, arm=arm[v], v=v, n=int(fit.nobs), n_topics=d.topic.nunique(),
                                     beta=coef(fit, "x")["est"], delta_b=db["est"], delta_b_se=db["se"],
                                     delta_b_p=db["p"], delta_b_lo=db["lo"], delta_b_hi=db["hi"],
                                     theta=th["est"], theta_se=th["se"], theta_p=th["p"], theta_lo=th["lo"],
                                     theta_hi=th["hi"], support=coef(fit, "x:support")["est"], mean_abs_x_contra=mx,
                                     implied_shift=-db["est"] * mx, implied_shift_se=db["se"] * mx))
            for v in (1, 2):
                d0, d1 = frames[(b, 0)], frames[(b, v)]
                both = set(d0.trial_id) & set(d1.trial_id)        # sensitivity: trials usable in both arms
                st_i = change(d0[d0.trial_id.isin(both)], d1[d1.trial_id.isin(both)])
                chg_rows.append(dict(tier=tier_, model=b, arm=arm[v], v=v, **change(d0, d1), n_both_usable=len(both),
                                     delta_b_change_both=st_i["delta_b_change"],
                                     delta_b_change_both_se=st_i["delta_b_change_se"],
                                     theta_change_both=st_i["theta_change"],
                                     theta_change_both_se=st_i["theta_change_se"]))

        # One regression stacking every arm, every term (and intercept) arm-specific, so the covariance
        # of all the changes is available for joint tests.
        parts, rhs = [], []
        for k, d in enumerate(frames.values()):
            parts.append(d.assign(**{f"A{j}": float(j == k) for j in range(len(frames))}))
        for j in range(len(frames)):
            rhs += [f"A{j}"] + [f"{t}:A{j}" for t in TERMS]
        d = pd.concat(parts, ignore_index=True)
        fit = clustered_ols("lp_after ~ 0 + " + " + ".join(rhs), d)
        idx = {key: j for j, key in enumerate(frames)}
        p = lambda b, v, term="x:contra": f"{term}:A{idx[(b, v)]}"  # noqa: E731
        chg = lambda b, v: {p(b, v): 1, p(b, 0): -1}                # noqa: E731

        def minus(r1, r2):
            r = dict(r1)
            for k, w in r2.items():
                r[k] = r.get(k, 0) - w
            return r
        for b in bases:
            s, pv, k = wald_rows(fit, [minus(chg(b, 1), chg(b, 2))])
            tests.append(dict(tier=tier_, test=f"{b}: wording 1 change = wording 2 change (Delta_b)", chi2=s, df=k, p=pv))
            s, pv, k = wald_rows(fit, [{p(b, 1, "x:contra:m"): 1, p(b, 0, "x:contra:m"): -1},
                                       {p(b, 2, "x:contra:m"): 1, p(b, 0, "x:contra:m"): -1}])
            tests.append(dict(tier=tier_, test=f"{b}: theta_cong equal across original, wording 1, wording 2",
                              chi2=s, df=k, p=pv))
        allc = [(b, v) for b in bases for v in (1, 2)]
        for label, rows in ((f"all {len(allc)} Delta_b changes equal", [minus(chg(*c), chg(*allc[0])) for c in allc[1:]]),
                            (f"all {len(allc)} Delta_b changes zero", [chg(*c) for c in allc])):
            s, pv, k = wald_rows(fit, rows)
            tests.append(dict(tier=tier_, test=label, chi2=s, df=k, p=pv))
        avg = lambda b: {p(b, 1): 0.5, p(b, 2): 0.5, p(b, 0): -1}  # noqa: E731
        if len(bases) > 1:
            s, pv, k = wald_rows(fit, [minus(avg(b), avg(bases[0])) for b in bases[1:]])
            tests.append(dict(tier=tier_, test=f"model-average Delta_b change equal across {len(bases)} models",
                              chi2=s, df=k, p=pv))
        # Ratio variant / original Delta_b with a delta-method SE
        V = fit.cov_params()
        for b, v in allc:
            a, o = fit.params[p(b, v)], fit.params[p(b, 0)]
            g = np.array([1 / o, -a / o ** 2])
            cv = V.loc[[p(b, v), p(b, 0)], [p(b, v), p(b, 0)]].values
            row = next(r for r in chg_rows if r["tier"] == tier_ and r["model"] == b and r["v"] == v)
            row.update(ratio=a / o, ratio_se=float(np.sqrt(g @ cv @ g)))

    E, C = pd.DataFrame(est_rows), pd.DataFrame(chg_rows)
    # Tier 2 vs Tier 1 change: disjoint topics, so independent clusters
    rows = []
    for b, v in [(b, v) for b in bases for v in (1, 2)]:
        c2 = C[(C.tier == 2) & (C.model == b) & (C.v == v)].iloc[0]
        c1 = C[(C.tier == 1) & (C.model == b) & (C.v == v)].iloc[0]
        diff, se = c2.delta_b_change - c1.delta_b_change, np.hypot(c2.delta_b_change_se, c1.delta_b_change_se)
        rd, rse = c2.ratio - c1.ratio, np.hypot(c2.ratio_se, c1.ratio_se)
        rows.append(dict(model=b, arm=arm[v], change_t2=c2.delta_b_change, change_t1=c1.delta_b_change, diff=diff,
                         diff_se=se, diff_p=2 * stats.norm.sf(abs(diff / se)),
                         same_direction=np.sign(c2.delta_b_change) == np.sign(c1.delta_b_change),
                         ratio_t2=c2.ratio, ratio_t1=c1.ratio, ratio_diff=rd, ratio_diff_se=rse,
                         ratio_diff_p=2 * stats.norm.sf(abs(rd / rse))))
    summ = []
    for tier_ in (2, 1):
        for b in bases:
            e = E[(E.tier == tier_) & (E.model == b)]
            summ.append(dict(tier=tier_, model=b, delta_b_min=e.delta_b.min(), delta_b_max=e.delta_b.max(),
                             delta_b_range=e.delta_b.max() - e.delta_b.min(),
                             shift_min=e.implied_shift.min(), shift_max=e.implied_shift.max(),
                             delta_b_all_negative=bool((e.delta_b < 0).all()),
                             delta_b_all_sig=bool((e.delta_b_p < 0.05).all()),
                             theta_min=e.theta.min(), theta_max=e.theta.max(),
                             theta_signs="".join("+" if t > 0 else "-" for t in e.theta),
                             theta_sig_pattern="".join("*" if q < 0.05 else "." for q in e.theta_p)))
    return {"P6": p6_table(df), "estimates": E, "changes": C, "wald_tests": pd.DataFrame(tests),
            "tier_compare": pd.DataFrame(rows), "summary": pd.DataFrame(summ)}


def verdict_parts(parts: list[bool]) -> str:
    return "supported" if all(parts) else ("not supported" if not any(parts) else "partly supported")


def predictions(df: pd.DataFrame) -> dict:
    """Verdicts for the exploratory predictions that the trial data can test (P1 needs the evidence
    diagnostic, P4 the saved activations; both are outside this dataset)."""
    pm = primary_all(df)
    ch = pm[pm.fmt == "chat"].set_index("model")
    preds = []
    if has(df, *OLMO7_STAGES[1:]):
        # P0: chat Delta_b grows SFT -> DPO -> Instruct; no completion-format stage significant after Holm
        co = pm[(pm.fmt == "completion") & pm.model.isin(OLMO7_STAGES)]
        grows = ch.delta_b["olmo_sft"] > ch.delta_b["olmo_dpo"] > ch.delta_b["olmo_instruct"]
        weak = bool((~co.holm_delta_b.astype(bool)).all())
        preds.append(dict(prediction="P0", verdict="supported" if grows and weak else "not supported",
                          evidence=f"chat Delta_b SFT {ch.delta_b['olmo_sft']:.3f}, DPO {ch.delta_b['olmo_dpo']:.3f}, "
                                   f"Instruct {ch.delta_b['olmo_instruct']:.3f}; completion significant after Holm: "
                                   f"{int(co.holm_delta_b.astype(bool).sum())} of {len(co)}"))
    big = pm[pm.model.isin(["olmo13_instruct", "phi4"]) & (pm.fmt == "chat")]
    if len(big):
        # P2: beta > 0.1 for at least one 13-14B instruct model
        preds.append(dict(prediction="P2", verdict="supported" if (big.beta_neutral > 0.1).any() else "not supported",
                          evidence="; ".join(f"{r.model} beta {r.beta_neutral:.3f}" for r in big.itertuples())))
    p3 = system_prompt(df).get("system_prompt")
    if p3 is not None:
        # P3: (a) Delta_b shrinks significantly, (b) stays significant, (c) |beta| < 0.1 with the prompt
        parts = [bool(((p3.delta_b_change > 0) & (p3.delta_b_change_p < 0.05)).all()),
                 bool((p3.delta_b_sp_p < 0.05).all()), bool((p3.beta_sp.abs() < 0.1).all())]
        preds.append(dict(prediction="P3", verdict=verdict_parts(parts),
                          evidence="; ".join(f"{r.model}: Delta_b {r.delta_b_original:.3f} -> {r.delta_b_sp:.3f} "
                                             f"(change {r.delta_b_change:+.3f}, p={r.delta_b_change_p:.3g}), beta "
                                             f"{r.beta_original:.3f} -> {r.beta_sp:.3f}" for r in p3.itertuples())))
    if has(df, *TULU_STAGES[1:]):
        # P5: the Tulu SFT -> DPO step in Delta_b is negative and significant (same trials)
        s1, s2 = stacked_variant_test(df, "tulu_sft", "tulu_dpo"), stacked_variant_test(df, "tulu_dpo", "tulu_final")
        jump = s1["delta_b_change"] < 0 and s1["delta_b_change_p"] < 0.05
        preds.append(dict(prediction="P5", verdict="supported" if jump else "not supported",
                          evidence=f"chat Delta_b SFT {ch.delta_b['tulu_sft']:.3f}, DPO {ch.delta_b['tulu_dpo']:.3f}, "
                                   f"final {ch.delta_b['tulu_final']:.3f}; SFT->DPO change {s1['delta_b_change']:+.3f} "
                                   f"(p={s1['delta_b_change_p']:.2g}); DPO->final change {s2['delta_b_change']:+.3f} "
                                   f"(p={s2['delta_b_change_p']:.2g})"))
    p6 = p6_table(df)
    if len(p6) and "same_sign_delta_b" in p6:
        # P6: signs of Delta_b and the support shift unchanged, Delta_b inside the original's 95% CI
        ok = p6.dropna(subset=["same_sign_delta_b"])
        keep = ok.same_sign_delta_b.astype(bool) & ok.same_sign_support.astype(bool)
        preds.append(dict(prediction="P6", verdict=verdict_parts([bool(keep.all()), bool(ok.within_ci.astype(bool).all())]),
                          evidence=f"{int(keep.sum())}/{len(ok)} variants keep both signs; "
                                   f"{int(ok.within_ci.astype(bool).sum())}/{len(ok)} Delta_b within the original's 95% CI"))
    return {"predictions": pd.DataFrame(preds)} if preds else {}


# ==================================================================================================
# Shape of the belief effect: local slopes by evidence strength
# ==================================================================================================
X_EDGES = [0.0005, 0.6, 1.4, 2.1, 3.0]           # |x| bins 0.32-0.49 | 0.81-1.29 | 1.61-2.03 | 2.26-2.91
X_BINS = ["0.32-0.49", "0.81-1.29", "1.61-2.03", "2.26-2.91"]


def local_slopes(df: pd.DataFrame) -> dict:
    """The primary model with its three evidence slopes (beta, Delta_b, s) varying by |x| bin; all other
    terms as in the primary model. Variant 'bin_intercepts' adds an intercept per bin. Also: the
    implied shift -Delta_b * mean|x| per bin, and Wald tests that Delta_b (or the shift) is equal
    across bins. Tier 2, chat. The x = 0 trials (null tables) are already outside the sample."""
    K = len(X_BINS)
    rows, tests = [], []
    for m in [m for m in HEADLINE_CHAT if has(df, m)]:
        d0 = frame(df, 2, m, "chat")
        n_x0 = int((d0.x.abs() <= X_EDGES[0]).sum())
        pooled = coef(fit_ols(d0), "x:contra")
        d = d0[d0.x.abs() > X_EDGES[0]].copy()
        d["xbin"] = pd.cut(d.x.abs(), X_EDGES, labels=False)
        for k in range(K):
            d[f"B{k}"] = (d.xbin == k).astype(float)
        mxs = [float(d.loc[d.xbin == k, "x"].abs().mean()) for k in range(K)]
        for spec in ("primary", "bin_intercepts"):
            # x is kept as the bin-0 slope so the control slopes keep their usual coding;
            # local beta_k = x + x:B_k for k >= 1
            terms = ("lp_before + x + " + " + ".join(f"x:B{k}" for k in range(1, K)) + " + "
                     + " + ".join(f"x:contra:B{k} + x:support:B{k}" for k in range(K))
                     + " + x:contra:m + x:support:m + x:dpos + x:C(cue_type) + x:C(source)"
                     + (" + C(xbin)" if spec == "bin_intercepts" else ""))
            fit = fit_ols(d, terms)
            for k in range(K):
                b = coef(fit, "x") if k == 0 else dict(zip(("est", "se"), lincomb(fit, {"x": 1, f"x:B{k}": 1})[:2]))
                db, s = coef(fit, f"x:contra:B{k}"), coef(fit, f"x:support:B{k}")
                rows.append(dict(model=m, spec=spec, bin=k, abs_x_range=X_BINS[k], n=int((d.xbin == k).sum()),
                                 beta=b["est"], beta_se=b["se"], delta_b=db["est"], delta_b_se=db["se"],
                                 delta_b_p=db["p"], s=s["est"], s_se=s["se"], mean_abs_x=mxs[k],
                                 implied_shift=db["est"] * mxs[k], implied_shift_se=db["se"] * mxs[k],
                                 share_removed=(-db["est"] / b["est"]) if b["est"] > 2 * b["se"] and b["est"] > 0.05 else np.nan,
                                 pooled_delta_b=pooled["est"], pooled_delta_b_se=pooled["se"], n_x0_dropped=n_x0))
            for label, h in (("Delta_b equal across bins",
                              ", ".join(f"x:contra:B{k} = x:contra:B{k+1}" for k in range(K - 1))),
                             ("Delta_b*|x| (constant shift) equal across bins",
                              ", ".join(f"{mxs[k]:.6f}*x:contra:B{k} = {mxs[k+1]:.6f}*x:contra:B{k+1}"
                                        for k in range(K - 1)))):
                w = fit.wald_test(h, use_f=True, scalar=True)
                tests.append(dict(model=m, spec=spec, test=label, F=float(w.statistic), p=float(w.pvalue), df_num=K - 1))
    # implied_shift is Delta_b * mean|x| (negative = toward the belief); the paper reports -implied_shift
    return {"local_slopes": pd.DataFrame(rows), "equality_tests": pd.DataFrame(tests)}


# ==================================================================================================
# Shape of the belief effect: additivity and anchoring (matched no-belief answers)
# ==================================================================================================
CLIP = 5.29   # |logit| of the 0.5% / 99.5% clipping


def additivity_anchoring(df: pd.DataFrame) -> dict:
    """Each belief trial is matched to the same model's no-belief trials on the same item (topic, slot,
    H1 wording: the table and x are fixed); N = their mean lp_after ('item'), or the mean over the same
    item and source ('src'). Signed toward the belief by b, never by belief_dir:
      s = b * x                 evidence position relative to the belief (s < 0 contradicts it)
      E = b * (lp_after - N)    the belief's total shift on the trial
    Check 1 (additivity): E = a + c*|x| on contradicting and on supporting trials. A constant shift
      gives c ~ 0; slope flattening gives c > 0 on contradicting trials only.
    Check 2 (anchoring): A = b*lp_after on Nb = b*N. Slope 1 = additive shift, 0 = a fixed answer
      whatever the item. N averages few trials, so the slope is divided by its reliability:
      variance components lambda (primary) or split-half Spearman-Brown (50 random splits).
    Chat format, both tiers, SEs clustered by topic. A stated belief treated as testimony also predicts
    a constant shift: check 1 separates a shift from flattening, not a social rule from testimony."""
    d = df[(~df.excluded) & (df.fmt == "chat") & df.model.isin(HEADLINE_CHAT) & (df.e != "null")].copy()
    d["s"], d["absx"] = d.b * d.x, d.x.abs()
    bel = d[d.b != 0]
    assert ((bel.s < 0) == (bel.kappa == 1)).all() and ((bel.s < 0) == (bel.e == "contra")).all()
    assert ((bel.s > 0) == (bel.e == "support")).all() and (bel.s != 0).all()
    assert (d.groupby(["model", "item_id"]).x.nunique() == 1).all(), "x not constant within item"
    nb = d[d.b == 0]
    for key, tag in ((["model", "item_id"], "item"), (["model", "item_id", "source"], "src")):
        g = nb.groupby(key).lp_after.agg(["mean", "size"]).rename(columns={"mean": f"N_{tag}", "size": f"k_{tag}"})
        d = d.merge(g, left_on=key, right_index=True, how="left")
    out = {}

    # Context: the no-belief evidence slope (where it is ~0, E cannot depend on the evidence)
    rows = []
    for (m, t), z in d[d.b == 0].groupby(["model", "tier"]):
        b1, b2 = clustered_ols("lp_after ~ x", z), clustered_ols("lp_after ~ lp_before + x", z)
        rows.append(dict(model=m, tier=t, readout=z.readout.iloc[0], beta_nobelief=b1.params["x"], se=b1.bse["x"],
                         beta_ctrl_prior=b2.params["x"], se_ctrl_prior=b2.bse["x"], n=len(z), n_topics=z.topic.nunique()))
    out["context_beta"] = pd.DataFrame(rows)

    bel = d[d.b != 0]
    out["matching_coverage"] = bel.groupby(["model", "tier"]).agg(
        n_belief=("b", "size"), frac_matched_item=("N_item", lambda v: v.notna().mean()),
        frac_matched_src=("N_src", lambda v: v.notna().mean()), median_k_item=("k_item", "median"),
        median_k_src=("k_src", "median"), frac_N_clip=("N_item", lambda v: (v.abs() >= CLIP).mean())).reset_index()

    # Check 1
    rows, joint = [], []
    for match in ("item", "src"):
        for (m, t), z in bel.groupby(["model", "tier"]):
            z = z[z[f"N_{match}"].notna()].copy()
            z["E"] = z.b * (z.lp_after - z[f"N_{match}"])
            z["contra"] = (z.s < 0).astype(int)
            z["support"] = 1 - z.contra
            for side, zz in (("contra", z[z.s < 0]), ("support", z[z.s > 0])):
                f, fm = clustered_ols("E ~ absx", zz), clustered_ols("E ~ 1", zz)
                rows.append(dict(model=m, tier=t, match=match, side=side, readout=zz.readout.iloc[0],
                                 a=f.params["Intercept"], se_a=f.bse["Intercept"], p_a=f.pvalues["Intercept"],
                                 c=f.params["absx"], se_c=f.bse["absx"], p_c=f.pvalues["absx"],
                                 meanE=fm.params["Intercept"], se_meanE=fm.bse["Intercept"],
                                 mean_absx=zz.absx.mean(), n=len(zz), n_topics=zz.topic.nunique()))
            # Joint fit: equal intercepts on both sides? In s-space the belief trials' slope is
            # beta - c_contra (s < 0) and beta + c_support (s > 0), so flattening specific to
            # contradicting evidence means c_contra + c_support > 0.
            f = clustered_ols("E ~ 0 + contra + support + contra:absx + support:absx", z)
            a, c = f.t_test("contra - support = 0"), f.t_test("contra:absx + support:absx = 0")
            joint.append(dict(model=m, tier=t, match=match, a_contra_minus_support=float(a.effect.item()),
                              se_a_diff=float(a.sd.item()), p_a_diff=float(a.pvalue),
                              c_contra_plus_support=float(c.effect.item()), se_c_sum=float(c.sd.item()),
                              p_c_sum=float(c.pvalue), flattening_contra=float(f.params["contra:absx"]),
                              flattening_support=-float(f.params["support:absx"]), n=len(z)))
    out["check1_additivity"], out["check1_joint"] = pd.DataFrame(rows), pd.DataFrame(joint)

    # Check 2
    rng = np.random.default_rng(15)

    def split_half(nb_: pd.DataFrame, reps: int = 50) -> tuple[float, float]:
        z = nb_.groupby("item_id").filter(lambda g: len(g) >= 2)
        if z.item_id.nunique() < 10:
            return np.nan, np.nan
        order = np.argsort(z.item_id.to_numpy(), kind="stable")
        items, y = z.item_id.to_numpy()[order], z.lp_after.to_numpy()[order]
        rs = []
        for _ in range(reps):
            tmp = pd.DataFrame(dict(i=items, y=y, u=rng.random(len(y))))
            tmp["half"] = (tmp.groupby("i").u.rank(method="first") > tmp.groupby("i").y.transform("size") / 2).astype(int)
            h = tmp.groupby(["i", "half"]).y.mean().unstack()
            rs.append(np.corrcoef(h[0], h[1])[0, 1])
        r = float(np.mean(rs))
        return r, 2 * r / (1 + r)

    rows = []
    for match in ("item", "src"):
        for (m, t), zall in d.groupby(["model", "tier"]):
            nb_ = zall[zall.b == 0]
            key = ["item_id"] if match == "item" else ["item_id", "source"]
            v = nb_.groupby(key).lp_after.var(ddof=1).dropna()
            sig2w = float(np.average(v, weights=nb_.groupby(key).size().loc[v.index] - 1))   # pooled within-item variance
            r_half, rel_sb = split_half(nb_ if match == "item" else nb_.assign(item_id=nb_.item_id + "|" + nb_.source))
            b_ = zall[(zall.b != 0) & zall[f"N_{match}"].notna()].copy()
            b_["A"], b_["Nb"], b_["k"] = b_.b * b_.lp_after, b_.b * b_[f"N_{match}"], b_[f"k_{match}"]
            for sample in ("all", "noclip"):
                bs = b_ if sample == "all" else b_[b_[f"N_{match}"].abs() < CLIP]
                for side, zz in (("all", bs), ("contra", bs[bs.s < 0]), ("support", bs[bs.s > 0])):
                    f = clustered_ols("A ~ Nb", zz)
                    sl, se = float(f.params["Nb"]), float(f.bse["Nb"])
                    lam = 1 - np.mean(sig2w / zz.k) / zz.Nb.var(ddof=1)
                    rows.append(dict(model=m, tier=t, match=match, sample=sample, side=side, readout=zz.readout.iloc[0],
                                     slope_raw=sl, se_raw=se, p_slope_eq_1=float(f.t_test("Nb = 1").pvalue),
                                     intercept=float(f.params["Intercept"]), se_intercept=float(f.bse["Intercept"]),
                                     r_splithalf=r_half, rel_sb=rel_sb, lambda_vc=lam,
                                     slope_disatt_vc=sl / lam, se_disatt_vc=se / lam,
                                     slope_disatt_sb=sl / rel_sb, se_disatt_sb=se / rel_sb,
                                     mean_A=zz.A.mean(), mean_Nb=zz.Nb.mean(), sd_A=zz.A.std(), sd_Nb=zz.Nb.std(),
                                     sd_ratio=zz.A.std() / zz.Nb.std(), corr_A_Nb=np.corrcoef(zz.A, zz.Nb)[0, 1],
                                     sigma2_within=sig2w, mean_k=zz.k.mean(), n=len(zz), n_topics=zz.topic.nunique()))
    out["check2_anchoring"] = pd.DataFrame(rows)
    return out


# ==================================================================================================
# Numeracy demands: heuristic-conflicting vs consistent items
# ==================================================================================================
HC_SLOPES = {"beta": "x", "delta_b": "x:contra", "s": "x:support", "theta_cong": "x:contra:m"}
HC_PRIMARY = ("lp_before + hc + x + x:hc + x:contra + x:contra:hc + x:support + x:support:hc"
              " + x:contra:m + x:contra:m:hc + x:support:m + x:support:m:hc + x:dpos + x:C(cue_type) + x:C(source)")
HC_FULL = HC_PRIMARY + " + x:dpos:hc + x:C(cue_type):hc + x:C(source):hc"


def hc_rows(fit, d: pd.DataFrame, model: str, tier_: int, spec: str) -> list[dict]:
    """Per stratum: each slope on consistent items, on conflicting items, and the difference (the :hc term)."""
    rows = []
    mx = {k: float(d.loc[(d.contra == 1) & (d.hc == k), "x"].abs().mean()) for k in (0.0, 1.0)}
    for par, t in HC_SLOPES.items():
        c0, c1, df_ = lincomb(fit, {t: 1}), lincomb(fit, {t: 1, f"{t}:hc": 1}), lincomb(fit, {f"{t}:hc": 1})
        rows.append(dict(param=par, consistent=c0[0], consistent_se=c0[1], conflicting=c1[0], conflicting_se=c1[1],
                         diff=df_[0], diff_se=df_[1], diff_p=df_[2]))
    # beta averaged over evidence direction (x:dpos is shared across strata unless interacted)
    p = {k: float(d.loc[d.hc == k, "dpos"].mean()) for k in (0.0, 1.0)}
    inter = {"x:dpos:hc": p[1.0]} if "x:dpos:hc" in fit.params else {}
    c0 = lincomb(fit, {"x": 1, "x:dpos": p[0.0]})
    c1 = lincomb(fit, {"x": 1, "x:hc": 1, "x:dpos": p[1.0], **inter})
    cd = lincomb(fit, {"x:hc": 1, "x:dpos": p[1.0] - p[0.0], **inter})
    rows.append(dict(param="beta_dir_avg", consistent=c0[0], consistent_se=c0[1], conflicting=c1[0],
                     conflicting_se=c1[1], diff=cd[0], diff_se=cd[1], diff_p=cd[2]))
    # implied shift toward the belief, -Delta_b * mean|x| over the stratum's contradicting trials
    s0 = lincomb(fit, {"x:contra": -mx[0.0]})
    s1 = lincomb(fit, {"x:contra": -mx[1.0], "x:contra:hc": -mx[1.0]})
    sd = lincomb(fit, {"x:contra": -(mx[1.0] - mx[0.0]), "x:contra:hc": -mx[1.0]})
    rows.append(dict(param="implied_shift", consistent=s0[0], consistent_se=s0[1], conflicting=s1[0],
                     conflicting_se=s1[1], diff=sd[0], diff_se=sd[1], diff_p=sd[2]))
    for r in rows:
        r.update(model=model, tier=tier_, spec=spec, n=len(d), n_conflicting=int((d.hc == 1).sum()),
                 n_topics=d.topic.nunique(), mean_abs_x_contra_consistent=mx[0.0],
                 mean_abs_x_contra_conflicting=mx[1.0])
    return rows


def heuristic_conflict(df: pd.DataFrame) -> dict:
    """A table is heuristic-conflicting when its raw success counts point opposite to its rates. Per
    model and tier (chat): the primary model with every evidence slope interacted with hc (consistent =
    reference). Robustness: controls also interacted ('controls_interacted'); separate fits per stratum
    ('separate_fits'). Count-following: on trials with evidence, the share of moves toward the side the
    rates favor vs the side the raw counts favor (on conflicting items these are opposite). Total shift: on
    contradicting trials, b * (logit P_after - mean no-belief logit P_after on the same item), by stratum, and
    its conflicting-minus-consistent difference (topic-clustered); it does not depend on how the prior is modelled."""
    rows, cf, ts = [], [], []
    for tier_ in (2, 1):
        for m in [m for m in CHAT10 if has(df, m)]:
            d = frame(df, tier_, m, "chat")
            if d.empty:
                continue
            d["hc"] = d.heuristic_conflict.astype(float)
            for spec, terms in (("primary", HC_PRIMARY), ("controls_interacted", HC_FULL)):
                rows += hc_rows(fit_ols(d, terms), d, m, tier_, spec)
            sep = {k: fit_ols(d[d.heuristic_conflict == k].reset_index(drop=True)) for k in (False, True)}
            for par, t in HC_SLOPES.items():
                a, b = sep[False], sep[True]
                rows.append(dict(param=par, consistent=a.params[t], consistent_se=a.bse[t], conflicting=b.params[t],
                                 conflicting_se=b.bse[t], diff=b.params[t] - a.params[t], diff_se=np.nan, diff_p=np.nan,
                                 model=m, tier=tier_, spec="separate_fits"))
            # total shift toward the belief on contradicting trials, by stratum
            nb = d[d.b == 0].groupby("item_id").lp_after.mean().rename("nb_after")
            c = d[(d.b != 0) & (d.e == "contra")].join(nb, on="item_id").dropna(subset=["nb_after"]).copy()
            if len(c) and c.hc.nunique() == 2:
                c["E"] = c.b * (c.lp_after - c.nb_after)
                f = clustered_ols("E ~ hc", c.reset_index(drop=True))
                ts.append(dict(model=m, tier=tier_, consistent=float(c.E[c.hc == 0].mean()),
                               conflicting=float(c.E[c.hc == 1].mean()), diff=f.params["hc"], diff_se=f.bse["hc"],
                               diff_z=f.tvalues["hc"], n=len(c), n_matched_share=float(len(c) / max(
                                   ((d.b != 0) & (d.e == "contra")).sum(), 1))))
            # count-following
            d = d[d.x != 0]
            cdir = np.sign(d.k_t - d.k_c) * d.h_dir       # direction the raw counts favor for H1
            mv = np.sign(d.y)
            for hc in (False, True):
                for e in ("neutral", "contra", "support"):
                    s = (d.heuristic_conflict == hc) & (d.e == e)
                    moved = s & (mv != 0)
                    cf.append(dict(model=m, tier=tier_, stratum="conflicting" if hc else "consistent", e=e,
                                   n=int(s.sum()), share_moved=float(moved.sum() / max(s.sum(), 1)),
                                   share_toward_rates=float((mv[moved] == d.d[moved]).mean()),
                                   share_toward_counts=float((mv[moved] == cdir[moved]).mean()),
                                   share_counts_tied=float((cdir[s] == 0).mean()),
                                   mean_y_signed_by_rates=float((d.y[s] * d.d[s]).mean())))
    cols = ["model", "tier", "spec", "param", "consistent", "consistent_se", "conflicting", "conflicting_se",
            "diff", "diff_se", "diff_p"]
    R = pd.DataFrame(rows)
    return {"stratified": R[cols + [c for c in R.columns if c not in cols]], "count_following": pd.DataFrame(cf),
            "total_shift": pd.DataFrame(ts)}


# ==================================================================================================
# Driver
# ==================================================================================================
FUNCS = {name: globals()[name] for name in ANALYSES}

# Columns printed for wide tables (every column is in the saved CSV)
SHOW = {
    ("primary", "per_model"): ["model", "fmt", "n", "beta_neutral", "rho", "delta_b", "delta_b_se", "delta_b_p",
                               "holm_delta_b", "theta_cong", "theta_se", "theta_p", "holm_theta", "support_shift"],
    ("mixed_model", "per_model"): ["model", "fmt", "n", "beta_neutral", "delta_b", "delta_b_se", "holm_delta_b",
                                   "theta_cong", "theta_se", "holm_theta"],
    ("training_stages", "ladders"): ["ladder", "fmt", "model", "n", "beta_neutral", "delta_b", "delta_b_se",
                                     "holm_delta_b", "delta_b_change_from_prev", "theta_cong"],
    ("training_stages", "scale_13_14b"): ["model", "fmt", "n", "beta_neutral", "delta_b", "theta_cong"],
    ("prompt_wording", "estimates"): ["tier", "model", "arm", "n", "delta_b", "delta_b_se", "theta", "theta_se",
                                      "theta_p", "implied_shift"],
    ("prompt_wording", "changes"): ["tier", "model", "arm", "delta_b_change", "delta_b_change_se", "delta_b_change_p",
                                    "theta_change", "theta_change_se", "ratio", "ratio_se", "delta_b_change_both"],
    ("local_slopes", "local_slopes"): ["model", "spec", "abs_x_range", "n", "beta", "beta_se", "delta_b",
                                       "delta_b_se", "delta_b_p", "implied_shift", "s"],
    ("additivity_anchoring", "check2_anchoring"): ["model", "tier", "match", "sample", "side", "slope_raw", "se_raw",
                                                   "lambda_vc", "slope_disatt_vc", "rel_sb", "mean_A", "n"],
    ("heuristic_conflict", "stratified"): ["model", "tier", "spec", "param", "consistent", "consistent_se",
                                           "conflicting", "conflicting_se", "diff", "diff_se", "diff_p"],
}
MAX_ROWS = 60


def show(analysis: str, name: str, tab: pd.DataFrame) -> None:
    cols = [c for c in SHOW.get((analysis, name), tab.columns) if c in tab.columns]
    print(f"\n== {analysis} / {name}  ({len(tab)} rows)")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 80):
        print(tab[cols].head(MAX_ROWS).round(4).to_string(index=False))
    if len(tab) > MAX_ROWS:
        print(f"   ... {len(tab) - MAX_ROWS} more rows (write them all with --save DIR)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="trial data (default: datasets/main_trials.csv.gz)")
    ap.add_argument("--only", help="comma-separated analyses to run (ignores the defaults)")
    ap.add_argument("--skip", help="comma-separated analyses to skip")
    ap.add_argument("--models", help="comma-separated model keys (default: every model in the data)")
    ap.add_argument("--save", type=Path, help="also write every result table as CSV to this folder")
    ap.add_argument("--list", action="store_true", help="list the analyses and exit")
    a = ap.parse_args()

    if a.list:
        src = Path(__file__).read_text().split("ANALYSES = {")[1].split("\n}")[0]
        print("analysis               default  what it reports")
        for line in src.strip().splitlines():
            name, rest = line.split(":", 1)
            on, desc = rest.split("#", 1)
            print(f"{name.strip().strip(chr(34)):22s} {on.strip().rstrip(','):7s}  {desc.strip()}")
        return 0

    run = {k for k, v in ANALYSES.items() if v}
    for opt, val in (("--only", a.only), ("--skip", a.skip)):
        names = set(val.split(",")) if val else set()
        if names - set(ANALYSES):
            ap.error(f"{opt}: unknown analyses {sorted(names - set(ANALYSES))}; see --list")
    if a.only:
        run = set(a.only.split(","))
    if a.skip:
        run -= set(a.skip.split(","))

    sys.stdout.reconfigure(line_buffering=True)      # show progress as it happens, also when redirected
    t0 = time.time()
    df = load(a.data)
    if a.models:
        models = a.models.split(",")
        missing = sorted(set(models) - set(df.model))
        if missing:
            ap.error(f"--models: not in the data: {missing}")
        df = df[df.model.isin(models)]
    print(f"read {len(df):,} trials, {df.model.nunique()} models ({time.time() - t0:.1f} s)")
    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)

    for name in ANALYSES:                      # in the order of the toggle block
        if name not in run:
            continue
        t = time.time()
        print(f"\n######## {name}")
        tables = {k: v for k, v in FUNCS[name](df).items() if len(v)}
        if not tables:
            print("   (not applicable to the selected models)")
        for tname, tab in tables.items():
            show(name, tname, tab)
            if a.save:
                tab.to_csv(a.save / f"{name}__{tname}.csv", index=False)
        print(f"-------- {name}: {time.time() - t:.1f} s")
    print(f"\ndone in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
