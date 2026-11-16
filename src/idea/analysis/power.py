"""Power by simulation (Section 4.7 notes). Plug in pilot residual SD, topic-slope SD and
the spread of m; choose the smallest per-group counts with power >= 0.8 for both
co-primary tests and acceptable delta_j interval widths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def simulate(n_contra=300, n_neutral=150, n_groups=22, delta_b=-0.25, theta=-0.15, delta_sd=0.05,
             sigma=1.0, topic_sd=0.1, m_sd=0.6, rho=0.9, beta=0.8, n_topics=24, x_lim=3.0, seed=0):
    rng = np.random.default_rng(seed)
    topic_slope = rng.normal(0, topic_sd, n_topics)
    group_delta = rng.normal(0, delta_sd, n_groups)
    rows = []
    for j in range(n_groups):
        for contra, n in ((1, n_contra), (0, n_neutral)):
            t = rng.integers(n_topics, size=n)
            x = rng.uniform(-x_lim, x_lim, n)
            m = rng.normal(0, m_sd, n) if contra else np.zeros(n)
            lp_b = rng.normal(0, 1, n)
            slope = beta + topic_slope[t] + contra * (delta_b + group_delta[j] + theta * m)
            lp_a = rho * lp_b + slope * x + rng.normal(0, sigma, n)
            rows.append(pd.DataFrame(dict(group=j, topic=t, contra=contra, m=m, x=x, lp_before=lp_b, lp_after=lp_a)))
    return pd.concat(rows, ignore_index=True)


def power(reps=200, alpha=0.05, **kw) -> dict:
    """Holm-corrected joint power for (delta_b, theta) and mean delta_j CI half-width."""
    hit_db = hit_th = 0
    hw = []
    for r in range(reps):
        d = simulate(seed=r, **kw)
        fit = smf.ols("lp_after ~ lp_before + x + x:contra + x:contra:m", d).fit(
            cov_type="cluster", cov_kwds={"groups": d.topic})
        p = sorted([("db", fit.pvalues["x:contra"]), ("th", fit.pvalues["x:contra:m"])], key=lambda t: t[1])
        sig = {}
        if p[0][1] < alpha / 2:
            sig[p[0][0]] = True
            sig[p[1][0]] = p[1][1] < alpha
        hit_db += sig.get("db", False)
        hit_th += sig.get("th", False)
        if r < 20:  # delta_j precision on a subset (costly)
            fg = smf.ols("lp_after ~ lp_before + x + x:C(group) + x:contra + x:contra:C(group) + x:contra:m", d).fit()
            se = fit_se = fg.bse.filter(like="x:contra:C(group)")
            hw.append(float(1.96 * se.mean()))
    return dict(power_delta_b=hit_db / reps, power_theta=hit_th / reps, delta_j_halfwidth=float(np.mean(hw)))


def grid(n_contra_list=(100, 200, 300, 400), reps=100, **kw) -> pd.DataFrame:
    rows = []
    for nc in n_contra_list:
        rows.append(dict(n_contra=nc, n_neutral=nc // 2, **power(reps=reps, n_contra=nc, n_neutral=nc // 2, **kw)))
    return pd.DataFrame(rows)


def pilot_inputs(df: pd.DataFrame, model: str) -> dict:
    """Estimate sigma, topic_sd and m_sd from pilot data for `simulate`."""
    from .rq1 import analysis_frame, fit_ols
    d = analysis_frame(df, 2, model)
    fit = fit_ols(d)
    sigma = float(np.sqrt(fit.scale))
    slopes = []
    for t, g in d[d.b == 0].groupby("topic"):
        if len(g) > 20:
            slopes.append(smf.ols("lp_after ~ lp_before + x", g).fit().params["x"])
    topic_sd = float(np.std(slopes)) if len(slopes) > 2 else 0.1
    m_sd = float(d[d.contra == 1].m.std()) if (d.contra == 1).any() else 0.6
    return dict(sigma=sigma, topic_sd=topic_sd, m_sd=m_sd, rho=float(fit.params["lp_before"]),
                beta=float(fit.params["x"]))
