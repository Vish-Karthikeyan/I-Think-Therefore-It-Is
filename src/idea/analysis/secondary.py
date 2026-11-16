"""Preregistered secondary specifications (Section 4.7):
1. spline in x instead of a single slope (saturation on strong evidence);
2. two-component mixture: each trial updates or ignores the evidence, with identity and
   belief predicting P(ignore) (EM);
3. latent-prior measurement model: the two prior replicates as noisy measures of an
   unobserved prior (errors-in-variables via the replicate as an instrument, 2SLS).
All three are reported regardless of the primary results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .rq1 import analysis_frame


def spline_fit(d: pd.DataFrame, df_spline: int = 4):
    """Separate spline-in-x response curves for neutral / contra / support trials. Reports
    the fitted response at x in a grid for each evidence type, and the overall F-test of
    the contra-specific curve against a linear contra slope."""
    d = d.copy()
    f_lin = "lp_after ~ lp_before + x + x:contra + x:support + x:contra:m"
    f_spl = (f"lp_after ~ lp_before + bs(x, df={df_spline}) + bs(x, df={df_spline}):contra"
             f" + bs(x, df={df_spline}):support + x:contra:m")
    lin = smf.ols(f_lin, d).fit()
    spl = smf.ols(f_spl, d).fit()
    grid = np.linspace(d.x.min(), d.x.max(), 25)
    rows = []
    for e in ("neutral", "contra", "support"):
        nd = pd.DataFrame(dict(x=grid, lp_before=0.0, contra=int(e == "contra"), support=int(e == "support"), m=0.0))
        rows.append(pd.DataFrame(dict(e=e, x=grid, response=spl.predict(nd))))
    f_test = spl.compare_f_test(lin) if spl.df_model > lin.df_model else (np.nan, np.nan, 0)
    return dict(curves=pd.concat(rows, ignore_index=True), f_stat=f_test[0], f_p=f_test[1], fit=spl)


def mixture_em(d: pd.DataFrame, covariates: str = "contra + support + contra:m + C(dimension)",
               n_iter: int = 200, tol: float = 1e-6, seed: int = 0) -> dict:
    """Two components:
        updated:  lp_after = a1 + r1*lp_before + b1*x + e,  e ~ N(0, s1)
        ignored:  lp_after = a0 + r0*lp_before      + e,  e ~ N(0, s0)
    P(ignored) = logistic(Z gamma) with Z from `covariates`. EM with weighted least squares
    for the components and a weighted logistic GLM for the mixing model."""
    import patsy

    rng = np.random.default_rng(seed)
    y = d.lp_after.values
    X1 = np.column_stack([np.ones(len(d)), d.lp_before.values, d.x.values])
    X0 = X1[:, :2]
    Z = np.asarray(patsy.dmatrix(covariates, d, return_type="dataframe"))
    znames = patsy.dmatrix(covariates, d, return_type="dataframe").columns.tolist()
    w = np.clip(rng.uniform(0.2, 0.8, len(d)), 1e-3, 1 - 1e-3)  # P(ignored | data)
    ll_old = -np.inf
    from scipy.stats import norm
    for it in range(n_iter):
        b1 = np.linalg.lstsq(X1 * np.sqrt(1 - w)[:, None], y * np.sqrt(1 - w), rcond=None)[0]
        b0 = np.linalg.lstsq(X0 * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        s1 = np.sqrt(np.sum((1 - w) * (y - X1 @ b1) ** 2) / np.sum(1 - w))
        s0 = np.sqrt(np.sum(w * (y - X0 @ b0) ** 2) / np.sum(w))
        # fractional-response logistic regression of the E-step weights on Z
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gam = sm.GLM(w, Z, family=sm.families.Binomial()).fit().params
        pi = 1 / (1 + np.exp(-(Z @ gam)))
        f1 = norm.pdf(y, X1 @ b1, s1) * (1 - pi)
        f0 = norm.pdf(y, X0 @ b0, s0) * pi
        tot = f1 + f0 + 1e-300
        ll = np.sum(np.log(tot))
        w = np.clip(f0 / tot, 1e-6, 1 - 1e-6)
        if abs(ll - ll_old) < tol * (1 + abs(ll)):
            break
        ll_old = ll
    return dict(updated=dict(alpha=b1[0], rho=b1[1], beta=b1[2], sigma=s1),
                ignored=dict(alpha=b0[0], rho=b0[1], sigma=s0),
                p_ignore_coefs=pd.Series(gam, index=znames), loglik=ll, iterations=it + 1,
                mean_p_ignore=float(pi.mean()), posterior_ignore=w)


def latent_prior_iv(d: pd.DataFrame, terms: str = "x + x:contra + x:support + x:contra:m") -> dict:
    """lp_before = true prior + error; lp_before_rep is an independent measurement of the
    same prior, so it is a valid instrument. 2SLS gives rho corrected for measurement error
    and slopes conditional on the latent prior."""
    import patsy

    X_ex = patsy.dmatrix(terms, d, return_type="dataframe")  # exogenous incl. intercept
    endog = d.lp_before.values
    inst = d.lp_before_rep.values
    Zfull = np.column_stack([X_ex.values, inst])
    # first stage
    g = np.linalg.lstsq(Zfull, endog, rcond=None)[0]
    endog_hat = Zfull @ g
    X2 = np.column_stack([X_ex.values, endog_hat])
    beta = np.linalg.lstsq(X2, d.lp_after.values, rcond=None)[0]
    # correct residuals use the actual endogenous regressor
    X_act = np.column_stack([X_ex.values, endog])
    resid = d.lp_after.values - X_act @ beta
    sigma2 = resid @ resid / (len(d) - X2.shape[1])
    cov = sigma2 * np.linalg.inv(X2.T @ X2)
    names = list(X_ex.columns) + ["lp_before(latent)"]
    first_stage_r2 = 1 - np.sum((endog - endog_hat) ** 2) / np.sum((endog - endog.mean()) ** 2)
    return dict(params=pd.Series(beta, names), se=pd.Series(np.sqrt(np.diag(cov)), names),
                first_stage_r2=first_stage_r2)


def run_secondary(df: pd.DataFrame, model: str, fmt: str = "chat") -> dict:
    d = analysis_frame(df[df.fmt == fmt], 2, model)
    out = {}
    try:
        out["spline"] = spline_fit(d)
    except Exception as e:  # noqa: BLE001
        out["spline_error"] = repr(e)
    nz = d[d.x.abs() > 0.05]
    out["mixture"] = mixture_em(nz)
    out["latent_prior"] = latent_prior_iv(d)
    return out
