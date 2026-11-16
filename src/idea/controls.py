"""Instrument controls (Section 4.5): test-retest reliability, null-evidence drift,
P(H1)+P(H0) complement sums, ceiling-control x ranges, coverage/exclusions, readout
calibration, and the pilot omega histogram."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .design import x_range


def retest_reliability(df: pd.DataFrame, by=("model", "fmt")) -> pd.DataFrame:
    """Correlation of the two paraphrased priors in log-odds, over unique prior prompts."""
    d = df[~df.excluded].drop_duplicates(["model", "fmt", "cue_id", "topic", "h_dir", "b", "source", "q",
                                          "treated_first"])
    return d.groupby(list(by)).apply(
        lambda g: pd.Series({"r": g.lp_before.corr(g.lp_before_rep), "n": len(g),
                             "mean_abs_diff": (g.lp_before - g.lp_before_rep).abs().mean()}),
        include_groups=False).reset_index()


def null_evidence(df: pd.DataFrame, by=("model", "fmt", "tier", "e_null")) -> pd.DataFrame:
    """On D=0 tables the Bayesian update is zero. Mean |y| is the noise floor; mean signed y
    toward the stated belief (b * y) is drift the data cannot justify."""
    d = df[(df.D == 0) & ~df.excluded].copy()
    d["e_null"] = np.where(d.b == 0, "no_belief", "belief")
    d["toward_belief"] = d.b * d.y
    return d.groupby(list(by)).agg(n=("y", "size"), mean_abs_y=("y", lambda v: v.abs().mean()),
                                   mean_y=("y", "mean"), drift_toward_belief=("toward_belief", "mean")).reset_index()


def complement_sums(df: pd.DataFrame) -> pd.DataFrame:
    if "sum_before" not in df:
        return pd.DataFrame()
    d = df[df.complement & df.sum_before.notna()]
    return d.groupby(["model", "fmt"])[["sum_before", "sum_after"]].describe().round(3)


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "fmt", "tier", "cue_type"])
    return g.agg(n=("trial_id", "size"), excluded_frac=("excluded", "mean"),
                 cov_prior=("cov_prior", "mean"), cov_post=("cov_post", "mean"),
                 reprompt_frac=("att_post", "mean")).reset_index()


def prior_means(df: pd.DataFrame) -> dict:
    """Pilot cell means of the prior logit, keyed (tier, topic, h_dir, b, group), for
    `DesignConfig.prior_means`."""
    d = df[~df.excluded]
    m = d.groupby(["tier", "topic", "h_dir", "b", "group"]).lp_before.mean()
    return {tuple(k): float(v) for k, v in m.items()}


def realized_x_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Report the realized x span per cell (asymmetric spans weaken piecewise slopes)."""
    return df.groupby(["tier", "topic", "b", "group"]).x.agg(["min", "max", "size"]).reset_index()


def prior_distribution(df: pd.DataFrame) -> pd.DataFrame:
    d = df[~df.excluded]
    return d.groupby(["model", "tier", "e"]).p_prior.describe(percentiles=[0.05, 0.5, 0.95]).reset_index()


def readout_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration subset: the same trials read by full distribution and by K-sample
    sampling. Reports agreement in log-odds and the slope estimate both ways."""
    import statsmodels.formula.api as smf

    from .elicit import logit
    if not {"p_prior_s", "p_post_s", "calibration"} <= set(df.columns):
        return pd.DataFrame()   # no sampling-readout calibration data (only open models produce it)
    d = df[df.calibration.fillna(False).astype(bool) & df.p_prior_s.notna() & df.p_post_s.notna()
           & ~df.excluded].copy()
    if d.empty:
        return pd.DataFrame()
    d["lp_before_s"] = logit(d.p_prior_s.clip(0.005, 0.995))
    d["lp_after_s"] = logit(d.p_post_s.clip(0.005, 0.995))
    rows = []
    for (model, fmt), g in d.groupby(["model", "fmt"]):
        rec = dict(model=model, fmt=fmt, n=len(g), r_prior=g.lp_before.corr(g.lp_before_s),
                   r_post=g.lp_after.corr(g.lp_after_s))
        for tag, a, b in (("dist", "lp_after", "lp_before"), ("sample", "lp_after_s", "lp_before_s")):
            try:
                fit = smf.ols(f"{a} ~ {b} + x + x:contra", g).fit()
                rec[f"beta_{tag}"] = fit.params["x"]
                rec[f"delta_b_{tag}"] = fit.params.get("x:contra", np.nan)
            except Exception:
                pass
        rows.append(rec)
    return pd.DataFrame(rows)


def omega_histogram(df: pd.DataFrame, path=None, bins=np.linspace(-1, 2, 61)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[~df.excluded & (df.x.abs() > 0.5)]
    models = sorted(d.model.unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(7, 2.2 * len(models)), squeeze=False)
    for ax, m in zip(axes[:, 0], models):
        g = d[d.model == m]
        for e, col in (("neutral", "#4a7"), ("contra", "#c54"), ("support", "#46b")):
            ax.hist(g[g.e == e].omega.clip(bins[0], bins[-1]), bins=bins, alpha=0.5, label=e, color=col)
        ax.axvline(1, color="k", lw=0.8); ax.axvline(0.5, color="gray", ls=":"); ax.axvline(0.8, color="gray", ls=":")
        ax.set_title(m, fontsize=9); ax.legend(fontsize=7)
    axes[-1, 0].set_xlabel("omega = y / x (share of the Bayesian update)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def pilot_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "coverage": coverage_report(df),
        "retest": retest_reliability(df),
        "null_evidence": null_evidence(df),
        "complement_sums": complement_sums(df),
        "prior_distribution": prior_distribution(df),
        "readout_calibration": readout_calibration(df),
    }
