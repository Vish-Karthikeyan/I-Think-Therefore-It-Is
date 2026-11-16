"""Step 4: preregistered robustness checks for RQ1 (main run).

    python analysis/04_robustness.py [--mixed]

Each check refits the primary model (02_rq1_primary.py) with one thing changed:

  1. congruence_svy     m_svy = b * g_svy: identity stance from Pew survey gaps instead of the
                        model's own judgment. Topics without a matching Pew item get m_svy = 0.
  2. congruence_beh     m_beh = b * g_beh: identity stance from the model's own BEFORE answers
                        (how far a group's no-belief prior sits from Unspecified's).
  3. svy_weak_dropped   check 1 without the two Tier 2 topics whose Pew item is the weakest proxy
                        (religious_exemptions, concealed_carry). Two other weak proxies,
                        faith_reentry and public_transit, are not Tier 2 topics in the main run.
  4. completion_cov08   completion format with the stricter 0.8 coverage cutoff (primary: 0.5).
                        Chat-format results cannot change (cutoff 0.9 there either way).
  5. tier1_low_g        Tier 1 restricted to (topic, cue) pairs whose mean |g| < 0.25 in the pilot
                        (the original Tier 1 screen, applied as a subset; see the protocol
                        changes). Tier 2 is unchanged, so this affects only the Tier 1 and
                        Tier 2 - Tier 1 contrasts and delta_j on Tier 1.
  6. mixed (optional)   random item intercepts and topic slopes instead of clustered OLS
                        (--mixed; takes several minutes per model).

Outputs: analysis/output/robustness/*.csv, plus robustness_summary.csv that puts Delta_b and
theta_cong from every check next to the primary estimate.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import load_dataset, model_formats, save

PRIMARY = __import__("02_rq1_primary")
SECONDARY = __import__("03_rq1_secondary")
WEAK_PROXY_TOPICS = ["religious_exemptions", "concealed_carry", "faith_reentry", "public_transit"]


def tier1_subset_contrasts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tier contrast and Tier 1 delta_j with Tier 1 limited to low-|g| (topic, cue) pairs."""
    sub = df[(df.tier == 2) | df.tier1_low_g]
    tier, dj1 = [], []
    for model, fmt in model_formats(sub):
        c = SECONDARY.contrasts(sub, model, fmt)
        if "tier" in c:
            tier.append(c["tier"].assign(model=model, fmt=fmt))
        if "delta_j_tier1" in c:
            dj1.append(c["delta_j_tier1"].assign(model=model, fmt=fmt))
    return pd.concat(tier, ignore_index=True), pd.concat(dj1, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", action="store_true", help="also fit the mixed-effects engine (slow)")
    a = ap.parse_args()
    df = load_dataset("main")
    df = df[df.study == "main"]

    checks = {
        "primary": PRIMARY.per_model(df),
        "congruence_svy": PRIMARY.per_model(df, m_col="m_svy"),
        "congruence_beh": PRIMARY.per_model(df, m_col="m_beh"),
        "svy_weak_dropped": PRIMARY.per_model(df[~df.topic.isin(WEAK_PROXY_TOPICS)], m_col="m_svy"),
        "completion_cov08": PRIMARY.per_model(df[df.fmt == "completion"], excluded_col="excluded_cov08"),
    }
    if a.mixed:
        checks["mixed"] = PRIMARY.per_model(df, engine="mixed")
    for name, tab in checks.items():
        if name != "primary":
            save(tab, "robustness", name, show=False)

    tier, dj1 = tier1_subset_contrasts(df)
    save(tier, "robustness", "tier1_low_g_tier_contrast")
    save(dj1, "robustness", "tier1_low_g_delta_j_tier1", show=False)

    # One table: every check's Delta_b and theta_cong next to the primary estimate
    keep = ["model", "fmt", "n", "delta_b", "delta_b_se", "holm_delta_b", "theta_cong", "theta_se", "holm_theta"]
    rows = [tab.reindex(columns=keep).assign(check=name) for name, tab in checks.items()]
    summ = pd.concat(rows, ignore_index=True)[["check"] + keep]
    save(summ.sort_values(["model", "fmt", "check"]), "robustness", "robustness_summary")


if __name__ == "__main__":
    main()
