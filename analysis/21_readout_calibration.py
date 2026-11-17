"""Step 21 (exploratory, reviewer request): does GPT-4o's readout bias beta or Delta_b?

    python analysis/21_readout_calibration.py                 # all results/readout_calibration/*.parquet
    python analysis/21_readout_calibration.py --simulate      # self-test on released data, planted effects
    python analysis/21_readout_calibration.py --gpt-only      # Part 0 only (GPT-4o internal checks)

GPT-4o's answers are read from a structured {"probability": k} reply, at the number token, over
the integers among the top-20 log-probabilities (renormalized). The open models are read from the
full next-token distribution over 0..100 after a plain prompt. scripts/readout_calibration.py
re-reads the SAME trials of an open model both ways, plus the two factors separately:

  A    full distribution, plain prompt (+ answer-prefix retry) - the open models' readout
  A20  A's forward pass truncated to the top-20 tokens        - truncation only   ("B" in the plan)
  A1   A's argmax integer                                     - greedy answer
  C    JSON reply prefilled to the number, full distribution  - format only
  C20  C truncated to the top 20                              - GPT-4o's pipeline on an open model

For each readout, Eq. 3 (common.BASE_TERMS) is fitted on the same trials (the released analysis
sample of that model, restricted to trials with a valid p under every compared readout), with
topic-clustered SEs. Differences against A come from one stacked regression in which every term
is interacted with a readout indicator, clustered by topic, so the SE accounts for the pairing.

Part 0 (no new runs): GPT-4o's own answers, split by how much probability its top-20 list held
(raw["top20_mass"] from the run cache runs/main-*/cache/gpt.sqlite, read-only; the only input here
that is not a released dataset), and re-read as the argmax integer.

Outputs (analysis/output/readout_calibration/): estimates.csv, differences.csv, coverage.csv,
gpt_internal.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import BASE_TERMS, ROOT, analysis_frame, estimate, fit_ols, formula, load_dataset, save

SUB = "readout_calibration"
IN_DIR = ROOT / "results" / "readout_calibration"
READOUTS = {"A": "full, plain prompt (open-model readout)",
            "A20": "B: top-20 truncation of A",
            "A10": "top-10 truncation of A", "A5": "top-5 truncation of A", "A1": "argmax of A",
            "C": "JSON reply, full distribution",
            "C20": "JSON reply, top-20 (GPT-4o pipeline)",
            "C10": "JSON, top-10", "C5": "JSON, top-5", "C1": "JSON, argmax"}
CORE = ["A", "A20", "C", "C20"]          # the common sample needs a valid p under these
CONTRASTS = [("A20", "A", "truncation (plain)"), ("C", "A", "format (full)"),
             ("C20", "C", "truncation (JSON)"), ("C20", "A", "GPT pipeline vs open-model readout"),
             ("A1", "A", "argmax vs mean"), ("A10", "A", "top-10"), ("A5", "A", "top-5")]
GPT_MASS_P75 = 0.714   # GPT-4o posterior top-20 integer mass, 75th percentile (Tier 2 analysis sample)
COEFS = {"beta": "x", "delta_b": "x:contra", "theta_cong": "x:contra:m"}


def lp(p):
    p = np.asarray(p, dtype=float).clip(0.005, 0.995)     # same clipping as the released lp_* columns
    return np.log(p / (1 - p))


def with_readout(d: pd.DataFrame, r: str) -> pd.DataFrame:
    out = d.copy()
    out["lp_before"] = lp(out[f"p_prior_{r}"])
    out["lp_after"] = lp(out[f"p_post_{r}"])
    return out


def paired_diff(d: pd.DataFrame, r1: str, r0: str) -> dict:
    """Stacked fit of readout r1 vs r0 on the same trials; coefficient differences with topic-clustered SEs."""
    s = pd.concat([with_readout(d, r0).assign(_r=0), with_readout(d, r1).assign(_r=1)], ignore_index=True)
    rhs = formula(BASE_TERMS, d).split("~", 1)[1]
    fit = smf.ols(f"lp_after ~ ({rhs}) * _r", s).fit(cov_type="cluster",
                                                      cov_kwds={"groups": s.topic.astype("category").cat.codes})
    out = {}
    for k, term in COEFS.items():
        e = estimate(fit, f"{term}:_r")
        out.update({f"d_{k}": e["est"], f"d_{k}_se": e["se"], f"d_{k}_p": e["p"]})
    return out


def merge_frame(cal: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    src = cal.trials_from.iloc[0]
    base = analysis_frame(df, model=src, fmt="chat")
    keep = ["trial_id"] + [c for c in cal.columns if c.startswith(("p_prior_", "p_post_", "cov_", "mass_",
                                                                   "nint_", "att_", "tried_", "argmax_"))
                           and not c.endswith("_released")]
    d = base.merge(cal[keep], on="trial_id", how="inner")
    ok = np.ones(len(d), bool)
    for r in CORE:
        ok &= d[f"p_prior_{r}"].notna().values & d[f"p_post_{r}"].notna().values
    print(f"{cal.model.iloc[0]}: {len(base)} released analysis trials ({src}), {len(d)} read out, "
          f"{int(ok.sum())} valid under {CORE}")
    return d[ok].reset_index(drop=True)


def analyse(name: str, d: pd.DataFrame, released: pd.DataFrame | None = None):
    est, dif, cov = [], [], []
    valid = {r: (d[f"p_prior_{r}"].notna() & d[f"p_post_{r}"].notna()).values
             for r in READOUTS if f"p_post_{r}" in d}   # all True for CORE; top-k/argmax can be empty
    readouts = [r for r in valid if valid[r].sum() >= 50]
    for r in readouts:
        dr = d[valid[r]].reset_index(drop=True)
        f = fit_ols(with_readout(dr, r))
        row = dict(model=name, readout=r, label=READOUTS[r], n=len(dr), topics=dr.topic.nunique())
        for k, term in COEFS.items():
            e = estimate(f, term)
            row.update({k: e["est"], f"{k}_se": e["se"]})
        est.append(row)
    for r1, r0, what in CONTRASTS:
        if r1 in readouts and r0 in readouts:
            dd = d[valid[r1] & valid[r0]].reset_index(drop=True)
            dif.append(dict(model=name, contrast=f"{r1} - {r0}", what=what, n=len(dd), **paired_diff(dd, r1, r0)))
    # GPT-like truncation severity: trials whose JSON top-20 integer mass is at or below GPT-4o's
    # posterior 75th percentile (0.714; Part 0). Conditional on a readout-derived variable, so
    # descriptive, but still a within-trial comparison.
    if "mass_post_C20" in d and {"C20", "A"} <= set(readouts):
        dd = d[(d.mass_post_C20 <= GPT_MASS_P75).values & valid["C20"] & valid["A"]].reset_index(drop=True)
        if len(dd) >= 200 and dd.contra.sum() >= 50:
            dif.append(dict(model=name, contrast="C20 - A", what=f"GPT-like truncation (C20 mass <= {GPT_MASS_P75})",
                            n=len(dd), **paired_diff(dd, "C20", "A")))
    for call in ("prior", "post"):
        row = dict(model=name, call=call)
        for t in ("A", "C"):
            for c in (f"cov_{call}_{t}", f"mass_{call}_{t}20", f"nint_{call}_{t}20"):
                if c in d:
                    row[f"median_{c.replace(f'_{call}', '')}"] = float(d[c].median())
        if f"mass_{call}_C20" in d:
            row["share_C20_mass_lt_0.9"] = float((d[f"mass_{call}_C20"] < 0.9).mean())
        for r in ("A20", "C", "C20", "A1"):
            if f"p_{call}_{r}" in d:
                row[f"mean_abs_dp_{r}_vs_A"] = float(np.mean(np.abs(d[f"p_{call}_{r}"] - d[f"p_{call}_A"])))
        if released is not None and f"p_{call}" in released:
            ok = released[f"p_{call}"].notna().values
            row["corr_A_vs_released"] = float(np.corrcoef(d[f"p_{call}_A"][ok], released[f"p_{call}"][ok])[0, 1])
            row["mean_abs_A_vs_released"] = float(np.mean(np.abs(d[f"p_{call}_A"].values[ok] - released[f"p_{call}"].values[ok])))
        cov.append(row)
    return pd.DataFrame(est), pd.DataFrame(dif), pd.DataFrame(cov)


# --------------------------------------------------------------------------------------
# Part 0: GPT-4o internal checks (no new runs)
# --------------------------------------------------------------------------------------

def gpt_internal(df: pd.DataFrame) -> pd.DataFrame | None:
    caches = sorted(glob.glob(str(ROOT / "runs" / "main-*" / "cache" / "gpt.sqlite")))
    resp = sorted(glob.glob(str(ROOT / "runs" / "main-*" / "responses" / "gpt_chat*.parquet")))
    if not caches or not resp:
        print("Part 0 skipped: no GPT-4o run cache under runs/main-*/")
        return None
    raw = {}
    for c in caches:
        con = sqlite3.connect(f"file:{c}?mode=ro", uri=True)
        raw.update({k: json.loads(v).get("raw") or {} for k, v in con.execute("SELECT key, result FROM results")})
    r = pd.concat([pd.read_parquet(f, columns=["trial_id", "key_prior", "key_post"]) for f in resp])
    for call in ("prior", "post"):
        r[f"mass_{call}"] = r[f"key_{call}"].map(lambda k: raw.get(k, {}).get("top20_mass"))
        r[f"ans_{call}"] = r[f"key_{call}"].map(lambda k: raw.get(k, {}).get("answer"))
    d = analysis_frame(df, model="gpt", fmt="chat").merge(r.drop(columns=["key_prior", "key_post"]),
                                                          on="trial_id", how="left")
    d = d[d.mass_post.notna() & d.mass_prior.notna()].reset_index(drop=True)
    rows = []

    def add(label, dd, note=""):
        f = fit_ols(dd)
        row = dict(sample=label, n=len(dd), note=note,
                   mass_post_median=float(dd.mass_post.median()), mass_prior_median=float(dd.mass_prior.median()))
        for k, term in COEFS.items():
            e = estimate(f, term)
            row.update({k: e["est"], f"{k}_se": e["se"]})
        rows.append(row)

    add("all (released readout)", d)
    d["tert"] = pd.qcut(d.mass_post, 3, labels=["low", "mid", "high"])
    for t, g in d.groupby("tert", observed=True):
        add(f"posterior top-20 mass tercile: {t}", g.reset_index(drop=True),
            f"mass {g.mass_post.min():.2f}-{g.mass_post.max():.2f}")
    dd = d.copy()
    dd["lp_before"] = lp((dd.ans_prior + 0.5) / 101)
    dd["lp_after"] = lp((dd.ans_post + 0.5) / 101)
    add("argmax readout (the integer GPT-4o wrote)", dd)
    # moderation: does the slope or the discount change with top-20 mass (standardized)?
    d["c"] = (d.mass_post - d.mass_post.mean()) / d.mass_post.std()
    f = fit_ols(d, BASE_TERMS + " + c + x:c + x:contra:c")
    for term in ("x:c", "x:contra:c"):
        e = estimate(f, term)
        rows.append(dict(sample=f"moderation {term} (per SD of top-20 mass)", n=len(d),
                         note="endogenous: spread answers are also uncertain answers",
                         beta=e["est"] if term == "x:c" else np.nan, beta_se=e["se"] if term == "x:c" else np.nan,
                         delta_b=e["est"] if term != "x:c" else np.nan,
                         delta_b_se=e["se"] if term != "x:c" else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Self-test with planted readout effects (released data; no model)
# --------------------------------------------------------------------------------------

def simulate(df: pd.DataFrame, model: str = "olmo_instruct") -> None:
    """A = the released answers; A20 = A; C = A with +0.05 * x * contra added to the posterior
    log-odds (so Delta_b must move by exactly +0.05); C20 = C with independent noise on both answers."""
    d = analysis_frame(df, model=model, fmt="chat")
    rng = np.random.default_rng(0)
    cal = pd.DataFrame({"trial_id": d.trial_id, "model": model + "_sim", "trials_from": model})
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    lpb, lpa = lp(d.p_prior), lp(d.p_post)
    for r, (b, a) in {"A": (lpb, lpa), "A20": (lpb, lpa),
                      "C": (lpb, lpa + 0.05 * d.x * d.contra),
                      "C20": (lpb + rng.normal(0, .3, len(d)), lpa + 0.05 * d.x * d.contra + rng.normal(0, .3, len(d)))}.items():
        cal[f"p_prior_{r}"], cal[f"p_post_{r}"] = sig(b), sig(a)
    m = merge_frame(cal, df)
    est, dif, _ = analyse(model + "_sim", m)
    print(est[["readout", "beta", "beta_se", "delta_b", "delta_b_se"]].round(4).to_string(index=False))
    print(dif[["contrast", "d_beta", "d_beta_se", "d_delta_b", "d_delta_b_se"]].round(4).to_string(index=False))
    di = dif.set_index("contrast")
    assert abs(di.at["A20 - A", "d_delta_b"]) < 1e-9 and abs(di.at["A20 - A", "d_beta"]) < 1e-9
    assert abs(di.at["C - A", "d_delta_b"] - 0.05) < 1e-6, di.at["C - A", "d_delta_b"]
    assert abs(di.at["C - A", "d_beta"]) < 1e-6
    e = est.set_index("readout")
    assert abs(e.at["C", "delta_b"] - e.at["A", "delta_b"] - 0.05) < 1e-6
    print("self-test passed: identical readouts give 0 difference; the planted +0.05 shift in Delta_b is recovered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--gpt-only", action="store_true")
    ap.add_argument("--inputs", nargs="*", help="calibration parquets (default results/readout_calibration/*.parquet)")
    a = ap.parse_args()
    df = load_dataset("main")
    if a.simulate:
        simulate(df)
        return
    g = gpt_internal(df)
    if g is not None:
        save(g, SUB, "gpt_internal")
    if a.gpt_only:
        return
    files = a.inputs or sorted(f for f in glob.glob(str(IN_DIR / "*.parquet")) if not f.endswith(".prompts.parquet"))
    if not files:
        print(f"no calibration outputs in {IN_DIR}; run scripts/readout_calibration.py first")
        return
    cal = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cal["model"] = cal.model.str.replace(r"_s\d+of\d+$", "", regex=True)
    E, D, C = [], [], []
    for name, c in cal.groupby("model"):
        c = c.drop_duplicates("trial_id", keep="last")
        d = merge_frame(c, df)
        if len(d) < 50:
            print(f"{name}: only {len(d)} trials; skipping the regressions")
            continue
        rel = d[["trial_id"]].merge(c[["trial_id", "p_prior_released", "p_post_released"]], on="trial_id", how="left")
        rel = rel.rename(columns={"p_prior_released": "p_prior", "p_post_released": "p_post"})
        e, di, co = analyse(name, d, rel)
        E.append(e); D.append(di); C.append(co)
    if E:
        save(pd.concat(E), SUB, "estimates")
        save(pd.concat(D), SUB, "differences")
        save(pd.concat(C), SUB, "coverage")


if __name__ == "__main__":
    main()
