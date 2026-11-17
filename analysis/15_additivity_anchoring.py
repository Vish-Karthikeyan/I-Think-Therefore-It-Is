"""15 - Additivity and anchoring checks: is the belief effect a fixed shift, or a flattened slope?

Two CPU-only checks on existing data (`datasets/main_trials.csv.gz`), chat format, models
gpt, claude, gemini, phi4, llama, olmo_instruct, olmo13_instruct, tulu_final; tier 2 (social
topics, primary) and tier 1 (control) reported separately.

Sign convention (everything is signed toward the user's belief by `b`, never by `belief_dir`):
  lp_after   logit P(H1) after the evidence
  x          Bayesian log-odds update for H1
  b          +1 user believes H1, -1 believes H0, 0 no belief
  s = b*x    evidence position relative to the belief (s > 0 supports, s < 0 contradicts);
             checked against `e` / `kappa` below.

Matching. The evidence table (and hence x) is constant within `item_id` (= topic x base_slot x
H1 wording h_dir); `base_slot` alone is only a slot index within a topic, and x is NOT constant
within topic x base_slot because the two H1 wordings flip its sign. So the matching unit is
`item_id`. Counterfactual N = mean lp_after of the no-belief (b = 0) trials of the same item
(primary), or of the same item and same source (lab/tank; robustness). Belief effect
E = b*(lp_after - N).

Check 1 (additivity). Separately for contradicting (s < 0) and supporting (s > 0) belief trials:
  E = a + c*|x|        (OLS, SEs clustered by topic)
  category shift      -> a > 0, c ~ 0, similar a on both sides
  slope flattening    -> on s < 0: a ~ 0, c > 0 (effect grows with the evidence it resists);
                         on s > 0: E ~ 0
  A joint fit E ~ side + side:|x| tests a_contra = a_support, and c_contra + c_support = 0
  (in s-space the belief-trial evidence slope is beta - c_contra for s<0 and beta + c_support
  for s>0, so contradicting-specific flattening means c_contra + c_support > 0; equal and
  opposite c's mean the slope flattens symmetrically on both sides).
  NOTE: a model that treats the stated belief as testimony (one more piece of evidence) also
  predicts a constant shift, so this check separates "shift" from "slope flattening"; it does not
  separate a social rule from rational testimony.

Check 2 (anchoring). In belief-signed space, A = b*lp_after (belief trial) on Nb = b*N:
  slope 1 -> additive shift on top of what the model would otherwise say
  slope 0 -> a fixed belief-side anchor regardless of the baseline
  N averages only a few no-belief trials, so the slope is attenuated. Two reliability estimates:
   (i)  split-half: items with >= 2 no-belief trials, random halves, Pearson r of half-means
        across items, Spearman-Brown to full length 2r/(1+r) (median 2 trials per item, so this
        is roughly the reliability of a 2-trial mean; averaged over 50 random splits);
   (ii) variance components (used as primary for disattenuation, because it allows for the
        variable number of no-belief trials k_i behind each N):
        lambda = 1 - mean_i(sigma2_within / k_i) / var(Nb), sigma2_within pooled within items.
  Disattenuated slope = raw / reliability (its SE is raw SE / reliability, ignoring the
  uncertainty in the reliability). Also reported: anchor level mean(A), sd(A)/sd(Nb), and a
  robustness fit excluding |N| >= 5.29 (the 0.5%/99.5% clip is at |logit| = 5.29).

Readout: claude and gemini use readout == "sample" (one sampled answer per question, much
noisier; claude has only 6,000 trials); the others read the full next-token distribution.

Context: the no-belief evidence slope beta (lp_after on x, b = 0 trials) is printed per model;
where beta ~ 0, E on contradicting trials is mechanically evidence-independent (there is no slope
to flatten), so open and frontier models must be read separately.

Outputs: analysis/output/additivity/*.csv, NOTES.txt, and the figure
../paper/secondary/figures/candidates/F16_additivity_anchoring.{pdf,png} (tier 2), plus a tier-1
version of the figure in analysis/output/additivity/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_dataset, out_dir, save  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUB = "additivity"
MODELS = ["gpt", "claude", "gemini", "phi4", "llama", "olmo_instruct", "olmo13_instruct", "tulu_final"]
FRONTIER = {"gpt", "claude", "gemini", "phi4"}
LABEL = dict(gpt="GPT-4o", claude="Claude Sonnet 5", gemini="Gemini 2.5 Flash", phi4="Phi-4",
             llama="Llama 3.1 8B Instruct", olmo_instruct="OLMo 2 7B Instruct",
             olmo13_instruct="OLMo 2 13B Instruct", tulu_final="Tulu 3 8B")
CLIP = 5.29
FIG_DIR = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"
# Okabe-Ito
OI_BLUE, OI_ORANGE, OI_GREY = "#0072B2", "#E69F00", "#999999"
RNG = np.random.default_rng(15)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def build() -> pd.DataFrame:
    df = load_dataset("main")
    d = df[(~df.excluded) & (df.fmt == "chat") & df.model.isin(MODELS) & (df.e != "null")].copy()
    d["s"] = d.b * d.x
    d["absx"] = d.x.abs()

    # Sign checks: s must agree with e / kappa; x constant within item_id.
    bel = d[d.b != 0]
    assert ((bel.s < 0) == (bel.kappa == 1)).all(), "s<0 does not match kappa"
    assert ((bel.s < 0) == (bel.e == "contra")).all() and ((bel.s > 0) == (bel.e == "support")).all()
    assert (bel.s != 0).all()
    assert (d.groupby(["model", "item_id"]).x.nunique() == 1).all(), "x not constant within item"
    assert (d[d.b == 0].e == "neutral").all()

    nb = d[d.b == 0]
    for key, tag in [(["model", "item_id"], "item"), (["model", "item_id", "source"], "src")]:
        g = nb.groupby(key).lp_after.agg(["mean", "size"]).rename(
            columns={"mean": f"N_{tag}", "size": f"k_{tag}"})
        d = d.merge(g, left_on=key, right_index=True, how="left")
    return d


def sign_check_table(d: pd.DataFrame) -> pd.DataFrame:
    bel = d[d.b != 0]
    t = pd.crosstab([bel.model, bel.e], np.sign(bel.s)).reset_index()
    t.columns = [str(c) for c in t.columns]
    return t


def clustered(formula: str, data: pd.DataFrame):
    g = data.topic.astype("category").cat.codes
    return smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": g})


def coef(fit, name):
    return float(fit.params[name]), float(fit.bse[name]), float(fit.pvalues[name])


# --------------------------------------------------------------------------------------
# Context: no-belief evidence slope
# --------------------------------------------------------------------------------------
def context_beta(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, t), z in d[d.b == 0].groupby(["model", "tier"]):
        f = clustered("lp_after ~ x", z)
        b, se, p = coef(f, "x")
        f2 = clustered("lp_after ~ lp_before + x", z)
        b2, se2, _ = coef(f2, "x")
        rows.append(dict(model=m, tier=t, readout=z.readout.iloc[0], beta_nobelief=b, se=se,
                         beta_ctrl_prior=b2, se_ctrl_prior=se2, n=len(z), n_topics=z.topic.nunique()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Check 1: additivity
# --------------------------------------------------------------------------------------
def check1(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, joint = [], []
    for match in ["item", "src"]:
        for (m, t), z in d[d.b != 0].groupby(["model", "tier"]):
            z = z[z[f"N_{match}"].notna()].copy()
            z["E"] = z.b * (z.lp_after - z[f"N_{match}"])
            z["contra"] = (z.s < 0).astype(int)
            z["support"] = 1 - z.contra
            for side, zz in [("contra", z[z.s < 0]), ("support", z[z.s > 0])]:
                f = clustered("E ~ absx", zz)
                a, sa, pa = coef(f, "Intercept")
                c, sc, pc = coef(f, "absx")
                fm = clustered("E ~ 1", zz)
                rows.append(dict(model=m, tier=t, match=match, side=side, readout=zz.readout.iloc[0],
                                 a=a, se_a=sa, p_a=pa, c=c, se_c=sc, p_c=pc,
                                 meanE=float(fm.params["Intercept"]), se_meanE=float(fm.bse["Intercept"]),
                                 mean_absx=zz.absx.mean(), n=len(zz), n_topics=zz.topic.nunique()))
            # joint: E ~ 0 + contra + support + contra:absx + support:absx
            f = clustered("E ~ 0 + contra + support + contra:absx + support:absx", z)
            dif = f.t_test("contra - support = 0")
            # In s-space the belief trials' evidence slope is beta - c_contra on s<0 (|x| = -s)
            # and beta + c_support on s>0. Flattening specific to contradicting evidence means
            # c_contra + c_support > 0; symmetric flattening (both slopes shrink) gives
            # c_contra ~ -c_support, i.e. a sum ~ 0.
            difc = f.t_test("contra:absx + support:absx = 0")
            joint.append(dict(model=m, tier=t, match=match,
                              a_contra_minus_support=float(dif.effect.item()), se_a_diff=float(dif.sd.item()),
                              p_a_diff=float(dif.pvalue),
                              c_contra_plus_support=float(difc.effect.item()),
                              se_c_sum=float(difc.sd.item()), p_c_sum=float(difc.pvalue),
                              flattening_contra=float(f.params["contra:absx"]),
                              flattening_support=-float(f.params["support:absx"]), n=len(z)))
    return pd.DataFrame(rows), pd.DataFrame(joint)


# --------------------------------------------------------------------------------------
# Check 2: anchoring
# --------------------------------------------------------------------------------------
def split_half_rel(nb: pd.DataFrame, reps: int = 50) -> tuple[float, float]:
    """Split-half reliability of the item mean of no-belief lp_after (Spearman-Brown)."""
    z = nb.groupby("item_id").filter(lambda g: len(g) >= 2)
    if z.item_id.nunique() < 10:
        return np.nan, np.nan
    items = z.item_id.to_numpy()
    y = z.lp_after.to_numpy()
    order = np.argsort(items, kind="stable")
    items, y = items[order], y[order]
    rs = []
    for _ in range(reps):
        u = RNG.random(len(y))
        tmp = pd.DataFrame(dict(i=items, y=y, u=u))
        tmp["rk"] = tmp.groupby("i").u.rank(method="first")
        tmp["n"] = tmp.groupby("i").y.transform("size")
        tmp["half"] = (tmp.rk > tmp.n / 2).astype(int)
        h = tmp.groupby(["i", "half"]).y.mean().unstack()
        rs.append(np.corrcoef(h[0], h[1])[0, 1])
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r)


def check2(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match in ["item", "src"]:
        for (m, t), zall in d.groupby(["model", "tier"]):
            nb = zall[zall.b == 0]
            key = ["item_id"] if match == "item" else ["item_id", "source"]
            sig2w = float(nb.groupby(key).lp_after.var(ddof=1).dropna().pipe(
                lambda v: np.average(v, weights=nb.groupby(key).size().loc[v.index] - 1)))
            if match == "item":
                r_half, rel_sb = split_half_rel(nb)
            else:
                tmp = nb.assign(item_id=nb.item_id + "|" + nb.source)
                r_half, rel_sb = split_half_rel(tmp)
            bel = zall[(zall.b != 0) & zall[f"N_{match}"].notna()].copy()
            bel["A"] = bel.b * bel.lp_after
            bel["Nb"] = bel.b * bel[f"N_{match}"]
            bel["k"] = bel[f"k_{match}"]
            for sample in ["all", "noclip"]:
                bs = bel if sample == "all" else bel[bel[f"N_{match}"].abs() < CLIP]
                for side, zz in [("all", bs), ("contra", bs[bs.s < 0]), ("support", bs[bs.s > 0])]:
                    f = clustered("A ~ Nb", zz)
                    sl, se, _ = coef(f, "Nb")
                    ic, sei, _ = coef(f, "Intercept")
                    varNb = zz.Nb.var(ddof=1)
                    lam = 1 - np.mean(sig2w / zz.k) / varNb
                    t1 = f.t_test("Nb = 1")
                    rows.append(dict(
                        model=m, tier=t, match=match, sample=sample, side=side,
                        readout=zz.readout.iloc[0],
                        slope_raw=sl, se_raw=se, p_slope_eq_1=float(t1.pvalue),
                        intercept=ic, se_intercept=sei,
                        r_splithalf=r_half, rel_sb=rel_sb, lambda_vc=lam,
                        slope_disatt_vc=sl / lam, se_disatt_vc=se / lam,
                        slope_disatt_sb=sl / rel_sb, se_disatt_sb=se / rel_sb,
                        mean_A=zz.A.mean(), mean_Nb=zz.Nb.mean(),
                        sd_A=zz.A.std(), sd_Nb=zz.Nb.std(), sd_ratio=zz.A.std() / zz.Nb.std(),
                        corr_A_Nb=np.corrcoef(zz.A, zz.Nb)[0, 1],
                        sigma2_within=sig2w, mean_k=zz.k.mean(),
                        n=len(zz), n_topics=zz.topic.nunique()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------
EDGES = np.array([-3.0, -2.0, -1.2, -0.6, 0.0, 0.6, 1.2, 2.0, 3.0])


def binned(s, y):
    b = np.digitize(s, EDGES[1:-1])
    t = pd.DataFrame(dict(b=b, s=s, y=y)).groupby("b").agg(s=("s", "mean"), y=("y", "mean"),
                                                             sd=("y", "std"), n=("y", "size"))
    t["se"] = t.sd / np.sqrt(t.n)
    return t[t.n >= 20]


def figure(d: pd.DataFrame, tier: int, paths: list[Path]):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True)
    z0 = d[d.tier == tier]
    for ax, m in zip(axes.ravel(), MODELS):
        z = z0[z0.model == m]
        bel = z[z.b != 0]
        nb = z[z.b == 0]
        # no-belief: duplicate with virtual b = +/-1
        s_nb = np.concatenate([nb.x.to_numpy(), -nb.x.to_numpy()])
        y_nb = np.concatenate([nb.lp_after.to_numpy(), -nb.lp_after.to_numpy()])
        tn = binned(s_nb, y_nb)
        tb = binned(bel.s.to_numpy(), (bel.b * bel.lp_after).to_numpy())
        ax.axhline(0, color=OI_GREY, lw=0.6)
        ax.axvline(0, color=OI_GREY, lw=0.6)
        ax.errorbar(tn.s, tn.y, yerr=1.96 * tn.se, fmt="--s", color=OI_ORANGE, mfc="white", ms=4.5,
                    lw=1.4, capsize=0, label="no belief (signed both ways)")
        ax.errorbar(tb.s, tb.y, yerr=1.96 * tb.se, fmt="-o", color=OI_BLUE, ms=4.5, lw=1.8,
                    capsize=0, label="belief stated")
        rd = " [sampled readout]" if z.readout.iloc[0] == "sample" else ""
        ax.set_title(f"{LABEL[m]}{rd}", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
    fig.supxlabel("s = b·x  (evidence position relative to the belief; s < 0 contradicts it)", fontsize=9, y=0.06)
    for ax in axes[:, 0]:
        ax.set_ylabel("b · logit P(H1) after", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    tname = "Tier 2 (social topics)" if tier == 2 else "Tier 1 (control topics)"
    fig.suptitle(f"Answer signed toward the belief vs evidence position, {tname}. Parallel shift = "
                 "additive; flatter only for s<0 = slope flattening; flat = anchoring", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)


NOTES = """Additivity / anchoring checks (analysis/15_additivity_anchoring.py)

Sign: everything signed toward the belief by b (not belief_dir). s = b*x; s<0 <=> kappa==1 <=> e=='contra'
(asserted in the script; see sign_check.csv).
Matching unit: item_id (x constant within it). base_slot alone is a slot index within a topic and
pools both H1 wordings (x flips sign), so it is NOT a valid matching unit. match = 'item' (primary)
or 'src' (same item and same source; fewer matched trials).
Check 1 caveat: a constant shift (a>0, c~0) is also what rational treatment of the stated belief
as testimony predicts. This check separates 'shift' from 'slope flattening', not 'social rule'
from 'testimony'.
Open models have no-belief evidence slope beta ~ 0 (context_beta.csv); for them E on contradicting
trials is mechanically evidence-independent, so c ~ 0 there is uninformative about flattening.
claude and gemini use sampled readout (one sampled answer): much noisier; claude n is small.
Check 2: disattenuated slopes divide by reliability of N (lambda_vc primary, rel_sb split-half SB);
their SEs ignore uncertainty in the reliability. SEs clustered by topic; tier 1 has few topics.
"""


def main():
    d = build()
    save(sign_check_table(d), SUB, "sign_check", show=False)
    ctx = context_beta(d)
    save(ctx, SUB, "context_beta")

    # matching coverage
    bel = d[d.b != 0]
    cov = bel.groupby(["model", "tier"]).agg(
        n_belief=("b", "size"), frac_matched_item=("N_item", lambda v: v.notna().mean()),
        frac_matched_src=("N_src", lambda v: v.notna().mean()), median_k_item=("k_item", "median"),
        median_k_src=("k_src", "median"), frac_N_clip=("N_item", lambda v: (v.abs() >= CLIP).mean())
    ).reset_index()
    save(cov, SUB, "matching_coverage")

    c1, c1j = check1(d)
    save(c1, SUB, "check1_additivity")
    save(c1j, SUB, "check1_joint_intercept_test")
    c2 = check2(d)
    save(c2, SUB, "check2_anchoring", show=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(c2[(c2.side == "all")][["model", "tier", "match", "sample", "slope_raw", "se_raw",
                                      "rel_sb", "lambda_vc", "slope_disatt_vc", "slope_disatt_sb",
                                      "mean_A", "mean_Nb", "sd_ratio", "n"]].round(3).to_string(index=False))
    (out_dir(SUB) / "NOTES.txt").write_text(NOTES)

    figure(d, 2, [FIG_DIR / "F16_additivity_anchoring.pdf", FIG_DIR / "F16_additivity_anchoring.png"])
    figure(d, 1, [out_dir(SUB) / "F16_tier1_additivity_anchoring.png"])
    print("\nfigure written to", FIG_DIR / "F16_additivity_anchoring.png")


if __name__ == "__main__":
    main()
