"""Step 18: sensitivity of Delta_b and theta_cong to prompt wording (reviewer check).

    python analysis/18_wording_sensitivity.py

Reviewer question: how sensitive are Delta_b and theta_cong to the wording of the belief statement?

What the data can answer. The belief sentence itself was never varied: every topic has one fixed
sentence per direction (data/topics.csv, data/tier1_topics.csv; all of the form "I think the <policy>
helped/hurt <outcome>."). The only wording manipulation is P6 (07_exploratory.py): two rewordings of the
*surrounding* prompt (hypothesis header and question; src/idea/prompts.py VARIANT_HEADS / VARIANT_QUESTIONS)
for four open-weight models on a fixed third of trials. This script re-analyses P6 for both estimands and both
tiers; it says nothing directly about weaker or stronger belief phrasings.

Sample. Exactly as P6 in 07_exploratory.py: the trial ids run under the variant (<model>_pv1; pv2 uses the
same ids, checked below) select the original model's chat-format trials; each arm then goes through
common.analysis_frame (its own exclusions, x = 0 dropped, one tier).

Estimates. Per arm and tier: the primary model (common.BASE_TERMS, common.fit_ols, topic-clustered SEs),
Delta_b = x:contra and theta_cong = x:contra:m. Variant - original changes come from the P6 stacked
regression (every term interacted with a variant dummy; change = coefficient on x:contra:var or
x:contra:m:var), reimplemented here with a tier argument. Implied shift = -Delta_b * mean|x| over the arm's
contradicting trials (as in 17_local_delta_b.py): the log-odds by which contradicting evidence of average
strength moves the answer less than it would without the belief.

Uniformity. One fully interacted regression per tier stacks all 12 arms (4 models x original/wording 1/
wording 2; every term arm-specific, topic-clustered SEs), so the covariance of all changes is available.
Wald tests: (a) per model, wording 1 change = wording 2 change; (b) all 8 changes equal; (c) the model-average
change equal across the 4 models; (d) theta_cong equal across a model's 3 arms. Tier 1 vs tier 2 changes use
disjoint topic sets (independent clusters), so their difference has SE sqrt(se1^2 + se2^2).

Outputs: analysis/output/wording/{estimates,changes,wald_tests,tier_compare,belief_sentences}.csv,
../paper/secondary/tables/wording_sensitivity.tex, figure F21_wording_sensitivity.{pdf,png}.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_TERMS, ROOT, analysis_frame, estimate, fit_ols, load_dataset, out_dir  # noqa: E402

warnings.filterwarnings("ignore")
BASES = ["olmo_instruct", "llama", "olmo13_instruct", "phi4"]
NAME = {"olmo_instruct": "OLMo 2 7B Instr.", "llama": "Llama 3.1 8B Instr.",
        "olmo13_instruct": "OLMo 2 13B Instr.", "phi4": "Phi-4 14B"}
ARM_LABEL = {0: "original", 1: "wording 1", 2: "wording 2"}
COLORS = {0: "#000000", 1: "#E69F00", 2: "#0072B2"}     # Okabe-Ito
MARKERS = {0: "o", 1: "s", 2: "D"}
PAPER = ROOT.parent / "paper" / "secondary"
TERMS = [t.strip() for t in BASE_TERMS.split("+")]


def arm_name(base: str, v: int) -> str:
    return base if v == 0 else f"{base}_pv{v}"


def arm_frame(df: pd.DataFrame, base: str, v: int, ids: set, tier: int) -> pd.DataFrame:
    d = df[(df.model == arm_name(base, v)) & (df.fmt == "chat") & df.trial_id.isin(ids)]
    return analysis_frame(d, tier)


def stacked_change(d0: pd.DataFrame, d1: pd.DataFrame) -> dict:
    """P6 stacked test (07_exploratory.stacked_variant_test), on already-built analysis frames."""
    d = pd.concat([d0.assign(var=0), d1.assign(var=1)], ignore_index=True)
    terms = TERMS if d.cue_type.nunique() > 1 else [t for t in TERMS if "cue_type" not in t]
    fit = fit_ols(d, " + ".join(terms + ["var"] + [f"{t}:var" for t in terms]))
    out = {"n_stacked": int(fit.nobs)}
    for name, term in (("delta_b_change", "x:contra:var"), ("theta_change", "x:contra:m:var")):
        e = estimate(fit, term)
        out.update({name: e["est"], f"{name}_se": e["se"], f"{name}_p": e["p"]})
    return out


def all_arms_fit(frames: dict):
    """One regression, all 12 arms, every term (incl. intercept) arm-specific; topic-clustered SEs."""
    parts, rhs = [], []
    for k, ((base, v), d) in enumerate(frames.items()):
        dd = d.copy()
        for j in range(len(frames)):
            dd[f"A{j}"] = float(j == k)
        parts.append(dd)
    d = pd.concat(parts, ignore_index=True)
    for j in range(len(frames)):
        rhs += [f"A{j}"] + [f"{t}:A{j}" for t in TERMS]
    fit = smf.ols("lp_after ~ 0 + " + " + ".join(rhs), d).fit(
        cov_type="cluster", cov_kwds={"groups": d.topic.astype("category").cat.codes})
    return fit, {key: j for j, key in enumerate(frames)}


def wald(fit, rows: list[dict]) -> tuple[float, float, int]:
    """Wald chi2 test of R b = 0 (each row: {param_name: weight})."""
    names = list(fit.params.index)
    R = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for n, w in r.items():
            R[i, names.index(n)] += w
    rb = R @ fit.params.values
    V = R @ fit.cov_params().values @ R.T
    stat = float(rb @ np.linalg.solve(V, rb))
    return stat, float(stats.chi2.sf(stat, len(rows))), len(rows)


def belief_sentences() -> pd.DataFrame:
    rows = []
    for f in ("topics.csv", "tier1_topics.csv"):
        t = pd.read_csv(ROOT / "data" / f)
        for _, r in t.iterrows():
            for direction in ("pos", "neg"):
                s = r[f"belief_{direction}"]
                rows.append(dict(topic=r.topic_id, tier=r.tier, direction=direction, sentence=s,
                                 opener=" ".join(s.split()[:2]), n_words=len(s.split()),
                                 hedge_or_intensifier=any(w in s.lower() for w in
                                     ("sure", "maybe", "perhaps", "certain", "might", "probably", "strongly",
                                      "know", "believe", "suspect", "convinced", "definitely", "guess"))))
    return pd.DataFrame(rows)


def main() -> None:
    df = load_dataset("main")
    od = out_dir("wording")

    bs = belief_sentences()
    bs.to_csv(od / "belief_sentences.csv", index=False)
    print(f"belief sentences: {len(bs)}; openers: {bs.opener.value_counts().to_dict()}; "
          f"hedges/intensifiers: {int(bs.hedge_or_intensifier.sum())}")

    est_rows, chg_rows, tests, tier_rows = [], [], [], []
    for tier in (2, 1):
        frames = {}
        for base in BASES:
            ids = set(df[df.model == arm_name(base, 1)].trial_id)
            assert ids == set(df[df.model == arm_name(base, 2)].trial_id), base
            assert ids <= set(df[(df.model == base) & (df.fmt == "chat")].trial_id), base
            for v in (0, 1, 2):
                d = arm_frame(df, base, v, ids, tier)
                frames[(base, v)] = d
                fit = fit_ols(d)
                db, th = estimate(fit, "x:contra"), estimate(fit, "x:contra:m")
                mx = float(d.loc[d.contra == 1, "x"].abs().mean())
                est_rows.append(dict(tier=tier, model=base, arm=ARM_LABEL[v], v=v, n=int(fit.nobs),
                                     n_topics=d.topic.nunique(), beta=estimate(fit, "x")["est"],
                                     delta_b=db["est"], delta_b_se=db["se"], delta_b_p=db["p"],
                                     delta_b_lo=db["lo"], delta_b_hi=db["hi"],
                                     theta=th["est"], theta_se=th["se"], theta_p=th["p"],
                                     theta_lo=th["lo"], theta_hi=th["hi"],
                                     support=estimate(fit, "x:support")["est"], mean_abs_x_contra=mx,
                                     implied_shift=-db["est"] * mx, implied_shift_se=db["se"] * mx))
            for v in (1, 2):
                d0, d1 = frames[(base, 0)], frames[(base, v)]
                st = stacked_change(d0, d1)
                both = set(d0.trial_id) & set(d1.trial_id)       # sensitivity: trials usable in both arms
                st_i = stacked_change(d0[d0.trial_id.isin(both)], d1[d1.trial_id.isin(both)])
                chg_rows.append(dict(tier=tier, model=base, arm=ARM_LABEL[v], v=v, **st,
                                     n_both_usable=len(both),
                                     delta_b_change_both=st_i["delta_b_change"],
                                     delta_b_change_both_se=st_i["delta_b_change_se"],
                                     theta_change_both=st_i["theta_change"],
                                     theta_change_both_se=st_i["theta_change_se"]))

        fit, idx = all_arms_fit(frames)
        p = lambda base, v, term="x:contra": f"{term}:A{idx[(base, v)]}"  # noqa: E731
        # check: stacked fit reproduces the per-arm point estimates
        E = pd.DataFrame(est_rows)
        for (base, v), j in idx.items():
            ref = E[(E.tier == tier) & (E.model == base) & (E.v == v)].delta_b.iloc[0]
            assert abs(fit.params[p(base, v)] - ref) < 1e-8, (base, v)
        chg = lambda base, v: {p(base, v): 1, p(base, 0): -1}  # noqa: E731
        for base in BASES:
            r = {**chg(base, 1)}
            for k, w in chg(base, 2).items():
                r[k] = r.get(k, 0) - w
            s, pv, k = wald(fit, [r])
            tests.append(dict(tier=tier, test=f"{base}: wording 1 change = wording 2 change (Delta_b)",
                              chi2=s, df=k, p=pv))
            s, pv, k = wald(fit, [{p(base, 1, "x:contra:m"): 1, p(base, 0, "x:contra:m"): -1},
                                  {p(base, 2, "x:contra:m"): 1, p(base, 0, "x:contra:m"): -1}])
            tests.append(dict(tier=tier, test=f"{base}: theta_cong equal across original, wording 1, wording 2",
                              chi2=s, df=k, p=pv))
        allc = [(b, v) for b in BASES for v in (1, 2)]
        rows = []
        for (b, v) in allc[1:]:
            r = dict(chg(b, v))
            for kk, w in chg(*allc[0]).items():
                r[kk] = r.get(kk, 0) - w
            rows.append(r)
        s, pv, k = wald(fit, rows)
        tests.append(dict(tier=tier, test="all 8 Delta_b changes equal", chi2=s, df=k, p=pv))
        s, pv, k = wald(fit, [{p(b, v): 1, p(b, 0): -1} for (b, v) in allc])
        tests.append(dict(tier=tier, test="all 8 Delta_b changes zero", chi2=s, df=k, p=pv))
        avg = lambda b: {p(b, 1): 0.5, p(b, 2): 0.5, p(b, 0): -1}  # noqa: E731
        rows = []
        for b in BASES[1:]:
            r = dict(avg(b))
            for kk, w in avg(BASES[0]).items():
                r[kk] = r.get(kk, 0) - w
            rows.append(r)
        s, pv, k = wald(fit, rows)
        tests.append(dict(tier=tier, test="model-average Delta_b change equal across 4 models", chi2=s, df=k, p=pv))
        # proportional change (variant / original) with delta-method SEs
        V = fit.cov_params()
        for (b, v) in allc:
            a, o = fit.params[p(b, v)], fit.params[p(b, 0)]
            g = np.array([1 / o, -a / o ** 2])
            cv = V.loc[[p(b, v), p(b, 0)], [p(b, v), p(b, 0)]].values
            C = pd.DataFrame(chg_rows)
            m = (C.tier == tier) & (C.model == b) & (C.v == v)
            idx_row = C.index[m][0]
            chg_rows[idx_row].update(ratio=a / o, ratio_se=float(np.sqrt(g @ cv @ g)))

    E, C, T = pd.DataFrame(est_rows), pd.DataFrame(chg_rows), pd.DataFrame(tests)
    # tier 1 vs tier 2 change (disjoint topics -> independent clusters)
    for b in BASES:
        for v in (1, 2):
            c2 = C[(C.tier == 2) & (C.model == b) & (C.v == v)].iloc[0]
            c1 = C[(C.tier == 1) & (C.model == b) & (C.v == v)].iloc[0]
            diff = c2.delta_b_change - c1.delta_b_change
            se = np.hypot(c2.delta_b_change_se, c1.delta_b_change_se)
            rd = c2.ratio - c1.ratio
            rse = np.hypot(c2.ratio_se, c1.ratio_se)
            tier_rows.append(dict(model=b, arm=ARM_LABEL[v], change_t2=c2.delta_b_change, change_t1=c1.delta_b_change,
                                  diff=diff, diff_se=se, diff_p=2 * stats.norm.sf(abs(diff / se)),
                                  same_direction=np.sign(c2.delta_b_change) == np.sign(c1.delta_b_change),
                                  ratio_t2=c2.ratio, ratio_t1=c1.ratio, ratio_diff=rd, ratio_diff_se=rse,
                                  ratio_diff_p=2 * stats.norm.sf(abs(rd / rse))))
    TC = pd.DataFrame(tier_rows)
    for name, tab in (("estimates", E), ("changes", C), ("wald_tests", T), ("tier_compare", TC)):
        tab.to_csv(od / f"{name}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(E[["tier", "model", "arm", "n", "delta_b", "delta_b_se", "theta", "theta_se", "theta_p",
                 "implied_shift"]].round(3).to_string(index=False))
        print(C[["tier", "model", "arm", "delta_b_change", "delta_b_change_se", "delta_b_change_p", "theta_change",
                 "theta_change_se", "theta_change_p", "ratio", "ratio_se", "delta_b_change_both"]].round(3).to_string(index=False))
        print(T.round(4).to_string(index=False))
        print(TC.round(3).to_string(index=False))
    summary(E, C)
    write_table(E, C)
    figure(E)


def summary(E: pd.DataFrame, C: pd.DataFrame) -> None:
    rows = []
    for tier in (2, 1):
        for b in BASES:
            e = E[(E.tier == tier) & (E.model == b)]
            rows.append(dict(tier=tier, model=b, delta_b_min=e.delta_b.min(), delta_b_max=e.delta_b.max(),
                             delta_b_range=e.delta_b.max() - e.delta_b.min(),
                             shift_min=e.implied_shift.min(), shift_max=e.implied_shift.max(),
                             delta_b_all_negative=bool((e.delta_b < 0).all()),
                             delta_b_all_sig=bool((e.delta_b_p < 0.05).all()),
                             theta_min=e.theta.min(), theta_max=e.theta.max(),
                             theta_signs="".join("+" if t > 0 else "-" for t in e.theta),
                             theta_sig_pattern="".join("*" if q < 0.05 else "." for q in e.theta_p)))
    S = pd.DataFrame(rows)
    S.to_csv(out_dir("wording") / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(S.round(3).to_string(index=False))


def fmt(v: float, d: int = 3) -> str:
    s = f"{v:.{d}f}"
    return s.replace("-", "$-$")


def pfmt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_table(E: pd.DataFrame, C: pd.DataFrame) -> None:
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
         r"\caption{Sensitivity of $\Delta_b$ and $\theta_{\mathrm{cong}}$ to prompt wording (P6 rewordings of the "
         r"hypothesis header and question; the belief sentence is unchanged). Same trials per model (the third run "
         r"under the rewordings); change = variant $-$ original from the stacked P6 regression, topic-clustered SEs. "
         r"Shift = $-\Delta_b\cdot\overline{|x|}$ on contradicting trials (log-odds).}\label{tab:wording}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & & \multicolumn{5}{c}{Tier 2} & \multicolumn{2}{c}{Tier 1} \\",
         r"\cmidrule(lr){3-7}\cmidrule(lr){8-9}",
         r"Model & Prompt & $\Delta_b$ (SE) & change (SE) & shift & $\theta_{\mathrm{cong}}$ (SE) & change (SE) "
         r"& $\Delta_b$ (SE) & change (SE) \\", r"\midrule"]
    for b in BASES:
        for v in (0, 1, 2):
            e2 = E[(E.tier == 2) & (E.model == b) & (E.v == v)].iloc[0]
            e1 = E[(E.tier == 1) & (E.model == b) & (E.v == v)].iloc[0]
            star = lambda p: r"$^{*}$" if p < 0.05 else ""  # noqa: E731
            if v == 0:
                c2d = c2t = c1d = "---"
            else:
                c2 = C[(C.tier == 2) & (C.model == b) & (C.v == v)].iloc[0]
                c1 = C[(C.tier == 1) & (C.model == b) & (C.v == v)].iloc[0]
                c2d = f"{fmt(c2.delta_b_change)} ({c2.delta_b_change_se:.3f}){star(c2.delta_b_change_p)}"
                c2t = f"{fmt(c2.theta_change)} ({c2.theta_change_se:.3f}){star(c2.theta_change_p)}"
                c1d = f"{fmt(c1.delta_b_change)} ({c1.delta_b_change_se:.3f}){star(c1.delta_b_change_p)}"
            L.append(f"{NAME[b] if v == 0 else ''} & {ARM_LABEL[v]} & {fmt(e2.delta_b)} ({e2.delta_b_se:.3f}) & {c2d} & "
                     f"{fmt(e2.implied_shift, 2)} & {fmt(e2.theta)} ({e2.theta_se:.3f}){star(e2.theta_p)} & {c2t} & "
                     f"{fmt(e1.delta_b)} ({e1.delta_b_se:.3f}) & {c1d} \\\\")
        if b != BASES[-1]:
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\multicolumn{9}{l}{$^{*}$ $p<0.05$ (unadjusted). Tier 2: 17 topics; Tier 1: 13 topics.}\\",
          r"\end{tabular}", r"\end{table}"]
    p = PAPER / "tables" / "wording_sensitivity.tex"
    p.write_text("\n".join(L) + "\n")
    print(f"wrote {p}")


def figure(E: pd.DataFrame) -> None:
    panels = [(2, "delta_b", r"Tier 2: $\Delta_b$"), (2, "theta", r"Tier 2: $\theta_{\mathrm{cong}}$"),
              (1, "delta_b", r"Tier 1: $\Delta_b$")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    xs = np.arange(len(BASES))
    for ax, (tier, col, title) in zip(axes, panels):
        se = "delta_b_se" if col == "delta_b" else "theta_se"
        for v in (0, 1, 2):
            q = E[(E.tier == tier) & (E.v == v)].set_index("model").loc[BASES]
            ax.errorbar(xs + (v - 1) * 0.22, q[col], yerr=1.96 * q[se], fmt=MARKERS[v], color=COLORS[v],
                        ms=6, capsize=2, lw=1.2, label=ARM_LABEL[v])
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([NAME[b].replace(" Instr.", "\nInstr.").replace(" 14B", "\n14B") for b in BASES], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("estimate (95% CI)", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left", title="prompt", title_fontsize=8)
    fig.tight_layout()
    out = PAPER / "figures" / "candidates" / "F21_wording_sensitivity"
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
