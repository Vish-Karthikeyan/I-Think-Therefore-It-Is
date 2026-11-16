"""Step 9: LaTeX tables for the paper, generated from analysis/output/ and results/ (never typed by hand).

    python analysis/09_paper_tables.py

Writes paper/secondary/tables/*.tex; the paper \\input{}s them. Rerun after any analysis rerun
(e.g. when the Llama audit finishes) and recompile the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis" / "output"
TAB = ROOT.parent / "paper" / "secondary" / "tables"

NAMES = {"gpt": "GPT-4o", "claude": "Claude Sonnet 5", "gemini": "Gemini 2.5 Flash", "llama": "Llama 3.1 8B Instr.",
         "olmo_base": "OLMo 2 7B base", "olmo_sft": "OLMo 2 7B SFT", "olmo_dpo": "OLMo 2 7B DPO",
         "olmo_instruct": "OLMo 2 7B Instr.", "olmo13_base": "OLMo 2 13B base", "olmo13_sft": "OLMo 2 13B SFT",
         "olmo13_dpo": "OLMo 2 13B DPO", "olmo13_instruct": "OLMo 2 13B Instr.", "phi4": "Phi-4 14B",
         "tulu_base": "Llama 3.1 8B base", "tulu_sft": "T\\\"ulu 3 SFT", "tulu_dpo": "T\\\"ulu 3 DPO",
         "tulu_final": "T\\\"ulu 3 final", "olmo_instruct_sp": "OLMo 7B Instr. + sys. prompt",
         "llama_sp": "Llama 8B + sys. prompt"}
for b in ("olmo_instruct", "llama", "olmo13_instruct", "phi4"):
    for v in (1, 2):
        NAMES[f"{b}_pv{v}"] = f"{NAMES[b]} (wording {v})"
ORDER = ["gpt", "claude", "gemini", "llama", "olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]


def name(m):
    return NAMES.get(m, m.replace("_", "\\_"))


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


STEP = {"sft": "SFT", "dpo": "DPO", "final": "final", "instruct": "Instruct", "base": "base"}
LADDER = {"tulu": "T\\\"ulu 3", "olmo7": "OLMo 2 7B", "olmo13": "OLMo 2 13B"}
DIRECTION = {"hat_k4": "bias, topic PCA $k=4$", "hat_k8": "bias, topic PCA $k=8$", "hat_k16": "bias, topic PCA $k=16$",
             "hat_k8_amplify": "bias, $k=8$, reversed", "raw": "bias, raw", "random": "random (control)",
             "topic_only_k8": "topic component only", "prompt_hat_k8": "prompt contrast, $k=8$"}


def direction_name(d):
    d = str(d)
    if d.startswith("per_dim_"):
        return f"bias, {d[len('per_dim_'):]} only"
    return DIRECTION.get(d, esc(d))


def step_name(s):
    return " $\\to$ ".join(STEP.get(x, esc(x)) for x in str(s).split("->"))


def prose_tex(s):
    """Turn a plain-text evidence string from the analysis CSVs into LaTeX: model keys to names, Delta_b, arrows,
    scientific notation, number ranges (en dash) and signs (math minus)."""
    import re
    s, keep = str(s), {}
    for i, k in enumerate(sorted(NAMES, key=len, reverse=True)):
        tok = f"@@{i}@@"
        s2 = re.sub(rf"(?<![\w]){re.escape(k)}(?![\w])", tok, s)
        if s2 != s:
            keep[tok], s = NAMES[k], s2
    s = s.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")
    s = s.replace("Delta\\_b", "$\\Delta_b$")
    s = re.sub(r"(\d(?:\.\d+)?)e-(\d+)", lambda m: f"${m.group(1)}\\times10^{{-{int(m.group(2))}}}$", s)
    s = s.replace("->", "$\\to$")
    s = re.sub(r"(?<=\d)-(?=\d)", "--", s)
    s = re.sub(r"(^|[\s(=,:])-(?=\d)", r"\1$-$", s)
    s = re.sub(r"(^|[\s(=,:])\+(?=\d)", r"\1$+$", s)
    for tok, v in keep.items():
        s = s.replace(tok, v)
    return s


def num(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return "--"
    return f"{x:.{d}f}".replace("-", "$-$")


def pval(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(est, lo, hi, d=3):
    return f"{num(est, d)} [{num(lo, d)}, {num(hi, d)}]"


def mark(b):
    return "\\checkmark" if bool(b) and str(b) != "nan" else ""


def table(rows, header, colspec, caption, label, note=None, size="\\small", longt=False, fit=False):
    """Plain booktabs table (or longtable) from a list of row lists of strings."""
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    hdr = " & ".join(header) + " \\\\"
    if longt:
        out = (f"{size}\n\\begin{{longtable}}{{{colspec}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{hdr}\n"
               f"\\midrule\n\\endfirsthead\n\\toprule\n{hdr}\n\\midrule\n\\endhead\n{body}\n\\bottomrule\n\\end{{longtable}}\n")
        if note:
            out += f"\\noindent{{\\footnotesize {note}}}\n"
        return out + "\\normalsize\n"
    tab = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{hdr}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    if fit:   # scale a wide table down to the text width
        tab = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{tab}}}"
    out = (f"\\begin{{table}}[t]\n\\centering\n{size}\n\\caption{{{caption}}}\\label{{{label}}}\n{tab}\n")
    if note:
        out += f"\\par\\vspace{{2pt}}{{\\footnotesize {note}}}\n"
    return out + "\\end{table}\n"


def write(fname, tex):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / fname).write_text(tex)
    print("wrote", (TAB / fname).relative_to(ROOT.parent))


def sort_models(df, col="model"):
    key = {m: i for i, m in enumerate(ORDER)}
    return df.assign(_k=df[col].map(lambda m: key.get(m, 100)), _f=df.fmt).sort_values(["_k", col, "_f"]).drop(columns=["_k", "_f"])


def rq1_table(study, label, caption):
    f = A / "rq1" / f"per_model_{study}.csv"
    if not f.exists():
        return
    d = sort_models(pd.read_csv(f))
    rows = []
    for r in d.itertuples():
        if pd.isna(getattr(r, "delta_b", np.nan)):
            rows.append([name(r.model), r.fmt, str(int(r.n)), "\\multicolumn{5}{l}{fewer than 50 usable trials}"])
            continue
        rows.append([name(r.model), r.fmt, f"{int(r.n):,}", num(r.beta_neutral), num(r.rho),
                     ci(r.delta_b, r.delta_b_lo, r.delta_b_hi) + (" $^\\ast$" if r.holm_delta_b else ""),
                     ci(r.theta_cong, r.theta_lo, r.theta_hi) + (" $^\\ast$" if r.holm_theta else ""),
                     num(r.support_shift)])
    write(f"rq1_{study}.tex", table(
        rows, ["Model", "Format", "$n$", "$\\beta$", "$\\rho$", "$\\Delta_b$ [95\\% CI]", "$\\theta_{\\text{cong}}$ [95\\% CI]", "support"],
        "llrrrlll", caption, label,
        note="Tier 2, usable trials with evidence. $\\beta$: no-belief evidence slope (1 = Bayesian); $\\rho$: weight on the "
             "prior; $\\Delta_b$: change in slope when evidence contradicts the stated belief; $\\theta_{\\text{cong}}$: change in "
             "that slope per unit of identity--belief congruence $m$; support: change in slope when evidence supports the "
             "belief. $^\\ast$ significant after Holm correction over the two co-primary tests. SEs clustered by topic.",
        size="\\scriptsize"))


def main():
    design_tables()
    rq1_table("main", "tab:rq1-main", "RQ1 primary estimates, main run.")
    rq1_table("pilot", "tab:rq1-pilot", "RQ1 estimates in the pilot (including Claude Sonnet 5).")
    rq1_table("exploratory", "tab:rq1-exploratory", "RQ1 estimates for the exploratory models.")

    # joint tests
    rows = []
    for fmt in ("chat", "completion"):
        f = A / "rq1" / f"joint_main_{fmt}.csv"
        if f.exists():
            j = pd.read_csv(f)
            rows.append([fmt, str(int(j.n_restrictions.iloc[0])), num(j.wald_stat.iloc[0], 1), pval(j.wald_p.iloc[0])])
    write("joint.tex", table(rows, ["Format", "restrictions", "Wald $\\chi^2$", "$p$"], "lrrr",
                             "Do $\\Delta_b$ and $\\theta_{\\text{cong}}$ differ across models? Joint Wald test on the model "
                             "interactions with $x{:}\\text{contra}$ and $x{:}\\text{contra}{:}m$ (main run).", "tab:joint"))

    # robustness
    f = A / "robustness" / "robustness_summary.csv"
    if f.exists():
        r = sort_models(pd.read_csv(f))
        checks = ["primary", "congruence_svy", "congruence_beh", "svy_weak_dropped", "completion_cov08"]
        lab = {"primary": "primary ($m$)", "congruence_svy": "survey stance $m_{\\text{svy}}$", "congruence_beh":
               "behavioral stance $m_{\\text{beh}}$", "svy_weak_dropped": "$m_{\\text{svy}}$, weak proxies dropped",
               "completion_cov08": "completion cutoff 0.8"}
        rows = []
        for (m, fmt), g in r.groupby(["model", "fmt"], sort=False):
            for c in checks:
                x = g[g.check == c]
                if x.empty:
                    continue
                x = x.iloc[0]
                rows.append([name(m) if c == "primary" else "", fmt if c == "primary" else "", lab[c], f"{int(x.n):,}",
                             num(x.delta_b) + (" $^\\ast$" if x.holm_delta_b else ""), num(x.delta_b_se),
                             num(x.theta_cong) + (" $^\\ast$" if x.holm_theta else ""), num(x.theta_se)])
        write("robustness.tex", table(rows, ["Model", "Format", "Check", "$n$", "$\\Delta_b$", "SE", "$\\theta_{\\text{cong}}$", "SE"],
                                      "lllrrrrr", "Preregistered robustness checks (main run).", "tab:robust",
                                      note="$^\\ast$ significant after Holm correction.", size="\\scriptsize", longt=True))
    f = A / "robustness" / "tier1_low_g_tier_contrast.csv"
    g = A / "rq1_secondary" / "tier_main.csv"
    if f.exists() and g.exists():
        a, b = pd.read_csv(g), pd.read_csv(f)
        rows = []
        for (m, fmt), x in sort_models(a).groupby(["model", "fmt"], sort=False):
            y = b[(b.model == m) & (b.fmt == fmt)]
            def get(t, src):
                s = src[src.term == t]
                return (s.est.iloc[0], s.se.iloc[0]) if len(s) else (np.nan, np.nan)
            t1, amp = get("delta_b_tier1", x), get("amplification_tier2_minus_tier1", x)
            t1r, ampr = get("delta_b_tier1", y), get("amplification_tier2_minus_tier1", y)
            rows.append([name(m), fmt, f"{num(t1[0])} ({num(t1[1])})", f"{num(amp[0])} ({num(amp[1])})",
                         f"{num(t1r[0])} ({num(t1r[1])})", f"{num(ampr[0])} ({num(ampr[1])})"])
        write("tier.tex", table(rows, ["Model", "Format", "$\\Delta_b^{(1)}$", "$\\Delta_b^{(2)}-\\Delta_b^{(1)}$",
                                       "$\\Delta_b^{(1)}$ low-$|g|$", "amplif. low-$|g|$"], "llllll",
                                "Tier 1 (non-social topics) and social amplification, all Tier 1 vs. the low-$|g|$ Tier 1 subset. Estimate (SE).",
                                "tab:tier", size="\\scriptsize"))

    # delta_j equality and cue type and theta by dim
    f = A / "rq1_secondary" / "delta_j_equality_main.csv"
    if f.exists():
        e = sort_models(pd.read_csv(f))
        rows = [[name(r.model), r.fmt, esc(r.dimension), str(int(r.df)), num(r.stat, 2), pval(r.p)] for r in e.itertuples()]
        write("dj_equality.tex", table(rows, ["Model", "Format", "Dimension", "df", "Wald", "$p$"], "lllrrr",
                                       "Unequal deference: Wald test that all groups in a dimension have equal $\\delta_j$ (main run, Tier 2).",
                                       "tab:dj-eq", size="\\scriptsize", longt=True))
    f = A / "rq1_secondary" / "delta_j_main.csv"
    if f.exists():
        e = sort_models(pd.read_csv(f))
        rows = [[name(r.model), r.fmt, esc(r.dimension), esc(r.group), ci(r.delta, r.lo, r.hi), pval(r.p)] for r in e.itertuples()]
        write("dj_all.tex", table(rows, ["Model", "Format", "Dimension", "Group", "$\\delta_j$ [95\\% CI]", "$p$"], "llllll",
                                  "$\\delta_j$ for every group relative to the Unspecified user (main run, Tier 2).", "tab:dj-all",
                                  note="Negative $\\delta_j$: the group's contradicted beliefs receive extra protection from evidence.",
                                  size="\\scriptsize", longt=True))
    f = A / "rq1_secondary" / "cue_type_main.csv"
    if f.exists():
        c = sort_models(pd.read_csv(f))
        rows = [[name(r.model), r.fmt, esc(r.cue_type), f"{num(r.delta_b)} ({num(r.delta_b_se)})", f"{num(r.theta)} ({num(r.theta_se)})"]
                for r in c.itertuples()]
        write("cue_type.tex", table(rows, ["Model", "Format", "Cue", "$\\Delta_b$ (SE)", "$\\theta_{\\text{cong}}$ (SE)"], "lllll",
                                    "Explicit labels vs.\\ vignettes (main run, Tier 2).", "tab:cue", size="\\scriptsize",
                                    note="Label and vignette rows: separate fits. Difference row: the cue-type interaction in one joint "
                                         "fit of both cue types (shared prior, slope and control terms), so it need not equal the "
                                         "difference of the separate estimates."))
    f = A / "rq1_secondary" / "theta_by_dimension_main.csv"
    if f.exists():
        t = sort_models(pd.read_csv(f)).dropna(subset=["est"])
        rows = [[name(r.model), r.fmt, esc(r.dimension), ci(r.est, r.lo, r.hi), pval(r.p)] for r in t.itertuples()]
        write("theta_dim.tex", table(rows, ["Model", "Format", "Dimension", "$\\theta_{\\text{cong}}+\\tau_k$ [95\\% CI]", "$p$"],
                                     "lllll", "Congruence effect by identity dimension (main run, Tier 2).", "tab:theta-dim",
                                     size="\\scriptsize", longt=True))

    # controls
    rt = A / "controls" / "retest_main.csv"
    ne = A / "controls" / "null_evidence_main.csv"
    sm = A / "controls" / "sample_main.csv"
    if rt.exists() and ne.exists() and sm.exists():
        r, n, s = pd.read_csv(rt), pd.read_csv(ne), pd.read_csv(sm)
        rows = []
        for x in sort_models(r).itertuples():
            ss = s[(s.model == x.model) & (s.fmt == x.fmt)]
            nn = n[(n.model == x.model) & (n.fmt == x.fmt) & (n.e_null == "belief")]
            excl = 1 - ss.usable.sum() / ss.trials.sum()
            drift = np.average(nn.drift_toward_belief, weights=nn.n) if len(nn) else np.nan
            noise = np.average(n[(n.model == x.model) & (n.fmt == x.fmt)].mean_abs_y,
                               weights=n[(n.model == x.model) & (n.fmt == x.fmt)].n)
            rows.append([name(x.model), x.fmt, f"{int(ss.trials.sum()):,}", f"{100 * excl:.1f}\\%", num(x.r), num(noise), num(drift)])
        write("controls.tex", table(rows, ["Model", "Format", "trials", "excluded", "retest $r$", "null $|y|$", "null drift"],
                                    "llrrrrr", "Instrument controls (main run).", "tab:controls",
                                    note="Retest $r$: correlation of the two paraphrased priors (log-odds). Null $|y|$: mean absolute update "
                                         "on tables with no evidence ($x=0$); null drift: mean update toward the stated belief on those tables.",
                                    size="\\scriptsize"))
    f = A / "controls" / "readout_calibration_main.csv"
    if f.exists():
        c = sort_models(pd.read_csv(f))
        rows = [[name(r.model), r.fmt, str(int(r.n)), num(r.r_prior), num(r.r_post), num(r.beta_dist), num(r.beta_sample),
                 num(r.delta_b_dist), num(r.delta_b_sample)] for r in c.itertuples()]
        write("calibration.tex", table(rows, ["Model", "Format", "$n$", "$r$ prior", "$r$ post", "$\\beta$ dist", "$\\beta$ sample",
                                              "$\\Delta_b$ dist", "$\\Delta_b$ sample"], "llrrrrrrr",
                                       "Readout calibration on open models: the same trials read by the full next-token distribution and by 10 samples.",
                                       "tab:calib", size="\\scriptsize"))

    # RQ2
    rows = []
    for f in sorted((A / "rq2").glob("*_chat_main_summary.csv")):
        m = f.name.replace("_chat_main_summary.csv", "")
        s = pd.read_csv(f)
        for r in s.itertuples():
            ok = (r.full_rmse < r.no_W_rmse) and (r.full_rmse < r.mean_rmse) and (r.full_spearman > max(r.no_W_spearman, r.mean_spearman))
            rows.append([name(m), r.split.replace("_", " "), num(r.full_rmse), num(r.no_W_rmse), num(r.mean_rmse),
                         num(r.full_spearman), num(r.no_W_spearman), num(r.mean_spearman), "yes" if ok else "no"])
    if rows:
        write("rq2.tex", table(rows, ["Model", "Split", "RMSE full", "no $W$", "mean", "$\\rho_S$ full", "no $W$", "mean", "success"],
                               "llrrrrrrr", "RQ2: predicting slope shifts on held-out topics and identity dimensions (chat format).", "tab:rq2",
                               note="Success requires the full model to beat both baselines on RMSE and Spearman correlation.",
                               size="\\scriptsize"))
    # secondary specs
    rows = []
    for f in sorted((A / "secondary").glob("*_main_summary.csv")):
        mf = f.name.replace("_main_summary.csv", "")
        m, fmt = mf.rsplit("_", 1)
        s = pd.read_csv(f)
        sp, mx = s[s.spec == "spline"].iloc[0], s[s.spec == "mixture"].iloc[0]
        stable = bool(mx.get("stable_across_starts", False))
        rows.append([name(m), fmt, pval(sp.f_p), num(mx.mean_p_ignore) if stable else "not identified",
                     num(mx.updated_beta) if stable else "--"])
    if rows:
        write("secondary_specs.tex", table(rows, ["Model", "Format", "spline $p$", "share ignoring", "slope of updaters"], "lllll",
                                           "Secondary specifications: nonlinearity in $x$ (spline vs.\\ linear, $F$ test) and the two-component "
                                           "mixture (best of 10 EM starts).", "tab:secondary",
                                           note="``Not identified'': starts with near-equal likelihood disagree by more than 0.05 on the share.",
                                           size="\\scriptsize"))

    # RQ3
    f = A / "rq3" / "steering_test.csv"
    if f.exists():
        t = pd.read_csv(f)
        rows = [[name(r.model), direction_name(r.direction), str(int(r.layer)), num(r.psi, 2), r.mode, num(r.t2_contra_gap_before),
                 num(r.t2_contra_gap_after), num(r.t2_neutral_change), num(r.tier1_neutral_change), "yes" if r.success else "no"]
                for r in t.itertuples()]
        write("steering.tex", table(rows, ["Model", "Direction", "Layer", "$\\psi$", "Mode", "gap before", "gap after",
                                           "$\\Delta\\beta_{b=0}$", "$\\Delta\\beta_{\\text{T1}}$", "success"], "lllrlrrrrl",
                                    "RQ3 held-out steering test. Gap = contradicted-evidence slope minus the unsteered neutral slope (Tier 2).",
                                    "tab:steer", note="Success (preregistered): the gap shrinks in every dimension and the neutral and Tier 1 "
                                                      "slopes change by less than 0.05.", size="\\scriptsize", longt=True))
    f = A / "rq3" / "chosen_settings.csv"
    if f.exists():
        c = pd.read_csv(f)
        rows = [[name(r.model), str(int(r.layer)), num(r.multiple, 1), r["mode"], str(int(r.probe_layer)), num(r.probe_r2),
                 num(r.masking)] for _, r in c.iterrows()]
        write("rq3_settings.tex", table(rows, ["Model", "Layer", "Multiple", "Mode", "Probe layer", "Probe $R^2$", "Masking"],
                                        "lrrlrrr", "RQ3 settings chosen on validation topics, and the masking probe.", "tab:rq3set",
                                        size="\\scriptsize"))
    f = A / "rq3" / "audit_summary.csv"
    if f.exists():
        a = pd.read_csv(f)
        rows = [[name(r.model), r.condition, num(r.mmlu_pro_acc), str(int(r.mmlu_pro_subjects)), num(r.bbq_acc),
                 num(r.bbq_bias_ambiguous), num(r.syceval_initial_acc), num(r.syceval_regressive_flip),
                 "" if r.complete else "partial"] for r in a.itertuples()]
        write("audit.tex", table(rows, ["Model", "Condition", "MMLU-Pro", "subj.", "BBQ acc.", "BBQ amb. bias", "SycEval acc.",
                                        "regress. flip", ""], "llrrrrrrl", "Capability and bias audit under steering.", "tab:audit",
                                 note="MMLU-Pro: 14 subjects, up to 500 items each. BBQ: 500 items (age category only). SycEval: 300 MMLU items.",
                                 size="\\scriptsize"))

    # RQ3, corrected (answer position; analysis/11_rq3_corrected.py)
    FAM = {"cr_single": "steer, one layer", "cr_all_layers": "steer, all layers", "erase_answer": "erase at answer",
           "erase_span": "erase at belief tokens", "erase_span_multi": "erase at belief tokens, 4 layers",
           "control_random_subspace": "random subspace (control)"}
    f = A / "rq3_v2" / "test.csv"
    if f.exists():
        t = pd.read_csv(f)
        rows = [[name(r.model), FAM.get(r.family, esc(r.family)), esc(r.setting), num(r.t2_contra_gap_before),
                 num(r.t2_contra_gap_after), f"{100 * r.gap_closed_frac:.0f}\\%".replace("-", "$-$") + ("$^\\dagger$" if r.overshoot else ""),
                 num(r.t2_neutral_change), num(r.tier1_neutral_change), num(r.tier1_contra_change),
                 mark(r.success_preregistered), mark(r.success_exploratory), mark(r.success_partial)]
                for r in t.itertuples()]
        write("rq3v2_test.tex", table(
            rows, ["Model", "Intervention", "Setting", "gap before", "gap after", "closed", "$\\Delta\\beta_{b=0}$",
                   "$\\Delta\\beta^{\\text{T1}}_{b=0}$", "$\\Delta\\beta^{\\text{T1}}_{\\text{contra}}$", "prereg.", "expl.", "partial"],
            "lllrrrrrrccc", "Corrected RQ3: held-out test at the answer position.", "tab:rq3v2",
            note="Gap: Tier~2 contradicted-evidence slope minus the unsteered no-belief slope; closed: share of the gap removed "
                 "($^\\dagger$ past 100\\%: the slope overshoots the no-belief slope). $\\Delta\\beta$: change in the no-belief "
                 "slope (Tier~2, Tier~1) and in Tier~1's contradicted slope. Success tiers: preregistered; exploratory (Tier~1 "
                 "belief-following may change, only its no-belief slope must stay within 0.05); partial (pooled only: at least "
                 "25\\% of the gap and 10 points more than the random control closed, no-belief slopes within 0.05). Settings: "
                 "layer (L) and subspace size (k) or strength multiple (x).", size="\\scriptsize", fit=True))
    f = A / "rq3_v2" / "sufficiency.csv"
    if f.exists():
        s = pd.read_csv(f)
        rows = [[name(r.model), str(int(r.layer)), num(r.answer_U2, 2), num(r.answer_U4, 2), num(r.whole_last, 2),
                 num(r.span_U2, 2), num(r.random_U2, 2)] for r in s.itertuples()]
        write("rq3v2_sufficiency.tex", table(
            rows, ["Model", "Layer", "answer, 2-D", "answer, 4-D", "answer, whole", "belief tokens, 2-D", "random, 2-D"],
            "lrrrrrr", "Corrected RQ3: interchange (sufficiency) test.", "tab:rq3v2-suff",
            note="Share of the belief effect on the answer's log-odds recovered by patching the named subspace (or the whole "
                 "hidden state) from a belief-stated input into the matching no-belief input; mean of the two swap directions, "
                 "300 held-out counterfactual pairs.", size="\\scriptsize", longt=True))
    f = A / "rq3_v2" / "audit.csv"
    if f.exists():
        a = pd.read_csv(f)
        # the cancelled automatic choice for Llama (overshooting single-layer steering) stays, marked as stopped
        stopped = (a.model == "llama") & a.label.isin(["corrected", "control"])
        a = a.assign(condition=np.where(stopped, a.condition.map({"corrected": "stopped_steer", "control": "stopped_rand"}),
                                        a.condition))
        cond_name = {"unsteered": "unsteered", "corrected": "erasure", "control": "random control",
                     "stopped_steer": "overshooting steering (stopped)", "stopped_rand": "random vectors, same norm (stopped)"}
        rows = []
        for r in a.sort_values(["model", "condition"], key=lambda c: c.map({"unsteered": 0, "corrected": 1, "control": 2, "stopped_steer": 3, "stopped_rand": 4})
                               if c.name == "condition" else c).itertuples():
            setting = f"{FAM.get(r.family, '')} {r.setting}".strip() if isinstance(r.setting, str) and r.setting else ""
            if r.condition.startswith("stopped"):
                setting = r.setting
            elif r.condition == "control" and setting:   # matched random subspace of the same layer and size
                setting = f"random subspace {r.setting}"
            rows.append([name(r.model), cond_name.get(r.condition, r.condition) + (f" ({esc(setting)})" if setting else ""),
                         num(getattr(r, "mmlu_pro_acc", np.nan)), num(getattr(r, "bbq_acc", np.nan)),
                         num(getattr(r, "bbq_amb_bias", np.nan)), num(getattr(r, "syceval_initial_acc", np.nan)),
                         num(getattr(r, "syceval_regressive_flip", np.nan)), num(getattr(r, "l2d_truth_discernment", np.nan)),
                         num(getattr(r, "l2d_source_discernment", np.nan)),
                         "" if r.status == "complete" else ("stopped" if r.condition.startswith("stopped") else "running")])
        write("rq3v2_audit.tex", table(
            rows, ["Model", "Condition", "MMLU-Pro", "BBQ acc.", "BBQ amb. bias", "SycEval acc.", "regress. flip",
                   "L2D TD", "L2D SD", ""], "llrrrrrrrl", "Corrected RQ3: capability and bias audit.", "tab:rq3v2-audit",
            note="MMLU-Pro: 14 subjects, 200 items each (sample-weighted). BBQ: 500 items (age category only). SycEval: 300 MMLU "
                 "items. L2D: 4,000 tuples; TD/SD = truth and source discernment \\citep{ashkinaze2026discernment}. ``running'': audit "
                 "not finished when the table was built.", size="\\scriptsize", fit=True))

    # exploratory
    f = A / "exploratory" / "predictions.csv"
    if f.exists():
        p = pd.read_csv(f)
        rows = [[r.prediction, r.verdict, prose_tex(r.evidence)] for r in p.itertuples()]
        write("predictions.tex", table(rows, ["", "Verdict", "Evidence"], "llp{10cm}", "Exploratory predictions P0--P6.", "tab:pred",
                                       size="\\scriptsize"))
    f = A / "exploratory" / "P1_evidence_diagnostic.csv"
    if f.exists():
        d = sort_models(pd.read_csv(f))
        rows = [[name(r.model), r.fmt, num(r.rate_mae, 1), num(r.compare_acc, 2), num(r.compare_p_A, 2), num(r.support_acc, 2),
                 num(r.post_dir_acc, 2), num(r.post_slope_vs_x)] for r in d.itertuples()]
        write("diagnostic.tex", table(rows, ["Model", "Format", "rate MAE", "compare acc.", "$P$(A)", "support acc.", "post. dir. acc.",
                                             "post. slope"], "llrrrrrr",
                                      "Evidence diagnostic (no persona, no belief; 480 tables).", "tab:diag",
                                      note="Rate MAE in percentage points. $P$(A): mean probability on the first option (position bias).",
                                      size="\\scriptsize"))
    f = A / "exploratory" / "P5_P0_stage_steps.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = [[LADDER.get(r.ladder, esc(r.ladder)), step_name(r.step), f"{num(r.delta_b_change)} ({num(r.delta_b_change_se)})", pval(r.delta_b_change_p),
                 f"{num(r.support_shift_change)} ({num(r.support_shift_change_se)})"] for r in d.itertuples()]
        write("stage_steps.tex", table(rows, ["Ladder", "Step", "change in $\\Delta_b$ (SE)", "$p$", "change in support (SE)"], "lllll",
                                       "Training-stage steps in chat format (same trials, stacked regression).", "tab:steps",
                                       size="\\scriptsize"))
    for key, cap, lab in (("P3_system_prompt", "Debiasing system prompt vs.\\ original (same trials).", "tab:sp"),
                          ("P6_prompt_wording", "Reworded prompts vs.\\ original on the same third of trials.", "tab:pv")):
        f = A / "exploratory" / f"{key}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if key == "P3_system_prompt":
            rows = [[name(r.model), num(r.delta_b_original), num(r.delta_b_sp), f"{num(r.delta_b_change)} ({num(r.delta_b_change_se)})",
                     pval(r.delta_b_change_p), num(r.beta_original), num(r.beta_sp)] for r in d.itertuples()]
            hdr = ["Model", "$\\Delta_b$ orig.", "$\\Delta_b$ prompt", "change (SE)", "$p$", "$\\beta$ orig.", "$\\beta$ prompt"]
            spec = "lrrlrrr"
        else:
            d = d.dropna(subset=["delta_b_variant"])
            rows = [[name(r.variant), ci(r.delta_b_original, r.delta_b_lo, r.delta_b_hi), num(r.delta_b_variant),
                     f"{num(r.delta_b_change)} ({num(r.delta_b_change_se)})", pval(r.delta_b_change_p),
                     num(r.support_original), num(r.support_variant)] for r in d.itertuples()]
            hdr = ["Variant", "$\\Delta_b$ orig. [95\\% CI]", "$\\Delta_b$ variant", "change (SE)", "$p$", "support orig.", "support var."]
            spec = "llrlrrr"
        write(f"{key}.tex", table(rows, hdr, spec, cap, lab, size="\\scriptsize"))


    f = A / "exploratory" / "P4_stage_probes_late_layers.csv"
    if f.exists():
        d = pd.read_csv(f)
        rows = [[name(r.stage), num(r.probe_acc), num(r.majority), num(r.probe_auc), num(r.cos_raw_to_instruct, 2),
                 num(r.cos_hat_k8_to_instruct, 2)] for r in d.itertuples()]
        write("p4.tex", table(rows, ["Stage", "probe acc.", "majority", "AUC", "cos raw", "cos hat$_{k8}$"], "lrrrrr",
                              "P4: contradiction signal and bias direction across OLMo 2 7B training stages (layers 18--26, mean).",
                              "tab:p4", note="Probe: logistic regression, held-out topics, contradicted vs.\\ supported trials. Cosine: "
                              "each stage's bias direction vs.\\ OLMo Instruct's saved RQ3 direction (Instruct row: labels-only rebuild, a "
                              "sanity check).", size="\\scriptsize"))
    f = ROOT / "results" / "p4_v2" / "p4_v2_summary.csv"
    if f.exists():   # P4 rerun at the answer position (rq3v2.run_p4 / p4_summary)
        d = pd.read_csv(f)
        late = d[d.layer.between(18, 26)].groupby("stage", sort=False)[
            ["probe_acc", "majority", "probe_auc", "cos_raw_to_instruct", "cos_hat_k8_to_instruct"]].mean()
        late = late.reindex([s for s in ("olmo_sft", "olmo_dpo", "olmo_instruct") if s in late.index])
        rows = [[name(s), num(r.probe_acc), num(r.majority), num(r.probe_auc), num(r.cos_raw_to_instruct, 2),
                 num(r.cos_hat_k8_to_instruct, 2)] for s, r in late.iterrows()]
        write("p4v2.tex", table(rows, ["Stage", "probe acc.", "majority", "AUC", "cos raw", "cos hat$_{k8}$"], "lrrrrr",
                                "P4 rerun at the answer position (layers 18--26, mean).", "tab:p4v2",
                                note="As Table~\\ref{tab:p4}, from activations at the position that produced the answer. SFT has no "
                                     "bias direction: none of its contradicted trials resisted the belief (Section~\\ref{sec:exploratory}). "
                                     "Instruct's cosines are 1 by construction (its own direction).", size="\\scriptsize"))
    # compute
    f = ROOT / "results" / "compute" / "farmshare_jobs.csv"
    if f.exists():
        j = pd.read_csv(f)
        j["model"] = j.model.fillna("")
        g = j.groupby(["phase", "task"]).agg(jobs=("job_id", "size"), runs=("runs", "sum"), gpu_hours=("gpu_hours", "sum")).reset_index()
        g = g[g.gpu_hours > 0].sort_values(["phase", "gpu_hours"], ascending=[True, False])
        rows = [[esc(r.phase), esc(r.task), str(int(r.jobs)), str(int(r.runs)), num(r.gpu_hours, 1)] for r in g.itertuples()]
        rows.append(["\\textbf{total}", "", str(int(j.job_id.nunique())), str(int(j.runs.sum())), f"\\textbf{{{j.gpu_hours.sum():.1f}}}"])
        write("compute.tex", table(rows, ["Phase", "Task", "jobs", "runs", "GPU-h"], "llrrr",
                                   "Compute on Stanford FarmShare (NVIDIA L40S 48 GB). Tasks with zero GPU time omitted; totals include all jobs.",
                                   "tab:compute", size="\\scriptsize", longt=True))



def tex_text(s):
    """Escape free text (topic titles, cue wording) for LaTeX."""
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace('"', "''")


def design_tables():
    """Appendix: every topic (both tiers) and every identity cue (label and first vignette per group)."""
    import yaml
    data = ROOT / "data"
    rows = []
    for f in ("topics.csv", "tier1_topics.csv"):
        d = pd.read_csv(data / f, keep_default_na=False)
        for r in d.itertuples():
            dims = r.dims_engaged.replace(";", ", ") if r.dims_engaged else "--"
            pew = "yes" if str(getattr(r, "pew_item", "")).strip() else ""
            rows.append([str(r.tier), tex_text(r.title), tex_text(r.claim_pos), dims, pew])
    write("topics.tex", table(rows, ["Tier", "Topic", "$H_1$ (one direction; the other is its negation)", "dimensions engaged",
                                     "Pew"], "llp{6.2cm}p{3.2cm}l",
                              f"All {len(rows)} topics.", "tab:topics",
                              note="Tier~2: socially meaningful policy topics; Tier~1: technical topics without social meaning. "
                                   "Dimensions engaged: identity dimensions the topic is paired with (pairing rule). Pew: a "
                                   "survey item gives the survey stance $g_{\\text{svy}}$.", size="\\scriptsize", longt=True))
    labels = yaml.safe_load((data / "cues" / "labels.yaml").read_text())
    vign = yaml.safe_load((data / "cues" / "vignettes.yaml").read_text())
    first = {}
    for v in vign:
        if v.get("status") == "final":
            first.setdefault(v["group_id"], v["text"])
    rows = [[l["dimension"], tex_text(l["text"]), tex_text(first.get(l["group_id"], "--"))] for l in labels]
    write("cues.tex", table(rows, ["Dimension", "Label", "Example vignette (first of the group's final vignettes)"],
                            "llp{9.4cm}", "Identity cues: the explicit label and one vignette for each of the 22 groups.",
                            "tab:cues", note="The Unspecified control has no identity sentence. All vignettes passed the "
                            "identification and leakage checks in the pilot; trials whose vignette failed for a given model "
                            "are dropped for that model.", size="\\scriptsize", longt=True))

if __name__ == "__main__":
    main()
