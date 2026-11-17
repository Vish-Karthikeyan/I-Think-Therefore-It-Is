"""Step 17: local evidence slopes by evidence strength (reviewer check on nonlinearity).

    python analysis/17_local_delta_b.py

The primary model (Eq. 3, common.BASE_TERMS) has one evidence slope per condition: beta (no belief),
beta + Delta_b (contradicting), beta + s (supporting). Spline fits show the x -> update relation is nonlinear,
so a single Delta_b could average opposite effects at weak and strong evidence. Here the three evidence slopes
are allowed to differ by |x| bin (4 bins of roughly equal size, cut at natural gaps in the 18 distinct |x|
values); all other terms (prior, congruence, d>0 slope, cue type, source) are kept as in the primary model.
Within a bin the slope is a local (chord) slope across that range of evidence strength.

  lp_after ~ lp_before + sum_k [ x:B_k + x:contra:B_k + x:support:B_k ] + x:contra:m + x:support:m
             + x:dpos + x:C(cue_type) + x:C(source)          [+ C(bin) intercepts in the robustness variant]

Reported per model (tier 2, chat): local beta_k, Delta_b_k, s_k with topic-clustered SEs; the pooled Delta_b
from the primary model for reference; a Wald test that Delta_b is equal across bins; and, where beta_k is
clearly positive, the share of the no-belief slope removed on contradicting evidence (-Delta_b_k / beta_k).
Writes analysis/output/local_slopes/{local_slopes.csv, equality_tests.csv} and figure F20.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_TERMS, analysis_frame, estimate, fit_ols, load_dataset, out_dir  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ["gpt", "claude", "gemini", "phi4", "llama", "olmo_instruct", "olmo13_instruct", "tulu_final"]
NAME = {"gpt": "GPT-4o", "claude": "Claude Sonnet 5", "gemini": "Gemini 2.5 Flash", "phi4": "Phi-4",
        "llama": "Llama 3.1 8B", "olmo_instruct": "OLMo 2 7B Instr.", "olmo13_instruct": "OLMo 2 13B Instr.",
        "tulu_final": "Tulu 3 8B"}
EDGES = [0.0005, 0.6, 1.4, 2.1, 3.0]            # bins: 0.32-0.49 | 0.81-1.29 | 1.61-2.03 | 2.26-2.91
BIN_LABEL = ["weak\n0.32-0.49", "moderate\n0.81-1.29", "strong\n1.61-2.03", "very strong\n2.26-2.91"]
K = len(EDGES) - 1
FIG = Path(__file__).resolve().parents[2] / "paper" / "secondary" / "figures" / "candidates"


def terms(intercepts: bool) -> str:
    # keep the x main effect (bin 0 = reference) so x:C(cue_type) / x:C(source) keep their reduced coding;
    # local beta_k = coef(x) + coef(x:B_k), k >= 1
    beta = "x + " + " + ".join(f"x:B{k}" for k in range(1, K))
    slope = " + ".join(f"x:contra:B{k} + x:support:B{k}" for k in range(K))
    rest = " + x:contra:m + x:support:m + x:dpos + x:C(cue_type) + x:C(source)"
    return "lp_before + " + beta + " + " + slope + rest + (" + C(xbin)" if intercepts else "")


def local_beta(fit, k: int) -> dict:
    if k == 0:
        return estimate(fit, "x")
    c = np.zeros(len(fit.params)); names = list(fit.params.index)
    c[names.index("x")] = 1; c[names.index(f"x:B{k}")] = 1
    est = float(c @ fit.params.values); se = float(np.sqrt(c @ fit.cov_params().values @ c))
    return dict(est=est, se=se)


def fit_model(d: pd.DataFrame, intercepts: bool):
    d = d[d.x.abs() > EDGES[0]].copy()
    d["xbin"] = pd.cut(d.x.abs(), EDGES, labels=False)
    for k in range(K):
        d[f"B{k}"] = (d.xbin == k).astype(float)
    return fit_ols(d, terms(intercepts)), d


def main() -> None:
    df = load_dataset("main")
    od = out_dir("local_slopes")
    rows, tests = [], []
    for m in MODELS:
        d = analysis_frame(df, tier=2, model=m, fmt="chat")
        if d.empty:
            continue
        n_x0 = int((d.x.abs() <= EDGES[0]).sum())
        pooled = estimate(fit_ols(d), "x:contra")
        for intercepts in (False, True):
            fit, dd = fit_model(d, intercepts)
            spec = "bin_intercepts" if intercepts else "primary"
            for k in range(K):
                b, db, s = (local_beta(fit, k), estimate(fit, f"x:contra:B{k}"), estimate(fit, f"x:support:B{k}"))
                mx = float(dd.loc[dd.xbin == k, "x"].abs().mean())
                rows.append(dict(model=m, spec=spec, bin=k, abs_x_range=BIN_LABEL[k].split("\n")[1],
                                 n=int((dd.xbin == k).sum()), beta=b["est"], beta_se=b["se"],
                                 delta_b=db["est"], delta_b_se=db["se"], delta_b_p=db["p"],
                                 s=s["est"], s_se=s["se"], mean_abs_x=mx,
                                 implied_shift=db["est"] * mx, implied_shift_se=db["se"] * mx,
                                 share_removed=(-db["est"] / b["est"]) if b["est"] > 2 * b["se"] and b["est"] > 0.05 else np.nan,
                                 pooled_delta_b=pooled["est"], pooled_delta_b_se=pooled["se"], n_x0_dropped=n_x0))
            w = fit.wald_test(", ".join(f"x:contra:B{k} = x:contra:B{k+1}" for k in range(K - 1)), use_f=True, scalar=True)
            tests.append(dict(model=m, spec=spec, test="Delta_b equal across bins", F=float(w.statistic),
                              p=float(w.pvalue), df_num=K - 1))
            mxs = [float(dd.loc[dd.xbin == k, "x"].abs().mean()) for k in range(K)]
            w2 = fit.wald_test(", ".join(f"{mxs[k]:.6f}*x:contra:B{k} = {mxs[k+1]:.6f}*x:contra:B{k+1}" for k in range(K - 1)),
                               use_f=True, scalar=True)
            tests.append(dict(model=m, spec=spec, test="Delta_b*|x| (constant shift) equal across bins",
                              F=float(w2.statistic), p=float(w2.pvalue), df_num=K - 1))
        print(f"  {m}: done (dropped {n_x0} trials with x = 0)")
    R = pd.DataFrame(rows); T = pd.DataFrame(tests)
    R.to_csv(od / "local_slopes.csv", index=False); T.to_csv(od / "equality_tests.csv", index=False)
    print(f"wrote {od/'local_slopes.csv'} and equality_tests.csv")
    write_table(R, T)

    P = R[R.spec == "primary"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, squeeze=False)
    for ax, m in zip(axes.ravel(), [x for x in MODELS if x in set(P.model)]):
        q = P[P.model == m].sort_values("bin"); xs = np.arange(K)
        ax.errorbar(xs - 0.08, q.beta, yerr=1.96 * q.beta_se, fmt="s--", color="#999999", ms=5, capsize=2,
                    label=r"no-belief slope $\beta$")
        ax.errorbar(xs + 0.08, q.delta_b, yerr=1.96 * q.delta_b_se, fmt="o-", color="#D55E00", ms=6, capsize=2,
                    label=r"$\Delta_b$ (contradicting)")
        pb = q.pooled_delta_b.iloc[0]
        ax.axhline(pb, color="#D55E00", lw=0.9, ls=":", label=r"pooled $\Delta_b$")
        ax.axhline(0, color="black", lw=0.6)
        t = T[(T.model == m) & (T.spec == "primary") & (T.test == "Delta_b equal across bins")].iloc[0]
        ax.set_title(f"{NAME[m]}\nequal-$\\Delta_b$ test p={t.p:.2f}", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(BIN_LABEL, fontsize=7); ax.tick_params(axis="y", labelsize=7)
    axes[0][0].legend(fontsize=7, loc="best")
    for r in (0, 1):
        axes[r][0].set_ylabel("slope on x (log-odds per unit x)", fontsize=8)
    fig.suptitle(r"Local evidence slopes by evidence strength $|x|$ (tier 2, chat; 95% CIs, topic-clustered)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"F20_local_delta_b.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {FIG/'F20_local_delta_b.pdf'}")


def write_table(R: pd.DataFrame, T: pd.DataFrame) -> None:
    """LaTeX table for the paper: local Delta_b (SE), implied shift Delta_b*|x|, and the two equality tests."""
    P = R[R.spec == "primary"]
    fmtp = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"  # noqa: E731
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Local belief-following by evidence strength (Tier~2, chat; primary model with the evidence slopes "
             r"varying across four $|x|$ bins, topic-clustered SEs). Left: $\Delta_b$ in each bin. Right: the implied "
             r"constant shift toward the belief, $-\Delta_b\cdot\overline{|x|}$ (log-odds; positive = toward the belief). Last columns: $p$ for equal "
             r"$\Delta_b$ across bins and for an equal implied shift. Mean $|x|$ per bin: 0.42, 1.04, 1.84, 2.67. "
             r"Claude and Gemini use sampled answers.}\label{tab:local-slopes}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l cccc cccc cc}", r"\toprule",
             r" & \multicolumn{4}{c}{$\Delta_b$ (SE)} & \multicolumn{4}{c}{implied shift $-\Delta_b\cdot\overline{|x|}$} & "
             r"\multicolumn{2}{c}{$p$, equal across bins} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-11}",
             r"Model & weak & moderate & strong & very strong & weak & moderate & strong & very strong & $\Delta_b$ & shift \\",
             r"\midrule"]
    for m in [x for x in MODELS if x in set(P.model)]:
        q = P[P.model == m].sort_values("bin")
        db = [f"{a:.2f} ({b:.2f})" for a, b in zip(q.delta_b, q.delta_b_se)]
        sh = [f"{-a:.2f}" for a in q.implied_shift]          # positive = toward the belief
        t1 = T[(T.model == m) & (T.spec == "primary") & (T.test == "Delta_b equal across bins")].p.iloc[0]
        t2 = T[(T.model == m) & (T.spec == "primary") & (T.test.str.startswith("Delta_b*|x|"))].p.iloc[0]
        lines.append(f"{NAME[m].replace('Tulu', chr(84)+chr(92)+chr(34)+'ulu')} & " + " & ".join(db + sh) + f" & {fmtp(t1)} & {fmtp(t2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    out = Path(__file__).resolve().parents[2] / "paper" / "secondary" / "tables" / "local_slopes.tex"
    out.write_text("\n".join(lines) + "\n"); print(f"wrote {out}")


if __name__ == "__main__":
    main()
