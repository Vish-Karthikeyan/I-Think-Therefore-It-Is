"""Step 19: does belief-following differ on heuristic-conflicting items? (reviewer check on numeracy demands)

    python analysis/19_heuristic_conflict.py

`heuristic_conflict` (hc) is True when a table's raw success counts point the opposite way from its success
rates (about half the non-null tables). On those items a count-based shortcut gives the wrong answer, so they
demand more numeracy. The reviewer asks whether beta (no-belief evidence slope) or Delta_b (the change in that
slope when the evidence contradicts the user's belief) differ between conflicting and consistent items.

One model per model x tier (chat format), the primary model (common.BASE_TERMS) with every evidence slope of
interest interacted with hc; the x main effect is KEPT (consistent items = reference), so the control slopes
x:C(cue_type), x:C(source) keep their usual reduced coding:

  lp_after ~ lp_before + hc + x + x:hc + x:contra + x:contra:hc + x:support + x:support:hc
             + x:contra:m + x:contra:m:hc + x:support:m + x:support:m:hc + x:dpos + x:C(cue_type) + x:C(source)

Per stratum: beta, Delta_b, s (support slope), theta_cong; the conflicting - consistent difference is the :hc
coefficient (topic-clustered SEs). Delta_b behaves roughly like a constant shift toward the belief, so we also
report the implied shift -Delta_b * mean|x| (mean over contradicting trials of that stratum; mean |x| differs a
little between strata) and test its difference. Robustness: (a) the controls x:dpos, x:C(cue_type), x:C(source)
also interacted with hc; (b) separate fits of the primary model in each stratum.

Count-following check: on no-belief trials, the share of updates (y = lp_after - lp_before, y != 0) that move
in the direction the RATES favor (= sign of x) vs the direction the raw COUNTS favor; on conflicting items the
two are opposite, so a share below 0.5 means the model follows the counts.

Also reproduces figure F5 (12_candidate_figures.py: polyfit of y on x, no-belief trials, both tiers, chat).

Writes analysis/output/heuristic_conflict/*.csv, paper/secondary/tables/heuristic_conflict.tex and
paper/secondary/figures/candidates/F22_heuristic_conflict.{pdf,png}.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import analysis_frame, fit_ols, load_dataset, out_dir  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ["gpt", "claude", "gemini", "phi4", "llama", "olmo_sft", "olmo_dpo", "olmo_instruct",
          "olmo13_instruct", "tulu_final"]
NAME = {"gpt": "GPT-4o", "claude": "Claude Sonnet 5", "gemini": "Gemini 2.5 Flash", "phi4": "Phi-4",
        "llama": "Llama 3.1 8B", "olmo_sft": "OLMo 2 7B SFT", "olmo_dpo": "OLMo 2 7B DPO",
        "olmo_instruct": "OLMo 2 7B Instr.", "olmo13_instruct": "OLMo 2 13B Instr.", "tulu_final": "Tulu 3 8B"}
SAMPLED = {"claude", "gemini"}
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "secondary" / "figures" / "candidates"
TAB = ROOT / "paper" / "secondary" / "tables" / "heuristic_conflict.tex"

SLOPES = {"beta": "x", "delta_b": "x:contra", "s": "x:support", "theta_cong": "x:contra:m"}
PRIMARY = ("lp_before + hc + x + x:hc + x:contra + x:contra:hc + x:support + x:support:hc"
           " + x:contra:m + x:contra:m:hc + x:support:m + x:support:m:hc"
           " + x:dpos + x:C(cue_type) + x:C(source)")
FULL = PRIMARY + " + x:dpos:hc + x:C(cue_type):hc + x:C(source):hc"
BASE = ("lp_before + x + x:contra + x:support + x:contra:m + x:support:m"
        " + x:dpos + x:C(cue_type) + x:C(source)")


def hc_name(term: str) -> str:
    """statsmodels name of `term` interacted with hc (patsy puts hc last for x:hc, x:contra:hc, ...)."""
    return f"{term}:hc"


def lincomb(fit, weights: dict[str, float]) -> tuple[float, float, float]:
    names = list(fit.params.index); c = np.zeros(len(names))
    for k, w in weights.items():
        c[names.index(k)] = w
    est = float(c @ fit.params.values); se = float(np.sqrt(c @ fit.cov_params().values @ c))
    return est, se, float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan


def check_names(fit) -> None:
    for t in SLOPES.values():
        assert t in fit.params and hc_name(t) in fit.params, (t, list(fit.params.index))
    assert "x" in fit.params


def fit_interacted(d: pd.DataFrame, terms: str):
    d = d.copy(); d["hc"] = d.heuristic_conflict.astype(float)
    fit = fit_ols(d, terms); check_names(fit)
    return fit, d


def stratum_rows(fit, d, model, tier, spec):
    rows = []
    mx = {k: float(d.loc[(d.contra == 1) & (d.hc == k), "x"].abs().mean()) for k in (0.0, 1.0)}
    mx_all = {k: float(d.loc[d.hc == k, "x"].abs().mean()) for k in (0.0, 1.0)}
    for par, t in SLOPES.items():
        c0 = lincomb(fit, {t: 1}); c1 = lincomb(fit, {t: 1, hc_name(t): 1}); df_ = lincomb(fit, {hc_name(t): 1})
        rows.append(dict(model=model, tier=tier, spec=spec, param=par,
                         consistent=c0[0], consistent_se=c0[1], conflicting=c1[0], conflicting_se=c1[1],
                         diff=df_[0], diff_se=df_[1], diff_p=df_[2]))
    # direction-averaged beta: the x slope averaged over evidence favoring H1 (dpos = 1) and H0 (x:dpos is a
    # separate slope for d > 0; it is shared across strata in the primary spec, stratum-specific in the full spec)
    p = {k: float(d.loc[d.hc == k, "dpos"].mean()) for k in (0.0, 1.0)}
    inter = "x:dpos:hc" in fit.params
    w0 = {"x": 1, "x:dpos": p[0.0]}
    w1 = {"x": 1, "x:hc": 1, "x:dpos": p[1.0], **({"x:dpos:hc": p[1.0]} if inter else {})}
    wd = {"x:hc": 1, "x:dpos": p[1.0] - p[0.0], **({"x:dpos:hc": p[1.0]} if inter else {})}
    c0, c1, cd = lincomb(fit, w0), lincomb(fit, w1), lincomb(fit, wd)
    rows.append(dict(model=model, tier=tier, spec=spec, param="beta_dir_avg", consistent=c0[0], consistent_se=c0[1],
                     conflicting=c1[0], conflicting_se=c1[1], diff=cd[0], diff_se=cd[1], diff_p=cd[2]))
    # implied shift toward the belief: -Delta_b * mean|x| (contradicting trials of each stratum)
    t = SLOPES["delta_b"]
    s0 = lincomb(fit, {t: -mx[0.0]}); s1 = lincomb(fit, {t: -mx[1.0], hc_name(t): -mx[1.0]})
    sd = lincomb(fit, {t: -(mx[1.0] - mx[0.0]), hc_name(t): -mx[1.0]})
    rows.append(dict(model=model, tier=tier, spec=spec, param="implied_shift",
                     consistent=s0[0], consistent_se=s0[1], conflicting=s1[0], conflicting_se=s1[1],
                     diff=sd[0], diff_se=sd[1], diff_p=sd[2]))
    for r in rows:
        r.update(n=int(len(d)), n_conflicting=int((d.hc == 1).sum()), n_topics=int(d.topic.nunique()),
                 mean_abs_x_contra_consistent=mx[0.0], mean_abs_x_contra_conflicting=mx[1.0],
                 mean_abs_x_consistent=mx_all[0.0], mean_abs_x_conflicting=mx_all[1.0],
                 sampled_readout=model in SAMPLED)
    return rows


def separate_rows(d, model, tier):
    """Robustness: the primary model fitted separately in each stratum (no shared prior/control terms)."""
    out = {}
    for k, lab in ((False, "consistent"), (True, "conflicting")):
        f = fit_ols(d[d.heuristic_conflict == k].reset_index(drop=True), BASE)
        out[lab] = {p: (float(f.params[t]), float(f.bse[t])) for p, t in SLOPES.items()}
    rows = []
    for p in SLOPES:
        (a, sa), (b, sb) = out["consistent"][p], out["conflicting"][p]
        rows.append(dict(model=model, tier=tier, spec="separate_fits", param=p, consistent=a, consistent_se=sa,
                         conflicting=b, conflicting_se=sb, diff=b - a, diff_se=np.nan, diff_p=np.nan))
    return rows


def count_following(df: pd.DataFrame) -> pd.DataFrame:
    """Direction of the update relative to rates vs counts, by stratum and evidence condition (tier x model)."""
    rows = []
    for tier in (2, 1):
        for m in MODELS:
            d = analysis_frame(df, tier=tier, model=m, fmt="chat")
            if d.empty:
                continue
            d = d[d.x != 0]
            cdir = np.sign(d.k_t - d.k_c) * d.h_dir      # direction the raw counts favor for H1
            mv = np.sign(d.y)
            for hc in (False, True):
                for e in ("neutral", "contra", "support"):
                    s = (d.heuristic_conflict == hc) & (d.e == e)
                    moved = s & (mv != 0)
                    rows.append(dict(model=m, tier=tier, stratum="conflicting" if hc else "consistent", e=e,
                                     n=int(s.sum()), share_moved=float(moved.sum() / max(s.sum(), 1)),
                                     share_toward_rates=float((mv[moved] == d.d[moved]).mean()),
                                     share_toward_counts=float((mv[moved] == cdir[moved]).mean()),
                                     share_counts_tied=float((cdir[s] == 0).mean()),
                                     mean_y_signed_by_rates=float((d.y[s] * d.d[s]).mean())))
    return pd.DataFrame(rows)


def reproduce_f5(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df.study == "main") & (df.fmt == "chat") & (~df.excluded) & (df.b == 0) & (df.e != "null")]
    rows = []
    for m in ["gpt", "llama", "olmo_instruct", "olmo_dpo", "olmo_sft"]:
        for conf, lab in [(False, "consistent"), (True, "conflicting")]:
            s = d[(d.model == m) & (d.heuristic_conflict == conf)]
            rows.append(dict(model=m, kind=lab, beta_polyfit=float(np.polyfit(s.x, s.y, 1)[0]), n=len(s)))
    return pd.DataFrame(rows)


def main() -> None:
    df = load_dataset("main")
    od = out_dir("heuristic_conflict")
    f5 = reproduce_f5(df); f5.to_csv(od / "f5_reproduction.csv", index=False)
    print(f5.round(3).to_string(index=False))

    rows = []
    for tier in (2, 1):
        for m in MODELS:
            d = analysis_frame(df, tier=tier, model=m, fmt="chat")
            if d.empty:
                print(f"  skip {m} tier {tier} (no chat data)"); continue
            fit, dd = fit_interacted(d, PRIMARY); rows += stratum_rows(fit, dd, m, tier, "primary")
            fit2, dd2 = fit_interacted(d, FULL); rows += stratum_rows(fit2, dd2, m, tier, "controls_interacted")
            rows += separate_rows(d, m, tier)
            print(f"  {m} tier {tier}: n={len(d)}")
    R = pd.DataFrame(rows); R.to_csv(od / "stratified_estimates.csv", index=False)
    C = count_following(df); C.to_csv(od / "count_following.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        P = R[R.spec == "primary"]
        for par in ("beta", "beta_dir_avg", "delta_b", "implied_shift", "s", "theta_cong"):
            print(f"\n== {par} (primary)")
            print(P[P.param == par][["model", "tier", "consistent", "consistent_se", "conflicting",
                                     "conflicting_se", "diff", "diff_se", "diff_p"]].round(3).to_string(index=False))
        print("\n== count following (neutral trials, share of moves toward rates / counts)")
        print(C[C.e == "neutral"].round(3).to_string(index=False))
    write_table(R)
    figure(R)


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def num(v, d=2):
    s = f"{v:.{d}f}"
    if s == "-" + "0." + "0" * d:      # no negative zero
        s = s[1:]
    return s.replace("-", "$-$")


def write_table(R: pd.DataFrame) -> None:
    P = R[R.spec == "primary"]
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\caption{Belief-following on heuristic-conflicting vs.\ heuristic-consistent items (chat). One fit per "
             r"model and tier: the primary model with $\beta$, $\Delta_b$, $s$ and $\theta_{\text{cong}}$ (and the "
             r"supporting congruence slope) interacted with an indicator for heuristic conflict (raw counts point opposite to the rates); "
             r"topic-clustered SEs. Cons.\ = consistent, Confl.\ = conflicting; $p$ tests Confl.\ $-$ Cons. Shift = "
             r"implied constant shift toward the belief, $-\Delta_b\cdot\overline{|x|}$ over contradicting trials of the "
             r"stratum (log-odds). Claude and Gemini use sampled answers; Claude has 5{,}000 Tier~2 and 1{,}000 Tier~1 "
             r"trials.}\label{tab:hc}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{$\beta$ (SE)} & \multicolumn{3}{c}{$\Delta_b$ (SE)} & "
             r"\multicolumn{3}{c}{Shift $-\Delta_b\cdot\overline{|x|}$} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Model & Cons. & Confl. & $p$ & Cons. & Confl. & $p$ & Cons. & Confl. & $p$ \\", r"\midrule"]
    for tier, lab in ((2, r"\textit{Tier 2 (socially meaningful topics; primary)}"),
                      (1, r"\textit{Tier 1 (control topics)}")):
        lines.append(rf"\multicolumn{{10}}{{l}}{{{lab}}} \\")
        for m in MODELS:
            q = P[(P.model == m) & (P.tier == tier)].set_index("param")
            if q.empty:
                continue
            cells = []
            for par in ("beta", "delta_b"):
                r = q.loc[par]
                cells += [f"{num(r.consistent)} ({r.consistent_se:.2f})", f"{num(r.conflicting)} ({r.conflicting_se:.2f})",
                          fmt_p(r.diff_p)]
            r = q.loc["implied_shift"]
            cells += [num(r.consistent), num(r.conflicting), fmt_p(r.diff_p)]
            nm = NAME[m].replace("Tulu", "T\\\"ulu") + ("$^{\\dagger}$" if m in SAMPLED else "")
            lines.append(nm + " & " + " & ".join(cells) + r" \\")
        if tier == 2:
            lines.append(r"\midrule")
    mx = {t: (P[P.tier == t].mean_abs_x_contra_consistent.agg(["min", "max"]).values,
              P[P.tier == t].mean_abs_x_contra_conflicting.agg(["min", "max"]).values) for t in (2, 1)}
    rng = lambda a: f"{a[0]:.2f}" if f"{a[0]:.2f}" == f"{a[1]:.2f}" else f"{a[0]:.2f}--{a[1]:.2f}"  # noqa: E731
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\par\vspace{2pt}{\footnotesize $^{\dagger}$Sampled readout (noisier). Mean $|x|$ on contradicting trials "
              rf"is nearly the same in both strata (Tier~2: {rng(mx[2][0])} consistent vs.\ {rng(mx[2][1])} conflicting; "
              rf"Tier~1: {rng(mx[1][0])} vs.\ {rng(mx[1][1])}, across models), so the shift column mainly rescales $\Delta_b$. "
              r"$\beta$ is the coefficient on $x$ as in the primary model; averaging it over evidence direction "
              r"(the $x\cdot\mathbb{1}[d>0]$ slope) gives the same conflicting $-$ consistent differences within 0.07.}",
              r"\end{table}"]
    TAB.write_text("\n".join(lines) + "\n"); print(f"wrote {TAB}")


def figure(R: pd.DataFrame) -> None:
    P = R[(R.spec == "primary") & (R.tier == 2)]
    models = [m for m in MODELS if m in set(P.model)]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(models))[::-1]
    styles = [("consistent", "#0072B2", "o", "white", "heuristic-consistent"),
              ("conflicting", "#D55E00", "s", None, "heuristic-conflicting")]
    for ax, (par, title) in zip(axes, [("beta", r"no-belief slope $\beta$"),
                                       ("delta_b", r"$\Delta_b$ (contradicting $-$ no belief)"),
                                       ("implied_shift", r"implied shift toward belief $-\Delta_b\cdot\overline{|x|}$")]):
        for j, (col, color, mk, face, lab) in enumerate(styles):
            v = [P[(P.model == m) & (P.param == par)].iloc[0] for m in models]
            est = np.array([r[col] for r in v]); se = np.array([r[col + "_se"] for r in v])
            ax.errorbar(est, y + (0.17 if j == 0 else -0.17), xerr=1.96 * se, fmt=mk, color=color, ms=5.5,
                        mfc=face if face else color, mec=color, capsize=2, lw=1.1, label=lab)
        for i, m in enumerate(models):
            p = P[(P.model == m) & (P.param == par)].iloc[0].diff_p
            if p < 0.05:
                ax.annotate("*" if p >= 0.001 else "***", xy=(1.01, y[i]), xycoords=("axes fraction", "data"),
                            ha="left", va="center", fontsize=10, annotation_clip=False)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("log-odds per unit x" if par != "implied_shift" else "log-odds")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([NAME[m] + (" (sampled)" if m in SAMPLED else "") for m in models])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Belief-following on heuristic-conflicting vs. consistent items (Tier 2, chat; 95% CIs, "
                 "topic-clustered; * conflicting $-$ consistent p<0.05, *** p<0.001)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"F22_heuristic_conflict.{ext}", bbox_inches="tight", dpi=220)
    print(f"wrote {FIG/'F22_heuristic_conflict.pdf'}")


if __name__ == "__main__":
    main()
