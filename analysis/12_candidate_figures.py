"""Step 12: candidate figures for the paper (a variety, so the best can be picked for the primary).

    python analysis/12_candidate_figures.py

Writes PDF + PNG into paper/secondary/figures/candidates/. Each figure stands alone; a figure is
skipped (with a note) if its input is missing. Colorblind-safe Okabe-Ito palette, one y-axis per panel,
direct labels where they fit. These are static paper figures (matplotlib), so the dataviz principles
(form first, CVD-safe hues, no dual axis, legend/labels not color-alone) are applied in that medium.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
A = ROOT / "analysis" / "output"
OUT = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", grey="#999999")
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300, "font.size": 11, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})
NAME = {"gpt": "GPT-4o", "claude": "Claude S5", "gemini": "Gemini 2.5F", "llama": "Llama 3.1 8B",
        "olmo_base": "OLMo base", "olmo_sft": "OLMo SFT", "olmo_dpo": "OLMo DPO", "olmo_instruct": "OLMo Instr.",
        "tulu_base": "Tulu base", "tulu_sft": "Tulu SFT", "tulu_dpo": "Tulu DPO", "tulu_final": "Tulu final",
        "phi4": "Phi-4", "olmo13_instruct": "OLMo13 Instr."}
MODELNAME = {"olmo_instruct": "OLMo 2 7B Instruct", "llama": "Llama 3.1 8B Instruct", "tulu_final": "Tulu 3 final"}


def save(fig, stem, caption):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png — {caption}")


def read(p):
    f = A / p
    return pd.read_csv(f) if f.exists() else None


# F1: Delta_b by model (main run, chat) with 95% CI — the headline belief-following effect
def f1_delta_b_main():
    d = read("rq1/per_model_main.csv")
    if d is None: return print("  skip F1 (no rq1/per_model_main)")
    d = d[d.fmt == "chat"].copy().sort_values("delta_b")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(d))
    err = np.vstack([d.delta_b - d.delta_b_lo, d.delta_b_hi - d.delta_b]).clip(min=0)
    ax.errorbar(d.delta_b, y, xerr=err, fmt="o", color=OI["blue"], ecolor=OI["grey"], capsize=3, ms=7)
    ax.axvline(0, color=OI["black"], lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME.get(m, m) for m in d.model])
    ax.set_xlabel(r"$\Delta_b$  (change in evidence slope when evidence contradicts the stated belief)")
    ax.set_title("Belief-following: a stated belief flattens the evidence slope (main run, chat)")
    for yi, (v, hi) in zip(y, zip(d.delta_b, d.delta_b_hi)):
        ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    save(fig, "F1_delta_b_main", "Delta_b per model with 95% CI")


# F2: no-belief evidence slope beta by model — context for the very low beta in open models
def f2_beta():
    dm = read("rq1/per_model_main.csv"); de = read("rq1/per_model_exploratory.csv")
    if dm is None: return print("  skip F2")
    d = pd.concat([dm[dm.fmt == "chat"], de[de.fmt == "chat"]] if de is not None else [dm[dm.fmt == "chat"]])
    keep = ["gpt", "gemini", "claude", "llama", "olmo_instruct", "tulu_final", "phi4", "olmo13_instruct"]
    d = d[d.model.isin(keep)].copy().sort_values("beta_neutral")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = [OI["vermillion"] if b < 0.1 else OI["blue"] for b in d.beta_neutral]
    ax.barh(np.arange(len(d)), d.beta_neutral, color=colors, height=0.66)
    ax.axvline(1.0, color=OI["black"], ls="--", lw=0.9); ax.text(1.0, len(d)-0.4, " Bayesian = 1", fontsize=8, va="top")
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels([NAME.get(m, m) for m in d.model])
    ax.set_xlabel(r"no-belief evidence slope $\beta$ (0 = ignores evidence, 1 = Bayesian)")
    ax.set_title("Most open models barely use the evidence even with no belief stated")
    for i, v in enumerate(d.beta_neutral):
        ax.annotate(f"{v:.2f}", (v, i), textcoords="offset points", xytext=(4, 0), va="center", fontsize=9)
    save(fig, "F2_beta_neutral", "no-belief slope beta per model")


# F3: theta_cong by model with CI — the congruence effect (null / opposite sign)
def f3_theta():
    d = read("rq1/per_model_main.csv")
    if d is None: return print("  skip F3")
    d = d[d.fmt == "chat"].copy().sort_values("theta_cong")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(d))
    err = np.vstack([d.theta_cong - d.theta_lo, d.theta_hi - d.theta_cong]).clip(min=0)
    sig = d.holm_theta.astype(bool).values
    ax.errorbar(d.theta_cong[~sig], y[~sig], xerr=err[:, ~sig], fmt="o", color=OI["grey"], capsize=3, ms=6, label="n.s. (Holm)")
    if sig.any():
        ax.errorbar(d.theta_cong[sig], y[sig], xerr=err[:, sig], fmt="s", color=OI["purple"], capsize=3, ms=7, label="sig. (Holm)")
    ax.axvline(0, color=OI["black"], lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME.get(m, m) for m in d.model])
    ax.set_xlabel(r"$\theta_{\mathrm{cong}}$  (hypothesis: $<0$; found: $\geq 0$ where significant)")
    ax.set_title("Identity–belief congruence does NOT amplify belief-following")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "F3_theta_cong", "theta_cong per model with CI")


# F4: training-stage ladders — Delta_b grows with preference tuning (OLMo 7B and Tulu 3)
def f4_ladders():
    dm = read("rq1/per_model_main.csv"); de = read("rq1/per_model_exploratory.csv")
    if dm is None or de is None: return print("  skip F4")
    allm = pd.concat([dm, de])
    olmo = ["olmo_base", "olmo_sft", "olmo_dpo", "olmo_instruct"]
    tulu = ["tulu_base", "tulu_sft", "tulu_dpo", "tulu_final"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for ladder, col, lab in [(olmo, OI["blue"], "OLMo 2 7B"), (tulu, OI["orange"], "Tulu 3 (Llama 3.1 8B)")]:
        xs, ys, es = [], [], []
        for i, m in enumerate(ladder):
            r = allm[(allm.model == m) & (allm.fmt == "chat")]
            if len(r): xs.append(i); ys.append(float(r.delta_b.iloc[0])); es.append(float(r.delta_b_se.iloc[0]))
        ax.errorbar(xs, ys, yerr=es, marker="o", color=col, capsize=3, lw=2, label=lab)
    ax.axhline(0, color=OI["black"], lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["base", "SFT", "DPO", "Instr./final"])
    ax.set_ylabel(r"$\Delta_b$ (chat)"); ax.set_xlabel("training stage")
    ax.set_title("Belief-following grows through preference tuning, in two model families")
    ax.legend(loc="lower left", fontsize=10)
    save(fig, "F4_stage_ladders", "Delta_b across OLMo and Tulu training stages")


# F5: heuristic-conflicting vs -consistent evidence slope — why open-model beta is low
def f5_heuristic_conflict():
    try:
        from common import load_dataset
    except Exception:
        return print("  skip F5 (no common)")
    df = load_dataset("main")
    df = df[(df.study == "main") & (df.fmt == "chat") & (~df.excluded) & (df.b == 0) & (df.e != "null")]
    rows = []
    for m in ["gpt", "llama", "olmo_instruct", "olmo_dpo", "olmo_sft"]:
        d = df[df.model == m]
        for conf, lab in [(False, "consistent"), (True, "conflicting")]:
            s = d[d.heuristic_conflict == conf]
            if len(s) > 30:
                b = np.polyfit(s.x, s.y, 1)[0]
                rows.append(dict(model=m, kind=lab, beta=b, n=len(s)))
    if not rows: return print("  skip F5 (no rows)")
    r = pd.DataFrame(rows); models = r.model.unique()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    w = 0.38; x = np.arange(len(models))
    for j, (kind, col) in enumerate([("consistent", OI["green"]), ("conflicting", OI["vermillion"])]):
        vals = [float(r[(r.model == m) & (r.kind == kind)].beta.iloc[0]) if len(r[(r.model == m) & (r.kind == kind)]) else np.nan for m in models]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=f"heuristic-{kind}")
    ax.axhline(0, color=OI["black"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NAME.get(m, m) for m in models], rotation=15, ha="right")
    ax.set_ylabel(r"no-belief evidence slope $\beta$")
    ax.set_title("No-belief evidence slope $\\beta$, split by heuristic conflict")
    ax.legend(fontsize=9)
    save(fig, "F5_heuristic_conflict", "evidence slope split by heuristic conflict")


# F6: corrected RQ3 — share of the contradicted-evidence gap closed, per intervention & model
def f6_gap_closed():
    d = read("rq3_v2/test.csv")
    if d is None: return print("  skip F6")
    fams = {"erase_answer": "erase@answer", "erase_span": "erase@belief-tokens", "cr_single": "steer (1 layer)",
            "cr_all_layers": "steer (all layers)", "erase_span_multi": "erase span×4", "control_random_subspace": "random ctrl"}
    order = ["erase_answer", "erase_span", "cr_single", "cr_all_layers", "erase_span_multi", "control_random_subspace"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    models = ["olmo_instruct", "llama", "tulu_final"]
    cols = {"olmo_instruct": OI["blue"], "llama": OI["orange"], "tulu_final": OI["green"]}
    w = 0.26; x = np.arange(len(order))
    for j, m in enumerate(models):
        dm = d[d.model == m]
        vals = []
        for fam in order:
            r = dm[dm.family == fam]
            vals.append(100 * float(r.gap_closed_frac.iloc[0]) if len(r) else np.nan)
        ax.bar(x + (j - 1) * w, vals, w, color=cols[m], label=MODELNAME[m])
    ax.axhline(100, color=OI["black"], ls="--", lw=0.8); ax.text(len(order)-1, 102, "full closure", fontsize=8)
    ax.axhline(0, color=OI["black"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([fams[f] for f in order], rotation=20, ha="right")
    ax.set_ylabel("% of contradicted-evidence gap closed")
    ax.set_title("Corrected RQ3: answer-position belief erasure removes most belief-following")
    ax.legend(fontsize=9)
    save(fig, "F6_gap_closed", "gap closed per intervention/model")


# F7: sufficiency — where the belief is represented (recovery vs layer)
def f7_sufficiency():
    d = read("rq3_v2/sufficiency.csv")
    if d is None: return print("  skip F7")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    for ax, m in zip(axes, ["olmo_instruct", "llama", "tulu_final"]):
        dm = d[d.model == m].sort_values("layer")
        if dm.empty: continue
        ax.plot(dm.layer, dm.answer_U4, marker="o", color=OI["blue"], lw=2, label="answer pos., 4-D")
        ax.plot(dm.layer, dm.whole_last, marker="s", color=OI["sky"], lw=1.5, ls="--", label="answer pos., whole")
        ax.plot(dm.layer, dm.span_U2, marker="^", color=OI["orange"], lw=1.5, label="belief tokens, 2-D")
        ax.plot(dm.layer, dm.random_U2, marker="x", color=OI["grey"], lw=1, label="random 2-D")
        ax.set_title(MODELNAME[m], fontsize=10); ax.set_xlabel("layer")
    axes[0].set_ylabel("belief effect recovered")
    axes[-1].legend(fontsize=8, loc="center right")
    fig.suptitle("Sufficiency: belief is at the belief-tokens early, at the answer position from mid-layers", fontsize=11)
    save(fig, "F7_sufficiency", "interchange recovery vs layer")


# F8: belief-direction geometry — cosine(believe-H1, believe-H0) vs layer
def f8_geometry():
    d = read("rq3_v2/geometry.csv")
    if d is None: return print("  skip F8")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cols = {"olmo_instruct": OI["blue"], "llama": OI["orange"], "tulu_final": OI["green"]}
    for m in ["olmo_instruct", "llama", "tulu_final"]:
        dm = d[d.model == m].sort_values("layer")
        if dm.empty: continue
        ax.plot(dm.layer, dm.cos_pos_neg, marker="o", color=cols[m], lw=2, label=MODELNAME[m])
    ax.axhline(0, color=OI["black"], lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel(r"cosine(believe-$H_1$, believe-$H_0$) directions")
    ax.set_title("Early: one 'a belief was stated' axis; later: the two beliefs separate")
    ax.legend(fontsize=9)
    save(fig, "F8_geometry", "belief-direction cosine vs layer")


# F9: capability audit — erasure preserves capabilities (per model, per metric)
def f9_audit():
    d = read("rq3_v2/audit.csv")
    if d is None: return print("  skip F9")
    d = d[d.condition.isin(["unsteered", "corrected", "control"]) & (d.chosen != "automatic (success_partial)")]
    metrics = [("mmlu_pro_acc", "MMLU-Pro"), ("bbq_acc", "BBQ"), ("syceval_initial_acc", "SycEval")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    condcol = {"unsteered": OI["grey"], "corrected": OI["blue"], "control": OI["orange"]}
    condlab = {"unsteered": "unsteered", "corrected": "erased", "control": "random ctrl"}
    models = ["olmo_instruct", "llama", "tulu_final"]
    for ax, (mc, mlab) in zip(axes, metrics):
        x = np.arange(len(models)); w = 0.26
        for j, cond in enumerate(["unsteered", "corrected", "control"]):
            vals = []
            for m in models:
                r = d[(d.model == m) & (d.condition == cond)]
                vals.append(float(r[mc].iloc[0]) if len(r) and pd.notna(r[mc].iloc[0]) else np.nan)
            ax.bar(x + (j - 1) * w, vals, w, color=condcol[cond], label=condlab[cond])
        ax.set_xticks(x); ax.set_xticklabels(["OLMo", "Llama", "Tulu"], fontsize=9)
        ax.set_title(mlab, fontsize=10); ax.set_ylim(0, 0.75)
    axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8)
    fig.suptitle("Belief erasure preserves capabilities (erased ≈ unsteered ≈ control)", fontsize=11)
    save(fig, "F9_audit", "capability audit per model/metric")


# F10: tier contrast — belief-following is as strong on non-social (Tier 1) topics
def f10_tier():
    d = read("rq1_secondary/tier_main.csv")
    if d is None: return print("  skip F10")
    t1 = d[d.term == "delta_b_tier1"].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    t1 = t1[t1.fmt == "chat"].sort_values("est")
    y = np.arange(len(t1))
    err = np.vstack([t1.est - t1.lo, t1.hi - t1.est]).clip(min=0)
    ax.errorbar(t1.est, y, xerr=err, fmt="o", color=OI["vermillion"], capsize=3, ms=7)
    ax.axvline(0, color=OI["black"], lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([NAME.get(m, m) for m in t1.model])
    ax.set_xlabel(r"$\Delta_b$ on Tier 1 (technical, non-social) topics")
    ax.set_title("Belief-following is at least as strong on topics with no social meaning")
    save(fig, "F10_tier1", "Tier 1 Delta_b per model")


def main():
    print(f"writing candidate figures to {OUT.relative_to(ROOT.parent)}")
    for fn in [f1_delta_b_main, f2_beta, f3_theta, f4_ladders, f5_heuristic_conflict,
               f6_gap_closed, f7_sufficiency, f8_geometry, f9_audit, f10_tier]:
        try: fn()
        except Exception as e: print(f"  ERROR in {fn.__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
