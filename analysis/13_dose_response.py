"""Step 13: dose-response for the belief subspace (RQ3 add-on).

    python analysis/13_dose_response.py

Does the *magnitude* of the belief representation on a trial predict *how much* that trial discounts
contradicting evidence? For each model we have:
  DOSE      = per-trial projection of the answer-position hidden state onto the belief axis, signed toward the
              stated belief (scripts/capture_doses.py -> results/rq3_v2/<model>/doses.parquet).
  RESPONSE  = per-trial behavioral belief-congruent deviation from the Bayesian benchmark, signed toward the
              stated belief: belief_dir * (logit P_post - bayes_post), where bayes_post = logit P_prior + x is the
              ideal (Bayes) posterior logit. Computed self-contained from cf_trials.parquet (CPU, no GPU); the
              counterfactual trials are all belief-stated (no neutral b=0 rows), so the deviation is measured
              against each trial's own Bayesian benchmark rather than a matched no-belief counterpart.
On belief-CONTRADICTING trials (kappa==1) the evidence points away from the stated belief, so a positive response
means the answer stayed toward the belief and under-updated on the contradicting evidence (discounting). We
correlate dose vs response, partial out evidence strength |x|, per model and pooled, and write a figure (scatter +
binned dose-response curve) to paper/secondary/figures/candidates/F11_dose_response.{pdf,png}.

Writes analysis/output/rq3_v2/dose_response.csv and the figure. Skips a model whose doses.parquet is absent.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "rq3_v2"
OUTD = ROOT / "analysis" / "output" / "rq3_v2"; OUTD.mkdir(parents=True, exist_ok=True)
FIG = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"; FIG.mkdir(parents=True, exist_ok=True)
OI = dict(olmo_instruct="#0072B2", llama="#E69F00", tulu_final="#009E73", grey="#999999")
NAME = {"olmo_instruct": "OLMo 2 7B Instruct", "llama": "Llama 3.1 8B Instruct", "tulu_final": "Tulu 3 final"}
def logit(p): return np.log(np.clip(p, 0.005, 0.995) / (1 - np.clip(p, 0.005, 0.995)))


def responses(cf: pd.DataFrame) -> pd.DataFrame:
    """Per belief-stated trial: belief-congruent deviation from the Bayesian benchmark, signed toward belief.

    All counterfactual trials carry a stated belief (no neutral b=0 rows), so we measure each trial's final answer
    against its own Bayesian ideal rather than a matched no-belief counterpart. lp_after = logit P_post and
    bayes_post = logit P_prior + x (the exact Bayesian posterior logit) are precomputed in cf_trials; the residual
    lp_after - bayes_post, signed by belief_dir, is positive when the answer sits toward the stated belief
    relative to what the evidence alone warrants.
    """
    cf = cf.copy()
    cf["response"] = np.sign(cf.belief_dir) * (cf.lp_after - cf.bayes_post)
    return cf[["trial_id", "kappa", "x", "belief_dir", "response"]]


def analyze(model: str):
    dp = RES / f"{model}_chat" / "doses.parquet"
    cfp = RES / f"{model}_chat" / "cf_trials.parquet"
    if not dp.exists() or not cfp.exists():
        print(f"  skip {model} (missing doses/cf)"); return None, None
    dose = pd.read_parquet(dp)[["trial_id", "dose", "belief_dir"]]
    resp = responses(pd.read_parquet(cfp))
    d = resp.merge(dose, on="trial_id", how="inner", suffixes=("", "_d"))
    contra = d[d.kappa == 1].dropna(subset=["dose", "response", "x"])
    if len(contra) < 30:
        print(f"  skip {model} (only {len(contra)} contra trials)"); return None, None
    r, p = stats.pearsonr(contra.dose, contra.response)
    rho, prho = stats.spearmanr(contra.dose, contra.response)
    # partial correlation of dose vs response controlling for |x|
    def resid(y, xctl):
        b = np.polyfit(xctl, y, 1); return y - np.polyval(b, xctl)
    pr = stats.pearsonr(resid(contra.dose.values, np.abs(contra.x.values)),
                        resid(contra.response.values, np.abs(contra.x.values)))[0]
    row = dict(model=model, n_contra=len(contra), pearson_r=round(r, 3), pearson_p=p,
               spearman=round(rho, 3), partial_r_given_absx=round(pr, 3))
    print(f"  {model}: n={len(contra)} r={r:.3f} (p={p:.1e}) rho={rho:.3f} partial(|x|)={pr:.3f}")
    return row, contra


def main():
    rows, panels = [], []
    for m in ["olmo_instruct", "llama", "tulu_final"]:
        row, contra = analyze(m)
        if row: rows.append(row); panels.append((m, contra))
    if rows:
        pd.DataFrame(rows).to_csv(OUTD / "dose_response.csv", index=False)
        print(f"wrote {OUTD/'dose_response.csv'}")
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.3), sharey=True, squeeze=False)
        for ax, (m, c) in zip(axes[0], panels):
            ax.scatter(c.dose, c.response, s=8, alpha=0.25, color=OI.get(m, "#0072B2"))
            qs = pd.qcut(c.dose, min(8, max(2, len(c) // 40)), duplicates="drop")
            g = c.groupby(qs); mx = g.dose.mean(); my = g.response.mean()
            ax.plot(mx, my, "-o", color="black", lw=2, ms=5, label="binned mean")
            ax.axhline(0, color=OI["grey"], lw=0.8); ax.axvline(0, color=OI["grey"], lw=0.8)
            r = next(x for x in rows if x["model"] == m)
            ax.set_title(f"{NAME[m]}\n$r$={r['pearson_r']}, partial={r['partial_r_given_absx']}", fontsize=9)
            ax.set_xlabel("belief dose (projection)")
        axes[0][0].set_ylabel("discounting response (log-odds toward belief)")
        axes[0][-1].legend(fontsize=8)
        fig.suptitle("Dose-response: per-trial belief-representation magnitude vs belief-congruent evidence "
                     "discounting (contradicting trials)", fontsize=10.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"F11_dose_response.{ext}", bbox_inches="tight", dpi=200)
        print(f"wrote {FIG/'F11_dose_response.pdf'}")


if __name__ == "__main__":
    main()
