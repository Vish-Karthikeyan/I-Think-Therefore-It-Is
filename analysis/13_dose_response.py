"""Step 13: dose-response for the belief subspace (RQ3 add-on).

    python analysis/13_dose_response.py

Does the *magnitude* of the belief representation on a trial predict *how much* that trial discounts
contradicting evidence?

Sign convention (important). The model reports P(H1); x is the Bayesian log-odds update for H1; the user's
belief relative to H1 is b (+1 believes H1, -1 believes H0). "Toward the belief" in this space is therefore
signed by b -- NOT by belief_dir = b*h_dir, which is the concrete claim (policy helps/hurts) and flips sign
with H1's wording. (An earlier version of this script signed by belief_dir; that manufactured a spurious
correlation, see below.)

  RESPONSE = b * (logit P_post - bayes_post), bayes_post = logit P_prior + x (each trial's own Bayesian
             benchmark; cf trials are all belief-stated, so there is no neutral counterpart). On contradicting
             trials (kappa==1) a positive response = the answer stayed toward the belief (discounting).
  DOSE     = position on the belief axis of the 2-D answer-position subspace ans_U2 (the subspace whose
             interchange recovers 74-90% of the belief effect), signed by b, with the axis cross-fitted
             (fit leaving out the trial's topic). From results/rq3_v2/<model>_chat/subspace_proj.parquet.

Reported per model: pooled r and partial r given |x| (contradicting trials), and r within each stated belief
(b=+1, b=-1), which removes the between-belief separation and asks whether *graded* magnitude matters.
For transparency the old content-axis dose (doses.parquet, from scripts/capture_doses.py) is also reported
with the corrected response, plus its correlation with the answer itself: that axis largely reads out the
answer the model is about to give (answer-position state), which is why signing dose and response by the
same variable produced a large spurious r.

Writes analysis/output/rq3_v2/dose_response.csv and F11_dose_response.{pdf,png} (candidates dir).
"""
from __future__ import annotations
import warnings
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


def crossfit_dose(df: pd.DataFrame) -> np.ndarray:
    """b-signed position on the ans_U2 belief axis, axis fit leaving out the trial's topic."""
    X = df[["u2_0", "u2_1"]].to_numpy(np.float64); b = df.b.to_numpy(); out = np.full(len(df), np.nan)
    for t in df.topic.unique():
        te = (df.topic == t).to_numpy(); tr = ~te
        w = X[tr & (b > 0)].mean(0) - X[tr & (b < 0)].mean(0); w /= (np.linalg.norm(w) or 1.0)
        out[te] = np.sign(b[te]) * ((X[te] - X[tr].mean(0)) @ w)
    return out


def partial_r(a, c, z):
    ra = a - np.polyval(np.polyfit(z, a, 1), z); rc = c - np.polyval(np.polyfit(z, c, 1), z)
    return stats.pearsonr(ra, rc)[0]


def analyze(model: str):
    cfp = RES / f"{model}_chat" / "cf_trials.parquet"; spp = RES / f"{model}_chat" / "subspace_proj.parquet"
    if not cfp.exists() or not spp.exists():
        print(f"  skip {model} (missing cf_trials/subspace_proj)"); return None, None
    cf = pd.read_parquet(cfp)
    df = cf.merge(pd.read_parquet(spp)[["trial_id", "u2_0", "u2_1"]], on="trial_id", how="inner")
    df["response"] = np.sign(df.b) * (df.lp_after - df.bayes_post)
    df["dose"] = crossfit_dose(df)
    c = df[df.kappa == 1].dropna(subset=["dose", "response", "x"])
    ax_ = np.abs(c.x.to_numpy())
    r, p = stats.pearsonr(c.dose, c.response)
    row = dict(model=model, n_contra=len(c), mean_response=round(c.response.mean(), 3),
               pearson_r=round(r, 3), pearson_p=p, partial_r_given_absx=round(partial_r(c.dose.values, c.response.values, ax_), 3))
    for v in (1, -1):
        cc = c[c.b == v]; row[f"r_within_b{v:+d}"] = round(stats.pearsonr(cc.dose, cc.response)[0], 3)
    dp = RES / f"{model}_chat" / "doses.parquet"            # old content-axis dose, for transparency
    if dp.exists():
        old = c.merge(pd.read_parquet(dp)[["trial_id", "dose"]].rename(columns={"dose": "dose_content"}), on="trial_id")
        row["r_content_axis_dose_corrected"] = round(stats.pearsonr(old.dose_content, old.response)[0], 3)
        full = df.merge(pd.read_parquet(dp)[["trial_id", "dose_raw"]], on="trial_id")
        row["r_content_axis_vs_answer"] = round(stats.pearsonr(full.dose_raw, full.h_dir * full.lp_after)[0], 3)
    print(f"  {row}")
    return row, c


def main():
    rows, panels = [], []
    for m in ["olmo_instruct", "llama", "tulu_final"]:
        row, c = analyze(m)
        if row: rows.append(row); panels.append((m, c))
    if rows:
        pd.DataFrame(rows).to_csv(OUTD / "dose_response.csv", index=False)
        print(f"wrote {OUTD/'dose_response.csv'}")
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=True, squeeze=False)
        for ax, (m, c) in zip(axes[0], panels):
            r = next(x for x in rows if x["model"] == m)
            for v, mk in ((1, "o"), (-1, "^")):
                cc = c[c.b == v]
                ax.scatter(cc.dose, cc.response, s=8, alpha=0.25, marker=mk, color=OI[m], linewidths=0)
                qs = pd.qcut(cc.dose, 6, duplicates="drop"); g = cc.groupby(qs)
                ax.plot(g.dose.mean(), g.response.mean(), "-" + mk, color="black", lw=1.6, ms=4,
                        label=f"binned mean, believes {'H1' if v > 0 else 'H0'}")
            ax.axhline(0, color=OI["grey"], lw=0.8)
            ax.set_title(f"{NAME[m]}\nwithin-belief $r$ = {r['r_within_b+1']:+.2f} / {r['r_within_b-1']:+.2f}", fontsize=9)
            ax.set_xlabel("belief dose (ans_U2 axis, cross-fit)")
        axes[0][0].set_ylabel("discounting (log-odds toward belief\nbeyond the Bayesian posterior)")
        axes[0][-1].legend(fontsize=7)
        fig.suptitle("Dose-response on contradicting trials: belief magnitude within a stated belief does not "
                     "robustly predict discounting", fontsize=10)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"F11_dose_response.{ext}", bbox_inches="tight", dpi=200)
        print(f"wrote {FIG/'F11_dose_response.pdf'}")


if __name__ == "__main__":
    main()
