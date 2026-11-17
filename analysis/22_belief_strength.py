"""Step 22: belief-strength and third-party experiments (exploratory, 2026-11-16).

    python analysis/22_belief_strength.py

Levels, all run fresh on the same trials (configs/bs.yaml, configs/tp.yaml):
  weak      *_bs3  "I'm not sure, but I think <clause>."
  user      *_bs0  "I think <clause>."               (the original sentence, re-run)
  strong    *_bs2  "I'm certain that <clause>."
  tp        *_bs6  "<Name> thinks <clause>."         (third party; Alex/Jordan/Taylor/Casey by trial)
  tp_strong *_bs7  "<Name> is certain that <clause>." (open models only)
  nb        *_nb   no belief (b = 0 trials; open models only)
Open models: the 0/3 trial-id third; GPT-4o: the 0/12 twelfth (its answers drifted since the main run, so every GPT-4o
comparison is within this fresh run). Exclusions and log-odds by idea.runner.derive (as in the main run).

Outputs (analysis/output/belief_strength/):
  reproducibility.csv  fresh original level vs the released main-run answers (same trials)
  paired_shifts.csv    b*(logit P_A - logit P_B) on the same trials (posterior and prior), topic-clustered SE
  total_shift.csv      b*(logit P_post - mean no-belief logit P_post on the same item), by level (open models; fresh
                       no-belief baseline where the item has one, else the released no-belief answers)
  slopes_by_level.csv  primary model per level (fresh no-belief + that level's belief trials): Delta_b, theta_cong,
                       support slope, implied shift (open models)
Figure F24 (paper/secondary/figures/candidates).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from idea.runner import derive  # noqa: E402
from common import BASE_TERMS, analysis_frame, estimate, fit_ols, load_dataset, out_dir  # noqa: E402

RUNS = ROOT / "runs"
FIG = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"
MODELS = ["gpt", "phi4", "llama", "olmo_instruct", "tulu_final"]
NAME = {"gpt": "GPT-4o", "phi4": "Phi-4", "llama": "Llama 3.1 8B", "olmo_instruct": "OLMo 2 7B Instr.",
        "tulu_final": "Tulu 3 8B"}
LEVELS = {"weak": ("bs", 3), "user": ("bs", 0), "strong": ("bs", 2), "tp": ("tp", 6), "tp_strong": ("tp", 7)}
PAIRS = [("strong", "user"), ("user", "weak"), ("strong", "weak"), ("user", "tp"), ("strong", "tp_strong"),
         ("tp_strong", "tp")]


def load(run: str, alias: str) -> pd.DataFrame | None:
    fs = sorted((RUNS / f"{run}-8b15a8c4" / "responses").glob(f"{alias}_chat*.parquet"))
    if not fs:
        return None
    return derive(pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True))


def clustered_mean(v: pd.Series, groups: pd.Series) -> tuple[float, float, int]:
    d = pd.DataFrame({"v": v.values, "g": groups.values}).dropna()
    if len(d) < 10 or d.g.nunique() < 3:
        return np.nan, np.nan, len(d)
    f = smf.ols("v ~ 1", d).fit(cov_type="cluster", cov_kwds={"groups": d.g.astype("category").cat.codes})
    return float(f.params.iloc[0]), float(f.bse.iloc[0]), len(d)


def main() -> None:
    od = out_dir("belief_strength")
    rel = load_dataset("main")
    rep_rows, pair_rows, tot_rows, slope_rows = [], [], [], []
    for m in MODELS:
        L = {k: load(run, f"{m}_bs{v}") for k, (run, v) in LEVELS.items()}
        L = {k: v for k, v in L.items() if v is not None}
        nb = load("tp", f"{m}_nb")
        base = rel[(rel.model == m) & (rel.fmt == "chat")]
        # ---- reproducibility of the fresh original level
        u = L["user"][~L["user"].excluded].merge(base[["trial_id", "p_post", "p_prior"]], on="trial_id", suffixes=("", "_rel"))
        rep_rows.append(dict(model=m, n=len(u), r_post=np.corrcoef(u.p_post, u.p_post_rel)[0, 1],
                             mad_post=float((u.p_post - u.p_post_rel).abs().mean()),
                             r_prior=np.corrcoef(u.p_prior, u.p_prior_rel)[0, 1],
                             mad_prior=float((u.p_prior - u.p_prior_rel).abs().mean())))
        # ---- paired shifts (same trials, usable at both levels)
        for a, b in PAIRS:
            if a not in L or b not in L:
                continue
            A = L[a][~L[a].excluded]; B = L[b][~L[b].excluded]
            j = A[["trial_id", "b", "tier", "topic", "kappa", "lp_after", "lp_before"]].merge(
                B[["trial_id", "lp_after", "lp_before"]], on="trial_id", suffixes=("_a", "_b"))
            for tier in (2, 1):
                for side, q in (("all", j.tier == tier), ("contra", (j.tier == tier) & (j.kappa == 1))):
                    s = j[q]
                    for which in ("post", "prior"):
                        col = "lp_after" if which == "post" else "lp_before"
                        est, se, n = clustered_mean(s.b * (s[f"{col}_a"] - s[f"{col}_b"]), s.topic)
                        pair_rows.append(dict(model=m, contrast=f"{a} - {b}", tier=tier, side=side, answer=which,
                                              est=est, se=se, n=n, n_topics=s.topic.nunique()))
        # ---- total shift vs no belief (open models: fresh baseline; also released no-belief answers)
        if nb is not None:
            nbf = nb[~nb.excluded].groupby("item_id").lp_after.mean().rename("N_fresh")
            nbr = base[(base.b == 0) & ~base.excluded.astype(bool)].groupby("item_id").lp_after.mean().rename("N_rel")
            for k, D in L.items():
                D = D[~D.excluded].join(nbf, on="item_id").join(nbr, on="item_id")
                for tier in (2, 1):
                    for side, q in (("contra", (D.tier == tier) & (D.kappa == 1)), ("all", D.tier == tier)):
                        s = D[q]
                        for src in ("N_fresh", "N_rel"):
                            est, se, n = clustered_mean(s.b * (s.lp_after - s[src]), s.topic)
                            tot_rows.append(dict(model=m, level=k, tier=tier, side=side, baseline=src.split("_")[1],
                                                 est=est, se=se, n=n, matched=float(s[src].notna().mean())))
        # ---- primary model per level (open models, fresh no-belief baseline)
        if nb is not None:
            mm = base[["trial_id", "m"]]
            for k, D in L.items():
                frame = pd.concat([nb, D], ignore_index=True).drop(columns=["m"], errors="ignore").merge(mm, on="trial_id", how="left")
                frame["model"] = m
                for tier in (2, 1):
                    d = analysis_frame(frame.assign(excluded=frame.excluded.astype(bool)), tier=tier, model=m, fmt="chat")
                    f = fit_ols(d, BASE_TERMS)
                    db, th, sp, be = (estimate(f, t) for t in ("x:contra", "x:contra:m", "x:support", "x"))
                    mx = float(d.loc[d.contra == 1, "x"].abs().mean())
                    slope_rows.append(dict(model=m, level=k, tier=tier, n=len(d), beta=be["est"], beta_se=be["se"],
                                           delta_b=db["est"], delta_b_se=db["se"], theta=th["est"], theta_se=th["se"],
                                           theta_p=th["p"], support=sp["est"], implied_shift=-db["est"] * mx))
        print(f"  {m}: levels {sorted(L)}{' + no-belief' if nb is not None else ''}")
    R, P, T, S = (pd.DataFrame(x) for x in (rep_rows, pair_rows, tot_rows, slope_rows))
    for name, x in (("reproducibility", R), ("paired_shifts", P), ("total_shift", T), ("slopes_by_level", S)):
        x.to_csv(od / f"{name}.csv", index=False)
    print(f"wrote {od}")
    figure(P, T)


def figure(P: pd.DataFrame, T: pd.DataFrame) -> None:
    """Shift of the answer toward the stated belief at each level, relative to the user's original 'I think' (paired)."""
    order = [("weak", "user - weak", -1), ("strong", "strong - user", 1), ("third party", "user - tp", -1),
             ("third party, certain", None, None)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    col = {"gpt": "#000000", "phi4": "#E69F00", "llama": "#56B4E9", "olmo_instruct": "#009E73", "tulu_final": "#D55E00"}
    mk = {"gpt": "o", "phi4": "s", "llama": "^", "olmo_instruct": "D", "tulu_final": "v"}
    for ax, tier in zip(axes, (2, 1)):
        for i, m in enumerate(MODELS):
            q = P[(P.model == m) & (P.tier == tier) & (P.side == "all") & (P.answer == "post")].set_index("contrast")
            xs, ys, es = [], [], []
            for j, (lab, c, sgn) in enumerate(order):
                if lab == "third party, certain":
                    if "strong - tp_strong" in q.index and "strong - user" in q.index:
                        # tp_strong relative to user = (strong - user) - (strong - tp_strong)
                        v = q.loc["strong - user", "est"] - q.loc["strong - tp_strong", "est"]
                        e = np.sqrt(q.loc["strong - user", "se"] ** 2 + q.loc["strong - tp_strong", "se"] ** 2)
                    else:
                        continue
                elif c in q.index:
                    v, e = sgn * q.loc[c, "est"], q.loc[c, "se"]
                else:
                    continue
                xs.append(j + (i - 2) * 0.1); ys.append(v); es.append(1.96 * e)
            ax.errorbar(xs, ys, yerr=es, fmt=mk[m], color=col[m], ms=6, capsize=2, label=NAME[m], lw=0, elinewidth=1.2)
        ax.axhline(0, color="#999999", lw=0.9)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([o[0] for o in order], fontsize=8.5)
        ax.set_title(f"Tier {tier}", fontsize=10)
    axes[0].set_ylabel("change in shift toward the belief vs \"I think\"\n(log-odds, posterior; 95% CI)", fontsize=8.5)
    axes[1].legend(fontsize=7.5, loc="best")
    fig.suptitle("How the belief's wording and speaker change belief-following (same trials, paired)", fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"F24_belief_strength_third_party.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {FIG/'F24_belief_strength_third_party.pdf'}")


if __name__ == "__main__":
    main()
