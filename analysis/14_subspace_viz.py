"""Step 14: visualize the belief subspace (RQ3 add-on).

    python analysis/14_subspace_viz.py

The corrected RQ3 finds a low-dimensional subspace at the answer position that encodes the user's stated
belief. scripts/capture_subspace_viz.py projects every counterfactual trial's answer-position hidden state
(at each model's erase layer) onto that subspace and writes results/rq3_v2/<model>_chat/subspace_proj.parquet
with coordinates u2_0,u2_1 (the 2-D belief subspace ans_U2) and u4_0..u4_2 (first 3 dims of ans_U4).

This script makes the figures (candidates dir), mirroring how the linear-representation / interpretability
literature visualizes a concept subspace (2-D projection onto the concept directions, class-colored scatter,
and a layerwise trajectory). Two belief labels matter and must not be confused:
  b          = +1 if the user believes H1, -1 if H0 -- the belief *relative to the hypothesis the model scores*
               (the answer is P(H1)); this is what the counterfactual interchange swaps.
  belief_dir = b * h_dir -- the *concrete claim* asserted (policy helps / hurts), independent of H1's wording.
The 2-D answer-position subspace ans_U2 separates b very strongly and belief_dir hardly at all.
  F12  ans_U2 plane, one column per model. Top row colored by b (believe H1 vs H0): two separated clouds
       [KEY FIGURE]. Bottom row, same points colored by the concrete claim (belief_dir): mixed.
  F13  ans_U2 plane colored by IDENTITY DIMENSION, marker = b: the b split holds in every dimension.
  F14  layerwise geometry trajectory (geometry.csv): cosine between the believe-H1 and believe-H0 mean
       directions vs layer: ~ +1 early (a belief is present) -> negative late (which belief).
  F15  1-D density of each trial's score on the b-discriminant axis within ans_U2, per model.
Cohen's d is reported in-sample and cross-fitted (axis fit leaving out the trial's topic).

All cf trials here are belief-stated (b in {-1,+1}); there are no no-belief rows.
CPU only; reads the parquet produced on GPU. Saves PDF+PNG. Skips a model whose parquet is absent.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "rq3_v2"
FIG = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["olmo_instruct", "llama", "tulu_final"]
NAME = {"olmo_instruct": "OLMo 2 7B Instruct", "llama": "Llama 3.1 8B Instruct", "tulu_final": "Tulu 3 final"}
ERASE_LAYER = {"olmo_instruct": 24, "llama": 16, "tulu_final": 24}
# Okabe-Ito, colorblind-safe
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermillion="#D55E00",
          skyblue="#56B4E9", yellow="#F0E442", purple="#CC79A7", black="#000000", grey="#999999")
BELIEF_COLOR = {1: OI["blue"], -1: OI["vermillion"]}        # believe-H1 vs believe-H0
BELIEF_MARK = {1: "o", -1: "^"}                              # distinct markers -> legible in B&W
BELIEF_LABEL = {1: "believe H1", -1: "believe H0"}
# 8 identity dimensions -> Okabe-Ito palette
DIM_ORDER = ["political", "religion", "gender", "sexuality", "race", "age", "geography", "none"]
DIM_COLOR = dict(zip(DIM_ORDER, [OI["blue"], OI["orange"], OI["green"], OI["vermillion"],
                                 OI["skyblue"], OI["purple"], OI["yellow"], OI["grey"]]))


def load(model: str):
    p = RES / f"{model}_chat" / "subspace_proj.parquet"
    cfp = RES / f"{model}_chat" / "cf_trials.parquet"
    if not p.exists() or not cfp.exists():
        print(f"  skip {model} (missing {p.name} or cf_trials)"); return None
    df = pd.read_parquet(p)
    return df.merge(pd.read_parquet(cfp)[["trial_id", "b", "h_dir"]], on="trial_id", how="left")


SUB_COLS = ("u2_0", "u2_1")   # the 2-D answer-position belief subspace ans_U2 (the one the interchange patches)


def cohen_d(pos, neg):
    sp = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2) or 1.0
    return (pos.mean() - neg.mean()) / sp


def belief_axis(df, label="b", cols=SUB_COLS):
    """Within the subspace, the unit difference-of-class-means direction separating label=+1 from -1, each trial's
    (centered) score on it, the leading orthogonal direction's score (for the 2-D plot), and Cohen's d."""
    X = df[list(cols)].to_numpy().astype(np.float64)
    lab = df[label].to_numpy()
    w = X[lab > 0].mean(0) - X[lab < 0].mean(0)
    w = w / (np.linalg.norm(w) or 1.0)
    Xc = X - X.mean(0)
    s = Xc @ w
    d = cohen_d(s[lab > 0], s[lab < 0])
    R = Xc - np.outer(s, w)
    if R.shape[1] > 1 and np.linalg.norm(R) > 0:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        y = R @ Vt[0]
    else:
        y = np.zeros(len(df))
    return w, s, y, d


def crossfit_d(df, label="b", cols=SUB_COLS):
    """Cohen's d with the axis fit leaving out each trial's topic (guards against in-sample axis fitting)."""
    X = df[list(cols)].to_numpy().astype(np.float64); lab = df[label].to_numpy(); s = np.full(len(df), np.nan)
    for t in df.topic.unique():
        te = (df.topic == t).to_numpy(); tr = ~te
        w = X[tr & (lab > 0)].mean(0) - X[tr & (lab < 0)].mean(0); w /= (np.linalg.norm(w) or 1.0)
        s[te] = (X[te] - X[tr].mean(0)) @ w
    return cohen_d(s[lab > 0], s[lab < 0])


# --------------------------------------------------------------------------------------
CLAIM_COLOR = {1: OI["green"], -1: OI["purple"]}
CLAIM_MARK = {1: "s", -1: "D"}
CLAIM_LABEL = {1: "claims policy helps", -1: "claims policy hurts"}


def fig_scatter_belief(data):
    """F12: the ans_U2 plane (x = b-discriminant axis, y = the orthogonal U2 direction), one column per model.
    Top row colored by b (believe H1 vs H0, the belief relative to the scored hypothesis); bottom row, the same
    points colored by the concrete claim (belief_dir). The axis and plane are identical in both rows."""
    fig, axes = plt.subplots(2, len(data), figsize=(3.5 * len(data), 6.6), squeeze=False)
    for j, (m, df) in enumerate(data):
        _, s, y, d = belief_axis(df, "b"); dcf = crossfit_d(df, "b")
        b = df.b.to_numpy(); bd = df.belief_dir.to_numpy()
        _, s_c, _, _ = belief_axis(df, "belief_dir"); d_claim = cohen_d(s_c[bd > 0], s_c[bd < 0])
        ax = axes[0][j]
        for v in (1, -1):
            ax.scatter(s[b == v], y[b == v], s=10, alpha=0.35, color=BELIEF_COLOR[v],
                       marker=BELIEF_MARK[v], linewidths=0, label=BELIEF_LABEL[v])
        for v in (1, -1):
            ax.scatter([s[b == v].mean()], [y[b == v].mean()], s=150, color=BELIEF_COLOR[v],
                       marker=BELIEF_MARK[v], edgecolor="black", linewidths=1.4, zorder=5)
        ax.axvline(0, color="black", lw=1.1, ls="--", zorder=4)
        ax.set_title(f"{NAME[m]}, L{ERASE_LAYER[m]}\nbelief vs scored hypothesis: $d$={d:.1f} "
                     f"(cross-fit {dcf:.1f})", fontsize=9)
        ax = axes[1][j]
        for v in (1, -1):
            ax.scatter(s[bd == v], y[bd == v], s=10, alpha=0.35, color=CLAIM_COLOR[v],
                       marker=CLAIM_MARK[v], linewidths=0, label=CLAIM_LABEL[v])
        ax.axvline(0, color="black", lw=1.1, ls="--", zorder=4)
        ax.set_title(f"same points, by concrete claim: best-axis $d$={d_claim:.1f}", fontsize=9)
        ax.set_xlabel("belief axis within ans_U2")
        for r in (0, 1):
            axes[r][j].axhline(0, color=OI["grey"], lw=0.6)
    for r in (0, 1):
        axes[r][0].set_ylabel("orthogonal direction within ans_U2")
        axes[r][0].legend(fontsize=7.5, loc="best", framealpha=0.9)
    fig.suptitle("The 2-D answer-position belief subspace encodes agreement with the scored hypothesis (top),\n"
                 "not the concrete claim asserted (bottom)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "F12_belief_subspace_scatter")


def fig_scatter_dimension(data):
    """F13: same ans_U2 plane colored by identity dimension, marker = b."""
    fig, axes = plt.subplots(1, len(data), figsize=(3.5 * len(data), 3.6), squeeze=False)
    for ax, (m, df) in zip(axes[0], data):
        _, s, y, d = belief_axis(df, "b")
        bd = df.b.to_numpy(); dims = df.dimension.to_numpy()
        for dim in [x for x in DIM_ORDER if x in set(df.dimension)]:
            for b in (1, -1):
                sel = (dims == dim) & (bd == b)
                if sel.any():
                    ax.scatter(s[sel], y[sel], s=10, alpha=0.5, color=DIM_COLOR[dim],
                               marker=BELIEF_MARK[b], linewidths=0)
        ax.axvline(0, color="black", lw=1.0, ls="--", zorder=4)
        ax.set_title(f"{NAME[m]}  (belief $d$={d:.1f})", fontsize=9)
        ax.set_xlabel("belief axis within ans_U2")
        ax.axhline(0, color=OI["grey"], lw=0.6)
    axes[0][0].set_ylabel("orthogonal direction within ans_U2")
    dims_present = [x for x in DIM_ORDER if any(x in set(df.dimension) for _, df in data)]
    handles = [Line2D([], [], marker="s", ls="", color=DIM_COLOR[x], label=x) for x in dims_present]
    handles += [Line2D([], [], marker=BELIEF_MARK[b], ls="", color="black", label=BELIEF_LABEL[b]) for b in (1, -1)]
    axes[0][-1].legend(handles=handles, fontsize=7, loc="best", framealpha=0.9, ncol=1)
    fig.suptitle("Stated belief (vs the scored hypothesis) by identity dimension in the ans_U2 plane",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "F13_belief_subspace_by_dimension")


def fig_layer_trajectory():
    """F14: cos between believe-H1 and believe-H0 directions vs layer (presence -> polarity)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    color = {"olmo_instruct": OI["blue"], "llama": OI["orange"], "tulu_final": OI["green"]}
    mark = {"olmo_instruct": "o", "llama": "s", "tulu_final": "^"}
    any_data = False
    for m in MODELS:
        gp = RES / f"{m}_chat" / "geometry.csv"
        if not gp.exists():
            continue
        g = pd.read_csv(gp).sort_values("layer")
        any_data = True
        ax.plot(g.layer, g.cos_pos_neg, "-", marker=mark[m], color=color[m], lw=1.8, ms=5, label=NAME[m])
        el = ERASE_LAYER[m]
        row = g[g.layer == el]
        if len(row):
            ax.scatter([el], [row.cos_pos_neg.iloc[0]], s=150, facecolors="none",
                       edgecolors=color[m], linewidths=1.8, zorder=5)
    ax.axhline(0, color=OI["grey"], lw=0.8, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel(r"cos(believe-H1 dir, believe-H0 dir)")
    ax.set_title("Presence -> polarity: belief directions align early, point apart late\n"
                 "(open circle = erase layer)", fontsize=10)
    ax.annotate("shared 'belief present' axis", xy=(8, 0.9), fontsize=8, color=OI["grey"])
    ax.annotate("beliefs point apart\n(polarity)", xy=(20, -0.25), fontsize=8, color=OI["grey"])
    if any_data:
        ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, "F14_belief_geometry_trajectory")


def fig_discriminant_density(data):
    """F15: 1-D density of the within-subspace belief-discriminant projection, per model."""
    fig, axes = plt.subplots(1, len(data), figsize=(3.5 * len(data), 3.0), sharey=True, squeeze=False)
    for ax, (m, df) in zip(axes[0], data):
        _, s, _, d = belief_axis(df, "b"); dcf = crossfit_d(df, "b")
        bd = df.b.to_numpy()
        lo, hi = np.percentile(s, [1, 99]); bins = np.linspace(lo, hi, 36)
        for b in (1, -1):
            ax.hist(s[bd == b], bins=bins, density=True, alpha=0.55, color=BELIEF_COLOR[b],
                    label=BELIEF_LABEL[b])
        ax.axvline(0, color=OI["grey"], lw=0.8)
        ax.set_title(f"{NAME[m]}  ($d$={d:.1f}, cross-fit {dcf:.1f})", fontsize=9)
        ax.set_xlabel("score on the belief axis within ans_U2")
    axes[0][0].set_ylabel("density"); axes[0][0].legend(fontsize=8)
    fig.suptitle("Believe-H1 vs believe-H0 along the belief axis of the 2-D answer-position subspace", fontsize=11)
    fig.tight_layout()
    save(fig, "F15_belief_discriminant_density")


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG/stem}.pdf/.png")


def main():
    data = [(m, df) for m in MODELS if (df := load(m)) is not None]
    if data:
        rows = []
        for m, df in data:
            _, _, _, d_b = belief_axis(df, "b"); _, _, _, d_c = belief_axis(df, "belief_dir")
            d_b4 = belief_axis(df, "b", ("u4_0", "u4_1", "u4_2"))[3]
            d_c4 = belief_axis(df, "belief_dir", ("u4_0", "u4_1", "u4_2"))[3]
            rows.append(dict(model=m, n=len(df), d_b_U2=round(d_b, 2), d_b_U2_crossfit=round(crossfit_d(df, "b"), 2),
                             d_claim_U2=round(d_c, 2), d_b_U4first3=round(d_b4, 2), d_claim_U4first3=round(d_c4, 2)))
            print(f"  {rows[-1]}")
        outd = ROOT / "analysis" / "output" / "rq3_v2"; outd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(outd / "subspace_separation.csv", index=False)
        fig_scatter_belief(data)
        fig_scatter_dimension(data)
        fig_discriminant_density(data)
    else:
        print("no subspace_proj.parquet found; run scripts/capture_subspace_viz.py on GPU first")
    fig_layer_trajectory()   # uses geometry.csv (already on CPU), independent of the parquet


if __name__ == "__main__":
    main()
