"""Step 14: visualize the belief subspace (RQ3 add-on).

    python analysis/14_subspace_viz.py

The corrected RQ3 finds a low-dimensional subspace at the answer position that encodes the user's stated
belief. scripts/capture_subspace_viz.py projects every counterfactual trial's answer-position hidden state
(at each model's erase layer) onto that subspace and writes results/rq3_v2/<model>_chat/subspace_proj.parquet
with coordinates u2_0,u2_1 (the 2-D belief subspace ans_U2) and u4_0..u4_2 (first 3 dims of ans_U4).

This script makes the figures (candidates dir), mirroring how the linear-representation / interpretability
literature visualizes a concept subspace (2-D projection onto the concept directions, class-colored scatter,
and a layerwise trajectory):
  F12  belief-subspace scatter, colored by STATED BELIEF (believe-H1 vs believe-H0), one panel per model.
       The two beliefs separate along the subspace -> the "see the concept direction" plot. [KEY FIGURE]
  F13  same scatter, colored by IDENTITY DIMENSION, marker = stated belief. The belief split holds across
       dimensions/topics -> the belief axis is identity-/topic-agnostic.
  F14  layerwise geometry trajectory (geometry.csv): cosine between the believe-H1 and believe-H0 mean
       directions vs layer. cos ~ +1 early (shared "a belief is present" direction) -> negative late
       (the two beliefs point apart: polarity). The presence->polarity transition.
  F15  1-D density of each trial's projection onto the within-subspace belief-discriminant axis, per model;
       the believe-H1 and believe-H0 distributions separate (classifier view of the same subspace).

All cf trials here are belief-stated (belief_dir in {-1,+1}); there are no no-belief rows.
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
    if not p.exists():
        print(f"  skip {model} (missing {p.name})"); return None
    return pd.read_parquet(p)


SUB_COLS = ("u4_0", "u4_1", "u4_2")   # the captured belief subspace (first 3 dims of ans_U4)


def belief_axis(df, cols=SUB_COLS):
    """Within the captured belief subspace, the unit difference-of-class-means (LDA-style) direction that
    separates believe-H1 from believe-H0, each trial's score on it (signed so believe-H1 is positive), the
    Cohen's d of that separation, and each trial's score on the leading ORTHOGONAL direction (residual PC1,
    for a 2-D plot of 'belief axis vs orthogonal variation'). Standard concept-direction + residual-PC view.

    We use ans_U4 because the top-2 subspace ans_U2 carries mostly the common-mode 'belief present' magnitude;
    the *polarity* (which belief) lives in the added ans_U4 dimension (see the note / F14)."""
    X = df[list(cols)].to_numpy().astype(np.float64)
    bd = df.belief_dir.to_numpy()
    w = X[bd > 0].mean(0) - X[bd < 0].mean(0)
    w = w / (np.linalg.norm(w) or 1.0)
    Xc = X - X.mean(0)
    s = Xc @ w                                               # belief-axis score (centered)
    pos, neg = s[bd > 0], s[bd < 0]
    sp = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2) or 1.0
    d = (pos.mean() - neg.mean()) / sp
    R = Xc - np.outer(s, w)                                  # remove the belief axis
    if R.shape[1] > 1 and np.linalg.norm(R) > 0:
        _, _, Vt = np.linalg.svd(R, full_matrices=False)
        y = R @ Vt[0]                                        # leading orthogonal direction
    else:
        y = np.zeros(len(df))
    return w, s, y, d


# --------------------------------------------------------------------------------------
def fig_scatter_belief(data):
    """F12: belief-subspace scatter colored by stated belief, one panel per model.
    x = belief axis (ans_U4 difference-of-means direction), y = leading orthogonal direction."""
    fig, axes = plt.subplots(1, len(data), figsize=(3.5 * len(data), 3.6), squeeze=False)
    for ax, (m, df) in zip(axes[0], data):
        _, s, y, d = belief_axis(df)
        bd = df.belief_dir.to_numpy()
        for b in (1, -1):
            ax.scatter(s[bd == b], y[bd == b], s=10, alpha=0.35, color=BELIEF_COLOR[b],
                       marker=BELIEF_MARK[b], linewidths=0, label=BELIEF_LABEL[b])
        for b in (1, -1):                                    # class centroids
            ax.scatter([s[bd == b].mean()], [y[bd == b].mean()], s=150, color=BELIEF_COLOR[b],
                       marker=BELIEF_MARK[b], edgecolor="black", linewidths=1.4, zorder=5)
        ax.axvline(0, color="black", lw=1.1, ls="--", zorder=4)
        ax.set_title(f"{NAME[m]}\nL{ERASE_LAYER[m]} belief subspace, Cohen's $d$={d:.2f}", fontsize=9)
        ax.set_xlabel("belief axis  (ans_U4 discriminant)")
        ax.axhline(0, color=OI["grey"], lw=0.6)
    axes[0][0].set_ylabel("leading orthogonal direction")
    axes[0][0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle("Belief subspace at the answer position: stated beliefs separate along the belief axis",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "F12_belief_subspace_scatter")


def fig_scatter_dimension(data):
    """F13: same scatter colored by identity dimension, marker = stated belief."""
    fig, axes = plt.subplots(1, len(data), figsize=(3.5 * len(data), 3.6), squeeze=False)
    for ax, (m, df) in zip(axes[0], data):
        _, s, y, d = belief_axis(df)
        bd = df.belief_dir.to_numpy(); dims = df.dimension.to_numpy()
        for dim in [x for x in DIM_ORDER if x in set(df.dimension)]:
            for b in (1, -1):
                sel = (dims == dim) & (bd == b)
                if sel.any():
                    ax.scatter(s[sel], y[sel], s=10, alpha=0.5, color=DIM_COLOR[dim],
                               marker=BELIEF_MARK[b], linewidths=0)
        ax.axvline(0, color="black", lw=1.0, ls="--", zorder=4)
        ax.set_title(f"{NAME[m]}  (belief $d$={d:.2f})", fontsize=9)
        ax.set_xlabel("belief axis (ans_U4 discriminant)")
        ax.axhline(0, color=OI["grey"], lw=0.6)
    axes[0][0].set_ylabel("leading orthogonal direction")
    dims_present = [x for x in DIM_ORDER if any(x in set(df.dimension) for _, df in data)]
    handles = [Line2D([], [], marker="s", ls="", color=DIM_COLOR[x], label=x) for x in dims_present]
    handles += [Line2D([], [], marker=BELIEF_MARK[b], ls="", color="black", label=BELIEF_LABEL[b]) for b in (1, -1)]
    axes[0][-1].legend(handles=handles, fontsize=7, loc="best", framealpha=0.9, ncol=1)
    fig.suptitle("The belief axis is identity-agnostic: beliefs split the same way within every identity dimension",
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
        _, s, _, d = belief_axis(df)
        bd = df.belief_dir.to_numpy()
        lo, hi = np.percentile(s, [1, 99]); bins = np.linspace(lo, hi, 36)
        for b in (1, -1):
            ax.hist(s[bd == b], bins=bins, density=True, alpha=0.55, color=BELIEF_COLOR[b],
                    label=BELIEF_LABEL[b])
        ax.axvline(0, color=OI["grey"], lw=0.8)
        ax.set_title(f"{NAME[m]}  (Cohen's $d$={d:.2f})", fontsize=9)
        ax.set_xlabel("projection onto belief-discriminant axis")
    axes[0][0].set_ylabel("density"); axes[0][0].legend(fontsize=8)
    fig.suptitle("Separation along the belief-discriminant axis within the subspace", fontsize=11)
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
        for m, df in data:
            _, _, _, d = belief_axis(df)
            print(f"  {m}: n={len(df)} belief-axis Cohen's d={d:.3f}")
        fig_scatter_belief(data)
        fig_scatter_dimension(data)
        fig_discriminant_density(data)
    else:
        print("no subspace_proj.parquet found; run scripts/capture_subspace_viz.py on GPU first")
    fig_layer_trajectory()   # uses geometry.csv (already on CPU), independent of the parquet


if __name__ == "__main__":
    main()
