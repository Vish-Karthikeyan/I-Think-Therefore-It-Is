"""Step 16: the belief subspace across layers, positions and erasure (RQ3 visualization add-on).

    python analysis/16_subspace_layers_viz.py

Inputs (from scripts/capture_subspace_layers.py, per model in results/rq3_v2/<model>_chat/):
  subspace_layers.parquet   set (cf = training topics, held = held-out validation topics), cond (pos = believes
                            H1, neg = believes H0, none = no belief), idx, layer, pos (last = answer position,
                            span = belief-sentence tokens), c0, c1 = coordinates in that layer's own 2-D subspace.
  subspace_erasure.parquet  held trials, run (unerased / erased at the erase layer), downstream layers.
  erasure_answers.parquet   answers P(H1) for the same held trials and runs.

Two descriptions of a stated belief (see analysis/14): b = relative to the scored hypothesis (pos = +1, neg = -1),
and the concrete claim belief_dir = b * h_dir (policy helps / hurts). Every axis is fit on the training-topic
(cf) trials and evaluated on the held-out trials, so separations are out-of-sample.

Figures (paper/secondary/figures/candidates/):
  F17  the split forming: answer-position plane at each saved layer (rows = models), held-out trials,
       believes H1 / believes H0 / no belief.
  F18  erasure: downstream of the erase layer, erased vs unerased held-out trials in each layer's plane, plus
       the answers (b-signed logit P(H1)) unerased vs erased.
  F19  what is encoded where: out-of-sample separation (Cohen's d) of the answer-relative belief b and of the
       concrete claim, by layer, at the answer position and at the belief-sentence tokens.
Writes analysis/output/rq3_v2/subspace_layers_summary.csv and erasure_summary.csv.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "rq3_v2"
OUTD = ROOT / "analysis" / "output" / "rq3_v2"; OUTD.mkdir(parents=True, exist_ok=True)
FIG = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["olmo_instruct", "llama", "tulu_final"]
NAME = {"olmo_instruct": "OLMo 2 7B Instruct", "llama": "Llama 3.1 8B Instruct", "tulu_final": "Tulu 3 final"}
ERASE = {"olmo_instruct": 24, "llama": 16, "tulu_final": 24}
COL = {"pos": "#0072B2", "neg": "#D55E00", "none": "#999999"}
MK = {"pos": "o", "neg": "^", "none": "s"}
LAB = {"pos": "believes H1", "neg": "believes H0", "none": "no belief"}
CLAIM = {"b": "#000000", "claim": "#CC79A7"}


def cohen_d(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) or 1.0
    return (np.mean(a) - np.mean(b)) / sp


def load(m):
    d = RES / f"{m}_chat"
    if not (d / "subspace_layers.parquet").exists():
        print(f"  skip {m}: no subspace_layers.parquet"); return None
    L = pd.read_parquet(d / "subspace_layers.parquet")
    meta = {"cf": pd.read_parquet(d / "cf_trials.parquet"), "held": pd.read_parquet(d / "held_trials.parquet")}
    hd = pd.concat([pd.DataFrame(dict(set=s, idx=np.arange(len(f)), h_dir=f.h_dir.to_numpy(), topic=f.topic.to_numpy()))
                    for s, f in meta.items()])
    L = L.merge(hd, on=["set", "idx"], how="left")
    L["b"] = L.cond.map({"pos": 1, "neg": -1, "none": 0})
    L["claim"] = L.b * L.h_dir
    E = pd.read_parquet(d / "subspace_erasure.parquet") if (d / "subspace_erasure.parquet").exists() else None
    A = pd.read_parquet(d / "erasure_answers.parquet") if (d / "erasure_answers.parquet").exists() else None
    if E is not None:
        E = E.merge(hd[hd.set == "held"].drop(columns="set"), on="idx", how="left")
        E["b"] = E.cond.map({"pos": 1, "neg": -1, "none": 0})
    if A is not None:
        A = A.merge(hd[hd.set == "held"].drop(columns="set"), on="idx", how="left")
    return L, E, A


def frame(train, label):
    """Unit axis within the 2-D plane separating label=+1 from -1 on training rows, the orthogonal axis, and a
    centre (training no-belief mean if present, else the midpoint) -> function mapping (c0,c1) to (x, y)."""
    X = train[["c0", "c1"]].to_numpy(); y = train[label].to_numpy()
    w = X[y > 0].mean(0) - X[y < 0].mean(0); w = w / (np.linalg.norm(w) or 1.0)
    o = np.array([-w[1], w[0]])
    none = train[train.cond == "none"]
    c = none[["c0", "c1"]].to_numpy().mean(0) if len(none) else (X[y > 0].mean(0) + X[y < 0].mean(0)) / 2
    return lambda df: ((df[["c0", "c1"]].to_numpy() - c) @ w, (df[["c0", "c1"]].to_numpy() - c) @ o)


def oos_d(L, pos, layer, label):
    """Out-of-sample d: axis fit on cf (training topics), evaluated on held-out trials (belief-stated only)."""
    tr = L[(L.set == "cf") & (L.pos == pos) & (L.layer == layer) & (L.b != 0)]
    te = L[(L.set == "held") & (L.pos == pos) & (L.layer == layer) & (L.b != 0)]
    if len(tr) < 20 or len(te) < 20:
        return np.nan
    f = frame(tr, label); x, _ = f(te); y = te[label].to_numpy()
    return cohen_d(x[y > 0], x[y < 0])


# ---------------------------------------------------------------------------------------------
def fig17(data):
    layers = sorted(set().union(*[set(L.layer.unique()) for _, (L, _, _) in data]))
    fig, axes = plt.subplots(len(data), len(layers), figsize=(2.25 * len(layers), 2.4 * len(data)), squeeze=False)
    rows = []
    for i, (m, (L, _, _)) in enumerate(data):
        for j, l in enumerate(layers):
            ax = axes[i][j]
            tr = L[(L.set == "cf") & (L.pos == "last") & (L.layer == l)]
            te = L[(L.set == "held") & (L.pos == "last") & (L.layer == l)]
            if tr.empty:
                ax.axis("off"); continue
            f = frame(tr[tr.b != 0].assign(cond=tr.cond).pipe(lambda t: pd.concat([t, tr[tr.b == 0]])), "b")
            x, y = f(te)
            sd = np.sqrt((np.var(x[te.b.values > 0], ddof=1) + np.var(x[te.b.values < 0], ddof=1)) / 2) or 1.0
            x, y = x / sd, y / sd
            for c in ("none", "neg", "pos"):
                s = (te.cond == c).to_numpy()
                ax.scatter(x[s], y[s], s=5, alpha=0.35, color=COL[c], marker=MK[c], linewidths=0)
                ax.scatter([x[s].mean()], [y[s].mean()], s=60, color=COL[c], marker=MK[c], edgecolor="black",
                           linewidths=1.0, zorder=5)
            bpos, bneg, bnon = (te.b.values > 0), (te.b.values < 0), (te.b.values == 0)
            d_pol = cohen_d(x[bpos], x[bneg])
            pres = np.linalg.norm([np.r_[x[bpos], x[bneg]].mean() - x[bnon].mean(), np.r_[y[bpos], y[bneg]].mean() - y[bnon].mean()])
            rows.append(dict(model=m, layer=l, d_H1_vs_H0=round(d_pol, 2), presence_shift_sd=round(pres, 2)))
            ax.set_title(f"L{l}: $d$={d_pol:.1f}", fontsize=8.5)
            ax.axvline(0, color="black", lw=0.6, ls=":"); ax.tick_params(labelsize=6.5)
            if j == 0:
                ax.set_ylabel(NAME[m], fontsize=8.5)
    handles = [Line2D([], [], marker=MK[c], ls="", color=COL[c], label=LAB[c]) for c in ("pos", "neg", "none")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The belief split forming across layers (answer position, held-out trials; each layer's own 2-D "
                 "subspace, x = belief axis, units = within-belief sd)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "F17_belief_split_by_layer")
    return pd.DataFrame(rows)


def fig18(data):
    have = [(m, E, A) for m, (_, E, A) in data if E is not None and A is not None]
    if not have:
        print("  F18 skipped: no erasure data"); return None
    fig, axes = plt.subplots(len(have), 4, figsize=(12.5, 3.0 * len(have)), squeeze=False)
    rows = []
    for i, (m, E, A) in enumerate(have):
        layers = sorted(E.layer.unique()); l0 = layers[0]
        dun, der = [], []
        for l in layers:                                   # separation by downstream layer, same frame per layer
            un = E[(E.run == "unerased") & (E.layer == l)]; er = E[(E.run == "erased") & (E.layer == l)]
            f = frame(un, "b"); xu, _ = f(un); xe, _ = f(er)
            dun.append(cohen_d(xu[un.b.values > 0], xu[un.b.values < 0]))
            der.append(cohen_d(xe[er.b.values > 0], xe[er.b.values < 0]))
            rows.append(dict(model=m, layer=l, d_unerased=round(dun[-1], 2), d_erased=round(der[-1], 2)))
        un = E[(E.run == "unerased") & (E.layer == l0)]; er = E[(E.run == "erased") & (E.layer == l0)]
        f = frame(un, "b"); xu, yu = f(un); xe, ye = f(er)
        sd = np.sqrt((np.var(xu[un.b.values > 0], ddof=1) + np.var(xu[un.b.values < 0], ddof=1)) / 2) or 1.0
        lim_x = np.percentile(np.abs(np.r_[xu, xe]) / sd, 99.5) * 1.1
        lim_y = np.percentile(np.abs(np.r_[yu, ye]) / sd, 99.5) * 1.1
        for j, (df, x, y, tag) in enumerate(((un, xu, yu, "before erasure"), (er, xe, ye, "after erasure"))):
            ax = axes[i][j]
            for c in ("none", "neg", "pos"):
                s = (df.cond == c).to_numpy()
                ax.scatter(x[s] / sd, y[s] / sd, s=6, alpha=0.35, color=COL[c], marker=MK[c], linewidths=0)
                ax.scatter([x[s].mean() / sd], [y[s].mean() / sd], s=70, color=COL[c], marker=MK[c],
                           edgecolor="black", linewidths=1.0, zorder=6)
            ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y)
            ax.axvline(0, color="black", lw=0.6, ls=":"); ax.tick_params(labelsize=7)
            dd = dun[0] if j == 0 else der[0]
            ax.set_title(f"L{l0}, {tag} (cut at L{ERASE[m]}): $d$={dd:.1f}", fontsize=8.5)
        axes[i][0].set_ylabel(NAME[m], fontsize=9)
        ax = axes[i][2]
        ax.plot(layers, dun, "-o", color="black", lw=1.6, ms=4, label="before erasure")
        ax.plot(layers, der, "--o", color="black", markerfacecolor="white", lw=1.6, ms=4, label="after erasure")
        ax.set_ylim(0, max(dun) * 1.15); ax.set_xticks(layers); ax.tick_params(labelsize=7)
        ax.set_title("H1-vs-H0 separation $d$ in later layers", fontsize=8.5); ax.set_xlabel("layer", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7.5)
        ax = axes[i][3]                                    # answers
        A = A.copy(); A["lp"] = np.log(np.clip(A.p, .005, .995) / (1 - np.clip(A.p, .005, .995)))
        base = A[A.cond == "none"].pivot_table(index="idx", columns="run", values="lp")
        for k, run in enumerate(("unerased", "erased")):
            for c, off in (("pos", -0.12), ("neg", 0.12)):
                s = A[(A.cond == c) & (A.run == run)].set_index("idx")
                eff = (1 if c == "pos" else -1) * (s.lp - base[run].reindex(s.index))
                ax.errorbar(k + off, eff.mean(), yerr=1.96 * eff.std(ddof=1) / np.sqrt(eff.notna().sum()),
                            fmt=MK[c], color=COL[c], ms=7, capsize=3)
                rows.append(dict(model=m, layer="answer", run=run, cond=c, belief_effect=round(eff.mean(), 3)))
        ax.axhline(0, color="#999999", lw=0.8); ax.set_xticks([0, 1]); ax.set_xlim(-0.5, 1.5)
        ax.set_xticklabels(["before", "after"], fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title("answer pulled toward the belief\n(log-odds vs no belief)", fontsize=8.5)
    handles = [Line2D([], [], marker=MK[c], ls="", color=COL[c], label=LAB[c]) for c in ("pos", "neg", "none")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Erasing the belief subspace at one layer: the split stays mostly gone downstream, and so does the "
                 "answer's pull toward the belief (held-out trials)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, "F18_erasure_downstream")
    return pd.DataFrame(rows)


def fig19(data):
    fig, axes = plt.subplots(1, len(data), figsize=(3.6 * len(data), 3.3), sharey=True, squeeze=False)
    rows = []
    for ax, (m, (L, _, _)) in zip(axes[0], data):
        layers = sorted(L.layer.unique())
        for pos, ls in (("span", "--"), ("last", "-")):
            for label in ("b", "claim"):
                ds = [oos_d(L, pos, l, label) for l in layers]
                for l, dv in zip(layers, ds):
                    rows.append(dict(model=m, pos=pos, label=label, layer=l, d_oos=round(dv, 2) if np.isfinite(dv) else np.nan))
                ax.plot(layers, np.abs(ds), ls, marker="o" if label == "b" else "D", color=CLAIM[label], lw=1.6, ms=4)
        ax.axvline(ERASE[m], color="#999999", lw=0.8, ls=":")
        ax.set_title(NAME[m], fontsize=9.5); ax.set_xlabel("layer")
    axes[0][0].set_ylabel("held-out separation |Cohen's d|")
    handles = [Line2D([], [], color=CLAIM["b"], marker="o", label="belief vs the scored hypothesis (b)"),
               Line2D([], [], color=CLAIM["claim"], marker="D", label="concrete claim (helps / hurts)"),
               Line2D([], [], color="black", ls="-", label="answer position"),
               Line2D([], [], color="black", ls="--", label="belief-sentence tokens")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("What is encoded where: the two descriptions of the belief, by layer and position "
                 "(axis fit on training topics, tested on held-out topics)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, "F19_what_is_encoded_where")
    return pd.DataFrame(rows)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig); print(f"wrote {FIG/stem}.pdf/.png")


def main():
    data = [(m, x) for m in MODELS if (x := load(m)) is not None]
    if not data:
        return
    s17 = fig17(data); s19 = fig19(data); s18 = fig18(data)
    s17.merge(s19.pivot_table(index=["model", "layer"], columns=["pos", "label"], values="d_oos").pipe(
        lambda t: t.set_axis([f"d_oos_{p}_{q}" for p, q in t.columns], axis=1)).reset_index(),
        on=["model", "layer"], how="outer").to_csv(OUTD / "subspace_layers_summary.csv", index=False)
    if s18 is not None:
        s18.to_csv(OUTD / "erasure_summary.csv", index=False)
    print(f"wrote {OUTD/'subspace_layers_summary.csv'}")


if __name__ == "__main__":
    main()
