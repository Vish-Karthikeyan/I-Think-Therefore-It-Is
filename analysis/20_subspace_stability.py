"""Step 20: stability of the corrected-RQ3 belief subspaces (reviewer question; exploratory, CPU only).

Question. How stable are the answer-position belief subspaces that the corrected RQ3 erases (U2, U4) under
resampling, under a change of topics (unseen Tier 2 validation topics), and compared with chance? And how much
of the belief displacement on unseen topics does a subspace estimated on other data still capture?

How the subspaces are built (src/idea/mech/rq3v2.py, run(), stages 1-2). Every counterfactual trial is rendered
three ways - user believes H1 (pos), believes H0 (neg), states no belief (none) - and the hidden state at the
answer position ("last") is saved at layers 4..24. Then
    U2 = orth([mean(pos - none), mean(neg - none)])
    U4 = top-4 right singular vectors of concat(pos - none, neg - none)   (uncentered, as in rq3v2)
This construction has no random seed (only the random-control subspaces use one), so "seed" variability is
read here as SAMPLING variability: which trials and which topics happened to be drawn.

Analyses, per model at layers 12, 16, 20, 24 (erase layers: OLMo Instruct L24, Llama L16, Tulu final L24):
  trial_boot     resample the 1500 cf trials with replacement (200 reps)
  (cf = 15 training topics: 9 Tier 1 (neutral) + 6 Tier 2 (political); held = 5 unseen Tier 2 topics)
  topic_boot     resample the cf topics with replacement within tier, keep all their trials (200 reps)
  split_half     random half split of the cf topics within tier, subspace on each half, half vs half (20 splits)
  unseen_topics  for each of the 6 ways to pick 5 of the 6 seen Tier 2 topics: reference U_R on all OTHER cf
                 topics; subspace on the 300 held trials (5 unseen Tier 2 topics) vs U_R; paired with
  matched_ref    a size-matched subspace (300 trials) from the 5 picked SEEN Tier 2 topics vs the same U_R -
                 the cosine expected from small-sample noise and the Tier 2 vs rest difference, without the
                 topics being new (10 subsamples per pick)
  held_vs_full, held_vs_tier2   held-built subspace vs the full-cf / the cf-Tier-2-only subspace
  tier1_vs_tier2 cf Tier 1 topics only vs cf Tier 2 topics only (a large shift: neutral vs political topics)
  rand_iso       random k-dim subspaces, isotropic in R^d (200 reps)
  rand_data      random k-dim subspaces inside the data: span of random Gaussian mixtures of the centred
                 no-belief states (variance-weighted; a harder chance baseline than isotropic) (200 reps)
For each subspace U' we report the principal-angle cosines with the reference (singular values of U' U_ref^T):
cos_max, cos_min, cos_mean and overlap = mean cos^2 (= ||U' U_ref^T||_F^2 / k).

Geometric retention proxy (NOT behavioural). For the held trials (unseen topics), X = concat(pos - none,
neg - none) at that layer. share_trial(U) = ||X U^T||^2 / ||X||^2 (share of per-trial belief displacement
energy inside U); share_mean(U) = share of the two mean displacements mean(pos - none), mean(neg - none) inside
U (the erasure resets the subspace coordinates to the no-belief mean, so this is the part that tracks the mean
shift it removes). retention = share(U') / share(U_full_cf). U_full_cf never sees the held trials.

Behavioural retention (from existing results, no new computation): the erasure setting was chosen on
validation topics (sweep_validation_v2.csv) and tested once on held-out test topics (steering_test_v2.csv);
gap closed = 1 - gap_after / gap_before.

Usage
  On FarmShare (CPU job), per model:
    python analysis/20_subspace_stability.py compute --model olmo_instruct \
        --capture-dir /scratch/users/$USER/idea/runs/main-8b15a8c4/results/rq3_v2/olmo_instruct_chat --out DIR
  Locally, after copying DIR/*.csv into analysis/output/rq3_v2/stability/:
    python analysis/20_subspace_stability.py report      # summary CSVs, behavioural table, figure F23
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "rq3_v2"
OUTD = ROOT / "analysis" / "output" / "rq3_v2" / "stability"
FIG = ROOT.parent / "paper" / "secondary" / "figures" / "candidates"
MODELS = ["olmo_instruct", "llama", "tulu_final"]
NAME = {"olmo_instruct": "OLMo 2 7B Instruct", "llama": "Llama 3.1 8B Instruct", "tulu_final": "Tülu 3 8B (final)"}
ERASE = {"olmo_instruct": 24, "llama": 16, "tulu_final": 24}
LAYERS = [12, 16, 20, 24]
KS = (2, 4)
MCOL = dict(olmo_instruct="#0072B2", llama="#E69F00", tulu_final="#009E73")   # Okabe-Ito, as in analysis/13


# ---------------------------------------------------------------------------------------------- linear algebra
def orth(M):
    q, _ = np.linalg.qr(np.asarray(M, np.float64).T)
    return q.T


def top_rsv_exact(X, k):
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k].astype(np.float64)


def top_rsv(X, k, rng, p=12, q=4):
    """Top-k right singular vectors by randomized SVD with q power iterations (checked against exact SVD)."""
    Om = rng.standard_normal((X.shape[1], k + p)).astype(X.dtype)
    Y, _ = np.linalg.qr(X @ Om)
    for _ in range(q):
        Z, _ = np.linalg.qr(X.T @ Y)
        Y, _ = np.linalg.qr(X @ Z)
    _, _, Vt = np.linalg.svd(Y.T @ X, full_matrices=False)
    return Vt[:k].astype(np.float64)


def pcos(A, B):
    """Principal-angle cosines between the row spaces of orthonormal A [k,d] and B [k,d], descending."""
    return np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0, 1)


def build(P, N, Z, k, rng=None, exact=False):
    """U2 (from the two mean displacements) or U4 (top-k SVD of the stacked displacements), as in rq3v2.run()."""
    dp, dn = P - Z, N - Z
    if k == 2:
        return orth(np.stack([dp.mean(0), dn.mean(0)]))
    X = np.concatenate([dp, dn])
    return top_rsv_exact(X, k) if exact else top_rsv(X, k, rng)


def shares(U, X, mp, mn):
    st = float(np.linalg.norm(X @ U.T) ** 2 / np.linalg.norm(X) ** 2)
    sm = float((np.linalg.norm(U @ mp) ** 2 + np.linalg.norm(U @ mn) ** 2) / (mp @ mp + mn @ mn))
    return st, sm


# ---------------------------------------------------------------------------------------------- compute (FarmShare)
def compute(model, cdir: Path, out: Path, reps=200, n_split=20, n_shift=10, seed=20261116):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    C = np.load(cdir / "capture.npz")
    Dsaved = np.load(cdir / "directions_v2.npz") if (cdir / "directions_v2.npz").exists() else None
    cf = pd.read_parquet(cdir / "cf_trials.parquet")
    held = pd.read_parquet(cdir / "held_trials.parquet")
    cf_topic, cf_tier = cf.topic.to_numpy(), cf.tier.to_numpy()
    topics = np.unique(cf_topic)
    by_topic = {t: np.flatnonzero(cf_topic == t) for t in topics}
    tier_topics = {tr: np.unique(cf_topic[cf_tier == tr]) for tr in sorted(np.unique(cf_tier))}
    print(f"[{model}] cf {len(cf)} trials / {len(topics)} topics; held {len(held)} trials / "
          f"{held.topic.nunique()} topics (tiers {sorted(held.tier.unique())})", flush=True)
    rows, ref = [], []

    for l in LAYERS:
        g = lambda key: C[f"{key}_last_L{l}"].astype(np.float32)  # noqa: E731
        P, N, Z = g("cf_pos"), g("cf_neg"), g("cf_none")
        hP, hN, hZ = g("held_pos"), g("held_neg"), g("held_none")
        okh = np.isfinite(hP).all(1) & np.isfinite(hN).all(1) & np.isfinite(hZ).all(1)
        hP, hN, hZ = hP[okh], hN[okh], hZ[okh]
        Xh = np.concatenate([hP - hZ, hN - hZ]).astype(np.float64)
        mp, mn = (hP - hZ).mean(0).astype(np.float64), (hN - hZ).mean(0).astype(np.float64)
        Zc = (Z - Z.mean(0)).astype(np.float32)
        d = P.shape[1]
        for k in KS:
            rng = np.random.default_rng([seed, l, k])
            Uf = build(P, N, Z, k, exact=True)
            Ufr = build(P, N, Z, k, rng=rng)
            Uh = build(hP, hN, hZ, k, exact=True)
            base_t, base_m = shares(Uf, Xh, mp, mn)
            held_t, held_m = shares(Uh, Xh, mp, mn)
            saved_cos = (pcos(Uf, orth(Dsaved[f"L{l}/ans_U{k}"])).min()
                         if Dsaved is not None and f"L{l}/ans_U{k}" in Dsaved else np.nan)
            ref.append(dict(model=model, layer=l, k=k, d=d, n_cf=len(P), n_held=int(okh.sum()),
                            share_trial_fullcf=base_t, share_mean_fullcf=base_m,
                            share_trial_heldbuilt=held_t, share_mean_heldbuilt=held_m,
                            cos_min_vs_saved_directions=float(saved_cos),
                            cos_min_rsvd_vs_exact=float(pcos(Ufr, Uf).min()) if k > 2 else 1.0,
                            cos_min_held_vs_full=float(pcos(Uh, Uf).min()),
                            overlap_held_vs_full=float((pcos(Uh, Uf) ** 2).mean())))

            def rec(analysis, rep, U, Uref, n_used=np.nan):
                c = pcos(U, Uref)
                st, sm = shares(U, Xh, mp, mn)
                rows.append(dict(model=model, layer=l, k=k, analysis=analysis, rep=rep, n_used=n_used,
                                 **{f"cos{i + 1}": float(c[i]) for i in range(k)},
                                 cos_max=float(c[0]), cos_min=float(c[-1]), cos_mean=float(c.mean()),
                                 overlap=float((c ** 2).mean()), share_trial=st, share_mean=sm,
                                 retention_trial=st / base_t, retention_mean=sm / base_m))

            rec("held_vs_full", 0, Uh, Uf, int(okh.sum()))
            n = len(P)
            rsvd_err = []
            for r in range(reps):                                     # trial bootstrap
                idx = rng.integers(0, n, n)
                Ub = build(P[idx], N[idx], Z[idx], k, rng=rng)
                if k > 2 and r < 3:                                   # randomized vs exact SVD on resamples
                    rsvd_err.append(1 - pcos(Ub, build(P[idx], N[idx], Z[idx], k, exact=True)).min())
                rec("trial_boot", r, Ub, Uf, n)
            ref[-1]["max_1mcos_rsvd_vs_exact_boot"] = float(max(rsvd_err)) if rsvd_err else 0.0
            for r in range(reps):                                     # topic bootstrap, stratified by tier
                pick = np.concatenate([rng.choice(tt, len(tt), replace=True) for tt in tier_topics.values()])
                idx = np.concatenate([by_topic[t] for t in pick])
                rec("topic_boot", r, build(P[idx], N[idx], Z[idx], k, rng=rng), Uf, len(idx))
            for r in range(n_split):                                  # topic split-half, stratified by tier
                ha, hb = [], []
                for tt in tier_topics.values():
                    pp = rng.permutation(tt)
                    (ha if r % 2 else hb).extend(pp[: len(pp) // 2]); (hb if r % 2 else ha).extend(pp[len(pp) // 2:])
                a = np.concatenate([by_topic[t] for t in ha])
                b = np.concatenate([by_topic[t] for t in hb])
                Ua, Ub = build(P[a], N[a], Z[a], k, rng=rng), build(P[b], N[b], Z[b], k, rng=rng)
                rec("split_half", r, Ua, Ub, len(a))                  # half A vs half B; shares are for half A
                rec("split_half_vs_full", r, Ua, Uf, len(a))
            # domain shift. cf = 6 Tier 2 (political) + 9 Tier 1 (neutral) topics; held = 5 unseen Tier 2 topics.
            # For each choice of 5 of the 6 cf Tier 2 topics: reference subspace U_R on all OTHER cf topics
            # (9 Tier 1 + 1 Tier 2); compare U_held with U_R (unseen topics) and a size-matched U from those
            # 5 seen Tier 2 topics with U_R (matched_ref: same small-sample noise, no topic novelty).
            t2 = list(tier_topics[2])
            n_small = held.topic.nunique()
            for ci, drop in enumerate(itertools.combinations(t2, n_small)):
                rest = np.flatnonzero(~np.isin(cf_topic, drop))
                Ur = build(P[rest], N[rest], Z[rest], k, rng=rng)
                pool = np.flatnonzero(np.isin(cf_topic, drop))
                rec("rest_vs_full", ci, Ur, Uf, len(rest))
                for s_ in range(n_shift):
                    rep = ci * n_shift + s_
                    sm_idx = rng.permutation(pool)[: int(okh.sum())]
                    rec("matched_ref", rep, build(P[sm_idx], N[sm_idx], Z[sm_idx], k, rng=rng), Ur, len(sm_idx))
                rec("unseen_topics", ci, Uh, Ur, int(okh.sum()))
            i1, i2 = np.flatnonzero(cf_tier == 1), np.flatnonzero(cf_tier == 2)
            U1, U2t = build(P[i1], N[i1], Z[i1], k, exact=True), build(P[i2], N[i2], Z[i2], k, exact=True)
            rec("tier1_vs_tier2", 0, U1, U2t, len(i1))                # strong shift: neutral vs political topics
            rec("tier2_vs_full", 0, U2t, Uf, len(i2))
            rec("held_vs_tier2", 0, Uh, U2t, int(okh.sum()))          # unseen vs seen Tier 2 topics
            for r in range(reps):                                     # chance
                rec("rand_iso", r, orth(rng.standard_normal((k, d))), Uf)
                rec("rand_data", r, orth(rng.standard_normal((k, len(Zc))).astype(np.float32) @ Zc), Uf)
            print(f"[{model}] L{l} k{k} done ({time.time() - t0:.0f}s); full-cf share of held displacement "
                  f"trial={base_t:.3f} mean={base_m:.3f}; held-vs-full cos_min={ref[-1]['cos_min_held_vs_full']:.3f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(out / f"reps_{model}.csv", index=False)
    pd.DataFrame(ref).to_csv(out / f"reference_{model}.csv", index=False)
    print(f"[{model}] wrote {out}/reps_{model}.csv, reference_{model}.csv ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------- report (local)
ORDER = ["trial_boot", "topic_boot", "split_half", "matched_ref", "unseen_topics", "held_vs_tier2",
         "tier1_vs_tier2", "rand_data", "rand_iso"]
LABEL = {"trial_boot": "trial bootstrap", "topic_boot": "topic bootstrap", "split_half": "topic split-half",
         "matched_ref": "5 seen Tier 2 topics\n(size-matched)", "unseen_topics": "5 unseen Tier 2\ntopics",
         "held_vs_tier2": "unseen vs seen\nTier 2 topics", "tier1_vs_tier2": "Tier 1 vs Tier 2\ntopics",
         "held_vs_full": "unseen vs full", "rand_data": "random (in data)", "rand_iso": "random (isotropic)"}


def q(x, p):
    return float(np.nanpercentile(x, p)) if len(x) else np.nan


def summarize(R):
    out = []
    for (m, l, k, a), g in R.groupby(["model", "layer", "k", "analysis"], sort=False):
        row = dict(model=m, layer=l, k=k, analysis=a, n_reps=len(g))
        for c in ["cos_max", "cos_min", "cos_mean", "overlap", "share_trial", "share_mean",
                  "retention_trial", "retention_mean"]:
            row[f"{c}_median"], row[f"{c}_p05"], row[f"{c}_p95"] = q(g[c], 50), q(g[c], 5), q(g[c], 95)
        out.append(row)
    return pd.DataFrame(out)


def behavioural():
    rows = []
    for m in MODELS:
        d = RES / f"{m}_chat"
        summ = json.loads((d / "summary_v2.json").read_text())
        fam_set = dict((f, s) for f, s in summ["chosen"])
        s = fam_set.get("erase_answer")
        val = pd.read_csv(d / "sweep_validation_v2.csv")
        test = pd.read_csv(d / "steering_test_v2.csv")
        v = val[(val.family == "erase_answer") & (val.setting == s)].iloc[0]
        t = test[(test.family == "erase_answer") & (test.setting == s)].iloc[0]
        c = test[(test.family == "control_random_subspace") & (test.setting == s)]
        suf = pd.read_csv(d / "sufficiency.csv")
        L = int(s.split("_")[0][1:])
        sf = suf[(suf.layer == L) & (suf.patch == f"answer_U{s[-1]}")]
        rows.append(dict(model=m, setting=s,
                         val_gap_before=v.t2_contra_gap_before, val_gap_after=v.t2_contra_gap_after,
                         val_gap_closed=1 - v.t2_contra_gap_after / v.t2_contra_gap_before,
                         test_gap_before=t.t2_contra_gap_before, test_gap_after=t.t2_contra_gap_after,
                         test_gap_closed=t.gap_closed_frac,
                         test_over_val=t.gap_closed_frac / (1 - v.t2_contra_gap_after / v.t2_contra_gap_before),
                         random_control_test_gap_closed=float(c.gap_closed_frac.iloc[0]) if len(c) else np.nan,
                         sufficiency_recovery_heldtopics=float(sf.recovery.mean()) if len(sf) else np.nan,
                         test_t2_neutral_change=t.t2_neutral_change, test_tier1_contra_change=t.tier1_contra_change))
    return pd.DataFrame(rows)


def figure(S, B, ref, fig_dir=FIG):
    """F23, four panels, erase layer of each model. a: U2 smallest principal-angle cosine; b: U4 overlap (mean
    squared cosine) and its two leading cosines; c: geometric retention; d: behavioural gap closed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    rows = ["trial_boot", "topic_boot", "split_half", "matched_ref", "unseen_topics", "held_vs_tier2",
            "tier1_vs_tier2", "rand_data", "rand_iso"]
    rlab = {"trial_boot": "resample trials (bootstrap)", "topic_boot": "resample topics (bootstrap)",
            "split_half": "half of topics vs other half", "matched_ref": "5 seen Tier 2 topics vs rest (n=300)",
            "unseen_topics": "5 UNSEEN Tier 2 topics vs rest (n=300)",
            "held_vs_tier2": "unseen vs seen Tier 2 topics", "tier1_vs_tier2": "Tier 1 vs Tier 2 topics",
            "rand_data": "chance: random, in data span", "rand_iso": "chance: random, isotropic"}
    off = {m: (1 - j) * 0.24 for j, m in enumerate(MODELS)}   # first model on top

    def g(m):
        return S[(S.model == m) & (S.layer == ERASE[m])]

    def dots(ax, k, col, rws, labels):
        y = np.arange(len(rws))[::-1]
        for m in MODELS:
            h = g(m)[g(m).k == k].set_index("analysis")
            med = np.array([h.loc[a, f"{col}_median"] for a in rws])
            lo = np.array([h.loc[a, f"{col}_p05"] for a in rws])
            hi = np.array([h.loc[a, f"{col}_p95"] for a in rws])
            ax.errorbar(med, y + off[m], xerr=[np.maximum(0, med - lo), np.maximum(0, hi - med)], fmt="o", ms=3.5,
                        color=MCOL[m], lw=0.9, capsize=1.5, label=f"{NAME[m]} (L{ERASE[m]})")
        ax.set_yticks(y); ax.set_yticklabels([labels[a] for a in rws])
        ax.grid(axis="x", color="#e5e5e5", lw=0.6); ax.set_xlim(-0.02, 1.05)
        return y

    ax = axes[0, 0]
    dots(ax, 2, "cos_min", rows, rlab)
    ax.set_xlabel("smallest principal-angle cosine (1 = same plane)")
    ax.set_title("a  2-D belief plane U2 (median, 5-95%)", loc="left", fontsize=8)
    ax = axes[0, 1]
    y = dots(ax, 4, "overlap", rows, rlab)
    for m in MODELS:   # leading two principal cosines of U4 (the plane inside it), from the per-rep medians
        h = g(m)[g(m).k == 4].set_index("analysis")
        ax.scatter([h.loc[a, "cos_max_median"] for a in rows], y + off[m], marker="|", s=30, color=MCOL[m], lw=1)
    ax.set_yticklabels([])
    ax.set_xlabel("overlap = mean squared cosine (1 = same subspace)")
    ax.set_title("b  4-D erased subspace U4 (|: largest cosine)", loc="left", fontsize=8)
    ax = axes[1, 0]
    rr = ["topic_boot", "split_half", "rest_vs_full", "rand_data"]
    rl = {"topic_boot": "resample topics (bootstrap)", "split_half": "half of topics",
          "rest_vs_full": "drop 5 of the 6 Tier 2 topics", "rand_data": "chance: random, in data span"}
    dots(ax, 4, "retention_mean", rr, rl)
    ax.axvline(1, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("share of unseen-topic mean belief shift inside U4,\nrelative to the full-sample U4 (geometric)")
    ax.set_title("c  geometric retention on unseen topics", loc="left", fontsize=8)
    ax = axes[1, 1]
    cats = [("val_gap_closed", "erasure, validation topics\n(where it was chosen)"),
            ("test_gap_closed", "erasure, held-out test topics\n(tested once)"),
            ("random_control_test_gap_closed", "random 4-D control,\ntest topics")]
    yy = np.arange(len(cats))[::-1]
    for m in MODELS:
        b = B[B.model == m].iloc[0]
        ax.barh(yy + off[m], [b[c] for c, _ in cats], height=0.22, color=MCOL[m])
        for yv, (c, _) in zip(yy, cats):
            ax.text(b[c] + 0.01, yv + off[m], f"{b[c]:.2f}", va="center", fontsize=6, color="#333333")
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l in cats]); ax.set_xlim(0, 1.1)
    ax.set_xlabel("share of the Tier 2 contradiction gap closed (behavioural)")
    ax.set_title("d  behavioural retention", loc="left", fontsize=8)
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"F23_subspace_stability.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {fig_dir}/F23_subspace_stability.pdf/.png")


def report(outd=OUTD, fig_dir=FIG):
    OUTD = outd
    R = pd.concat([pd.read_csv(f) for f in sorted(OUTD.glob("reps_*.csv"))], ignore_index=True)
    ref = pd.concat([pd.read_csv(f) for f in sorted(OUTD.glob("reference_*.csv"))], ignore_index=True)
    S = summarize(R)
    S.to_csv(OUTD / "stability_summary.csv", index=False)
    B = behavioural()
    B.to_csv(OUTD / "behavioral_retention.csv", index=False)
    # compact headline table: erase layer, both k
    keep = ["trial_boot", "topic_boot", "split_half", "split_half_vs_full", "matched_ref", "unseen_topics",
            "held_vs_full", "held_vs_tier2", "tier1_vs_tier2", "tier2_vs_full", "rest_vs_full", "rand_data", "rand_iso"]
    H = S[S.apply(lambda r: r.layer == ERASE[r.model], axis=1) & S.analysis.isin(keep)]
    cols = ["model", "layer", "k", "analysis", "cos_min_median", "cos_min_p05", "cos_mean_median", "overlap_median",
            "retention_mean_median", "retention_mean_p05", "retention_trial_median", "retention_trial_p05"]
    H = H[cols].copy()
    H["analysis"] = pd.Categorical(H.analysis, keep, ordered=True)
    H = H.sort_values(["model", "k", "analysis"])
    H.to_csv(OUTD / "headline_erase_layer.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(H.round(3).to_string(index=False))
        print(ref.round(3).to_string(index=False))
        print(B.round(3).to_string(index=False))
    figure(S, B, ref, fig_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--model", required=True)
    c.add_argument("--capture-dir", required=True, type=Path)
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--reps", type=int, default=200)
    r = sub.add_parser("report")
    r.add_argument("--in-dir", type=Path, default=OUTD, help="folder with reps_*.csv / reference_*.csv")
    r.add_argument("--fig-dir", type=Path, default=FIG)
    a = ap.parse_args()
    if a.cmd == "compute":
        compute(a.model, a.capture_dir, a.out, reps=a.reps)
    else:
        report(a.in_dir, a.fig_dir)
