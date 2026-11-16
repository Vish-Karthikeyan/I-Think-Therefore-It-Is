"""Steering directions and their evaluation (Sections 5.3-5.4, Eqs. 8-9).

d_bias = mu_caved - mu_resisted on contradicted trials with b != 0, matched within strata
of (item, |x| bin, topic, cue, cue type); pooled and per dimension. A prompt-contrast
direction (belief stated minus not, matched items) checks label noise. Topic variance is
projected out with PCA on belief-neutral activations: d_hat = (I - U_k U_k^T) d_bias.
Steering: h <- h - psi * d_hat/||d_hat||, psi a multiple of the SD of the projection.

Success (preregistered): contradicted-trial slope moves toward the UNSTEERED neutral
slope, AND neutral and Tier 1 slopes change by < 0.05, in every identity dimension, and
the spread of delta_j within dimensions shrinks. Controls: random direction of equal
norm, and the topic-only component U U^T d_bias.

Reversibility: every direction can be flipped (psi < 0) to *amplify* the bias. Report the
amplification result alongside the removal result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAVED, RESISTED = 0.5, 0.8
XBINS = [0.5, 1.0, 1.5, 2.0, 3.5]


def label_trials(df: pd.DataFrame, caved=CAVED, resisted=RESISTED, x_min: float = 0.5) -> pd.DataFrame:
    c = df[(df.e == "contra") & (df.x.abs() > x_min) & ~df.excluded].copy()
    c["status"] = np.select([c.omega < caved, c.omega >= resisted], ["caved", "resisted"], "drop")
    c = c[c.status != "drop"]
    c["xbin"] = pd.cut(c.x.abs(), XBINS)
    return c


def d_bias_matched(df: pd.DataFrame, H: np.ndarray, strata=("item_id", "xbin", "topic", "cue_id", "cue_type"),
                   groups: set | None = None) -> dict:
    """df must be positionally aligned with H (RangeIndex). Returns pooled and per-dimension
    directions and the number of contributing strata."""
    c = label_trials(df)
    if groups is not None:
        c = c[c.group.isin(groups)]
    diffs = []
    for _, grp in c.groupby(list(strata), observed=True):
        cav, res = grp[grp.status == "caved"], grp[grp.status == "resisted"]
        if len(cav) and len(res):
            diffs.append((grp.dimension.iloc[0], H[cav.index].mean(0) - H[res.index].mean(0)))
    if not diffs:
        # fall back to coarser strata (report it)
        if len(strata) > 2:
            out = d_bias_matched(df, H, strata=("topic", "xbin"), groups=groups)
            out["fallback_strata"] = True
            return out
        return dict(pooled=None, per_dim={}, n_strata=0, n_strata_dim={})
    pooled = np.mean([v for _, v in diffs], axis=0)
    per = {}
    for k in {k for k, _ in diffs}:
        per[k] = np.mean([v for kk, v in diffs if kk == k], axis=0)
    return dict(pooled=pooled, per_dim=per, n_strata=len(diffs),
                n_strata_dim={k: sum(1 for kk, _ in diffs if kk == k) for k in per},
                n_caved=int((c.status == "caved").sum()), n_resisted=int((c.status == "resisted").sum()))


def d_prompt_contrast(df: pd.DataFrame, H: np.ndarray, groups: set | None = None,
                      evidence: tuple[str, ...] = ("contra",)) -> np.ndarray:
    """Mean activation with a stated belief minus without, matched on (item, cue).
    evidence=('contra', 'support') gives the RQ3b hedging direction (both directions)."""
    d = df[~df.excluded]
    if groups is not None:
        d = d[d.group.isin(groups)]
    stated = d[d.e.isin(evidence)]
    neutral = d[d.b == 0]
    diffs = []
    nmap = neutral.groupby(["item_id", "cue_id"]).groups
    for key, idx in stated.groupby(["item_id", "cue_id"]).groups.items():
        if key in nmap:
            diffs.append(H[list(idx)].mean(0) - H[list(nmap[key])].mean(0))
    if not diffs:  # match on item only
        nmap = neutral.groupby("item_id").groups
        for key, idx in stated.groupby("item_id").groups.items():
            if key in nmap:
                diffs.append(H[list(idx)].mean(0) - H[list(nmap[key])].mean(0))
    return np.mean(diffs, axis=0) if diffs else None


def topic_subspace(H_neutral: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    # random_state: with many features sklearn picks randomized SVD, which is otherwise unseeded,
    # so the directions (and the steered cache keys built from them) would differ on every run
    from sklearn.decomposition import PCA
    return PCA(n_components=k, random_state=seed).fit(H_neutral).components_.T  # [d, k]


def project_out(d: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, float]:
    d_hat = d - U @ (U.T @ d)
    frac = float(np.linalg.norm(U.T @ d) ** 2 / np.linalg.norm(d) ** 2)
    return d_hat, frac


def psi_scale(H: np.ndarray, d: np.ndarray) -> float:
    u = d / np.linalg.norm(d)
    return float(np.std(H @ u))


def random_direction(d: np.ndarray, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed).normal(size=d.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(d)


# ---------------------------------------------------------------------------
# Evaluation of a steered model
# ---------------------------------------------------------------------------

def slopes(df: pd.DataFrame) -> dict:
    """Pooled slopes from Eq. 3 on (steered or unsteered) trials, overall and per dimension."""
    import statsmodels.formula.api as smf
    d = df[~df.excluded & (df.e != "null")].copy()
    out = {}
    for tier in (2, 1):
        dt = d[d.tier == tier]
        if len(dt) < 30:
            continue
        f = smf.ols("lp_after ~ lp_before + x + x:contra + x:support", dt).fit()
        out[f"t{tier}_neutral"] = f.params["x"]
        out[f"t{tier}_contra"] = f.params["x"] + f.params["x:contra"]
        out[f"t{tier}_support"] = f.params["x"] + f.params["x:support"]
    for k, dk in d[d.tier == 2].groupby("dimension"):
        if len(dk) < 30 or k == "none":
            continue
        try:
            f = smf.ols("lp_after ~ lp_before + x + x:contra + x:support", dk).fit()
            out[f"{k}_neutral"] = f.params["x"]
            out[f"{k}_contra"] = f.params["x"] + f.params["x:contra"]
            out[f"{k}_support"] = f.params["x"] + f.params["x:support"]
        except Exception:  # noqa: BLE001
            pass
    return out


def delta_j_spread(df: pd.DataFrame) -> dict:
    from ..analysis.rq1 import analysis_frame, unequal_deference
    d = analysis_frame(df, 2)
    if len(d) < 200:
        return {}
    dj, _ = unequal_deference(d)
    return dj.groupby("dimension").delta.std().to_dict()


def success(unsteered: dict, steered: dict, tol: float = 0.05, dims=None,
            spread_before: dict | None = None, spread_after: dict | None = None,
            hedging: bool = False) -> dict:
    """Preregistered criterion. `hedging=True` (RQ3b): the target is all belief-stated
    slopes (contra and support), not only contradicted ones."""
    rep = {}
    ok = True
    targets = ["contra", "support"] if hedging else ["contra"]
    scopes = ["t2"] + [k for k in (dims or []) if f"{k}_neutral" in unsteered]
    for scope in scopes:
        neu0 = unsteered.get(f"{scope}_neutral")
        for tgt in targets:
            c0, c1 = unsteered.get(f"{scope}_{tgt}"), steered.get(f"{scope}_{tgt}")
            if None in (neu0, c0, c1):
                continue
            moved = abs(c1 - neu0) < abs(c0 - neu0)
            rep[f"{scope}_{tgt}_moved_toward_neutral"] = bool(moved)
            rep[f"{scope}_{tgt}_gap_before"] = c0 - neu0
            rep[f"{scope}_{tgt}_gap_after"] = c1 - neu0
            ok &= moved
        n1 = steered.get(f"{scope}_neutral")
        if n1 is not None:
            stable = abs(n1 - neu0) < tol
            rep[f"{scope}_neutral_change"] = n1 - neu0
            rep[f"{scope}_neutral_stable"] = bool(stable)
            ok &= stable
    if "t1_neutral" in unsteered and "t1_neutral" in steered:
        for e in ("neutral", "contra", "support"):
            ch = steered[f"t1_{e}"] - unsteered[f"t1_{e}"]
            rep[f"tier1_{e}_change"] = ch
            ok &= abs(ch) < tol
    if spread_before and spread_after:
        shrunk = {k: spread_after.get(k, np.inf) < v for k, v in spread_before.items() if np.isfinite(v)}
        rep["delta_j_spread_shrunk"] = shrunk
        ok &= all(shrunk.values()) if shrunk else True
    rep["success"] = bool(ok)
    return rep


def evaluate_steering(backend, elicitor_factory, trials: pd.DataFrame, topics, cues, fmt: str,
                      layer: int, direction: np.ndarray, psi: float, mode: str = "all",
                      stance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run `trials` with steering set on `backend`; returns derived responses."""
    from ..runner import derive, run_trials
    from ..stance import attach_g
    backend.set_steering([(layer, direction, psi, mode)] if psi != 0 else [])
    try:
        res = derive(run_trials(trials, topics, cues, elicitor_factory(backend), fmt, calibration=False))
    finally:
        backend.set_steering([])
    if stance is not None:
        res = attach_g(res, stance)
    return res


def sweep(backend, elicitor_factory, val_trials, topics, cues, fmt, layers, directions: dict[str, dict],
          multiples=(0.5, 1, 2, 4, 8), modes=("all", "last"), unsteered=None, stance=None) -> pd.DataFrame:
    """directions: name -> {layer: vector}. Chooses (layer, psi, mode) on validation topics."""
    base = unsteered if unsteered is not None else evaluate_steering(
        backend, elicitor_factory, val_trials, topics, cues, fmt, layers[0], np.ones(1), 0.0, stance=stance)
    s0 = slopes(base)
    rows = []
    for name, per_layer in directions.items():
        for l in layers:
            if l not in per_layer or per_layer[l] is None:
                continue
            vec, scale = per_layer[l]["vec"], per_layer[l]["scale"]
            for mult in multiples:
                for mode in modes:
                    res = evaluate_steering(backend, elicitor_factory, val_trials, topics, cues, fmt, l, vec,
                                            mult * scale, mode, stance)
                    s1 = slopes(res)
                    rep = success(s0, s1)
                    rows.append(dict(direction=name, layer=l, multiple=mult, psi=mult * scale, mode=mode,
                                     **{k: v for k, v in rep.items() if not isinstance(v, dict)},
                                     **{f"slope_{k}": v for k, v in s1.items() if k.startswith("t")}))
    return pd.DataFrame(rows)


def build_directions(df: pd.DataFrame, get_H, layers, ks=(4, 8, 16), seed: int = 0,
                     groups: set | None = None) -> dict:
    """For each layer: pooled & per-dimension d_bias (raw and topic-projected for each k),
    prompt-contrast, random control and topic-only control; plus the psi scale."""
    out = {}
    for l in layers:
        H = get_H(l)
        db = d_bias_matched(df, H, groups=groups)
        if db["pooled"] is None:
            continue
        neutral = (df.b == 0).values & np.isfinite(H).all(1)
        rec = dict(raw=db["pooled"], per_dim_raw=db["per_dim"], n_strata=db["n_strata"],
                   n_strata_dim=db["n_strata_dim"], prompt=d_prompt_contrast(df, H, groups))
        for k in ks:
            U = topic_subspace(H[neutral], k, seed)
            dh, frac = project_out(db["pooled"], U)
            rec[f"hat_k{k}"] = dh
            rec[f"topic_frac_k{k}"] = frac
            rec[f"topic_only_k{k}"] = U @ (U.T @ db["pooled"])
            rec[f"per_dim_hat_k{k}"] = {kk: project_out(v, U)[0] for kk, v in db["per_dim"].items()}
            if rec["prompt"] is not None:
                rec[f"prompt_hat_k{k}"] = project_out(rec["prompt"], U)[0]
        rec["random"] = random_direction(db["pooled"], seed + l)
        rec["scale"] = {name: psi_scale(H[np.isfinite(H).all(1)], v) for name, v in rec.items()
                        if isinstance(v, np.ndarray) and v.ndim == 1}
        out[l] = rec
    return out


def save_directions(dirs: dict, path) -> None:
    """Flatten to an .npz (vectors) + .json (metadata)."""
    import json
    arrays, meta = {}, {}
    for l, rec in dirs.items():
        for name, v in rec.items():
            if isinstance(v, np.ndarray):
                arrays[f"L{l}/{name}"] = v
            elif isinstance(v, dict) and v and all(isinstance(x, np.ndarray) for x in v.values()):
                for kk, vv in v.items():
                    arrays[f"L{l}/{name}/{kk}"] = vv
            else:
                meta[f"L{l}/{name}"] = v
    np.savez(str(path) + ".npz", **arrays)
    with open(str(path) + ".json", "w") as f:
        json.dump(meta, f, indent=1, default=float)
        f.write("\n")
    with open(str(path) + ".README.txt", "w") as f:
        f.write("IDEA steering directions. Subtracting a direction (psi > 0) is intended to remove "
                "identity-conditioned evidence discounting; adding it (psi < 0) can AMPLIFY the bias. "
                "Every released direction is reversible in this way; see the paper's amplification result.\n")


def load_directions(path) -> dict:
    """Inverse of save_directions: {layer: rec}, with the same keys and the exact saved values,
    so a resumed steering run uses the very vectors (and psi scales) its cached results used."""
    import json
    out: dict = {}
    with open(str(path) + ".json") as f:
        meta = json.load(f)
    for key, v in meta.items():
        l, name = key.split("/", 1)
        out.setdefault(int(l[1:]), {})[name] = v
    with np.load(str(path) + ".npz") as z:
        for key in z.files:
            parts = key.split("/")
            rec = out.setdefault(int(parts[0][1:]), {})
            if len(parts) == 2:
                rec[parts[1]] = z[key]
            else:
                rec.setdefault(parts[1], {})[parts[2]] = z[key]
    return out


def signature_directions(df: pd.DataFrame, H: np.ndarray, signatures: pd.DataFrame, train_topics: set) -> dict:
    """RQ3b: one direction per dominant signature, built from that signature's defining
    behavior on TRAINING topics only."""
    d = df[df.topic.isin(train_topics)]
    out = {}
    for sig, rows in signatures.groupby("signature"):
        groups = set(rows.group)
        if sig in ("contradiction_discounting", "discounting(other)", "discounting+confirmation"):
            r = d_bias_matched(d, H, groups=groups)
            out[sig] = dict(vec=r["pooled"], groups=groups, kind="discounting")
        elif sig == "general_hedging":
            out[sig] = dict(vec=d_prompt_contrast(d, H, groups, evidence=("contra", "support")),
                            groups=groups, kind="hedging")
        elif sig == "confirmation_boost":
            # steepened supported slopes: supported minus neutral on matched items
            out[sig] = dict(vec=d_prompt_contrast(d, H, groups, evidence=("support",)), groups=groups,
                            kind="confirmation")
    return {k: v for k, v in out.items() if v["vec"] is not None}


def specificity_matrix(results: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """results[(direction_signature, evaluated_on_signature)] = success report."""
    rows = [dict(direction=a, applied_to=b, success=r.get("success"),
                 **{k: v for k, v in r.items() if k.endswith("gap_after") or k.endswith("gap_before")})
            for (a, b), r in results.items()]
    return pd.DataFrame(rows)
