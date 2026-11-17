"""Corrected RQ3 at the answer position (exploratory except where it repeats the preregistered test).

Why. The preregistered RQ3 built its directions and probes from activations at the end of the plain chat
prompt, but the answers it explained were mostly produced after the answer prefix "Probability (0-100): "
(99% of OLMo Instruct's posteriors, 54% of Llama's), so activations and answers came from different
positions. The corrected analysis works at the answer position and adds interventions from the literature:

  * Gupta et al. (2026): caved-minus-resisted difference of means at the answer position, steered at one
    layer or at all layers at once.
  * Park, Choe & Veitch (2024): concept directions from counterfactual inputs that differ ONLY in the concept.
    Every training trial is rendered three ways - user believes H1 / believes H0 / states no belief - and the
    activation differences give a clean "stated belief" signal. Geometry is also measured after whitening by
    the no-belief activation covariance.
  * Engels et al. (2025): a concept can occupy a low-dimensional subspace; intervene by SETTING the subspace
    coordinates to a target value (the no-belief value: erase the stated belief), test sufficiency by patching
    the subspace between counterfactual inputs, and intervene where the concept enters (the tokens of the
    belief sentence) in early layers.

This module holds the building blocks; analysis/rq3.py runs them in order:
  1. capture       answer-position activations of the training-topic trials, and counterfactual belief
                   triples (answer position + mean over the belief-sentence tokens)       [forward_states]
  2. directions    caved-resisted (cr_hat_k8), belief subspaces (answer position k=2,4; belief tokens k=1,2),
                   random controls                                                       [build_subspaces]
                   geometry: line vs triangle, whitened                                  [geometry_table]
  3. sufficiency   interchange patching on held-out counterfactual trials                [sufficiency]
  4. sweep         interventions on validation topics                                    [intervention_grid]
  5. test          best setting per family + matched random control on held-out test topics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..elicit import to_p

CAPTURE_LAYERS = [4, 8, 12, 16, 20, 24]   # layers captured and intervened on
N_CF_TRAIN = 1500                          # counterfactual triples from training topics (Tier 2 + Tier 1)
N_CF_HELDOUT = 300                         # counterfactual triples from validation topics (Tier 2), sufficiency
P4_LAYERS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

# Intervention families swept on validation topics (one entry per family; values are the settings tried).
GRID = dict(
    cr_single_layers=(12, 16, 20), cr_single_mult=(2, 4, 8, 16),   # caved-resisted vector at one layer, all tokens
    cr_all_mult=(0.5, 1, 2, 4),                                      # at every captured layer >= 8, last token
    erase_answer_layers=(8, 12, 16, 20, 24), erase_answer_k=(2, 4),  # erase the answer-position subspace
    erase_span_layers=(4, 8, 12), erase_span_k=(1, 2),               # erase at the belief-sentence tokens
    erase_span_multi=(4, 8, 12, 16),                                 # ... at several early layers at once
)


# ---------------------------------------------------------------------------------------------
# low-level forward passes (no cache): capture states, or patch per-row subspace coordinates
# ---------------------------------------------------------------------------------------------
def _batches(n, bs):
    for s in range(0, n, bs):
        yield slice(s, min(n, s + bs))


def forward_states(backend, reqs, layers, prefill=True, patch=None, bs=16):
    """Run `reqs` (chat number requests) rendered with the answer prefix (prefill=True) or without it.
    Returns p [n] (answer readout P(H1)), last {layer: [n, d]} (state at the last position = the answer
    position), span {layer: [n, d]} (mean over the belief-sentence tokens; NaN where there is no belief
    sentence). patch: optional dict(layer=l, U=[k,d], targets=[n,k], pos='last'|'span', full=None|[n,d]) that
    SETS per-row subspace coordinates (or, with `full`, replaces the whole last-position vector)."""
    torch = backend.torch
    texts = [backend.render(r, prefill=prefill) for r in reqs]
    spans = [r.meta.get("span") for r in reqs]
    n = len(reqs)
    P = np.full(n, np.nan)
    last = {l: None for l in layers}
    span = {l: None for l in layers}
    L = backend.layers()
    dev = backend.model.get_input_embeddings().weight.device
    for sl in _batches(n, bs):
        tx, sp = texts[sl], spans[sl]
        enc = backend.tok(tx, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True,
                          max_length=backend.max_length, return_offsets_mapping=True)
        mask = backend._span_mask(tx, sp, enc.pop("offset_mapping").numpy())
        enc = {k: v.to(dev) for k, v in enc.items()}
        mask_t = torch.as_tensor(mask, device=dev, dtype=torch.float32)
        got = {}
        handles = []
        for l in layers:
            def mk(l):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    hf = h.float()
                    m = mask_t[:, :, None]
                    cnt = m.sum(1).clamp(min=1)
                    got[l] = (hf[:, -1, :].cpu().numpy(), ((hf * m).sum(1) / cnt).cpu().numpy(),
                              (mask_t.sum(1) > 0).cpu().numpy())
                return hook
            handles.append(L[l].register_forward_hook(mk(l)))
        if patch is not None:
            def mk_p(pt, sl):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    if pt.get("full") is not None:
                        h = h.clone()
                        h[:, -1, :] = torch.as_tensor(pt["full"][sl], device=h.device, dtype=h.dtype)
                    else:
                        Ut = torch.as_tensor(pt["U"], device=h.device, dtype=torch.float32)
                        tg = torch.as_tensor(pt["targets"][sl], device=h.device, dtype=torch.float32)  # [b,k]
                        hf = h.float()
                        if pt["pos"] == "last":
                            sel = hf[:, -1, :]
                            hf[:, -1, :] = sel + (tg - sel @ Ut.T) @ Ut
                        else:
                            m = mask_t[:, :, None]
                            hf = hf + ((tg[:, None, :] - hf @ Ut.T) @ Ut) * m
                        h = hf.to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return hook
            # registered after the capture hooks, so captured states at that layer are pre-patch
            handles.append(L[patch["layer"]].register_forward_hook(mk_p(patch, sl)))
        try:
            with torch.no_grad():
                logits = backend.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        finally:
            for h in handles:
                h.remove()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for j, i in enumerate(range(sl.start, sl.stop)):
            mass, cov = backend._dist(probs[j], backend.num_ids)
            P[i] = float(((mass / cov) * to_p(np.arange(101))).sum()) if cov > 0 else np.nan
        for l in layers:
            a, s, has = got[l]
            if last[l] is None:
                d = a.shape[1]
                last[l] = np.zeros((n, d), np.float32)
                span[l] = np.full((n, d), np.nan, np.float32)
            last[l][sl] = a
            s = s.copy()
            s[~has] = np.nan
            span[l][sl] = s
    return P, last, span


def _logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def _orth(M):
    """Orthonormal rows spanning the rows of M."""
    q, _ = np.linalg.qr(np.asarray(M, np.float64).T)
    return q.T.astype(np.float32)


def _whiten_cos(u, v, S):
    Si = np.linalg.pinv(S)
    return float(u @ Si @ v / np.sqrt((u @ Si @ u) * (v @ Si @ v)))


def _shrunk_cov(X, lam=0.1):
    X = X - X.mean(0)
    S = X.T @ X / max(1, len(X) - 1)
    return (1 - lam) * S + lam * np.trace(S) / S.shape[0] * np.eye(S.shape[0])


# ---------------------------------------------------------------------------------------------
# 1-2. counterfactual trials, subspaces, geometry
# ---------------------------------------------------------------------------------------------
def select_cf_trials(d: pd.DataFrame, train_t, val_t, rng, n_train=N_CF_TRAIN, n_held=N_CF_HELDOUT):
    """Counterfactual trials: belief-stated, evidence-present trials of the training topics (both tiers), and
    of the validation topics (Tier 2 only) for the held-out sufficiency test. `rng` is consumed in this order
    (cf, then held), exactly as in the study's run."""
    dtrain = d[d.topic.isin(train_t)].reset_index(drop=True)
    pool = dtrain[(dtrain.b != 0) & (dtrain.e != "null")]
    cf = pool.iloc[rng.permutation(len(pool))[:n_train]].reset_index(drop=True)
    heldpool = d[d.topic.isin(val_t) & (d.b != 0) & (d.e != "null") & (d.tier == 2)]
    held = heldpool.iloc[rng.permutation(len(heldpool))[:n_held]].reset_index(drop=True)
    return dtrain, cf, held


def capture_counterfactuals(backend, req, dtrain, cf, held, layers, log=print) -> dict:
    """capture.npz contents: train_last_L<l> (answer position, every training-topic trial as asked) and, for
    tag in (cf, held) x name in (pos = believes H1, neg = believes H0, none = no belief):
    <tag>_<name>_p, <tag>_<name>_last_L<l>, <tag>_<name>_span_L<l> (float16)."""
    log(f"[capture] {len(dtrain)} training trials at the answer position")
    _, A, _ = forward_states(backend, [req(r) for r in dtrain.itertuples(index=False)], layers)
    res = {f"train_last_L{l}": A[l].astype(np.float16) for l in layers}
    for tag, frame in (("cf", cf), ("held", held)):
        rws = list(frame.itertuples(index=False))
        for name in ("pos", "neg", "none"):
            # believes H1: belief_dir = h_dir; believes H0: belief_dir = -h_dir; none: 0
            reqs = [req(r, (r.h_dir if name == "pos" else -r.h_dir) if name != "none" else 0) for r in rws]
            log(f"[capture] {tag} {name}: {len(reqs)}")
            p, la, sp = forward_states(backend, reqs, layers)
            res[f"{tag}_{name}_p"] = p
            for l in layers:
                res[f"{tag}_{name}_last_L{l}"] = la[l].astype(np.float16)
                res[f"{tag}_{name}_span_L{l}"] = sp[l].astype(np.float16)
    return res


def build_subspaces(C: dict, dtrain: pd.DataFrame, layers, rng, seed: int) -> dict:
    """directions_v2.npz contents, keys L<l>/<name> (see steering_vectors/README.md)."""
    from . import steer
    f32 = lambda k: C[k].astype(np.float32)  # noqa: E731
    D = {}
    cr = steer.build_directions(dtrain, lambda l: f32(f"train_last_L{l}"), layers, ks=(8,), seed=seed)
    for l in layers:
        pos, neg, non = f32(f"cf_pos_last_L{l}"), f32(f"cf_neg_last_L{l}"), f32(f"cf_none_last_L{l}")
        dp, dn = (pos - non).mean(0), (neg - non).mean(0)
        diffs = np.concatenate([pos - non, neg - non])
        _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
        U2, U4 = _orth(np.stack([dp, dn])), Vt[:4].astype(np.float32)
        D[f"L{l}/ans_U2"], D[f"L{l}/ans_U4"] = U2, U4
        D[f"L{l}/ans_t2"], D[f"L{l}/ans_t4"] = U2 @ non.mean(0), U4 @ non.mean(0)
        sp, sn = f32(f"cf_pos_span_L{l}"), f32(f"cf_neg_span_L{l}")
        sdiff = sp - sn
        _, _, Vs = np.linalg.svd(sdiff, full_matrices=False)
        ds = sdiff.mean(0)
        Us1, Us2 = (ds / np.linalg.norm(ds))[None].astype(np.float32), Vs[:2].astype(np.float32)
        mid = ((sp + sn) / 2).mean(0)
        D[f"L{l}/span_U1"], D[f"L{l}/span_U2"] = Us1, Us2
        D[f"L{l}/span_t1"], D[f"L{l}/span_t2"] = Us1 @ mid, Us2 @ mid
        R = rng.normal(size=(4, U2.shape[1]))
        D[f"L{l}/rand_U2"], D[f"L{l}/rand_U4"] = _orth(R[:2]), _orth(R)
        D[f"L{l}/rand_t2"], D[f"L{l}/rand_t4"] = _orth(R[:2]) @ non.mean(0), _orth(R) @ non.mean(0)
        if l in cr:
            D[f"L{l}/cr_hat_k8"] = cr[l]["hat_k8"].astype(np.float32)
            D[f"L{l}/cr_scale"] = np.float32(cr[l]["scale"]["hat_k8"])
    return D


def geometry_table(C: dict, D: dict, layers) -> pd.DataFrame:
    """Per layer: norms and cosine of the two mean belief displacements (believes H1 - none, believes H0 - none),
    raw and whitened; sv_ratio (0 = the three means on one line); variance share inside U2/U4; overlap of the
    caved-resisted vector with the belief plane; mean answers per condition."""
    f32 = lambda k: C[k].astype(np.float32)  # noqa: E731
    geo = []
    for l in layers:
        pos, neg, non = f32(f"cf_pos_last_L{l}"), f32(f"cf_neg_last_L{l}"), f32(f"cf_none_last_L{l}")
        dp, dn = (pos - non).mean(0), (neg - non).mean(0)
        diffs = np.concatenate([pos - non, neg - non])
        U2, U4 = D[f"L{l}/ans_U2"], D[f"L{l}/ans_U4"]
        ev = np.linalg.svd(np.stack([dp, dn]), compute_uv=False)
        cr = D.get(f"L{l}/cr_hat_k8")
        geo.append(dict(layer=l, norm_pos=float(np.linalg.norm(dp)), norm_neg=float(np.linalg.norm(dn)),
                        cos_pos_neg=float(dp @ dn / np.linalg.norm(dp) / np.linalg.norm(dn)),
                        cos_pos_neg_whitened=_whiten_cos(dp, dn, _shrunk_cov(non)),
                        sv_ratio=float(ev[1] / ev[0]),
                        share_var_U2=float(np.linalg.norm(diffs @ U2.T) ** 2 / np.linalg.norm(diffs) ** 2),
                        share_var_U4=float(np.linalg.norm(diffs @ U4.T) ** 2 / np.linalg.norm(diffs) ** 2),
                        cos_cr_vs_belief=(float(np.abs(cr @ U2.T).max() / np.linalg.norm(cr)) if cr is not None
                                          else np.nan),
                        p_pos=float(np.nanmean(C["cf_pos_p"])), p_neg=float(np.nanmean(C["cf_neg_p"])),
                        p_none=float(np.nanmean(C["cf_none_p"]))))
    return pd.DataFrame(geo)


# ---------------------------------------------------------------------------------------------
# 3. sufficiency: interchange patching on held-out counterfactual pairs
# ---------------------------------------------------------------------------------------------
def sufficiency(backend, req, held: pd.DataFrame, C: dict, D: dict, layers, log=print) -> pd.DataFrame:
    """Run the believes-<src> input, set its subspace coordinates to those of the believes-<dst> input, and
    report recovery = mean moved / mean gap in the answer's log-odds (1 = the subspace carries the whole
    belief effect). Compared with patching the whole last-position vector and with a random subspace."""
    f32 = lambda k: C[k].astype(np.float32)  # noqa: E731
    hr = list(held.itertuples(index=False))
    rows = []
    for src, dst in (("pos", "neg"), ("neg", "pos")):
        reqs = [req(r, r.h_dir if src == "pos" else -r.h_dir) for r in hr]
        p_src, p_dst = C[f"held_{src}_p"], C[f"held_{dst}_p"]
        gap = _logit(p_dst) - _logit(p_src)
        ok = np.isfinite(gap) & (np.abs(gap) > 0.05)
        for l in layers:
            for name, U, key in (("answer_U2", D[f"L{l}/ans_U2"], "last"), ("answer_U4", D[f"L{l}/ans_U4"], "last"),
                                 ("random_U2", D[f"L{l}/rand_U2"], "last"), ("span_U1", D[f"L{l}/span_U1"], "span"),
                                 ("span_U2", D[f"L{l}/span_U2"], "span"), ("whole_last", None, "full")):
                if key == "full":
                    pt = dict(layer=l, full=f32(f"held_{dst}_last_L{l}"), pos="last")
                elif key == "last":
                    pt = dict(layer=l, U=U, targets=f32(f"held_{dst}_last_L{l}") @ U.T, pos="last")
                else:
                    tg = f32(f"held_{dst}_span_L{l}") @ U.T
                    pt = dict(layer=l, U=U, targets=np.nan_to_num(tg), pos="span")
                p, _, _ = forward_states(backend, reqs, [l], patch=pt)
                moved = _logit(p) - _logit(p_src)
                rec = float(np.nanmean(moved[ok]) / np.nanmean(gap[ok]))
                rows.append(dict(direction=f"{src}->{dst}", layer=l, patch=name, n=int(ok.sum()),
                                 mean_gap=float(np.nanmean(gap[ok])), mean_moved=float(np.nanmean(moved[ok])),
                                 recovery=rec))
                log(f"[sufficiency] {src}->{dst} L{l} {name}: recovery {rec:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 4-5. interventions
# ---------------------------------------------------------------------------------------------
def intervention_grid(D: dict, layers, grid: dict = GRID) -> list:
    """(family, setting, steering entries) for every intervention swept on validation topics. Tuple entries are
    (layer, vector, psi, mode) additive steering; dict entries set subspace coordinates (erasure)."""
    g = {**GRID, **(grid or {})}
    configs = []
    for l in [x for x in g["cr_single_layers"] if f"L{x}/cr_hat_k8" in D]:
        for m in g["cr_single_mult"]:
            configs.append(("cr_single", f"L{l}_x{m}", entries_for(D, "cr_single", f"L{l}_x{m}", layers)))
    for m in g["cr_all_mult"]:
        configs.append(("cr_all_layers", f"x{m}", entries_for(D, "cr_all_layers", f"x{m}", layers)))
    for l in [x for x in g["erase_answer_layers"] if x in layers]:
        for k in g["erase_answer_k"]:
            configs.append(("erase_answer", f"L{l}_k{k}", entries_for(D, "erase_answer", f"L{l}_k{k}", layers)))
    for l in [x for x in g["erase_span_layers"] if x in layers]:
        for k in g["erase_span_k"]:
            configs.append(("erase_span", f"L{l}_k{k}", entries_for(D, "erase_span", f"L{l}_k{k}", layers)))
    multi = [x for x in g["erase_span_multi"] if x in layers]
    if multi:
        configs.append(("erase_span_multi", f"L{multi[0]}-{multi[-1]}_k1",
                        [dict(kind="erase", layer=x, U=D[f"L{x}/span_U1"], target=D[f"L{x}/span_t1"], pos="span")
                         for x in multi]))
    # (a family with no vectors, e.g. cr_all_layers when no stratum had both caved and resisted trials, is kept
    # with empty entries = unsteered, as in the study's run)
    return configs


def choose_per_family(sw: pd.DataFrame) -> set:
    """Per family: among settings whose no-belief slope stays within 0.05 on validation topics (if any), the one
    with the smallest remaining contradicted-evidence gap."""
    best = []
    for fam, g in sw.groupby("family"):
        ok = g[g.t2_neutral_change.abs() < 0.05] if (g.t2_neutral_change.abs() < 0.05).any() else g
        best.append(ok.loc[ok.t2_contra_gap_after.abs().idxmin()])
    return {(b.family, b.setting) for b in best}


def entries_for(D, family, setting, layers):
    """Rebuild a setting's steering entries from the saved subspaces/vectors."""
    if family == "cr_single":
        l, m = setting.split("_x")
        l = int(l[1:])
        return [(l, D[f"L{l}/cr_hat_k8"], float(float(m) * D[f"L{l}/cr_scale"]), "all")]
    if family == "cr_all_layers":
        m = float(setting[1:])
        return [(x, D[f"L{x}/cr_hat_k8"], float(m * D[f"L{x}/cr_scale"]), "last")
                for x in layers if x >= 8 and f"L{x}/cr_hat_k8" in D]
    if family in ("erase_answer", "control_random_subspace"):
        l, k = setting.split("_k")
        l = int(l[1:])
        src = "ans" if family == "erase_answer" else "rand"
        return [dict(kind="erase", layer=l, U=D[f"L{l}/{src}_U{k}"], target=D[f"L{l}/{src}_t{k}"], pos="last")]
    if family == "erase_span":
        l, k = setting.split("_k")
        l = int(l[1:])
        return [dict(kind="erase", layer=l, U=D[f"L{l}/span_U{k}"], target=D[f"L{l}/span_t{k}"], pos="span")]
    if family == "erase_span_multi":
        return [dict(kind="erase", layer=x, U=D[f"L{x}/span_U1"], target=D[f"L{x}/span_t1"], pos="span")
                for x in (4, 8, 12, 16) if f"L{x}/span_U1" in D]
    raise ValueError(f"unknown family {family}")


def control_for(D, family, setting, entries, seed=0):
    """Matched random control: random subspace of the same size for erasure; for steering vectors, random
    directions of equal norm at the same layers and strengths."""
    if family == "erase_answer":
        return entries_for(D, "control_random_subspace", setting, [])
    rng = np.random.default_rng(seed + 31)
    out = []
    for e in entries:
        if isinstance(e, dict):
            out.append(e)   # span erasure: no control needed (no effect on benchmarks by construction)
        else:
            l, d, psi, mode = e
            out.append((l, rng.normal(size=np.asarray(d).shape).astype(np.float32), psi, mode))
    return out


def partial_success(t: pd.DataFrame, bar: float) -> pd.Series:
    """Tier 3 'partial success' (defined before any corrected-RQ3 test result existed): the pooled
    contradicted-evidence gap closes by at least `bar` (25%, and 10 points more than the best random control),
    the contradicted slope moves toward neutral, and the no-belief slopes in Tier 2 and Tier 1 change by less
    than 0.05. No per-dimension requirement."""
    def col(c):
        return t[c] if c in t else pd.Series(np.nan, index=t.index)
    return ((col("gap_closed_frac") >= bar) & col("t2_contra_moved_toward_neutral").fillna(False).astype(bool)
            & (col("t2_neutral_change").abs() < 0.05) & (col("tier1_neutral_change").abs() < 0.05))


def audit_bar(t: pd.DataFrame) -> float:
    """At least 25% of the gap closed, and 10 points more than the best random control."""
    ctrl = t[t.family == "control_random_subspace"]
    return max(0.25, (float(ctrl.gap_closed_frac.max()) if len(ctrl) else 0.0) + 0.10)


def pick_winner(out: Path):
    """Automatic choice (used in the study for Tulu 3): best tested intervention that passes the preregistered
    criterion, else the exploratory one, else the partial one (largest share of the gap closed, above the audit
    bar). Returns (winner dict, '') or (None, reason)."""
    f = Path(out) / "steering_test_v2.csv"
    if not f.exists():
        return None, "corrected RQ3 has not produced steering_test_v2.csv"
    t = pd.read_csv(f)
    bar = audit_bar(t)
    t = t[t.family != "control_random_subspace"]
    t = t.assign(success_partial=partial_success(t, bar))
    for col in ("success_preregistered", "success_exploratory", "success_partial"):
        ok = t[t[col].astype(bool) & (t.gap_closed_frac >= bar)]
        if len(ok):
            w = ok.loc[ok.gap_closed_frac.idxmax()]
            return dict(family=w.family, setting=w.setting, criterion=col, gap_closed=float(w.gap_closed_frac),
                        audit_bar=bar), ""
    return None, "no corrected intervention passed a criterion and closed enough of the gap"
