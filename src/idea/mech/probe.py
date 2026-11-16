"""Masking probe (Section 5.3). A ridge probe trained on belief-neutral trials (b=0) to read
the Bayesian posterior logit (l + x) from the last-token residual stream, split by topic,
then applied to contradicted trials with b != 0.

masking index = median((read - stated) / (bayes - stated)) over trials where
|bayes - stated| is large: ~1 means the internal readout sits at the Bayesian answer while
the stated answer does not (response-selection masking); ~0 means the hidden state is as
distorted as the output."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


def split_topics(topics, frac_test: float = 0.3, frac_val: float = 0.0, seed: int = 0):
    t = sorted(set(topics))
    rng = np.random.default_rng(seed)
    rng.shuffle(t)
    n_te = max(1, int(round(len(t) * frac_test)))
    n_va = int(round(len(t) * frac_val))
    return set(t[n_te + n_va:]), set(t[n_te:n_te + n_va]), set(t[:n_te])  # train, val, test


def train_probe(H: np.ndarray, df: pd.DataFrame, train_topics: set, test_topics: set):
    target = (df.lp_before + df.x).values
    neutral = (df.b == 0).values & np.isfinite(H).all(1)
    tr = neutral & df.topic.isin(train_topics).values
    te = neutral & df.topic.isin(test_topics).values
    probe = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(H[tr], target[tr])
    r2 = probe.score(H[te], target[te]) if te.sum() > 2 else np.nan
    return probe, float(r2)


def masking_index(probe, H: np.ndarray, df: pd.DataFrame, min_gap: float = 0.5, by=None) -> pd.DataFrame:
    contra = ((df.b != 0) & (df.b * np.sign(df.x) < 0)).values & np.isfinite(H).all(1)
    d = df[contra].copy()
    d["read"] = probe.predict(H[contra])
    d["stated"] = d.lp_after
    d["bayes"] = d.lp_before + d.x
    d = d[(d.bayes - d.stated).abs() >= min_gap]
    d["mi"] = (d.read - d.stated) / (d.bayes - d.stated)
    if by is None:
        return pd.DataFrame([dict(n=len(d), masking_index=float(d.mi.median()),
                                  iqr_lo=float(d.mi.quantile(0.25)), iqr_hi=float(d.mi.quantile(0.75)))])
    return d.groupby(by).mi.agg(n="size", masking_index="median",
                                iqr_lo=lambda v: v.quantile(0.25), iqr_hi=lambda v: v.quantile(0.75)).reset_index()


def layer_sweep(get_H, df: pd.DataFrame, layers, train_topics, val_topics) -> pd.DataFrame:
    """Choose the probe layer by held-out R^2 on validation topics (never test topics)."""
    rows = []
    for l in layers:
        H = get_H(l)
        _, r2 = train_probe(H, df, train_topics, val_topics)
        rows.append(dict(layer=l, r2_val=r2))
    return pd.DataFrame(rows)


def run_masking(get_H, df: pd.DataFrame, layers, seed: int = 0, signatures: pd.DataFrame | None = None,
                r2_min: float = 0.5) -> dict:
    train_t, val_t, test_t = split_topics(df.topic.unique(), 0.25, 0.2, seed)
    sweep = layer_sweep(get_H, df, layers, train_t, val_t)
    best = int(sweep.loc[sweep.r2_val.idxmax(), "layer"])
    H = get_H(best)
    probe, r2 = train_probe(H, df, train_t | val_t, test_t)
    test_mask = df.topic.isin(test_t).values
    dft, Ht = df[test_mask].reset_index(drop=True), H[test_mask]
    out = dict(sweep=sweep, layer=best, r2_test=r2, interpretable=bool(r2 >= r2_min),
               overall=masking_index(probe, Ht, dft),
               by_dimension=masking_index(probe, Ht, dft, by="dimension"),
               by_cue_type=masking_index(probe, Ht, dft, by="cue_type"))
    if signatures is not None:
        sig = signatures[["group", "cue_type", "signature"]]
        out["by_signature"] = masking_index(probe, Ht, dft.merge(sig, on=["group", "cue_type"], how="left"),
                                            by="signature")
    return out
