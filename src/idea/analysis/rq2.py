"""RQ2: predicting slope shifts on held-out topics and a held-out identity dimension
(Section 5.1, Eq. 7).

Targets are per-cell slopes estimated WITHOUT pooling (one simple regression per cell, for
training and test cells alike) to avoid leakage through the targets. Each cell slope is
the through-origin slope of (lp_after - alpha - rho * lp_before) on x, with alpha and rho
taken from the training fold's belief-neutral trials only.

Cells (deviation approved 2026-11-10): identity group x cue type x source x belief x evidence
direction, pooling topic and H1 wording. With topic in the cell (the notes' definition) the
main run would have a median of 1 trial per cell. The held-out-topic split uses 4 folds of
topics: the same identity-by-condition cells are estimated on training topics and on unseen
topics; g and v_k enter as within-cell averages.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..cues import DIMENSIONS

CELL_KEYS = ["group", "dimension", "cue_type", "source", "b", "d"]
N_TOPIC_FOLDS = 4


def neutral_reference(d: pd.DataFrame) -> tuple[float, float, float]:
    import statsmodels.formula.api as smf
    n = d[d.b == 0]
    f = smf.ols("lp_after ~ lp_before + x", n).fit()
    return float(f.params["Intercept"]), float(f.params["lp_before"]), float(f.params["x"])


def cell_slopes(d: pd.DataFrame, alpha: float, rho: float, keys=CELL_KEYS, min_n: int = 3,
                min_sxx: float = 1.0) -> pd.DataFrame:
    d = d[(d.b != 0) & (d.x != 0)].copy()
    d["r"] = d.lp_after - alpha - rho * d.lp_before
    d["xr"], d["xx"] = d.x * d.r, d.x ** 2
    agg = d.groupby(keys, observed=True).agg(n=("x", "size"), sxy=("xr", "sum"), sxx=("xx", "sum"),
                                              g=("g", "mean"), v=("v_k", "mean"), has_v=("has_v", "mean"),
                                              conflict=("heuristic_conflict", "mean"), tier=("tier", "first"))
    agg = agg[(agg.n >= min_n) & (agg.sxx >= min_sxx)].reset_index()
    if agg.empty:
        return agg
    agg["beta"] = agg.sxy / agg.sxx
    # residual SD per cell is noisy at small n; use the residual SD pooled over the kept cells
    kept = d.merge(agg[keys + ["beta"]], on=keys, how="inner")
    sig = float(np.sqrt(((kept.r - kept.x * kept.beta) ** 2).mean()))
    agg["se"] = sig / np.sqrt(agg.sxx)
    return agg


class Encoder:
    def __init__(self, cells: pd.DataFrame, use_source: bool = True):
        self.groups = sorted(cells.group.unique())
        self.sbd = sorted(set(zip(cells.source, cells.b, cells.d))) if use_source else sorted(set(zip(cells.b, cells.d)))
        self.bd = sorted(set(zip(cells.b, cells.d)))
        self.use_source = use_source

    def encode(self, cells: pd.DataFrame, known_groups: set[str]):
        import torch
        phi = np.column_stack([(cells.cue_type == "vignette").astype(float), (cells.source == "tank").astype(float),
                               cells.conflict.astype(float), (cells.tier == 1).astype(float)])
        gi = {g: i for i, g in enumerate(self.groups)}
        si = {k: i for i, k in enumerate(self.sbd)}
        bi = {k: i for i, k in enumerate(self.bd)}
        ident = cells.group.map(lambda g: gi.get(g, 0)).values
        sbd_keys = list(zip(cells.source, cells.b, cells.d)) if self.use_source else list(zip(cells.b, cells.d))
        sbd = np.array([si[k] for k in sbd_keys])
        bd = np.array([bi[k] for k in zip(cells.b, cells.d)])
        known = cells.group.isin(known_groups).astype(float).values
        T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt)
        return (T(phi), T(cells.g.fillna(0).values), T(cells.v.fillna(0).values), T(cells.has_v.fillna(0).values),
                T(ident, torch.long), T(sbd, torch.long), T(bd, torch.long), T(known))


def make_model(n_phi, n_ident, n_sbd, n_bd, rank=2):
    import torch
    import torch.nn as nn

    class LowRankBeta(nn.Module):
        def __init__(self):
            super().__init__()
            self.b0 = nn.Parameter(torch.zeros(1))
            self.w = nn.Linear(n_phi, 1, bias=False)
            self.lam = nn.Embedding(n_bd, 1)
            self.eta = nn.Embedding(n_bd, 1)
            self.bdi = nn.Embedding(n_bd, 1)      # per (b, d) intercept (the mean baseline's structure)
            self.rank = rank
            if rank > 0:
                self.A = nn.Embedding(n_ident, rank)
                self.B = nn.Embedding(n_sbd, rank)
            for e in (self.lam, self.eta, self.bdi):
                nn.init.zeros_(e.weight)
            if rank > 0:
                nn.init.normal_(self.A.weight, std=0.1); nn.init.normal_(self.B.weight, std=0.1)

        def forward(self, phi, g, v, has_v, ident, sbd, bd, known):
            out = self.b0 + self.w(phi).squeeze(-1) + self.bdi(bd).squeeze(-1)
            out = out + g * self.lam(bd).squeeze(-1) + has_v * v * self.eta(bd).squeeze(-1)
            if self.rank > 0:
                out = out + known * (self.A(ident) * self.B(sbd)).sum(-1)
            return out

    return LowRankBeta()


def fit_model(model, inputs, beta, se, ridge, steps=1500, lr=2e-2):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = torch.tensor(beta, dtype=torch.float32)
    w = torch.tensor(1 / np.asarray(se) ** 2, dtype=torch.float32)
    w = w / w.mean()
    for _ in range(steps):
        opt.zero_grad()
        loss = (w * (model(*inputs) - y) ** 2).mean()
        loss = loss + ridge * sum((p ** 2).sum() for n, p in model.named_parameters() if n not in ("b0", "bdi.weight"))
        loss.backward()
        opt.step()
    return model


def _predict(train: pd.DataFrame, test: pd.DataFrame, rank: int, ridge: float, known_groups: set, seed=0,
             use_source=True):
    import torch
    torch.manual_seed(seed)
    enc = Encoder(pd.concat([train, test]), use_source)
    m = make_model(4, len(enc.groups), len(enc.sbd), len(enc.bd), rank)
    fit_model(m, enc.encode(train, known_groups), train.beta.values, train.se.values, ridge)
    with torch.no_grad():
        return m(*enc.encode(test, known_groups)).numpy()


def _inner_ridge(trials: pd.DataFrame, rank: int, grid, split_col: str, alpha: float, rho: float,
                 k: int = 3, seed=0, use_source: bool = True):
    """Tune the ridge penalty by inner CV on the training trials only, splitting on the same
    unit as the outer split (topics or dimensions) and rebuilding cells inside each fold."""
    units = sorted(trials[split_col].unique())
    rng = random.Random(seed)
    rng.shuffle(units)
    folds = [set(units[i::k]) for i in range(k)]
    best, best_err = grid[0], np.inf
    for lam in grid:
        err, used = 0.0, 0
        for f in folds:
            tr = cell_slopes(trials[~trials[split_col].isin(f)], alpha, rho)
            va = cell_slopes(trials[trials[split_col].isin(f)], alpha, rho)
            if tr.empty or va.empty:
                continue
            known = set(tr.group) - (set(va.group) if split_col == "dimension" else set())
            pred = _predict(tr, va, rank, lam, known, seed, use_source)
            err += float(np.sum((pred - va.beta.values) ** 2 / va.se.values ** 2))
            used += 1
        if used and err < best_err:
            best, best_err = lam, err
    return best


def metrics(pred_shift, true_shift) -> dict:
    pred_shift, true_shift = np.asarray(pred_shift), np.asarray(true_shift)
    return dict(sign_acc=float(np.mean(np.sign(pred_shift) == np.sign(true_shift))),
                spearman=float(spearmanr(pred_shift, true_shift).statistic) if len(pred_shift) > 2 else np.nan,
                rmse=float(np.sqrt(np.mean((pred_shift - true_shift) ** 2))))


def topic_folds(topics, k: int = N_TOPIC_FOLDS, seed: int = 0) -> list[tuple[str, ...]]:
    t = sorted(topics)
    random.Random(seed).shuffle(t)
    return [tuple(t[i::k]) for i in range(k)]


def evaluate(df: pd.DataFrame, model: str, fmt: str = "chat", rank: int = 2,
             ridge_grid=(1e-4, 1e-3, 1e-2, 1e-1), seed: int = 0, use_source: bool = True,
             max_folds: int | None = None) -> dict:
    """Run both splits; returns per-fold and aggregate metrics for full, no-W, and mean
    baselines, plus the success flag (full beats both baselines on each split)."""
    d = df[(df.model == model) & (df.fmt == fmt) & ~df.excluded & (df.tier == 2)].copy()
    d["v_k"] = d.get("v_k", pd.Series(0.0, index=d.index)).fillna(0.0)
    d["has_v"] = d.get("has_v", pd.Series(0.0, index=d.index)).fillna(0.0)
    results = []
    splits = []
    for fold in topic_folds(d.topic.unique(), seed=seed):
        splits.append(("held_out_topics", "topic", set(fold)))
    for k in DIMENSIONS:
        splits.append(("held_out_dimension", "dimension", {k}))
    if max_folds:
        by = {}
        for s in splits:
            by.setdefault(s[0], []).append(s)
        splits = [s for v in by.values() for s in v[:max_folds]]
    for split, col, held in splits:
        tr_trials, te_trials = d[~d[col].isin(held)], d[d[col].isin(held)]
        if te_trials.empty:
            continue
        alpha, rho, beta_neu = neutral_reference(tr_trials)
        train = cell_slopes(tr_trials, alpha, rho)
        test = cell_slopes(te_trials, alpha, rho)
        if train.empty or test.empty:
            continue
        known = set(train.group)
        if split == "held_out_dimension":
            known -= set(test.group)
        true_shift = test.beta.values - beta_neu
        rec = dict(split=split, held=",".join(sorted(held)), n_train=len(train), n_test=len(test))
        for name, rank_ in (("full", rank), ("no_W", 0)):
            lam = _inner_ridge(tr_trials, rank_, ridge_grid, col, alpha, rho, seed=seed, use_source=use_source)
            pred = _predict(train, test, rank_, lam, known, seed, use_source)
            rec.update({f"{name}_{k}": v for k, v in metrics(pred - beta_neu, true_shift).items()})
            rec[f"{name}_ridge"] = lam
        means = train.assign(w=1 / train.se ** 2).groupby(["b", "d"]).apply(
            lambda g: np.average(g.beta, weights=g.w), include_groups=False)
        pred = test.apply(lambda r: means.get((r.b, r.d), train.beta.mean()), axis=1).values
        rec.update({f"mean_{k}": v for k, v in metrics(pred - beta_neu, true_shift).items()})
        results.append(rec)
    folds = pd.DataFrame(results)
    if folds.empty:
        return dict(folds=folds, summary=pd.DataFrame(), success={},
                    note="no fold had cells with enough trials to estimate unpooled slopes")
    summary = folds.groupby("split").mean(numeric_only=True)
    success = {}
    for split, row in summary.iterrows():
        success[split] = bool(row.full_rmse < row.no_W_rmse and row.full_rmse < row.mean_rmse
                              and row.full_spearman > max(row.no_W_spearman, row.mean_spearman))
    return dict(folds=folds, summary=summary.reset_index(), success=success)
