"""RQ2: can identity information predict how much a group's evidence slope shifts, for topics or
identity dimensions the predictor never saw?

    python analysis/rq2.py [--only a,b] [--skip a,b] [--models gpt,llama,...]
                           [--data datasets/main_trials.csv.gz] [--save DIR] [--list]

Preregistered for the main study, chat format, tier-2 (socially meaningful) topics, non-excluded
trials. Requires pandas, numpy, scipy, statsmodels and torch (the predictor is a small low-rank
model fitted with Adam; torch is what makes the numbers reproduce exactly).

Method
  * Cell = identity group x cue type x source x belief b x evidence direction d (topic and H1
    wording pooled). Each cell's evidence slope is estimated WITHOUT pooling: the through-origin
    slope of (lp_after - alpha - rho * lp_before) on x, with alpha, rho taken from the training
    fold's no-belief (b = 0) trials. Target = cell slope minus the training no-belief slope.
  * Predictors: an intercept per (b, d); cue type, source, heuristic conflict, tier; the model's
    perceived stance g and the survey gap v_k, each with a slope per (b, d); and, in the "full"
    model, a rank-2 interaction W = A[group] . B[source, b, d] (only for groups seen in training).
  * Baselines: "no_W" (same model without W) and "mean" (inverse-variance weighted training mean
    per (b, d)). Ridge penalties are tuned by 3-fold inner CV on the training trials.
  * Splits: 4 folds of held-out topics (seeded shuffle) and 7 folds each holding out one identity
    dimension. Metrics per fold (RMSE, Spearman, sign accuracy), averaged over folds.
  * Success (preregistered): on a split, the full model has lower RMSE than both baselines AND a
    higher Spearman correlation than both.
  * RQ2b (behavioral signatures, exploratory) is gated: it runs only for models whose RQ2
    succeeded on the held-out-topic split. No model passes the gate, so RQ2b is not run here.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# --------------------------------------------------------------------------------------
# Toggles
# --------------------------------------------------------------------------------------
ANALYSES = {                       # set False to skip
    "held_out_topics": True,       # 4 folds of held-out topics (Table rq2, "held out topics")
    "held_out_dimensions": True,   # 7 folds, one identity dimension held out (Table rq2, "held out dimension")
    "success_rule": True,          # preregistered rule: full beats no_W and mean on RMSE and Spearman
    "rq2b_gate": True,             # RQ2b runs only if RQ2 succeeds on held-out topics; reports the gate
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "datasets" / "main_trials.csv.gz"

STUDY, FMT, TIER = "main", "chat", 2          # preregistered sample for RQ2
DIMENSIONS = ["political", "race", "sexuality", "gender", "religion", "geography", "age"]
CELL_KEYS = ["group", "dimension", "cue_type", "source", "b", "d"]
N_TOPIC_FOLDS = 4
RANK = 2                                      # rank of W in the full model
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
SEED = 0
BOOL_COLS = ("excluded", "excluded_cov08", "tier1_low_g", "matched_pair", "calibration", "heuristic_conflict")


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the trials CSV as the codebook says: only empty cells are missing (the evidence
    label "null" must stay a string) and the boolean columns are cast back to bool."""
    df = pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])
    for c in BOOL_COLS:
        if c in df:
            df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df


# --------------------------------------------------------------------------------------
# Targets: unpooled per-cell slopes
# --------------------------------------------------------------------------------------
def neutral_reference(d: pd.DataFrame) -> tuple[float, float, float]:
    """alpha, rho and the evidence slope of the no-belief trials: lp_after ~ lp_before + x."""
    f = smf.ols("lp_after ~ lp_before + x", d[d.b == 0]).fit()
    return float(f.params["Intercept"]), float(f.params["lp_before"]), float(f.params["x"])


def cell_slopes(d: pd.DataFrame, alpha: float, rho: float, min_n: int = 3, min_sxx: float = 1.0) -> pd.DataFrame:
    """One through-origin slope per cell (b != 0, x != 0 trials), keeping cells with at least
    min_n trials and sum(x^2) >= min_sxx. SE uses the residual SD pooled over the kept cells
    (a per-cell SD is too noisy at small n)."""
    d = d[(d.b != 0) & (d.x != 0)].copy()
    d["r"] = d.lp_after - alpha - rho * d.lp_before
    d["xr"], d["xx"] = d.x * d.r, d.x ** 2
    agg = d.groupby(CELL_KEYS, observed=True).agg(
        n=("x", "size"), sxy=("xr", "sum"), sxx=("xx", "sum"), g=("g", "mean"), v=("v_k", "mean"),
        has_v=("has_v", "mean"), conflict=("heuristic_conflict", "mean"), tier=("tier", "first"))
    agg = agg[(agg.n >= min_n) & (agg.sxx >= min_sxx)].reset_index()
    if agg.empty:
        return agg
    agg["beta"] = agg.sxy / agg.sxx
    kept = d.merge(agg[CELL_KEYS + ["beta"]], on=CELL_KEYS, how="inner")
    sig = float(np.sqrt(((kept.r - kept.x * kept.beta) ** 2).mean()))
    agg["se"] = sig / np.sqrt(agg.sxx)
    return agg


# --------------------------------------------------------------------------------------
# Predictor: low-rank model of the cell slope
# --------------------------------------------------------------------------------------
class Encoder:
    """Index the groups, (source, b, d) and (b, d) combinations of a set of cells and turn
    cells into model inputs."""

    def __init__(self, cells: pd.DataFrame):
        self.groups = sorted(cells.group.unique())
        self.sbd = sorted(set(zip(cells.source, cells.b, cells.d)))
        self.bd = sorted(set(zip(cells.b, cells.d)))

    def encode(self, cells: pd.DataFrame, known_groups: set[str]):
        phi = np.column_stack([(cells.cue_type == "vignette").astype(float), (cells.source == "tank").astype(float),
                               cells.conflict.astype(float), (cells.tier == 1).astype(float)])
        gi = {g: i for i, g in enumerate(self.groups)}
        si = {k: i for i, k in enumerate(self.sbd)}
        bi = {k: i for i, k in enumerate(self.bd)}
        ident = cells.group.map(lambda g: gi.get(g, 0)).values
        sbd = np.array([si[k] for k in zip(cells.source, cells.b, cells.d)])
        bd = np.array([bi[k] for k in zip(cells.b, cells.d)])
        known = cells.group.isin(known_groups).astype(float).values   # W applies only to groups seen in training

        def t(a, dt=torch.float32):
            return torch.tensor(np.asarray(a), dtype=dt)
        return (t(phi), t(cells.g.fillna(0).values), t(cells.v.fillna(0).values), t(cells.has_v.fillna(0).values),
                t(ident, torch.long), t(sbd, torch.long), t(bd, torch.long), t(known))


class LowRankBeta(nn.Module):
    """beta = b0 + phi.w + c[b,d] + g * lam[b,d] + has_v * v * eta[b,d] + known * A[group].B[s,b,d]"""

    def __init__(self, n_phi: int, n_ident: int, n_sbd: int, n_bd: int, rank: int):
        super().__init__()
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Linear(n_phi, 1, bias=False)
        self.lam = nn.Embedding(n_bd, 1)
        self.eta = nn.Embedding(n_bd, 1)
        self.bdi = nn.Embedding(n_bd, 1)          # per (b, d) intercept (the mean baseline's structure)
        self.rank = rank
        if rank > 0:
            self.A = nn.Embedding(n_ident, rank)
            self.B = nn.Embedding(n_sbd, rank)
        for e in (self.lam, self.eta, self.bdi):
            nn.init.zeros_(e.weight)
        if rank > 0:
            nn.init.normal_(self.A.weight, std=0.1)
            nn.init.normal_(self.B.weight, std=0.1)

    def forward(self, phi, g, v, has_v, ident, sbd, bd, known):
        out = self.b0 + self.w(phi).squeeze(-1) + self.bdi(bd).squeeze(-1)
        out = out + g * self.lam(bd).squeeze(-1) + has_v * v * self.eta(bd).squeeze(-1)
        if self.rank > 0:
            out = out + known * (self.A(ident) * self.B(sbd)).sum(-1)
        return out


def fit_model(model: nn.Module, inputs, beta, se, ridge: float, steps: int = 1500, lr: float = 2e-2):
    """Inverse-variance weighted least squares plus a ridge penalty on everything except the
    intercepts, minimized with Adam."""
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


def predict(train: pd.DataFrame, test: pd.DataFrame, rank: int, ridge: float, known_groups: set) -> np.ndarray:
    torch.manual_seed(SEED)
    enc = Encoder(pd.concat([train, test]))
    m = LowRankBeta(4, len(enc.groups), len(enc.sbd), len(enc.bd), rank)
    fit_model(m, enc.encode(train, known_groups), train.beta.values, train.se.values, ridge)
    with torch.no_grad():
        return m(*enc.encode(test, known_groups)).numpy()


def inner_ridge(trials: pd.DataFrame, rank: int, split_col: str, alpha: float, rho: float, k: int = 3) -> float:
    """Tune the ridge penalty by k-fold CV on the training trials only, splitting on the same
    unit as the outer split (topics or dimensions) and rebuilding cells inside each fold."""
    units = sorted(trials[split_col].unique())
    random.Random(SEED).shuffle(units)
    folds = [set(units[i::k]) for i in range(k)]
    best, best_err = RIDGE_GRID[0], np.inf
    for lam in RIDGE_GRID:
        err, used = 0.0, 0
        for f in folds:
            tr = cell_slopes(trials[~trials[split_col].isin(f)], alpha, rho)
            va = cell_slopes(trials[trials[split_col].isin(f)], alpha, rho)
            if tr.empty or va.empty:
                continue
            known = set(tr.group) - (set(va.group) if split_col == "dimension" else set())
            pred = predict(tr, va, rank, lam, known)
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


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def topic_folds(topics) -> list[set]:
    t = sorted(topics)
    random.Random(SEED).shuffle(t)
    return [set(t[i::N_TOPIC_FOLDS]) for i in range(N_TOPIC_FOLDS)]


def evaluate_fold(d: pd.DataFrame, split: str, col: str, held: set) -> dict | None:
    """Fit on the training trials, predict the held-out cells' slope shifts with the full model,
    the no-W model and the cell-mean baseline."""
    tr_trials, te_trials = d[~d[col].isin(held)], d[d[col].isin(held)]
    if te_trials.empty:
        return None
    alpha, rho, beta_neu = neutral_reference(tr_trials)
    train, test = cell_slopes(tr_trials, alpha, rho), cell_slopes(te_trials, alpha, rho)
    if train.empty or test.empty:
        return None
    known = set(train.group)
    if split == "held_out_dimension":
        known -= set(test.group)
    true_shift = test.beta.values - beta_neu
    rec = dict(split=split, held=",".join(sorted(held)), n_train=len(train), n_test=len(test))
    for name, rank in (("full", RANK), ("no_W", 0)):
        lam = inner_ridge(tr_trials, rank, col, alpha, rho)
        pred = predict(train, test, rank, lam, known)
        rec.update({f"{name}_{k}": v for k, v in metrics(pred - beta_neu, true_shift).items()})
        rec[f"{name}_ridge"] = lam
    means = train.assign(w=1 / train.se ** 2).groupby(["b", "d"]).apply(
        lambda g: np.average(g.beta, weights=g.w), include_groups=False)
    pred = test.apply(lambda r: means.get((r.b, r.d), train.beta.mean()), axis=1).values
    rec.update({f"mean_{k}": v for k, v in metrics(pred - beta_neu, true_shift).items()})
    return rec


def run_model(df: pd.DataFrame, model: str, on: dict) -> pd.DataFrame:
    d = df[(df.model == model) & ~df.excluded & (df.tier == TIER)].copy()
    d["v_k"] = d.v_k.fillna(0.0)
    d["has_v"] = d.has_v.fillna(0.0)
    splits = []
    if on["held_out_topics"]:
        splits += [("held_out_topics", "topic", f) for f in topic_folds(d.topic.unique())]
    if on["held_out_dimensions"]:
        splits += [("held_out_dimension", "dimension", {k}) for k in DIMENSIONS]
    rows = [evaluate_fold(d, *s) for s in splits]
    return pd.DataFrame([r for r in rows if r is not None])


def success(summary_row: pd.Series) -> bool:
    r = summary_row
    return bool(r.full_rmse < r.no_W_rmse and r.full_rmse < r.mean_rmse
                and r.full_spearman > max(r.no_W_spearman, r.mean_spearman))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma-separated analyses to run (all others off)")
    ap.add_argument("--skip", help="comma-separated analyses to skip")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--models", help="comma-separated model keys (default: every main-study chat model)")
    ap.add_argument("--save", help="directory to write the result tables as CSV")
    ap.add_argument("--list", action="store_true", help="list the analyses and exit")
    a = ap.parse_args()
    on = dict(ANALYSES)
    for flag, value in (("only", a.only), ("skip", a.skip)):
        names = [x for x in (value or "").split(",") if x]
        unknown = set(names) - set(ANALYSES)
        if unknown:
            ap.error(f"unknown analyses for --{flag}: {', '.join(sorted(unknown))}")
        if flag == "only" and names:
            on = {k: k in names for k in ANALYSES}
        for k in names if flag == "skip" else []:
            on[k] = False
    return a, on


def main():
    a, on = parse_args()
    if a.list:
        comments = {}
        for line in Path(__file__).read_text().splitlines():
            for k in ANALYSES:
                if line.strip().startswith(f'"{k}"'):
                    comments[k] = line.split("#", 1)[1].strip()
        for k, v in ANALYSES.items():
            print(f"{k:22s} {'on ' if v else 'off'}  {comments.get(k, '')}")
        return 0
    t0 = time.time()
    df = load_dataset(Path(a.data))
    df = df[(df.study == STUDY) & (df.fmt == FMT)]
    models = sorted(df.model.unique())
    if a.models:
        wanted = a.models.split(",")
        missing = set(wanted) - set(models)
        if missing:
            sys.exit(f"no main-study chat trials for: {', '.join(sorted(missing))} (available: {', '.join(models)})")
        models = [m for m in models if m in wanted]
    save_dir = Path(a.save) if a.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if not (on["held_out_topics"] or on["held_out_dimensions"]):
        print("Both splits are switched off; nothing to evaluate.")
        return 0

    summaries, all_folds = [], []
    for model in models:
        folds = run_model(df, model, on)
        if folds.empty:
            print(f"RQ2 {model}: no fold had cells with enough trials to estimate unpooled slopes")
            continue
        all_folds.append(folds.assign(model=model))
        s = folds.groupby("split").mean(numeric_only=True).reset_index().assign(model=model)
        summaries.append(s)
        print(f"RQ2 {model}: {len(folds)} folds done ({time.time() - t0:.0f}s)")
    if not summaries:
        return 0
    folds = pd.concat(all_folds, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    if on["success_rule"]:
        summary["success"] = summary.apply(success, axis=1)

    cols = ["model", "split", "full_rmse", "no_W_rmse", "mean_rmse",
            "full_spearman", "no_W_spearman", "mean_spearman"] + (["success"] if on["success_rule"] else [])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n== RQ2: predicting slope shifts (chat format; metrics averaged over folds)")
        print(summary[cols].round(3).to_string(index=False))
        if on["success_rule"]:
            print("Success = full model beats both baselines on RMSE and on Spearman correlation.")

    if on["rq2b_gate"]:
        print("\n== RQ2b gate (preregistered: RQ2b runs only if RQ2 succeeds on held-out topics)")
        if not (on["held_out_topics"] and on["success_rule"]):
            print("cannot evaluate the gate: needs held_out_topics and success_rule switched on")
        else:
            topics = summary[summary.split == "held_out_topics"].set_index("model").success
            for model, ok in topics.items():
                print(f"{model}: {'PASSES the gate (RQ2b is not part of this release script)' if ok else 'RQ2b skipped (RQ2 did not succeed on held-out topics)'}")

    if save_dir:
        folds.to_csv(save_dir / "rq2_folds.csv", index=False)
        summary.to_csv(save_dir / "rq2_summary.csv", index=False)
        print(f"\nsaved rq2_folds.csv and rq2_summary.csv to {save_dir}")
    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
