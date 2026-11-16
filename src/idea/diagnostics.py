"""Evidence diagnostic (exploratory addition, 2026-11-11; see the design-changes doc).

The pilot found that the 7-8B open models' posteriors do not track the evidence even when no
belief or persona is present. This module asks whether they can use the table at all, on the
same 2x2 tables as the main run, with no persona and no belief, at four levels:

  rate_t / rate_c  read one group's percentage from the Results block (number, 0-100)
  compare          which group had the higher rate (choice A/B, order counterbalanced)
  support          which hypothesis the results support more (choice A/B, counterbalanced)
  post             the original posterior question for H1 (number), without persona or belief

Each call is an independent fresh context, read out like the main run (next-token distribution).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .elicit import CachedElicitor, Request
from .prompts import (COMPLETION_HEADER, QUESTIONS, Topic, _cap, hypotheses_text, results_text,
                      study_text)
from .tables import Table

LETTERS = ["A", "B"]
KINDS = ["rate_t", "rate_c", "compare", "support", "post"]


def select_items(trials: pd.DataFrame, n: int = 480, seed: int = 20261111) -> pd.DataFrame:
    """Distinct evidence tables from the main trials (posterior-bearing rows), spread over topics."""
    cols = ["topic", "tier", "h_dir", "q", "n_t", "k_t", "n_c", "k_c", "x", "source",
            "results_as_list", "treated_first"]
    t = trials[cols].drop_duplicates(["topic", "h_dir", "q", "n_t", "k_t", "n_c", "k_c"])
    t = t[(t.k_t * t.n_c) != (t.k_c * t.n_t)]          # no tied rates (compare needs a right answer)
    rng = np.random.default_rng(seed)
    per = max(1, n // t.topic.nunique())
    parts = [g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 31))) for _, g in t.groupby("topic")]
    out = pd.concat(parts).head(n).reset_index(drop=True)
    out["diag_id"] = [f"d{i:04d}" for i in range(len(out))]
    out["flip"] = rng.integers(0, 2, len(out)).astype(bool)   # counterbalance A/B order
    return out


def _table(r) -> Table:
    return Table(int(r.n_t), int(r.k_t), int(r.n_c), int(r.k_c), float(r.q))


def build_requests(items: pd.DataFrame, topics: dict[str, Topic], fmt: str):
    reqs, index = [], []
    for r in items.itertuples(index=False):
        tp = topics[r.topic]
        study = study_text(tp, r.source)
        res = results_text(tp, _table(r), bool(r.results_as_list), bool(r.treated_first))
        hyp = hypotheses_text(tp, float(r.q), int(r.h_dir), bool(r.treated_first))
        groups = [tp.treated, tp.control]
        rate_t, rate_c = 100 * r.k_t / r.n_t, 100 * r.k_c / r.n_c
        opts_g = groups[::-1] if r.flip else groups
        opts_h = ["H0", "H1"] if r.flip else ["H1", "H0"]
        q = {
            "rate_t": (f"In these results, what percentage of {tp.treated} {tp.outcome_past}? "
                       "Answer with a single integer from 0 to 100 and nothing else."),
            "rate_c": (f"In these results, what percentage of {tp.control} {tp.outcome_past}? "
                       "Answer with a single integer from 0 to 100 and nothing else."),
            "compare": (f"In these results, which group had the higher percentage who {tp.outcome_past}?\n"
                        f"(A) {_cap(opts_g[0])}\n(B) {_cap(opts_g[1])}\nAnswer with the letter only."),
            "support": ("Which hypothesis do these results support more?\n"
                        f"(A) {opts_h[0]}\n(B) {opts_h[1]}\nAnswer with the letter only."),
            "post": QUESTIONS[(0, "H1")],
        }
        bodies = {"rate_t": [study, res], "rate_c": [study, res], "compare": [study, res],
                  "support": [study, hyp, res], "post": [study, hyp, res]}
        truth = {"rate_t": rate_t, "rate_c": rate_c,
                 "compare": opts_g.index(tp.treated if rate_t > rate_c else tp.control),
                 "support": opts_h.index("H1" if r.x > 0 else "H0"), "post": np.nan}
        for kind in KINDS:
            body = "\n\n".join(bodies[kind])
            choice = kind in ("compare", "support")
            if fmt == "chat":
                req = Request("choice" if choice else "number", "chat",
                              messages=[{"role": "user", "content": body + "\n\n" + q[kind]}],
                              letters=LETTERS if choice else None, answer_format="text")
            else:
                tail = "Answer:" if choice else "Answer (0-100):"
                req = Request("choice" if choice else "number", "completion",
                              text=f"{COMPLETION_HEADER}\n\n{body}\n\n{q[kind]}\n{tail}",
                              letters=LETTERS if choice else None, answer_format="text")
            reqs.append(req)
            index.append(dict(diag_id=r.diag_id, kind=kind, truth=truth[kind]))
    return reqs, index


def run_diag(items: pd.DataFrame, topics: list[Topic], elicitor: CachedElicitor, fmt: str) -> pd.DataFrame:
    tmap = {t.topic_id: t for t in topics}
    reqs, index = build_requests(items, tmap, fmt)
    res = elicitor.run(reqs)
    df = pd.DataFrame(index)
    df["p"] = [r.p for r in res]
    df["probs"] = [r.probs for r in res]
    df["coverage"] = [r.coverage for r in res]
    df["model"], df["fmt"] = elicitor.b.spec.key, fmt
    return df.merge(items[["diag_id", "topic", "tier", "x", "flip"]], on="diag_id")


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Per model x format: rate-reading error, choice accuracies, posterior-evidence slope."""
    from .elicit import logit
    rows = []
    for (m, f), g in df.groupby(["model", "fmt"]):
        rate = g[g.kind.isin(["rate_t", "rate_c"]) & g.p.notna()]
        est = 100 * rate.p.astype(float)
        err = (est - rate.truth.astype(float)).abs()
        row = dict(model=m, fmt=f, n_items=g.diag_id.nunique(),
                   rate_mae=err.mean(), rate_within5=(err <= 5).mean(),
                   rate_corr=np.corrcoef(est, rate.truth.astype(float))[0, 1] if len(rate) > 2 else np.nan)
        for kind in ("compare", "support"):
            c = g[(g.kind == kind) & g.probs.notna()]
            p_right = np.array([pr[int(t)] for pr, t in zip(c.probs, c.truth)]) if len(c) else np.array([])
            row[f"{kind}_acc"] = float((p_right > 0.5).mean()) if len(c) else np.nan
            row[f"{kind}_p_right"] = float(p_right.mean()) if len(c) else np.nan
            # position bias: probability on "A" regardless of truth
            row[f"{kind}_p_A"] = float(np.mean([pr[0] for pr in c.probs])) if len(c) else np.nan
        post = g[(g.kind == "post") & g.p.notna()]
        lp = logit(post.p.astype(float).clip(0.005, 0.995))
        row["post_slope_vs_x"] = float(np.polyfit(post.x, lp, 1)[0]) if len(post) > 5 else np.nan
        row["post_dir_acc"] = float((np.sign(post.p - 0.5) == np.sign(post.x)).mean()) if len(post) else np.nan
        row["mean_coverage"] = g.coverage.mean()
        rows.append(row)
    return pd.DataFrame(rows)
