"""Turn trial designs into model calls and assemble per-trial responses.

Per trial: `prior` (template 0), `prior_rep` (paraphrased template 1, the test-retest
replicate), `post` (template 0 with Results). Pilot subset (`complement`): also
`prior_H0` and `post_H0`. Calibration subset: `prior_s`, `post_s` read out by sampling.
Each call is an independent fresh context. Prior prompts contain no table, so they are
shared across trials and deduplicated by the cache.
"""
from __future__ import annotations

from pathlib import Path

import os

import numpy as np
import pandas as pd

from .cues import CueLibrary
from .elicit import CachedElicitor, Request, logit
from .prompts import Topic, TrialSlots, chat_trial, completion_trial, hedge_belief
from .tables import Table


def slots_for(row, topics: dict[str, Topic], cues: CueLibrary, with_table: bool, belief_strength: int = 0) -> TrialSlots:
    tab = Table(int(row.n_t), int(row.k_t), int(row.n_c), int(row.k_c), float(row.q)) if with_table else None
    return TrialSlots(cue_text=cues.by_id(row.cue_id).text, topic=topics[row.topic], source=row.source,
                      h_dir=int(row.h_dir), q=float(row.q), belief_dir=int(row.belief_dir), table=tab,
                      results_as_list=bool(row.results_as_list), treated_first=bool(row.treated_first),
                      belief_strength=int(belief_strength),
                      belief_key=str(row.trial_id) if int(belief_strength) else "")


def make_request(row, topics, cues, fmt: str, call: str, capture: bool = False,
                 variant: int = 0, system: str | None = None, belief_strength: int = 0) -> Request:
    with_table = call.startswith("post")
    paraphrase = 1 if call == "prior_rep" else 0
    target = "H0" if call.endswith("_H0") else "H1"
    s = slots_for(row, topics, cues, with_table, belief_strength)
    meta = dict(call="post" if with_table else ("prior_rep" if call == "prior_rep" else "prior"),
                target=target, topic=row.topic, h_dir=int(row.h_dir), cue_id=row.cue_id, group=row.group,
                b=int(row.b), x=float(row.x), tier=int(row.tier),
                # stated-belief sentence (for span interventions; not part of the prompt key)
                span=hedge_belief(s.topic.belief(s.belief_dir), s.belief_strength, s.belief_key or None)
                if s.belief_dir != 0 else None)
    if belief_strength:
        meta["belief_strength"] = int(belief_strength)
    readout = "sample" if call.endswith("_s") else None
    if fmt == "chat":
        return Request("number", "chat", messages=chat_trial(s, paraphrase, target, variant, system), readout=readout,
                       capture=capture, meta=meta)
    return Request("number", "completion", text=completion_trial(s, paraphrase, target), readout=readout,
                   capture=capture, meta=meta)


def calls_for(row, calibration: bool) -> list[str]:
    calls = ["prior", "prior_rep", "post"]
    if not bool(getattr(row, "retest", True)):
        calls.remove("prior_rep")   # test-retest asked on a subset only (budget-limited API models)
    if getattr(row, "complement", False):
        calls += ["prior_H0", "post_H0"]
    if calibration and getattr(row, "calibration", False):
        calls += ["prior_s", "post_s"]
    return calls


def run_trials(trials: pd.DataFrame, topics: list[Topic], cues: CueLibrary, elicitor: CachedElicitor,
               fmt: str, capture: bool = False, calibration: bool = True) -> pd.DataFrame:
    """Return one row per trial with p_* and cov_* per call, plus request keys for the
    posterior (used to look up activations)."""
    tmap = {t.topic_id: t for t in topics}
    spec = elicitor.b.spec
    variant, system = getattr(spec, "prompt_variant", 0), getattr(spec, "system_prompt", None)
    strength = getattr(spec, "belief_strength", 0)
    if (variant or system or strength) and fmt != "chat":
        raise ValueError(f"{spec.key}: prompt variants, system prompts and belief strengths are chat-only")
    reqs, index = [], []
    for row in trials.itertuples(index=False):
        for call in calls_for(row, calibration):
            cap = capture and call == "post"   # probes and directions use the posterior prompt
            reqs.append(make_request(row, tmap, cues, fmt, call, capture=cap, variant=variant, system=system,
                                     belief_strength=strength))
            index.append((row.trial_id, call))
    results = elicitor.run(reqs)
    recs: dict[str, dict] = {}
    for (tid, call), req, res in zip(index, reqs, results):
        r = recs.setdefault(tid, {"trial_id": tid})
        r[f"p_{call}"] = res.p
        r[f"cov_{call}"] = res.coverage
        r[f"att_{call}"] = res.attempt
        if call in ("post", "prior"):
            r[f"key_{call}"] = elicitor.key(req)
        r["model_version"] = res.model_version
    out = pd.DataFrame(list(recs.values()))
    out["fmt"] = fmt
    out["model"] = elicitor.b.spec.key
    out["readout"] = elicitor.b.spec.readout
    return trials.merge(out, on="trial_id", how="left")


# Completion-format (plain-text) prompts: the pilot (2026-11-10) found OLMo puts a few percent
# of next-token mass on words, so integer coverage sits near 0.85-0.9 and the 0.9 rule kept only
# 25-35% of answers. Approved deviation: completion-format cutoff 0.5 (mass is renormalized over
# the integers, as before), with 0.8 as a robustness check (env IDEA_COVERAGE_COMPLETION=0.8).
COVERAGE_MIN_COMPLETION = float(os.environ.get("IDEA_COVERAGE_COMPLETION", 0.5))


def derive(df: pd.DataFrame, coverage_min: float = 0.9,
           coverage_min_completion: float | None = None) -> pd.DataFrame:
    """Add analysis columns: lp_before, lp_before_rep, lp_after, y, omega, exclusion flag."""
    df = df.copy()
    for c in ("p_prior_rep", "cov_prior_rep"):   # absent when no trial in a file had the retest asked
        if c not in df:
            df[c] = np.nan
    if coverage_min_completion is None:
        coverage_min_completion = COVERAGE_MIN_COMPLETION
    cov_cols = [c for c in ("cov_prior", "cov_prior_rep", "cov_post") if c in df]
    ok = np.ones(len(df), bool)
    # dist readouts: mass on valid integers >= coverage_min (chat) or coverage_min_completion
    # (completion); sampling: >= half the samples parse
    readout = df["readout"] if "readout" in df else pd.Series("dist", index=df.index)
    fmt = df["fmt"] if "fmt" in df else pd.Series("chat", index=df.index)
    dist_min = np.where(fmt.values == "completion", coverage_min_completion, coverage_min)
    thresh = np.where(readout.values == "dist", dist_min, 0.5)
    # trials whose paraphrased prior was not asked by design (retest == False) are not excluded for it
    no_rep = (df["retest"].astype("boolean").fillna(True) == False).values if "retest" in df else np.zeros(len(df), bool)  # noqa: E712
    for c in cov_cols:
        cond = df[c].fillna(0).values >= thresh
        ok &= (cond | no_rep) if c == "cov_prior_rep" else cond
    for c in ("p_prior", "p_prior_rep", "p_post"):
        cond = df[c].notna().values
        ok &= (cond | no_rep) if c == "p_prior_rep" else cond
    df["excluded"] = ~ok
    df["lp_before"] = logit(df.p_prior.astype(float).clip(0.005, 0.995))
    df["lp_before_rep"] = logit(df.p_prior_rep.astype(float).clip(0.005, 0.995))
    df["lp_after"] = logit(df.p_post.astype(float).clip(0.005, 0.995))
    df["y"] = df.lp_after - df.lp_before
    with np.errstate(divide="ignore", invalid="ignore"):
        df["omega"] = np.where(df.x != 0, df.y / df.x, np.nan)
    df["bayes_post"] = df.lp_before + df.x
    df["contra"] = (df.e == "contra").astype(int)
    df["support"] = (df.e == "support").astype(int)
    df["has_belief"] = df.b != 0
    for c in ("p_prior_H0", "p_post_H0"):
        if c in df:
            pass
    if "p_prior_H0" in df:
        df["sum_before"] = df.p_prior + df.p_prior_H0
        df["sum_after"] = df.p_post + df.p_post_H0
    return df
