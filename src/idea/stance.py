"""Perceived identity stance (Section 4.6, Eq. 2), survey stance, behavioral stance,
topic screening and the Tier 1 rule.

g(c, H1) = logit p_c - logit p_unsp, where p_unsp is the answer for the matched
Unspecified cue (same cue type). Each cue (label or vignette) gets its own g, per model,
per hypothesis direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cues import DIMENSIONS, GROUP_TO_DIM, UNSPECIFIED, CueLibrary
from .elicit import CachedElicitor, Request, logit
from .prompts import Topic, stance_completion, stance_prompt


def stance_requests(cues: CueLibrary, topics: list[Topic], fmt: str = "chat") -> tuple[list[Request], list[dict]]:
    reqs, index = [], []
    for t in topics:
        for direction in (1, -1):
            for c in cues.cues:
                if fmt == "chat":
                    r = Request("number", "chat", messages=stance_prompt(c.text, t, direction))
                else:
                    r = Request("number", "completion", text=stance_completion(c.text, t, direction))
                r.answer_format = "text"  # stance stays free text for every model (decision 2026-11-10)
                r.meta = dict(call="stance", cue_id=c.cue_id, topic=t.topic_id, h_dir=direction)
                reqs.append(r)
                index.append(dict(cue_id=c.cue_id, group=c.group_id, dimension=c.dimension,
                                  cue_type=c.cue_type, topic=t.topic_id, tier=t.tier, h_dir=direction))
    return reqs, index


def run_stance(cues: CueLibrary, topics: list[Topic], elicitor: CachedElicitor, fmt: str = "chat") -> pd.DataFrame:
    reqs, index = stance_requests(cues, topics, fmt)
    res = elicitor.run(reqs)
    df = pd.DataFrame(index)
    df["p"] = [r.p for r in res]
    df["coverage"] = [r.coverage for r in res]
    df["model"] = elicitor.b.spec.key
    return compute_g(df)


def compute_g(df: pd.DataFrame) -> pd.DataFrame:
    """Add g = logit p - logit p_unsp (matched cue type; mean over Unspecified cues)."""
    df = df.copy()
    df["lp"] = logit(df.p.astype(float).clip(0.005, 0.995))
    base = (df[df.group == UNSPECIFIED].groupby(["model", "topic", "h_dir", "cue_type"]).lp.mean()
            .rename("lp_unsp"))
    df = df.join(base, on=["model", "topic", "h_dir", "cue_type"])
    df["g"] = df.lp - df.lp_unsp
    return df


def g_lookup(stance: pd.DataFrame) -> pd.Series:
    """(model, cue_id, topic, h_dir) -> g, for merging onto trials."""
    return stance.set_index(["model", "cue_id", "topic", "h_dir"]).g


def attach_g(trials: pd.DataFrame, stance: pd.DataFrame) -> pd.DataFrame:
    s = stance[["model", "cue_id", "topic", "h_dir", "g"]]
    out = trials.drop(columns=[c for c in ("g",) if c in trials]).merge(
        s, on=["model", "cue_id", "topic", "h_dir"], how="left")
    out["g"] = out.g.fillna(0.0)
    out["m"] = out.b * out.g
    return out


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen_topics(stance: pd.DataFrame, g_min: float = 0.5, min_dims: int = 2,
                  tier1_max: float = 0.25) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Tier 2: keep a topic if >= min_dims dimensions have some group with |g| >= g_min
    (averaged over that group's cues and over models). Tier 1: keep an item only if every
    cue has |g| < tier1_max. Returns (per-topic report, dimension -> engaged topics)."""
    s = stance[stance.group != UNSPECIFIED]
    grp = s.groupby(["tier", "topic", "dimension", "group", "h_dir"]).g.mean().abs().reset_index()
    by_dim = grp.groupby(["tier", "topic", "dimension"]).g.max().unstack("dimension")
    rep = by_dim.copy()
    rep["n_dims_engaged"] = (by_dim >= g_min).sum(axis=1)
    rep["max_abs_g_any_cue"] = s.groupby(["tier", "topic"]).g.apply(lambda v: v.abs().max())
    # Tier 1 (deviation approved 2026-11-11): every Tier 1 topic is kept, because the
    # preregistered "|g| < tier1_max for every cue" rule removed all of them in the pilot.
    # The rule is applied instead as a robustness subset (tier1_low_g_pairs).
    rep["tier1_rule_pass"] = rep.max_abs_g_any_cue < tier1_max
    rep["keep"] = np.where(rep.index.get_level_values("tier") == 2, rep.n_dims_engaged >= min_dims, True)
    rep = rep.reset_index()
    engaged = {d: sorted(rep[(rep.tier == 2) & rep.keep & (rep.get(d, 0) >= g_min)].topic) for d in DIMENSIONS}
    return rep, engaged


def tier1_low_g_pairs(stance: pd.DataFrame, tier1_max: float = 0.25) -> list[list[str]]:
    """Tier 1 robustness subset: (topic, cue_id) pairs whose |mean g| over models, samples and
    H1 directions is below tier1_max. Unspecified cues always qualify (g = 0 by definition)."""
    s = stance[(stance.tier == 1) & (stance.group != UNSPECIFIED)]
    pc = s.groupby(["topic", "cue_id"]).g.mean().abs()
    return [list(k) for k in pc[pc < tier1_max].index]


def check_dim_coverage(engaged: dict[str, list[str]], min_topics: int = 3) -> dict[str, int]:
    """Dimensions with fewer than `min_topics` engaged topics (need more topics)."""
    return {d: len(v) for d, v in engaged.items() if len(v) < min_topics}


# ---------------------------------------------------------------------------
# Survey and behavioral stance
# ---------------------------------------------------------------------------

def g_survey(survey: pd.DataFrame) -> pd.DataFrame:
    """survey columns: topic, h_dir, group, pct_group, pct_all (percent agreeing with the
    matched Pew item). g_svy = logit(group) - logit(all adults)."""
    s = survey.copy()
    pg = (s.pct_group / 100).clip(0.005, 0.995)
    pa = (s.pct_all / 100).clip(0.005, 0.995)
    s["g_svy"] = logit(pg) - logit(pa)
    return s


def g_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """g_beh(c, H1) = E[l | c, b=0] - E[l | Unspecified, b=0], per model/topic/h_dir/cue type."""
    neu = df[(df.b == 0) & ~df.get("excluded", False)]
    means = neu.groupby(["model", "cue_id", "group", "cue_type", "topic", "h_dir"]).lp_before.mean().rename("lp").reset_index()
    unsp = (means[means.group == UNSPECIFIED].groupby(["model", "topic", "h_dir", "cue_type"]).lp.mean()
            .rename("lp_unsp"))
    means = means.join(unsp, on=["model", "topic", "h_dir", "cue_type"])
    means["g_beh"] = means.lp - means.lp_unsp
    return means


def stance_spread_report(stance: pd.DataFrame) -> pd.DataFrame:
    """Pilot check for deflection: per model/dimension/cue type, the SD of g, share of
    |g| < 0.1, and mean coverage."""
    s = stance[stance.group != UNSPECIFIED]
    return s.groupby(["model", "dimension", "cue_type"]).agg(
        g_sd=("g", "std"), share_near_zero=("g", lambda v: float((v.abs() < 0.1).mean())),
        coverage=("coverage", "mean"), n=("g", "size")).reset_index()


def attach_survey_gap(trials: pd.DataFrame, topics: list) -> pd.DataFrame:
    """v_k(t) for the trial's identity dimension, in units of 100 percentage points,
    with has_v = 0 (and v_k = 0) where Pew has no breakdown."""
    tm = {t.topic_id: t for t in topics}
    v = [tm[t].v_k(dim) if (t in tm and dim in DIMENSIONS) else None for t, dim in zip(trials.topic, trials.dimension)]
    out = trials.copy()
    out["has_v"] = [0.0 if x is None else 1.0 for x in v]
    out["v_k"] = [0.0 if x is None else x / 100 for x in v]
    return out
