"""Trial sampling: base items, identity-topic pairing rule, randomized slots, controls.

A *base item* is (tier, topic, H1 direction, table). Trials reuse base items across
identity cues, beliefs and sources, so item random intercepts and matched caved/resisted
strata (RQ3) are well defined. Within each cell x is drawn at random, independent of the
prior, as the design requires (Section 4.5 notes).
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cues import ALL_GROUPS, DIMENSIONS, GROUP_TO_DIM, UNSPECIFIED, CueLibrary
from .prompts import SOURCES, Topic
from .tables import N_RANGE, Q_VALUES, RATE_RANGE, Table, sample_null_table, sample_table, trial_x, validate_tables

LOGIT_03, LOGIT_97 = math.log(0.03 / 0.97), math.log(0.97 / 0.03)


def x_range(mean_prior_logit: float, x_min: float = -3.0, x_max: float = 3.0) -> tuple[float, float]:
    """Ceiling control: keep the Bayesian posterior within [0.03, 0.97] for the cell."""
    return max(x_min, LOGIT_03 - mean_prior_logit), min(x_max, LOGIT_97 - mean_prior_logit)


@dataclass
class DesignConfig:
    seed: int = 0
    tiers: tuple[int, ...] = (2, 1)
    cue_types: tuple[str, ...] = ("label", "vignette")
    n_neutral: int = 150            # b = 0 trials per (group, cue_type) in Tier 2
    n_belief: int = 600             # b != 0 trials per (group, cue_type) in Tier 2 (~half contradicted)
    tier1_scale: float = 0.4        # Tier 1 trials as a fraction of Tier 2 counts
    include_unspecified: bool = True
    groups: tuple[str, ...] | None = None   # None = all 22
    items_per_topic_dir: int = 30
    p_match: float = 0.75           # share of trials on topics that engage the identity's dimension
    x_min: float = -3.0
    x_max: float = 3.0
    null_frac: float = 0.05
    conflict_frac: float = 0.5
    q_values: tuple[float, ...] = Q_VALUES
    n_range: tuple[int, int] = N_RANGE
    rate_range: tuple[float, float] = RATE_RANGE
    complement_frac: float = 0.0    # pilot: share of trials that also elicit P(H0)
    calibration_frac: float = 0.0   # share read out by sampling on open models
    engaged: dict | None = None     # dimension -> list of topic_ids (from stance screen); None = CSV
    prior_means: dict | None = None  # (tier, topic, h_dir, b, group) -> mean prior logit (pilot)
    scale: float = 1.0              # multiply all counts (pilot/smoke runs)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignConfig":
        d = dict(d or {})
        for k in ("tiers", "cue_types", "q_values", "n_range", "rate_range", "groups"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**d)


def _stable_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def build_items(topics: list[Topic], cfg: DesignConfig, rng: random.Random) -> pd.DataFrame:
    """For each (tier, topic, h_dir) draw items_per_topic_dir tables with x stratified over
    [x_min, x_max], q cycled, a share heuristic-conflicting, and a share null (x = 0).
    Tier 1 and Tier 2 use the *same* table sets, so the tiers differ only in framing."""
    n = cfg.items_per_topic_dir
    # One shared pool of tables per slot index, reused across topics and tiers.
    pool: list[Table] = []
    n_null = max(1, round(n * cfg.null_frac)) if cfg.null_frac > 0 else 0
    edges = np.linspace(cfg.x_min, cfg.x_max, n - n_null + 1)
    # Conflict status is assigned by a low-discrepancy sequence (exactly balanced at 0.5:
    # alternating), and q changes every two slots, so q, conflict status and x strata are
    # not confounded with one another.
    offset = rng.random()
    flipped = 0
    for i in range(n - n_null):
        target = rng.uniform(edges[i], edges[i + 1])
        if abs(target) < 0.15:
            target = math.copysign(0.15, target or 1.0)
        q = cfg.q_values[(i // 2) % len(cfg.q_values)]
        conflict = ((i * cfg.conflict_frac + offset) % 1.0) < cfg.conflict_frac
        try:
            pool.append(sample_table(target, q, conflict, rng, cfg.n_range, cfg.rate_range))
        except RuntimeError:
            flipped += 1
            pool.append(sample_table(target, q, not conflict, rng, cfg.n_range, cfg.rate_range))
    if flipped:
        import warnings
        warnings.warn(f"{flipped} table slots could not take their assigned conflict status")
    for i in range(n_null):
        pool.append(sample_null_table(cfg.q_values[i % len(cfg.q_values)], rng, cfg.n_range, cfg.rate_range))
    validate_tables(pool)

    rows = []
    for t in topics:
        if t.tier not in cfg.tiers:
            continue
        for h_dir in (1, -1):
            for slot, tab in enumerate(pool):
                rows.append(dict(item_id=_stable_id(t.tier, t.topic_id, h_dir, slot), base_slot=slot,
                                 tier=t.tier, topic=t.topic_id, h_dir=h_dir,
                                 n_t=tab.n_t, k_t=tab.k_t, n_c=tab.n_c, k_c=tab.k_c, q=tab.q,
                                 D=tab.D, table_x=tab.x, x=trial_x(tab, h_dir),
                                 heuristic_conflict=tab.heuristic_conflict))
    return pd.DataFrame(rows)


def engaged_topics(topics: list[Topic], cfg: DesignConfig) -> dict[str, set[str]]:
    if cfg.engaged:
        return {d: set(v) for d, v in cfg.engaged.items()}
    out = {d: set() for d in DIMENSIONS}
    for t in topics:
        for d in t.dims_engaged:
            out.setdefault(d, set()).add(t.topic_id)
    return out


def build_trials(topics: list[Topic], cues: CueLibrary, cfg: DesignConfig,
                 cue_exclude: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (items, trials). `cue_exclude`: vignettes that failed identification for the
    model this design is for (they are dropped before sampling)."""
    rng = random.Random(cfg.seed)
    items = build_items(topics, cfg, rng)
    by_cell = {k: g for k, g in items.groupby(["tier", "topic", "h_dir"])}
    tier_topics = {tier: [t for t in topics if t.tier == tier] for tier in cfg.tiers}
    eng = engaged_topics(topics, cfg)
    groups = list(cfg.groups or ALL_GROUPS) + ([UNSPECIFIED] if cfg.include_unspecified else [])

    trials = []
    for tier in cfg.tiers:
        scale = cfg.scale * (1.0 if tier == 2 else cfg.tier1_scale)
        tlist = tier_topics[tier]
        if not tlist:
            continue
        for group in groups:
            dim = GROUP_TO_DIM.get(group, "none")
            for cue_type in cfg.cue_types:
                pool = cues.select(group_id=group, cue_type=cue_type, exclude=cue_exclude)
                if not pool:
                    continue
                n_neu = max(1, round(cfg.n_neutral * scale))
                n_bel = max(2, round(cfg.n_belief * scale))
                specs = [0] * n_neu + [1, -1] * (n_bel // 2)
                for b in specs:
                    # pairing rule (Tier 2 only; Tier 1 topics engage nothing)
                    if tier == 2 and dim in eng and eng[dim]:
                        matched = [t for t in tlist if t.topic_id in eng[dim]]
                        unmatched = [t for t in tlist if t.topic_id not in eng[dim]] or matched
                        is_match = rng.random() < cfg.p_match
                        topic = rng.choice(matched if is_match else unmatched)
                        is_match = topic.topic_id in eng[dim]
                    else:
                        topic, is_match = rng.choice(tlist), False
                    h_dir = rng.choice((1, -1))
                    cand = by_cell[(tier, topic.topic_id, h_dir)]
                    if cfg.prior_means:
                        mp = cfg.prior_means.get((tier, topic.topic_id, h_dir, b, group))
                        if mp is not None:
                            lo, hi = x_range(mp, cfg.x_min, cfg.x_max)
                            sub = cand[((cand.x >= lo) & (cand.x <= hi)) | (cand.D == 0)]
                            cand = sub if len(sub) else cand
                    item = cand.iloc[rng.randrange(len(cand))]
                    cue = rng.choice(pool)
                    source = rng.choice(list(SOURCES))
                    rec = item.to_dict()
                    rec.update(
                        group=group, dimension=dim, cue_id=cue.cue_id, cue_type=cue_type,
                        b=b, belief_dir=b * h_dir,  # belief_dir: which claim the user states
                        source=source, matched_pair=bool(is_match),
                        results_as_list=rng.random() < 0.5, treated_first=rng.random() < 0.5,
                        complement=rng.random() < cfg.complement_frac,
                        calibration=rng.random() < cfg.calibration_frac,
                    )
                    trials.append(rec)
    df = pd.DataFrame(trials)
    df.insert(0, "trial_id", [f"t{i:06d}" for i in range(len(df))])
    df["d"] = np.sign(df.x).astype(int)
    df["kappa"] = ((df.b * df.d) < 0).astype(int)
    df["e"] = np.select([df.b == 0, df.b * df.d > 0, df.b * df.d < 0], ["neutral", "support", "contra"], "null")
    return items, df
