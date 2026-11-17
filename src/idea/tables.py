"""Synthetic 2x2 contingency tables with a known log-likelihood ratio (Eq. 1).

H1: success rate q under the policy (treated) and 1-q without it (control).
H0: the reverse. With both hypotheses stipulated exclusive and exhaustive,

    x = log P(E|H1)/P(E|H0) = log(q/(1-q)) * [(2 k_T - n_T) - (2 k_C - n_C)].

`Table.x` is always expressed for the "treated has the higher rate" hypothesis.
A trial whose H1 is written in the negative direction uses x_trial = -Table.x
(see `trial_x`).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Iterable

# Approved 2026-11-10 after pilot 1: small groups and q near 0.6 make evidence of |x| <= 3 visible
# (median rate gap ~10 points instead of ~3). The notes' original values were q 0.52-0.58, groups 50-400.
Q_VALUES = (0.58, 0.60, 0.62)
N_RANGE = (10, 30)


@dataclass(frozen=True)
class Table:
    n_t: int
    k_t: int
    n_c: int
    k_c: int
    q: float

    def __post_init__(self):
        if not (0 <= self.k_t <= self.n_t and 0 <= self.k_c <= self.n_c):
            raise ValueError(f"invalid counts {self}")
        if not (0.5 < self.q < 1):
            raise ValueError("q must be in (0.5, 1)")

    @property
    def L(self) -> float:
        return math.log(self.q / (1 - self.q))

    @property
    def D(self) -> int:
        return (2 * self.k_t - self.n_t) - (2 * self.k_c - self.n_c)

    @property
    def x(self) -> float:
        return self.L * self.D

    @property
    def rate_t(self) -> float:
        return self.k_t / self.n_t

    @property
    def rate_c(self) -> float:
        return self.k_c / self.n_c

    @property
    def rate_agrees(self) -> bool:
        """The evidence direction matches the visible comparison of rates: the group whose
        rate is higher is the one the favored hypothesis says is higher. Null tables (D = 0)
        must show identical groups. Without this, the exact log-LR (which scores each group
        against the hypothesized absolute rates) can point against the obvious reading of
        the table; 45% of the first pilot's tables did (2026-11-10)."""
        if self.D == 0:
            return self.n_t == self.n_c and self.k_t == self.k_c
        diff = self.k_t * self.n_c - self.k_c * self.n_t  # sign of rate_t - rate_c, exact
        return diff != 0 and (diff > 0) == (self.D > 0)

    @property
    def heuristic_conflict(self) -> bool:
        """Raw success counts point the opposite way from the success rates."""
        raw = self.k_t - self.k_c
        rate = self.rate_t - self.rate_c
        return raw * rate < 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(D=self.D, x=self.x, heuristic_conflict=self.heuristic_conflict)
        return d


def trial_x(table: Table, h_dir: int) -> float:
    """Evidence for the trial's H1. h_dir=+1: H1 says treated rate is q; -1: reverse."""
    if h_dir not in (1, -1):
        raise ValueError("h_dir must be +1 or -1")
    return h_dir * table.x


def loglik_ratio_binomial(table: Table) -> float:
    """Direct computation via binomial log-pmfs (used to verify Eq. 1)."""
    from scipy.stats import binom

    q = table.q
    l1 = binom.logpmf(table.k_t, table.n_t, q) + binom.logpmf(table.k_c, table.n_c, 1 - q)
    l0 = binom.logpmf(table.k_t, table.n_t, 1 - q) + binom.logpmf(table.k_c, table.n_c, q)
    return float(l1 - l0)


RATE_RANGE = (0.20, 0.80)


def sample_table(target_x: float, q: float, conflict: bool, rng: random.Random,
                 n_range=N_RANGE, rate_range=RATE_RANGE, tries: int = 200000) -> Table:
    """Sample counts whose x is as close as possible to target_x (D is an integer).

    Every table satisfies `rate_agrees` (evidence direction = visible rate comparison).
    conflict=True requires a heuristic-conflicting table (unequal group sizes make raw
    successes and rates disagree). Always analyze the realized `x`, never the target.
    """
    L = math.log(q / (1 - q))
    D = round(target_x / L)
    if D == 0 and target_x != 0:
        D = 1 if target_x > 0 else -1  # a non-null target must not silently become a null table
    if D == 0:
        return sample_null_table(q, rng, n_range, rate_range)
    lo_r, hi_r = rate_range
    for _ in range(tries):
        if conflict:
            # Conflict needs quite unequal arms; draw one small and one large.
            a, b = rng.randint(n_range[0], (n_range[0] + n_range[1]) // 2), rng.randint(
                (n_range[0] + n_range[1]) // 2, n_range[1])
            n_t, n_c = (a, b) if rng.random() < 0.5 else (b, a)
        else:
            n_t, n_c = rng.randint(*n_range), rng.randint(*n_range)
        k_c = rng.randint(math.ceil(lo_r * n_c), math.floor(hi_r * n_c))
        two_k_t = D + (2 * k_c - n_c) + n_t
        if two_k_t % 2:
            continue
        k_t = two_k_t // 2
        if not (0 <= k_t <= n_t) or not (lo_r <= k_t / n_t <= hi_r):
            continue
        t = Table(n_t, k_t, n_c, k_c, q)
        if t.heuristic_conflict == conflict and t.rate_agrees:
            return t
    raise RuntimeError(f"no table for target_x={target_x}, q={q}, conflict={conflict}; "
                       "widen n_range or rate_range")


def sample_null_table(q: float, rng: random.Random, n_range=N_RANGE,
                      rate_range=RATE_RANGE) -> Table:
    """A D = 0 table, so x = 0 exactly (null-evidence control): two identical groups, so
    the table also looks like no evidence."""
    n = rng.randint(*n_range)
    k = rng.randint(math.ceil(rate_range[0] * n), math.floor(rate_range[1] * n))
    return Table(n, k, n, k, q)


def validate_tables(tables) -> None:
    """Raise if any table's evidence direction disagrees with its visible rate comparison."""
    bad = [t for t in tables if not t.rate_agrees]
    if bad:
        raise ValueError(f"{len(bad)} tables have evidence pointing against their rate comparison, e.g. {bad[0]}")
