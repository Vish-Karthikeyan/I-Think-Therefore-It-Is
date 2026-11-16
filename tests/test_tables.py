import math
import random

import pytest

from idea.tables import (Q_VALUES, Table, loglik_ratio_binomial, sample_null_table, sample_table,
                         sample_tables, trial_x)


def test_running_example():
    t = Table(n_t=10, k_t=4, n_c=10, k_c=6, q=0.6)
    assert t.D == -4
    assert t.x == pytest.approx(-4 * math.log(1.5))
    assert t.x == pytest.approx(-1.62, abs=0.01)


def test_proposal_example():
    t = Table(n_t=200, k_t=110, n_c=190, k_c=100, q=0.54)
    assert t.D == 10
    assert t.x == pytest.approx(1.6, abs=0.01)


@pytest.mark.parametrize("seed", range(20))
def test_x_matches_binomial(seed):
    rng = random.Random(seed)
    for _ in range(50):
        n_t, n_c = rng.randint(5, 400), rng.randint(5, 400)
        t = Table(n_t, rng.randint(0, n_t), n_c, rng.randint(0, n_c), rng.choice([0.52, 0.54, 0.56, 0.58, 0.6]))
        assert t.x == pytest.approx(loglik_ratio_binomial(t), abs=1e-8)


def test_sample_table_hits_target_and_constraints():
    rng = random.Random(0)
    for target in [-2.9, -1.0, 0.4, 1.7, 2.95]:
        for q in Q_VALUES:
            for conflict in [False, True]:
                t = sample_table(target, q, conflict, rng)
                assert abs(t.x - target) <= t.L / 2 + 1e-9
                assert t.heuristic_conflict == conflict
                assert 0.2 <= t.rate_t <= 0.8 and 0.2 <= t.rate_c <= 0.8 and t.rate_agrees
                assert 10 <= t.n_t <= 30 and 10 <= t.n_c <= 30


def test_heuristic_conflict_example():
    # 120/260 (46%) vs 100/190 (53%): more raw successes, lower rate
    t = Table(260, 120, 190, 100, 0.54)
    assert t.heuristic_conflict


def test_null_table():
    t = sample_null_table(0.56, random.Random(1))
    assert t.D == 0 and t.x == 0


def test_trial_x_direction():
    t = Table(10, 4, 10, 6, 0.6)
    assert trial_x(t, -1) == pytest.approx(-t.x)


def test_sample_tables_coverage():
    tabs = sample_tables(400, random.Random(3))
    xs = [t.x for t in tabs]
    assert min(xs) < -2.5 and max(xs) > 2.5
    frac_conf = sum(t.heuristic_conflict for t in tabs) / len(tabs)
    assert 0.35 < frac_conf < 0.6


# --- evidence direction must match the visible rate comparison (bug found in pilot 1) ---

def test_running_example_direction():
    # 4/10 vs 6/10: program group does worse, evidence favors H0 -> agrees
    assert Table(10, 4, 10, 6, 0.6).rate_agrees


def test_pilot_bug_example_is_rejected():
    # treated 178/372 = 48% beats control 82/177 = 46%, yet x < 0: must be flagged
    t = Table(372, 178, 177, 82, 0.58)
    assert t.x < 0 and not t.rate_agrees


@pytest.mark.parametrize("seed", range(5))
def test_every_sampled_table_agrees(seed):
    rng = random.Random(seed)
    for target in [-2.9, -1.7, -0.4, -0.15, 0.15, 0.6, 1.9, 2.95]:
        for q in Q_VALUES:
            for conflict in [False, True]:
                try:
                    t = sample_table(target, q, conflict, rng)
                except RuntimeError:
                    continue
                assert t.rate_agrees, t
                assert t.D != 0
                assert 0.2 <= t.rate_t <= 0.8 and 0.2 <= t.rate_c <= 0.8
                if t.heuristic_conflict != conflict:
                    raise AssertionError(t)


def test_conflict_tables_available_across_x_range():
    rng = random.Random(0)
    for target in [-2.9, -1.5, -0.3, 0.3, 1.5, 2.9]:
        for q in (min(Q_VALUES), max(Q_VALUES)):
            t = sample_table(target, q, True, rng)
            assert t.heuristic_conflict and t.rate_agrees


def test_null_tables_are_identical_groups():
    rng = random.Random(4)
    for _ in range(50):
        t = sample_null_table(0.54, rng)
        assert t.x == 0 and t.n_t == t.n_c and t.k_t == t.k_c and t.rate_agrees


def test_small_target_never_becomes_null():
    t = sample_table(0.15, 0.58, False, random.Random(1))  # 0.15 / L rounds to 0 for q=0.58
    assert t.D != 0


def test_design_pools_validate():
    from idea.cues import load_cues
    from idea.design import DesignConfig, build_items
    from idea.prompts import load_topics
    items = build_items(load_topics(), DesignConfig(items_per_topic_dir=30), random.Random(0))
    pool = items.drop_duplicates("base_slot")
    tabs = [Table(int(r.n_t), int(r.k_t), int(r.n_c), int(r.k_c), float(r.q)) for r in pool.itertuples()]
    assert all(t.rate_agrees for t in tabs)
    nonnull = [t for t in tabs if t.D != 0]
    xs = [t.x for t in nonnull]
    assert min(xs) < -2.5 and max(xs) > 2.5
    frac = sum(t.heuristic_conflict for t in nonnull) / len(nonnull)
    assert 0.45 <= frac <= 0.55, frac
