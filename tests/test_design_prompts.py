import numpy as np
import pytest

from idea.cues import ALL_GROUPS, UNSPECIFIED, load_cues
from idea.design import DesignConfig, build_trials, x_range
from idea.prompts import TrialSlots, chat_trial, completion_trial, load_topics, stance_prompt
from idea.tables import Table


@pytest.fixture(scope="module")
def lib():
    return load_cues()


@pytest.fixture(scope="module")
def topics():
    return load_topics()


@pytest.fixture(scope="module")
def design(topics, lib):
    return build_trials(topics, lib, DesignConfig(seed=3, scale=0.05, items_per_topic_dir=10))


def test_every_group_has_label_and_vignettes(lib):
    for g in ALL_GROUPS + [UNSPECIFIED]:
        assert lib.select(group_id=g, cue_type="label")
        assert len(lib.select(group_id=g, cue_type="vignette")) == 4


def test_topics_cover_dimensions(topics):
    tier2 = [t for t in topics if t.tier == 2]
    for k in ("political", "race", "sexuality", "gender", "religion", "geography", "age"):
        assert sum(k in t.dims_engaged for t in tier2) >= 3
    assert any(t.tier == 1 for t in topics)


def test_context_a_b_differ_only_by_results(topics, lib):
    t = topics[0]
    tab = Table(200, 110, 190, 100, 0.54)
    a = TrialSlots(lib.by_id("catholic_label").text, t, "lab", 1, 0.54, 1, None)
    b = TrialSlots(**{**a.__dict__, "table": tab})
    ta, tb = chat_trial(a)[0]["content"], chat_trial(b)[0]["content"]
    assert "Results" not in ta and "Results" in tb
    blocks_b = [x for x in tb.split("\n\n") if not x.startswith("Results")]
    assert blocks_b == ta.split("\n\n")
    assert completion_trial(b).rstrip().endswith("Probability (0-100):")


def test_belief_and_hypothesis_directions(topics, lib):
    t = topics[0]
    s = TrialSlots("I'm Catholic.", t, "tank", -1, 0.56, -1, None)  # H1 = claim_neg, belief = claim_neg
    txt = chat_trial(s)[0]["content"]
    assert f"H1: {t.claim_neg}" in txt and t.belief_neg in txt
    assert "{Source}" not in txt and "A policy think tank" in txt


def test_design_invariants(design):
    items, tr = design
    assert set(tr.e.unique()) <= {"neutral", "contra", "support", "null"}
    assert (tr[tr.e == "contra"].b * tr[tr.e == "contra"].d < 0).all()
    assert (tr.belief_dir == tr.b * tr.h_dir).all()
    # x independent of belief within tier: means of x close across b
    m = tr[tr.D != 0].groupby("b").x.mean()
    assert m.abs().max() < 0.5
    # Tier 1 and Tier 2 share the same table pool
    t1 = items[items.tier == 1][["n_t", "k_t", "n_c", "k_c"]].drop_duplicates()
    t2 = items[items.tier == 2][["n_t", "k_t", "n_c", "k_c"]].drop_duplicates()
    assert len(t1.merge(t2)) == len(t1)


def test_x_range_ceiling():
    lo, hi = x_range(2.0)
    assert hi == pytest.approx(np.log(0.97 / 0.03) - 2.0)
    assert lo == -3.0


def test_stance_prompt_has_no_study(topics):
    p = stance_prompt("I'm Jewish.", topics[0], 1)[0]["content"]
    assert "Results" not in p and "H1" not in p and topics[0].stance_pos in p
