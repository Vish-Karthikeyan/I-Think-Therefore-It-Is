"""Belief-strength experiment (exploratory, 2026-11-16): the hedge rule and that original prompts are unchanged."""
import pandas as pd
import pytest

from idea.models import MODELS
from idea.pipeline import _cues, _topics, load_config, run_dir
from idea.prompts import BELIEF_HEDGES, hedge_belief, load_topics
from idea.runner import make_request

TOPICS = load_topics()


def test_every_belief_sentence_has_the_original_hedge():
    sents = [s for t in TOPICS for s in (t.belief_pos, t.belief_neg)]
    assert len(sents) == 2 * len(TOPICS)
    assert all(s.startswith("I think ") and s.endswith(".") for s in sents)


@pytest.mark.parametrize("level", sorted(BELIEF_HEDGES))
def test_hedge_swaps_only_the_prefix(level):
    for t in TOPICS:
        for d in (1, -1):
            orig = t.belief(d)
            h = hedge_belief(orig, level)
            assert h.startswith(BELIEF_HEDGES[level])
            assert h[len(BELIEF_HEDGES[level]):] == orig[len("I think "):]
            if level == 0:
                assert h is orig


def test_hedge_rejects_unexpected_sentence():
    with pytest.raises(ValueError):
        hedge_belief("We believe the policy helped.", 1)


def test_aliases_exist_and_differ_only_in_strength():
    for b in ("olmo_instruct", "llama", "tulu_final", "phi4", "gpt"):
        for v in (0, 1, 2):
            a = MODELS[f"{b}_bs{v}"]
            assert a.belief_strength == v and a.formats == ("chat",)
            assert (a.hf_id, a.revision, a.openrouter_id, a.readout) == (MODELS[b].hf_id, MODELS[b].revision,
                                                                       MODELS[b].openrouter_id, MODELS[b].readout)


def _some_belief_rows(n=30):
    cfg = load_config("configs/main.yaml")
    path = run_dir(cfg) / "trials.parquet"
    if not path.exists():   # run folders are not part of the public release
        pytest.skip(f"needs the main run's trial table ({path})")
    trials = pd.read_parquet(path)
    return cfg, trials[trials.b != 0].head(n)


@pytest.mark.parametrize("level", [1, 2])
def test_hedged_prompt_equals_original_with_sentence_swapped(level):
    cfg, rows = _some_belief_rows()
    tmap = {t.topic_id: t for t in _topics(cfg)}
    lib = _cues(cfg)
    for row in rows.itertuples(index=False):
        for call in ("prior", "post"):
            o = make_request(row, tmap, lib, "chat", call)
            h = make_request(row, tmap, lib, "chat", call, belief_strength=level)
            orig_s = tmap[row.topic].belief(int(row.belief_dir))
            hedged_s = hedge_belief(orig_s, level)
            assert h.messages[0]["content"].count(hedged_s) == 1
            assert h.messages[0]["content"].replace(hedged_s, orig_s) == o.messages[0]["content"]
            assert h.meta["span"] == hedged_s and h.meta["belief_strength"] == level


def test_level_zero_request_identical_to_original():
    cfg, rows = _some_belief_rows()
    tmap = {t.topic_id: t for t in _topics(cfg)}
    lib = _cues(cfg)
    for row in rows.itertuples(index=False):
        for call in ("prior", "prior_rep", "post"):
            a = make_request(row, tmap, lib, "chat", call)
            b = make_request(row, tmap, lib, "chat", call, belief_strength=0)
            assert a.prompt_repr() == b.prompt_repr() and a.meta == b.meta


def test_third_party_levels_swap_only_the_speaker():
    from idea.prompts import THIRD_PARTY, THIRD_PARTY_NAMES, third_party_name
    for lvl, tmpl in THIRD_PARTY.items():
        for t in TOPICS:
            for d in (1, -1):
                key = f"{t.topic_id}:{d}"
                h = hedge_belief(t.belief(d), lvl, key=key)
                prefix = tmpl.format(name=third_party_name(key))
                assert h == prefix + t.belief(d)[len("I think "):]
                assert third_party_name(key) in THIRD_PARTY_NAMES
    with pytest.raises(ValueError):
        hedge_belief(TOPICS[0].belief(1), 6)          # a third-party level needs a key


def test_third_party_names_are_balanced_over_trials():
    from idea.prompts import THIRD_PARTY_NAMES, third_party_name
    cfg, rows = _some_belief_rows(n=4000)
    counts = pd.Series([third_party_name(t) for t in rows.trial_id]).value_counts()
    assert set(counts.index) == set(THIRD_PARTY_NAMES)
    assert counts.min() / counts.max() > 0.9


def test_third_party_request_uses_the_trial_id_name():
    from idea.prompts import third_party_name
    cfg, rows = _some_belief_rows(n=10)
    tmap = {t.topic_id: t for t in _topics(cfg)}
    lib = _cues(cfg)
    for row in rows.itertuples(index=False):
        h = make_request(row, tmap, lib, "chat", "post", belief_strength=6)
        name = third_party_name(row.trial_id)
        assert f"{name} thinks " in h.messages[0]["content"] and "I think " not in h.messages[0]["content"]
