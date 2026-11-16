import json

import pandas as pd

from idea.cues import UNSPECIFIED
from idea.pipeline import mark_tier1_low_g
from idea.stance import screen_topics, tier1_low_g_pairs


def _stance():
    rows = []
    for model in ("a", "b"):
        rows += [
            dict(model=model, tier=1, topic="t1", cue_id="c_low", group="g1", dimension="political", h_dir=1, g=0.1),
            dict(model=model, tier=1, topic="t1", cue_id="c_high", group="g2", dimension="political", h_dir=1, g=2.0),
            dict(model=model, tier=1, topic="t1", cue_id="u", group=UNSPECIFIED, dimension="none", h_dir=1, g=0.0),
            dict(model=model, tier=2, topic="t2", cue_id="c_high", group="g2", dimension="political", h_dir=1, g=2.0),
        ]
    return pd.DataFrame(rows)


def test_tier1_topics_always_kept():
    rep, _ = screen_topics(_stance())
    r = rep[rep.tier == 1].iloc[0]
    assert bool(r.keep) and not bool(r.tier1_rule_pass)


def test_low_g_pairs_and_marking(tmp_path):
    low = tier1_low_g_pairs(_stance())
    assert low == [["t1", "c_low"]]
    (tmp_path / "screen.json").write_text(json.dumps(dict(keep=[], engaged={}, tier1_low_g=low)))
    trials = pd.DataFrame(dict(tier=[1, 1, 1, 2], topic=["t1", "t1", "t1", "t2"],
                               cue_id=["c_low", "c_high", "u", "c_high"], group=["g1", "g2", UNSPECIFIED, "g2"]))
    assert mark_tier1_low_g(trials, str(tmp_path)).tolist() == [True, False, True, True]
    assert mark_tier1_low_g(trials, None).all()


def test_cache_path_per_piece(tmp_path):
    from idea.pipeline import cache_path
    assert cache_path(tmp_path, "llama").name == "llama.sqlite"
    assert cache_path(tmp_path, "llama", "3/8").name == "llama_s3of8.sqlite"
