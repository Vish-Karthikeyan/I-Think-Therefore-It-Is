"""Independent check: read the counts and hypothesized rates back out of every rendered
posterior prompt, recompute the log-likelihood ratio from the text alone, and compare with
the stored x. Also checks that the evidence direction agrees with the visible rate
comparison. (Added after the 2026-11-10 pilot found tables whose evidence pointed against
their visible rates.)"""
import math
import re

import pytest

from idea.cues import load_cues
from idea.design import DesignConfig, build_trials
from idea.prompts import load_topics
from idea.runner import make_request


def parse_prompt(txt, topic):
    h1 = re.search(r"H1: .*?\. (\d+)% of (.+?) " + re.escape(topic.outcome_present) + r", versus (\d+)% of (.+?)\.", txt)
    first_pct, first_grp, second_pct = int(h1.group(1)), h1.group(2), int(h1.group(3))
    h1_treated = first_pct if first_grp == topic.treated else second_pct
    res = txt.split("Results")[1].split("\n\n")[0]
    counts = {}
    for g in (topic.treated, topic.control):
        m = (re.search(r"(?m)^\s+" + re.escape(g[:1].upper() + g[1:]) + r":\s+(\d+) of (\d+)", res)
             or re.search(r"(?:among|;) " + re.escape(g.lower()) + r", (\d+) of (\d+)", res))
        counts[g] = (int(m.group(1)), int(m.group(2)))
    return h1_treated, counts[topic.treated], counts[topic.control]


@pytest.fixture(scope="module")
def trials():
    topics = load_topics()
    _, tr = build_trials(topics, load_cues(), DesignConfig(seed=11, scale=0.08, items_per_topic_dir=12))
    return topics, tr


def test_x_recomputed_from_prompt_text(trials):
    topics, tr = trials
    T = {t.topic_id: t for t in topics}
    L = load_cues()
    for row in tr.itertuples(index=False):
        t = T[row.topic]
        txt = make_request(row, T, L, "chat", "post").messages[0]["content"]
        h1_treated, (kt, nt), (kc, nc) = parse_prompt(txt, t)
        q = h1_treated / 100
        a = math.log(q / (1 - q))
        x_text = a * ((2 * kt - nt) - (2 * kc - nc))
        assert abs(x_text - row.x) < 1e-6, (row.trial_id, row.x, x_text)
        if row.D != 0:
            h1_says_treated_higher = h1_treated > 50
            treated_higher = kt / nt > kc / nc
            assert (treated_higher == h1_says_treated_higher) == (row.x > 0), row.trial_id
