"""Per-run spending cap (approved 2026-11-10: $15 per run). No network."""
import dataclasses

import pandas as pd
import pytest

from idea.cues import load_cues
from idea.elicit import BudgetExceeded, Cache, CachedElicitor, OpenRouterBackend, Request
from idea.models import get_model
from idea.prompts import load_topics
from idea.runner import calls_for, make_request

SONNET = {"prompt": 2e-6, "completion": 1e-5}     # $/token, OpenRouter list prices 2026-11-10
GPT4O = {"prompt": 2.5e-6, "completion": 1e-5}


def _test_requests():
    tr = pd.read_parquet("tests/fixtures/evtest_trials.parquet")
    tr = tr[tr.set == "wider"].head(20).assign(complement=False, calibration=False)
    T = {t.topic_id: t for t in load_topics()}
    L = load_cues()
    reqs, seen = [], set()
    for row in tr.itertuples(index=False):
        for call in calls_for(row, False):
            r = make_request(row, T, L, "chat", call)
            if r.prompt_repr() not in seen:
                seen.add(r.prompt_repr()); reqs.append(r)
    return reqs


def test_estimate_matches_measured_costs():
    reqs = _test_requests()
    sonnet = OpenRouterBackend(dataclasses.replace(get_model("claude"), openrouter_id="x"), api_key="t",
                               sample_k=5, pricing=SONNET)
    gpt = OpenRouterBackend(get_model("gpt"), api_key="t", pricing=GPT4O)
    # measured: Sonnet 5 tool format $0.595 for these requests x5 samples; GPT json $0.034
    assert 0.595 <= sonnet.estimate_cost(reqs) <= 0.595 * 1.8
    assert 0.034 <= gpt.estimate_cost(reqs) <= 0.034 * 1.8


class Fake(OpenRouterBackend):
    def __init__(self, cap):
        super().__init__(dataclasses.replace(get_model("gpt"), answer_format="text"), api_key="t",
                         max_cost=cap, pricing=GPT4O, concurrency=1)
        self.calls = 0

    async def _post(self, client, body):
        # mimic the real method's cap check, then pretend each call costs $1
        if self.usage["cost"] >= self.max_cost:
            raise BudgetExceeded("cap")
        self.calls += 1
        self.usage["cost"] += 1.0
        import math
        return {"model": "m", "choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "50", "logprob": math.log(0.99)}]}]}, "message": {"content": "50"}}]}


def req(i):
    return Request("number", "chat", messages=[{"role": "user", "content": f"question {i}"}])


def test_refuses_to_start_when_estimate_exceeds_cap(tmp_path):
    b = Fake(cap=1e-9)
    el = CachedElicitor(b, Cache(tmp_path / "c.sqlite"), progress=False)
    with pytest.raises(BudgetExceeded):
        el.run([req(i) for i in range(5)])
    assert b.calls == 0


def test_stops_mid_run_when_spend_reaches_cap(tmp_path):
    b = Fake(cap=3.0)
    b.estimate_cost = lambda reqs: 0.0          # estimate too optimistic: the hard stop must catch it
    el = CachedElicitor(b, Cache(tmp_path / "c.sqlite"), chunk=1, progress=False)
    with pytest.raises(BudgetExceeded):
        el.run([req(i) for i in range(10)])
    assert b.calls == 3
    assert len(el.cache.get_many([el.key(req(i)) for i in range(10)])) == 3   # paid answers kept
