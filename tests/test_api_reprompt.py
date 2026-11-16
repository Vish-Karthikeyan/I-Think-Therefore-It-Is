"""API re-prompt (free-text answer format): a low-coverage logprob answer is re-asked once with API_REPROMPT_SUFFIX,
and cached low-coverage results from before the re-prompt existed are re-run. No network."""
import math

import pytest

from idea.elicit import API_REPROMPT_SUFFIX, Cache, CachedElicitor, OpenRouterBackend, Request
from idea.models import get_model


def _resp(tokens):
    return {"model": "openai/gpt-4o-2024-11-20", "choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": t, "logprob": math.log(p)} for t, p in tokens]}]}, "message": {"content": tokens[0][0]}}]}


class FakeBackend(OpenRouterBackend):
    def __init__(self):
        import dataclasses
        super().__init__(dataclasses.replace(get_model("gpt"), answer_format="text"), api_key="test", coverage_min=0.9)
        self.sent = []

    async def _post(self, client, body):
        content = body["messages"][-1]["content"]
        self.sent.append(content)
        if content.endswith(API_REPROMPT_SUFFIX):
            return _resp([("70", 0.95), ("60", 0.03)])
        if "easy" in content:
            return _resp([("40", 0.97)])
        return _resp([("To", 0.8), ("62", 0.1)])  # starts reasoning: coverage 0.1


def req(text):
    return Request("number", "chat", messages=[{"role": "user", "content": text}])


def test_low_coverage_is_reprompted_once():
    b = FakeBackend()
    hard, easy = b.run([req("hard question"), req("easy question")])
    assert hard.attempt == 1 and hard.coverage > 0.9 and abs(hard.p - (70.5 / 101 * 0.95 + 60.5 / 101 * 0.03) / 0.98) < 1e-9
    assert easy.attempt == 0 and easy.coverage > 0.9
    assert sum(s.endswith(API_REPROMPT_SUFFIX) for s in b.sent) == 1   # only the hard one
    assert len(b.sent) == 3


def test_cached_low_coverage_results_are_rerun(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    b = FakeBackend()
    el = CachedElicitor(b, cache, progress=False)
    r = req("hard question")
    # simulate a result cached before the re-prompt existed
    from idea.elicit import Result
    cache.put_many({el.key(r): Result(0.6, 0.1, "dist", attempt=0)}, "gpt")
    out = el.run([r])[0]
    assert out.attempt == 1 and out.coverage > 0.9
    b.sent.clear()
    again = el.run([r])[0]           # now cached with attempt=1: no new calls
    assert again.attempt == 1 and b.sent == []


def test_parse_int_requires_leading_number():
    from idea.elicit import parse_int
    assert parse_int("57") == 57 and parse_int(" 62\n\nBecause...") == 62 and parse_int("**40**") == 40
    assert parse_int("50%") == 50 and parse_int("100") == 100
    assert parse_int("Non-participants: 92/180") is None
    assert parse_int("Looking at the data: 45 of 100") is None
    assert parse_int("92/180") is None and parse_int("0.62") is None and parse_int("101") is None


class FakeSampler(OpenRouterBackend):
    """A sampled model that uses the suffix re-prompt."""
    def __init__(self):
        import dataclasses
        super().__init__(dataclasses.replace(get_model("claude"), reprompt="suffix", answer_format="text"), api_key="test", sample_k=4)
        self.sent = []

    async def _post(self, client, body):
        content = body["messages"][-1]["content"]
        self.sent.append(content)
        txt = "35" if content.endswith(API_REPROMPT_SUFFIX) else "I need to calculate the likelihood"
        return {"model": "anthropic/claude-sonnet-5", "choices": [{"message": {"content": txt}}]}


def test_sampling_reprompt_when_most_samples_unparsed():
    b = FakeSampler()
    r = b.run([req("hard question")])[0]
    assert r.attempt == 1 and r.coverage == 1.0 and r.samples == [35] * 4
    assert len(b.sent) == 8


def test_cached_lenient_parse_is_redone():
    from idea.elicit import Result
    b = FakeSampler()
    old = Result(0.9, 1.0, "sample", samples=[92, 50], raw=["Non-participants: 92/180", "50"])
    assert b.needs_retry(old)
    good = Result(0.5, 1.0, "sample", samples=[50, 62], raw=["50", "62\n\nBecause"])
    assert not b.needs_retry(good)


class FakePrefill(OpenRouterBackend):
    """Claude-like: reasons unless its reply is prefilled with the answer prefix."""
    def __init__(self):
        import dataclasses
        super().__init__(dataclasses.replace(get_model("claude"), reprompt="prefill", answer_format="text"), api_key="test", sample_k=3)
        self.bodies = []

    async def _post(self, client, body):
        self.bodies.append(body)
        last = body["messages"][-1]
        txt = " 41" if last["role"] == "assistant" else "Looking at the data:"
        return {"model": "anthropic/claude-sonnet-5", "choices": [{"message": {"content": txt}}]}


def test_claude_reprompt_prefills_the_answer():
    from idea.prompts import ANSWER_PREFILL
    b = FakePrefill()
    r = b.run([req("hard question")])[0]
    assert r.attempt == 1 and r.samples == [41, 41, 41]
    re_bodies = [x for x in b.bodies if x["messages"][-1]["role"] == "assistant"]
    assert len(re_bodies) == 3
    assert re_bodies[0]["messages"][-1]["content"] == ANSWER_PREFILL.rstrip()
    assert re_bodies[0]["messages"][0]["content"] == "hard question"   # question text unchanged


def test_in_flight_402_waits_and_retries(monkeypatch):
    """OpenRouter's low-balance 'in_flight_budget_exhausted' refusal is retried after Retry-After;
    a plain 'insufficient credits' 402 still stops the run."""
    import asyncio
    import httpx

    sleeps = []

    async def fake_sleep(t):
        sleeps.append(t)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    responses = [httpx.Response(402, headers={"retry-after": "2"}, text='{"error":{"metadata":{"reason":"in_flight_budget_exhausted"}}}'),
                 httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "50"}}], "usage": {"cost": 0.01}})]

    class Client:
        async def post(self, url, json=None, headers=None):
            r = responses.pop(0); r.request = httpx.Request("POST", url); return r

    b = OpenRouterBackend(get_model("gpt"), api_key="t")
    out = asyncio.run(b._post(Client(), {"messages": []}))
    assert out["choices"][0]["message"]["content"] == "50" and len(sleeps) == 1 and sleeps[0] >= 2

    responses[:] = [httpx.Response(402, text='{"error":{"message":"Insufficient credits"}}')]
    with pytest.raises(RuntimeError, match="402"):
        asyncio.run(b._post(Client(), {"messages": []}))
