"""Constrained answer formats for API models (no network): GPT structured JSON with logprobs
at the number token, Claude forced tool call. Shapes match live probes of 2026-11-10."""
import math

import pytest

from idea.elicit import (Cache, CachedElicitor, OpenRouterBackend, Request, parse_answer_json,
                         request_key, to_p)
from idea.models import get_model


def req(text, kind="number"):
    if kind == "choice":
        return Request("choice", "chat", messages=[{"role": "user", "content": text}], letters=["A", "B"])
    return Request("number", "chat", messages=[{"role": "user", "content": text}])


def test_parse_answer_json():
    assert parse_answer_json('{"probability":85}') == 85 and parse_answer_json('{"probability": 0}') == 0
    for bad in ('{"probability": 101}', '{"probability": "50"}', '{"probability": true}', "50", "", "{}"):
        assert parse_answer_json(bad) is None


class FakeGPT(OpenRouterBackend):
    def __init__(self):
        super().__init__(get_model("gpt"), api_key="t")
        self.bodies = []

    async def _post(self, client, body):
        self.bodies.append(body)
        top = [{"token": "85", "logprob": math.log(0.3)}, {"token": "70", "logprob": math.log(0.2)},
               {"token": "65", "logprob": math.log(0.1)}]
        toks = [{"token": t, "logprob": 0.0, "top_logprobs": []} for t in ('{"', "prob", "ability", '":')]
        toks.append({"token": "85", "logprob": math.log(0.3), "top_logprobs": top})
        return {"model": "openai/gpt-4o-2024-11-20", "choices": [{"message": {"content": '{"probability":85}'},
                                                                   "logprobs": {"content": toks}}]}


def test_gpt_json_reads_number_position():
    b = FakeGPT()
    r = b.run([req("q")])[0]
    expected = (0.3 * to_p(85) + 0.2 * to_p(70) + 0.1 * to_p(65)) / 0.6
    assert r.coverage == 1.0 and abs(r.p - expected) < 1e-12 and abs(r.raw["top20_mass"] - 0.6) < 1e-12
    body = b.bodies[0]
    assert body["response_format"]["json_schema"]["schema"]["properties"]["probability"]["type"] == "integer"
    assert len(b.bodies) == 1                         # valid answer: no re-prompt
    assert not b.needs_retry(r)


class FakeClaude(OpenRouterBackend):
    def __init__(self, answers):
        super().__init__(get_model("claude"), api_key="t", sample_k=len(answers))
        self.answers = list(answers)
        self.bodies = []

    async def _post(self, client, body):
        self.bodies.append(body)
        a = self.answers.pop(0)
        return {"model": "anthropic/claude-sonnet-5", "choices": [{"message": {
            "content": None, "tool_calls": [{"function": {"name": "answer", "arguments": a}}]}}]}


def test_claude_tool_sampling():
    b = FakeClaude(['{"probability": 13}', '{"probability": 73}', '{"probability": 40}', "oops", '{"probability": 50}'])
    r = b.run([req("q")])[0]
    assert r.samples == [13, 73, 40, None, 50] and r.coverage == 0.8
    assert abs(r.p - to_p([13, 73, 40, 50]).mean()) < 1e-12
    assert all(x["tool_choice"]["function"]["name"] == "answer" for x in b.bodies)
    assert not b.needs_retry(r)                       # tool answers are not re-read as text


def test_keys_separate_formats_only_for_numbers():
    gpt = get_model("gpt")
    import dataclasses
    text_gpt = dataclasses.replace(gpt, answer_format="text")
    n, c = req("q"), req("q", "choice")
    assert request_key(n, gpt, "v", {}) != request_key(n, text_gpt, "v", {})
    assert request_key(c, gpt, "v", {}) == request_key(c, text_gpt, "v", {})
