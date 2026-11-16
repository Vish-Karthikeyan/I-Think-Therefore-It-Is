"""HF backend on a tiny random Llama-3 model (downloads ~10 MB). Skipped offline."""
import numpy as np
import pytest

pytest.importorskip("transformers")
TINY = "trl-internal-testing/tiny-LlamaForCausalLM-3.1"


@pytest.fixture(scope="module")
def backend():
    from idea.elicit import HFBackend
    from idea.models import ModelSpec
    try:
        return HFBackend(ModelSpec("tiny", TINY, formats=("chat", "completion")), batch_size=4,
                         capture_layers=[0, 1], coverage_min=0.9)
    except OSError as e:
        pytest.skip(f"cannot download {TINY}: {e}")


def _reqs():
    from idea.elicit import Request
    msgs = [{"role": "user", "content": "How likely is it that H1 is true? Answer with a single integer from 0 to 100."}]
    return [Request("number", "chat", messages=msgs, capture=True),
            Request("number", "completion", text="Question.\nProbability (0-100):", capture=True),
            Request("choice", "chat", messages=[{"role": "user", "content": "(A) x (B) y. Letter only."}],
                    letters=["A", "B", "C"]),
            Request("number", "chat", messages=msgs, readout="sample")]


def test_integer_tokens_single(backend):
    assert len(backend.num_ids) == 101
    assert all(len(ids) >= 1 for ids in backend.num_ids)


def test_readouts_and_capture(backend):
    res = backend.run(_reqs())
    for r in res[:2]:
        assert r.p is None or 0.005 <= r.p <= 0.995
        assert 0 <= r.coverage <= 1
    assert res[2].probs is None or abs(sum(res[2].probs) - 1) < 1e-6
    assert res[3].readout == "sample" and len(res[3].samples) == backend.sample_k
    acts = backend.last_activations
    assert set(acts) == {0, 1} and set(acts[0]) == {0, 1}
    assert acts[0][1].shape == (backend.hidden_size,)


def test_left_padding_invariance(backend):
    """Batched (left-padded) results must match single-prompt results."""
    from idea.elicit import Request
    short = Request("number", "completion", text="P:")
    long = Request("number", "completion", text="A much longer prompt with many more tokens in it. P:")
    batched = backend.run([short, long])
    single = backend.run([short])
    assert batched[0].coverage == pytest.approx(single[0].coverage, rel=1e-3, abs=1e-6)


def test_steering_changes_output_and_cache_key(backend):
    from idea.elicit import Request
    r = Request("number", "completion", text="Question.\nProbability (0-100):", capture=True)
    s0 = dict(backend.settings)
    base = backend.run([r])[0]
    h0 = backend.last_activations[0][1].astype(np.float32)
    d = np.random.default_rng(0).normal(size=backend.hidden_size)
    backend.set_steering([(0, d, 50.0, "all")])
    assert backend.settings != s0
    steered = backend.run([r])[0]
    h1 = backend.last_activations[0][1].astype(np.float32)
    backend.set_steering([])
    assert not np.allclose(h0, h1)  # layer-1 activations change when layer 0 is steered
    assert backend.settings.get("steer", "") == ""
