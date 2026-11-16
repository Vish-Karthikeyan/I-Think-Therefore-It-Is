"""Exploratory additions (2026-11-11): prompt variants, system prompt, subset, diagnostic."""
import hashlib

import numpy as np
import pandas as pd

from idea.diagnostics import KINDS, build_requests, score, select_items
from idea.models import DEBIAS_SYSTEM_PROMPT, MODELS, get_model
from idea.prompts import QUESTIONS, TrialSlots, chat_trial, load_topics
from idea.tables import Table


def _slots():
    t = load_topics()[0]
    return t, TrialSlots(cue_text="I am a person.", topic=t, source="lab", h_dir=1, q=0.6, belief_dir=1,
                         table=Table(20, 12, 20, 8, 0.6))


def test_default_chat_trial_unchanged():
    _, s = _slots()
    a = chat_trial(s, 0, "H1")
    assert len(a) == 1 and a[0]["role"] == "user" and a[0]["content"].endswith(QUESTIONS[(0, "H1")])
    assert chat_trial(s, 0, "H1", 0, None) == a


def test_variants_and_system_prompt():
    _, s = _slots()
    base = chat_trial(s, 0, "H1")[0]["content"]
    for v in (1, 2):
        c = chat_trial(s, 0, "H1", v)[0]["content"]
        assert c != base and "Results" in c and "H1" in c
    sp = chat_trial(s, 0, "H1", 0, DEBIAS_SYSTEM_PROMPT)
    assert sp[0] == {"role": "system", "content": DEBIAS_SYSTEM_PROMPT} and sp[1]["content"] == base


def test_aliases_share_weights_and_are_chat_only():
    for k in ("olmo_instruct_sp", "llama_sp", "olmo_instruct_pv1", "phi4_pv2"):
        m = get_model(k)
        base = MODELS[k.rsplit("_", 1)[0]]
        assert m.hf_id == base.hf_id and m.revision == base.revision and m.formats == ("chat",)
    assert get_model("olmo_instruct").system_prompt is None and get_model("olmo_instruct").prompt_variant == 0


def test_subset_hash_is_fixed_third():
    ids = pd.Series([f"t{i:06d}" for i in range(3000)])
    h = ids.map(lambda t: int(hashlib.md5(t.encode()).hexdigest(), 16) % 3)
    assert 900 < (h == 0).sum() < 1100
    assert (h == ids.map(lambda t: int(hashlib.md5(t.encode()).hexdigest(), 16) % 3)).all()


def _trials():
    topics = load_topics()
    rows = []
    rng = np.random.default_rng(0)
    for t in topics[:6]:
        for j in range(10):
            n_t, n_c = 20, 25
            k_t, k_c = int(rng.integers(2, 18)), int(rng.integers(2, 23))
            x = np.log((k_t / n_t + 1e-9) / (k_c / n_c + 1e-9))
            rows.append(dict(topic=t.topic_id, tier=t.tier, h_dir=1, q=0.6, n_t=n_t, k_t=k_t, n_c=n_c, k_c=k_c,
                             x=x, source="lab", results_as_list=True, treated_first=bool(j % 2)))
    return pd.DataFrame(rows), {t.topic_id: t for t in topics}


def test_diag_requests_and_truth():
    tr, tmap = _trials()
    items = select_items(tr, n=24, seed=1)
    assert not ((items.k_t * items.n_c) == (items.k_c * items.n_t)).any()
    for fmt in ("chat", "completion"):
        reqs, index = build_requests(items, tmap, fmt)
        assert len(reqs) == len(items) * len(KINDS)
        for r, ix in zip(reqs, index):
            assert (r.kind == "choice") == (ix["kind"] in ("compare", "support"))
            body = r.messages[0]["content"] if fmt == "chat" else r.text
            assert "Results" in body
            assert ("H1" in body) == (ix["kind"] in ("support", "post"))
    # compare truth points at the group with the higher rate
    reqs, index = build_requests(items, tmap, "chat")
    for r, ix in zip(reqs, index):
        if ix["kind"] == "compare":
            it = items.set_index("diag_id").loc[ix["diag_id"]]
            t = tmap[it.topic]
            higher = t.treated if it.k_t / it.n_t > it.k_c / it.n_c else t.control
            line = [l for l in r.messages[0]["content"].splitlines() if l.startswith(f"({'AB'[int(ix['truth'])]})")][0]
            assert higher.lower() in line.lower()


def test_diag_score_perfect_model():
    tr, tmap = _trials()
    items = select_items(tr, n=24, seed=1)
    _, index = build_requests(items, tmap, "chat")
    df = pd.DataFrame(index)
    df["p"] = [ix["truth"] / 100 if ix["kind"].startswith("rate") else
               (0.8 if items.set_index("diag_id").loc[ix["diag_id"]].x > 0 else 0.2) if ix["kind"] == "post" else None
               for ix in index]
    df["probs"] = [([1.0, 0.0] if int(ix["truth"]) == 0 else [0.0, 1.0]) if ix["kind"] in ("compare", "support") else None
                   for ix in index]
    df["coverage"] = 1.0
    df["model"], df["fmt"] = "perfect", "chat"
    df = df.merge(items[["diag_id", "topic", "tier", "x", "flip"]], on="diag_id")
    s = score(df).iloc[0]
    assert s.rate_mae < 1e-6 and s.compare_acc == 1 and s.support_acc == 1 and s.post_dir_acc == 1
