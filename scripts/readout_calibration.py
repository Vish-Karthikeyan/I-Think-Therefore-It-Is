#!/usr/bin/env python
"""Readout calibration (reviewer question on GPT-4o's structured, top-20 readout).

    .venv/bin/python scripts/readout_calibration.py --model olmo_instruct --subset 0/3          # GPU
    .venv/bin/python scripts/readout_calibration.py --model llama --subset 0/3 --dry-run        # count only
    .venv/bin/python scripts/readout_calibration.py --model olmo1b_test --trials-from olmo_instruct \
        --limit 4 --dtype float32 --out /tmp/x.parquet                                         # CPU test

On the SAME trials and the SAME open model, read each prior and posterior answer several ways
(docs/READOUT_CALIBRATION_PLAN.md):

  A    current open-model readout, exactly as HFBackend.run does it: plain chat prompt; if integer
       coverage < 0.9, a second pass with the answer prefix "Probability (0-100): "; the pass with
       the higher coverage is kept; p = sum_k pi(k) (k+0.5)/101 over the integers 0..100,
       renormalized.
  B<k> the same forward pass as A, truncated to the top-k tokens of the FULL vocabulary and
       renormalized over the integers among them (k = 20 is GPT-4o's top_logprobs; isolates
       truncation). B1 = the argmax integer (greedy answer).
  C    structured readout: the same user message, plus a system message giving the JSON schema
       that OpenAI's structured output enforces; the reply is prefilled with '{"probability": '
       and the next-token distribution over integers is read (full: isolates format).
  C<k> C truncated to the top-k tokens and renormalized over the integers among them
       (C20 = GPT-4o's pipeline, apart from server-side constrained decoding).

Uses the project's own HF backend (rendering, tokenizer handling, integer token ids, forward
pass) by import only; nothing under src/ is modified. Trials are the released rows of
datasets/main_trials.csv.gz for the model (same prompts, vignettes and tables as the main run).

Output: one row per trial with p_<call>_<readout> for call in (prior, post), plus coverage,
top-k integer mass (the analogue of GPT's raw["top20_mass"]) and the number of integers in the
top 20. A per-prompt checkpoint (<out>.prompts.parquet) makes reruns resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from idea.cues import load_cues  # noqa: E402
from idea.elicit import ANSWER_SCHEMA, Request, to_p  # noqa: E402
from idea.models import get_model  # noqa: E402
from idea.prompts import load_topics  # noqa: E402
from idea.runner import make_request  # noqa: E402

COVERAGE_MIN = 0.9          # HFBackend default (configs/main.yaml elicit.coverage_min)
TOPK = (1, 5, 10, 20)       # 20 = GPT-4o's top_logprobs
CALLS = ("prior", "post")   # the two answers Eq. 3 uses (prior_rep only enters the exclusion rule)

# Readout C. The schema is the one sent to OpenAI (elicit.ANSWER_SCHEMA); OpenAI passes it to the
# model out of band, so it goes in a system message here and the user message is left unchanged.
JSON_SYSTEM = ("Respond only with a JSON object that matches this JSON schema: "
               + json.dumps(ANSWER_SCHEMA, separators=(",", ":")))
JSON_PREFIX = '{"probability": '   # GPT-4o itself writes '{"probability":50}'; see --json-prefix


# ---------------------------------------------------------------------------
# Readout arithmetic (pure numpy; tested without a model)
# ---------------------------------------------------------------------------

def integer_mass(probs_row: np.ndarray, num_ids: list[list[int]], keep: np.ndarray | None = None) -> np.ndarray:
    """Mass per integer 0..100 (summing its token variants), optionally only over token ids in `keep`."""
    if keep is None:
        return np.array([probs_row[ids].sum() for ids in num_ids])
    ks = set(int(i) for i in keep)
    return np.array([sum(probs_row[i] for i in ids if i in ks) for ids in num_ids])


def p_from_mass(mass: np.ndarray) -> float | None:
    tot = float(mass.sum())
    return float((mass / tot * to_p(np.arange(101))).sum()) if tot > 0 else None


def readouts(probs_row: np.ndarray, num_ids: list[list[int]], tag: str, topk=TOPK) -> dict:
    """p under the full distribution and under top-k truncation of the full vocabulary."""
    full = integer_mass(probs_row, num_ids)
    out = {f"p_{tag}": p_from_mass(full), f"cov_{tag}": float(full.sum()),
           f"argmax_{tag}": int(full.argmax())}
    kmax = max(topk)
    top = np.argpartition(-probs_row, kmax)[:kmax]
    top = top[np.argsort(-probs_row[top])]          # sorted, most likely first
    for k in topk:
        m = integer_mass(probs_row, num_ids, top[:k])
        out[f"p_{tag}{k}"] = p_from_mass(m)
        out[f"mass_{tag}{k}"] = float(m.sum())       # integer mass inside the top k (GPT: top20_mass)
        out[f"nint_{tag}{k}"] = int((m > 0).sum())   # distinct integers inside the top k
    return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def json_messages(messages: list[dict], mode: str) -> list[dict]:
    if mode == "none":
        return messages
    if mode == "user":
        msgs = [dict(m) for m in messages]
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n" + JSON_SYSTEM
        return msgs
    return [{"role": "system", "content": JSON_SYSTEM}] + list(messages)


def render_json(backend, req: Request, mode: str, prefix: str) -> str:
    msgs = json_messages(req.messages, mode)
    text = backend.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if prefix.endswith(" ") and not backend.space_is_separate:
        prefix = prefix.rstrip()    # the number token carries its own leading space (as HFBackend.render)
    return text + prefix


def prompt_hash(req: Request) -> str:
    return hashlib.sha256(req.prompt_repr().encode()).hexdigest()[:20]


def load_trials(trials_from: str, tier: int | None, subset: str | None, limit: int | None,
                shard: str | None) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "analysis"))
    from common import load_dataset
    df = load_dataset("main")
    d = df[(df.model == trials_from) & (df.fmt == "chat") & (df.e != "null")]
    if tier is not None:
        d = d[d.tier == tier]
    if subset:   # same rule as pipeline.stage_elicit's subset_by_model ("0/3" = the fixed trial-id third)
        i, n = map(int, subset.split("/"))
        d = d[d.trial_id.map(lambda t: int(hashlib.md5(t.encode()).hexdigest(), 16) % n) == i]
    d = d.sort_values("trial_id").reset_index(drop=True)
    if shard:
        i, n = map(int, shard.split("/"))
        d = d[d.index % n == i]
    if limit:
        d = d.head(limit)
    if d.empty:
        raise SystemExit(f"no trials for {trials_from}")
    return d.reset_index(drop=True)


def build_requests(trials: pd.DataFrame) -> tuple[dict[str, Request], list[tuple[str, str, str]]]:
    tmap = {t.topic_id: t for t in load_topics()}
    lib = load_cues(include_draft=True)
    uniq: dict[str, Request] = {}
    index = []
    for row in trials.itertuples(index=False):
        for call in CALLS:
            r = make_request(row, tmap, lib, "chat", call)
            h = prompt_hash(r)
            uniq.setdefault(h, r)
            index.append((row.trial_id, call, h))
    return uniq, index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_prompts(backend, reqs: dict[str, Request], done: pd.DataFrame | None, ckpt: Path, batch: int,
                json_mode: str, json_prefix: str, save_every: int = 20) -> pd.DataFrame:
    have = set(done.h) if done is not None and len(done) else set()
    todo = [h for h in reqs if h not in have]
    rows = [] if done is None else done.to_dict("records")
    num_ids = backend.num_ids
    t0 = time.time()
    print(f"{len(todo)} prompts to run ({len(have)} already done), batch {batch}", flush=True)
    for s in range(0, len(todo), batch):
        hs = todo[s:s + batch]
        rq = [reqs[h] for h in hs]
        # A: attempt 0 (plain prompt), as HFBackend.run
        p0, _ = backend._forward([backend.render(r, prefill=False) for r in rq], False)
        rec = [readouts(p0[j], num_ids, "A") | {"att_A": 0} for j in range(len(hs))]
        # A: attempt 1 (answer prefix) where coverage < 0.9; keep the pass with higher coverage
        retry = [j for j, r in enumerate(rec) if r["cov_A"] < backend.coverage_min]
        if retry:
            p1, _ = backend._forward([backend.render(rq[j], prefill=True) for j in retry], False)
            for jj, j in enumerate(retry):
                r1 = readouts(p1[jj], num_ids, "A") | {"att_A": 1}
                if r1["cov_A"] > rec[j]["cov_A"]:   # as HFBackend: att = 1 only when the retry is kept
                    rec[j] = r1
                rec[j]["tried_A"] = 1
        for r in rec:
            r.setdefault("tried_A", 0)
        # C: structured JSON reply, prefilled to the number
        pc, _ = backend._forward([render_json(backend, r, json_mode, json_prefix) for r in rq], False)
        for j, h in enumerate(hs):
            rows.append({"h": h, **rec[j], **readouts(pc[j], num_ids, "C")})
        if (s // batch) % save_every == save_every - 1 or s + batch >= len(todo):
            pd.DataFrame(rows).to_parquet(ckpt)
            el = time.time() - t0
            n = min(s + batch, len(todo))
            print(f"  {n}/{len(todo)} prompts, {el / 60:.1f} min, {el / max(n, 1):.3f} s/prompt", flush=True)
    return pd.DataFrame(rows)


def to_wide(index, per_prompt: pd.DataFrame, trials: pd.DataFrame, model_key: str, version: str) -> pd.DataFrame:
    pp = per_prompt.set_index("h")
    cols = [c for c in pp.columns]
    recs: dict[str, dict] = {}
    for tid, call, h in index:
        r = recs.setdefault(tid, {"trial_id": tid})
        for c in cols:
            stem, rest = c.split("_", 1)            # e.g. "p", "A20"
            r[f"{stem}_{call}_{rest}"] = pp.at[h, c]
    out = pd.DataFrame(list(recs.values()))
    keep = ["trial_id", "p_prior", "p_post", "cov_prior", "cov_post", "att_prior", "att_post", "excluded"]
    rel = trials[[c for c in keep if c in trials]].rename(
        columns={c: f"{c}_released" for c in keep if c != "trial_id"})
    out = out.merge(rel, on="trial_id", how="left")
    out.insert(1, "model", model_key)
    out.insert(2, "trials_from", trials.model.iloc[0])
    out["model_version"] = version
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model key (olmo_instruct, llama, olmo1b_test, ...)")
    ap.add_argument("--trials-from", help="whose released trials to use (default: --model)")
    ap.add_argument("--hf-id", help="override the weights (e.g. a tiny random model for tests)")
    ap.add_argument("--tier", type=int, default=2, help="2 = primary analysis (default); 0 = both tiers")
    ap.add_argument("--subset", default="0/3", help="trial-id hash subset i/n ('' = all)")
    ap.add_argument("--shard", help="i/n split of the selected trials, for parallel jobs")
    ap.add_argument("--limit", type=int, help="first N trials only (tests)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--json-mode", choices=("system", "user", "none"), default="system")
    ap.add_argument("--json-prefix", default=JSON_PREFIX)
    ap.add_argument("--out", help="output parquet (default results/readout_calibration/<model>[_s].parquet)")
    ap.add_argument("--dry-run", action="store_true", help="count trials and unique prompts; load no model")
    a = ap.parse_args()

    trials_from = a.trials_from or a.model
    trials = load_trials(trials_from, a.tier or None, a.subset or None, a.limit, a.shard)
    reqs, index = build_requests(trials)
    n_prior = len({h for _, c, h in index if c == "prior"})
    n_post = len({h for _, c, h in index if c == "post"})
    chars = sum(len(r.messages[-1]["content"]) for r in reqs.values())
    print(f"{a.model}: {len(trials)} trials from {trials_from} (tier {a.tier or 'all'}, subset {a.subset or 'all'}"
          f"{', shard ' + a.shard if a.shard else ''}); unique prompts: {n_prior} prior + {n_post} post = {len(reqs)}; "
          f"forward passes: {len(reqs)} (A) + up to {len(reqs)} (A retry) + {len(reqs)} (C); "
          f"~{chars / len(reqs) / 4:.0f} tokens per user message")
    if a.dry_run:
        return
    from idea.elicit import HFBackend
    spec = get_model(a.model)
    if a.hf_id:   # test weights: the pinned revision belongs to the real checkpoint
        from dataclasses import replace
        spec = replace(spec, revision="main")
    backend = HFBackend(spec, dtype=a.dtype, batch_size=a.batch_size, coverage_min=COVERAGE_MIN,
                        model_id=a.hf_id)
    tag = f"_s{a.shard.replace('/', 'of')}" if a.shard else ""
    out = Path(a.out) if a.out else ROOT / "results" / "readout_calibration" / f"{a.model}{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.with_suffix(".prompts.parquet")
    done = pd.read_parquet(ckpt) if ckpt.exists() else None
    print(f"backend {backend.version}; space_is_separate={backend.space_is_separate}; "
          f"JSON prompt end: {render_json(backend, next(iter(reqs.values())), a.json_mode, a.json_prefix)[-60:]!r}")
    per = run_prompts(backend, reqs, done, ckpt, a.batch_size, a.json_mode, a.json_prefix)
    wide = to_wide(index, per, trials, a.model, backend.version)
    wide.attrs["settings"] = json.dumps(dict(json_mode=a.json_mode, json_prefix=a.json_prefix, topk=TOPK))
    wide.to_parquet(out)
    print(f"wrote {len(wide)} trials -> {out}")
    s = wide
    for call in CALLS:
        print(f"  {call}: cov_A median {s[f'cov_{call}_A'].median():.3f}, cov_C median {s[f'cov_{call}_C'].median():.3f}, "
              f"top-20 integer mass A {s[f'mass_{call}_A20'].median():.3f} / C {s[f'mass_{call}_C20'].median():.3f}, "
              f"mean |p_A - p_C| {np.nanmean(np.abs(s[f'p_{call}_A'] - s[f'p_{call}_C'])):.3f}, "
              f"mean |p_A - p_A20| {np.nanmean(np.abs(s[f'p_{call}_A'] - s[f'p_{call}_A20'])):.3f}")


if __name__ == "__main__":
    main()
