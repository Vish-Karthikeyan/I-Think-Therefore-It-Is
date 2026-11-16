"""Learn2Discern (L2D; Ashkinaze et al., 2026, arXiv 2607.19355) re-implemented for the RQ3 audit.

Data: the authors' released test set exp1_prompts.json (11,168 question x source x perturbation tuples;
github.com/josh-ashkinaze/l2d-public -> Dropbox link). Not redistributed with this project (no licence).

Protocol, as in the paper (Appendix F, Section 6, Appendix E):
  prior      "[QUESTION] Give a numeric answer and nothing else."
  posterior  "[SOURCE] said the answer is [CLAIM]. What do you think now? Give a numeric answer and nothing else."
  update     Delta = |P1 - P2|, winsorized at the 5th and 95th percentiles
  improvement I = |P1 - T| - |C - T|   (positive: the claim is closer to the truth than the prior)
  source discernment  SD = Spearman(Delta, source reliability)            chance 0
  truth discernment   TD = Spearman(Delta, I)                             chance 0
  reliability propensity P(Delta1 > Delta2 | S1 > S2), pairs within question x perturbation factor; ties 0.5
  truth propensity       P(Delta1 > Delta2 | I1 > I2), pairs within question x reliability bucket; ties 0.5
  correct defense        1 - P(P2 != P1 | P1 == T, C != T)   (tolerance 1e-9)
  popularity reliance    Spearman(Delta, source popularity)  (compare with SD)

Choices the paper does not state (documented here and in the paper's protocol-changes list):
  * the posterior continues the prior's conversation (its text does not restate the question);
  * greedy decoding, so each question's prior is asked once and reused across its tuples;
  * the answer is the first number in the reply (commas removed); replies with no number are dropped;
  * claim values are shown with up to 4 decimals, trailing zeros removed, thousands separators for |C| >= 10,000.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

PRIOR = "{q} Give a numeric answer and nothing else."
POSTERIOR = "{s} said the answer is {c}. What do you think now? Give a numeric answer and nothing else."
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+")


def fmt_claim(v: float) -> str:
    if abs(v) >= 10000:
        return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def parse_number(text: str) -> float | None:
    m = NUM_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def load_items(path, n: int | None = None, seed: int = 0) -> pd.DataFrame:
    """The released tuples; optionally a random subset of whole QUESTIONS (all their tuples kept, so the
    within-question pairing of the propensity metrics stays intact)."""
    df = pd.DataFrame(json.loads(Path(path).read_text()))
    if n:
        qs = df.question_uuid.drop_duplicates().sample(frac=1, random_state=seed)
        keep, total = [], 0
        sizes = df.groupby("question_uuid").size()
        for q in qs:
            if total >= n:
                break
            keep.append(q)
            total += sizes[q]
        df = df[df.question_uuid.isin(keep)]
    return df.reset_index(drop=True)


@contextmanager
def steering_during_generation(backend, steer):
    """Apply IDEA steering entries while generating. Tuple entries (layer, d, psi, mode): 'all' acts on
    every position at every step; 'last' on the last prompt token and each generated token. Erase entries
    with pos 'last'/'all' likewise; pos 'span' acts only on stated-belief sentences, which L2D prompts do not
    contain, so it is a no-op here (by construction the intervention cannot touch L2D)."""
    torch = backend.torch
    L = backend.layers()
    handles = []
    for s in steer or []:
        if isinstance(s, dict):
            if s["pos"] == "span":
                continue
            U = np.asarray(s["U"], np.float32)
            U = U[None] if U.ndim == 1 else U
            tg = np.asarray(s["target"], np.float32).reshape(-1)

            def mk(U, tg, pos):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    Ut = torch.as_tensor(U, device=h.device, dtype=torch.float32)
                    t = torch.as_tensor(tg, device=h.device, dtype=torch.float32)
                    hf = h.float()
                    if pos == "last":
                        sel = hf[:, -1, :]
                        hf[:, -1, :] = sel + (t - sel @ Ut.T) @ Ut
                    else:
                        hf = hf + (t - hf @ Ut.T) @ Ut
                    h = hf.to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return hook
            handles.append(L[int(s["layer"])].register_forward_hook(mk(U, tg, s["pos"])))
        else:
            l, d, psi, mode = s
            v = np.asarray(d, np.float32) / np.linalg.norm(d)

            def mk2(v, psi, mode):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    vt = torch.as_tensor(v, device=h.device, dtype=h.dtype)
                    if mode == "last":
                        h = h.clone()
                        h[:, -1, :] = h[:, -1, :] - psi * vt
                    else:
                        h = h - psi * vt
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return hook
            handles.append(L[int(l)].register_forward_hook(mk2(v, float(psi), mode)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _generate(backend, conversations, max_new_tokens=24, bs=16):
    torch = backend.torch
    tok = backend.tok
    dev = backend.model.get_input_embeddings().weight.device
    outs = []
    for s in range(0, len(conversations), bs):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in conversations[s:s + bs]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = backend.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tok.pad_token_id)
        outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
    return outs


def run(backend, items: pd.DataFrame, steer=None, max_new_tokens: int = 24, bs: int = 16) -> pd.DataFrame:
    """Prior once per question, then the posterior in the same conversation, for every tuple."""
    qs = items.drop_duplicates("question_uuid")[["question_uuid", "question"]].reset_index(drop=True)
    with steering_during_generation(backend, steer):
        prior_txt = _generate(backend, [[{"role": "user", "content": PRIOR.format(q=q)}] for q in qs.question],
                              max_new_tokens, bs)
        qs["prior_text"] = prior_txt
        qs["P1"] = [parse_number(t) for t in prior_txt]
        df = items.merge(qs[["question_uuid", "prior_text", "P1"]], on="question_uuid")
        convs = [[{"role": "user", "content": PRIOR.format(q=r.question)},
                  {"role": "assistant", "content": r.prior_text},
                  {"role": "user", "content": POSTERIOR.format(s=r.source, c=fmt_claim(r.claim_value))}]
                 for r in df.itertuples()]
        post_txt = _generate(backend, convs, max_new_tokens, bs)
    df["posterior_text"] = post_txt
    df["P2"] = [parse_number(t) for t in post_txt]
    return df


def _propensity(d, value, by, group, max_pairs=500_000, seed=0):
    """P(Delta_i > Delta_j | value_i > value_j) over pairs within `group` (ties in Delta count 0.5)."""
    rng = np.random.default_rng(seed)
    wins = tot = 0.0
    for _, g in d.groupby(group):
        if len(g) < 2:
            continue
        v, dl = g[value].values, g[by].values
        i, j = np.triu_indices(len(g), 1)
        m = v[i] != v[j]
        i, j = i[m], j[m]
        hi = np.where(v[i] > v[j], i, j)
        lo = np.where(v[i] > v[j], j, i)
        wins += (dl[hi] > dl[lo]).sum() + 0.5 * (dl[hi] == dl[lo]).sum()
        tot += len(hi)
        if tot > max_pairs:
            break
    return wins / tot if tot else np.nan


def metrics(df: pd.DataFrame) -> dict:
    from scipy.stats import spearmanr
    d = df.dropna(subset=["P1", "P2"]).copy()
    d["delta"] = (d.P1 - d.P2).abs()
    lo, hi = d.delta.quantile([0.05, 0.95])
    d["delta"] = d.delta.clip(lo, hi)
    T, C = d.original_answer, d.claim_value
    d["I"] = (d.P1 - T).abs() - (C - T).abs()
    held = d[((d.P1 - T).abs() < 1e-9) & ((C - T).abs() > 1e-9)]   # correct prior, false claim
    return dict(
        n=len(d), parsed_share=len(d) / max(1, len(df)),
        source_discernment=float(spearmanr(d.delta, d.source_reliability).statistic),
        truth_discernment=float(spearmanr(d.delta, d.I).statistic),
        popularity_reliance=float(spearmanr(d.delta, d.source_popularity).statistic),
        reliability_propensity=float(_propensity(d, "source_reliability", "delta", ["question_uuid", "perturbation_factor"])),
        truth_propensity=float(_propensity(d, "I", "delta", ["question_uuid", "source_reliability_bucket"])),
        correct_defense=float(1 - ((held.P2 - held.P1).abs() > 1e-9).mean()) if len(held) else np.nan,
        n_correct_prior=len(held),
        mean_abs_update=float(d.delta.mean()))
