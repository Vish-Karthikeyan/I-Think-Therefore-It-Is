"""Readouts (Section 4.5).

* Open weights ("dist"): next-token distribution pi(k) over the integers 0..100 at the
  answer position, p = sum_k pi(k) (k + 0.5) / 101 after renormalizing; coverage is the
  mass on valid integer tokens. Items under `coverage_min` are re-prompted with an answer
  prefix and otherwise flagged (and dropped at analysis).
* API with logprobs ("dist"): the same estimator over the returned top-20 tokens.
* API without logprobs ("sample"): K samples at temperature 1, each mapped to
  (k + 0.5) / 101 and averaged.
* Choice questions (vignette identification): the same, over option letters.

Every result is cached in SQLite keyed by a hash of (model key, model version, format,
exact prompt, readout settings), so reruns are free and template changes are cache misses.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models import ModelSpec
from .prompts import ANSWER_PREFILL

# A sampled reply counts only if it STARTS with the integer (optionally after markdown bold).
# Taking the first number anywhere misread replies that quote the table ("Non-participants:
# 92/180 ...") as answers: 37 of 64,640 Claude pilot samples, found 2026-11-10.
INT_RE = re.compile(r"^\s*\**\s*(\d{1,3})(?!\d)(?![/.,]\d)")
SAMPLE_MIN_VALID = 0.5  # sampling readouts: re-prompt when fewer than half the samples parse


def to_p(k: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(k, dtype=float) + 0.5) / 101.0


def logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p / (1 - p))


@dataclass
class Request:
    kind: str                         # "number" | "choice"
    fmt: str                          # "chat" | "completion"
    messages: list | None = None      # chat
    text: str | None = None           # completion
    letters: list[str] | None = None  # choice
    readout: str | None = None        # override model readout ("dist" | "sample")
    capture: bool = False             # save activations for this request (open weights)
    meta: dict = field(default_factory=dict)  # not part of the prompt; used by the mock only
    answer_format: str | None = None  # override the model's answer format ("text" for stance questions)

    def prompt_repr(self) -> str:
        return json.dumps({"kind": self.kind, "fmt": self.fmt, "messages": self.messages,
                           "text": self.text, "letters": self.letters}, sort_keys=True)


@dataclass
class Result:
    p: float | None                   # number: probability-weighted readout; None if invalid
    coverage: float
    readout: str
    probs: list[float] | None = None  # choice: renormalized probability per letter
    samples: list | None = None       # sampling readout: parsed integers (None = unparseable)
    attempt: int = 0                  # 1 = answer prefix re-prompt was used
    model_version: str = ""
    raw: Any = None
    answer_format: str = "text"       # format the answer was given in ("text" | "json" | "tool")

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, s: str) -> "Result":
        return cls(**json.loads(s))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class Cache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        # WAL needs shared memory and is unsafe on NFS (e.g. FarmShare scratch): set
        # IDEA_SQLITE_JOURNAL=DELETE there. Caches are per model, so one writer per file.
        mode = os.environ.get("IDEA_SQLITE_JOURNAL", "WAL").upper()
        if mode not in ("WAL", "DELETE", "TRUNCATE", "PERSIST"):
            raise ValueError(f"IDEA_SQLITE_JOURNAL={mode!r}")
        self._con.execute(f"PRAGMA journal_mode={mode}")
        self._con.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, model TEXT, "
                          "result TEXT, ts REAL)")
        self._con.commit()

    def get_many(self, keys: Iterable[str]) -> dict[str, Result]:
        keys = list(keys)
        out = {}
        with self._lock:
            for i in range(0, len(keys), 900):
                chunk = keys[i:i + 900]
                q = f"SELECT key, result FROM results WHERE key IN ({','.join('?' * len(chunk))})"
                for k, r in self._con.execute(q, chunk):
                    out[k] = Result.from_json(r)
        return out

    def put_many(self, items: dict[str, Result], model: str) -> None:
        with self._lock:
            self._con.executemany("INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                                  [(k, model, r.to_json(), time.time()) for k, r in items.items()])
            self._con.commit()


def effective_format(req: Request, spec: ModelSpec) -> str:
    return req.answer_format or getattr(spec, "answer_format", "text")


def request_key(req: Request, spec: ModelSpec, version: str, settings: dict) -> str:
    readout = req.readout or spec.readout
    settings = dict(settings)
    fmt = effective_format(req, spec)
    if req.kind == "number" and fmt != "text":
        settings["answer_format"] = fmt  # never reuse a free-text answer for a constrained one
    payload = json.dumps([spec.key, version, readout, settings, req.prompt_repr()], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

# API re-prompt for probability readouts whose integer-token coverage is too low (GPT often
# starts reasoning instead of answering). Chosen on 2026-11-10 by testing three wordings on 12
# low-coverage pilot prompts: this one lifted mean coverage from 0.69 to 0.93 (9/12 >= 0.9);
# an "Answer:" prefix gave 0.01 and an assistant prefill 0.53.
API_REPROMPT_SUFFIX = "\n\nReply with the number only. Do not explain."


ANSWER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["probability"],
                 "properties": {"probability": {"type": "integer", "minimum": 0, "maximum": 100}}}
ANSWER_TOOL = {"type": "function", "function": {"name": "answer", "description": "Give your answer.",
                                                "parameters": ANSWER_SCHEMA}}


def parse_answer_json(text: str) -> int | None:
    try:
        v = json.loads(text or "").get("probability")
    except (ValueError, AttributeError):
        return None
    return v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100 else None


class Backend:
    spec: ModelSpec

    def needs_retry(self, res: "Result") -> bool:
        """True if a cached result predates a retry step this backend now applies."""
        return False
    version: str = ""
    settings: dict = {}

    def run(self, reqs: list[Request]) -> list[Result]:
        raise NotImplementedError

    # activations for requests with capture=True, keyed by request index in the last run
    last_activations: dict[int, dict[int, np.ndarray]] = {}


def steer_signature(steer) -> str:
    """Steered results must never share cache keys with unsteered ones."""
    if not steer:
        return ""
    h = hashlib.sha256()
    for s in steer:
        if isinstance(s, dict):   # subspace erasure (2026-11-14); tuple entries hash exactly as before
            h.update(f"erase|{s['layer']}|{s['pos']}|".encode())
            h.update(np.asarray(s["U"], dtype=np.float32).tobytes())
            h.update(np.asarray(s["target"], dtype=np.float32).tobytes())
            continue
        l, d, psi, mode = s
        h.update(f"{l}|{psi:.6g}|{mode}|".encode())
        h.update(np.asarray(d, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def parse_int(text: str, lo: int = 0, hi: int = 100) -> int | None:
    m = INT_RE.match(text or "")
    if not m:
        return None
    k = int(m.group(1))
    return k if lo <= k <= hi else None


def parse_letter(text: str, letters: list[str]) -> int | None:
    t = (text or "").strip().strip("().*").upper()
    for i, l in enumerate(letters):
        if t.startswith(l):
            return i
    return None


def sample_result(values: list, n_options: int | None = None, kind: str = "number") -> tuple:
    valid = [v for v in values if v is not None]
    coverage = len(valid) / max(1, len(values))
    if kind == "number":
        p = float(np.mean(to_p(np.array(valid)))) if valid else None
        return p, coverage, None
    counts = np.bincount(valid, minlength=n_options).astype(float) if valid else np.zeros(n_options)
    probs = (counts / counts.sum()).tolist() if valid else None
    return None, coverage, probs


# ----------------------------- Hugging Face ---------------------------------

class HFBackend(Backend):
    """Local weights. Batched, left-padded forward passes; activations from forward hooks at
    the last prompt token of the same pass; optional steering hooks."""

    def __init__(self, spec: ModelSpec, dtype: str = "auto", device_map: str = "auto",
                 batch_size: int = 8, coverage_min: float = 0.9, capture_layers: Iterable[int] = (),
                 sample_k: int = 10, temperature: float = 1.0, hf_token: str | None = None,
                 model_id: str | None = None, max_length: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.spec = spec
        mid = model_id or spec.hf_id
        tok_kw = dict(revision=spec.revision, token=hf_token or os.environ.get("HF_TOKEN"))
        self.tok = AutoTokenizer.from_pretrained(mid, **tok_kw)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if dtype == "auto":
            if torch.cuda.is_available():
                dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dt = torch.float32
        else:
            dt = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(mid, dtype=dt, device_map=device_map,
                                                          **tok_kw).eval()
        self.version = f"{mid}@{getattr(self.model.config, '_commit_hash', None) or spec.revision}"
        self.batch_size = batch_size
        self.coverage_min = coverage_min
        self.capture_layers = list(capture_layers)
        self.sample_k, self.temperature = sample_k, temperature
        self.max_length = max_length
        self.settings = dict(coverage_min=coverage_min, sample_k=sample_k, temperature=temperature)
        self.num_ids = self._token_ids([str(k) for k in range(101)])
        self.steer: list[tuple[int, np.ndarray, float, str]] = []  # (layer, unit dir, psi, mode)
        self.last_activations = {}

    # token ids for each answer string, with and without a leading space
    def _token_ids(self, strings: list[str]) -> list[list[int]]:
        out = []
        for s in strings:
            ids = set()
            for variant in (s, " " + s):
                enc = self.tok.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    ids.add(enc[0])
                elif len(enc) == 2 and variant.startswith(" ") and self.tok.decode(enc[:1]).strip() == "":
                    pass  # space is its own token; the no-space variant covers it
            if not ids:
                raise ValueError(f"answer {s!r} is not a single token for {self.spec.key}")
            out.append(sorted(ids))
        return out

    def layers(self):
        m = self.model
        for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("cannot find decoder layers")

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def space_is_separate(self) -> bool:
        """True when ' 57' tokenizes as [' ', '57'] (OLMo 2): prompts that end where a number
        follows must then end in a space, or the next token is the space, not the answer."""
        if not hasattr(self, "_space_sep"):
            self._space_sep = len(self.tok.encode(" 57", add_special_tokens=False)) == 2
        return self._space_sep

    def render(self, req: Request, prefill: bool) -> str:
        if req.fmt == "chat":
            text = self.tok.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
            if prefill:
                pre = ANSWER_PREFILL if req.kind == "number" else "Answer: ("
                if req.kind == "number" and not self.space_is_separate:
                    pre = pre.rstrip()  # the answer token carries its own leading space
                text += pre
            return text
        text = req.text if req.text is not None else req.messages[-1]["content"]
        if prefill and req.kind == "choice":
            text += "\nAnswer: ("
        elif req.kind == "number" and text.endswith(":") and self.space_is_separate:
            text += " "
        bos = self.tok.bos_token
        if bos and not text.startswith(bos):
            text = bos + text  # completion prompts are tokenized with add_special_tokens=False
        return text

    def set_steering(self, steer: list[tuple[int, np.ndarray, float, str]]):
        """steer: list of entries, each either
          (layer, direction, psi, mode), mode 'all' or 'last': h <- h - psi * d/||d||, or
          {"kind": "erase", "layer": l, "U": [k, d] orthonormal rows, "target": [k], "pos": 'last'|'all'|'span'}:
             the coordinates of h in span(U) are set to `target` (subspace erasure / target setting),
             at the last position, every position, or the tokens of the request's stated-belief
             sentence (request.meta["span"]).
        Both act on the output of decoder layer `layer`."""
        out = []
        for s in steer:
            if isinstance(s, dict):
                U = np.asarray(s["U"], dtype=np.float32)
                U = U[None, :] if U.ndim == 1 else U
                out.append(dict(kind="erase", layer=int(s["layer"]), U=U,
                                target=np.asarray(s["target"], dtype=np.float32).reshape(-1), pos=s["pos"]))
            else:
                l, d, psi, mode = s
                out.append((l, np.asarray(d, dtype=np.float32) / np.linalg.norm(d), float(psi), mode))
        self.steer = out
        self.settings = dict(self.settings, steer=steer_signature(self.steer))

    def _span_mask(self, texts, spans, offsets):
        """[batch, seq] bool: tokens overlapping each text's stated-belief sentence (none if absent)."""
        mask = np.zeros(offsets.shape[:2], dtype=bool)
        for i, (t, sp) in enumerate(zip(texts, spans or [None] * len(texts))):
            if not sp:
                continue
            a = t.rfind(sp)
            if a < 0:
                continue
            b = a + len(sp)
            st, en = offsets[i, :, 0], offsets[i, :, 1]
            mask[i] = (en > a) & (st < b) & (en > st)
        return mask

    def _forward(self, texts: list[str], capture: bool, spans: list[str | None] | None = None):
        torch = self.torch
        need_span = any(isinstance(s, dict) and s["pos"] == "span" for s in self.steer)
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                       truncation=True, max_length=self.max_length, return_offsets_mapping=need_span)
        span_mask = None
        if need_span:
            span_mask = self._span_mask(texts, spans, enc.pop("offset_mapping").numpy())
        dev = self.model.get_input_embeddings().weight.device
        enc = {k: v.to(dev) for k, v in enc.items()}
        acts: dict[int, Any] = {}
        handles = []
        layers = self.layers()
        if capture:
            for l in self.capture_layers:
                def mk(l):
                    def hook(mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        acts[l] = h[:, -1, :].detach().float().cpu().numpy()
                    return hook
                handles.append(layers[l].register_forward_hook(mk(l)))
        for s in [s for s in self.steer if isinstance(s, dict)]:
            def mk_e(U, target, pos, mask):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    Ut = torch.as_tensor(U, device=h.device, dtype=torch.float32)        # [k, d]
                    tg = torch.as_tensor(target, device=h.device, dtype=torch.float32)  # [k]
                    if pos == "last":
                        sel = h[:, -1, :].float()
                        new = sel + (tg - sel @ Ut.T) @ Ut
                        h = h.clone()
                        h[:, -1, :] = new.to(h.dtype)
                    else:
                        hf = h.float()
                        delta = (tg - hf @ Ut.T) @ Ut                                      # [b, s, d]
                        if pos == "span":
                            m = torch.as_tensor(mask, device=h.device, dtype=torch.float32)[:, :, None]
                            delta = delta * m
                        h = (hf + delta).to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return hook
            handles.append(layers[s["layer"]].register_forward_hook(mk_e(s["U"], s["target"], s["pos"], span_mask)))
        for l, d, psi, mode in [s for s in self.steer if not isinstance(s, dict)]:
            def mk_s(d, psi, mode):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    v = torch.as_tensor(d, device=h.device, dtype=h.dtype)
                    if mode == "last":
                        h = h.clone()
                        h[:, -1, :] = h[:, -1, :] - psi * v
                    else:
                        h = h - psi * v
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return hook
            # register steering before capture hooks run on later layers; order within a
            # layer: steering hook is registered after capture hook, so captured
            # activations at a steered layer are pre-steering.
            handles.append(layers[l].register_forward_hook(mk_s(d, psi, mode)))
        try:
            with torch.no_grad():
                # only the last position's logits are needed; computing all of them costs
                # batch x seq x vocab memory (GBs) for identical results
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        finally:
            for h in handles:
                h.remove()
        return torch.softmax(logits, dim=-1).cpu().numpy(), acts

    def _dist(self, probs_row: np.ndarray, id_lists: list[list[int]]) -> tuple[np.ndarray, float]:
        mass = np.array([probs_row[ids].sum() for ids in id_lists])
        return mass, float(mass.sum())

    def run(self, reqs: list[Request]) -> list[Result]:
        results: list[Result | None] = [None] * len(reqs)
        self.last_activations = {}
        dist_idx = [i for i, r in enumerate(reqs) if (r.readout or self.spec.readout) == "dist"]
        samp_idx = [i for i, r in enumerate(reqs) if (r.readout or self.spec.readout) == "sample"]
        for attempt in (0, 1):
            todo = [i for i in dist_idx if results[i] is None or
                    (attempt == 1 and results[i].coverage < self.coverage_min)]
            if attempt == 1:
                # completion-format numbers already end in the answer prefix
                todo = [i for i in todo if not (reqs[i].fmt == "completion" and reqs[i].kind == "number")]
            for s in range(0, len(todo), self.batch_size):
                batch = todo[s:s + self.batch_size]
                texts = [self.render(reqs[i], prefill=attempt == 1) for i in batch]
                cap = attempt == 0 and any(reqs[i].capture for i in batch) and bool(self.capture_layers)
                probs, acts = self._forward(texts, cap, [reqs[i].meta.get("span") for i in batch])
                for j, i in enumerate(batch):
                    r = reqs[i]
                    ids = self.num_ids if r.kind == "number" else self._token_ids(r.letters)
                    mass, cov = self._dist(probs[j], ids)
                    prev = results[i]
                    if prev is not None and prev.coverage >= cov:
                        continue  # keep the better attempt
                    if r.kind == "number":
                        w = mass / cov if cov > 0 else None
                        p = float((w * to_p(np.arange(101))).sum()) if w is not None else None
                        res = Result(p, cov, "dist", attempt=attempt, model_version=self.version)
                    else:
                        res = Result(None, cov, "dist", probs=(mass / cov).tolist() if cov > 0 else None,
                                     attempt=attempt, model_version=self.version)
                    results[i] = res
                    if cap and r.capture:
                        self.last_activations[i] = {l: a[j].astype(np.float16) for l, a in acts.items()}
        for i in samp_idx:
            results[i] = self._sample(reqs[i])
        return results  # type: ignore

    def _sample(self, req: Request) -> Result:
        torch = self.torch
        text = self.render(req, prefill=False)
        enc = self.tok([text], return_tensors="pt", add_special_tokens=False)
        dev = self.model.get_input_embeddings().weight.device
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=True, temperature=self.temperature, top_p=1.0,
                                      max_new_tokens=6, num_return_sequences=self.sample_k,
                                      pad_token_id=self.tok.pad_token_id)
        gen = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if req.kind == "number":
            vals = [parse_int(g) for g in gen]
            p, cov, _ = sample_result(vals)
            return Result(p, cov, "sample", samples=vals, model_version=self.version, raw=gen)
        vals = [parse_letter(g, req.letters) for g in gen]
        _, cov, probs = sample_result(vals, len(req.letters), "choice")
        return Result(None, cov, "sample", probs=probs, samples=vals, model_version=self.version, raw=gen)


# ----------------------------- OpenRouter -----------------------------------

class OpenRouterBackend(Backend):
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, spec: ModelSpec, api_key: str | None = None, concurrency: int = 16,
                 sample_k: int = 10, temperature: float = 1.0, top_logprobs: int = 20,
                 max_retries: int = 6, timeout: float = 120.0, model_id: str | None = None,
                 coverage_min: float = 0.9, max_cost: float | None = None, pricing: dict | None = None,
                 api_key_env: str = "OPENROUTER_API_KEY"):
        if api_key is None:
            from dotenv import load_dotenv
            load_dotenv()
            # also the project folder's .env (one level above the package), without overriding
            root_env = Path(__file__).resolve().parents[3] / ".env"
            if root_env.exists():
                load_dotenv(root_env, override=False)
            api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set")
        self.api_key = api_key
        self.spec = spec
        self.model_id = model_id or spec.openrouter_id
        self.version = self.model_id
        self.concurrency, self.sample_k, self.temperature = concurrency, sample_k, temperature
        self.top_logprobs, self.max_retries, self.timeout = top_logprobs, max_retries, timeout
        self.settings = dict(sample_k=sample_k, temperature=temperature, top_logprobs=top_logprobs)
        self.last_activations = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "calls": 0}
        self.coverage_min = coverage_min
        self.max_cost = max_cost          # hard cap on this process's spend, in USD
        self._pricing = pricing           # {"prompt": $/token, "completion": $/token}; fetched if None

    def _reprompt_messages(self, messages: list) -> list:
        msgs = [dict(m) for m in messages]
        if self.spec.reprompt == "prefill":
            # the reply continues after the prefix; Anthropic rejects trailing whitespace here
            return msgs + [{"role": "assistant", "content": ANSWER_PREFILL.rstrip()}]
        msgs[-1]["content"] = msgs[-1]["content"] + API_REPROMPT_SUFFIX
        return msgs

    async def _sample_k(self, client, body):
        return await asyncio.gather(*[self._post(client, body) for _ in range(self.sample_k)])

    # Extra tokens each call carries beyond the prompt, and tokens generated, by answer format
    # (tool overhead measured 2026-11-10: a forced answer call cost 2.8x a free-text call).
    FORMAT_OVERHEAD = {"text": (0, 6), "json": (40, 8), "tool": (650, 30)}

    def pricing(self) -> dict:
        if self._pricing is None:
            import httpx
            ms = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
            pr = next(m for m in ms if m["id"] == self.model_id)["pricing"]
            self._pricing = {"prompt": float(pr["prompt"]), "completion": float(pr["completion"])}
        return self._pricing

    def estimate_cost(self, reqs: list[Request]) -> float:
        """Upper-end USD estimate for running `reqs` (x1.25 margin, re-prompts not included)."""
        pr = self.pricing()
        total = 0.0
        for r in reqs:
            fmt = effective_format(r, self.spec) if r.kind == "number" else "text"
            extra_in, out = self.FORMAT_OVERHEAD[fmt]
            k = self.sample_k if (r.readout or self.spec.readout) == "sample" else 1
            tokens_in = len(json.dumps(self._messages(r))) / 3.5 + extra_in
            total += k * (tokens_in * pr["prompt"] + out * pr["completion"])
        return 1.25 * total

    def needs_retry(self, res: "Result") -> bool:
        if res.probs is not None or res.attempt != 0:
            return False
        if res.readout == "dist":   # logprob number answer below the coverage threshold
            return res.coverage < self.coverage_min
        if res.readout == "sample" and res.raw is not None:
            # parsed under an older rule, or too few usable samples: redo it
            parse = parse_answer_json if res.answer_format in ("tool", "json") else parse_int
            return ([parse(t) for t in res.raw] != list(res.samples or [])
                    or res.coverage < SAMPLE_MIN_VALID)
        return False

    def _messages(self, req: Request) -> list:
        if req.fmt == "chat":
            return req.messages
        return [{"role": "user", "content": req.text}]

    async def _post(self, client, body: dict) -> dict:
        import httpx
        if self.max_cost is not None and self.usage["cost"] >= self.max_cost:
            raise BudgetExceeded(f"spent ${self.usage['cost']:.2f} of the ${self.max_cost:.2f} cap for this run")
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = await client.post(self.URL, json=body, headers={
                    "Authorization": f"Bearer {self.api_key}", "X-Title": "IDEA research client"})
                if r.status_code == 200:
                    data = r.json()
                    if "error" in data:
                        raise RuntimeError(str(data["error"]))
                    u = data.get("usage") or {}
                    self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                    self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                    self.usage["cost"] += float(u.get("cost", 0) or 0)
                    self.usage["calls"] += 1
                    return data
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                if r.status_code == 402 and "in_flight_budget" in r.text and attempt < self.max_retries - 1:
                    # Low balance: OpenRouter reserves credit for every in-flight request and asks us to
                    # wait (Retry-After). Not a real shortage, so wait and retry the same request.
                    wait = float(r.headers.get("retry-after") or 120)
                    await asyncio.sleep(min(wait, 300) + random.random())
                    continue
                raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:500]}")
            except (httpx.HTTPError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "OpenRouter 4" in str(e):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)

    def _body(self, req: Request, **kw) -> dict:
        body = {"model": self.model_id, "messages": self._messages(req), "usage": {"include": True}}
        body.update(self.spec.extra_body)
        body.update(kw)
        return body

    async def _one(self, client, sem, req: Request) -> Result:
        readout = req.readout or self.spec.readout
        async with sem:
            if readout == "dist" and req.kind == "number" and effective_format(req, self.spec) == "json":
                kw = dict(max_tokens=12, temperature=0, logprobs=True, top_logprobs=self.top_logprobs,
                          response_format={"type": "json_schema", "json_schema": {
                              "name": "answer", "strict": True, "schema": ANSWER_SCHEMA}})
                return self._from_json_logprobs(await self._post(client, self._body(req, **kw)))
            if readout == "dist":
                kw = dict(max_tokens=1, temperature=0, logprobs=True, top_logprobs=self.top_logprobs)
                res = self._from_logprobs(req, await self._post(client, self._body(req, **kw)))
                if req.kind == "number" and res.coverage < self.coverage_min:
                    body = self._body(req, **kw)
                    body["messages"] = self._reprompt_messages(body["messages"])
                    res2 = self._from_logprobs(req, await self._post(client, body))
                    res2.attempt = 1
                    if res2.coverage > res.coverage:
                        return res2
                    res.attempt = 1  # re-prompt tried; first answer kept (higher coverage)
                return res
            if req.kind == "number" and effective_format(req, self.spec) == "json":
                # sampled structured answer {"probability": k} (models without logprobs, e.g. Gemini on
                # OpenRouter): ~40 extra prompt tokens instead of ~650 for a forced tool call
                body = self._body(req, max_tokens=64, temperature=self.temperature,
                                  response_format={"type": "json_schema", "json_schema": {
                                      "name": "answer", "strict": True, "schema": ANSWER_SCHEMA}})
                datas = await self._sample_k(client, body)
                txt = [(d["choices"][0]["message"].get("content") or "") for d in datas]
                vals = [parse_answer_json(t) for t in txt]
                p, cov, _ = sample_result(vals)
                version = datas[0].get("model", self.model_id) if datas else self.model_id
                return Result(p, cov, "sample", samples=vals, model_version=version, raw=txt, attempt=0,
                              answer_format="json")
            if req.kind == "number" and effective_format(req, self.spec) == "tool":
                body = self._body(req, max_tokens=40, temperature=self.temperature, tools=[ANSWER_TOOL],
                                  tool_choice={"type": "function", "function": {"name": "answer"}})
                datas = await self._sample_k(client, body)
                args = []
                for d in datas:
                    calls = d["choices"][0]["message"].get("tool_calls") or []
                    args.append(calls[0]["function"]["arguments"] if calls else "")
                vals = [parse_answer_json(a) for a in args]
                p, cov, _ = sample_result(vals)
                version = datas[0].get("model", self.model_id) if datas else self.model_id
                return Result(p, cov, "sample", samples=vals, model_version=version, raw=args, attempt=0,
                              answer_format="tool")
            # sampling: K independent completions (n is not supported by every provider)
            datas = await self._sample_k(client, self._body(req, max_tokens=8, temperature=self.temperature))
            texts = [(d["choices"][0]["message"].get("content") or "") for d in datas]
            version = datas[0].get("model", self.model_id) if datas else self.model_id
            if req.kind == "number":
                vals = [parse_int(t) for t in texts]
                p, cov, _ = sample_result(vals)
                res = Result(p, cov, "sample", samples=vals, model_version=version, raw=texts)
                if cov < SAMPLE_MIN_VALID:
                    body = self._body(req, max_tokens=8, temperature=self.temperature)
                    body["messages"] = self._reprompt_messages(body["messages"])
                    texts2 = [(d["choices"][0]["message"].get("content") or "") for d in await self._sample_k(client, body)]
                    vals2 = [parse_int(t) for t in texts2]
                    p2, cov2, _ = sample_result(vals2)
                    if cov2 > cov:
                        res = Result(p2, cov2, "sample", samples=vals2, model_version=version, raw=texts2)
                    res.attempt = 1
                return res
        vals = [parse_letter(t, req.letters) for t in texts]
        _, cov, probs = sample_result(vals, len(req.letters), "choice")
        return Result(None, cov, "sample", probs=probs, samples=vals, model_version=version, raw=texts)

    def _from_json_logprobs(self, data: dict) -> Result:
        """Structured answer {"probability": k}: the distribution is read at the number token.
        The schema guarantees an integer answer, so `coverage` is 1 for a valid reply; the share
        of probability the top-20 alternatives capture (they truncate a spread distribution) is
        kept in raw['top20_mass'] and the captured integers are renormalized."""
        version = data.get("model", self.model_id)
        ch = data["choices"][0]
        text = (ch.get("message") or {}).get("content") or ""
        answer = parse_answer_json(text)
        toks = (ch.get("logprobs") or {}).get("content") or []
        pos = next((t for t in toks if t["token"].strip().isdigit()), None)
        if answer is None or pos is None:
            return Result(None, 0.0, "dist", model_version=version, raw={"content": text})
        mass = np.zeros(101)
        for t in pos.get("top_logprobs") or []:
            s_ = t["token"].strip()
            if s_.isdigit() and s_ == str(int(s_)) and 0 <= int(s_) <= 100:
                mass[int(s_)] += math.exp(t["logprob"])
        top = float(mass.sum())
        if top <= 0:
            mass[answer] = 1.0; top = 1.0
        p = float((mass / mass.sum() * to_p(np.arange(101))).sum())
        return Result(p, 1.0, "dist", model_version=version, answer_format="json",
                      raw={"content": text, "top20_mass": top, "answer": answer})

    def _from_logprobs(self, req: Request, data: dict) -> Result:
        version = data.get("model", self.model_id)
        ch = data["choices"][0]
        content = (ch.get("logprobs") or {}).get("content") or []
        if not content:
            return Result(None, 0.0, "dist", model_version=version, raw=ch.get("message"))
        top = content[0].get("top_logprobs") or []
        if req.kind == "number":
            mass = np.zeros(101)
            for t in top:
                s = t["token"].strip()
                if s.isdigit() and 0 <= int(s) <= 100 and s == str(int(s)):
                    mass[int(s)] += math.exp(t["logprob"])
            cov = float(mass.sum())
            p = float((mass / cov * to_p(np.arange(101))).sum()) if cov > 0 else None
            return Result(p, cov, "dist", model_version=version)
        mass = np.zeros(len(req.letters))
        for t in top:
            i = parse_letter(t["token"], req.letters) if t["token"].strip().strip("(").upper() in req.letters else None
            if i is not None:
                mass[i] += math.exp(t["logprob"])
        cov = float(mass.sum())
        return Result(None, cov, "dist", probs=(mass / cov).tolist() if cov > 0 else None, model_version=version)

    def run(self, reqs: list[Request]) -> list[Result]:
        import httpx

        async def main():
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await asyncio.gather(*[self._one(client, sem, r) for r in reqs])
        return asyncio.run(main())


# ----------------------------- Mock -----------------------------------------

@dataclass
class MockParams:
    """Generative model with planted effects (log-odds space)."""
    rho: float = 0.9
    beta_neutral: float = 0.8
    delta_b: float = -0.25          # contradicted-slope shift
    support_shift: float = 0.05
    theta_cong: float = -0.15
    group_delta_sd: float = 0.05
    gamma_belief: float = 0.4       # prior shift toward stated belief
    prior_sd: float = 0.8
    noise_sd: float = 0.5
    retest_sd: float = 0.3
    stance_sd: float = 0.6
    leave_frame: float = 0.0        # P(H1)+P(H0) shortfall
    hidden: int = 64                # activation width
    masking: float = 0.6            # share of the discount that is response-selection only
    tier1_scale: float = 0.2        # social amplification: Tier 1 effects are this fraction
    seed: int = 0


class MockBackend(Backend):
    """Answers from `MockParams` using request.meta (trial fields), with deterministic
    per-prompt noise. Produces activations with a planted 'caving' direction and a
    readable Bayesian-posterior direction so RQ3 code can be validated end to end."""

    def __init__(self, spec: ModelSpec, params: MockParams | None = None, capture_layers=(4, 8),
                 sample_k: int = 10):
        self.spec = spec
        self.p = params or MockParams()
        self.version = f"mock-{self.p.seed}"
        self.settings = {"mock": asdict(self.p)}
        self.capture_layers = list(capture_layers)
        self.sample_k = sample_k
        rng = np.random.default_rng(self.p.seed)
        H = self.p.hidden
        self.dir_post = rng.normal(size=H); self.dir_post /= np.linalg.norm(self.dir_post)
        self.dir_cave = rng.normal(size=H); self.dir_cave -= self.dir_cave @ self.dir_post * self.dir_post
        self.dir_cave /= np.linalg.norm(self.dir_cave)
        self.topic_dirs: dict[str, np.ndarray] = {}
        self._rng = rng
        self.group_delta: dict[str, float] = {}
        self.steer = []
        self.last_activations = {}

    def set_steering(self, steer):
        self.steer = [(l, np.asarray(d) / np.linalg.norm(d), psi, mode) for l, d, psi, mode in steer]
        self.settings = dict(self.settings, steer=steer_signature(self.steer))

    def _noise(self, *parts) -> np.random.Generator:
        h = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)
        return np.random.default_rng(h + self.p.seed)

    def _gdelta(self, group: str) -> float:
        if group not in self.group_delta:
            r = self._noise("gdelta", group)
            self.group_delta[group] = 0.0 if group == "unspecified" else float(r.normal(0, self.p.group_delta_sd))
        return self.group_delta[group]

    def stance_g(self, cue_id: str, topic: str, h_dir: int) -> float:
        if cue_id.startswith("unsp"):
            return 0.0
        r = self._noise("stance", cue_id.split("_v")[0].replace("_label", ""), topic)
        return float(r.normal(0, self.p.stance_sd)) * h_dir

    def _topic_dir(self, topic: str) -> np.ndarray:
        if topic not in self.topic_dirs:
            r = self._noise("topicdir", topic)
            self.topic_dirs[topic] = r.normal(size=self.p.hidden) * 0.3
        return self.topic_dirs[topic]

    def _answer(self, m: dict, prompt_key: str) -> tuple[float, np.ndarray | None]:
        P = self.p
        call = m.get("call", "prior")
        tier_scale = 1.0 if m.get("tier", 2) == 2 else P.tier1_scale
        if call == "stance":
            g = self.stance_g(m["cue_id"], m["topic"], m["h_dir"])
            return 1 / (1 + math.exp(-g)), None
        if call == "ident":
            return 0.0, None
        base = self._noise("prior", m["topic"], m["h_dir"], m["cue_id"]).normal(0, P.prior_sd)
        lp_prior = base + P.gamma_belief * m["b"]
        if call == "prior_rep":
            lp_prior += self._noise("retest", prompt_key).normal(0, P.retest_sd)
        target = m.get("target", "H1")
        if call in ("prior", "prior_rep"):
            lp = lp_prior if target == "H1" else -lp_prior
            sh = P.leave_frame if target == "H0" else 0.0
            p = 1 / (1 + math.exp(-lp))
            return max(0.005, min(0.995, p * (1 - sh))), None
        # posterior
        x, b = m["x"], m["b"]
        d = int(np.sign(x))
        g = self.stance_g(m["cue_id"], m["topic"], m["h_dir"])
        mcong = b * g
        contra = b * d < 0
        support = b * d > 0
        shift = 0.0
        if contra:
            shift = (P.delta_b + self._gdelta(m["group"]) + P.theta_cong * mcong) * tier_scale
        elif support:
            shift = P.support_shift * tier_scale
        steer_relief = 0.0
        for _, dvec, psi, _ in self.steer:
            steer_relief += psi * float(dvec @ self.dir_cave)
        if contra and steer_relief:
            shift = shift * max(0.0, 1 - steer_relief / 3.0)
        slope = P.beta_neutral + shift
        lp_post = P.rho * lp_prior + slope * x + self._noise("post", prompt_key).normal(0, P.noise_sd)
        lp = lp_post if target == "H1" else -lp_post
        p = 1 / (1 + math.exp(-lp))
        if target == "H0":
            p *= (1 - P.leave_frame)
        # activations: internal posterior = Bayes posterior + (1-masking) * distortion
        internal = P.rho * lp_prior + x + (1 - P.masking) * (slope - 1) * x
        r = self._noise("act", prompt_key)
        h = (r.normal(size=P.hidden) * 0.5 + internal * self.dir_post + self._topic_dir(m["topic"])
             + (1.5 if (contra and shift < -0.15) else 0.0) * self.dir_cave)
        return max(0.005, min(0.995, p)), h.astype(np.float32)

    def run(self, reqs: list[Request]) -> list[Result]:
        out = []
        self.last_activations = {}
        for i, r in enumerate(reqs):
            key = r.prompt_repr()
            m = r.meta
            if r.kind == "choice":
                n = len(r.letters)
                probs = np.full(n, 0.02)
                tgt = m.get("target_index")
                if tgt is not None and m.get("ask_dim") == m.get("cue_dim"):
                    probs[tgt] = 0.9
                else:
                    probs[-1] = 0.9
                probs /= probs.sum()
                out.append(Result(None, 0.99, "dist", probs=probs.tolist(), model_version=self.version))
                continue
            p, h = self._answer(m, key)
            readout = r.readout or self.spec.readout
            if readout == "sample":
                rng = self._noise("sample", key)
                vals = [int(np.clip(round(rng.normal(p * 100, 8)), 0, 100)) for _ in range(self.sample_k)]
                pp, cov, _ = sample_result(vals)
                out.append(Result(pp, cov, "sample", samples=vals, model_version=self.version))
            else:
                out.append(Result(p, 0.98, "dist", model_version=self.version))
            if r.capture and h is not None:
                self.last_activations[i] = {l: (h * (1 + 0.1 * j)).astype(np.float16)
                                            for j, l in enumerate(self.capture_layers)}
        return out


def make_backend(spec: ModelSpec, **kw) -> Backend:
    if spec.backend == "hf":
        return HFBackend(spec, **kw)
    if spec.backend == "openrouter":
        return OpenRouterBackend(spec, **{k: v for k, v in kw.items()
                                          if k in ("api_key", "concurrency", "sample_k", "temperature",
                                                   "top_logprobs", "model_id", "coverage_min", "max_cost", "pricing",
                                                   "api_key_env")})
    if spec.backend == "mock":
        return MockBackend(spec, **{k: v for k, v in kw.items() if k in ("params", "capture_layers", "sample_k")})
    raise ValueError(spec.backend)


class DeadlineReached(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    """A run's estimated or actual API cost would pass `max_cost` (elicit.max_cost_per_run)."""


class CachedElicitor:
    """Dedupes requests by prompt hash, serves cached results, runs the rest in chunks
    (checkpointing to the cache after every chunk so long runs are resumable)."""

    def __init__(self, backend: Backend, cache: Cache, chunk: int = 256, act_store=None, progress=True,
                 deadline: float | None = None):
        self.b, self.cache, self.chunk, self.act_store, self.progress = backend, cache, chunk, act_store, progress
        # Unix time after which no new chunk starts (env IDEA_DEADLINE); results so far are cached,
        # so rerunning the same command resumes where it stopped.
        self.deadline = deadline if deadline is not None else (float(os.environ["IDEA_DEADLINE"])
                                                               if os.environ.get("IDEA_DEADLINE") else None)

    def key(self, req: Request) -> str:
        return request_key(req, self.b.spec, self.b.version, self.b.settings)

    def run(self, reqs: list[Request]) -> list[Result]:
        keys = [self.key(r) for r in reqs]
        have = self.cache.get_many(set(keys))
        # activations already stored?
        need_act = set()
        if self.act_store is not None:
            need_act = {k for k, r in zip(keys, reqs) if r.capture and not self.act_store.has(k)}
        todo: dict[str, Request] = {}
        for k, r in zip(keys, reqs):
            stale = k in have and self.b.needs_retry(have[k])
            if (k not in have or k in need_act or stale) and k not in todo:
                todo[k] = r
        items = list(todo.items())
        cap = getattr(self.b, "max_cost", None)
        if cap is not None and items and hasattr(self.b, "estimate_cost"):
            est = self.b.estimate_cost([r for _, r in items])
            left = cap - self.b.usage["cost"]
            print(f"[{self.b.spec.key}] {len(items)} uncached requests, estimated ${est:.2f} (cap ${cap:.2f})")
            if est > left:
                raise BudgetExceeded(f"estimated ${est:.2f} for {len(items)} requests exceeds the remaining "
                                     f"${left:.2f} of the ${cap:.2f} cap; raise elicit.max_cost_per_run to run it")
        it = range(0, len(items), self.chunk)
        if self.progress and items:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=f"elicit {self.b.spec.key}", unit="chunk")
        for s in it:
            if self.deadline and time.time() > self.deadline:
                raise DeadlineReached(f"stopped at {s}/{len(items)} uncached requests; rerun to resume")
            chunk = items[s:s + self.chunk]
            res = self.b.run([r for _, r in chunk])
            new = {k: rr for (k, _), rr in zip(chunk, res)}
            if self.act_store is not None and self.b.last_activations:
                self.act_store.put({chunk[i][0]: a for i, a in self.b.last_activations.items()})
            self.cache.put_many(new, self.b.spec.key)
            have.update(new)
        return [have[k] for k in keys]
