#!/usr/bin/env python
"""Build an IDEA trial dataset for one model, in the schema of datasets/main_trials.csv.gz.

The output has the same 84 columns as the released dataset, so the analysis scripts
(analysis/rq1.py, analysis/rq2.py) run on it unchanged:

    python analysis/run_all.py --data my_trials.csv.gz

Steps (each answer is cached, so an interrupted run resumes when you rerun the same command):

  1. design     the main run's 25,254 trials, regenerated with the same seeded generator and
                settings (MAIN_* constants below); identical trial_ids, items, cues and x
  2. cue check  vignette identification: vignettes the model fails to identify (< 0.7 on the
                correct group) are replaced by a passing vignette of the same group, or the trial
                is dropped (the preregistered rule)
  3. stance     perceived stance g: how likely the model thinks a person with each cue agrees
                with each hypothesis, relative to an unspecified person
  4. trials     three fresh conversations per trial: prior, paraphrased prior, posterior (after
                the results table); answers below the coverage cutoff are re-prompted once
  5. derive     exclusions, log-odds, update y, congruence m (and the survey and behavioral
                stance variants), the descriptive model columns; written to --out

Backends
  --backend openrouter --model <OpenRouter id | gpt | claude | gemini>
        API models through OpenRouter. The key is read from the environment variable
        OPENROUTER_API_KEY (or a .env file in the repository root). Default answer format: a
        structured integer reply {"probability": k}; the readout is the token distribution
        (`--readout dist`, needs a provider that returns logprobs) or K sampled answers
        (`--readout sample`). A cost estimate is printed before anything is sent, the run
        refuses to start above --max-cost (default $15) and stops when spend reaches it.
  --backend hf --model <Hugging Face id | olmo_instruct | llama | ...> [--revision <commit>]
        Open weights on a local GPU: the full next-token distribution over the integers
        0-100 at the answer position. A 7-8B model in bf16 needs one ~48 GB GPU at the
        default batch size (lower --batch-size for smaller GPUs).
  --backend mock
        A simulated model with planted effects: checks the whole pipeline in seconds at no cost.

Our model keys (gpt, llama, olmo_instruct, olmo_sft, claude, ...; see src/idea/models.py) select
the pinned model revision, readout and answer format of the main run, and that model's part of
the design (e.g. labels only for the OLMo SFT/DPO/base stages). Any other id is run on the full
design unless --tiers / --cue-types / --n-trials / --subset narrow it.

Always start with --dry-run: it prints the number of calls, the cost estimate and a few rendered
prompts, and sends nothing.

Examples
    python elicit.py --backend openrouter --model openai/gpt-4o-2024-11-20 --dry-run
    python elicit.py --backend openrouter --model gpt --out my_gpt.csv.gz
    python elicit.py --backend openrouter --model google/gemini-2.5-flash --readout sample \\
        --sample-k 1 --extra-body '{"reasoning": {"enabled": false}}' --n-trials 2000 --out gem.csv.gz
    python elicit.py --backend hf --model olmo_instruct --out my_olmo.csv.gz
    python elicit.py --backend hf --model allenai/OLMo-2-1124-7B --fmt completion --cue-types label

Elicitation is stochastic: sampled answers, provider-side model updates and GPU batching
nondeterminism mean a re-run gives a comparable dataset, not an identical one.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from idea.cues import UNSPECIFIED, CueLibrary, identification_requests, load_cues, score_identification  # noqa: E402
from idea.design import DesignConfig, build_trials  # noqa: E402
from idea.elicit import (BudgetExceeded, Cache, CachedElicitor, DeadlineReached, OpenRouterBackend,  # noqa: E402
                         Request, make_backend)
from idea.models import MODELS, ModelSpec  # noqa: E402
from idea.paths import DATA_DIR  # noqa: E402
from idea.prompts import load_topics  # noqa: E402
from idea.runner import calls_for, derive, make_request, run_trials  # noqa: E402
from idea.stance import attach_g, attach_survey_gap, g_behavioral, g_survey, stance_requests, compute_g  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Settings of the main run. Changing any MAIN_DESIGN value gives a
# different set of trials.
# ---------------------------------------------------------------------------------------------
MAIN_SEED = 20261110
MAIN_DESIGN = dict(
    q_values=(0.58, 0.60, 0.62),   # H1's success rate under the policy (1 - q without)
    n_range=(10, 30),              # group sizes of the 2x2 results tables
    rate_range=(0.20, 0.80),       # observed success rates allowed in a table
    n_neutral=75,                  # Tier 2 trials with no stated belief, per group x cue type
    n_belief=318,                  # Tier 2 trials with a stated belief (half each way), per group x cue type
    tier1_scale=0.4,               # Tier 1 counts as a fraction of Tier 2
    items_per_topic_dir=30,        # evidence tables per topic x H1 direction
    p_match=0.75,                  # share of trials on a topic that engages the cue's dimension
    complement_frac=0.0,           # P(H0) questions: pilot only
    calibration_frac=0.06,         # trials also read out by sampling (open models; a readout check)
)
# Inputs the main design took from the pilot (pilot-296551ab): the topics kept by the stance
# screen, the topics each identity dimension engages (pairing rule), the Tier 1 (topic, cue) pairs
# of the |g| < 0.25 robustness subset, and the pilot's mean prior per cell (ceiling control on x).
DESIGN_DIR = DATA_DIR / "design"
SCREEN_FILE = DESIGN_DIR / "main_screen.json"
PRIOR_MEANS_FILE = DESIGN_DIR / "main_prior_means.json"

COVERAGE_MIN = 0.9        # valid-integer mass required (chat); completion format uses 0.5 (runner.derive)
TEMPERATURE = 1.0         # sampled answers
SAMPLE_K = 10             # sampled answers per question (sampling readout and calibration trials)
VIGNETTE_PASS = 0.7       # cue check: probability on the correct group a vignette needs
BATCH_SIZE = 32           # HF forward-pass batch (one 48 GB GPU, 7-8B model)
CONCURRENCY = 24          # parallel OpenRouter requests
CHUNK = 512               # requests per cache checkpoint
MAX_COST = 15.0           # USD per run (OpenRouter)

# The main run's per-model departures from the full design (budget and stage comparisons).
MAIN_RUN_MODEL_DESIGN = {
    "olmo_base": dict(cue_types=("label",)),
    "olmo_sft": dict(cue_types=("label",)),
    "olmo_dpo": dict(cue_types=("label",)),
    "olmo_instruct:completion": dict(cue_types=("label",)),
    # Claude Sonnet 5: a seeded 6,000-trial subset of the labels-only trials, 1 sampled answer per
    # question (5 for stance), paraphrased prior on 300 trials.
    "claude": dict(cue_types=("label",), tier_subset=dict(n_tier2=5000, n_tier1=1000, seed=20261115),
                   retest_n=300, sample_k=1, stance_sample_k=5),
    # Gemini 2.5 Flash: full design, 1 sampled answer per question, paraphrased prior on 1,500 trials.
    "gemini": dict(retest_n=1500, sample_k=1, stance_sample_k=1),
}

FAMILY = {"olmo": "OLMo 2", "olmo13": "OLMo 2", "tulu": "Tulu 3 (Llama 3.1)", "llama": "Llama 3.1",
          "phi4": "Phi-4", "gpt": "GPT-4o", "claude": "Claude Sonnet 5", "gemini": "Gemini 2.5 Flash"}
SIZE_B = {"olmo": 7, "olmo13": 13, "tulu": 8, "llama": 8, "phi4": 14}

# Column order of datasets/main_trials.csv.gz (see datasets/CODEBOOK.md).
SCHEMA = [
    "study", "model", "fmt", "trial_id", "base_model", "model_family", "model_size_b", "training_stage",
    "variant_tag", "prompt_variant", "debias_system_prompt", "open_weights", "hf_id", "hf_revision", "api_id",
    "design", "n_trials_model_fmt", "item_id", "base_slot", "tier", "topic", "h_dir", "n_t", "k_t", "n_c", "k_c",
    "q", "D", "table_x", "x", "heuristic_conflict", "group", "dimension", "cue_id", "cue_type", "b",
    "belief_dir", "source", "matched_pair", "results_as_list", "treated_first", "complement", "calibration",
    "d", "kappa", "e", "tier1_low_g", "retest", "p_prior", "cov_prior", "att_prior", "model_version", "p_post",
    "cov_post", "att_post", "p_prior_rep", "cov_prior_rep", "att_prior_rep", "readout", "p_prior_s",
    "cov_prior_s", "att_prior_s", "p_post_s", "cov_post_s", "att_post_s", "excluded", "lp_before",
    "lp_before_rep", "lp_after", "y", "omega", "bayes_post", "contra", "support", "has_belief", "g", "m",
    "g_beh", "m_beh", "g_svy", "m_svy", "has_v", "v_k", "excluded_cov08",
]

EXIT_DEADLINE, EXIT_BUDGET = 3, 4


# ---------------------------------------------------------------------------------------------
# 1. Design
# ---------------------------------------------------------------------------------------------
def build_design() -> tuple[pd.DataFrame, list, CueLibrary]:
    """The main run's trial table (25,254 trials), exactly as generated for the study."""
    screen = json.loads(SCREEN_FILE.read_text())
    cfg = DesignConfig.from_dict({**MAIN_DESIGN, "seed": MAIN_SEED})
    cfg.engaged = screen["engaged"]
    pm = json.loads(PRIOR_MEANS_FILE.read_text())
    cfg.prior_means = {tuple(json.loads(k)): v for k, v in pm.items()}
    topics = [t for t in load_topics() if t.topic_id in set(screen["keep"])]
    lib = load_cues(include_draft=True)
    _, trials = build_trials(topics, lib, cfg)
    # Tier 1 robustness subset: every Tier 2 trial, Unspecified-cue Tier 1 trials, and Tier 1
    # trials whose (topic, cue) pair had |g| < 0.25 in the pilot.
    pairs = {tuple(x) for x in screen["tier1_low_g"]}
    in_pair = pd.Series([(t, c) in pairs for t, c in zip(trials.topic, trials.cue_id)], index=trials.index)
    trials["tier1_low_g"] = (trials.tier != 1) | (trials.group == UNSPECIFIED) | in_pair
    return trials, topics, lib


def reassign_failed_vignettes(trials: pd.DataFrame, failed: set, lib: CueLibrary, seed: int = MAIN_SEED):
    """Replace each failed vignette by a random passing vignette of the same group; drop the trial
    when the group has none."""
    if not failed:
        return trials, 0
    rng = random.Random(seed)
    t = trials.copy()
    drop = []
    for i in t.index[t.cue_id.isin(failed)]:
        pool = [c.cue_id for c in lib.select(group_id=t.at[i, "group"], cue_type="vignette") if c.cue_id not in failed]
        if pool:
            t.at[i, "cue_id"] = rng.choice(pool)
        else:
            drop.append(i)
    return t.drop(index=drop), len(drop)


def select_trials(trials: pd.DataFrame, a) -> pd.DataFrame:
    t = trials[trials.tier.isin(a.tiers) & trials.cue_type.isin(a.cue_types)]
    if a.subset:
        ids = set(pd.read_csv(a.subset).trial_id) if a.subset.endswith((".csv", ".gz")) else \
            set(Path(a.subset).read_text().split())
        t = t[t.trial_id.isin(ids)]
    if a.tier_subset:   # fixed seeded subset per tier (as the main run drew Claude's trials)
        rs = np.random.default_rng(int(a.tier_subset.get("seed", 0)))
        parts = []
        for tier, n in ((2, a.tier_subset.get("n_tier2")), (1, a.tier_subset.get("n_tier1"))):
            pool = t[t.tier == tier]
            parts.append(pool.iloc[np.sort(rs.permutation(len(pool))[:int(n)])] if n is not None else pool)
        t = pd.concat(parts).sort_index()
    if a.n_trials and a.n_trials < len(t):
        rs = np.random.default_rng(a.subset_seed)
        t = t.iloc[np.sort(rs.permutation(len(t))[:a.n_trials])]
    if a.retest_n:      # paraphrased prior asked on a seeded subset only
        rs = np.random.default_rng(MAIN_SEED + 99)
        keep = set(t.trial_id.iloc[rs.permutation(len(t))[:int(a.retest_n)]])
        t = t.assign(retest=t.trial_id.isin(keep))
    return t


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------
def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_id.split("/")[-1].lower()).strip("_")


def resolve_model(a) -> ModelSpec:
    """Our key -> the pinned spec of the main run; anything else -> a new spec for that id."""
    s = MODELS.get(a.model)
    if a.backend == "mock":
        s = MODELS["mock"]
        return replace(s, key=a.name or "mock")
    if s is not None and s.backend == a.backend and a.backend in ("hf", "openrouter"):
        over = {}
        if a.revision:
            over["revision"] = a.revision
        if a.readout:
            over["readout"] = a.readout
        if a.answer_format:
            over["answer_format"] = a.answer_format
        if a.extra_body:
            over["extra_body"] = json.loads(a.extra_body)
        return replace(s, key=a.name or s.key, **over)
    if a.backend == "hf":
        return ModelSpec(a.name or slug(a.model), hf_id=a.model, revision=a.revision or "main",
                         formats=("chat", "completion"), readout="dist")
    return ModelSpec(a.name or slug(a.model), openrouter_id=a.model, backend="openrouter",
                     readout=a.readout or "dist", open_weights=False,
                     answer_format=a.answer_format or "json",
                     extra_body=json.loads(a.extra_body) if a.extra_body else {})


def model_columns(spec: ModelSpec, registry_key: str | None) -> dict:
    """Descriptive model columns (as in the released dataset)."""
    key = registry_key or spec.key
    root = key.split("_")[0]
    known = registry_key is not None
    stage = spec.stage or ("instruct" if spec.open_weights and root in ("llama", "phi4") and known else None)
    return dict(
        base_model=spec.key,
        model_family=FAMILY.get(root, root) if known else (spec.hf_id or spec.openrouter_id or spec.key),
        model_size_b=SIZE_B.get(root, np.nan) if known else np.nan,
        training_stage=stage or ("api_chat" if not spec.open_weights else ""),
        hf_id=spec.hf_id or "", hf_revision=spec.revision if spec.hf_id else "", api_id=spec.openrouter_id or "",
        open_weights=bool(spec.open_weights), prompt_variant=int(spec.prompt_variant),
        debias_system_prompt=bool(spec.system_prompt), variant_tag="",
    )


def make_elicitor(spec, a, cache: Cache, sample_k: int, max_cost: float | None):
    if spec.backend == "hf":
        kw = dict(batch_size=a.batch_size, coverage_min=COVERAGE_MIN, sample_k=sample_k, temperature=TEMPERATURE,
                  dtype=a.dtype)
    elif spec.backend == "openrouter":
        kw = dict(concurrency=a.concurrency, sample_k=sample_k, coverage_min=COVERAGE_MIN, max_cost=max_cost,
                  temperature=TEMPERATURE, api_key_env=a.api_key_env)
    else:
        kw = dict(sample_k=sample_k)
    return CachedElicitor(make_backend(spec, **kw), cache, chunk=CHUNK)


# ---------------------------------------------------------------------------------------------
# Requests per stage
# ---------------------------------------------------------------------------------------------
def cue_check_requests(lib: CueLibrary):
    meta = identification_requests(lib, leakage=True)
    reqs = []
    for m in meta:
        ti = m["options"].index(m["group_id"]) if m["group_id"] in m["options"] else None
        reqs.append(Request("choice", "chat", messages=[{"role": "user", "content": m["prompt"]}],
                            letters=m["letters"], meta=dict(call="ident", target_index=ti,
                                                            ask_dim=m["ask_dim"], cue_dim=m["cue_dim"])))
    return reqs, meta


def stance_cues(lib: CueLibrary, cue_types) -> CueLibrary:
    return CueLibrary([c for c in lib.cues if c.cue_type in cue_types], lib.options)


def trial_requests(trials, topics, lib, fmt, spec, calibration):
    tmap = {t.topic_id: t for t in topics}
    out = []
    for row in trials.itertuples(index=False):
        for call in calls_for(row, calibration):
            out.append(make_request(row, tmap, lib, fmt, call, variant=spec.prompt_variant,
                                    system=spec.system_prompt, belief_strength=spec.belief_strength))
    return out


def unique(reqs):
    seen, out = set(), []
    for r in reqs:
        k = r.prompt_repr() + str(r.readout) + str(r.answer_format)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def show(req: Request) -> str:
    if req.fmt == "chat":
        return "\n".join(f"[{m['role']}] {m['content']}" for m in req.messages)
    return req.text


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["openrouter", "hf", "mock"])
    ap.add_argument("--model", help="OpenRouter id, Hugging Face id, or one of our keys (src/idea/models.py)")
    ap.add_argument("--revision", help="hf: weights commit (default: our pinned commit for our keys, else main)")
    ap.add_argument("--name", help="value of the `model` column (default: our key, or a slug of the id)")
    ap.add_argument("--fmt", choices=["chat", "completion"],
                    help="chat template or plain-text completion (default: chat if the model has one)")
    ap.add_argument("--out", default="my_trials.csv.gz", help="output CSV (gzip if it ends in .gz)")
    ap.add_argument("--study", default="main", help="value of the `study` column (the analysis scripts use 'main')")
    ap.add_argument("--dry-run", action="store_true", help="print call counts, cost estimate and prompts; send nothing")
    ap.add_argument("--yes", action="store_true", help="openrouter: start without the confirmation prompt")
    # design subset
    ap.add_argument("--tiers", default=None, help="comma list of tiers, e.g. 2 (default: 1,2)")
    ap.add_argument("--cue-types", default=None, help="comma list: label,vignette (default: both)")
    ap.add_argument("--n-trials", type=int, help="seeded random subset of this many trials (after the filters)")
    ap.add_argument("--subset-seed", type=int, default=20261115, help="seed of the --n-trials subset")
    ap.add_argument("--subset", help="file of trial_ids to run (a .csv with a trial_id column, or one id per line)")
    ap.add_argument("--retest-n", type=int, help="ask the paraphrased prior on this many trials only (default: all)")
    ap.add_argument("--no-calibration", action="store_true", help="hf: skip the sampled-readout calibration trials")
    # readout
    ap.add_argument("--readout", choices=["dist", "sample"],
                    help="openrouter: token distribution (needs logprobs) or sampled answers (default: dist)")
    ap.add_argument("--answer-format", choices=["json", "tool", "text"],
                    help="openrouter: structured integer (json, default), forced tool call, or free text")
    ap.add_argument("--sample-k", type=int, help=f"sampled answers per question (default {SAMPLE_K})")
    ap.add_argument("--stance-sample-k", type=int, help="sampled answers per stance/cue-check question (default: --sample-k)")
    ap.add_argument("--extra-body", help='openrouter: JSON merged into each request, e.g. \'{"reasoning": {"enabled": false}}\'')
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="environment variable holding the key")
    ap.add_argument("--max-cost", type=float, default=MAX_COST, help="openrouter: USD cap for this run (default 15)")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--price-in", type=float, help="USD per million prompt tokens (default: OpenRouter's list price)")
    ap.add_argument("--price-out", type=float, help="USD per million completion tokens")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="hf: forward-pass batch")
    ap.add_argument("--dtype", default="auto", help="hf: torch dtype (auto = bf16 on GPU)")
    ap.add_argument("--cache-dir", default="elicit_cache", help="answer cache (rerun the same command to resume)")
    a = ap.parse_args(argv)
    if a.backend != "mock" and not a.model:
        ap.error("--model is required")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    spec = resolve_model(a)
    registry_key = a.model if (a.model in MODELS and MODELS[a.model].backend == a.backend) else None
    fmt = a.fmt or ("chat" if "chat" in spec.formats else "completion")
    if spec.backend == "openrouter" and fmt != "chat":
        raise SystemExit("API models use the chat format")
    # the main run's design for this model (our keys only), unless overridden on the command line
    md = {}
    if registry_key:
        md = MAIN_RUN_MODEL_DESIGN.get(f"{registry_key}:{fmt}") or MAIN_RUN_MODEL_DESIGN.get(registry_key) or {}
    a.tiers = tuple(int(x) for x in a.tiers.split(",")) if a.tiers else (1, 2)
    a.cue_types = tuple(a.cue_types.split(",")) if a.cue_types else md.get("cue_types", ("label", "vignette"))
    a.tier_subset = None if (a.n_trials or a.subset) else md.get("tier_subset")
    a.retest_n = a.retest_n or md.get("retest_n")
    sample_k = a.sample_k or md.get("sample_k", SAMPLE_K)
    stance_k = a.stance_sample_k or md.get("stance_sample_k", sample_k)
    calibration = spec.open_weights and spec.readout == "dist" and not a.no_calibration

    # 1. design
    design, topics, lib = build_design()
    print(f"design: {len(design):,} trials ({design.item_id.nunique()} base items, {design.topic.nunique()} topics)")
    pre = select_trials(design, a)          # to size the stages; the cue check may still swap vignettes
    if pre.empty:
        raise SystemExit("no trials selected")
    need_cue_check = (pre.cue_type == "vignette").any()
    used_topics = [t for t in topics if t.topic_id in set(pre.topic)]
    slib = stance_cues(lib, set(pre.cue_type))
    print(f"model: {spec.key} ({spec.hf_id or spec.openrouter_id or 'simulated'}), backend {spec.backend}, "
          f"format {fmt}, readout {spec.readout}" + (f", answer format {spec.answer_format}"
                                                       if spec.backend == "openrouter" else ""))
    print(f"selected: {len(pre):,} trials, tiers {a.tiers}, cue types {a.cue_types}"
          + (f", paraphrased prior on {a.retest_n}" if a.retest_n else ""))

    cue_reqs, cue_meta = cue_check_requests(lib) if need_cue_check else ([], [])
    st_reqs, _ = stance_requests(slib, used_topics, fmt)
    tr_reqs = unique(trial_requests(pre, topics, lib, fmt, spec, calibration))
    stage_counts = dict(cue_check=len(cue_reqs), stance=len(unique(st_reqs)), trials=len(tr_reqs))
    print("unique questions per stage:", stage_counts)

    est = None
    if spec.backend == "openrouter":
        est_b = OpenRouterBackend(spec, api_key="not-used-for-the-estimate", sample_k=stance_k)
        if a.price_in is not None and a.price_out is not None:
            est_b._pricing = {"prompt": a.price_in / 1e6, "completion": a.price_out / 1e6}
        try:
            e1 = est_b.estimate_cost(cue_reqs + unique(st_reqs))
            est_b.sample_k = sample_k
            e2 = est_b.estimate_cost(tr_reqs)
            est = e1 + e2
            pr = est_b.pricing()
            print(f"price: ${pr['prompt'] * 1e6:.2f} / ${pr['completion'] * 1e6:.2f} per million prompt / completion tokens")
            per = ("one call per question (distribution readout)" if spec.readout == "dist" else
                   f"{sample_k} sampled answer(s) per trial question, {stance_k} per stance question")
            print(f"estimated cost: ${est:.2f} (cue check + stance ${e1:.2f}, trials ${e2:.2f}; {per}; "
                  f"x1.25 margin, re-prompts not included); cap ${a.max_cost:.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"could not look up the price ({e!r}); pass --price-in and --price-out (USD per million tokens)")
        if spec.readout == "dist":
            print("note: --readout dist needs a provider that returns token logprobs; without them every answer\n"
                  "      has coverage 0 and is excluded. Use --readout sample for models without logprobs.")
    elif spec.backend == "hf":
        print(f"forward passes: ~{sum(stage_counts.values()):,} (+ {SAMPLE_K} sampled answers per calibration question)"
              f"; batch size {a.batch_size}")

    if a.dry_run:
        t0 = pre.iloc[0]
        tmap = {t.topic_id: t for t in topics}
        print("\n--- example prompts (trial %s) ---" % t0.trial_id)
        for call in ("prior", "prior_rep", "post"):
            r = make_request(t0, tmap, lib, fmt, call, variant=spec.prompt_variant, system=spec.system_prompt)
            print(f"\n[{call}]\n{show(r)}")
        print("\n[stance]\n" + show(st_reqs[0]))
        if cue_reqs:
            print("\n[cue check]\n" + show(cue_reqs[0]))
        print("\ndry run: nothing was sent.")
        return 0

    if spec.backend == "openrouter":
        if est is None:
            raise SystemExit("no cost estimate; refusing to start (pass --price-in/--price-out)")
        import os
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        if not os.environ.get(a.api_key_env):
            raise SystemExit(f"{a.api_key_env} is not set: export your OpenRouter key (or put it in {ROOT / '.env'})")
        if est > a.max_cost:
            print(f"estimated ${est:.2f} exceeds --max-cost ${a.max_cost:.2f}; nothing sent. "
                  "Narrow the run (--n-trials, --tiers, --cue-types) or raise --max-cost.", file=sys.stderr)
            return EXIT_BUDGET
        if not a.yes and input(f"Start? This spends up to ${a.max_cost:.2f} on {a.api_key_env}. [y/N] ").strip().lower() != "y":
            print("not started.")
            return 0

    cache_dir = Path(a.cache_dir)
    cache = Cache(cache_dir / f"{spec.key}_{fmt}.sqlite")
    try:
        el_st = make_elicitor(spec, a, cache, stance_k, a.max_cost)
        el_tr = el_st if (stance_k == sample_k or spec.backend == "hf") else None

        # 2. cue check
        failed = set()
        if need_cue_check:
            print("\n[2] cue check")
            res = el_st.run(cue_reqs)
            rows = pd.DataFrame(cue_meta)
            rows["probs"] = [r.probs or [np.nan] * len(m["options"]) for r, m in zip(res, cue_meta)]
            rows["coverage"] = [r.coverage for r in res]
            sc = score_identification(rows, VIGNETTE_PASS)
            failed = set(sc[~sc.passed].cue_id)
            print(f"vignettes failed: {len(failed)} of {len(sc)}")
        full, n_drop = reassign_failed_vignettes(design, failed, lib)
        trials = select_trials(full, a)
        if n_drop:
            print(f"dropped {n_drop} design trials: no passing vignette for their group")

        # 3. stance
        print("\n[3] perceived stance")
        st_res = el_st.run(st_reqs)
        _, index = stance_requests(slib, used_topics, fmt)
        st = pd.DataFrame(index)
        st["p"] = [r.p for r in st_res]
        st["coverage"] = [r.coverage for r in st_res]
        st["model"] = spec.key
        st = compute_g(st)

        # 4. trials
        print(f"\n[4] trials: {len(trials):,}")
        if el_tr is None:
            left = a.max_cost - el_st.b.usage["cost"]
            el_tr = make_elicitor(spec, a, cache, sample_k, left)
        res = run_trials(trials, topics, lib, el_tr, fmt, capture=False, calibration=calibration)
        for el in {id(el_st): el_st, id(el_tr): el_tr}.values():
            if hasattr(el.b, "usage"):
                print("API usage:", el.b.usage)
    except DeadlineReached as e:
        print(f"stopped: {e}", file=sys.stderr)
        return EXIT_DEADLINE
    except BudgetExceeded as e:
        print(f"spending cap: {e}. Nothing further was sent; answers so far are cached "
              "(rerun to resume with a higher --max-cost).", file=sys.stderr)
        return EXIT_BUDGET

    # 5. derived columns
    df = build_columns(res, st, spec, registry_key, a.study, fmt, subset=bool(a.n_trials or a.subset or a.tier_subset))
    out = Path(a.out)
    df.to_csv(out, index=False, compression="gzip" if out.suffix == ".gz" else None)
    print(f"\nwrote {out}: {len(df):,} rows x {df.shape[1]} columns; excluded {df.excluded.mean():.1%}")
    return 0


def build_columns(res: pd.DataFrame, stance: pd.DataFrame, spec: ModelSpec, registry_key, study: str, fmt: str,
                  subset: bool = False) -> pd.DataFrame:
    """Derived analysis columns, exactly as the released dataset was built."""
    df = derive(res)
    df = attach_g(df, stance)
    gb = g_behavioral(df)
    df = df.merge(gb[["model", "cue_id", "topic", "h_dir", "g_beh"]], on=["model", "cue_id", "topic", "h_dir"],
                  how="left")
    df["m_beh"] = df.b * df.g_beh.fillna(0.0)
    s = g_survey(pd.read_csv(DATA_DIR / "survey_stance.csv"))
    df = df.merge(s[["topic", "h_dir", "group", "g_svy"]], on=["topic", "h_dir", "group"], how="left")
    df["m_svy"] = df.b * df.g_svy.fillna(0.0)
    df = attach_survey_gap(df, load_topics())

    df.insert(0, "study", study)
    for k, v in model_columns(spec, registry_key).items():
        df[k] = v
    cov = df[[c for c in ("cov_prior", "cov_prior_rep", "cov_post") if c in df]].fillna(0).min(axis=1)
    strict = (df.fmt == "completion") & (df.readout == "dist") & (cov < 0.8)
    df["excluded_cov08"] = df.excluded | strict
    has_vig = (df.cue_type == "vignette").any()
    df["design"] = ("full" if has_vig else "labels_only") + ("_subset" if subset else "")
    df["n_trials_model_fmt"] = len(df)
    df["retest"] = df["retest"].astype("boolean").fillna(True).astype(bool) if "retest" in df else True
    for c in SCHEMA:
        if c not in df:
            df[c] = np.nan
    return df[SCHEMA].sort_values("trial_id").reset_index(drop=True)


if __name__ == "__main__":
    sys.exit(main())
