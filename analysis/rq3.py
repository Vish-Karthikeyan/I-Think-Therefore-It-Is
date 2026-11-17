#!/usr/bin/env python
"""RQ3: where the belief-following lives inside open models, and whether removing it is safe.

Unlike RQ1 and RQ2, RQ3 cannot be computed from the dataset alone: it needs the model weights and a GPU.
This script re-runs the RQ3 experiments on a local open model (OLMo 2 7B Instruct, Llama 3.1 8B Instruct,
Tulu 3 8B final; Hugging Face, revisions pinned in src/idea/models.py) using the trials in
datasets/main_trials.csv.gz. Every prompt is re-rendered from the trial's columns with the `idea` package,
exactly as in the study. Outputs (activations, subspaces, steered answers, benchmark scores) go to --out;
the result tables are printed (and written as CSV with --save).

    python analysis/rq3.py --list                          # the analyses and what they do
    python analysis/rq3.py --models olmo_instruct          # default analyses: corrected + audit
    python analysis/rq3.py --only geometry,dose,stability  # add-ons (need the corrected run's outputs)
    python analysis/rq3.py --only corrected --use-released-vectors   # steer with steering_vectors/ instead
    python analysis/rq3.py --summarize-only --out rq3_output         # tables from existing outputs, no GPU
    python analysis/rq3.py --smoke --hf-model olmo_instruct=allenai/OLMo-2-0425-1B-Instruct --out /tmp/rq3_smoke
                                                           # CPU smoke test on a 1B model (numbers meaningless)

Analyses (toggle in ANALYSES below, or --only / --skip):
  corrected     Corrected RQ3 at the answer position (the position that produced the answer). Captures
                answer-position activations of every training-topic trial and of 1,500 counterfactual
                triples (each trial rendered as: user believes H1 / believes H0 / states no belief); builds
                the belief subspaces (U2 = the two mean belief displacements, U4 = top-4 SVD), the
                belief-sentence subspaces, random controls, and the caved-minus-resisted vector; geometry by
                layer; sufficiency (interchange patching of each subspace between counterfactual inputs, 300
                held-out trials); then a validation-topic sweep of five intervention families (caved-resisted
                vector at one layer / at all layers, erasure of the answer subspace, erasure at the belief
                tokens, at several early layers) and one held-out test of the best setting per family plus a
                matched random-subspace control, with the preregistered, exploratory and partial criteria.
  audit         Capability and bias audit of the erasure intervention vs the unsteered model vs its random
                control: MMLU-Pro (14 subjects), BBQ, SycEval (via lm-eval and idea.audit), and L2D only if
                you supply its items in data/l2d_external/exp1_prompts.json (not redistributed).
  geometry      What the subspace encodes: Cohen's d of the stated belief relative to the scored hypothesis (b)
                vs the concrete claim (belief_dir) in U2/U4 at the erase layer; layer-wise formation of the
                split (answer position and belief tokens, axis fit on training topics, tested on held-out
                ones); erasure followed downstream (GPU part: one extra pass over the counterfactual trials).
  dose          Dose-response: does a trial's position on the belief axis predict how much it discounts the
                contradicting evidence (cross-fitted axis, within-belief correlations)? CPU only.
  stability     Subspace stability under trial and topic resampling, split halves, unseen topics, against
                chance (CPU, from the capture), and under the two prompt rewordings (GPU).
  prereg        Preregistered RQ3 (prompt-end activations): masking probe + masking index, d_bias directions
                (caved minus resisted, matched strata, topic PCs removed), validation sweep, one held-out test
                with controls and amplification, per-dimension directions, angles to other cue-induced biases.
  prereg_audit  Preregistered audit: MMLU-Pro (500 items/subject), BBQ, SycEval, unsteered / steered / random.
  p4            Prediction P4 at the answer position: probe for contradicted vs supported evidence and the
                bias direction's cosine to Instruct, for OLMo 2 7B SFT, DPO and Instruct (always these three).

Requirements. torch, transformers, scikit-learn, statsmodels, pandas, numpy, scipy, the `idea` package
(pip install -e .), and for the audits lm-eval (0.4.x) and `datasets`. Llama 3.1 is gated: accept Meta's
licence on Hugging Face and set HF_TOKEN.

GPU memory and time (study: one NVIDIA L40S 48 GB per model, bf16; GPU-hours from the paper's compute table):
  * One 48 GB GPU per 7-8B model (weights ~16 GB in bf16; activations are moved to host RAM as they are
    captured). Host RAM: 32 GB or more (capture.npz ~1 GB per model; the preregistered prompt-end
    activations ~2 GB per model). Smaller GPUs work with a smaller --fwd-batch and --batch-size.
  * corrected     ~3.5 GPU-h per model (10.4 GPU-h for the three models).
  * audit         three conditions: ~2-6 GPU-h per model, dominated by MMLU-Pro (study: 17.6 GPU-h for the
                  automatic choice and 8.6 GPU-h for the named erasure audits, all three models).
  * geometry/dose ~0.1 GPU-h per model; stability: CPU minutes per model + ~0.15 GPU-h per model (rewording).
  * prereg        ~13 GPU-h per model (study: 26.1 GPU-h for two models, plus the prompt-end capture).
  * prereg_audit  ~35 GPU-h per model (MMLU-Pro at 500 items/subject x 3 conditions; study: ~72 GPU-h, 2 models).
  * p4            ~0.1 GPU-h per stage (0.3 GPU-h for the three).
Every stage saves its outputs and steered answers are cached in SQLite (--out/cache/), so a run stopped at a
time limit resumes where it stopped when restarted with the same arguments.

Reproducibility. The trial selections (which trials are counterfactual, held out, swept and tested) are
reproduced exactly from the dataset (checked against the study's saved trial lists). Activations and answers
depend on the GPU, dtype and batch composition, so numbers agree with the paper's to the precision of such
floating-point differences, not bit for bit. --use-released-vectors steers with the released vectors.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================================================
# Toggles: set False to skip (or use --only / --skip on the command line).
# =============================================================================================================
ANALYSES = {
    "corrected": True,      # answer-position RQ3: capture, subspaces, geometry, sufficiency, sweep, held-out test
    "audit": True,          # MMLU-Pro / BBQ / SycEval (+ L2D if supplied) for the erasure vs unsteered vs control
    "geometry": False,      # subspace content (Cohen's d, b vs concrete claim), layer-wise formation, erasure downstream
    "dose": False,          # dose-response of belief magnitude on discounting (CPU; needs corrected)
    "stability": False,     # subspace stability (resampling, unseen topics, chance; rewording on GPU)
    "prereg": False,        # preregistered RQ3: prompt-end masking probe, d_bias steering sweep + test (slow)
    "prereg_audit": False,  # preregistered audit: unsteered / steered / random (slow: ~35 GPU-h per model)
    "p4": False,            # P4 stage probes at the answer position (OLMo 2 7B SFT, DPO, Instruct)
}
NEEDS_CORRECTED = ("audit", "geometry", "dose", "stability")   # read the corrected run's outputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================================================
# Settings (the values used in the study)
# =============================================================================================================
SEED = 20261110                    # study seed; topic split uses SEED+1, counterfactual draws SEED+7
MODELS = ["olmo_instruct", "llama", "tulu_final"]   # RQ3 models (keys of idea.models.MODELS)
P4_STAGES = ["olmo_sft", "olmo_dpo", "olmo_instruct"]
DESIGN_MODEL = "gpt"               # the model whose rows carry the original trial design (no vignette was
                                   # reassigned for it); steering is evaluated on the design's trials
# The study's analysis frame listed each model's trials shard by shard (trial number mod n = the GPU job that
# elicited it). Several random draws of RQ3 index into that row order, so it is restored here.
ROW_ORDER_SHARDS = {"olmo_instruct": 8, "llama": 4, "tulu_final": 2, "olmo_sft": 3, "olmo_dpo": 3}

BACKEND = dict(batch_size=32, coverage_min=0.9, sample_k=10, temperature=1.0, dtype="auto")
FWD_BATCH = 16                     # batch size of the capture / patching forward passes

CORRECTED = dict(
    layers=[4, 8, 12, 16, 20, 24],  # captured and intervened layers (decoder-layer outputs)
    test_frac=0.3, val_frac=0.2,    # topic split: test / validation / train topics
    n_cf_train=1500, n_cf_held=300,  # counterfactual triples (training topics; held-out validation topics)
    n_eval=3000,                    # validation: n/2 Tier 2 + n/6 Tier 1 trials; test: n Tier 2 + n/3 Tier 1
    grid=None,                      # intervention grid (None = idea.mech.rq3v2.GRID)
)
PREREG = dict(
    layers=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26],   # prompt-end capture layers
    test_frac=0.3, val_frac=0.2, topic_k=(4, 8, 16), k_main=8,
    sweep_layers=[12, 16, 20, 24], multiples=(0.5, 1, 2, 4, 8), modes=("all", "last"),
    n_eval=3000, n_mmlu=400,        # n_mmlu: MMLU items for the sycophancy / authority contrast sets
)
AUDIT = dict(
    n_syceval=300,                  # MMLU validation items for SycEval
    bbq_limit=500,                  # BBQ items (lm-eval's first 500 = the age category)
    mmlu_limit=200,                 # MMLU-Pro items per subject (14 subjects), corrected audit
    prereg_mmlu_limit=500,          # MMLU-Pro items per subject, preregistered audit
    prereg_mmlu_groups=7,           # the preregistered audit ran MMLU-Pro in 7 pieces
    l2d_tuples=4000,                # L2D tuples (whole questions kept)
    lmeval_batch=24,                # same lm-eval batch size for every condition
    mmlu_subjects=None,             # None = all 14 MMLU-Pro subjects (a subset only for smoke tests)
)
STABILITY = dict(layers=[12, 16, 20, 24], ks=(2, 4), reps=200, n_split=20, n_shift=10, seed=20261116)
P4 = dict(layers=[8, 10, 12, 14, 16, 18, 20, 22, 24, 26], test_frac=0.3, summary_layers=(18, 26))

# Tiny settings for --smoke (a CPU test that every code path runs; numbers meaningless).
SMOKE = dict(
    max_trials=360, fwd_batch=8, batch_size=8,
    corrected=dict(layers=[4, 8, 12], n_cf_train=60, n_cf_held=12, n_eval=72,
                   grid=dict(cr_single_layers=(8,), cr_single_mult=(4,), cr_all_mult=(1,),
                             erase_answer_layers=(12,), erase_answer_k=(4,), erase_span_layers=(4,),
                             erase_span_k=(1,), erase_span_multi=(4, 8))),
    prereg=dict(layers=[4, 8, 12], topic_k=(2, 4), k_main=4, sweep_layers=[8], multiples=(1,), modes=("all",),
                n_eval=72, n_mmlu=12),
    audit=dict(n_syceval=6, bbq_limit=4, mmlu_limit=1, prereg_mmlu_limit=1, prereg_mmlu_groups=7, l2d_tuples=30,
               lmeval_batch=4, mmlu_subjects=("biology", "law")),
    stability=dict(layers=[8, 12], reps=5, n_split=2, n_shift=2),
    p4=dict(layers=[4, 8, 12], summary_layers=(8, 12)),
)

BOOL_COLS = ["excluded", "excluded_cov08", "matched_pair", "heuristic_conflict", "results_as_list", "treated_first",
             "complement", "calibration", "tier1_low_g", "retest", "open_weights", "debias_system_prompt", "has_v"]
# The trial design (one row per trial, before any model answered): what steering evaluations run.
DESIGN_COLS = ["trial_id", "item_id", "base_slot", "tier", "topic", "h_dir", "n_t", "k_t", "n_c", "k_c", "q", "D",
               "table_x", "x", "heuristic_conflict", "group", "dimension", "cue_id", "cue_type", "b", "belief_dir",
               "source", "matched_pair", "results_as_list", "treated_first", "complement", "calibration", "d",
               "kappa", "e", "tier1_low_g"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
warnings.filterwarnings("ignore", category=FutureWarning)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# =============================================================================================================
# Data
# =============================================================================================================
def load_dataset(path) -> pd.DataFrame:
    """Read the CSV as the codebook says ("null" evidence must survive; booleans cast back)."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BOOL_COLS:
        if c in df:
            df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df


def model_rows(data: pd.DataFrame, key: str) -> pd.DataFrame:
    """One model's chat trials (the 'analysis frame' RQ3 used), in the study's row order."""
    d = data[(data.model == key) & (data.fmt == "chat")].copy()
    if d.empty:
        return d.reset_index(drop=True)
    if d.trial_id.str.fullmatch(r"t\d+").all():
        num = d.trial_id.str[1:].astype(int)
        d = d.assign(_k=num % ROW_ORDER_SHARDS.get(key, 1), _n=num).sort_values(["_k", "_n"])
        d = d.drop(columns=["_k", "_n"])
    else:
        d = d.sort_values("trial_id")
    return d.reset_index(drop=True)


def design_trials(data: pd.DataFrame) -> pd.DataFrame:
    """The original trial design, ordered by trial id (identical to the study's trials table)."""
    src = data[(data.model == DESIGN_MODEL) & (data.fmt == "chat")]
    if src.empty:
        print(f"note: no {DESIGN_MODEL!r} rows; the trial design is taken from the first model listing each trial "
              "(cue_id may then carry that model's vignette reassignments)")
        src = data[data.fmt == "chat"].drop_duplicates("trial_id")
    return src[[c for c in DESIGN_COLS if c in src]].sort_values("trial_id").reset_index(drop=True)


def subsample(d: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Smoke tests only: a random subset of trials, keeping the row order."""
    if not n or len(d) <= n:
        return d
    idx = np.sort(np.random.default_rng(seed).choice(len(d), n, replace=False))
    return d.iloc[idx].reset_index(drop=True)


# =============================================================================================================
# One open model: its trials, lazily loaded weights, request rendering and steered evaluation
# =============================================================================================================
class Model:
    def __init__(self, key: str, args, data: pd.DataFrame, design: pd.DataFrame):
        from idea.cues import load_cues
        from idea.prompts import load_topics
        self.key, self.args = key, args
        self.d = subsample(model_rows(data, key), args.max_trials)
        self.design = design
        self.topics = load_topics()
        self.tmap = {t.topic_id: t for t in self.topics}
        self.lib = load_cues(include_draft=True)
        self.out = Path(args.out) / "rq3_v2" / f"{key}_chat"          # corrected RQ3 and its add-ons
        self.out_prereg = Path(args.out) / "rq3" / f"{key}_chat"      # preregistered RQ3
        self._backend = self._elicitor = None

    @property
    def backend(self):
        """The model on the GPU (loaded on first use)."""
        if self._backend is None:
            from idea.elicit import make_backend
            from idea.models import get_model
            spec = get_model(self.key)
            if self.key in self.args.hf_override:
                hf_id, _, rev = self.args.hf_override[self.key].partition("@")
                spec = replace(spec, hf_id=hf_id, revision=rev or "main")
            log(f"loading {self.key}: {spec.hf_id}@{spec.revision}")
            kw = dict(BACKEND, batch_size=self.args.batch_size)
            if self.args.dtype:
                kw["dtype"] = self.args.dtype
            self._backend = make_backend(spec, device_map=self.args.device_map, **kw)
        return self._backend

    @property
    def n_layers(self) -> int:
        return len(self.backend.layers())

    @property
    def elicitor(self):
        """Cached elicitation (SQLite per model): rerunning an evaluation is free, so runs resume."""
        if self._elicitor is None:
            from idea.elicit import Cache, CachedElicitor
            self._elicitor = CachedElicitor(self.backend, Cache(Path(self.args.out) / "cache" / f"{self.key}.sqlite"),
                                            chunk=256)
        return self._elicitor

    def req(self, row, belief_dir=None, variant: int = 0):
        """The posterior ('post') prompt of a trial; belief_dir overrides the stated belief (counterfactuals):
        +h_dir = believes H1, -h_dir = believes H0, 0 = no belief."""
        from idea.runner import make_request
        if belief_dir is not None:
            row = row._replace(belief_dir=int(belief_dir), b=int(belief_dir * row.h_dir))
        return make_request(row, self.tmap, self.lib, "chat", "post", variant=variant)

    def evaluate(self, entries, trials: pd.DataFrame) -> pd.DataFrame:
        """Run trials (prior, paraphrased prior, posterior) with steering entries on; derived columns added."""
        from idea.runner import derive, run_trials
        self.backend.set_steering(entries)
        try:
            return derive(run_trials(trials, self.topics, self.lib, self.elicitor, "chat", calibration=False))
        finally:
            self.backend.set_steering([])

    def release(self):
        self._backend = self._elicitor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def eval_sets(design: pd.DataFrame, val_t, test_t, n_eval: int, seed: int):
    """Validation and test trials for steering, drawn exactly as in the study (same generator, same order)."""
    trials = design[design.tier.isin([1, 2])]
    rng = np.random.default_rng(seed)
    t1 = trials[trials.tier == 1]

    def sample(sub, n):
        return sub.iloc[rng.permutation(len(sub))[:n]]
    val = pd.concat([sample(trials[(trials.tier == 2) & trials.topic.isin(val_t)], n_eval // 2), sample(t1, n_eval // 6)])
    test = pd.concat([sample(trials[(trials.tier == 2) & trials.topic.isin(test_t)], n_eval), sample(t1, n_eval // 3)])
    return val, test


def flat(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not isinstance(v, dict)}


def load_npz(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def chosen_setting(out: Path, family: str = "erase_answer"):
    """The setting of `family` chosen on validation topics (summary_v2.json), e.g. 'L24_k4'."""
    f = out / "summary_v2.json"
    if not f.exists():
        return None
    return dict((fa, s) for fa, s in json.loads(f.read_text())["chosen"]).get(family)


def erase_layer_k(out: Path):
    """Layer and subspace size of the chosen answer-position erasure (OLMo L24_k4, Llama L16_k4, Tulu L24_k4)."""
    s = chosen_setting(out)
    if s is None:
        return None, None
    l, k = s.split("_k")
    return int(l[1:]), int(k)


# =============================================================================================================
# corrected: answer-position RQ3
# =============================================================================================================
def run_corrected(M: Model, S: dict):
    from idea.mech import probe, rq3v2, steer
    out = M.out
    out.mkdir(parents=True, exist_ok=True)
    d = M.d
    train_t, val_t, test_t = probe.split_topics(d.topic.unique(), S["test_frac"], S["val_frac"], SEED + 1)
    layers = [l for l in S["layers"] if l < M.n_layers]
    rng = np.random.default_rng(SEED + 7)
    dtrain, cf, held = rq3v2.select_cf_trials(d, train_t, val_t, rng, S["n_cf_train"], S["n_cf_held"])
    log(f"[corrected] {M.key}: {len(d)} trials; topics train {len(train_t)} / val {len(val_t)} / test {len(test_t)}; "
        f"{len(cf)} counterfactual + {len(held)} held-out triples; layers {layers}")

    # 1. capture
    cap_f = out / "capture.npz"
    if not cap_f.exists():
        C = rq3v2.capture_counterfactuals(M.backend, M.req, dtrain, cf, held, layers, log)
        np.savez(cap_f, **C)
        cf.to_parquet(out / "cf_trials.parquet")
        held.to_parquet(out / "held_trials.parquet")
    C = load_npz(cap_f)

    # 2. subspaces + geometry
    dir_f = out / "directions_v2.npz"
    if M.args.use_released_vectors:
        rel = ROOT / "steering_vectors" / M.key / "belief_subspaces_v2.npz"
        if not rel.exists():
            sys.exit(f"--use-released-vectors: no released subspaces for {M.key} ({rel})")
        D = load_npz(rel)
        np.savez(dir_f, **D)
        log(f"[corrected] using the released subspaces {rel.relative_to(ROOT)}")
    elif not dir_f.exists():
        np.savez(dir_f, **rq3v2.build_subspaces(C, dtrain, layers, rng, SEED))
    D = load_npz(dir_f)
    layers = [l for l in layers if f"L{l}/ans_U2" in D]
    if not (out / "geometry.csv").exists():
        rq3v2.geometry_table(C, D, layers).to_csv(out / "geometry.csv", index=False)

    # 3. sufficiency
    if not (out / "sufficiency.csv").exists():
        rq3v2.sufficiency(M.backend, M.req, held, C, D, layers, log).to_csv(out / "sufficiency.csv", index=False)
    suff = pd.read_csv(out / "sufficiency.csv")

    # 4. sweep on validation topics
    val_trials, test_trials = eval_sets(M.design, val_t, test_t, S["n_eval"], SEED)
    configs = rq3v2.intervention_grid(D, layers, S.get("grid"))
    log(f"[sweep] {len(configs)} settings x {len(val_trials)} validation trials")
    s0v = steer.slopes(M.evaluate([], val_trials))
    rows = []
    for fam, name, ent in configs:
        rep = steer.success(s0v, steer.slopes(M.evaluate(ent, val_trials)))
        rows.append(dict(family=fam, setting=name, **flat(rep)))
        log(f"[sweep] {fam} {name}: gap {rep.get('t2_contra_gap_after', np.nan):.3f} "
            f"neutral_change {rep.get('t2_neutral_change', np.nan):.3f}")
        pd.DataFrame(rows).to_csv(out / "sweep_validation_v2.csv", index=False)
    chosen = rq3v2.choose_per_family(pd.DataFrame(rows))

    # 5. held-out test: chosen setting per family + matched random-subspace control for the erasure
    test_cfgs = [c for c in configs if (c[0], c[1]) in chosen]
    for fam, name, _ in list(test_cfgs):
        if fam == "erase_answer":
            test_cfgs.append(("control_random_subspace", name, rq3v2.entries_for(D, "control_random_subspace", name, [])))
    base = M.evaluate([], test_trials)
    s0, spread0 = steer.slopes(base), steer.delta_j_spread(base)
    dims = sorted(set(base.dimension) - {"none"})
    trows = []
    for fam, name, ent in test_cfgs:
        r = M.evaluate(ent, test_trials)
        rep = steer.success(s0, steer.slopes(r), dims=dims, spread_before=spread0, spread_after=steer.delta_j_spread(r))
        fl = flat(rep)
        # exploratory criterion: belief-following removed in Tier 1 counts as success (only the Tier 1 no-belief
        # slope must stay within 0.05); preregistered criterion = rep["success"]
        expl = all(v for k, v in fl.items() if k.endswith(("_moved_toward_neutral", "_neutral_stable"))) and \
            abs(fl.get("tier1_neutral_change", 0)) < 0.05
        g0, g1 = fl.get("t2_contra_gap_before", np.nan), fl.get("t2_contra_gap_after", np.nan)
        trows.append(dict(family=fam, setting=name, success_preregistered=rep["success"], success_exploratory=bool(expl),
                          gap_closed_frac=1 - g1 / g0 if g0 else np.nan, **fl))
        log(f"[test] {fam} {name}: prereg {rep['success']} explor {expl} gap {g0:.3f}->{g1:.3f}")
    pd.DataFrame(trows).to_csv(out / "steering_test_v2.csv", index=False)
    summary = dict(model=M.key, layers=layers, n_cf_train=len(cf), n_heldout=len(held),
                   released_vectors=bool(M.args.use_released_vectors),
                   sufficiency_best=suff.sort_values("recovery", ascending=False).head(5).to_dict("records"),
                   chosen=sorted([list(c) for c in chosen]))
    (out / "summary_v2.json").write_text(json.dumps(summary, indent=1, default=str))


def summarize_corrected(root: Path, models) -> dict:
    """test, sufficiency (averaged over both swap directions) and geometry tables."""
    from idea.mech.rq3v2 import audit_bar, partial_success
    tests, suffs, geos = [], [], []
    for m in models:
        d = root / "rq3_v2" / f"{m}_chat"
        if (d / "steering_test_v2.csv").exists():
            t = pd.read_csv(d / "steering_test_v2.csv")
            bar = audit_bar(t)
            t = t.assign(model=m, audit_bar=bar,
                         success_partial=partial_success(t, bar) & (t.family != "control_random_subspace"),
                         overshoot=t.gap_closed_frac > 1.0)
            tests.append(t[[c for c in ["model", "family", "setting", "t2_contra_gap_before", "t2_contra_gap_after",
                                        "gap_closed_frac", "overshoot", "t2_neutral_change", "tier1_neutral_change",
                                        "tier1_contra_change", "tier1_support_change", "success_preregistered",
                                        "success_exploratory", "success_partial", "audit_bar"] if c in t]])
        if (d / "sufficiency.csv").exists():
            s = pd.read_csv(d / "sufficiency.csv")
            suffs.append(s.groupby(["layer", "patch"]).recovery.mean().unstack("patch").reset_index().assign(model=m))
        if (d / "geometry.csv").exists():
            geos.append(pd.read_csv(d / "geometry.csv").assign(model=m))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()  # noqa: E731
    return {"test": cat(tests), "sufficiency": cat(suffs), "geometry": cat(geos)}


# =============================================================================================================
# audit (corrected intervention): SycEval, BBQ, MMLU-Pro, L2D
# =============================================================================================================
def resolve_intervention(out: Path, choice: str):
    """'erase' (default): the answer-position erasure chosen on validation topics (the intervention audited for
    all three models in the paper); 'auto': the study's automatic choice (rq3v2.pick_winner); or 'family:setting'."""
    from idea.mech.rq3v2 import pick_winner
    if choice == "erase":
        s = chosen_setting(out)
        return ("erase_answer", s, "named") if s else (None, None, "no erase_answer setting chosen (run corrected)")
    if choice == "auto":
        w, why = pick_winner(out)
        return (w["family"], w["setting"], f"automatic ({w['criterion']})") if w else (None, None, why)
    fam, _, s = choice.partition(":")
    return fam, s, "named"


def run_audit(M: Model, A: dict):
    from idea import l2d
    from idea.audit import MMLU_PRO_SUBJECTS, lm_eval_suite, syceval_mcq, syceval_rates
    from idea.mech.other_biases import mmlu_items
    from idea.mech.rq3v2 import control_for, entries_for
    os.environ.setdefault("IDEA_LMEVAL_BATCH", str(A["lmeval_batch"]))
    adir = M.out / "audit2"
    adir.mkdir(parents=True, exist_ok=True)
    fam, setting, how = resolve_intervention(M.out, M.args.audit_intervention)
    if fam is None:
        log(f"[audit] {M.key} skipped: {how}")
        return
    if fam.startswith("erase_span"):
        log(f"[audit] {fam} acts only on stated-belief sentences, which benchmark prompts do not contain: "
            "its effect on them is zero by construction; not run")
        return
    D = load_npz(M.out / "directions_v2.npz")
    layers = sorted({int(k.split("/")[0][1:]) for k in D})
    ent = entries_for(D, fam, setting, layers)
    conds = {"unsteered": [], f"corrected-{fam}-{setting}": ent,
             f"control-{fam}-{setting}": control_for(D, fam, setting, ent, SEED)}
    for label in conds:
        if label != "unsteered":
            (adir / f"winner_{label}.json").write_text(json.dumps(dict(family=fam, setting=setting, criterion=how), indent=1))
    l2d_path = Path(M.args.l2d_items)
    for label, st in conds.items():
        f = adir / f"{label}__syceval.csv"
        if not f.exists():
            log(f"[audit] {M.key} {label}: SycEval")
            items = mmlu_items(A["n_syceval"], seed=SEED + 7, split="validation")
            M.backend.set_steering(st)
            try:
                r = syceval_rates(syceval_mcq(M.backend, items))
            finally:
                M.backend.set_steering([])
            pd.DataFrame([dict(condition=label, **{f"syceval_{k}": v for k, v in r.items()})]).to_csv(f, index=False)
        for comp, tasks, limit in (("bbq", ["bbq"], A["bbq_limit"]),
                                   ("mmlu_pro", [f"mmlu_pro_{s}" for s in (A["mmlu_subjects"] or MMLU_PRO_SUBJECTS)],
                                    A["mmlu_limit"])):
            f = adir / f"{label}__{comp}.csv"
            if f.exists():
                continue
            log(f"[audit] {M.key} {label}: {comp}")
            res, ns = lm_eval_suite(M.backend, st, tasks, limit, cache=str(adir / f"lmeval_{label}_{comp}"), with_n=True)
            rec = dict(condition=label)
            for task, metrics in res.items():
                rec.update({f"{task}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
                if ns.get(task) is not None:
                    rec[f"{task}/n"] = ns[task]
            pd.DataFrame([rec]).to_csv(f, index=False)
        f = adir / f"{label}__l2d.csv"
        if not f.exists():
            if not l2d_path.exists():
                log(f"[audit] L2D skipped: put the authors' exp1_prompts.json at {l2d_path} to include it")
                continue
            log(f"[audit] {M.key} {label}: L2D")
            items = l2d.load_items(l2d_path, n=A["l2d_tuples"], seed=SEED)
            res = l2d.run(M.backend, items, steer=st)
            res.to_parquet(adir / f"{label}__l2d_answers.parquet")
            pd.DataFrame([dict(condition=label, **l2d.metrics(res))]).to_csv(f, index=False)


def summarize_audit(root: Path, models) -> dict:
    """One row per model x condition. Labels: 'unsteered'; 'corrected'/'control' = the automatic choice
    (winner.json); '<condition>-<family>-<setting>' = a named intervention."""
    def read_one(f):
        if not f.exists():
            return None
        d = pd.read_csv(f)
        return d.iloc[0] if len(d) else None

    rows = []
    for m in models:
        adir = root / "rq3_v2" / f"{m}_chat" / "audit2"
        if not adir.exists():
            continue
        labels = {"unsteered": ("unsteered", "", "", "")}
        if (adir / "winner.json").exists():
            j = json.loads((adir / "winner.json").read_text())
            for cond in ("corrected", "control"):
                labels[cond] = (cond, j["family"], j["setting"], f"automatic ({j.get('criterion', '')})")
        for f in adir.glob("*__*.csv"):
            lab = f.name.split("__")[0]
            if lab not in labels and "-" in lab:
                cond, fam, s = lab.split("-", 2)
                labels[lab] = (cond, fam, s, "named")
        for lab, (cond, fam, s, how) in sorted(labels.items()):
            r = dict(model=m, label=lab, condition=cond, family=fam, setting=s, chosen=how)
            x = read_one(adir / f"{lab}__mmlu_pro.csv")
            if x is not None:
                subj = [c[:-2] for c in x.index if c.startswith("mmlu_pro_") and c.endswith("/n")]
                n = np.array([x[f"{q}/n"] for q in subj], float)
                acc = np.array([x[f"{q}/exact_match,custom-extract"] for q in subj], float)
                r.update(mmlu_pro_acc=float(np.sum(n * acc) / np.sum(n)), mmlu_pro_subjects=len(subj), mmlu_pro_n=int(n.sum()))
            x = read_one(adir / f"{lab}__bbq.csv")
            if x is not None:
                r.update(bbq_acc=x["bbq/acc,none"], bbq_acc_amb=x["bbq/accuracy_amb,none"],
                         bbq_acc_disamb=x["bbq/accuracy_disamb,none"], bbq_amb_bias=x["bbq/amb_bias_score,none"],
                         bbq_disamb_bias=x["bbq/disamb_bias_score,none"])
            x = read_one(adir / f"{lab}__syceval.csv")
            if x is not None:
                r.update(syceval_initial_acc=x["syceval_initial_accuracy"],
                         syceval_regressive_flip=x["syceval_regressive_flip_rate"],
                         syceval_progressive_flip=x["syceval_progressive_flip_rate"])
            x = read_one(adir / f"{lab}__l2d.csv")
            if x is not None:
                r.update(l2d_n=x["n"], l2d_parsed=x["parsed_share"], l2d_truth_discernment=x["truth_discernment"],
                         l2d_source_discernment=x["source_discernment"], l2d_correct_defense=x["correct_defense"])
            have = [c for c, k in (("syceval", "syceval_initial_acc"), ("bbq", "bbq_acc"), ("mmlu_pro", "mmlu_pro_acc"),
                                   ("l2d", "l2d_truth_discernment")) if k in r]
            r["status"] = ("complete" if len(have) == 4 and r.get("mmlu_pro_subjects") == 14
                           else "partial: " + (", ".join(have) or "nothing yet"))
            rows.append(r)
    return {"audit": pd.DataFrame(rows)}


# =============================================================================================================
# geometry: what the subspace encodes, layer-wise formation, erasure downstream
# =============================================================================================================
def make_projections(M: Model):
    """subspace_proj.parquet (each counterfactual trial, as asked, projected on U2 and the first 3 dims of U4 at
    the erase layer) and doses.parquet (the earlier content-axis dose, reported for transparency). Computed from
    the saved capture: a trial as asked is its believes-H1 (b=+1) or believes-H0 (b=-1) rendering."""
    out = M.out
    if (out / "subspace_proj.parquet").exists() and (out / "doses.parquet").exists():
        return
    E, _ = erase_layer_k(out)
    C = np.load(out / "capture.npz")
    D = load_npz(out / "directions_v2.npz")
    cf = pd.read_parquet(out / "cf_trials.parquet")
    b = cf.b.to_numpy()
    H = np.where((b > 0)[:, None], C[f"cf_pos_last_L{E}"], C[f"cf_neg_last_L{E}"]).astype(np.float64)
    p2 = H @ D[f"L{E}/ans_U2"].astype(np.float64).T
    p4 = H @ D[f"L{E}/ans_U4"].astype(np.float64)[:3].T
    pd.DataFrame(dict(trial_id=cf.trial_id.to_numpy(), belief_dir=cf.belief_dir.to_numpy(), kappa=cf.kappa.to_numpy(),
                      x=cf.x.to_numpy(), topic=cf.topic.to_numpy(), dimension=cf.dimension.to_numpy(),
                      group=cf.group.to_numpy(), u2_0=p2[:, 0], u2_1=p2[:, 1], u4_0=p4[:, 0], u4_1=p4[:, 1],
                      u4_2=p4[:, 2])).to_parquet(out / "subspace_proj.parquet", index=False)
    # content axis: mean state of trials asserting 'policy helps' minus 'hurts' (signed by belief_dir; it largely
    # reads out the answer about to be given, so it is not a clean belief dose)
    bdir = cf.belief_dir.to_numpy()
    u = H[bdir > 0].mean(0) - H[bdir < 0].mean(0)
    u = u / (np.linalg.norm(u) or 1.0)
    raw = H @ u
    pd.DataFrame(dict(trial_id=cf.trial_id.to_numpy(), b=b, belief_dir=bdir, kappa=cf.kappa.to_numpy(), x=cf.x.to_numpy(),
                      dose_raw=raw, dose=np.where(bdir != 0, np.sign(bdir) * raw, raw))).to_parquet(out / "doses.parquet",
                                                                                                    index=False)


def run_geometry(M: Model, S: dict):
    from idea.mech.rq3v2 import _orth, forward_states
    out = M.out
    make_projections(M)
    C = np.load(out / "capture.npz")
    D = load_npz(out / "directions_v2.npz")
    # Part A (CPU): every layer's counterfactual states in that layer's own 2-D subspace
    if not (out / "subspace_layers.parquet").exists():
        layers = sorted(int(k.split("/")[0][1:]) for k in D if k.endswith("/ans_U2"))
        rows = []
        for st, cond, l, (pos, ukey) in itertools.product(("cf", "held"), ("pos", "neg", "none"), layers,
                                                          (("last", "ans_U2"), ("span", "span_U2"))):
            key = f"{st}_{cond}_{pos}_L{l}"
            if key not in C.files or (pos == "span" and cond == "none"):
                continue
            P = C[key].astype(np.float64) @ D[f"L{l}/{ukey}"].astype(np.float64).T
            rows.append(pd.DataFrame(dict(set=st, cond=cond, idx=np.arange(len(P)), layer=l, pos=pos, c0=P[:, 0], c1=P[:, 1])))
        pd.concat(rows, ignore_index=True).to_parquet(out / "subspace_layers.parquet", index=False)
    # Part B (GPU): erase at the erase layer and follow the held-out trials downstream
    if (out / "subspace_erasure.parquet").exists():
        return
    E, k = erase_layer_k(out)
    nl = M.n_layers
    down = sorted({l for l in range(E + 4, nl, 4)} | {nl - 1})
    cf, held = pd.read_parquet(out / "cf_trials.parquet"), pd.read_parquet(out / "held_trials.parquet")

    def reqs_for(frame, cond):
        return [M.req(r, r.h_dir if cond == "pos" else (-r.h_dir if cond == "neg" else 0))
                for r in frame.itertuples(index=False)]
    log(f"[geometry] {M.key}: erase L{E}_k{k}, downstream layers {down}")
    S_ = {c: forward_states(M.backend, reqs_for(cf, c), down, bs=M.args.fwd_batch)[1] for c in ("pos", "neg", "none")}
    B = {}
    for l in down:   # downstream 2-D answer subspace rebuilt from the training counterfactuals (same recipe)
        B[f"L{l}/ans_U2"] = _orth(np.stack([(S_["pos"][l] - S_["none"][l]).mean(0), (S_["neg"][l] - S_["none"][l]).mean(0)]))
    np.savez(out / "bases_downstream.npz", **B)
    del S_
    U, t = D[f"L{E}/ans_U{k}"].astype(np.float32), D[f"L{E}/ans_t{k}"].astype(np.float32)
    rows, ans = [], []
    for cond in ("pos", "neg", "none"):
        rq = reqs_for(held, cond)
        for run, patch in (("unerased", None), ("erased", dict(layer=E, U=U, targets=np.tile(t, (len(rq), 1)), pos="last"))):
            p, la, _ = forward_states(M.backend, rq, down, patch=patch, bs=M.args.fwd_batch)
            ans.append(pd.DataFrame(dict(cond=cond, run=run, idx=np.arange(len(rq)), p=p)))
            for l in down:
                P = la[l].astype(np.float64) @ B[f"L{l}/ans_U2"].astype(np.float64).T
                rows.append(pd.DataFrame(dict(set="held", cond=cond, run=run, idx=np.arange(len(rq)), layer=l,
                                              c0=P[:, 0], c1=P[:, 1])))
    pd.concat(rows, ignore_index=True).to_parquet(out / "subspace_erasure.parquet", index=False)
    pd.concat(ans, ignore_index=True).to_parquet(out / "erasure_answers.parquet", index=False)


def cohen_d(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) or 1.0
    return (np.mean(a) - np.mean(b)) / sp


def _belief_axis_d(df, label="b", cols=("u2_0", "u2_1")):
    """Cohen's d along the within-subspace difference-of-class-means axis (in-sample)."""
    X = df[list(cols)].to_numpy().astype(np.float64)
    lab = df[label].to_numpy()
    w = X[lab > 0].mean(0) - X[lab < 0].mean(0)
    w = w / (np.linalg.norm(w) or 1.0)
    s = (X - X.mean(0)) @ w
    return cohen_d(s[lab > 0], s[lab < 0])


def _crossfit_d(df, label="b", cols=("u2_0", "u2_1")):
    """Cohen's d with the axis fit leaving out each trial's topic."""
    X = df[list(cols)].to_numpy().astype(np.float64)
    lab = df[label].to_numpy()
    s = np.full(len(df), np.nan)
    for t in df.topic.unique():
        te = (df.topic == t).to_numpy()
        tr = ~te
        w = X[tr & (lab > 0)].mean(0) - X[tr & (lab < 0)].mean(0)
        w /= (np.linalg.norm(w) or 1.0)
        s[te] = (X[te] - X[tr].mean(0)) @ w
    return cohen_d(s[lab > 0], s[lab < 0])


def _frame(train, label):
    """Unit axis separating label=+1 from -1 in the 2-D plane (training rows), its orthogonal, and a centre (the
    training no-belief mean if present, else the midpoint) -> a map (c0, c1) -> (x, y)."""
    X = train[["c0", "c1"]].to_numpy()
    y = train[label].to_numpy()
    w = X[y > 0].mean(0) - X[y < 0].mean(0)
    w = w / (np.linalg.norm(w) or 1.0)
    o = np.array([-w[1], w[0]])
    none = train[train.cond == "none"]
    c = none[["c0", "c1"]].to_numpy().mean(0) if len(none) else (X[y > 0].mean(0) + X[y < 0].mean(0)) / 2
    return lambda df: ((df[["c0", "c1"]].to_numpy() - c) @ w, (df[["c0", "c1"]].to_numpy() - c) @ o)


def summarize_geometry(root: Path, models) -> dict:
    """subspace_separation (Cohen's d of b vs the concrete claim at the erase layer), subspace_layers_summary
    (in-plane separation and out-of-sample d by layer and position), erasure_summary (downstream d, answers)."""
    sep, lay, era = [], [], []
    for m in models:
        d = root / "rq3_v2" / f"{m}_chat"
        if (d / "subspace_proj.parquet").exists() and (d / "cf_trials.parquet").exists():
            df = pd.read_parquet(d / "subspace_proj.parquet").merge(
                pd.read_parquet(d / "cf_trials.parquet")[["trial_id", "b", "h_dir"]], on="trial_id", how="left")
            u4 = ("u4_0", "u4_1", "u4_2")
            sep.append(dict(model=m, n=len(df), d_b_U2=round(_belief_axis_d(df, "b"), 2),
                            d_b_U2_crossfit=round(_crossfit_d(df, "b"), 2),
                            d_claim_U2=round(_belief_axis_d(df, "belief_dir"), 2),
                            d_b_U4first3=round(_belief_axis_d(df, "b", u4), 2),
                            d_claim_U4first3=round(_belief_axis_d(df, "belief_dir", u4), 2)))
        if not (d / "subspace_layers.parquet").exists():
            continue
        L = pd.read_parquet(d / "subspace_layers.parquet")
        meta = {"cf": pd.read_parquet(d / "cf_trials.parquet"), "held": pd.read_parquet(d / "held_trials.parquet")}
        hd = pd.concat([pd.DataFrame(dict(set=s, idx=np.arange(len(f)), h_dir=f.h_dir.to_numpy(), topic=f.topic.to_numpy()))
                        for s, f in meta.items()])
        L = L.merge(hd, on=["set", "idx"], how="left")
        L["b"] = L.cond.map({"pos": 1, "neg": -1, "none": 0})
        L["claim"] = L.b * L.h_dir
        rows17 = []
        for l in sorted(L.layer.unique()):
            tr = L[(L.set == "cf") & (L.pos == "last") & (L.layer == l)]
            te = L[(L.set == "held") & (L.pos == "last") & (L.layer == l)]
            if tr.empty:
                continue
            x, y = _frame(tr, "b")(te)
            bpos, bneg, bnon = te.b.values > 0, te.b.values < 0, te.b.values == 0
            sd = np.sqrt((np.var(x[bpos], ddof=1) + np.var(x[bneg], ddof=1)) / 2) or 1.0
            x, y = x / sd, y / sd
            pres = np.linalg.norm([np.r_[x[bpos], x[bneg]].mean() - x[bnon].mean(),
                                   np.r_[y[bpos], y[bneg]].mean() - y[bnon].mean()])
            rows17.append(dict(model=m, layer=l, d_H1_vs_H0=round(cohen_d(x[bpos], x[bneg]), 2),
                               presence_shift_sd=round(pres, 2)))
        rows19 = []
        for pos, label, l in itertools.product(("span", "last"), ("b", "claim"), sorted(L.layer.unique())):
            tr = L[(L.set == "cf") & (L.pos == pos) & (L.layer == l) & (L.b != 0)]
            te = L[(L.set == "held") & (L.pos == pos) & (L.layer == l) & (L.b != 0)]
            dv = np.nan
            if len(tr) >= 20 and len(te) >= 20:
                x, _ = _frame(tr, label)(te)
                yy = te[label].to_numpy()
                dv = cohen_d(x[yy > 0], x[yy < 0])
            rows19.append(dict(model=m, pos=pos, label=label, layer=l, d_oos=round(dv, 2) if np.isfinite(dv) else np.nan))
        s19 = pd.DataFrame(rows19).pivot_table(index=["model", "layer"], columns=["pos", "label"], values="d_oos")
        s19 = s19.set_axis([f"d_oos_{p}_{q}" for p, q in s19.columns], axis=1).reset_index()
        lay.append(pd.DataFrame(rows17).merge(s19, on=["model", "layer"], how="outer"))
        if (d / "subspace_erasure.parquet").exists() and (d / "erasure_answers.parquet").exists():
            Er = pd.read_parquet(d / "subspace_erasure.parquet").merge(hd[hd.set == "held"].drop(columns="set"), on="idx",
                                                                       how="left")
            Er["b"] = Er.cond.map({"pos": 1, "neg": -1, "none": 0})
            for l in sorted(Er.layer.unique()):
                un, er = Er[(Er.run == "unerased") & (Er.layer == l)], Er[(Er.run == "erased") & (Er.layer == l)]
                f = _frame(un, "b")
                xu, _ = f(un)
                xe, _ = f(er)
                era.append(dict(model=m, layer=l, d_unerased=round(cohen_d(xu[un.b.values > 0], xu[un.b.values < 0]), 2),
                                d_erased=round(cohen_d(xe[er.b.values > 0], xe[er.b.values < 0]), 2)))
            A = pd.read_parquet(d / "erasure_answers.parquet")
            A["lp"] = np.log(np.clip(A.p, .005, .995) / (1 - np.clip(A.p, .005, .995)))
            base = A[A.cond == "none"].pivot_table(index="idx", columns="run", values="lp")
            for run in ("unerased", "erased"):
                for c in ("pos", "neg"):
                    s = A[(A.cond == c) & (A.run == run)].set_index("idx")
                    eff = (1 if c == "pos" else -1) * (s.lp - base[run].reindex(s.index))
                    era.append(dict(model=m, layer="answer", run=run, cond=c, belief_effect=round(eff.mean(), 3)))
    return {"subspace_separation": pd.DataFrame(sep),
            "subspace_layers_summary": pd.concat(lay, ignore_index=True) if lay else pd.DataFrame(),
            "erasure_summary": pd.DataFrame(era)}


# =============================================================================================================
# dose: does belief magnitude predict discounting?
# =============================================================================================================
def summarize_dose(root: Path, models) -> dict:
    """RESPONSE = b * (lp_after - bayes_post) on contradicting trials (positive = stayed toward the belief).
    DOSE = b-signed position on the belief axis of U2, axis cross-fitted by leaving out the trial's topic.
    Reported: pooled r, partial r given |x|, r within each stated belief; and the earlier content-axis dose."""
    from scipy import stats

    def partial_r(a, c, z):
        ra = a - np.polyval(np.polyfit(z, a, 1), z)
        rc = c - np.polyval(np.polyfit(z, c, 1), z)
        return stats.pearsonr(ra, rc)[0]

    rows = []
    for m in models:
        d = root / "rq3_v2" / f"{m}_chat"
        if not ((d / "cf_trials.parquet").exists() and (d / "subspace_proj.parquet").exists()):
            continue
        df = pd.read_parquet(d / "cf_trials.parquet").merge(
            pd.read_parquet(d / "subspace_proj.parquet")[["trial_id", "u2_0", "u2_1"]], on="trial_id", how="inner")
        df["response"] = np.sign(df.b) * (df.lp_after - df.bayes_post)
        X, b = df[["u2_0", "u2_1"]].to_numpy(np.float64), df.b.to_numpy()
        dose = np.full(len(df), np.nan)
        for t in df.topic.unique():
            te = (df.topic == t).to_numpy()
            tr = ~te
            w = X[tr & (b > 0)].mean(0) - X[tr & (b < 0)].mean(0)
            w /= (np.linalg.norm(w) or 1.0)
            dose[te] = np.sign(b[te]) * ((X[te] - X[tr].mean(0)) @ w)
        df["dose"] = dose
        c = df[df.kappa == 1].dropna(subset=["dose", "response", "x"])
        r, p = stats.pearsonr(c.dose, c.response)
        row = dict(model=m, n_contra=len(c), mean_response=round(c.response.mean(), 3), pearson_r=round(r, 3),
                   pearson_p=p, partial_r_given_absx=round(partial_r(c.dose.values, c.response.values, np.abs(c.x.values)), 3))
        for v in (1, -1):
            cc = c[c.b == v]
            row[f"r_within_b{v:+d}"] = round(stats.pearsonr(cc.dose, cc.response)[0], 3)
        if (d / "doses.parquet").exists():
            dz = pd.read_parquet(d / "doses.parquet")
            old = c.merge(dz[["trial_id", "dose"]].rename(columns={"dose": "dose_content"}), on="trial_id")
            row["r_content_axis_dose_corrected"] = round(stats.pearsonr(old.dose_content, old.response)[0], 3)
            full = df.merge(dz[["trial_id", "dose_raw"]], on="trial_id")
            row["r_content_axis_vs_answer"] = round(stats.pearsonr(full.dose_raw, full.h_dir * full.lp_after)[0], 3)
        rows.append(row)
    return {"dose_response": pd.DataFrame(rows)}


# =============================================================================================================
# stability of the belief subspaces
# =============================================================================================================
def _orth64(M_):
    q, _ = np.linalg.qr(np.asarray(M_, np.float64).T)
    return q.T


def _top_rsv(X, k, rng=None, exact=False, p=12, q=4):
    """Top-k right singular vectors: exact SVD, or randomized SVD with q power iterations."""
    if exact:
        return np.linalg.svd(X, full_matrices=False)[2][:k].astype(np.float64)
    Om = rng.standard_normal((X.shape[1], k + p)).astype(X.dtype)
    Y, _ = np.linalg.qr(X @ Om)
    for _ in range(q):
        Z, _ = np.linalg.qr(X.T @ Y)
        Y, _ = np.linalg.qr(X @ Z)
    return np.linalg.svd(Y.T @ X, full_matrices=False)[2][:k].astype(np.float64)


def _pcos(A, B):
    """Principal-angle cosines between the row spaces of orthonormal A [k,d] and B [k,d], descending."""
    return np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0, 1)


def _build(P, N, Z, k, rng=None, exact=False):
    """U2 (from the two mean displacements) or U4 (top-k SVD of the stacked displacements), as in the pipeline."""
    dp, dn = P - Z, N - Z
    if k == 2:
        return _orth64(np.stack([dp.mean(0), dn.mean(0)]))
    return _top_rsv(np.concatenate([dp, dn]), k, rng, exact)


def _shares(U, X, mp, mn):
    st = float(np.linalg.norm(X @ U.T) ** 2 / np.linalg.norm(X) ** 2)
    sm = float((np.linalg.norm(U @ mp) ** 2 + np.linalg.norm(U @ mn) ** 2) / (mp @ mp + mn @ mn))
    return st, sm


def stability_compute(model, cdir: Path, out: Path, layers, ks=(2, 4), reps=200, n_split=20, n_shift=10, seed=20261116):
    """Principal-angle cosines of resampled / shifted / random subspaces against the full-sample subspace, and
    the geometric retention of the unseen-topic belief shift (see summarize_stability). CPU only."""
    out.mkdir(parents=True, exist_ok=True)
    C = np.load(cdir / "capture.npz")
    Dsaved = load_npz(cdir / "directions_v2.npz")
    cf, held = pd.read_parquet(cdir / "cf_trials.parquet"), pd.read_parquet(cdir / "held_trials.parquet")
    cf_topic, cf_tier = cf.topic.to_numpy(), cf.tier.to_numpy()
    by_topic = {t: np.flatnonzero(cf_topic == t) for t in np.unique(cf_topic)}
    tier_topics = {tr: np.unique(cf_topic[cf_tier == tr]) for tr in sorted(np.unique(cf_tier))}
    rows, ref = [], []
    for l in [x for x in layers if f"cf_pos_last_L{x}" in C.files]:
        g = lambda key: C[f"{key}_last_L{l}"].astype(np.float32)  # noqa: E731
        P, N, Z = g("cf_pos"), g("cf_neg"), g("cf_none")
        hP, hN, hZ = g("held_pos"), g("held_neg"), g("held_none")
        okh = np.isfinite(hP).all(1) & np.isfinite(hN).all(1) & np.isfinite(hZ).all(1)
        hP, hN, hZ = hP[okh], hN[okh], hZ[okh]
        Xh = np.concatenate([hP - hZ, hN - hZ]).astype(np.float64)
        mp, mn = (hP - hZ).mean(0).astype(np.float64), (hN - hZ).mean(0).astype(np.float64)
        Zc = (Z - Z.mean(0)).astype(np.float32)
        d = P.shape[1]
        for k in ks:
            rng = np.random.default_rng([seed, l, k])
            Uf, Ufr = _build(P, N, Z, k, exact=True), _build(P, N, Z, k, rng=rng)
            Uh = _build(hP, hN, hZ, k, exact=True)
            base_t, base_m = _shares(Uf, Xh, mp, mn)
            held_t, held_m = _shares(Uh, Xh, mp, mn)
            saved = Dsaved.get(f"L{l}/ans_U{k}")
            ref.append(dict(model=model, layer=l, k=k, d=d, n_cf=len(P), n_held=int(okh.sum()),
                            share_trial_fullcf=base_t, share_mean_fullcf=base_m,
                            share_trial_heldbuilt=held_t, share_mean_heldbuilt=held_m,
                            cos_min_vs_saved_directions=float(_pcos(Uf, _orth64(saved)).min()) if saved is not None else np.nan,
                            cos_min_rsvd_vs_exact=float(_pcos(Ufr, Uf).min()) if k > 2 else 1.0,
                            cos_min_held_vs_full=float(_pcos(Uh, Uf).min()),
                            overlap_held_vs_full=float((_pcos(Uh, Uf) ** 2).mean())))

            def rec(analysis, rep, U, Uref, n_used=np.nan):
                c = _pcos(U, Uref)
                st, sm = _shares(U, Xh, mp, mn)
                rows.append(dict(model=model, layer=l, k=k, analysis=analysis, rep=rep, n_used=n_used,
                                 **{f"cos{i + 1}": float(c[i]) for i in range(k)},
                                 cos_max=float(c[0]), cos_min=float(c[-1]), cos_mean=float(c.mean()),
                                 overlap=float((c ** 2).mean()), share_trial=st, share_mean=sm,
                                 retention_trial=st / base_t, retention_mean=sm / base_m))

            rec("held_vs_full", 0, Uh, Uf, int(okh.sum()))
            n = len(P)
            err = []
            for r in range(reps):                                     # trial bootstrap
                idx = rng.integers(0, n, n)
                Ub = _build(P[idx], N[idx], Z[idx], k, rng=rng)
                if k > 2 and r < 3:                                   # randomized vs exact SVD on resamples
                    err.append(1 - _pcos(Ub, _build(P[idx], N[idx], Z[idx], k, exact=True)).min())
                rec("trial_boot", r, Ub, Uf, n)
            ref[-1]["max_1mcos_rsvd_vs_exact_boot"] = float(max(err)) if err else 0.0
            for r in range(reps):                                     # topic bootstrap, stratified by tier
                pick = np.concatenate([rng.choice(tt, len(tt), replace=True) for tt in tier_topics.values()])
                idx = np.concatenate([by_topic[t] for t in pick])
                rec("topic_boot", r, _build(P[idx], N[idx], Z[idx], k, rng=rng), Uf, len(idx))
            for r in range(n_split):                                  # topic split-half, stratified by tier
                ha, hb = [], []
                for tt in tier_topics.values():
                    pp = rng.permutation(tt)
                    (ha if r % 2 else hb).extend(pp[: len(pp) // 2])
                    (hb if r % 2 else ha).extend(pp[len(pp) // 2:])
                a = np.concatenate([by_topic[t] for t in ha]) if ha else np.array([], int)
                b = np.concatenate([by_topic[t] for t in hb]) if hb else np.array([], int)
                if not len(a) or not len(b):
                    continue
                Ua, Ub = _build(P[a], N[a], Z[a], k, rng=rng), _build(P[b], N[b], Z[b], k, rng=rng)
                rec("split_half", r, Ua, Ub, len(a))                  # half A vs half B; shares are for half A
                rec("split_half_vs_full", r, Ua, Uf, len(a))
            # unseen topics: for each choice of (number of held topics) of the cf Tier 2 topics, reference U_R on all
            # OTHER cf topics; held-built U vs U_R, and a size-matched U from the chosen SEEN topics vs U_R
            t2 = list(tier_topics.get(2, []))
            for ci, drop in enumerate(itertools.combinations(t2, held.topic.nunique())):
                rest = np.flatnonzero(~np.isin(cf_topic, drop))
                Ur = _build(P[rest], N[rest], Z[rest], k, rng=rng)
                pool = np.flatnonzero(np.isin(cf_topic, drop))
                rec("rest_vs_full", ci, Ur, Uf, len(rest))
                for s_ in range(n_shift):
                    sm_idx = rng.permutation(pool)[: int(okh.sum())]
                    rec("matched_ref", ci * n_shift + s_, _build(P[sm_idx], N[sm_idx], Z[sm_idx], k, rng=rng), Ur, len(sm_idx))
                rec("unseen_topics", ci, Uh, Ur, int(okh.sum()))
            i1, i2 = np.flatnonzero(cf_tier == 1), np.flatnonzero(cf_tier == 2)
            if len(i1) and len(i2):
                U1, U2t = _build(P[i1], N[i1], Z[i1], k, exact=True), _build(P[i2], N[i2], Z[i2], k, exact=True)
                rec("tier1_vs_tier2", 0, U1, U2t, len(i1))            # strong shift: neutral vs political topics
                rec("tier2_vs_full", 0, U2t, Uf, len(i2))
                rec("held_vs_tier2", 0, Uh, U2t, int(okh.sum()))      # unseen vs seen Tier 2 topics
            for r in range(reps):                                     # chance
                rec("rand_iso", r, _orth64(rng.standard_normal((k, d))), Uf)
                rec("rand_data", r, _orth64(rng.standard_normal((k, len(Zc))).astype(np.float32) @ Zc), Uf)
            log(f"[stability] {model} L{l} k{k}: full-cf share of held displacement trial={base_t:.3f} mean={base_m:.3f}")
    pd.DataFrame(rows).to_csv(out / f"reps_{model}.csv", index=False)
    pd.DataFrame(ref).to_csv(out / f"reference_{model}.csv", index=False)


def run_variant_stability(M: Model, layers):
    """Rebuild U2/U4 under the original wording (variant 0: must reproduce the saved subspaces) and the two
    reworded templates (1, 2), compare with the saved subspaces, and report the share of each wording's held-out
    belief shift captured by the original U4 vs the wording's own U4. GPU."""
    from idea.mech.rq3v2 import _orth, forward_states
    out = M.out
    if (out / "variant_stability.csv").exists():
        return
    D = load_npz(out / "directions_v2.npz")
    frames = {"cf": pd.read_parquet(out / "cf_trials.parquet"), "held": pd.read_parquet(out / "held_trials.parquet")}
    layers = [l for l in layers if l < M.n_layers and f"L{l}/ans_U2" in D]
    conds = ("pos", "neg", "none")

    def build(S):
        dp, dn = S["pos"] - S["none"], S["neg"] - S["none"]
        U2 = _orth(np.stack([dp.mean(0), dn.mean(0)])).astype(np.float64)
        return U2, np.linalg.svd(np.concatenate([dp, dn]), full_matrices=False)[2][:4].astype(np.float64)

    def shares(U, S):
        X = np.concatenate([S["pos"] - S["none"], S["neg"] - S["none"]])
        m = np.stack([(S["pos"] - S["none"]).mean(0), (S["neg"] - S["none"]).mean(0)])
        return tuple(float(((Y @ U.T) ** 2).sum() / (Y ** 2).sum()) for Y in (m, X))

    recs = []
    for v in (0, 1, 2):
        S = {}
        for st, fr in frames.items():
            if v == 0 and st == "held":
                continue   # the self-check needs only the cf rebuild
            S[st] = {}
            for c in conds:
                rq = [M.req(r, r.h_dir if c == "pos" else (-r.h_dir if c == "neg" else 0), variant=v)
                      for r in fr.itertuples(index=False)]
                S[st][c] = {l: a.astype(np.float64) for l, a in forward_states(M.backend, rq, layers, bs=M.args.fwd_batch)[1].items()}
            log(f"[stability] {M.key} wording {v}: {st} captured")
        for l in layers:
            U2v, U4v = build({c: S["cf"][c][l] for c in conds})
            c2 = _pcos(U2v, D[f"L{l}/ans_U2"].astype(np.float64))
            c4 = _pcos(U4v, D[f"L{l}/ans_U4"].astype(np.float64))
            rec = dict(model=M.key, wording=v, layer=l, n_cf=len(frames["cf"]), U2_cos_min=float(c2.min()),
                       U2_cos=";".join(f"{x:.4f}" for x in c2), U4_cos=";".join(f"{x:.4f}" for x in c4),
                       U4_overlap=float((c4 ** 2).mean()))
            if v > 0:
                H = {c: S["held"][c][l] for c in conds}
                rec["held_mean_share_origU4"], rec["held_trial_share_origU4"] = shares(D[f"L{l}/ans_U4"].astype(np.float64), H)
                rec["held_mean_share_ownU4"], rec["held_trial_share_ownU4"] = shares(U4v, H)
            recs.append(rec)
    pd.DataFrame(recs).to_csv(out / "variant_stability.csv", index=False)


def run_stability(M: Model, S: dict):
    sdir = Path(M.args.out) / "rq3_v2" / "stability"
    if not (sdir / f"reps_{M.key}.csv").exists():
        stability_compute(M.key, M.out, sdir, S["layers"], S["ks"], S["reps"], S["n_split"], S["n_shift"], S["seed"])
    run_variant_stability(M, S["layers"])


def summarize_stability(root: Path, models) -> dict:
    """stability_summary (median and 5-95% of every measure, per model x layer x k x analysis), headline at the
    erase layer, behavioural retention (validation vs test gap closed, random control, sufficiency), wording."""
    sdir = root / "rq3_v2" / "stability"
    out = {}
    reps = [sdir / f"reps_{m}.csv" for m in models if (sdir / f"reps_{m}.csv").exists()]
    erase = {m: erase_layer_k(root / "rq3_v2" / f"{m}_chat")[0] for m in models}
    if reps:
        R = pd.concat([pd.read_csv(f) for f in reps], ignore_index=True)
        q = lambda x, p: float(np.nanpercentile(x, p)) if len(x) else np.nan  # noqa: E731
        S = []
        for (m, l, k, a), g in R.groupby(["model", "layer", "k", "analysis"], sort=False):
            row = dict(model=m, layer=l, k=k, analysis=a, n_reps=len(g))
            for c in ["cos_max", "cos_min", "cos_mean", "overlap", "share_trial", "share_mean", "retention_trial",
                      "retention_mean"]:
                row[f"{c}_median"], row[f"{c}_p05"], row[f"{c}_p95"] = q(g[c], 50), q(g[c], 5), q(g[c], 95)
            S.append(row)
        S = pd.DataFrame(S)
        out["stability_summary"] = S
        keep = ["trial_boot", "topic_boot", "split_half", "split_half_vs_full", "matched_ref", "unseen_topics",
                "held_vs_full", "held_vs_tier2", "tier1_vs_tier2", "tier2_vs_full", "rest_vs_full", "rand_data", "rand_iso"]
        H = S[S.apply(lambda r: r.layer == erase.get(r.model), axis=1) & S.analysis.isin(keep)]
        H = H[["model", "layer", "k", "analysis", "cos_min_median", "cos_min_p05", "cos_mean_median", "overlap_median",
               "retention_mean_median", "retention_mean_p05", "retention_trial_median", "retention_trial_p05"]].copy()
        H["analysis"] = pd.Categorical(H.analysis, keep, ordered=True)
        out["headline_erase_layer"] = H.sort_values(["model", "k", "analysis"]).reset_index(drop=True)
    beh, var = [], []
    for m in models:
        d = root / "rq3_v2" / f"{m}_chat"
        s = chosen_setting(d)
        if s and all((d / f).exists() for f in ("sweep_validation_v2.csv", "steering_test_v2.csv", "sufficiency.csv")):
            val, test = pd.read_csv(d / "sweep_validation_v2.csv"), pd.read_csv(d / "steering_test_v2.csv")
            v = val[(val.family == "erase_answer") & (val.setting == s)].iloc[0]
            t = test[(test.family == "erase_answer") & (test.setting == s)].iloc[0]
            c = test[(test.family == "control_random_subspace") & (test.setting == s)]
            suf = pd.read_csv(d / "sufficiency.csv")
            sf = suf[(suf.layer == int(s.split("_")[0][1:])) & (suf.patch == f"answer_U{s[-1]}")]
            vg = 1 - v.t2_contra_gap_after / v.t2_contra_gap_before
            beh.append(dict(model=m, setting=s, val_gap_before=v.t2_contra_gap_before, val_gap_after=v.t2_contra_gap_after,
                            val_gap_closed=vg, test_gap_before=t.t2_contra_gap_before, test_gap_after=t.t2_contra_gap_after,
                            test_gap_closed=t.gap_closed_frac, test_over_val=t.gap_closed_frac / vg,
                            random_control_test_gap_closed=float(c.gap_closed_frac.iloc[0]) if len(c) else np.nan,
                            sufficiency_recovery_heldtopics=float(sf.recovery.mean()) if len(sf) else np.nan,
                            test_t2_neutral_change=t.get("t2_neutral_change"),
                            test_tier1_contra_change=t.get("tier1_contra_change")))
        if (d / "variant_stability.csv").exists():
            w = pd.read_csv(d / "variant_stability.csv")
            w = w[(w.layer == erase.get(m)) & (w.wording > 0)]
            for r in w.itertuples():
                c2 = [float(x) for x in str(r.U2_cos).split(";")]
                var.append(dict(model=m, layer=r.layer, wording=r.wording, U2_cos1=c2[0], U2_cos2=c2[-1],
                                U4_overlap=r.U4_overlap, captured_by_original_U4=r.held_mean_share_origU4,
                                captured_by_own_U4=r.held_mean_share_ownU4))
    out["behavioral_retention"] = pd.DataFrame(beh)
    out["variant_stability"] = pd.DataFrame(var)
    return out


# =============================================================================================================
# prereg: the preregistered RQ3 (prompt-end activations)
# =============================================================================================================
def capture_prompt_end(M: Model, layers, acts_dir: Path):
    """Prompt-end activations (the state the preregistered analysis used: end of the chat prompt, no answer
    prefix) of every trial of the model, saved per layer as float16 [n_trials, hidden]."""
    from idea.mech.rq3v2 import forward_states
    acts_dir.mkdir(parents=True, exist_ok=True)
    if (acts_dir / "COMPLETE").exists():
        return
    n, chunk = len(M.d), 2048
    rows = list(M.d.itertuples(index=False))
    mm = None
    for s in range(0, n, chunk):
        log(f"[prereg] prompt-end capture {s}/{n}")
        _, last, _ = forward_states(M.backend, [M.req(r) for r in rows[s:s + chunk]], layers, prefill=False,
                                    bs=M.args.fwd_batch)
        if mm is None:
            mm = {l: np.lib.format.open_memmap(acts_dir / f"L{l}.npy", "w+", np.float16, (n, last[l].shape[1]))
                  for l in layers}
        for l in layers:
            mm[l][s:s + chunk] = last[l].astype(np.float16)
    for a in mm.values():
        a.flush()
    (acts_dir / "COMPLETE").write_text(json.dumps(dict(n=n, layers=layers)))


def run_prereg(M: Model, S: dict):
    from idea.mech import probe, steer, subspace
    from idea.mech import rq1fit
    from idea.mech.capture import ActStore
    from idea.mech.other_biases import bias_directions, run_contrast_sets
    out = M.out_prereg
    out.mkdir(parents=True, exist_ok=True)
    d = M.d
    layers = [l for l in S["layers"] if l < M.n_layers]
    capture_prompt_end(M, layers, out / "acts")
    cacheH = {}

    def get_H(l):
        if l not in cacheH:
            cacheH[l] = np.load(out / "acts" / f"L{l}.npy").astype(np.float32)
        return cacheH[l]

    # RQ1 gate: is there anything to fix?
    prim = rq1fit.primary(rq1fit.analysis_frame(d, 2))
    exploratory = not (prim["holm"]["delta_b"] or prim["holm"]["theta_cong"])
    # 1. masking probe (ridge probe of the Bayesian posterior on belief-neutral trials, layer by validation R2)
    mask = probe.run_masking(get_H, d, layers, seed=SEED)
    mask["sweep"].to_csv(out / "probe_layer_sweep.csv", index=False)
    for k in ("overall", "by_dimension", "by_cue_type"):
        mask[k].to_csv(out / f"masking_{k}.csv", index=False)
    log(f"[prereg] probe layer {mask['layer']} test R2={mask['r2_test']:.3f}; "
        f"masking={mask['overall'].masking_index.iloc[0]:.3f}")
    # 2. directions (training topics)
    train_t, val_t, test_t = probe.split_topics(d.topic.unique(), S["test_frac"], S["val_frac"], SEED + 1)
    tr_idx = d.index[d.topic.isin(train_t)]
    dtrain = d[d.topic.isin(train_t)].reset_index(drop=True)
    get_H_train = lambda l: get_H(l)[tr_idx]  # noqa: E731
    ks, kmain = tuple(S["topic_k"]), S["k_main"]
    dpath = out / "directions"
    if M.args.use_released_vectors:
        if not (ROOT / "steering_vectors" / M.key / "directions.npz").exists():
            sys.exit(f"--use-released-vectors: no released preregistered directions for {M.key}")
        dirs = steer.load_directions(ROOT / "steering_vectors" / M.key / "directions")
        log("[prereg] using the released directions")
    elif Path(f"{dpath}.npz").exists():
        dirs = steer.load_directions(dpath)   # a resumed run reuses its directions (same cache keys)
    else:
        dirs = steer.build_directions(dtrain, get_H_train, layers, ks, SEED)
        steer.save_directions(dirs, dpath)
    if not dirs:
        (out / "STEERING_SKIPPED.txt").write_text("no steering directions: no training-topic stratum had both caved "
                                                  "and resisted contradicted trials\n")
        return
    thdim = rq1fit.theta_by_dimension(rq1fit.analysis_frame(d, 2))
    credible = set(thdim[thdim.p < 0.05].dimension)
    ang = [subspace.pairwise_directions({k: v for k, v in rec[f"per_dim_hat_k{kmain}"].items() if k in credible}).assign(layer=l)
           for l, rec in dirs.items() if f"per_dim_hat_k{kmain}" in rec]
    ang = [a for a in ang if len(a)]
    if ang:
        pd.concat(ang).to_csv(out / "per_dimension_angles.csv", index=False)
    # 3. sweep (validation topics): layer x strength x mode for hat_k<kmain>
    val_trials, test_trials = eval_sets(M.design, val_t, test_t, S["n_eval"], SEED)
    fac = lambda b: M.elicitor  # noqa: E731
    sweep_layers = [l for l in S["sweep_layers"] if l in dirs] or sorted(dirs)
    cand = {l: dict(vec=dirs[l][f"hat_k{kmain}"], scale=dirs[l]["scale"][f"hat_k{kmain}"]) for l in sweep_layers}
    base_val = steer.evaluate_steering(M.backend, fac, val_trials, M.topics, M.lib, "chat", sweep_layers[0], np.ones(1), 0.0)
    sw = steer.sweep(M.backend, fac, val_trials, M.topics, M.lib, "chat", sweep_layers, {f"hat_k{kmain}": cand},
                     multiples=tuple(S["multiples"]), modes=tuple(S["modes"]), unsteered=base_val)
    sw.to_csv(out / "sweep_validation.csv", index=False)
    ok = sw[sw.success] if sw.success.any() else sw
    best = ok.loc[ok.t2_contra_gap_after.abs().idxmin()] if "t2_contra_gap_after" in ok else ok.iloc[0]
    L, mult, mode = int(best.layer), float(best.multiple), best["mode"]
    log(f"[prereg] chosen layer={L} multiple={mult} mode={mode}")
    # 4. held-out test: main direction, controls, amplification, per-dimension directions
    base_test = steer.evaluate_steering(M.backend, fac, test_trials, M.topics, M.lib, "chat", L, np.ones(1), 0.0)
    s0, spread0 = steer.slopes(base_test), steer.delta_j_spread(base_test)
    dims = sorted(set(base_test.dimension) - {"none"})
    tests = {f"hat_k{k}": dirs[L][f"hat_k{k}"] for k in ks}
    tests.update({"raw": dirs[L]["raw"], "random": dirs[L]["random"], f"topic_only_k{kmain}": dirs[L][f"topic_only_k{kmain}"]})
    if dirs[L].get(f"prompt_hat_k{kmain}") is not None:
        tests[f"prompt_hat_k{kmain}"] = dirs[L][f"prompt_hat_k{kmain}"]
    rows = []
    for name, vec in tests.items():
        for sign, tag in ((1, ""), (-1, "_amplify")):
            if tag and name != f"hat_k{kmain}":
                continue
            psi = sign * mult * dirs[L]["scale"].get(name, steer.psi_scale(get_H(L), vec))
            res = steer.evaluate_steering(M.backend, fac, test_trials, M.topics, M.lib, "chat", L, vec, psi, mode)
            s1 = steer.slopes(res)
            rep = steer.success(s0, s1, dims=dims, spread_before=spread0, spread_after=steer.delta_j_spread(res))
            rows.append(dict(direction=name + tag, layer=L, psi=psi, mode=mode, **flat(rep),
                             **{f"slope_{k}": v for k, v in s1.items()}))
    for k, vec in dirs[L][f"per_dim_hat_k{kmain}"].items():
        if k in credible:
            psi = mult * steer.psi_scale(get_H(L), vec)
            res = steer.evaluate_steering(M.backend, fac, test_trials, M.topics, M.lib, "chat", L, vec, psi, mode)
            rows.append(dict(direction=f"per_dim_{k}", layer=L, psi=psi, mode=mode,
                             **flat(steer.success(s0, steer.slopes(res), dims=dims))))
    pd.DataFrame(rows).to_csv(out / "steering_test.csv", index=False)
    summary = dict(exploratory=exploratory, layer=L, multiple=mult, mode=mode, probe_layer=mask["layer"],
                   probe_r2=mask["r2_test"], masking=float(mask["overall"].masking_index.iloc[0]))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    # 5. angles to other cue-induced biases (sycophancy, authority, status quo) at the chosen layer
    if not (out / "bias_angles.csv").exists():
        old = M.backend.capture_layers
        M.backend.capture_layers = [L]
        try:
            from idea.elicit import CachedElicitor
            el = CachedElicitor(M.backend, M.elicitor.cache, act_store=ActStore(out / "acts_other_biases"))
            cs = run_contrast_sets(el, el.act_store, [L], n_mmlu=S["n_mmlu"], seed=SEED)
        finally:
            M.backend.capture_layers = old
        cs.to_parquet(out / "other_bias_contrast_sets.parquet")
        U = steer.topic_subspace(get_H_train(L)[(dtrain.b == 0).values], kmain, SEED)
        bd = bias_directions(cs, el.act_store, L, U)
        bd["idea_pooled"] = dirs[L][f"hat_k{kmain}"]
        subspace.pairwise_directions(bd).to_csv(out / "bias_angles.csv", index=False)


def summarize_prereg(root: Path, models) -> dict:
    tests, chosen, sweeps, masks, angles = [], [], [], [], []
    for m in models:
        d = root / "rq3" / f"{m}_chat"
        if not (d / "steering_test.csv").exists():
            continue
        t = pd.read_csv(d / "steering_test.csv")
        t = t[[c for c in ["direction", "layer", "psi", "mode", "success", "t2_contra_gap_before", "t2_contra_gap_after",
                           "t2_contra_moved_toward_neutral", "t2_neutral_change", "tier1_neutral_change",
                           "tier1_contra_change", "tier1_support_change"] if c in t]].assign(model=m)
        t["gap_closed_frac"] = 1 - t.t2_contra_gap_after / t.t2_contra_gap_before
        tests.append(t)
        chosen.append(dict(model=m, **json.loads((d / "summary.json").read_text())))
        sw = pd.read_csv(d / "sweep_validation.csv")
        sw["abs_gap"] = sw.t2_contra_gap_after.abs()
        sweeps.append(sw.loc[sw.groupby("layer").abs_gap.idxmin()].assign(model=m))
        mk = pd.read_csv(d / "masking_overall.csv").assign(dimension="overall")
        if (d / "masking_by_dimension.csv").exists():
            mk = pd.concat([mk, pd.read_csv(d / "masking_by_dimension.csv")])
        masks.append(mk.assign(model=m))
        if (d / "bias_angles.csv").exists():
            angles.append(pd.read_csv(d / "bias_angles.csv").assign(model=m))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()  # noqa: E731
    return {"prereg_steering_test": cat(tests), "prereg_chosen_settings": pd.DataFrame(chosen),
            "prereg_sweep_best": cat(sweeps), "prereg_masking": cat(masks), "prereg_bias_angles": cat(angles)}


def run_prereg_audit(M: Model, A: dict):
    """unsteered / steered (chosen hat_k8 setting) / random (equal-norm random direction), in pieces."""
    from idea.audit import all_parts, parse_part, run_part
    from idea.mech.other_biases import mmlu_items
    out = M.out_prereg
    os.environ.setdefault("IDEA_LMEVAL_BATCH", str(A["lmeval_batch"]))
    conds = {"unsteered": []}
    if (out / "summary.json").exists():
        s = json.loads((out / "summary.json").read_text())
        meta = json.loads((out / "directions.json").read_text()) if (out / "directions.json").exists() else \
            json.loads((ROOT / "steering_vectors" / M.key / "directions.json").read_text())
        vecs = load_npz(out / "directions.npz") if (out / "directions.npz").exists() else \
            load_npz(ROOT / "steering_vectors" / M.key / "directions.npz")
        L, k = s["layer"], PREREG_K(M)
        psi = s["multiple"] * meta[f"L{L}/scale"][f"hat_k{k}"]
        conds.update({"steered": [(L, vecs[f"L{L}/hat_k{k}"], psi, s["mode"])],
                      "random": [(L, vecs[f"L{L}/random"], psi, s["mode"])]})
    else:
        log("[prereg_audit] no preregistered steering summary: auditing the unsteered model only")
    pdir = out / "audit_parts"
    pdir.mkdir(parents=True, exist_ok=True)
    items = mmlu_items(A["n_syceval"], seed=SEED + 7, split="validation")
    for part in all_parts(list(conds), A["prereg_mmlu_groups"]):
        f = pdir / f"{part.replace('/', '__').replace(':', '_')}.csv"
        _, what, tasks = parse_part(part)
        if f.exists() or (A["mmlu_subjects"] and what.startswith("mmlu_pro")   # smoke tests: a subject subset
                          and not {f"mmlu_pro_{x}" for x in A["mmlu_subjects"]} & set(tasks)):
            continue
        log(f"[prereg_audit] {M.key} {part}")
        pd.DataFrame([run_part(M.backend, conds[part.split('/')[0]], part, items, A["bbq_limit"] if "bbq" in part
                               else A["prereg_mmlu_limit"], pdir)]).to_csv(f, index=False)


def summarize_prereg_audit(root: Path, models) -> dict:
    """Merged benchmark parts per condition, and two-proportion tests of steered / random against unsteered
    (MMLU-Pro compared on the subjects both conditions finished)."""
    from scipy import stats
    from idea.audit import merge_parts

    def two_prop_z(p1, n1, p2, n2):
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se if se > 0 else np.nan
        return p1 - p2, se, z, 2 * stats.norm.sf(abs(z))

    summ, tests = [], []
    cols = {"mmlu_pro/exact_match,custom-extract": "mmlu_pro_acc", "mmlu_pro/n": "mmlu_pro_n",
            "mmlu_pro_subjects_found": "mmlu_pro_subjects", "bbq/acc,none": "bbq_acc",
            "bbq/accuracy_amb,none": "bbq_acc_ambiguous", "bbq/accuracy_disamb,none": "bbq_acc_disambiguated",
            "bbq/amb_bias_score,none": "bbq_bias_ambiguous", "bbq/disamb_bias_score,none": "bbq_bias_disambiguated",
            "bbq/n": "bbq_n", "syceval_initial_accuracy": "syceval_initial_acc",
            "syceval_regressive_flip_rate": "syceval_regressive_flip", "syceval_progressive_flip_rate": "syceval_progressive_flip",
            "syceval_n": "syceval_n"}
    for m in models:
        files = sorted((root / "rq3" / f"{m}_chat" / "audit_parts").glob("*.csv"))
        if not files:
            continue
        merged = merge_parts(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
        conds = set(merged.condition)
        complete = merged.get("mmlu_pro_subjects_found", pd.Series(0)).eq(14).all() and {"unsteered", "steered", "random"} <= conds
        s = merged[["condition"] + [c for c in cols if c in merged]].rename(columns=cols)
        s.insert(0, "model", m)
        s["complete"] = complete
        summ.append(s)
        if "unsteered" not in conds:
            continue
        u, mu = s[s.condition == "unsteered"].iloc[0], merged[merged.condition == "unsteered"].iloc[0]
        for cond in [c for c in ("steered", "random") if c in conds]:
            r, mr = s[s.condition == cond].iloc[0], merged[merged.condition == cond].iloc[0]
            acc = "exact_match,custom-extract"
            subj = [k[:-2] for k in merged.columns if k.startswith("mmlu_pro_") and k.endswith("/n")
                    and pd.notna(mr.get(k)) and pd.notna(mu.get(k))]
            if subj:
                n_r, n_u = sum(mr[f"{x}/n"] for x in subj), sum(mu[f"{x}/n"] for x in subj)
                a_r = sum(mr[f"{x}/{acc}"] * mr[f"{x}/n"] for x in subj) / n_r
                a_u = sum(mu[f"{x}/{acc}"] * mu[f"{x}/n"] for x in subj) / n_u
                diff, se, z, p = two_prop_z(a_r, n_r, a_u, n_u)
                tests.append(dict(model=m, condition=cond, measure=f"mmlu_pro_acc ({len(subj)} shared subjects)", value=a_r,
                                  unsteered=a_u, diff=diff, se=se, z=z, p=p, n=n_r, complete=complete))
            for a_, n_ in (("bbq_acc", "bbq_n"), ("syceval_initial_acc", "syceval_n")):
                if a_ in s and pd.notna(r.get(a_)) and pd.notna(u.get(a_)):
                    diff, se, z, p = two_prop_z(r[a_], r[n_], u[a_], u[n_])
                    tests.append(dict(model=m, condition=cond, measure=a_, value=r[a_], unsteered=u[a_], diff=diff, se=se,
                                      z=z, p=p, n=r[n_], complete=complete))
    return {"prereg_audit_summary": pd.concat(summ, ignore_index=True) if summ else pd.DataFrame(),
            "prereg_audit_tests": pd.DataFrame(tests)}


def PREREG_K(M) -> int:  # noqa: N802  (k_main of the run's settings)
    return M.args.settings["prereg"]["k_main"]


# =============================================================================================================
# p4: training-stage probes at the answer position
# =============================================================================================================
def run_p4(M: Model, S: dict):
    """Answer-position activations of the stage's explicit-label chat trials (rendered with the answer prefix
    where the recorded answer came from the prefix retry), then per layer (1) a logistic probe for contradicted
    vs supported on held-out topics and (2) the stage's caved-minus-resisted direction (hat_k8)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from idea.mech import probe, steer
    from idea.mech.rq3v2 import forward_states
    out = Path(M.args.out) / "p4_v2" / f"{M.key}_chat"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "probes.csv").exists():
        return
    d = M.d[M.d.cue_type == "label"].reset_index(drop=True)
    cap_f = out / "capture.npz"
    if not cap_f.exists():
        layers = [l for l in S["layers"] if l < M.n_layers]
        reqs = [M.req(r) for r in d.itertuples(index=False)]
        att = d.att_post.fillna(0).astype(int).values
        H = {}
        for flag in (0, 1):
            idx = np.where(att == flag)[0]
            if not len(idx):
                continue
            log(f"[p4] {M.key}: {len(idx)} trials, prefill={bool(flag)}")
            _, last, _ = forward_states(M.backend, [reqs[i] for i in idx], layers, prefill=bool(flag), bs=M.args.fwd_batch)
            for l in layers:
                H.setdefault(l, np.zeros((len(d), last[l].shape[1]), np.float32))[idx] = last[l]
        np.savez(cap_f, **{f"L{l}": H[l].astype(np.float16) for l in layers})
    C = np.load(cap_f)
    train_t, _, test_t = probe.split_topics(d.topic.unique(), S["test_frac"], 0.0, SEED + 1)
    sub = ((d.tier == 2) & ~d.excluded & d.e.isin(["contra", "support"])).values
    tr_m = d.topic.isin(train_t).values
    dtrain = d[tr_m].reset_index(drop=True)
    rows, dirs = [], {}
    for l in sorted(int(k[1:]) for k in C.files):
        H = C[f"L{l}"].astype(np.float32)
        y = (d.e == "contra").astype(int).values
        tr, te = sub & tr_m, sub & d.topic.isin(test_t).values
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(H[tr], y[tr])
        p = clf.predict_proba(H[te])[:, 1]
        dd = steer.build_directions(dtrain, lambda _l, H=H: H[tr_m], [l], ks=(8,), seed=SEED)
        if l in dd:
            dirs[f"L{l}/hat_k8"], dirs[f"L{l}/raw"] = dd[l]["hat_k8"], dd[l]["raw"]
        rows.append(dict(stage=M.key, layer=l, n_train=int(tr.sum()), n_test=int(te.sum()),
                         probe_acc=float(((p > 0.5).astype(int) == y[te]).mean()),
                         probe_auc=float(roc_auc_score(y[te], p)) if len(set(y[te])) == 2 else np.nan,
                         majority=float(max(y[te].mean(), 1 - y[te].mean()))))
        log(f"[p4] {M.key} L{l}: acc {rows[-1]['probe_acc']:.3f} auc {rows[-1]['probe_auc']:.3f}")
    np.savez(out / "directions.npz", **dirs)
    pd.DataFrame(rows).to_csv(out / "probes.csv", index=False)


def summarize_p4(root: Path, stages=P4_STAGES, summary_layers=(18, 26)) -> dict:
    """Per stage and layer: probe metrics and the cosine of the stage's direction to Instruct's; and the mean over
    the later layers (the paper's table)."""
    base = root / "p4_v2"
    ref_f = base / f"{stages[-1]}_chat" / "directions.npz"
    if not ref_f.exists():
        return {}
    ref = load_npz(ref_f)
    rows = []
    for s in stages:
        if not (base / f"{s}_chat" / "probes.csv").exists():
            continue
        dd = load_npz(base / f"{s}_chat" / "directions.npz")
        for r in pd.read_csv(base / f"{s}_chat" / "probes.csv").itertuples(index=False):
            rec = r._asdict()
            for kind in ("hat_k8", "raw"):
                a, b = dd.get(f"L{r.layer}/{kind}"), ref.get(f"L{r.layer}/{kind}")
                rec[f"cos_{kind}_to_instruct"] = (float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
                                                  if a is not None and b is not None else np.nan)
            rows.append(rec)
    P = pd.DataFrame(rows)
    lo, hi = summary_layers
    late = P[(P.layer >= lo) & (P.layer <= hi)].groupby("stage", sort=False)[
        ["probe_acc", "majority", "probe_auc", "cos_raw_to_instruct", "cos_hat_k8_to_instruct"]].mean().reset_index()
    return {"p4_v2_summary": P, f"p4_v2_layers_{lo}_{hi}_mean": late}


# =============================================================================================================
# driver
# =============================================================================================================
RUNNERS = {"corrected": run_corrected, "audit": run_audit, "geometry": run_geometry, "dose": lambda M, S: make_projections(M),
           "stability": run_stability, "prereg": run_prereg, "prereg_audit": run_prereg_audit}
SETTINGS_KEY = {"corrected": "corrected", "audit": "audit", "geometry": "corrected", "dose": "corrected",
                "stability": "stability", "prereg": "prereg", "prereg_audit": "audit", "p4": "p4"}


def show(name: str, tab: pd.DataFrame, save: Path | None):
    print(f"\n== {name} ({len(tab)} rows)")
    print(tab.round(3).to_string(index=False) if len(tab) else "  (no results)")
    if save is not None and len(tab):
        save.mkdir(parents=True, exist_ok=True)
        tab.to_csv(save / f"{name}.csv", index=False)


def summarize(todo, root: Path, models, args):
    save = Path(args.save) if args.save else None
    fns = {"corrected": summarize_corrected, "audit": summarize_audit, "geometry": summarize_geometry,
           "dose": summarize_dose, "stability": summarize_stability, "prereg": summarize_prereg,
           "prereg_audit": summarize_prereg_audit}
    for a in todo:
        tabs = (summarize_p4(root, summary_layers=args.settings["p4"]["summary_layers"]) if a == "p4"
                else fns[a](root, models))
        for name, tab in tabs.items():
            if name == "geometry" and len(tab):   # printed in part (all columns are saved)
                show(name, tab[[c for c in ["model", "layer", "cos_pos_neg", "cos_pos_neg_whitened", "norm_pos",
                                            "norm_neg", "share_var_U2", "share_var_U4"] if c in tab]], None)
                if save is not None:
                    tab.to_csv(save / f"{name}.csv", index=False)
                continue
            if name == "stability_summary":   # large; saved in full, printed only as a count
                print(f"\n== stability_summary ({len(tab)} rows; written with --save)")
                if save is not None:
                    save.mkdir(parents=True, exist_ok=True)
                    tab.to_csv(save / f"{name}.csv", index=False)
                continue
            show(name, tab, save)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the analyses and exit")
    ap.add_argument("--only", help="comma-separated analyses to run (overrides ANALYSES)")
    ap.add_argument("--skip", help="comma-separated analyses to skip")
    ap.add_argument("--data", default=str(ROOT / "datasets" / "main_trials.csv.gz"))
    ap.add_argument("--models", help=f"comma-separated model keys (default: those of {MODELS} in the data)")
    ap.add_argument("--out", default="rq3_output", help="output folder (activations, subspaces, caches, raw results)")
    ap.add_argument("--save", help="also write every result table as CSV into this folder")
    ap.add_argument("--summarize-only", action="store_true",
                    help="no model: rebuild the tables from the outputs already in --out (layout: rq3_v2/<model>_chat, "
                         "rq3/<model>_chat, p4_v2/<stage>_chat, rq3_v2/stability)")
    ap.add_argument("--use-released-vectors", action="store_true",
                    help="steer with steering_vectors/<model>/ instead of re-estimating (corrected, prereg)")
    ap.add_argument("--audit-intervention", default="erase",
                    help="corrected audit: 'erase' (chosen answer-position erasure; default), 'auto', or family:setting")
    ap.add_argument("--l2d-items", default=str(ROOT / "data" / "l2d_external" / "exp1_prompts.json"),
                    help="L2D items (the authors' exp1_prompts.json); L2D is skipped if the file is absent")
    ap.add_argument("--hf-model", action="append", default=[], metavar="KEY=REPO[@REV]",
                    help="load other weights for a model key (testing), e.g. olmo_instruct=allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--smoke", action="store_true", help="tiny settings for a CPU smoke test (numbers meaningless)")
    ap.add_argument("--max-trials", type=int, help="use a random subset of each model's trials (testing only)")
    ap.add_argument("--batch-size", type=int, default=BACKEND["batch_size"], help="batch size of steered evaluations")
    ap.add_argument("--fwd-batch", type=int, default=FWD_BATCH, help="batch size of capture / patching passes")
    ap.add_argument("--dtype", help="torch dtype (default: bf16 on GPU, fp32 on CPU)")
    ap.add_argument("--device-map", default="auto")
    a = ap.parse_args(argv)
    a.hf_override = dict(x.split("=", 1) for x in a.hf_model)
    a.settings = dict(corrected=dict(CORRECTED), prereg=dict(PREREG), audit=dict(AUDIT), stability=dict(STABILITY),
                      p4=dict(P4))
    if a.smoke:
        for k in a.settings:
            a.settings[k].update(SMOKE.get(k, {}))
        a.max_trials = a.max_trials or SMOKE["max_trials"]
        a.fwd_batch, a.batch_size = SMOKE["fwd_batch"], SMOKE["batch_size"]
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        width = max(map(len, ANALYSES))
        for k, v in ANALYSES.items():
            desc = [ln for ln in __doc__.splitlines() if ln.strip().startswith(k + " ")]
            print(f"  {k:<{width}}  default {'on ' if v else 'off'}  {desc[0].strip()[len(k):].strip() if desc else ''}")
        return 0
    todo = [k for k, v in ANALYSES.items() if v]
    if args.only:
        todo = [x.strip() for x in args.only.split(",") if x.strip()]
    if args.skip:
        todo = [x for x in todo if x not in {s.strip() for s in args.skip.split(",")}]
    bad = [x for x in todo if x not in ANALYSES]
    if bad:
        sys.exit(f"unknown analyses {bad}; known: {list(ANALYSES)}")
    root = Path(args.out)
    models = [m.strip() for m in args.models.split(",")] if args.models else None

    if not args.summarize_only:
        log(f"reading {args.data}")
        data = load_dataset(args.data)
        models = models or [m for m in MODELS if ((data.model == m) & (data.fmt == "chat")).any()]
        design = design_trials(data)
        per_model = [a for a in todo if a != "p4"]
        for m in models if per_model else []:
            M = Model(m, args, data, design)
            if M.d.empty:
                log(f"{m}: no chat trials in the data; skipped")
                continue
            for a in per_model:
                if a in NEEDS_CORRECTED and not (M.out / "summary_v2.json").exists():
                    log(f"[{a}] {m}: needs the corrected analysis first (run with --only corrected,{a})")
                    continue
                log(f"=== {a}: {m}")
                RUNNERS[a](M, args.settings[SETTINGS_KEY[a]])
            M.release()
        if "p4" in todo:
            for s in P4_STAGES:
                M = Model(s, args, data, design)
                if M.d.empty:
                    log(f"[p4] {s}: no chat trials in the data; skipped")
                    continue
                log(f"=== p4: {s}")
                run_p4(M, args.settings["p4"])
                M.release()
    models = models or MODELS
    summarize(todo, root, models, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
