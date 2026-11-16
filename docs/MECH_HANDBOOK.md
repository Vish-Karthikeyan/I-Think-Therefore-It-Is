# IDEA mechanistic-tools handbook

The mechanistic interpretability code lives in `src/idea/mech/`. It supports the RQ3 results in the
secondary paper: reading the model's internal "Bayesian" posterior out of its hidden states (the
*masking probe*), finding and removing the activation directions that carry **belief-following**
(evidence discounting conditioned on the user's stated belief and identity), and the corrected,
answer-position analysis that builds a low-dimensional **belief subspace** and erases it.

This handbook is for a researcher who wants to reuse the tools. It describes what each module does,
the released vector files and their keys, and copy-pasteable usage. It is accurate to the function
and class signatures in the code; where a routine needs a running model it is noted, and a
numpy-only path is given where one exists.

> **Dual-use note.** Every released direction and subspace is reversible: subtracting it (or setting
> a subspace to its no-belief target) is meant to *remove* belief-following; adding it, or moving the
> subspace coordinates away from the target, *amplifies* it. The vectors are released openly; see the
> paper's ethics statement.

## Contents

- [Where everything is](#where-everything-is)
- [The pipeline at a glance](#the-pipeline-at-a-glance)
- [Module reference](#module-reference)
  - [`capture.py` — activation storage](#capturepy--activation-storage)
  - [`probe.py` — the masking probe](#probepy--the-masking-probe)
  - [`steer.py` — directions and steering (original RQ3)](#steerpy--directions-and-steering-original-rq3)
  - [`subspace.py` — principal angles between directions](#subspacepy--principal-angles-between-directions)
  - [`other_biases.py` — sycophancy / authority / status-quo contrasts](#other_biasespy--sycophancy--authority--status-quo-contrasts)
  - [`rq3v2.py` — corrected RQ3 and belief-subspace erasure](#rq3v2py--corrected-rq3-and-belief-subspace-erasure)
- [The released vector files](#the-released-vector-files)
- [Applying an intervention: `set_steering`](#applying-an-intervention-set_steering)
- [Worked examples](#worked-examples)

---

## Where everything is

```
src/idea/mech/
  capture.py        ActStore: float16 activation shards keyed by request cache key
  probe.py          masking probe (read the Bayesian posterior out of the residual stream)
  steer.py          d_bias directions, topic projection, steering sweep + success criterion
  subspace.py       principal angles between directions / subspaces, vs random baselines
  other_biases.py   contrast sets for opinion sycophancy, authority, status-quo bias
  rq3v2.py          corrected (answer-position) RQ3: counterfactual belief subspaces + erasure

steering_vectors/
  llama/            Llama 3.1 8B Instruct      directions.npz/.json, summary.json, belief_subspaces_v2.npz
  olmo_instruct/    OLMo 2 7B Instruct         directions.npz/.json, summary.json, belief_subspaces_v2.npz
  tulu_final/       Tülu 3 8B (final)          belief_subspaces_v2.npz
  README.md, BELIEF_SUBSPACES_README.txt
```

The per-model orchestration (which trials, which layers, caching, resume) is in
`src/idea/pipeline.py` (`stage_steer`) and in `rq3v2.run`. The CLI entry points are
`idea steer`, `idea steer2`, `idea audit`, `idea audit2`, `idea p4v2` (see `src/idea/cli.py`).

## The pipeline at a glance

1. **Capture** last-token residual-stream activations in the same forward pass that produces the
   answer, per layer, keyed by the request's cache key (`capture.ActStore`, or the answer-position
   `rq3v2.forward_states`).
2. **Build directions** that separate belief-following behavior from the rest:
   - *Original RQ3* (`steer.py`): `d_bias` = mean activation on contradicted trials where the model
     **caved** minus where it **resisted**, matched within strata, with topic variance projected out.
   - *Corrected RQ3* (`rq3v2.py`): a **belief subspace** from counterfactual inputs that differ only
     in the stated belief (believes H1 / believes H0 / no belief).
3. **Masking probe** (`probe.py`): a ridge probe trained on belief-neutral trials reads the Bayesian
   posterior logit `lp_before + x` out of the hidden state; applied to contradicted trials it says
   whether the internal readout is at the Bayesian answer while the output is not.
4. **Sufficiency / interchange test** (`rq3v2.py` stage 3): patch a subspace from one counterfactual
   input into the other and measure how much of the belief effect on the answer transfers.
5. **Steering sweep and held-out test** (`steer.sweep`, `rq3v2.run` stages 4–5): choose layer /
   strength / mode on validation topics, test once on held-out topics against the preregistered
   success criterion, with random and topic-only controls.
6. **Audit** (`idea audit` / `idea audit2`): MMLU-Pro, BBQ, SycEval, L2D under the intervention vs
   unsteered vs a random control, to check the intervention preserves general capability.

---

## Module reference

### `capture.py` — activation storage

`ActStore(root)` stores per-layer activations as float16 `.npz` shards with a JSON key index, keyed
by each request's cache key, so captures from different jobs accumulate and can be looked up later.

| Method | Signature | Returns / effect |
|---|---|---|
| `__init__` | `ActStore(root: Path | str)` | Opens/creates the store at `root`, loading the shard index. |
| `has` | `has(key: str) -> bool` | Whether `key` has saved activations. |
| `put` | `put(acts: dict[str, dict[int, np.ndarray]]) -> None` | Append activations (`{cache_key: {layer: vector}}`) as a new shard. Shard names are unique across concurrent processes. |
| `layers` | `layers() -> list[int]` | Layer indices present in the store. |
| `get` | `get(keys: list[str], layer: int) -> np.ndarray` | Stack activations for `keys` at `layer` → `[n, d]` float32; rows for missing keys are filled with NaN. |

```python
from idea.mech.capture import ActStore
store = ActStore(run_dir / "acts" / "olmo_instruct_chat")
H = store.get(list(df.key_post), layer=16)     # [n, d], NaN rows where a key is absent
ok = np.isfinite(H).all(1)                      # mask out missing rows before fitting
```

The **released** steering vectors were derived from such stores, but the raw activation stores are
large and are **not** in the release; `get` is used by `analysis/10_p4_stage_probes.py`, which runs
where the activations are (FarmShare).

### `probe.py` — the masking probe

A ridge probe trained on **belief-neutral** trials (`b == 0`) to read the Bayesian posterior logit
(`lp_before + x`) out of the last-token residual stream, split by topic, then applied to
**contradicted** trials (`b != 0`, evidence against the belief).

Masking index = `median((read − stated) / (bayes − stated))` over trials where `|bayes − stated|`
is large:

- **≈ 1** — the internal readout sits at the Bayesian answer while the stated (output) answer does
  not: *response-selection masking* (the model "knows" the Bayesian answer but does not say it).
- **≈ 0** — the hidden state is as distorted as the output.
- **< 0** — the hidden state is even further from Bayes than the output.

| Function | Signature | Purpose |
|---|---|---|
| `split_topics` | `split_topics(topics, frac_test=0.3, frac_val=0.0, seed=0) -> (train, val, test)` | Seeded topic split into three sets (topics, never trials, so train/test never share a topic). |
| `train_probe` | `train_probe(H, df, train_topics, test_topics) -> (RidgeCV, r2)` | Fit `RidgeCV` on neutral trials in `train_topics`; `r2` on neutral trials in `test_topics`. Target = `df.lp_before + df.x`. |
| `masking_index` | `masking_index(probe, H, df, min_gap=0.5, by=None) -> DataFrame` | Masking index over contradicted trials with `|bayes − stated| ≥ min_gap`; `by` groups it (e.g. `"dimension"`, `"cue_type"`). Reports median and IQR. |
| `layer_sweep` | `layer_sweep(get_H, df, layers, train_topics, val_topics) -> DataFrame` | Held-out R² per layer on **validation** topics (used to pick the probe layer without touching test). |
| `run_masking` | `run_masking(get_H, df, layers, seed=0, signatures=None, r2_min=0.5) -> dict` | Full pipeline: split topics, choose the layer by validation R², fit on train+val, report masking index overall / by dimension / by cue type (and by signature if given). `interpretable` flags `r2_test ≥ r2_min`. |

`get_H` is a callable `layer -> [n, d]` aligned with `df` (e.g. `lambda l: store.get(keys, l)`).

```python
from idea.mech import probe
res = probe.run_masking(lambda l: store.get(list(df.key_post), l),
                        df, layers=[8, 12, 16, 20, 24], seed=0)
print(res["layer"], res["r2_test"], res["interpretable"])
print(res["overall"])          # n, masking_index, iqr_lo, iqr_hi
print(res["by_dimension"])
```

### `steer.py` — directions and steering (original RQ3)

Builds the caved-minus-resisted bias direction, removes topic variance, sweeps steering settings and
scores them against the preregistered success criterion.

**`d_bias`** = mean activation on contradicted trials where the model *caved* (`omega < 0.5`) minus
where it *resisted* (`omega ≥ 0.8`), differenced **within strata** of `(item_id, |x| bin, topic,
cue_id, cue_type)` and averaged, pooled and per dimension. Topic variance is projected out with PCA
on belief-neutral activations: `d̂ = (I − Uₖ Uₖᵀ) d_bias`. Steering applies
`h ← h − ψ · d̂/‖d̂‖`, with ψ a multiple of the SD of activations projected on the direction.

Key functions:

| Function | Signature | Purpose |
|---|---|---|
| `label_trials` | `label_trials(df, caved=0.5, resisted=0.8, x_min=0.5) -> DataFrame` | Tag contradicted, non-excluded, `|x| > x_min` trials as `caved`/`resisted`/`drop` and bin `|x|`. |
| `d_bias_matched` | `d_bias_matched(df, H, strata=(...), groups=None) -> dict` | Pooled & per-dimension `d_bias` and strata counts. `df` must be positionally aligned with `H` (`RangeIndex`). Falls back to coarser strata if none match. |
| `d_prompt_contrast` | `d_prompt_contrast(df, H, groups=None, evidence=("contra",)) -> ndarray` | Mean activation with a stated belief minus without, matched on `(item, cue)` (a label-noise control; `("contra","support")` gives the RQ3b hedging direction). |
| `topic_subspace` | `topic_subspace(H_neutral, k, seed=0) -> [d, k]` | Top-k PCA components of belief-neutral activations (the topic subspace). Seeded (sklearn's randomized SVD is otherwise unseeded). |
| `project_out` | `project_out(d, U) -> (d_hat, frac)` | Remove the component of `d` in `span(U)`; `frac` = fraction of `d`'s squared norm that was in the subspace. |
| `psi_scale` | `psi_scale(H, d) -> float` | SD of `H @ (d/‖d‖)` — the unit of steering strength ψ. |
| `random_direction` | `random_direction(d, seed=0) -> ndarray` | Random vector of equal norm (the matched control). |
| `build_directions` | `build_directions(df, get_H, layers, ks=(4,8,16), seed=0, groups=None) -> dict` | Per layer: raw & per-dim `d_bias`, topic-projected `hat_k{4,8,16}`, `topic_only_k*`, prompt contrast, random control, and the ψ `scale` for each. |
| `save_directions` / `load_directions` | `(dirs, path)` / `(path) -> dict` | Flatten to `path.npz` (+ `path.json` metadata, + a README); `load_directions` restores the exact saved values so a resumed run reuses the same vectors and ψ scales. |
| `slopes` | `slopes(df) -> dict` | Pooled Eq.-3 evidence slopes (neutral/contra/support) on steered or unsteered responses, overall and per dimension. |
| `delta_j_spread` | `delta_j_spread(df) -> dict` | Per-dimension SD of the unequal-deference `delta_j` (uses `idea.analysis.rq1`). |
| `success` | `success(unsteered, steered, tol=0.05, dims=None, spread_before=None, spread_after=None, hedging=False) -> dict` | The **preregistered criterion** (see below) as a report dict with a top-level `success` bool. |
| `evaluate_steering` | `evaluate_steering(backend, elicitor_factory, trials, topics, cues, fmt, layer, direction, psi, mode="all", stance=None) -> DataFrame` | Run `trials` with one steering entry set on `backend`; returns derived responses. |
| `sweep` | `sweep(backend, elicitor_factory, val_trials, topics, cues, fmt, layers, directions, multiples=(0.5,1,2,4,8), modes=("all","last"), unsteered=None, stance=None) -> DataFrame` | Grid over directions × layers × ψ multiples × modes on validation topics. |
| `signature_directions`, `specificity_matrix` | | RQ3b: one direction per behavioral signature; a direction×applied-to success matrix. |

**Preregistered success criterion** (`steer.success`): the contradicted-trial slope moves toward
the *unsteered* neutral slope, AND the neutral and Tier-1 slopes change by < `tol` (0.05) in every
identity dimension, AND the spread of `delta_j` within dimensions shrinks. Controls: a random
direction of equal norm and the topic-only component `U Uᵀ d_bias`. With `hedging=True` the targets
are all belief-stated slopes (contra and support), not only contradicted ones.

### `subspace.py` — principal angles between directions

Compares directions/subspaces geometrically against random baselines (e.g. are the per-dimension
belief directions the same direction? is `d_bias` aligned with the sycophancy direction?).

| Function | Signature | Purpose |
|---|---|---|
| `basis` | `basis(vectors) -> [d, r]` | Orthonormal basis (QR) spanning a list of vectors. |
| `angles` | `angles(S1, S2) -> degrees` | Principal angles between two subspaces (handles 1-D inputs). |
| `random_baseline` | `random_baseline(d, r, S2, n=1000, seed=0) -> ndarray` | Min principal angle of `n` random r-dim subspaces to `S2`. |
| `pairwise_directions` | `pairwise_directions(dirs, n_random=1000, seed=0) -> DataFrame` | Angle & cosine between every pair of single directions, with the random-pair 1st percentile and a p-value vs random. |
| `subspace_comparison` | `subspace_comparison(set1, set2, n_random=1000, seed=0) -> dict` | Min principal angle between the spans of two vector sets, vs the random baseline. |

### `other_biases.py` — sycophancy / authority / status-quo contrasts

Builds directions for three other cue-induced biases (following Gupta et al.), each exactly like
`d_bias` (same layer, same topic projection), so angle differences reflect the biases rather than
the construction. Opinion **sycophancy** (MMLU item with vs. without the user stating a wrong
answer), **authority** (the same wrong suggestion from an expert vs. a layperson), **status-quo**
(one option framed as the current default vs. neither; 16 built-in scenarios).

| Function | Signature | Purpose |
|---|---|---|
| `mmlu_items` | `mmlu_items(n=400, seed=0, split="test") -> DataFrame` | Sample MMLU items and pick a wrong option per item (needs `datasets`/`cais/mmlu`). |
| `sycophancy_requests` / `authority_requests` / `status_quo_requests` | `(items)` / `(items)` / `(seed=0)` → `(list[Request], list[dict])` | Build the paired `Request`s and an index of `(bias, item, cond, …)`. |
| `run_contrast_sets` | `run_contrast_sets(elicitor, act_store, layers, n_mmlu=400, seed=0) -> DataFrame` | Run all three sets through an elicitor; returns choices, probs and cache keys. |
| `bias_directions` | `bias_directions(df, act_store, layer, U=None) -> dict` | Prompt-contrast direction per bias (plus a behavioral caved-vs-resisted sycophancy direction), each projected by the same topic subspace `U` as `d_bias`. |

### `rq3v2.py` — corrected RQ3 and belief-subspace erasure

The original RQ3 built directions and probes from activations at the end of the plain chat prompt,
but almost all answers were produced **after** the answer prefix `"Probability (0-100): "` (99% of
OLMo Instruct's posteriors, 54% of Llama's). This module redoes everything at the **answer
position** and adds interventions from the literature:

- **Gupta et al. (2026)** — caved-minus-resisted direction at the answer position (`cr_hat_k8`),
  steered at one layer or at all layers.
- **Park, Choe & Veitch (2024)** — concept directions from **counterfactual** inputs that differ
  only in the stated belief: every training trial is rendered three ways — believes H1 / believes H0
  / no belief — and the activation differences give a clean "stated belief" signal (also measured
  after whitening by the no-belief covariance).
- **Engels et al. (2025)** — a concept as a low-dimensional **subspace**: intervene by *setting* the
  subspace coordinates to the no-belief value (erase the belief), test **sufficiency** by patching
  the subspace between counterfactual inputs, and intervene at the **belief-sentence tokens** (span)
  in early layers.

`run(cfg, model_key, fmt="chat")` runs five resumable stages, each saving its output under
`<run>/results/rq3_v2/<model>_<fmt>/`:

| Stage | Output file(s) | What it computes |
|---|---|---|
| 1. capture | `capture.npz`, `cf_trials.parquet`, `held_trials.parquet` | Answer-position activations for training trials, and counterfactual belief triples (answer position + mean over belief-sentence tokens), for training and held-out (validation) topics. |
| 2. directions + geometry | `directions_v2.npz`, `geometry.csv` | Belief subspaces (answer position `ans_U2/U4`; belief tokens `span_U1/U2`), caved-resisted `cr_hat_k8`, random controls, and per-layer geometry (norms, cosine of the two belief directions, whitened cosine, SV ratio, variance share, counterfactual-answer probabilities). |
| 3. sufficiency | `sufficiency.csv` | Interchange patching on held-out counterfactual pairs: share of the belief effect recovered by setting each subspace from the source belief into the target input, per layer and patch type. |
| 4. sweep | `sweep_validation_v2.csv` | Interventions on validation topics (steer `cr_hat_k8`; erase `ans`/`span` subspaces at several layers and k). |
| 5. test | `steering_test_v2.csv`, `summary_v2.json` | Best setting per family + a matched random-subspace control, on held-out test topics; preregistered + exploratory success and share of gap closed. |

Low-level and selection helpers:

| Function | Signature | Purpose |
|---|---|---|
| `forward_states` | `forward_states(backend, reqs, layers, prefill=True, patch=None, bs=16) -> (p, last, span)` | Forward pass with the answer prefix; returns the answer readout `p[n]`, last-token states `last[layer] -> [n,d]`, and belief-sentence-span means `span[layer] -> [n,d]`. `patch` optionally **sets** per-row subspace coordinates (or replaces the whole last vector) during the pass. |
| `entries_for` | `entries_for(D, family, setting, layers) -> entries` | Rebuild a sweep setting's steering entries from a loaded `directions_v2.npz`. Families: `cr_single`, `cr_all_layers`, `erase_answer`, `erase_span`, `erase_span_multi`. |
| `control_for` | `control_for(D, family, setting, entries, seed=0)` | The matched random control (random subspace for erasure; equal-norm random directions for steering). |
| `partial_success` | `partial_success(t, bar) -> Series` | "Tier-3" rule: pooled gap closes ≥ `bar`, contradicted slope moves toward neutral, Tier-1 & Tier-2 neutral slopes stable. |
| `pick_winner` | `pick_winner(out) -> (dict | None, reason)` | Best tested intervention passing preregistered → exploratory → partial, requiring ≥ 25% of the gap closed (and ≥ 10 points over the best random control) before an audit is worth running. |
| `resolve_audit` / `run_audit` | `(cfg, model_key, part/cond, fmt="chat")` | Resolve and run the capability audit (SycEval, BBQ, MMLU-Pro, L2D) under `unsteered` / `corrected` / `control`, or a named intervention `<cond>@<family>:<setting>`. Span-erasure winners are skipped (they cannot affect benchmark prompts, which contain no belief sentence). |
| `run_p4` / `p4_summary` | `(cfg, model_key, fmt="chat")` / `(cfg, stages, fmt)` | P4 redone at the answer position: per-layer contradicted-vs-supported probe and the stage's caved-resisted direction; `p4_summary` compares stages' directions to Instruct's. |

---

## The released vector files

Each model folder under `steering_vectors/` holds the exact vectors the RQ3 experiments used. Hidden
size is 4096 for all three models.

### `directions.*` (original RQ3; `olmo_instruct/`, `llama/`)

- **`directions.npz`** — vectors keyed `L<layer>/<name>` for layers 8–26:
  - `hat_k4`, `hat_k8` (main), `hat_k16` — `d_bias` with the top-k topic PCA components removed
  - `raw` — `d_bias` itself (caved minus resisted, matched within strata)
  - `random` — equal-norm random control; `topic_only_k*` — the removed topic component
  - `prompt`, `prompt_hat_k*` — belief-stated-minus-not prompt contrast (label-noise control)
  - `per_dim_raw/<dim>`, `per_dim_hat_k*/<dim>` — per-dimension directions
- **`directions.json`** — per layer: `scale` (SD of the activation projection on each vector — the
  unit of ψ), strata counts, and the share of `d_bias` lying in the topic subspace.
- **`summary.json`** — the chosen setting: layer, strength multiple, mode (`all` tokens or `last`).

Load with `steer.load_directions("steering_vectors/llama/directions")` (note: no extension — it
appends `.npz`/`.json`), or read one array directly with `numpy.load`.

### `belief_subspaces_v2.npz` (corrected RQ3; all three models)

Keys `L<layer>/<name>` for layers 4, 8, 12, 16, 20, 24:

| Key | Shape | Meaning |
|---|---|---|
| `ans_U2`, `ans_U4` | `[2,d]`, `[4,d]` | Orthonormal basis of the belief subspace at the **answer position**, from counterfactual inputs differing only in the stated belief. |
| `ans_t2`, `ans_t4` | `[2]`, `[4]` | The no-belief **target** coordinates for erasure: `h ← h − Uᵀ(Uh − t)`. |
| `span_U1`, `span_U2` | `[1,d]`, `[2,d]` | The same at the **belief-sentence tokens**; `span_t1`, `span_t2` their targets. |
| `rand_U2`, `rand_U4` | | Random subspaces of the same size (controls); `rand_t2`, `rand_t4` their targets. |
| `cr_hat_k8` | `[d]` | Counterfactual caved-resisted direction with the top-8 topic components removed. |
| `cr_scale` | scalar | Its ψ norm scale. |

Applying the erasure in reverse (moving the coordinates *away* from the no-belief target)
strengthens belief-following. See the paper's "RQ3: masking and steering" section and
`src/idea/mech/rq3v2.py`.

```python
import numpy as np
Z = np.load("steering_vectors/olmo_instruct/belief_subspaces_v2.npz")
U = Z["L16/ans_U4"]        # [4, 4096] belief subspace at the answer position, layer 16
t = Z["L16/ans_t4"]        # [4] no-belief target
```

---

## Applying an intervention: `set_steering`

Interventions are installed on a Hugging Face backend (`idea.elicit.HFBackend.set_steering`) as a
list of entries; they act on the **output of decoder layer `layer`** and persist until cleared with
`set_steering([])`. Two entry forms:

- **Vector steering** — `(layer, direction, psi, mode)` with `mode` `"all"` or `"last"`:
  `h ← h − ψ · direction/‖direction‖`. ψ > 0 removes the bias; ψ < 0 amplifies it. The direction is
  normalized internally, so ψ carries the magnitude (use `multiple × scale`).
- **Subspace erasure** — `{"kind": "erase", "layer": l, "U": [k,d], "target": [k], "pos": "last"|"all"|"span"}`:
  the coordinates of `h` in `span(U)` are set to `target` (erase the belief / set a target value), at
  the last position, every position, or the belief-sentence tokens (`request.meta["span"]`).

```python
backend.set_steering([(16, direction, 4 * scale, "all")])       # steer
backend.set_steering([{"kind": "erase", "layer": 16,            # erase belief subspace
                       "U": U, "target": t, "pos": "last"}])
# ... run trials / audit ...
backend.set_steering([])                                        # clear
```

---

## Worked examples

### 1. Capture activations and build the original `d_bias` directions

```python
import numpy as np
from idea.mech.capture import ActStore
from idea.mech import steer, probe

store = ActStore(run_dir / "acts" / "olmo_instruct_chat")      # keyed by request cache key
get_H = lambda l: store.get(list(df.key_post), l)              # df aligned to the keys

layers = [8, 12, 16, 20, 24]
dirs = steer.build_directions(df.reset_index(drop=True), get_H, layers, ks=(4, 8, 16), seed=0)
steer.save_directions(dirs, "my_directions")                   # -> my_directions.npz/.json/.README.txt

d_hat = dirs[16]["hat_k8"]          # topic-projected bias direction at layer 16
scale = dirs[16]["scale"]["hat_k8"] # psi unit at that layer
```

### 2. Belief-subspace erasure from the released file

```python
import numpy as np
Z = np.load("steering_vectors/olmo_instruct/belief_subspaces_v2.npz")
layer, k = 16, 4
U, t = Z[f"L{layer}/ans_U{k}"], Z[f"L{layer}/ans_t{k}"]

backend.set_steering([{"kind": "erase", "layer": layer, "U": U, "target": t, "pos": "last"}])
# run the trials through the elicitor here; then:
backend.set_steering([])
```

To **amplify** instead of erase, steer away from the target along the subspace, or apply a
reversed-ψ vector steer with `cr_hat_k8`:

```python
backend.set_steering([(layer, Z[f"L{layer}/cr_hat_k8"], -4 * float(Z[f"L{layer}/cr_scale"]), "all")])
```

### 3. Sufficiency / interchange test (recover the belief effect by patching a subspace)

`rq3v2.forward_states(..., patch=...)` sets per-row subspace coordinates during the forward pass.
The `patch` dict is `{"layer": l, "U": [k,d], "targets": [n,k], "pos": "last"|"span"}` (or
`{"layer": l, "full": [n,d], "pos": "last"}` to replace the whole last-position vector). Patching the
**target** belief's coordinates into the **source** belief's input and measuring how far the answer
moves gives the recovery fraction:

```python
from idea.mech.rq3v2 import forward_states
# reqs rendered for the source belief; targets = source states projected onto the target's values
pt = dict(layer=16, U=U, targets=(target_states @ U.T), pos="last")
p_patched, _, _ = forward_states(backend, reqs, [16], patch=pt)
```

In practice this is driven by `rq3v2.run` stage 3, which writes `sufficiency.csv` (recovery by
layer, patch type, and swap direction). `analysis/11_rq3_corrected.py` averages the two swap
directions into the released table.

### 4. Masking probe

```python
from idea.mech import probe
res = probe.run_masking(lambda l: store.get(list(df.key_post), l),
                        df, layers=[8, 12, 16, 20, 24], seed=0)
# res["layer"], res["r2_test"], res["interpretable"],
# res["overall"], res["by_dimension"], res["by_cue_type"]
```

### 5. Steering sweep and the preregistered success check

```python
from idea.mech import steer
per_layer = {l: {"vec": dirs[l]["hat_k8"], "scale": dirs[l]["scale"]["hat_k8"]} for l in layers}
sweep = steer.sweep(backend, elicitor_factory, val_trials, topics, cues, fmt="chat",
                    layers=layers, directions={"hat_k8": per_layer},
                    multiples=(0.5, 1, 2, 4, 8), modes=("all", "last"))

unsteered = steer.slopes(steer.evaluate_steering(backend, elicitor_factory, test_trials,
                                                 topics, cues, "chat", layers[0], np.ones(1), 0.0))
steered  = steer.slopes(steer.evaluate_steering(backend, elicitor_factory, test_trials,
                                                 topics, cues, "chat", 16, dirs[16]["hat_k8"],
                                                 4 * dirs[16]["scale"]["hat_k8"], "all"))
report = steer.success(unsteered, steered)      # report["success"] is the preregistered bool
```

Neither released model met the preregistered success criterion under the original (prompt-position)
RQ3; the corrected answer-position erasure is the stronger intervention. See the secondary paper and
`analysis/06_rq3_steering_audit.py` / `analysis/11_rq3_corrected.py` for the published tables.
