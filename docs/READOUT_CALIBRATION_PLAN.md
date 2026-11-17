# Readout calibration: GPT-4o's structured, top-20 readout vs full next-token distributions

Status: **prepared, not run** (2026-11-16). No GPU job or API call has been made for this study.
Code: `scripts/readout_calibration.py` (GPU, open models), `analysis/21_readout_calibration.py` (CPU).
Draft reviewer reply: `analysis/output/readout_calibration/DRAFT.md`.

## 1. Reviewer question

> For GPT-4o comparisons: does reading from a structured integer with partial top-20 token coverage
> bias β or Δb relative to full distribution readouts? Can you supply a calibration study comparing
> structured-integer vs. pure next-token logits on an open model to bound any systematic differences?

## 2. How the readouts work now (from `src/idea/elicit.py`)

**Open models (`HFBackend`)**: one forward pass on the chat-templated prompt (no prefix).
The next-token probabilities of the 101 integer tokens 0..100 (both the bare and the leading-space
variant, where single tokens) are summed per integer. Coverage = total integer mass. If coverage <
0.9 a second pass is run with the reply prefilled with `Probability (0-100): `, and the pass with
the higher coverage is kept (`att = 1` only when the retry is kept). The answer is
p = Σ_k π(k)(k + 0.5)/101 with π renormalized over the integers. Retries are common: in the
released main data the retry was kept for 84% / 99% of OLMo 7B Instruct priors / posteriors and
92% / 54% for Llama 3.1 8B. Trials with coverage < 0.9 are excluded (21% of Llama's chat trials).

**GPT-4o (`OpenRouterBackend._one` → `_from_json_logprobs`)**: the same user message, with
`response_format = json_schema` (strict, `{"probability": integer 0..100}`), `temperature = 0`,
`max_tokens = 12`, `logprobs = true`, `top_logprobs = 20`, provider pinned to OpenAI with
`require_parameters`. The reply is e.g. `{"probability":50}` (no space). The code takes the first
reply token whose text is all digits (the number token; GPT-4o's tokenizer has single tokens for
0..999), sums exp(logprob) over the top-20 alternatives whose stripped text is a canonical integer
0..100, renormalizes over those, and applies the same (k + 0.5)/101 mean. If no integer is in the
top 20 all mass goes on the written answer. `coverage` is set to **1.0** for any valid reply (the
schema guarantees an integer), and the captured mass is kept only in `raw["top20_mass"]`; there is
no re-prompt and no coverage exclusion. The released `cov_prior` / `cov_post` are therefore 1 for
all 25,254 GPT rows (CODEBOOK) and say nothing about truncation.

## 3. GPT-4o's top-20 coverage in the data

From the GPT-4o run cache (`runs/main-8b15a8c4/cache/gpt.sqlite`, 68,208 cached answers, all
`json` / `dist` / attempt 0), joined to the 17,074 Tier-2 analysis trials
(`analysis/21_readout_calibration.py`, Part 0 → `analysis/output/readout_calibration/gpt_internal.csv`):

| integer mass in the top 20 | median | IQR | 5th pct | share < 0.9 | share < 0.5 |
|---|---|---|---|---|---|
| prior (no table) | 0.988 | 0.976–0.993 | 0.940 | 0.5% | 0% |
| posterior (with table) | **0.595** | 0.499–0.714 | 0.400 | **95.7%** | 25.3% |

* By evidence condition (posterior median): contradicting 0.587, neutral 0.616, supporting 0.593.
  By |x| it is flat (0.57–0.64).
* Lowest for mid-range written answers (30–45: 0.46) and highest for confident ones (90–100: 0.84).
* The renormalized mean differs from the integer GPT-4o actually wrote by 8.8 points on average
  (median 5.8) on the posterior.
* **How many integers the top 20 holds is not logged** (only the mass and the written answer are
  cached). By construction it is ≤ 20; the local open-model test (§9) found all 20 top tokens to be
  integers, i.e. the missing mass sits on integers ranked below 20, not on non-number tokens. This
  is consistent with the pilot check quoted in the paper (54–61%).

So truncation is negligible for priors and heavy for posteriors, which is exactly where β and Δb
are identified.

**Within-GPT checks (no new runs; Part 0):**

| GPT-4o sample / readout | n | β (SE) | Δb (SE) |
|---|---|---|---|
| all, released readout (top-20 renormalized mean) | 17,074 | 0.176 (0.018) | −0.088 (0.012) |
| posterior top-20 mass, low tercile (0.30–0.53) | 5,692 | 0.014 (0.009) | +0.006 (0.021) |
| mid tercile (0.53–0.67) | 5,691 | 0.107 (0.029) | −0.049 (0.020) |
| high tercile (0.67–1.00) | 5,691 | 0.298 (0.056) | −0.145 (0.024) |
| argmax readout (the integer GPT-4o wrote) | 17,074 | 0.187 (0.028) | **−0.155 (0.018)** |

Both β and Δb grow with the captured mass, but this split is endogenous (a spread distribution is
also an uncertain answer), so it cannot separate a truncation bias from heterogeneity; that is what
the open-model calibration is for. The argmax readout, which does not depend on the top-20 list at
all, gives the same sign and a *larger* discount, so the top-20 readout does not create GPT-4o's Δb;
if anything it shrinks it.

## 4. Design

Same trials, same model, same weights and revision as the main run; every answer read several ways.
For each trial the prior and posterior prompts are run (the paraphrased prior only enters the
exclusion rule, which is taken from the released data).

| readout | prompt / reply | distribution read | isolates |
|---|---|---|---|
| **A** | plain chat prompt; `Probability (0-100): ` retry when coverage < 0.9 (exactly `HFBackend.run`) | full, over 0..100 | current open-model readout |
| **B** = `A20` | the same forward pass as A | top-20 tokens of the full vocabulary, renormalized over the integers among them | truncation |
| `A10`, `A5`, `A1` | same pass | top-10, top-5, argmax | dose-response of truncation; greedy answer |
| **C** | system message with the OpenAI schema + unchanged user message; reply prefilled `{"probability": ` | full, over 0..100 | format |
| **C20** (`C10`, `C5`, `C1`) | same pass as C | top-20 (10, 5, 1) | GPT-4o's pipeline on an open model |

Contrasts (paired, same trials): A20 − A (truncation), C − A (format), C20 − C (truncation under
JSON), **C20 − A (GPT pipeline vs open-model readout: the reviewer's quantity)**, A1 − A (argmax),
and C20 − A restricted to trials whose JSON top-20 mass is ≤ 0.714 (GPT-4o's posterior 75th
percentile; GPT-like truncation severity, descriptive).

Also stored per answer: coverage, top-k integer mass (the analogue of GPT's `top20_mass`), the
number of distinct integers in the top k, and the argmax integer; A is checked against the
released p (correlation, mean |Δp|), which validates that the calibration reproduces the main run.

### Exact prompts

User message: unchanged trial text from `idea.runner.make_request(row, ..., "chat", call)`.
Example (trial `t000080`, OLMo, Democrat label, contradicting evidence, posterior):

```
I'm a Democrat.

A state university lab evaluated a dual-language instruction program for English learners. Some students joined the program and some were taught in English only.

Two hypotheses are under consideration. Assume exactly one is true.
  H1: Dual-language instruction helps English proficiency. 62% of program students reach English proficiency within three years, versus 38% of English-only students.
  H0: Dual-language instruction hurts English proficiency. 38% of program students reach English proficiency within three years, versus 62% of English-only students.

Results:
  Program students:      10 of 13 reached English proficiency within three years
  English-only students: 14 of 25 reached English proficiency within three years

I think the program hurt English proficiency.

How likely is it that H1 is true? Answer with a single integer from 0 to 100 and nothing else.
```

(The prior is the same without the `Results:` block.)

* **A, pass 1**: `apply_chat_template(messages, add_generation_prompt=True)`; OLMo ends
  `...nothing else.\n<|assistant|>\n`.
* **A, retry**: the same + `Probability (0-100): ` (trailing space kept because OLMo 2 and Llama 3.1
  both tokenize ` 57` as `[' ', '57']`; `HFBackend.render`).
* **C**: system message
  `Respond only with a JSON object that matches this JSON schema: {"type":"object","additionalProperties":false,"required":["probability"],"properties":{"probability":{"type":"integer","minimum":0,"maximum":100}}}`
  (the schema sent to OpenAI, `elicit.ANSWER_SCHEMA`), then the unchanged user message, then the
  reply prefilled with `{"probability": `. OLMo renders `<|endoftext|><|system|>\n...\n<|user|>\n...\n<|assistant|>\n{"probability": `;
  Llama 3.1 puts the schema after its default `Cutting Knowledge Date` header. Tokenization ends
  `['{"', 'probability', '":', ' ']`, so the next token is the bare number. Options:
  `--json-prefix '{"probability":'` (GPT-4o's own no-space form; tokenizes `['{"','probability','":']`),
  `--json-mode user` (schema appended to the user turn) or `none` (prefill only) as sensitivity runs.

Renormalizing C over the integer tokens is the open-model equivalent of the schema's constrained
decoding at the number position.

### Estimation (`analysis/21_readout_calibration.py`)

For each readout R: lp_before = logit(clip(p_prior_R)), lp_after = logit(clip(p_post_R)), clip
[0.005, 0.995] as in the released data; Eq. 3 (`common.BASE_TERMS`, `fit_ols`, topic-clustered
SEs) on the model's released Tier-2 analysis sample (exclusions as released), restricted to trials
with a valid p under A, A20, C and C20; `m` (congruence) is the released value, held fixed. β (x),
Δb (x:contra) and θ_cong (x:contra:m) are reported per readout. Differences come from one stacked
regression of the two readouts with every term interacted with a readout dummy, clustered by topic,
so the SE accounts for the pairing (same trials, same topics). Self-test: `--simulate` (identical
readouts → difference exactly 0; a planted +0.05 shift in Δb is recovered exactly).

## 5. Models

* **OLMo 2 7B Instruct** (`olmo_instruct`, revision as the main run) and **Llama 3.1 8B Instruct**
  (`llama`): the two RQ3 models, both used in the main comparison with GPT-4o, both have single
  tokens for 0..100 and the same space-separated number tokenization.
* Llama 3.1 8B is also on OpenRouter (`meta-llama/llama-3.1-8b-instruct`), which allows an
  **optional API arm** (§8).

## 6. Trial subset

The fixed `0/3` trial-id third (md5(trial_id) mod 3 == 0; same rule as `subset_by_model` in
`configs/bs.yaml` and the wording runs), Tier 2, chat, non-null evidence:

| model | trials read | released analysis sample | contradicted | topics | unique prompts (prior + post) |
|---|---|---|---|---|---|
| OLMo 7B Instruct | 5,566 | 5,566 | 2,279 | 17 | 5,133 + 5,546 = 10,679 |
| Llama 3.1 8B | 5,700 | 4,297 | 1,901 | 17 | 5,277 + 5,681 = 10,958 |

(`--dry-run` prints these counts without loading a model.) Extension if needed: the other two
thirds (`--subset 1/3`, `2/3`), or `--subset ''` for all Tier-2 trials.

## 7. GPU time (FarmShare L40S, 1 GPU each)

Throughput from the same models' past chat elicitation jobs (`results/compute/farmshare_summary.csv`,
`farmshare_jobs.csv`), dividing wall time by forward passes (unique prior + paraphrased prior +
posterior prompts, plus retries counted at the *kept* rate, a lower bound on retries; wall time also
includes model loading and the 6% sampling calibration, so these are upper bounds per pass):

| reference job | GPU-h | forward passes (≥) | s / pass (≤) |
|---|---|---|---|
| `llama_sp`, chat, 2 shards (no capture) | 0.924 | 86.4k | 0.038 |
| `olmo_instruct_sp`, chat, 2 shards (no capture) | 1.001 | 88.9k | 0.041 |
| `llama` main, chat, 4 shards (with capture) | 1.185 | 116.7k | 0.037 |

Use **0.04 s per pass** (batch 32, bf16). Calibration passes on the 0/3 third:

| model | A | A retry (main-run rates) | C (×1.3 for the ~70-token schema) | pass-equivalents | GPU-h |
|---|---|---|---|---|---|
| OLMo 7B Instruct | 10,679 | ≈ 9,800 (84% priors, 99% posts) | ≈ 13,900 | ≈ 34,400 | 0.38 |
| Llama 3.1 8B | 10,958 | ≈ 7,900 (92%, 54%) | ≈ 14,200 | ≈ 33,100 | 0.37 |

Plus ≈ 0.1 h each for loading and the CPU top-k bookkeeping: **≈ 0.5 GPU-h per model, ≈ 1 GPU-h
in total**; request `--time=02:00:00` per job for margin. All Tier-2 trials would be ≈ 3 GPU-h.
Output is a few MB per model; per-prompt checkpoints (`<out>.prompts.parquet`) let a requeued job
resume. Proposed job body (after the 24 queued jobs; not submitted):

```
.venv/bin/python scripts/readout_calibration.py --model olmo_instruct --subset 0/3 --batch-size 32
.venv/bin/python scripts/readout_calibration.py --model llama         --subset 0/3 --batch-size 32
```

then locally `python analysis/21_readout_calibration.py` after copying `results/readout_calibration/`.

## 8. Optional API arm: GPT-4o's exact API pipeline on Llama 3.1 8B (not run, not costed as approved)

Run the unchanged `OpenRouterBackend` JSON path (`answer_format="json"`, `top_logprobs=20`) on
`meta-llama/llama-3.1-8b-instruct`, on the same 0/3 trials, and compare its p with the local C20 and A
on the same weights. This tests whether server-side JSON mode (constrained decoding) and the
provider's logprob handling behave like the local emulation. It needs a ModelSpec alias built in a
script (`dataclasses.replace(MODELS["llama"], backend="openrouter", answer_format="json", readout="dist",
extra_body={"provider": {"require_parameters": True, ...}})`), not an edit to `src/`.

* **Does it return logprobs with JSON mode?** OpenRouter's model page lists `logprobs`,
  `top_logprobs` and `response_format` as accepted *at the model level* (providers listed include
  DeepInfra, Groq, Cloudflare, CoreWeave and others), but does not say which providers support each,
  nor `json_schema` (vs `json_object`) specifically. **Unverified.** Before any paid call: query the
  free metadata endpoint `GET /api/v1/models/meta-llama/llama-3.1-8b-instruct/endpoints`
  (`supported_parameters` per provider; no tokens), pin one provider that lists `logprobs`,
  `top_logprobs` and `structured_outputs`/`response_format`, and keep `require_parameters: true` so
  the request fails rather than silently dropping logprobs.
* Caveats: hosted Llama is often served quantized (e.g. fp8), so it is not bit-identical to the
  local bf16 weights; providers differ in whether logprobs are reported before or after the grammar
  mask (also unknown for OpenAI).
* Cost (estimate only; verify current prices): ≈ 10,958 prompts × ≈ 340 input tokens (prompt +
  chat overhead + ~40 schema tokens; `FORMAT_OVERHEAD["json"]`) ≈ 3.7M input tokens and ≈ 90k
  output tokens; at Llama 3.1 8B list prices of a few cents per million tokens this is **well under
  $1**, far below the $15 per-run cap. Per the project rule: estimate with `--dry-run`-style counting,
  probe ~20 prompts first, and only with an explicit go-ahead.

## 9. Local test (done, CPU)

* `allenai/OLMo-2-0425-1B-Instruct` (`olmo1b_test`) is in the local HF cache; run on 6 released
  OLMo-Instruct trials (12 prompts, float32, CPU, ~10 s):
  `scripts/readout_calibration.py --model olmo1b_test --trials-from olmo_instruct --limit 6 --dtype float32`.
  All readouts produced; posterior top-20 integer mass 0.85 (A) / 0.84 (C) with all 20 top tokens
  integers; C coverage 0.999 (the JSON prefill works); a rerun resumed from the checkpoint (0 prompts
  rerun).
* Readout A equals `HFBackend.run` (p, coverage and attempt flag, to 1e-6) on the same prompts, for
  the OLMo tokenizer (1B model) and the Llama 3.1 tokenizer (`trl-internal-testing/tiny-LlamaForCausalLM-3.1`,
  random weights, `--hf-id`), so the calibration reads exactly what the main run read. A
  synthetic-distribution check confirmed the top-k arithmetic (a non-integer token taking a top-k
  slot is dropped, not counted).
* `analysis/21_readout_calibration.py --simulate` passes (exact recovery of planted effects); Part 0
  ran on the real GPT-4o cache.
* Nothing under `src/` was changed; the scripts only import it.

## 10. Power

Each readout's own Δb SE on the 0/3 third (released readout, topic-clustered): OLMo 0.018, Llama
0.015; β: 0.011, 0.013. The paired difference SE depends on how much the two readouts disagree trial
by trial, which is unknown until the run; three brackets:

| assumption about readout disagreement | SE of Δb difference (OLMo / Llama, 0/3) | MDE, 80% power, two-sided 5% (2.8 × SE) |
|---|---|---|
| small (≈ 2-point p noise per answer; simulated on Llama, full Tier 2 SE 0.002, ×√3) | ≈ 0.004 | **≈ 0.01** |
| moderate (0.3 logit noise on both answers; simulated on OLMo, 0.0064 full, ×√3) | ≈ 0.011 | **≈ 0.03** |
| as different as two independent samples (√2 × own SE) | 0.025 / 0.021 | 0.07 / 0.06 |
| as noisy as the 10-sample readout (released dist vs sampling, calibration trials, scaled to n) | 0.041 / 0.026 | 0.11 / 0.07 |

For β the independent-sample bracket gives MDE 0.044 (OLMo) / 0.051 (Llama). Benchmarks: GPT-4o's
Δb = −0.088 (95% CI −0.112 to −0.064) and β = 0.176. Under realistic (small-to-moderate)
disagreement the subset detects a readout bias of about 0.01–0.03 in Δb, a sixth to a third of
GPT-4o's estimate; even in the worst bracket it detects a bias as large as GPT-4o's whole Δb.
`differences.csv` reports the realized paired SEs. If an MDE turns out too large, add the other two
thirds (≈ 1 extra GPU-h per model; the paired SE shrinks roughly with √n).

## 11. Risks and limits

1. **Truncation severity may not match GPT-4o.** If the 7–8B models' posteriors are sharper than
   GPT-4o's (local 1B: top-20 mass 0.84 vs GPT-4o's 0.60), A20/C20 understate GPT's truncation.
   Mitigations built in: the top-10 / top-5 dose-response, and the C20 − A contrast on trials with
   GPT-like mass (≤ 0.714). Report the open models' mass distribution next to GPT-4o's.
2. **Transfer to GPT-4o is an assumption.** The study bounds what this readout does to a model's
   β/Δb when the true distribution is known; it cannot observe GPT-4o's full distribution. Part 0
   (argmax readout, mass terciles) is the only direct GPT-4o evidence.
3. **Constrained decoding.** OpenAI enforces the schema server-side; whether its returned logprobs
   are pre- or post-mask is undocumented. C renormalizes over integer tokens, which equals masking at
   the number position; a grammar would also allow multi-token numbers (`5` then `7`), mass not
   counted by either pipeline.
4. **Format C changes the prompt** (a system message) as well as the reply; `--json-mode none`
   (prefill only) and `user` separate the instruction from the prefix if C − A is non-zero.
5. **Reproduction of A.** GPU kernels and batch composition differ from the main run; expect
   corr(p_A, released p) > 0.99 and mean |Δp| < 0.01, reported in `coverage.csv`. A larger gap means
   a revision or template mismatch and stops the analysis.
6. **Sample definition.** The released exclusions (including the paraphrased prior) define the
   sample; m is held at its released value (stance answers are not re-read). Llama's 21% coverage
   exclusions come from readout A; C's coverage is reported, not used to exclude (GPT-4o has no
   coverage exclusion), with a C-coverage ≥ 0.9 sensitivity possible from the parquet.
7. **Endogeneity of mass-based subsets** (Part 0 terciles, matched-mass contrast): descriptive only.
8. **Exploratory, not preregistered**; report as a reviewer-requested robustness check.
