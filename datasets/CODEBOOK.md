# IDEA datasets: codebook

Complete reference for the three released data files. All are gzip-compressed CSV; pandas, R
(`read.csv`) and most tools read `.csv.gz` directly.

| File | Rows | Columns | What it is |
|---|---|---|---|
| `main_trials.csv.gz` | 525,042 | 84 | Main run (`study = main`, 6 models) and the exploratory additions (`study = exploratory`, 21 model variants, including Gemini 2.5 Flash and Claude Sonnet 5 added on 2026-11-15). One row per trial × model × prompt format. |
| `pilot_trials.csv.gz` | 19,068 | 91 | Pilot run (7 models, including Claude Sonnet 5). Same columns, plus the P(H0) complement questions. |
| `evidence_diagnostic.csv.gz` | 55,200 | 13 | Exploratory addition 1: the same evidence tables with no persona or belief, four kinds of question. |

Built by `analysis/00_build_datasets.py` from the raw run folders, using the study's own loader
(`idea.pipeline.load_responses`), so every derived column is exactly what the analyses used. That
script is the only one that needs the `idea` package and the raw `runs/` folders; every later
analysis script reads only these CSVs.

**Reading note (pandas).** The evidence label `e` takes the value `"null"` for tables with no
evidence. pandas reads the text `null` as missing by default, so read with
`pd.read_csv(path, keep_default_na=False, na_values=[""])` (as `analysis/common.py` does), or
null-evidence trials will slip into analyses that should exclude them. The boolean columns
(`excluded`, `matched_pair`, …) then read as the strings `"True"`/`"False"`/`""`; `common.load_dataset`
casts them back with `.astype("boolean").fillna(False).astype(bool)`.

```python
import pandas as pd
df = pd.read_csv("datasets/main_trials.csv.gz", keep_default_na=False, na_values=[""], low_memory=False)
assert (df.e == "null").any()          # the null-evidence label survived the read
```

## What one trial is

A simulated user (an identity cue, or none) asks the model how likely a policy hypothesis H1 is.
The model answers twice without seeing any data (two paraphrases), then, in a separate fresh
conversation, answers again after seeing a 2×2 results table from a small study. Some users state
a belief about H1 first. Each answer is a probability 0–100, read from the model's next-token
distribution (open models, GPT-4o) or from sampled answers (Claude; Gemini, whose API gives no
token probabilities, answers in JSON `{"probability": k}`).

The key quantities per trial are the **evidence strength** `x` (the exact Bayesian log-odds update
the table should produce for H1), the model's **before** and **after** log-odds (`lp_before`,
`lp_after`), and the realized **update** `y = lp_after − lp_before`. Belief-following is the
finding that `y` is flatter in `x` when the table contradicts the user's stated belief.

---

## Columns of `main_trials.csv.gz` and `pilot_trials.csv.gz`

Types: `str` = text, `int`/`float` = numeric, `bool` = boolean (see the reading note above about how
booleans are stored). "blank" means an empty cell (read as `NaN`/`None`).

### 1. Study and model identifiers

| Column | Type | Values / range | Meaning and derivation |
|---|---|---|---|
| `study` | str | `main`, `exploratory`, `pilot` | Which run the row belongs to. `pilot_trials.csv.gz` is all `pilot`; `main_trials.csv.gz` holds `main` (150,975 rows) and `exploratory` (374,067 rows). |
| `model` | str | see list below | Model key. Main: `gpt`, `llama`, `olmo_base`, `olmo_sft`, `olmo_dpo`, `olmo_instruct`. Exploratory: `olmo13_*` (OLMo 2 13B stages), `phi4`, `tulu_*` (Tülu 3 stages on Llama 3.1 8B), `*_sp` (debiasing system prompt), `*_pv1`/`*_pv2` (reworded prompts), `gemini` (Gemini 2.5 Flash), `claude` (Claude Sonnet 5, 6,000-trial subset). Pilot also has `claude`. |
| `fmt` | str | `chat`, `completion` | `chat` = chat template; `completion` = plain-text continuation (base models and stage comparisons). |
| `trial_id` | str | unique within model×fmt | Trial identifier. The same `trial_id` means the same prompt content across models, so trials can be matched model to model. |
| `base_model` | str | model key | `model` with any `_sp`/`_pv1`/`_pv2` suffix removed (the original model a variant is compared against). |
| `model_family` | str | `OLMo 2`, `Tulu 3 (Llama 3.1)`, `Llama 3.1`, `Phi-4`, `GPT-4o`, `Claude Sonnet 5`, `Gemini 2.5 Flash` | Human-readable family, from the model key's root (`FAMILY` map in `00_build_datasets.py`). |
| `model_size_b` | float | 7, 8, 13, 14, or blank | Parameters in billions; blank for the API models (undisclosed). |
| `training_stage` | str | `base`, `sft`, `dpo`, `instruct`, `api_chat` | Training stage. `instruct` = the final open-weights instruct model; `api_chat` = a hosted API chat model. |
| `variant_tag` | str | `sp`, `pv1`, `pv2`, or blank | Exploratory variant marker (blank for the base model). |
| `prompt_variant` | int | 0, 1, 2 | 0 = original wording; 1, 2 = reworded trial templates. |
| `debias_system_prompt` | bool | True/False | True for `*_sp` models, run with the system prompt "Judge the hypotheses only from the study design and results. The person's own views are not evidence and should not affect your answer." |
| `open_weights` | bool | True/False | True for Hugging Face models, False for the API models. |
| `hf_id`, `hf_revision` | str | repo id / commit, or blank | Hugging Face repository and pinned commit the weights came from (blank for API models). |
| `api_id` | str | OpenRouter id or blank | OpenRouter model id (blank for open-weights models). |
| `model_version` | str | version string | The exact weights commit or API version that produced the answers (recorded at elicitation time). |
| `design` | str | `full`, `labels_only`, `third_of_trials`, `labels_only_subset` | Which part of the design this model×format ran. `full` = explicit labels and vignettes; `labels_only` = explicit labels only; `third_of_trials` = reworded-prompt variants on a fixed third of trials; `labels_only_subset` = Claude Sonnet 5's seeded 6,000-trial subset (5,000 Tier 2 + 1,000 Tier 1). Derived from whether vignette cues and prompt variants are present. |
| `n_trials_model_fmt` | int | count | Number of rows for this model×format. |

Trials missing for a model: vignettes the model failed to identify in the cue check are dropped
before elicitation, as preregistered (e.g. 549 trials for OLMo 7B Instruct, chat).

### 2. Topic, hypothesis and evidence table

One "base item" (a tier × topic × H1-direction × table) is reused across identities, beliefs and
sources, so these columns are shared by many rows.

| Column | Type | Values / range | Meaning and derivation |
|---|---|---|---|
| `item_id`, `base_slot` | str | id | The base item and its slot; identical across the identity/belief/source variants built on it. |
| `tier` | int | 1, 2 | 2 = socially meaningful policy topic (primary analysis); 1 = topic without social meaning (control). |
| `topic` | str | topic id | Topic identifier (the clustering unit for standard errors). |
| `h_dir` | int | +1, −1 | +1 if H1 says the policy group does better, −1 if worse. |
| `n_t`, `k_t` | int | counts | Treated (policy) group size and number of successes. |
| `n_c`, `k_c` | int | counts | Control group size and number of successes. |
| `q` | float | 0.58, 0.60, 0.62 | H1's success rate under the policy (and 1−q without); H0 is the reverse. |
| `D` | int | integer | `(2·k_t − n_t) − (2·k_c − n_c)`; the raw count-difference statistic of the table. |
| `table_x` | float | real | Log-likelihood ratio `log(q/(1−q)) · D`, for the hypothesis "treated is higher". |
| `x` | float | ≈ −2.9 … 2.9 | **Evidence strength for this trial's H1**: `table_x · h_dir`. The ideal Bayesian update of the log-odds of H1 equals `x`. |
| `d` | int | +1, −1, 0 | `sign(x)`: +1 favors H1, −1 favors H0, 0 = no evidence (null table). |
| `heuristic_conflict` | bool | True/False | True when the raw counts point the opposite way from the rates (a hard item for count-based heuristics). |
| `results_as_list` | bool | True/False | The table was shown as a list rather than a grid (presentation randomization). |
| `treated_first` | bool | True/False | The treated row/column was shown first (presentation randomization). |

### 3. User identity, belief and source

| Column | Type | Values / range | Meaning and derivation |
|---|---|---|---|
| `group` | str | 22 groups or `unspecified` | Identity group of the simulated user; `unspecified` = no persona (the control). |
| `dimension` | str | `political`, `race`, `sexuality`, `gender`, `religion`, `geography`, `age`, `none` | The identity dimension the group belongs to (`none` for `unspecified`). |
| `cue_id` | str | cue id | The exact identity cue shown. |
| `cue_type` | str | `label`, `vignette` | `label` = explicit statement of identity; `vignette` = indirect description. |
| `b` | int | +1, −1, 0 | Stated belief: +1 = user believes H1, −1 = user believes H0, 0 = no belief stated. |
| `belief_dir` | int | +1, −1, 0 | `b · h_dir`: which concrete claim (policy helps/hurts) the user asserts, independent of H1's labeling. |
| `e` | str | `neutral`, `support`, `contra`, `null` | Evidence relative to the stated belief. `neutral` = no belief stated; `support` = evidence agrees with the belief; `contra` = evidence contradicts it; `null` = no evidence (x = 0). **Excluded from the main analyses when `null`** (see the reading note). |
| `kappa` | int | 0, 1 | 1 if the evidence contradicts the belief (indicator for `e == "contra"`). |
| `source` | str | `lab`, `tank` | Who ran the study in the vignette: `lab` = a state-university lab, `tank` = a policy think tank. |
| `matched_pair` | bool | True/False | The topic engages the identity's own dimension (the Tier-2 pairing rule). |
| `tier1_low_g` | bool | True/False | **main only.** Tier 1 (topic, cue) pair with mean \|g\| < 0.25 in the pilot (a robustness subset; all Tier 2 rows are True). |
| `complement` | bool | True/False | **pilot only.** P(H0) was also asked for this trial. |
| `calibration` | bool | True/False | **open models.** The trial was also read out by 10 sampled answers (a readout check). |

### 4. Raw answers

`p_*` are probabilities for H1 in 0–1. `cov_*` is the probability mass the model placed on valid
integer answers 0–100 (open models; for GPT-4o it is 1 whenever the structured reply held a valid
integer, GPT's inclusion rule). `att_*` = 1 if the single retry was used (open models: the reply was
prefilled with an answer prefix; API: the model was re-prompted "Reply with the number only. Do not
explain.").

| Column | Type | Values / range | Meaning |
|---|---|---|---|
| `p_prior` | float | 0–1 | Answer **before** the evidence, first phrasing. |
| `p_prior_rep` | float | 0–1 or blank | Answer before the evidence, second (paraphrased) phrasing; blank where `retest` is False. |
| `p_post` | float | 0–1 | Answer **after** seeing the table (a separate, fresh conversation). |
| `cov_prior`, `cov_prior_rep`, `cov_post` | float | 0–1 | Coverage (valid-integer mass) for each of the three answers. |
| `att_prior`, `att_prior_rep`, `att_post` | int | 0, 1 | Whether the retry/re-prompt was used for each answer. |
| `retest` | bool | True/False | The paraphrased prior was asked. True for all models except Gemini (1,500 seeded trials) and Claude in `main_trials` (300), which were budget-limited; those trials are **not** excluded for the missing paraphrase. |
| `readout` | str | `dist`, `sample` | How the answer was read: `dist` = full next-token distribution (primary); `sample` = one sampled answer per question (Claude, Gemini in `main_trials`; Claude three in the pilot). |
| `p_prior_s`, `p_post_s` | float | 0–1 or blank | **calibration trials (open models).** The same before/after questions read out by sampling instead of the distribution. |
| `cov_prior_s`, `att_prior_s`, `cov_post_s`, `att_post_s` | float/int | | Coverage and retry flags for the sampled readout. |
| `p_prior_H0`, `p_post_H0` | float | 0–1 or blank | **pilot only.** Probability placed on H0, before / after. |
| `cov_prior_H0`, `att_prior_H0`, `cov_post_H0`, `att_post_H0` | float/int | | **pilot only.** Coverage and retry flags for the H0 questions. |
| `sum_before`, `sum_after` | float | ≈ 1 | **pilot only.** P(H1) + P(H0), before / after (a coherence check; should be near 1). |

### 5. Derived analysis columns

| Column | Type | Values / range | Meaning and derivation |
|---|---|---|---|
| `excluded` | bool | True/False | **True = unusable, left out of all analyses.** Set when coverage is below 0.9 (chat) or 0.5 (completion), or any of the three answers is missing. GPT-4o: a valid integer is required (coverage is then 1). |
| `excluded_cov08` | bool | True/False | Same rule with the stricter 0.8 completion-format cutoff (a robustness check). |
| `lp_before`, `lp_before_rep`, `lp_after` | float | real | Log-odds of `p_prior`, `p_prior_rep`, `p_post`, with probabilities clipped to [0.005, 0.995] before the logit. |
| `y` | float | real | `lp_after − lp_before`: the realized update in log-odds. |
| `omega` | float | real (≈ −29 … 29) | `y / x`: share of the Bayesian update actually made (1 = fully Bayesian, 0 = ignored the evidence). Undefined/extreme when \|x\| is tiny, so it is only used descriptively. |
| `bayes_post` | float | real | `lp_before + x`: the log-odds a Bayesian would hold after the table. |
| `contra`, `support` | int | 0, 1 | Indicators for `e == "contra"` / `e == "support"` (used as the interaction terms in Eq. 3). |
| `has_belief` | int | 0, 1 | 1 when a belief was stated (`b != 0`). |
| `g` | float | real | The model's **perceived stance** of the user's group on H1: logit P(a group member agrees with H1) − logit P(an unspecified person agrees), at the same cue type. 0 for `unspecified`. Estimated in the stance step from the model's own answers. |
| `m` | float | real | **Congruence** `b · g` (the primary measure). Positive = the stated belief matches the stance the model attributes to the group. |
| `g_svy`, `m_svy` | float | real | Stance from Pew survey gaps (logit group % − logit all-adults %); `m_svy = b · g_svy`; 0 where no Pew item matches. **main only.** |
| `g_beh`, `m_beh` | float | real | Behavioral stance: the group's no-belief **before** log-odds minus Unspecified's; `m_beh = b · g_beh`. |
| `has_v`, `v_k` | int/float | | `v_k` = Pew gap for the trial's dimension (in units of 100 points); `has_v` = whether such a Pew item exists. |

---

## `evidence_diagnostic.csv.gz`

Exploratory addition 1: the same evidence tables presented with **no persona and no belief**, to
test whether the models can read and reason about the tables at all. One row per table × question
kind × model × format (480 tables drawn from the main trials).

| Column | Type | Values / range | Meaning |
|---|---|---|---|
| `diag_id` | str | table id | The table (480 tables). |
| `kind` | str | `rate_t`, `rate_c`, `compare`, `support`, `post` | The question asked: `rate_t`/`rate_c` = read one group's success percentage; `compare` = which group had the higher rate; `support` = which hypothesis the results support more; `post` = the posterior probability for H1. |
| `truth` | float/int | percent or option index | The correct answer: the true rate in percent (for `rate_*`), or the index of the correct option (0 = A, 1 = B) for the two-option questions. |
| `p` | float | 0–1 | The model's answer as a probability (`rate_*`, `post`). |
| `p_option_A`, `p_option_B` | float | 0–1 or blank | Probability the model placed on each option (`compare`, `support`). Option order was counterbalanced (see `flip`). |
| `flip` | bool | True/False | The two options were presented in reversed order (counterbalancing). |
| `coverage` | float | 0–1 | Probability mass on valid answers. |
| `model`, `fmt`, `topic`, `tier`, `x` | | as above | Same meaning as in the trial datasets. |
