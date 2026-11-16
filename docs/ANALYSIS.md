# IDEA analysis scripts — full documentation

This is the complete reference for the scripts in `analysis/`. They reproduce every table, number
and candidate figure in the secondary paper from the released datasets. `analysis/README.md` is the
short index and points here.

Run from the **repository root**, in order, or all at once with `bash analysis/run_all.sh`. Every
script writes CSV tables to `analysis/output/<topic>/` and prints the key ones.

## How the scripts fit together

```
datasets/*.csv.gz  ──►  01 descriptives/controls      ─┐
                        02 RQ1 primary                  │
                        03 RQ1 secondary                ├─► analysis/output/*  ─►  09 paper_tables ─► paper/secondary/tables/*.tex
                        04 robustness                   │                          12 candidate_figures ─► paper/secondary/figures/candidates/
                        05 RQ2 + secondary specs        │
                        07 exploratory P0–P6            │
results/rq3/*      ──►  06 RQ3 steering + audit         │
results/rq3_v2/*   ──►  11 RQ3 corrected                │
results/p4/*       ──►  (07 reads it) / 10 P4 probes    │
results/compute/*  ──►  08 compute accounting          ─┘
runs/* (+ idea pkg)──►  00 build_datasets (rebuild only)
```

- **00** is the only script that needs the `idea` package and the raw `runs/` folders; it rebuilds
  `datasets/*.csv.gz`. Everything else reads the released CSVs (01–04, 07) or released results dirs
  (06, 11) and, for the long model code, the package (05, 06, 11).
- **Dependencies.** Scripts 01–04 and 07 use only pandas, numpy, scipy and statsmodels. 05, 06, 10,
  11 call the `idea` package (`pip install -e .` from the root). 12 also needs matplotlib.
- **`--study main|exploratory|pilot`** selects the data for 01, 02, 03 and 05 (default `main`).
  `pilot` reads `pilot_trials.csv.gz`; `main` and `exploratory` are two slices of
  `main_trials.csv.gz`.
- **Shared code** is in `analysis/common.py` (described below). Scripts import each other by number
  via `__import__("02_rq1_primary")` etc.

---

## `common.py` — shared definitions

Every analysis script reads the released datasets through `common.py`, which also defines the
analysis sample and the primary regression, so the definitions live in one place. The same
definitions exist in the `idea` package (`src/idea/analysis/rq1.py`), and
`tests/test_release_analysis.py` checks the two give identical numbers (to ~1e-15).

| Object | Purpose |
|---|---|
| `load_dataset(name="main")` | Read `datasets/<name>_trials.csv.gz` with `keep_default_na=False, na_values=[""]` (preserves the `e == "null"` label) and cast the boolean columns. Asserts the null label survived. |
| `analysis_frame(df, tier=2, model=None, fmt=None, cue_type=None, m_col="m", excluded_col="excluded")` | The **analysis sample**: drop excluded trials, drop null-evidence tables (`x = 0`), keep one tier (2 = primary, `None` = both), optionally one model / format / cue type; `m_col` swaps the congruence measure (`m`, `m_svy`, `m_beh`); `excluded_col` swaps the exclusion rule. Sets `Unspecified` as the reference group and adds `dpos`. |
| `BASE_TERMS` | The right-hand side of Eq. 3 (below). |
| `fit_ols(d, terms=BASE_TERMS)` | OLS with standard errors **clustered by topic**. `formula()` first drops control terms that are constant in the sample (e.g. `x:C(cue_type)` for labels-only models). |
| `estimate(fit, name)` | `est`, `se`, `p` and 95% CI of one coefficient (NaN if absent). |
| `holm(pvals, alpha=0.05)` | Holm step-down correction over a dict of p-values. |
| `DIMENSIONS` | The 7 identity dimensions and their 22 groups (identical to `idea.cues.DIMENSIONS`). |
| `OLMO7_STAGES`, `OLMO13_STAGES`, `TULU_STAGES` | Training-ladder orderings. |
| `model_formats(df)`, `save(tab, sub, name, show=True)`, `out_dir(sub)` | (model, fmt) pairs; write/print a table to `analysis/output/<sub>/<name>.csv`. |

### The primary regression (Eq. 3)

```
lp_after ~ lp_before + x + x:contra + x:support + x:contra:m + x:support:m
           + x:dpos + x:C(cue_type) + x:C(source)
```

with SEs clustered by `topic`. The two **co-primary estimands** (Holm-corrected together):

- **Δ_b** = coefficient on `x:contra` — how much flatter the evidence slope is when the evidence
  *contradicts* the stated belief (negative = contradicting evidence is discounted → belief-following).
- **θ_cong** = coefficient on `x:contra:m` — how that discount changes with identity–belief
  congruence `m = b·g` (negative = larger discount when the belief fits the user's identity).

Also reported: **β** (`x`, the no-belief evidence slope; 1 = Bayesian), **ρ** (`lp_before`), and the
support shift (`x:support`).

---

## Script-by-script

### `00_build_datasets.py` — build the datasets *(rebuild only)*

```
python analysis/00_build_datasets.py
```

- **Inputs:** `idea` package, raw run folders `runs/pilot-296551ab`, `runs/main-8b15a8c4`,
  `runs/ext-8b15a8c4`. Calls `idea.pipeline.load_responses` (the exact loader the preregistered
  analysis used) so the derived columns match.
- **Outputs:** `datasets/main_trials.csv.gz`, `datasets/pilot_trials.csv.gz`,
  `datasets/evidence_diagnostic.csv.gz`.
- **What it does:** adds the model/design descriptor columns, the `excluded_cov08` robustness flag,
  the `design` and `retest` columns; checks that the capture-only re-runs of OLMo SFT/DPO reproduce
  the main run's answers (`r > 0.999`, mean |Δ| < 0.01) before dropping them; labels the API models
  (Gemini, Claude) `study = "exploratory"`. Columns are documented in `datasets/CODEBOOK.md`.

### `01_descriptives_controls.py` — sample and instrument controls

```
python analysis/01_descriptives_controls.py [--study main|exploratory|pilot]
```

- **Inputs:** datasets. **Outputs:** `analysis/output/controls/<name>_<study>.csv`.
- **Tables:** `sample` (trials / usable / exclusion rate per model×fmt×tier×cue type);
  `coverage` (mean valid-answer mass before/after, re-prompt rate); `retest` (correlation of the two
  paraphrased priors `lp_before` vs `lp_before_rep`, per distinct prior prompt); `null_evidence`
  (D = 0 tables: mean |y| noise floor and drift toward the belief); `complement_sums` (pilot:
  P(H1)+P(H0)); `prior_distribution`; `readout_calibration` (distribution vs 10-sample readout on
  open models: correlations and slopes); `evidence_slopes` (raw unadjusted slope of `y` on `x` per
  evidence type, Tier 2).

### `02_rq1_primary.py` — RQ1 primary *(preregistered)*

```
python analysis/02_rq1_primary.py [--study main|exploratory|pilot] [--engine ols|mixed]
```

- **Inputs:** datasets. **Outputs:** `analysis/output/rq1/`.
- **Methods:** fits Eq. 3 per model×format on Tier 2 (`primary()`), reports Δ_b, θ_cong (Holm-
  corrected together), β, ρ, support shift. `--engine mixed` uses random item intercepts + random
  topic slopes on `x` (`fit_mixed`, slow). Models with < 50 usable trials are reported as missing.
- **Tables:** `per_model_<tag>.csv` (one row per model×format); `joint_<tag>_<fmt>.csv` (Wald test
  that Δ_b and θ_cong differ by model — reported "not estimable" when the number of restrictions
  exceeds topic clusters − 1, as for the exploratory set); `stage_contrasts_<tag>_<fmt>.csv` (OLMo 7B
  base→SFT→DPO→Instruct, step changes).

### `03_rq1_secondary.py` — RQ1 secondary *(preregistered as secondary)*

```
python analysis/03_rq1_secondary.py [--study main|exploratory|pilot]
```

- **Inputs:** datasets (model×format with ≥ 200 usable Tier-2 trials). **Outputs:**
  `analysis/output/rq1_secondary/`.
- **Methods / tables:**
  - `delta_j` — unequal deference (Eq. 6): per group, how much more its contradicted-evidence slope
    drops than the Unspecified control's (`x:contra:C(group)[T.j]` in one regression).
  - `delta_j_equality` — Wald test within each dimension that all its groups' `delta_j` are equal.
  - `theta_by_dimension` — θ_cong estimated per identity dimension.
  - `cue_type` — Δ_b and θ_cong for labels vs vignettes, and their difference.
  - `tier` — Δ_b on Tier 1, and the Tier 2 − Tier 1 difference ("social amplification").
  - `delta_j_tier1` — unequal deference on Tier 1 topics.
  - `simple_difference` — the `y = lp_after − lp_before` regression (ρ forced to 1) as a check.
  - `prior_shift` — γ: how far a (group, cue type, belief) cell's before log-odds sits from
    Unspecified-with-no-belief.

### `04_robustness.py` — RQ1 robustness *(preregistered; main run)*

```
python analysis/04_robustness.py [--mixed]
```

- **Inputs:** `main_trials.csv.gz` (`study == main`). **Outputs:** `analysis/output/robustness/`.
- **Checks** (each refits the primary model with one change): `congruence_svy` (Pew-survey stance
  `m_svy`), `congruence_beh` (behavioral stance `m_beh`), `svy_weak_dropped` (drop the weakest-proxy
  Tier-2 topics), `completion_cov08` (stricter 0.8 completion cutoff), `tier1_low_g` (Tier 1 limited
  to low-|g| pairs), and optional `mixed`. `robustness_summary.csv` places every check's Δ_b and
  θ_cong next to the primary estimate.

### `05_rq2_and_secondary_specs.py` — RQ2, RQ2b, secondary specifications

```
python analysis/05_rq2_and_secondary_specs.py [--study main] [--models gpt llama ...] \
       [--max-folds N] [--n-boot 1000] [--skip-secondary]
```

- **Inputs:** datasets **and** the `idea` package (RQ2 predictor in PyTorch, RQ2b bootstrap). Feeds
  the released dataset to `idea.analysis.{rq2, rq2b, secondary}`. **Outputs:** `analysis/output/rq2/`,
  `analysis/output/rq2b/`, `analysis/output/secondary/`.
- **RQ2** (chat format only): can identity information predict a group's evidence-slope shift on
  **held-out** topics / dimensions? A rank-2 interaction model ("full") vs no-W vs cell means, on two
  splits (4 held-out-topic folds; 7 held-out-dimension folds). Preregistered success = full model
  beats both baselines on RMSE **and** Spearman ρ.
- **RQ2b** (exploratory, gated): runs only for models whose RQ2 succeeded on the held-out-topic split.
- **Secondary specs** (per model×format): `spline` (F test of nonlinearity in `x`), `mixture`
  (two-component EM "updated vs ignored"; fitted from `N_STARTS = 10` seeds, best kept, stability
  across near-equal starts reported), `latent_prior` (IV version of Eq. 3 using the paraphrased
  prior as the instrument).

### `06_rq3_steering_audit.py` — RQ3 steering + capability audit *(original, prompt position)*

```
python analysis/06_rq3_steering_audit.py
```

- **Inputs:** `results/rq3/<model>_chat/` (produced on FarmShare by `idea steer`/`idea audit`) **and**
  the `idea` package (`idea.audit.merge_parts`). Models: `olmo_instruct`, `llama`. **Outputs:**
  `analysis/output/rq3/`.
- **What it reads / builds:** `steering_test.csv` (+ `gap_closed_frac`), `chosen_settings.csv`
  (from `summary.json`), `sweep_best.csv` (best setting per validation layer), `masking.csv`
  (overall + by dimension), `bias_angles.csv`, `audit_summary.csv`, and `audit_tests.csv`
  (two-proportion z-tests of steered/random vs unsteered on MMLU-Pro — restricted to subjects both
  conditions finished — BBQ and SycEval). The steering vectors themselves are in `steering_vectors/`;
  the methods are documented in `docs/MECH_HANDBOOK.md`.
- **Caveats printed:** SycEval reads one letter right after the user names one, so flip rates sit
  near 1 and it cannot show a steering effect; BBQ at the 500-item cap covers the age category only.

### `07_exploratory.py` — exploratory predictions P0–P6

```
python analysis/07_exploratory.py
```

- **Inputs:** `main_trials.csv.gz`, `evidence_diagnostic.csv.gz`, and (for P4)
  `results/p4/p4_stage_probes.csv` if present. **Outputs:** `analysis/output/exploratory/` plus
  `predictions.csv` (one verdict line per prediction).
- **Predictions** (each checked with the primary regression; variant-vs-original comparisons use one
  stacked regression with every term interacted with a variant dummy and topic-clustered SEs):
  - **P0** — belief-following (Δ_b < 0) grows OLMo SFT→DPO→Instruct in chat, weak/absent in completion.
  - **P1** — evidence diagnostic: models read rates and compare groups above chance, but are near
    chance on which hypothesis the data support and on posterior direction.
  - **P2** — scale: β > 0.1 for at least one 13–14B instruct model (OLMo 2 13B Instruct, Phi-4).
  - **P3** — the debiasing system prompt shrinks but does not remove belief-following; β stays near 0.
  - **P4** — the belief signal is weak at SFT and appears at DPO (needs saved activations; read from
    `results/p4/` here, computed by script 10).
  - **P5** — the Tülu 3 ladder shows the same DPO jump as OLMo.
  - **P6** — two reworded prompt templates keep the sign of Δ_b and the support shift, magnitudes
    within the original's 95% CI.

### `08_compute_accounting.py` — FarmShare compute accounting

```
# on FarmShare, refresh the raw records first:
sacct -u <user> -S 2026-10-01 -E now -D -X -P \
      -o JobID,JobName%80,Partition,QOS,State,Submit,Start,End,ElapsedRaw,AllocTRES,ReqMem,ExitCode,NodeList \
      > results/compute/sacct_raw.txt
python analysis/08_compute_accounting.py
```

- **Input:** `results/compute/sacct_raw.txt` (Slurm accounting, pipe-separated). **Outputs:**
  `results/compute/farmshare_runs.csv` (one row per run), `farmshare_jobs.csv` (one row per job id:
  purpose, runs, wall time, GPU-hours), `farmshare_summary.csv` (GPU-hours by phase and task).
- **Methods:** `classify()` maps each job name (from `scripts/farmshare/*.sh`) to (phase, task,
  model); GPU-hours = allocated GPUs × wall time (all GPUs are NVIDIA L40S 48 GB). Requeued runs are
  summed per job id. Disclosed waste (the Llama steering restarts before the unseeded-PCA fix on
  2026-11-14) is flagged. API models (GPT-4o, Claude) and the one Kaggle pilot backup are not
  FarmShare and are reported separately.

### `09_paper_tables.py` — LaTeX tables for the paper

```
python analysis/09_paper_tables.py
```

- **Inputs:** the CSVs in `analysis/output/` (so run 01–08, 11 first) and `data/` (topic and cue
  lists). **Outputs:** `paper/secondary/tables/*.tex` — the paper `\input{}`s them. *(This script
  writes under `paper/`, which is otherwise off-limits; it is the generator that produces the paper's
  tables and is listed here for completeness.)*
- **Tables written:** `rq1_<study>.tex`, `joint.tex`, `robustness.tex`, `tier.tex`, `dj_equality.tex`,
  `dj_all.tex`, `cue_type.tex`, `theta_dim.tex`, `controls.tex`, `calibration.tex`, `rq2.tex`,
  `secondary_specs.tex`, `steering.tex`, `rq3_settings.tex`, `audit.tex`, `rq3v2_test.tex`,
  `rq3v2_sufficiency.tex`, `rq3v2_audit.tex`, `predictions.tex`, `diagnostic.tex`, `stage_steps.tex`,
  `p4.tex`, `p4v2.tex`, `compute.tex`, `topics.tex`, `cues.tex`. Rerun after any analysis rerun and
  recompile the paper.

### `10_p4_stage_probes.py` — P4 stage probes on saved activations

```
python analysis/10_p4_stage_probes.py [--layers 8 16 24] [--max-n 4000]
```

- **Inputs:** the `idea` package and the **saved activations** (FarmShare; CPU only) for OLMo 2 7B
  SFT, DPO (exploratory run) and Instruct (main run), chat format, explicit labels. **Not in the
  release** — this script runs where the activations are. **Output:**
  `results/p4/p4_stage_probes.csv` (one row per stage×layer), read back by script 07.
- **Methods:** two measures per stage and layer — (1) a logistic probe (`C=0.05`) that reads
  contradicted-vs-supported from the activations, trained on 70% of topics and tested on the held-out
  30% (same split for every stage), reported as held-out accuracy/AUC vs majority; (2) cosine of the
  stage's own `d_bias` direction (`hat_k8` and `raw`) to OLMo Instruct's saved RQ3 direction at the
  same layer (random baseline |cos| ≈ 0.016). Caveat: these activations were recorded **before** the
  answer prefix, like the original RQ3 — see script 11 for the corrected version.

### `11_rq3_corrected.py` — corrected RQ3 *(answer position; exploratory)*

```
python analysis/11_rq3_corrected.py
```

- **Inputs:** `results/rq3_v2/<model>_chat/` (from `idea steer2`/`idea audit2` on FarmShare) and the
  `idea` package (`idea.mech.rq3v2.partial_success`). Models: `olmo_instruct`, `llama`, `tulu_final`.
  **Outputs:** `analysis/output/rq3_v2/`.
- **Tables:** `test.csv` (held-out test of every chosen intervention and its random control, with the
  three success tiers — preregistered, exploratory, partial — the share of the gap closed, and an
  overshoot flag); `sufficiency.csv` (interchange recovery by layer and patch type, averaged over the
  two swap directions); `geometry.csv` (norms and cosine of the two belief directions by layer);
  `audit.csv` (MMLU-Pro, BBQ, SycEval, L2D per model×condition, with a `status` column so partly
  finished audits are shown as such). The intervention families, subspaces and the erasure mechanism
  are documented in `docs/MECH_HANDBOOK.md`.

### `12_candidate_figures.py` — candidate paper figures

```
python analysis/12_candidate_figures.py
```

- **Inputs:** the CSVs in `analysis/output/` (and the dataset for F5), matplotlib. **Outputs:**
  `paper/secondary/figures/candidates/*.{pdf,png}` *(writes under `paper/`; it is the figure
  generator)*. Each figure is skipped with a note if its input CSV is missing.
- **Figures:** F1 Δ_b by model (main, chat); F2 no-belief slope β by model; F3 θ_cong by model;
  F4 training-stage ladders (OLMo 7B and Tülu); F5 evidence slope split by heuristic conflict;
  F6 corrected-RQ3 share of gap closed per intervention/model; F7 sufficiency recovery vs layer;
  F8 belief-direction geometry vs layer; F9 capability audit per model/metric; F10 Tier-1 Δ_b.
  Colorblind-safe Okabe–Ito palette, one y-axis per panel.

---

## `run_all.sh` — entry point

```bash
bash analysis/run_all.sh            # all analyses (datasets already built)
REBUILD=1 bash analysis/run_all.sh  # rebuild datasets/ from runs/ first (needs the idea package)
```

Uses `../.venv/bin/python` if present (override with `PYTHON=`). Runs 01–03 for `main`, `exploratory`
and `pilot`, then 04, 05 (`--study main`), 06, 11, 07, 08 and 09. Script 10 (needs saved activations)
and 12 (figures) are run separately. Before running 08, refresh `results/compute/sacct_raw.txt` on
FarmShare (see the script's header).

## Tests — `tests/`

Run `python -m pytest -q tests` from the root. The key check is
`tests/test_release_analysis.py`, which verifies that the standalone analysis scripts reproduce the
`idea` package's numbers (agreeing to ~1e-15) and that reading the datasets keeps the null-evidence
label. Other suites cover the trial/prompt construction, answer-format handling, the coverage and
Tier-1 rules, API re-prompting, the spending cap, audit-part merging, RQ2 and the LaTeX-table
generator (`test_tables.py`).

## Notes for readers

- **Sample.** Every regression uses usable trials (`excluded == False`) with evidence (`e != "null"`);
  the primary analysis is Tier 2. Standard errors are clustered by topic.
- **Exclusions differ by model.** Llama 3.1 8B loses ~21% of trials (low integer coverage); several
  exploratory runs lose far more (OLMo 13B completion 75–100%, Llama with the system prompt ~56%), so
  those estimates rest on fewer trials or are not estimable (< 50 usable trials).
- **P4** needs saved activations (not in the release) and is reported separately.
