# I Think, Therefore It Is: How User Beliefs Trump Credible Evidence Uptake in LLMs

**Repository:** https://github.com/Vish-Karthikeyan/I-Think-Therefore-It-Is

Code, data and steering vectors for the paper above (Vish Karthikeyan, Stanford University). A simulated user, sometimes with a stated identity and a stated belief, asks
a language model how likely a policy hypothesis is, before and after seeing a small study's results. The study measures
whether a stated belief changes how much the model's answer moves with the evidence, and whether identity amplifies it
(RQ1); whether identity information predicts the size of that change on unseen topics and groups (RQ2); and whether the
effect can be located in, and removed from, the internals of open-weight models (RQ3).

## Repository layout

| Path | Contents |
|---|---|
| `datasets/main_trials.csv.gz` | Trial-level dataset: 525,042 rows (one per trial × model × prompt format), 84 columns |
| `datasets/CODEBOOK.md` | Every column, how it was derived, and how to read the file |
| `analysis/` | `rq1.py`, `rq2.py`, `rq3.py` (all analyses for each research question, with on/off toggles) and `run_all.py` |
| `elicit.py` | Builds a new dataset in the same format for any OpenRouter or Hugging Face model |
| `src/idea/` | The package behind `elicit.py` and `rq3.py`: trial design, prompts, readouts, stance, steering |
| `data/` | Inputs to elicitation: topics, identity cues, Pew survey items, and the main design's pilot-derived inputs (`data/design/`) |
| `steering_vectors/` | RQ3 directions and belief subspaces for OLMo 2 7B Instruct, Llama 3.1 8B Instruct and Tülu 3 8B |

## Option 1: use our dataset

This reproduces the paper's numbers exactly.

```bash
git clone https://github.com/Vish-Karthikeyan/I-Think-Therefore-It-Is.git && cd I-Think-Therefore-It-Is
python -m venv .venv && source .venv/bin/activate
pip install -e .        # numpy, pandas, scipy, statsmodels, torch (CPU is enough)
python analysis/run_all.py
```

`rq2.py` fits its predictor with Adam in PyTorch, which is part of what makes the numbers reproduce exactly, so
torch is a core dependency; it runs on the CPU, no GPU needed.

`run_all.py` runs `rq1.py` and `rq2.py` on `datasets/main_trials.csv.gz` and prints labelled result tables
(`--save DIR` also writes them as CSV), then runs `rq3.py` if a CUDA GPU is available. Each script can also be run on
its own, and every analysis in it can be switched on or off:

```bash
python analysis/run_all.py --list                 # every analysis, with a one-line description
python analysis/rq1.py --list
python analysis/rq1.py --only <name>,<name>       # just these analyses (names from --list)
python analysis/rq1.py --skip <name>              # everything except these
python analysis/run_all.py --only rq1.<name>,rq2  # the same toggles across scripts (script or script.analysis)
python analysis/run_all.py --skip rq3
```

The toggles are also a block of `True`/`False` flags (`ANALYSES = {...}`) at the top of each script. `--models` limits
the analysis to some models; `--data` points at another dataset in the same format.

## Option 2: construct your own dataset

`elicit.py` runs the whole elicitation for one model and writes a CSV with the same columns as
`datasets/main_trials.csv.gz`, so the analysis scripts run on it unchanged. It regenerates the study's 25,254 trials
with the same seeded design (same trial ids, tables, cues and evidence strengths), then runs the vignette check, the
perceived-stance questions, and the three answers per trial (prior, paraphrased prior, posterior), and derives the
exclusions and analysis columns. You bring the model access: your own OpenRouter API key for API models, your own GPU
for open-weight models.

```bash
# API models (OpenRouter). Always start with a dry run: call counts, cost estimate and example prompts, nothing sent.
export OPENROUTER_API_KEY=...
python elicit.py --backend openrouter --model openai/gpt-4o-2024-11-20 --dry-run
python elicit.py --backend openrouter --model openai/gpt-4o-2024-11-20 --n-trials 3000 --out my_trials.csv.gz

# Open-weight models (Hugging Face, local GPU): the full next-token distribution over 0-100
python elicit.py --backend hf --model allenai/OLMo-2-1124-7B-Instruct --revision <commit> --out my_olmo.csv.gz

# Then analyse it
python analysis/run_all.py --data my_trials.csv.gz
```

- **Cost control.** The cost estimate is printed before anything is sent, and the run refuses to start if it is above
  `--max-cost` (default $15), asks for confirmation, and stops if actual spend reaches the cap. Narrow a run with
  `--n-trials`, `--tiers 2`, `--cue-types label` or `--subset ids.txt`.
- **Readout.** API answers are a structured integer reply. `--readout dist` (default) reads the token probabilities and
  needs a provider that returns logprobs; for models without them use `--readout sample` (`--sample-k` answers per
  question).
- **Our models.** Passing one of our model keys (`gpt`, `claude`, `gemini`, `olmo_instruct`, `llama`, `olmo_sft`, …;
  see `src/idea/models.py`) uses that model's pinned revision, readout and part of the design from the main run.
- **Resuming.** Every answer is cached in `elicit_cache/`; rerunning the same command resumes where it stopped.
- **Testing.** `--backend mock` runs the whole pipeline on a simulated model in seconds, at no cost.

**A re-run produces a comparable dataset, not an identical one.** Elicitation is stochastic: sampled answers differ from
run to run; API providers update and re-deploy models behind the same identifier, so answers to unchanged prompts drift
over time; and batched GPU inference is not bit-for-bit deterministic (batch composition and kernels change the
probabilities slightly). The design, prompts and analysis are fixed; the answers are a new sample. To reproduce the
paper's numbers exactly, use Option 1.

## RQ3

**RQ3 needs model weights and a GPU.** `python analysis/rq3.py` re-runs RQ3 on OLMo 2 7B Instruct, Llama 3.1 8B
Instruct and Tülu 3 8B (Hugging Face, pinned revisions; Llama is gated, so set `HF_TOKEN`). Use one 48 GB GPU per
7–8B model in bf16 (the study used an NVIDIA L40S) and at least 32 GB of host RAM. Source your own GPU; Stanford
affiliates can use FarmShare. Without a CUDA GPU, `run_all.py` skips RQ3 and prints how to run it.

Approximate GPU-hours per model: corrected analysis ~3.5; capability audit (3 conditions) ~2–6; geometry and
dose–response ~0.1; stability a few CPU minutes plus ~0.15; preregistered steering ~13 and its audit ~35; training-stage
probes ~0.1 per stage. The default runs the corrected analysis and its audit; `--list`, `--only` and `--skip` work as in
the other scripts. Runs resume after interruption. `--use-released-vectors` steers with `steering_vectors/` instead of
re-estimating; `--summarize-only` rebuilds the tables from existing outputs without a GPU; `--smoke --hf-model
olmo_instruct=allenai/OLMo-2-0425-1B-Instruct` tests every code path on a CPU (the numbers are meaningless). The audit
needs `lm-eval` 0.4.x and `datasets`, and runs L2D only if you place the authors' `exp1_prompts.json` in
`data/l2d_external/`. Hidden states and answers depend slightly on GPU, dtype and batch size, so RQ3 reproduces the
study's results closely but not bit for bit.

The directions and subspaces themselves are in `steering_vectors/`; see
[`steering_vectors/README.md`](steering_vectors/README.md) for what each file is, how to apply it, and the dual-use
warning (applied in reverse they strengthen belief-following).

**Synthetic data.** Every persona, study and results table in the prompts is invented. The perceived-stance values
describe what each model assumes about a group, never what the group believes.

## License

The code, datasets and steering vectors in this repository are released under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md), except where a third-party license below applies.

## Third-party models and data

- **Llama 3.1** (Llama 3.1 8B Instruct and base): Llama 3.1 is licensed under the Llama 3.1 Community License,
  Copyright © Meta Platforms, Inc. All Rights Reserved. Use is subject to that license and Meta's Acceptable Use
  Policy. A copy of the license is in `third_party/LLAMA_3.1_LICENSE` and the attribution notice in `NOTICE`.
- **Tülu 3** (8B SFT, DPO and final; Ai2, built on Llama 3.1 8B): Llama 3.1 Community License and Ai2's terms for
  Tülu 3.
- **OLMo 2** (7B and 13B: base, SFT, DPO, Instruct; Ai2): Apache License 2.0.
- **Phi-4** (Microsoft): MIT License.
- **GPT-4o, Claude Sonnet 5, Gemini 2.5 Flash** (through OpenRouter): only the numeric answers are released; use of
  these services is subject to their providers' terms.
- **Pew Research Center** American Trends Panel: only published aggregate percentages are used for the survey-stance
  measure; cite Pew as in the paper.
- **L2D** (Ashkinaze et al.): the benchmark items are **not** included. Obtain them from the authors and place
  `exp1_prompts.json` in `data/l2d_external/` to run that part of the RQ3 audit.

Exact model revisions (Hugging Face commits) and API model identifiers are pinned in `src/idea/models.py`.

## Citation

> TODO: arXiv citation once posted.
