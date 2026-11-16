# IDEA: Doctoring the Who

Code, data and steering vectors for *Doctoring the Who: Controlling Socially Conditioned Evidence Weighting in LLMs*
(Vish Karthikeyan, Stanford University). The secondary paper (full methods, every protocol change, all results) is in
`paper/secondary/`.

**Built with Llama.**

## What is here

| Path | Contents |
|---|---|
| `datasets/` | Trial-level data as gzip CSV: `main_trials.csv.gz` (525,042 rows), `pilot_trials.csv.gz`, `evidence_diagnostic.csv.gz`; columns in `datasets/CODEBOOK.md` |
| `analysis/` | Commented scripts that rebuild every table from the datasets (`00`–`11`, `run_all.sh`); see `analysis/README.md` |
| `steering_vectors/` | RQ3 directions and belief subspaces (OLMo 2 7B Instruct, Llama 3.1 8B Instruct, Tülu 3 8B) |
| `src/idea/` | The experiment package: trial design, elicitation, stance, regressions, probes, steering, audit |
| `configs/` | Pilot, main and exploratory configurations, with seeds and pinned model revisions |
| `scripts/farmshare/` | Slurm job scripts used on Stanford FarmShare |
| `results/compute/` | FarmShare job log and API spend |

**Synthetic data.** Every persona, study and results table in the prompts is invented. The perceived-stance values
describe what each model assumes about a group, never what the group believes.

**Steering vectors are released openly.** Applied in reverse they strengthen belief-following (see
`steering_vectors/` and the paper's ethics statement).

## Reproduce

```bash
uv venv .venv --python 3.11 && uv pip install -p .venv -e ".[dev]"
bash analysis/run_all.sh          # all tables from datasets/ (no GPU, no API keys)
python -m pytest -q tests         # checks that the standalone scripts reproduce the package's numbers
```

Re-running the models themselves needs GPUs (open weights) and OpenRouter keys (API models); see `scripts/` and the
configs.

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
  `exp1_prompts.json` in `data/l2d_external/` to run that part of the audit.

Exact model revisions (Hugging Face commits) and API model identifiers are pinned in `src/idea/models.py`.

## Citation

> TODO: arXiv citation once posted.

## Compute

Open-weight models ran on Farmshare (RRID:SCR_027070), Stanford University's shared computing cluster.
