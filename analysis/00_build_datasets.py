"""Step 0: build the two released trial-level datasets from the raw run folders.

    python analysis/00_build_datasets.py

Output (one row per trial x model x prompt format; columns documented in datasets/CODEBOOK.md):
    datasets/pilot_trials.csv.gz   pilot run (pilot-296551ab): 7 models incl. Claude
    datasets/main_trials.csv.gz    main run (main-8b15a8c4, study = "main") plus the
                                   exploratory additions (ext-8b15a8c4, study = "exploratory"),
                                   which used the same 25,254 trials; Gemini 2.5 Flash and
                                   Claude Sonnet 5 (added 2026-11-15, main run folder) are
                                   also labeled "exploratory"

This is the only script that needs the `idea` package and the raw run folders (runs/...):
it calls `idea.pipeline.load_responses`, the exact loader the preregistered analysis used,
so the derived columns (log-odds, exclusions, stance g and congruence m) are the same.
Every later script reads only the CSVs. The CSVs are gzip-compressed because the main file
is larger than GitHub's 100 MB limit uncompressed; pandas, R and most tools read .csv.gz
directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from idea.models import MODELS  # noqa: E402
from idea.pipeline import load_responses  # noqa: E402

RUNS = ROOT / "runs"
DATA = ROOT / "datasets"

# Raw run folders behind each dataset.
PILOT_RUN = "pilot-296551ab"
MAIN_RUN = "main-8b15a8c4"
EXT_RUN = "ext-8b15a8c4"

# In the exploratory folder, olmo_sft and olmo_dpo were re-run only to save their activations
# (exploratory addition 4). Their answers repeat the main run's, so they are checked against
# the main run and then left out instead of being listed twice.
EXT_CAPTURE_ONLY = ["olmo_sft", "olmo_dpo"]

# API models added on 2026-11-15, after the preregistered analysis (budget-limited, second
# OpenRouter key). Their answers sit in the main run folder (same trials, same stance step), but
# they are not preregistered models, so they are labeled study = "exploratory".
#   gemini: Gemini 2.5 Flash, full design (25,254 trials), 1 sampled JSON answer per question.
#   claude: Claude Sonnet 5, a seeded 6,000-trial subset of the labels-only trials
#           (5,000 Tier 2 + 1,000 Tier 1), 1 sampled answer per question.
# Both asked the paraphrased prior on a seeded subset only (`retest` = True; Gemini 1,500,
# Claude 300), to fit the budget.
MAIN_RUN_API_ADDED = ["gemini", "claude"]

# Internal cache keys (hashes of the prompts) mean nothing outside the pipeline.
DROP_COLS = ["key_prior", "key_post"]

FAMILY = {"olmo": "OLMo 2", "olmo13": "OLMo 2", "tulu": "Tulu 3 (Llama 3.1)", "llama": "Llama 3.1",
          "phi4": "Phi-4", "gpt": "GPT-4o", "claude": "Claude Sonnet 5", "gemini": "Gemini 2.5 Flash"}
SIZE_B = {"olmo": 7, "olmo13": 13, "tulu": 8, "llama": 8, "phi4": 14}


def model_info(key: str) -> dict:
    """Descriptive columns for one model key (from the pinned specs in src/idea/models.py)."""
    s = MODELS[key]
    root = key.split("_")[0]
    variant = key[len(root) + 1:] if key.startswith(root + "_") else ""
    base_key = key
    for suf in ("_sp", "_pv1", "_pv2"):
        if key.endswith(suf):
            base_key = key[: -len(suf)]
    stage = MODELS[base_key].stage or ("instruct" if s.open_weights and root in ("llama", "phi4") else None)
    return dict(
        base_model=base_key,
        model_family=FAMILY.get(root, root),
        model_size_b=SIZE_B.get(root, np.nan),                  # NaN for API models (undisclosed)
        training_stage=stage or ("api_chat" if not s.open_weights else ""),
        hf_id=s.hf_id or "", hf_revision=s.revision if s.hf_id else "", api_id=s.openrouter_id or "",
        open_weights=bool(s.open_weights),
        prompt_variant=int(s.prompt_variant),                   # 0 = original wording; 1, 2 = rewordings
        debias_system_prompt=bool(s.system_prompt),             # exploratory prompting baseline
        variant_tag=variant if variant in ("sp", "pv1", "pv2") else "",
    )


def add_columns(df: pd.DataFrame, study: str) -> pd.DataFrame:
    df = df.drop(columns=[c for c in DROP_COLS if c in df]).copy()
    df.insert(0, "study", study)
    info = pd.DataFrame([dict(model=m, **model_info(m)) for m in df.model.unique()])
    df = df.merge(info, on="model", how="left")
    # Robustness flag: exclusion under the stricter 0.8 completion-format coverage cutoff (the
    # primary rule uses 0.5 for completion format, 0.9 for chat; see CODEBOOK.md).
    cov = df[[c for c in ("cov_prior", "cov_prior_rep", "cov_post") if c in df]].fillna(0).min(axis=1)
    strict = (df.fmt == "completion") & (df.readout == "dist") & (cov < 0.8)
    df["excluded_cov08"] = df.excluded | strict
    # Which part of the design this model x format ran
    n = df.groupby(["model", "fmt"]).trial_id.transform("size")
    has_vig = df.groupby(["model", "fmt"]).cue_type.transform(lambda s: (s == "vignette").any())
    df["design"] = np.where(df.prompt_variant > 0, "third_of_trials",
                            np.where(has_vig, "full", "labels_only"))
    if study != "pilot":   # Claude in the main folder ran a fixed subset of the labels-only trials
        df.loc[df.model == "claude", "design"] = "labels_only_subset"
    df["n_trials_model_fmt"] = n
    # Paraphrased prior asked? True for every model except the budget-limited API models,
    # which asked it on a seeded subset (the rest have p_prior_rep blank).
    if "retest" in df:
        df["retest"] = df["retest"].astype("boolean").fillna(True).astype(bool)
    else:
        df["retest"] = True
    return df


def check_capture_reruns(main: pd.DataFrame, ext: pd.DataFrame) -> None:
    """The capture-only re-runs must repeat the main run's answers (same prompts, same weights).
    They are not bit-identical: saving activations changes GPU batching and kernels, which moves
    probabilities by about 0.002 on average (r > 0.9997 observed). Larger gaps would mean a
    different model or prompt, so the check requires r > 0.999 and mean |difference| < 0.01."""
    for m in EXT_CAPTURE_ONLY:
        e = ext[ext.model == m][["trial_id", "fmt", "p_post"]]
        if e.empty:
            continue
        j = e.merge(main[main.model == m][["trial_id", "fmt", "p_post"]], on=["trial_id", "fmt"],
                    suffixes=("_ext", "_main")).dropna()
        diff = (j.p_post_ext - j.p_post_main).abs()
        r = float(np.corrcoef(j.p_post_ext, j.p_post_main)[0, 1])
        print(f"  check {m}: {len(j)} shared trials, r = {r:.5f}, mean |diff| = {diff.mean():.4f}")
        if not (r > 0.999 and diff.mean() < 0.01):
            raise SystemExit(f"{m}: exploratory re-run does not match the main run (r={r}, mean diff={diff.mean()})")


def build_diagnostic() -> None:
    """Exploratory addition 1 (evidence diagnostic, 480 tables, no persona or belief) as a small
    supplementary CSV: one row per table x question kind x model x format. The two-option
    answers (compare, support) store the probability of each option as p_option_A / p_option_B
    (option order was counterbalanced; `truth` is the index of the correct option, 0 = A)."""
    files = sorted((RUNS / EXT_RUN / "diag").glob("*_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in files if f.name != "items.parquet"], ignore_index=True)
    probs = d.pop("probs")
    d["p_option_A"] = [p[0] if p is not None and len(p) else np.nan for p in probs]
    d["p_option_B"] = [p[1] if p is not None and len(p) > 1 else np.nan for p in probs]
    d = d.sort_values(["model", "fmt", "diag_id", "kind"]).reset_index(drop=True)
    path = DATA / "evidence_diagnostic.csv.gz"
    d.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path.relative_to(ROOT)}: {len(d):,} rows ({d.groupby(['model', 'fmt']).ngroups} model x format)")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    build_diagnostic()
    print("loading pilot ...")
    pilot = add_columns(load_responses(RUNS / PILOT_RUN), "pilot")
    print("loading main ...")
    main_df = load_responses(RUNS / MAIN_RUN)
    print("loading exploratory ...")
    ext = load_responses(RUNS / EXT_RUN)
    check_capture_reruns(main_df, ext)
    ext = ext[~ext.model.isin(EXT_CAPTURE_ONLY)]
    api_added = main_df[main_df.model.isin(MAIN_RUN_API_ADDED)]
    main_df = main_df[~main_df.model.isin(MAIN_RUN_API_ADDED)]
    assert not set(api_added.model) & set(ext.model), "an added API model is also in the exploratory folder"
    both = pd.concat([add_columns(main_df, "main"), add_columns(ext, "exploratory"),
                      add_columns(api_added, "exploratory")], ignore_index=True)

    # Column order: identifiers, then design, then answers, then derived analysis columns.
    first = ["study", "model", "fmt", "trial_id", "base_model", "model_family", "model_size_b",
             "training_stage", "variant_tag", "prompt_variant", "debias_system_prompt", "open_weights",
             "hf_id", "hf_revision", "api_id", "design", "n_trials_model_fmt"]
    for name, df in (("pilot", pilot), ("main", both)):
        cols = first + [c for c in df.columns if c not in first]
        df = df[cols].sort_values(["study", "model", "fmt", "trial_id"]).reset_index(drop=True)
        path = DATA / f"{name}_trials.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        print(f"wrote {path.relative_to(ROOT)}: {len(df):,} rows x {df.shape[1]} columns")
        print(df.groupby(["study", "model", "fmt"]).agg(trials=("trial_id", "size"),
                                                      excluded=("excluded", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
