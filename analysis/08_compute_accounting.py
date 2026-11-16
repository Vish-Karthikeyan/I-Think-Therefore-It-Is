"""Step 8: compute accounting - every Stanford FarmShare job, what it did, how long it ran, GPU-hours.

    # 1. on FarmShare (refresh the raw records; -D keeps every run of a requeued job, -X one line
    #    per allocation):
    sacct -u vk27 -S 2026-10-01 -E now -D -X -P \
          -o JobID,JobName%80,Partition,QOS,State,Submit,Start,End,ElapsedRaw,AllocTRES,ReqMem,ExitCode,NodeList \
          > results/compute/sacct_raw.txt
    # 2. here:
    python analysis/08_compute_accounting.py

Input:  results/compute/sacct_raw.txt (Slurm accounting records, pipe-separated)
Output: results/compute/farmshare_runs.csv     one row per run (a requeued job has several runs)
        results/compute/farmshare_jobs.csv     one row per job id: purpose, runs, wall time, GPU-hours
        results/compute/farmshare_summary.csv  GPU-hours by study phase and task
        (also printed)

Definitions
  wall time   End - Start of each run (Slurm ElapsedRaw); jobs cancelled before starting have 0
  GPU-hours   number of GPUs allocated x wall time, summed over runs. All GPUs are NVIDIA L40S 48 GB
              (FarmShare "oat" nodes). Jobs without a GPU (dataset generation, model fitting) count
              CPU-hours only.
  status      the job's final state; "in progress" rows are counted up to the time sacct was run

Not FarmShare: the API models (GPT-4o, Claude) ran through OpenRouter, and one early pilot backup
ran on Kaggle (T4); those are reported separately in the paper.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "compute" / "sacct_raw.txt"
OUT = ROOT / "results" / "compute"


def gpus(tres: str) -> int:
    """Allocated GPU count from AllocTRES, e.g. 'billing=4,cpu=4,gres/gpu=1,mem=32G' -> 1."""
    m = re.search(r"gres/gpu=(\d+)", tres or "")
    return int(m.group(1)) if m else 0


def cpus(tres: str) -> int:
    m = re.search(r"(?:^|,)cpu=(\d+)", tres or "")
    return int(m.group(1)) if m else 0


def classify(name: str, submit: str) -> tuple[str, str, str]:
    """(phase, task, model) from the job name. Names come from scripts/farmshare/*.sh; each rule was
    checked against the job logs (first line: model and stages; then the exact `idea` commands)."""
    day = submit[:10]
    n = name
    if n in ("gpucheck.sbatch", "bash"):
        return "setup", "GPU and environment check", ""
    if n == "idea-gen-main":
        return "main", "generate trials (CPU)", ""
    if n == "idea-fit":
        return "main", "RQ1/RQ2 model fitting (CPU)", ""
    m = re.fullmatch(r"idea-diag-(\w+)", n)
    if m:
        return "exploratory", "evidence diagnostic (addition 1)", m.group(1)
    m = re.fullmatch(r"idea-prep-(\w+)", n)
    if m:
        return "main", "cue check + perceived stance", m.group(1)
    m = re.fullmatch(r"idea-(\w+?)-\d+-of-\d+", n)
    if m and not n.startswith("idea-x-"):
        return "main", "trial elicitation (shard)", m.group(1)
    m = re.fullmatch(r"idea-x-(\w+?)-\d+-of-\d+", n)
    if m:
        return "exploratory", "trial elicitation (shard)", m.group(1)
    m = re.fullmatch(r"idea-xprep-(\w+)", n)
    if m:
        return "exploratory", "cue check + stance (+ diagnostic)", m.group(1)
    m = re.fullmatch(r"idea-s-x-(\w+?)-(prep|s\d+)", n)
    if m:
        task = "cue check + stance + first shard" if m.group(2) == "prep" else f"trial elicitation ({m.group(2)})"
        return "exploratory", task, m.group(1)
    if n == "idea-rq3-olmo_instruct":
        return "main RQ3", "steering + audit (single job; audit hit the 6 h limit)", "olmo_instruct"
    if n in ("idea-rq3-llama", "idea-steer-llama", "idea-s-llama-steer"):
        return "main RQ3", "steering: probes, directions, sweep, test", "llama"
    m = re.fullmatch(r"idea-aud-(\w+?)-(unsteered|steered|random)-(.+)", n)
    if m:
        return "main RQ3", f"audit {m.group(2)} {m.group(3)}", m.group(1)
    if n == "idea-bundle-olmo_instruct-us":
        return "main RQ3", "audit unsteered + steered (4-GPU bundle)", "olmo_instruct"
    m = re.fullmatch(r"idea-s-olmoi-rand-p(\d)", n)
    if m:
        return "main RQ3", f"audit random part {int(m.group(1)) + 1}/7", "olmo_instruct"
    m = re.fullmatch(r"idea-s-llama-(unst|steered|rand)-p(\d)", n)
    if m:
        cond = {"unst": "unsteered", "steered": "steered", "rand": "random"}[m.group(1)]
        return "main RQ3", f"audit {cond} part {int(m.group(2)) + 1}/7", "llama"
    if n == "idea-p4-probes":
        return "exploratory", "P4 stage probes on saved activations (CPU)", "olmo_sft/dpo/instruct"
    m = re.fullmatch(r"idea-s-rq3v2-(\w+)", n)
    if m:
        model = {"olmoi": "olmo_instruct", "llama": "llama", "tulu": "tulu_final"}.get(m.group(1), m.group(1))
        return "RQ3 corrected", "answer-position capture, counterfactual subspaces, steering", model
    m = re.fullmatch(r"idea-s-audit2(n?)-(\w+?)-(unsteered|corrected|control)", n)
    if m:   # corrected-RQ3 audit; "audit2" = automatic choice, "audit2n" = named intervention (2026-11-15)
        model = {"olmoi": "olmo_instruct", "llama": "llama", "tulu": "tulu_final"}.get(m.group(2), m.group(2))
        how = "named intervention" if m.group(1) else "automatic choice"
        return "RQ3 corrected", f"audit {m.group(3)} ({how})", model
    m = re.fullmatch(r"idea-s-p4v2-(\w+)", n)
    if m:
        return "exploratory", "P4 rerun at the answer position", m.group(1)
    if n.startswith("idea-bundle-"):
        return "other", "4-GPU bundle (superseded by 1-GPU jobs)", ""
    m = re.fullmatch(r"idea-(\w+)", n)
    if m:   # pilot jobs (2026-11-10): cue check, stance and pilot elicitation per open model
        return ("pilot", "cue check + stance + pilot elicitation", m.group(1)) if day <= "2026-11-10" else \
               ("pilot", "pilot rerun (superseded)", m.group(1))
    return "other", n, ""


def main() -> None:
    raw = pd.read_csv(RAW, sep="|", dtype=str, keep_default_na=False)
    raw = raw[raw.JobID.str.fullmatch(r"\d+")]           # allocations only (no .batch/.extern steps)
    runs = pd.DataFrame({
        "job_id": raw.JobID.astype(int), "job_name": raw.JobName, "partition": raw.Partition, "qos": raw.QOS,
        "state": raw.State.str.replace(r" by \d+", "", regex=True), "submit": raw.Submit,
        # pending jobs show Start "Unknown"/"None": they have not run
        "start": raw.Start.replace({"Unknown": "", "None": ""}), "end": raw.End.replace({"Unknown": "", "None": ""}),
        "wall_hours": pd.to_numeric(raw.ElapsedRaw, errors="coerce").fillna(0) / 3600,
        "gpus": raw.AllocTRES.map(gpus), "cpus": raw.AllocTRES.map(cpus), "mem_requested": raw.ReqMem,
        "node": raw.NodeList.replace("None assigned", ""),
    })
    cls = [classify(n, s) for n, s in zip(runs.job_name, runs.submit)]
    runs["phase"], runs["task"], runs["model"] = zip(*cls)
    runs.loc[runs.start == "", "wall_hours"] = 0.0
    runs["gpu_hours"] = runs.gpus * runs.wall_hours
    runs["cpu_hours"] = runs.cpus * runs.wall_hours
    # Disclosed waste: Llama steering (job 1981007) restarted 9 times before 2026-11-14 14:00 without
    # reusing its cache, because unseeded randomized PCA changed the steering vectors (and so the
    # cache keys) on every restart. Runs that ended before the fix are repeated work; the run that
    # ended at 13:26 saved the directions the final run reused, so it is not counted as waste.
    runs["note"] = ""
    waste = (runs.job_id == 1981007) & (runs.end != "") & (runs.end < "2026-11-14T13:00")
    runs.loc[waste, "note"] = "repeated work (unseeded PCA bug, fixed 2026-11-14)"
    runs = runs.sort_values(["submit", "job_id", "start"]).reset_index(drop=True)
    runs.to_csv(OUT / "farmshare_runs.csv", index=False)

    # one row per job: a requeued job keeps its id; its runs are summed
    def final_state(s):
        s = list(s)
        return "in progress" if any(x in ("RUNNING", "PENDING") for x in s) else s[-1]
    jobs = runs.groupby("job_id").agg(
        job_name=("job_name", "first"), phase=("phase", "first"), task=("task", "first"), model=("model", "first"),
        submit=("submit", "first"), first_start=("start", lambda v: min([x for x in v if x] or [""])),
        last_end=("end", lambda v: max([x for x in v if x] or [""])), runs=("start", lambda v: sum(1 for x in v if x)),
        wall_hours=("wall_hours", "sum"), gpus=("gpus", "max"), gpu_hours=("gpu_hours", "sum"),
        cpu_hours=("cpu_hours", "sum"), status=("state", final_state)).reset_index()
    jobs.to_csv(OUT / "farmshare_jobs.csv", index=False)

    jobs["model"] = jobs.model.fillna("")
    summ = (jobs.groupby(["phase", "task", "model"]).agg(jobs=("job_id", "size"), runs=("runs", "sum"),
                                                         wall_hours=("wall_hours", "sum"), gpu_hours=("gpu_hours", "sum"))
            .reset_index().sort_values(["phase", "gpu_hours"], ascending=[True, False]))
    by_phase = jobs.groupby("phase").agg(jobs=("job_id", "size"), gpu_hours=("gpu_hours", "sum")).reset_index()
    summ.to_csv(OUT / "farmshare_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 300):
        print(summ.round(2).to_string(index=False))
        print("\nby phase:\n" + by_phase.round(2).to_string(index=False))
    ran = jobs[jobs.runs > 0]
    print(f"\nFarmShare total: {len(jobs)} jobs submitted, {len(ran)} ran ({int(jobs.runs.sum())} runs incl. requeues); "
          f"{jobs.gpu_hours.sum():.1f} GPU-hours (NVIDIA L40S), {jobs.cpu_hours.sum():.1f} CPU-hours; "
          f"{int((jobs.status == 'in progress').sum())} jobs still in progress")
    w = runs[runs.note != ""]
    print(f"of which repeated work (Llama steering restarts before the fix): {len(w)} runs, {w.gpu_hours.sum():.1f} GPU-hours")


if __name__ == "__main__":
    main()
