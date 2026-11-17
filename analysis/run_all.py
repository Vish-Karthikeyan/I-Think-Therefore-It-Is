#!/usr/bin/env python
"""Run every analysis: rq1.py and rq2.py, then rq3.py if a CUDA GPU is available.

    python analysis/run_all.py                          # everything on datasets/main_trials.csv.gz
    python analysis/run_all.py --list                   # each script's analyses (the toggles)
    python analysis/run_all.py --only rq1               # one script
    python analysis/run_all.py --only rq1.primary,rq2   # rq1's "primary" analysis, and all of rq2
    python analysis/run_all.py --skip rq2,rq1.robustness
    python analysis/run_all.py --data my_trials.csv.gz --save results/   # your own dataset; save tables

--only / --skip take script names (rq1, rq2, rq3) or script.analysis names (see --list); a dotted
name is passed on to that script as its own --only / --skip. --data, --models and --save are
passed through (--save DIR writes each script's tables to DIR/<script>/).

rq1 and rq2 read only the dataset (CPU, a few minutes). rq3 runs the open-weight models
themselves (activations and steering), so it needs the model weights and a GPU; without one it is
skipped with instructions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["rq1", "rq2", "rq3"]


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def split(tokens: str | None) -> tuple[set[str], dict[str, list[str]]]:
    """'rq1,rq2.cv' -> ({'rq1'}, {'rq2': ['cv']})."""
    whole, part = set(), {}
    for t in filter(None, (x.strip() for x in (tokens or "").split(","))):
        s, _, a = t.partition(".")
        if s not in SCRIPTS:
            raise SystemExit(f"unknown script {s!r} in {t!r}; choose from {', '.join(SCRIPTS)}")
        if a:
            part.setdefault(s, []).append(a)
        else:
            whole.add(s)
    return whole, part


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list each script's analyses and exit")
    ap.add_argument("--only", help="scripts or script.analysis names to run (comma list)")
    ap.add_argument("--skip", help="scripts or script.analysis names to skip (comma list)")
    ap.add_argument("--data", help="dataset CSV (default: datasets/main_trials.csv.gz)")
    ap.add_argument("--models", help="comma list of model keys (default: all in the data)")
    ap.add_argument("--save", help="directory for the result tables (CSV), one subfolder per script")
    a = ap.parse_args(argv)

    only_whole, only_part = split(a.only)
    skip_whole, skip_part = split(a.skip)
    chosen = [s for s in SCRIPTS if (not a.only or s in only_whole or s in only_part) and s not in skip_whole]

    failed = []
    for s in chosen:
        cmd = [sys.executable, str(HERE / f"{s}.py")]
        if a.list:
            cmd.append("--list")
        else:
            if a.data:
                cmd += ["--data", a.data]
            if a.models:
                cmd += ["--models", a.models]
            if a.save:
                cmd += ["--save", str(Path(a.save) / s)]
            if s in only_part and s not in only_whole:
                cmd += ["--only", ",".join(only_part[s])]
            if s in skip_part:
                cmd += ["--skip", ",".join(skip_part[s])]
        print(f"\n{'=' * 100}\n{s}.py\n{'=' * 100}", flush=True)
        if s == "rq3" and not a.list and not cuda_available():
            print("rq3 skipped: no CUDA GPU found (torch.cuda.is_available() is False).\n"
                  "RQ3 re-runs the open-weight models (OLMo 2 7B Instruct, Llama 3.1 8B Instruct, Tulu 3 8B)\n"
                  "to read their hidden states and apply the steering and erasure interventions, so it needs\n"
                  "the model weights and a GPU (about one 48 GB GPU per 7-8B model). On a GPU machine run:\n"
                  "    python analysis/rq3.py --list\n"
                  "    python analysis/rq3.py" + (f" --data {a.data}" if a.data else ""))
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append(f"{s} (exit {rc})")
    if failed:
        print(f"\nfailed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
