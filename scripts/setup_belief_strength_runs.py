#!/usr/bin/env python
"""Prepare the run folders for the belief-strength experiment (exploratory, 2026-11-16).

    .venv/bin/python scripts/setup_belief_strength_runs.py --config configs/bs.yaml

Copies the main run's trial table and, for every *_bs{v} alias, its base model's vignette-check file
(cues/ident_<base>.csv -> cues/ident_<alias>.csv), so stage_elicit reassigns failed vignettes exactly as in
the base run and every alias gets the same trials and cues. Refuses to overwrite a differing file.
"""
from __future__ import annotations
import argparse, filecmp, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from idea.pipeline import load_config, run_dir  # noqa: E402

BASE_RUN = {"olmo_instruct": "main", "llama": "main", "gpt": "main", "tulu_final": "ext", "phi4": "ext"}


def copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if not filecmp.cmp(src, dst, shallow=False):
            raise SystemExit(f"refusing to overwrite differing {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    cfg = load_config(a.config)
    rd = run_dir(cfg)
    main_rd = run_dir(load_config(ROOT / "configs" / "main.yaml"))
    ext_rd = run_dir(load_config(ROOT / "configs" / "ext.yaml"))
    src_rd = {"main": main_rd, "ext": ext_rd}
    if not filecmp.cmp(main_rd / "trials.parquet", ext_rd / "trials.parquet", shallow=False):
        raise SystemExit("main and ext trial tables differ")
    copy(main_rd / "trials.parquet", rd / "trials.parquet")
    for alias in cfg["models"]:
        base = alias.rsplit("_", 1)[0]   # <base>_bs<v> or <base>_nb
        copy(src_rd[BASE_RUN[base]] / "cues" / f"ident_{base}.csv", rd / "cues" / f"ident_{alias}.csv")
    print(f"ready: {rd}")


if __name__ == "__main__":
    main()
