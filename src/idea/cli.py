"""idea <command> --config configs/X.yaml [...]

Commands, in order of work:
  generate   build items and trials (optionally from a pilot run's screen + prior means)
  cues       vignette identification + leakage check for a model
  stance     perceived-stance calls for a model
  screen     topic screen, Tier 1 rule, identity-topic engagement from stance results
  elicit     main trials for a model/format (use --capture to save activations)
  capture    alias for `elicit --capture`
  pilot      pilot report: coverage, retest, null evidence, sums, calibration, omega hist
  power      power simulation (optionally from pilot inputs)
  prereg     write the preregistration document from the config
  fit        RQ1, secondary specifications, RQ2, RQ2b
  steer2     corrected RQ3 at the answer position + counterfactual subspace erasure (exploratory)
  steer      RQ3 + RQ3b (masking probes, directions, steering, subspaces, other biases)
  diag       exploratory evidence diagnostic: can the model read and use the tables?
  audit-merge  combine audit parts (audit --part ...) into audit.csv
  audit      benchmark suite under steering (SycEval, BBQ, MMLU-Pro, L2D)
"""
from __future__ import annotations

import argparse

from . import pipeline


EXIT_DEADLINE = 3  # stopped at IDEA_DEADLINE; rerunning the same command resumes from the cache
EXIT_BUDGET = 4    # refused or stopped by the per-run spending cap (elicit.max_cost_per_run)


def main(argv=None):
    import sys

    from .elicit import BudgetExceeded, DeadlineReached
    try:
        _main(argv)
    except DeadlineReached as e:
        print(f"IDEA: deadline reached ({e}); rerun the same command to resume.", file=sys.stderr)
        sys.exit(EXIT_DEADLINE)
    except BudgetExceeded as e:
        print(f"IDEA: spending cap: {e}. Nothing further was sent; answers so far are cached.", file=sys.stderr)
        sys.exit(EXIT_BUDGET)


def _main(argv=None):
    ap = argparse.ArgumentParser(prog="idea", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["generate", "cues", "stance", "screen", "elicit", "capture", "pilot", "power",
                                        "prereg", "fit", "steer", "steer2", "audit2", "p4v2", "audit", "audit-merge", "diag"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model")
    ap.add_argument("--fmt", default="chat", choices=["chat", "completion"])
    ap.add_argument("--shard", help="i/n: run every n-th trial starting at i (e.g. one per GPU)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="elicit: count calls and estimate API cost only")
    ap.add_argument("--pilot-run", help="run folder of the pilot (screen.json, prior_means.json)")
    ap.add_argument("--engine", default="ols", choices=["ols", "mixed"])
    ap.add_argument("--rq2-folds", type=int)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--max-eval", type=int)
    ap.add_argument("--l2d-path")
    ap.add_argument("--part", help="audit: one part only, e.g. steered/mmlu_pro:1of4 (merge with audit-merge)")
    a = ap.parse_args(argv)
    cfg = pipeline.load_config(a.config)

    def need_model():
        if not a.model:
            ap.error(f"{a.command} needs --model")
        return a.model

    if a.command == "generate":
        pipeline.stage_generate(cfg, a.pilot_run)
    elif a.command == "cues":
        pipeline.stage_cues(cfg, need_model())
    elif a.command == "stance":
        pipeline.stage_stance(cfg, need_model(), a.fmt)
    elif a.command == "screen":
        pipeline.stage_screen(cfg)
    elif a.command in ("elicit", "capture"):
        pipeline.stage_elicit(cfg, need_model(), a.fmt, a.shard, capture=True if (a.capture or a.command == "capture") else None,
                              limit=a.limit, dry_run=a.dry_run)
    elif a.command == "pilot":
        pipeline.stage_pilot_report(cfg)
    elif a.command == "power":
        pipeline.stage_power(cfg, a.model, a.reps)
    elif a.command == "prereg":
        from .prereg import write_prereg
        print(write_prereg(cfg))
    elif a.command == "fit":
        pipeline.stage_fit(cfg, a.engine, a.rq2_folds, a.n_boot)
    elif a.command == "steer":
        pipeline.stage_steer(cfg, need_model(), a.fmt, a.max_eval)
    elif a.command == "steer2":   # corrected + extended RQ3 (exploratory; see mech/rq3v2.py)
        from .mech import rq3v2
        rq3v2.run(cfg, need_model(), a.fmt)
    elif a.command == "p4v2":     # P4 redone at the answer position (exploratory)
        from .mech import rq3v2
        rq3v2.run_p4(cfg, need_model(), a.fmt)
    elif a.command == "audit2":   # audit of the winning corrected intervention; --part unsteered|corrected|control
        from .mech import rq3v2
        rq3v2.run_audit(cfg, need_model(), a.part or "corrected", a.fmt)
    elif a.command == "diag":
        pipeline.stage_diag(cfg, need_model(), a.fmt, a.shard)
    elif a.command == "audit":
        pipeline.stage_audit(cfg, need_model(), a.fmt, a.l2d_path, a.part)
    elif a.command == "audit-merge":
        pipeline.stage_audit_merge(cfg, need_model(), a.fmt)


if __name__ == "__main__":
    main()
