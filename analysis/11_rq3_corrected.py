"""Step 11: corrected RQ3 (answer-position analysis; exploratory beyond the correction).

    python analysis/11_rq3_corrected.py

Reads the FarmShare outputs copied to results/rq3_v2/<model>_chat/ (`idea steer2`, `idea audit2`) and
writes the tables behind the paper's corrected-RQ3 paragraph to analysis/output/rq3_v2/:

  test.csv          held-out test of every chosen intervention and its random control, with the three
                    success tiers (preregistered, exploratory, partial; defined in the protocol-changes
                    document before any corrected test result existed) and the share of the gap closed
  sufficiency.csv   interchange test: share of the belief effect on the answer recovered by patching
                    a subspace from a belief-stated input into the matching no-belief input, by layer
  geometry.csv      norms and angle of the two belief directions (belief in H1, belief in H0) by layer
  audit.csv         capability and bias audit per model x condition (MMLU-Pro, BBQ, SycEval, L2D),
                    with a status column, so partly finished audits are shown as such

Audit labels: `unsteered` is the model without intervention. Audits named on the command line
(`--part corrected@erase_answer:L24_k4`) are labeled `<condition>-<family>-<setting>`; the earlier
automatic choice wrote `corrected` and `control`, recorded in winner.json. The automatic choice had two
flaws (it ranked criteria before effect size, and it took the largest gap closure even past 100%), so
it audited Llama's overshooting `cr_single L16_x16`; those two audits were cancelled on 2026-11-15
before finishing and are listed as secondary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from idea.mech.rq3v2 import partial_success  # noqa: E402  (the exact tier-3 rule the pipeline uses)

RES = ROOT / "results" / "rq3_v2"
OUT = Path(__file__).resolve().parent / "output" / "rq3_v2"
MODELS = {"olmo_instruct_chat": "olmo_instruct", "llama_chat": "llama", "tulu_final_chat": "tulu_final"}
N_MMLU_SUBJECTS = 14


def test_table() -> pd.DataFrame:
    rows = []
    for d, key in MODELS.items():
        f = RES / d / "steering_test_v2.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        ctrl = t[t.family == "control_random_subspace"]
        # Same bar as the pipeline: at least 25% of the gap, and 10 points above the best random control
        bar = max(0.25, (float(ctrl.gap_closed_frac.max()) if len(ctrl) else 0.0) + 0.10)
        t = t.assign(model=key, audit_bar=bar,
                     success_partial=partial_success(t, bar) & (t.family != "control_random_subspace"),
                     overshoot=t.gap_closed_frac > 1.0)
        rows.append(t[["model", "family", "setting", "t2_contra_gap_before", "t2_contra_gap_after", "gap_closed_frac",
                       "overshoot", "t2_neutral_change", "tier1_neutral_change", "tier1_contra_change",
                       "tier1_support_change", "success_preregistered", "success_exploratory", "success_partial",
                       "audit_bar"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sufficiency_table() -> pd.DataFrame:
    rows = []
    for d, key in MODELS.items():
        f = RES / d / "sufficiency.csv"
        if f.exists():
            s = pd.read_csv(f)
            # average over the two directions of the swap (belief H1 -> H0 and H0 -> H1)
            p = s.groupby(["layer", "patch"]).recovery.mean().unstack("patch").reset_index()
            rows.append(p.assign(model=key))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def geometry_table() -> pd.DataFrame:
    rows = [pd.read_csv(RES / d / "geometry.csv").assign(model=key)
            for d, key in MODELS.items() if (RES / d / "geometry.csv").exists()]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def label_map(adir: Path) -> dict:
    """label -> (condition, family, setting, how chosen)."""
    out = {"unsteered": ("unsteered", "", "", "")}
    w = adir / "winner.json"
    if w.exists():
        j = json.loads(w.read_text())
        for cond in ("corrected", "control"):
            out[cond] = (cond, j["family"], j["setting"], f"automatic ({j.get('criterion', '')})")
    for f in adir.glob("*__*.csv"):
        label = f.name.split("__")[0]
        if label not in out and "-" in label:
            cond, family, setting = label.split("-", 2)
            out[label] = (cond, family, setting, "named")
    return out


def read_one(f: Path) -> pd.Series | None:
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return d.iloc[0] if len(d) else None


def audit_table() -> pd.DataFrame:
    rows = []
    for d, key in MODELS.items():
        adir = RES / d / "audit2"
        if not adir.exists():
            continue
        for label, (cond, family, setting, how) in sorted(label_map(adir).items()):
            r = dict(model=key, label=label, condition=cond, family=family, setting=setting, chosen=how)
            m = read_one(adir / f"{label}__mmlu_pro.csv")
            if m is not None:
                subj = [c[:-2] for c in m.index if c.startswith("mmlu_pro_") and c.endswith("/n")]
                n = np.array([m[f"{s}/n"] for s in subj], float)
                acc = np.array([m[f"{s}/exact_match,custom-extract"] for s in subj], float)
                r.update(mmlu_pro_acc=float(np.sum(n * acc) / np.sum(n)), mmlu_pro_subjects=len(subj),
                         mmlu_pro_n=int(n.sum()))
            b = read_one(adir / f"{label}__bbq.csv")
            if b is not None:
                r.update(bbq_acc=b["bbq/acc,none"], bbq_acc_amb=b["bbq/accuracy_amb,none"],
                         bbq_acc_disamb=b["bbq/accuracy_disamb,none"], bbq_amb_bias=b["bbq/amb_bias_score,none"],
                         bbq_disamb_bias=b["bbq/disamb_bias_score,none"])
            s = read_one(adir / f"{label}__syceval.csv")
            if s is not None:
                r.update(syceval_initial_acc=s["syceval_initial_accuracy"],
                         syceval_regressive_flip=s["syceval_regressive_flip_rate"],
                         syceval_progressive_flip=s["syceval_progressive_flip_rate"])
            l2 = read_one(adir / f"{label}__l2d.csv")
            if l2 is not None:
                r.update(l2d_n=l2["n"], l2d_parsed=l2["parsed_share"], l2d_truth_discernment=l2["truth_discernment"],
                         l2d_source_discernment=l2["source_discernment"], l2d_correct_defense=l2["correct_defense"])
            have = [c for c, k in (("syceval", "syceval_initial_acc"), ("bbq", "bbq_acc"),
                                   ("mmlu_pro", "mmlu_pro_acc"), ("l2d", "l2d_truth_discernment")) if k in r]
            complete = len(have) == 4 and r.get("mmlu_pro_subjects") == N_MMLU_SUBJECTS
            r["status"] = "complete" if complete else "partial: " + (", ".join(have) or "nothing yet")
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    for name, tab in (("test", test_table()), ("sufficiency", sufficiency_table()),
                      ("geometry", geometry_table()), ("audit", audit_table())):
        tab.to_csv(OUT / f"{name}.csv", index=False)
        print(f"== {name} ({len(tab)} rows) -> {(OUT / f'{name}.csv').relative_to(ROOT)}")
        show = tab if name != "geometry" else tab[["model", "layer", "cos_pos_neg", "norm_pos", "norm_neg"]]
        print(show.round(3).to_string(index=False) if len(show) else "  (no results)")


if __name__ == "__main__":
    main()
