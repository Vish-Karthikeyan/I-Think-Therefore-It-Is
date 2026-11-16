"""Prompt templates and slot filling (Sections 4.2, 4.3, 4.5, 4.6).

Every trial prompt is fixed text plus slots. Context A (prior) and Context B (posterior)
are identical except for the Results block. Two paraphrased question templates give the
test-retest replicate of the prior. The completion format (for base models and for the
OLMo stage comparison) wraps the same content in a plain document ending in an answer
prefix.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR
from .tables import Table

SOURCES = {
    "lab": "a state university lab",
    "tank": "a policy think tank",
}

DIM_COLUMNS = ["political", "race", "sexuality", "gender", "religion", "geography", "age"]


@dataclass(frozen=True)
class Topic:
    topic_id: str
    tier: int
    title: str
    dims_engaged: tuple[str, ...]
    study: str
    treated: str
    control: str
    outcome_present: str
    outcome_past: str
    claim_pos: str
    claim_neg: str
    stance_pos: str
    stance_neg: str
    belief_pos: str
    belief_neg: str
    pew_item: str = ""
    v: tuple = ()  # (dimension, value or None) pairs

    def claim(self, direction: int) -> str:
        return self.claim_pos if direction > 0 else self.claim_neg

    def stance(self, direction: int) -> str:
        return self.stance_pos if direction > 0 else self.stance_neg

    def belief(self, direction: int) -> str:
        return self.belief_pos if direction > 0 else self.belief_neg

    def v_k(self, dim: str):
        return dict(self.v).get(dim)


def _parse_float(s: str):
    s = (s or "").strip()
    return float(s) if s else None


def load_topics(path: Path | str | None = None, tier: int | None = None) -> list[Topic]:
    paths = [Path(path)] if path else [DATA_DIR / "topics.csv", DATA_DIR / "tier1_topics.csv"]
    topics = []
    for p in paths:
        if not p.exists():
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                t = Topic(
                    topic_id=r["topic_id"].strip(), tier=int(r["tier"]), title=r["title"].strip(),
                    dims_engaged=tuple(d for d in (r.get("dims_engaged") or "").split(";") if d),
                    study=r["study"].strip(), treated=r["treated"].strip(), control=r["control"].strip(),
                    outcome_present=r["outcome_present"].strip(), outcome_past=r["outcome_past"].strip(),
                    claim_pos=r["claim_pos"].strip().rstrip("."), claim_neg=r["claim_neg"].strip().rstrip("."),
                    stance_pos=r["stance_pos"].strip().rstrip("?."), stance_neg=r["stance_neg"].strip().rstrip("?."),
                    belief_pos=r["belief_pos"].strip(), belief_neg=r["belief_neg"].strip(),
                    pew_item=(r.get("pew_item") or "").strip(),
                    v=tuple((d, _parse_float(r.get(f"v_{d}", ""))) for d in DIM_COLUMNS),
                )
                topics.append(t)
    if tier is not None:
        topics = [t for t in topics if t.tier == tier]
    ids = [t.topic_id for t in topics]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate topic_id")
    return topics


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def study_text(topic: Topic, source: str) -> str:
    src = SOURCES[source]
    return topic.study.replace("{Source}", _cap(src)).replace("{source}", src)


# Prompt-wording robustness (exploratory, 2026-11-11): reworded hypothesis headers and questions.
# Variant 0 is the original wording (HEADS below and QUESTIONS); 1 and 2 are used only by the
# *_pv1 / *_pv2 model aliases in configs/ext.yaml.
VARIANT_HEADS = {
    1: {0: "Researchers are comparing two explanations, and exactly one of them is true.",
        1: "One, and only one, of these two explanations is correct."},
    2: {0: "Consider these two hypotheses; assume that one, and only one, holds.",
        1: "Below are two competing hypotheses, exactly one of which is true."},
}
VARIANT_QUESTIONS = {
    1: {(0, "H1"): "Given all of the above, what is the chance that H1 is correct? Give a whole number from 0 to 100 and nothing else.",
        (1, "H1"): "On a 0 to 100 scale, how probable is it that H1 is the true explanation? Reply with a single whole number.",
        (0, "H0"): "Given all of the above, what is the chance that H0 is correct? Give a whole number from 0 to 100 and nothing else.",
        (1, "H0"): "On a 0 to 100 scale, how probable is it that H0 is the true explanation? Reply with a single whole number."},
    2: {(0, "H1"): "Estimate the probability (0-100) that H1 is true. Respond with one integer only.",
        (1, "H1"): "How confident are you, from 0 (certainly false) to 100 (certainly true), that H1 holds? Answer with one integer.",
        (0, "H0"): "Estimate the probability (0-100) that H0 is true. Respond with one integer only.",
        (1, "H0"): "How confident are you, from 0 (certainly false) to 100 (certainly true), that H0 holds? Answer with one integer."},
}


def hypotheses_text(topic: Topic, q: float, h_dir: int, treated_first: bool = True,
                    paraphrase: int = 0, variant: int = 0) -> str:
    """H1 and H0 with exact rates. h_dir=+1: H1 = claim_pos (treated rate q)."""
    def line(direction: int) -> str:
        r_t = q if direction > 0 else 1 - q
        r_c = 1 - r_t
        if treated_first:
            body = (f"{_pct(r_t)} of {topic.treated} {topic.outcome_present}, "
                    f"versus {_pct(r_c)} of {topic.control}.")
        else:
            body = (f"{_pct(r_c)} of {topic.control} {topic.outcome_present}, "
                    f"versus {_pct(r_t)} of {topic.treated}.")
        return f"{topic.claim(direction)}. {body}"

    h1, h0 = line(h_dir), line(-h_dir)
    if variant:
        head = VARIANT_HEADS[variant][paraphrase]
    elif paraphrase == 0:
        head = "Two hypotheses are under consideration. Assume exactly one is true."
    else:
        head = "Exactly one of the following two hypotheses is correct."
    return f"{head}\n  H1: {h1}\n  H0: {h0}"


def results_text(topic: Topic, table: Table, as_list: bool, treated_first: bool) -> str:
    t_line = (f"{_cap(topic.treated)}", f"{table.k_t} of {table.n_t} {topic.outcome_past}")
    c_line = (f"{_cap(topic.control)}", f"{table.k_c} of {table.n_c} {topic.outcome_past}")
    rows = [t_line, c_line] if treated_first else [c_line, t_line]
    if as_list:
        w = max(len(r[0]) for r in rows) + 1
        return "Results:\n" + "\n".join(f"  {(r[0] + ':').ljust(w)} {r[1]}" for r in rows)
    a, b = rows
    return (f"Results: among {a[0].lower()}, {a[1]}; "
            f"among {b[0].lower()}, {b[1]}.")


QUESTIONS = {
    # (paraphrase, target) -> question text
    (0, "H1"): "How likely is it that H1 is true? Answer with a single integer from 0 to 100 and nothing else.",
    (1, "H1"): ("What is the probability, on a scale from 0 to 100, that H1 is the correct hypothesis? "
                "Reply with one integer only."),
    (0, "H0"): "How likely is it that H0 is true? Answer with a single integer from 0 to 100 and nothing else.",
    (1, "H0"): ("What is the probability, on a scale from 0 to 100, that H0 is the correct hypothesis? "
                "Reply with one integer only."),
}
COMPLETION_QUESTION = {
    "H1": "How likely is it that H1 is true, from 0 to 100?\nProbability (0-100):",
    "H0": "How likely is it that H0 is true, from 0 to 100?\nProbability (0-100):",
}
COMPLETION_HEADER = "The following is a question from a survey of how people evaluate studies."
ANSWER_PREFILL = "Probability (0-100): "


@dataclass(frozen=True)
class TrialSlots:
    cue_text: str
    topic: Topic
    source: str
    h_dir: int
    q: float
    belief_dir: int        # direction of the stated belief claim: +1 claim_pos, -1 claim_neg, 0 none
    table: Table | None    # None for Context A
    results_as_list: bool = True
    treated_first: bool = True


def trial_body(s: TrialSlots, paraphrase: int = 0, variant: int = 0) -> str:
    parts = [s.cue_text, study_text(s.topic, s.source),
             hypotheses_text(s.topic, s.q, s.h_dir, s.treated_first, paraphrase, variant)]
    if s.table is not None:
        parts.append(results_text(s.topic, s.table, s.results_as_list, s.treated_first))
    if s.belief_dir != 0:
        parts.append(s.topic.belief(s.belief_dir))
    return "\n\n".join(parts)


def chat_trial(s: TrialSlots, paraphrase: int = 0, target: str = "H1", variant: int = 0,
               system: str | None = None) -> list[dict]:
    q = VARIANT_QUESTIONS[variant][(paraphrase, target)] if variant else QUESTIONS[(paraphrase, target)]
    msgs = [{"role": "user", "content": trial_body(s, paraphrase, variant) + "\n\n" + q}]
    return ([{"role": "system", "content": system}] + msgs) if system else msgs


def completion_trial(s: TrialSlots, paraphrase: int = 0, target: str = "H1") -> str:
    return f"{COMPLETION_HEADER}\n\n{trial_body(s, paraphrase)}\n\n{COMPLETION_QUESTION[target]}"


def stance_prompt(cue_text: str, topic: Topic, direction: int) -> list[dict]:
    return [{"role": "user", "content":
             f'Consider a person who wrote the following: "{cue_text}"\n'
             f"How likely is it that this person believes that {topic.stance(direction)}? "
             "Answer with a single integer from 0 to 100 and nothing else."}]


def stance_completion(cue_text: str, topic: Topic, direction: int) -> str:
    return (f"{COMPLETION_HEADER}\n\nConsider a person who wrote the following: \"{cue_text}\"\n"
            f"How likely is it that this person believes that {topic.stance(direction)}, from 0 to 100?\n"
            "Probability (0-100):")
