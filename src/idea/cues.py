"""Identity cue library: explicit labels, implicit vignettes, Unspecified cues, and the
vignette identification / leakage checks (Section 4.2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from .paths import DATA_DIR

DIMENSIONS: dict[str, list[str]] = {
    "political": ["democrat", "republican", "independent"],
    "race": ["white", "black", "hispanic", "asian_american"],
    "sexuality": ["straight", "gay_lesbian", "bisexual"],
    "gender": ["man", "woman", "nonbinary"],
    "religion": ["evangelical", "catholic", "jewish", "muslim", "atheist"],
    "geography": ["rural", "urban"],
    "age": ["age_20s", "age_60s"],
}
DIM_INDEX = {k: i + 1 for i, k in enumerate(DIMENSIONS)}  # k = 1..7 as in the notes
GROUP_TO_DIM = {g: d for d, gs in DIMENSIONS.items() for g in gs}
ALL_GROUPS = [g for gs in DIMENSIONS.values() for g in gs]
UNSPECIFIED = "unspecified"

# Wording of each dimension in the identification question.
DIM_QUESTION = {
    "political": "political affiliation",
    "race": "race or ethnicity",
    "sexuality": "sexual orientation",
    "gender": "gender",
    "religion": "religion",
    "geography": "type of place they live",
    "age": "age group",
}
CANNOT_TELL = "Cannot tell"


@dataclass(frozen=True)
class Cue:
    cue_id: str
    group_id: str          # one of ALL_GROUPS or "unspecified"
    dimension: str         # one of DIMENSIONS or "none"
    cue_type: str          # "label" | "vignette"
    text: str
    status: str = "final"

    @property
    def is_unspecified(self) -> bool:
        return self.group_id == UNSPECIFIED


@dataclass
class CueLibrary:
    cues: list[Cue]
    options: dict[str, str] = field(default_factory=dict)  # group_id -> option wording

    def by_id(self, cue_id: str) -> Cue:
        return self._index[cue_id]

    def __post_init__(self):
        self._index = {c.cue_id: c for c in self.cues}
        if len(self._index) != len(self.cues):
            raise ValueError("duplicate cue_id")

    def select(self, group_id=None, cue_type=None, dimension=None, exclude: set | None = None) -> list[Cue]:
        out = []
        for c in self.cues:
            if group_id is not None and c.group_id != group_id:
                continue
            if cue_type is not None and c.cue_type != cue_type:
                continue
            if dimension is not None and c.dimension != dimension:
                continue
            if exclude and c.cue_id in exclude:
                continue
            out.append(c)
        return out

    def unspecified(self, cue_type: str) -> list[Cue]:
        return self.select(group_id=UNSPECIFIED, cue_type=cue_type)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.cues])


def load_cues(data_dir: Path | str = DATA_DIR, include_draft: bool = True) -> CueLibrary:
    data_dir = Path(data_dir)
    cues: list[Cue] = []
    options = {}
    for row in yaml.safe_load((data_dir / "cues" / "labels.yaml").read_text()):
        cues.append(Cue(f"{row['group_id']}_label", row["group_id"], row["dimension"], "label", row["text"]))
        options[row["group_id"]] = row["option"]
    unsp_path = data_dir / "cues" / "unspecified_labels.yaml"
    if unsp_path.exists():
        for row in yaml.safe_load(unsp_path.read_text()):
            cues.append(Cue(row["cue_id"], UNSPECIFIED, "none", "label", row["text"]))
    else:
        cues.append(Cue("unsp_label_1", UNSPECIFIED, "none", "label", "I'm a reader of policy research."))
    vpath = data_dir / "cues" / "vignettes.yaml"
    if vpath.exists():
        for row in yaml.safe_load(vpath.read_text()) or []:
            status = row.get("status", "draft")
            if status == "rejected" or (status == "draft" and not include_draft):
                continue
            gid = row["group_id"]
            dim = "none" if gid == UNSPECIFIED else row.get("dimension", GROUP_TO_DIM[gid])
            cues.append(Cue(row["vignette_id"], gid, dim, "vignette", " ".join(row["text"].split()), status))
    lib = CueLibrary(cues, options)
    _validate(lib)
    return lib


def _validate(lib: CueLibrary) -> None:
    for c in lib.cues:
        if c.group_id != UNSPECIFIED and GROUP_TO_DIM.get(c.group_id) != c.dimension:
            raise ValueError(f"cue {c.cue_id}: group {c.group_id} not in dimension {c.dimension}")
    for g in ALL_GROUPS:
        if not lib.select(group_id=g, cue_type="label"):
            raise ValueError(f"no label for {g}")


# ---------------------------------------------------------------------------
# Identification and leakage checks
# ---------------------------------------------------------------------------

def identification_prompt(cue_text: str, dimension: str, options: list[str]) -> tuple[str, list[str]]:
    opts = options + [CANNOT_TELL]
    letters = [chr(65 + i) for i in range(len(opts))]
    q = (f"Here is something a person wrote:\n\n\"{cue_text}\"\n\n"
         f"Which of these best describes the writer's {DIM_QUESTION[dimension]}?\n"
         + "\n".join(f"({l}) {o}" for l, o in zip(letters, opts))
         + "\nAnswer with the letter only.")
    return q, letters


def identification_requests(lib: CueLibrary, leakage: bool = True) -> list[dict]:
    """One request per (vignette, dimension). The target dimension gives the pass/fail
    check; the other dimensions are the leakage check."""
    reqs = []
    for c in lib.select(cue_type="vignette"):
        dims = list(DIMENSIONS) if (leakage or c.is_unspecified) else [c.dimension]
        for dim in dims:
            groups = DIMENSIONS[dim]
            opts = [lib.options[g] for g in groups]
            text, letters = identification_prompt(c.text, dim, opts)
            reqs.append(dict(cue_id=c.cue_id, group_id=c.group_id, cue_dim=c.dimension,
                             ask_dim=dim, options=groups + ["cannot_tell"], letters=letters, prompt=text))
    return reqs


def score_identification(rows: pd.DataFrame, threshold: float = 0.7, leak_threshold: float = 0.5) -> pd.DataFrame:
    """rows: one row per request with columns cue_id, group_id, cue_dim, ask_dim, options,
    probs (list aligned to options). Returns per-vignette pass flag and leakage flags."""
    out = []
    for cue_id, grp in rows.groupby("cue_id"):
        r0 = grp.iloc[0]
        rec = dict(cue_id=cue_id, group_id=r0.group_id, dimension=r0.cue_dim)
        leaks = []
        for _, r in grp.iterrows():
            probs = dict(zip(r.options, r.probs))
            if r.ask_dim == r.cue_dim:
                rec["p_target"] = probs.get(r.group_id, float("nan"))
                rec["p_cannot_tell"] = probs.get("cannot_tell", float("nan"))
            else:
                top = max((g for g in r.options if g != "cannot_tell"), key=lambda g: probs[g])
                if probs[top] >= leak_threshold:
                    leaks.append(f"{r.ask_dim}:{top}:{probs[top]:.2f}")
        if r0.group_id == UNSPECIFIED:
            rec["passed"] = len(leaks) == 0
        else:
            rec["passed"] = rec.get("p_target", 0) >= threshold
        rec["leaks"] = ";".join(leaks)
        out.append(rec)
    return pd.DataFrame(out)
