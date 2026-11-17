"""Run-folder helpers used by the RQ3 code (idea.mech): config loading, run folders named by
the config's hash (runs/<name>-<hash8>/), backends and cached elicitors for a model key, and
the loader that assembles a run folder's responses into the analysis frame.

New datasets are built with elicit.py at the repository root."""
from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path

import pandas as pd
import yaml

from .cues import CueLibrary, load_cues
from .elicit import Cache, CachedElicitor, make_backend
from .models import get_model
from .paths import RUNS_DIR
from .prompts import load_topics


# ---------------------------------------------------------------------------
# Config and run folders
# ---------------------------------------------------------------------------

def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("name", Path(path).stem)
    return cfg


def config_hash(cfg: dict) -> str:
    keep = {k: cfg.get(k) for k in ("seed", "design", "topics", "cues")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:8]


def run_dir(cfg: dict, root: Path | None = None) -> Path:
    d = Path(cfg.get("run_dir") or (Path(root or RUNS_DIR) / f"{cfg['name']}-{config_hash(cfg)}"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return d


def _topics(cfg):
    t = load_topics()
    keep = (cfg.get("topics") or {}).get("include")
    drop = set((cfg.get("topics") or {}).get("exclude") or [])
    if keep:
        t = [x for x in t if x.topic_id in set(keep)]
    return [x for x in t if x.topic_id not in drop]


def _cues(cfg) -> CueLibrary:
    return load_cues(include_draft=(cfg.get("cues") or {}).get("include_draft", True))


def sample_k_for(cfg: dict, model_key: str) -> int:
    """Samples per prompt for the sampling readout: elicit.sample_k_by_model overrides
    elicit.sample_k (e.g. Claude at K=5 to halve API cost; open-model calibration stays 10)."""
    e = cfg.get("elicit") or {}
    return int((e.get("sample_k_by_model") or {}).get(model_key, e.get("sample_k", 10)))


def stance_sample_k(cfg: dict, model_key: str) -> int:
    """Samples per question for stance and vignette-check calls (may differ from trials)."""
    e = cfg.get("elicit") or {}
    return int((e.get("stance_sample_k_by_model") or {}).get(model_key, sample_k_for(cfg, model_key)))


def _backend(cfg, model_key: str, **over):
    spec = get_model(model_key)
    e = dict(cfg.get("elicit") or {})
    e.update(over)
    kw = {}
    if spec.backend == "hf":
        kw = dict(batch_size=(e.get("batch_size_by_model") or {}).get(model_key, e.get("batch_size", 8)),
                  coverage_min=e.get("coverage_min", 0.9),
                  sample_k=sample_k_for(cfg, model_key), temperature=e.get("temperature", 1.0),
                  capture_layers=e.get("capture_layers", ()), dtype=e.get("dtype", "auto"))
    elif spec.backend == "openrouter":
        kw = dict(concurrency=int(os.environ.get("IDEA_CONCURRENCY") or e.get("concurrency", 16)), sample_k=over.get("sample_k", sample_k_for(cfg, model_key)),
                  coverage_min=e.get("coverage_min", 0.9),
                  max_cost=float(os.environ.get("IDEA_MAX_COST") or e.get("max_cost_per_run", 15.0)),
                  temperature=e.get("temperature", 1.0),
                  api_key_env=(e.get("api_key_env_by_model") or {}).get(model_key, "OPENROUTER_API_KEY"))
    elif spec.backend == "mock":
        kw = dict(capture_layers=e.get("capture_layers", (4, 8)) or (4, 8))
    return make_backend(spec, **kw)


def cache_path(rd: Path, model_key: str, shard: str | None = None) -> Path:
    """One cache file per model, and per piece when a model is split (--shard i/n): SQLite on
    NFS (cluster scratch) is only safe with one writer per file, and pieces run at once."""
    tag = f"_s{shard.replace('/', 'of')}" if shard else ""
    return rd / "cache" / f"{model_key}{tag}.sqlite"


def _elicitor(rd: Path, backend, act_dir: Path | None = None, chunk: int = 256, shard: str | None = None):
    from .mech.capture import ActStore
    store = ActStore(act_dir) if act_dir else None
    cache = Cache(cache_path(rd, backend.spec.key, shard))
    return CachedElicitor(backend, cache, chunk=chunk, act_store=store)


def load_responses(rd: Path) -> pd.DataFrame:
    from .runner import derive
    from .stance import attach_g, attach_survey_gap, g_behavioral, g_survey
    files = sorted((rd / "responses").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no responses in {rd}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["trial_id", "model", "fmt"], keep="last")
    df = derive(df)
    if os.environ.get("IDEA_TIER1_RULE") == "1":
        # robustness: Tier 1 restricted to (topic, cue) pairs with |g| < 0.25 in the pilot screen
        tr = pd.read_parquet(rd / "trials.parquet")
        if "tier1_low_g" not in tr:
            raise RuntimeError("IDEA_TIER1_RULE=1 but trials.parquet has no tier1_low_g column")
        ok = set(tr.trial_id[tr.tier1_low_g])
        n0 = len(df)
        df = df[df.trial_id.isin(ok)].reset_index(drop=True)
        print(f"IDEA_TIER1_RULE=1: kept {len(df)}/{n0} responses (Tier 1 robustness subset)")
    stance_files = list((rd / "stance").glob("*.parquet")) if (rd / "stance").exists() else []
    if stance_files:
        st = pd.concat([pd.read_parquet(f) for f in stance_files], ignore_index=True)
        parts = []
        for fmt, g in df.groupby("fmt"):
            s = st[st.fmt == fmt] if (st.fmt == fmt).any() else st[st.fmt == "chat"]
            parts.append(attach_g(g, s))
        df = pd.concat(parts, ignore_index=True)
    else:
        df["g"], df["m"] = 0.0, 0.0
    gb = g_behavioral(df)
    df = df.merge(gb[["model", "cue_id", "topic", "h_dir", "g_beh"]], on=["model", "cue_id", "topic", "h_dir"],
                  how="left")
    df["m_beh"] = df.b * df.g_beh.fillna(0.0)
    from .paths import DATA_DIR
    svy = rd / "survey_stance.csv"
    svy = svy if svy.exists() else DATA_DIR / "survey_stance.csv"
    if svy.exists():
        s = g_survey(pd.read_csv(svy))
        df = df.merge(s[["topic", "h_dir", "group", "g_svy"]], on=["topic", "h_dir", "group"], how="left")
        df["m_svy"] = df.b * df.g_svy.fillna(0.0)
    return attach_survey_gap(df, load_topics())
