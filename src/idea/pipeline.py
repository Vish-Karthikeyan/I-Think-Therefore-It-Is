"""Stage functions behind the CLI. Each reads a config and writes into a run folder named
by the config's hash: runs/<name>-<hash8>/."""
from __future__ import annotations

import hashlib
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .cues import UNSPECIFIED, CueLibrary, identification_requests, load_cues, score_identification
from .design import DesignConfig, build_trials, summarize_design
from .elicit import Cache, CachedElicitor, Request, make_backend
from .models import RQ3_MODELS, get_model
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
    NFS (FarmShare scratch) is only safe with one writer per file, and pieces run at once."""
    tag = f"_s{shard.replace('/', 'of')}" if shard else ""
    return rd / "cache" / f"{model_key}{tag}.sqlite"


def _elicitor(rd: Path, backend, act_dir: Path | None = None, chunk: int = 256, shard: str | None = None):
    from .mech.capture import ActStore
    store = ActStore(act_dir) if act_dir else None
    cache = Cache(cache_path(rd, backend.spec.key, shard))
    return CachedElicitor(backend, cache, chunk=chunk, act_store=store)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_generate(cfg: dict, pilot_run: str | None = None) -> Path:
    rd = run_dir(cfg)
    dcfg = DesignConfig.from_dict({**(cfg.get("design") or {}), "seed": cfg.get("seed", 0)})
    if pilot_run:
        pr = Path(pilot_run)
        if (pr / "screen.json").exists():
            dcfg.engaged = json.loads((pr / "screen.json").read_text())["engaged"]
        if (pr / "prior_means.json").exists():
            pm = json.loads((pr / "prior_means.json").read_text())
            dcfg.prior_means = {tuple(json.loads(k)): v for k, v in pm.items()}
    topics = _topics(cfg)
    if pilot_run and (Path(pilot_run) / "screen.json").exists():
        keep = set(json.loads((Path(pilot_run) / "screen.json").read_text())["keep"])
        topics = [t for t in topics if t.topic_id in keep]
    items, trials = build_trials(topics, _cues(cfg), dcfg)
    trials["tier1_low_g"] = mark_tier1_low_g(trials, pilot_run)
    items.to_parquet(rd / "items.parquet")
    trials.to_parquet(rd / "trials.parquet")
    summary = summarize_design(trials)
    summary.to_csv(rd / "design_summary.csv")
    print(summary)
    print(f"{len(trials)} trials, {trials.item_id.nunique()} items -> {rd}")
    return rd


def mark_tier1_low_g(trials: pd.DataFrame, pilot_run: str | None) -> pd.Series:
    """True for trials in the Tier 1 robustness subset (deviation 2026-11-11): every Tier 2
    trial, Unspecified-cue Tier 1 trials, and Tier 1 trials whose (topic, cue) pair had
    |g| < 0.25 in the pilot screen. Without a pilot screen every trial is marked True."""
    low = None
    if pilot_run and (Path(pilot_run) / "screen.json").exists():
        low = json.loads((Path(pilot_run) / "screen.json").read_text()).get("tier1_low_g")
    if low is None:
        return pd.Series(True, index=trials.index)
    pairs = {tuple(x) for x in low}
    in_pair = pd.Series([(t, c) in pairs for t, c in zip(trials.topic, trials.cue_id)], index=trials.index)
    return (trials.tier != 1) | (trials.group == UNSPECIFIED) | in_pair


def stage_cues(cfg: dict, model_key: str, threshold: float = 0.7) -> pd.DataFrame:
    rd = run_dir(cfg)
    lib = _cues(cfg)
    backend = _backend(cfg, model_key, sample_k=stance_sample_k(cfg, model_key))
    reqs_meta = identification_requests(lib, leakage=True)
    reqs = []
    for m in reqs_meta:
        target_index = m["options"].index(m["group_id"]) if m["group_id"] in m["options"] else None
        reqs.append(Request("choice", "chat", messages=[{"role": "user", "content": m["prompt"]}],
                            letters=m["letters"], meta=dict(call="ident", target_index=target_index,
                                                            ask_dim=m["ask_dim"], cue_dim=m["cue_dim"])))
    res = _elicitor(rd, backend).run(reqs)
    rows = pd.DataFrame(reqs_meta)
    rows["probs"] = [r.probs or [np.nan] * len(m["options"]) for r, m in zip(res, reqs_meta)]
    rows["coverage"] = [r.coverage for r in res]
    scored = score_identification(rows, threshold)
    scored["model"] = model_key
    (rd / "cues").mkdir(exist_ok=True)
    rows.drop(columns=["prompt"]).to_parquet(rd / "cues" / f"ident_raw_{model_key}.parquet")
    scored.to_csv(rd / "cues" / f"ident_{model_key}.csv", index=False)
    rate = scored[scored.group_id != UNSPECIFIED].groupby("group_id").passed.mean()
    print(f"vignette pass rate per group ({model_key}):\n{rate.round(2).to_string()}")
    return scored


def stage_stance(cfg: dict, model_key: str, fmt: str = "chat") -> pd.DataFrame:
    from .stance import run_stance, stance_spread_report
    rd = run_dir(cfg)
    backend = _backend(cfg, model_key, sample_k=stance_sample_k(cfg, model_key))
    topics = _topics(cfg)
    if (cfg.get("elicit") or {}).get("stance_trial_topics_only"):   # only topics the trials use
        used = set(pd.read_parquet(rd / "trials.parquet").topic)
        topics = [t for t in topics if t.topic_id in used]
    st = run_stance(_cues(cfg), topics, _elicitor(rd, backend), fmt)
    st["fmt"] = fmt
    (rd / "stance").mkdir(exist_ok=True)
    st.to_parquet(rd / "stance" / f"{model_key}_{fmt}.parquet")
    print(stance_spread_report(st).round(3).to_string())
    return st


def stage_diag(cfg: dict, model_key: str, fmt: str = "chat", shard: str | None = None) -> pd.DataFrame:
    """Evidence diagnostic (exploratory, 2026-11-11): can the model read and use the tables?"""
    from .diagnostics import run_diag, score, select_items
    rd = run_dir(cfg)
    out_dir = rd / "diag"
    out_dir.mkdir(exist_ok=True)
    n = int((cfg.get("diag") or {}).get("n_items", 480))
    items = select_items(pd.read_parquet(rd / "trials.parquet"), n=n, seed=cfg.get("seed", 0))
    items.to_parquet(out_dir / "items.parquet")
    if shard:
        i, k = map(int, shard.split("/"))
        items = items[items.index % k == i]
    backend = _backend(cfg, model_key)
    tag = f"{model_key}_{fmt}" + (f"_s{shard.replace('/', 'of')}" if shard else "")
    # own cache file (…_sdiag.sqlite): diag jobs may run alongside this model's other jobs
    ctag = "diag" + (shard.replace("/", "of") if shard else "")
    df = run_diag(items, _topics(cfg), _elicitor(rd, backend, shard=ctag), fmt)
    df.to_parquet(out_dir / f"{tag}.parquet")
    print(score(df).round(3).to_string())
    return df


def stage_screen(cfg: dict, g_min=0.5, min_dims=2, tier1_max=0.25) -> dict:
    from .stance import check_dim_coverage, screen_topics, tier1_low_g_pairs
    rd = run_dir(cfg)
    st = pd.concat([pd.read_parquet(p) for p in (rd / "stance").glob("*.parquet")])
    st = st[st.fmt == "chat"] if (st.fmt == "chat").any() else st
    rep, engaged = screen_topics(st, g_min, min_dims, tier1_max)
    rep.to_csv(rd / "screen_report.csv", index=False)
    keep = sorted(rep[rep.keep].topic)
    short = check_dim_coverage(engaged)
    low = tier1_low_g_pairs(st, tier1_max)
    out = dict(keep=keep, engaged=engaged, dims_below_min=short, tier1_low_g=low)
    print(f"Tier 1 robustness subset: {len(low)} (topic, cue) pairs with |g| < {tier1_max}")
    (rd / "screen.json").write_text(json.dumps(out, indent=1))
    print(f"kept {len(keep)} topics; dims with < 3 engaged topics: {short or 'none'}")
    return out


def _reassign_failed_vignettes(trials: pd.DataFrame, rd: Path, model_key: str, lib: CueLibrary, seed: int):
    p = rd / "cues" / f"ident_{model_key}.csv"
    if not p.exists():
        return trials, 0
    sc = pd.read_csv(p)
    failed = set(sc[~sc.passed].cue_id)
    if not failed:
        return trials, 0
    rng = random.Random(seed)
    t = trials.copy()
    drop = []
    for i in t.index[t.cue_id.isin(failed)]:
        pool = [c.cue_id for c in lib.select(group_id=t.at[i, "group"], cue_type="vignette") if c.cue_id not in failed]
        if pool:
            t.at[i, "cue_id"] = rng.choice(pool)
        else:
            drop.append(i)
    return t.drop(index=drop), len(drop)


def stage_bcheck(cfg: dict, model_key: str) -> pd.DataFrame:
    """Manipulation check for the belief-strength experiment (exploratory, 2026-11-16): for every topic,
    direction and hedge level, (1) which side the writer is on (the stance question, about the claim the
    sentence asserts) and (2) how certain the writer sounds. 48 topics x 2 directions x 3 levels x 2 questions."""
    from .elicit import Request
    from .prompts import BELIEF_LEVELS, belief_certainty_prompt, hedge_belief, stance_prompt
    rd = run_dir(cfg)
    backend = _backend(cfg, model_key, sample_k=stance_sample_k(cfg, model_key))
    reqs, rows = [], []
    for tp in _topics(cfg):
        for d in (1, -1):
            for lvl in (cfg.get("bcheck_levels") or BELIEF_LEVELS):
                s = hedge_belief(tp.belief(d), lvl, key=f"{tp.topic_id}:{d}")
                for q, msgs in (("side", stance_prompt(s, tp, d)), ("certainty", belief_certainty_prompt(s))):
                    r = Request("number", "chat", messages=msgs)
                    r.answer_format = "text"
                    reqs.append(r)
                    rows.append(dict(model=model_key, topic=tp.topic_id, tier=tp.tier, direction=d, level=lvl,
                                     question=q, sentence=s))
    res = _elicitor(rd, backend).run(reqs)
    out = pd.DataFrame(rows).assign(p=[x.p for x in res], coverage=[x.coverage for x in res])
    (rd / "bcheck").mkdir(exist_ok=True)
    out.to_parquet(rd / "bcheck" / f"{model_key}.parquet")
    print(out.groupby(["question", "level"]).p.agg(["mean", "min", "count"]).round(3).to_string())
    return out


def stage_elicit(cfg: dict, model_key: str, fmt: str = "chat", shard: str | None = None,
                 capture: bool | None = None, limit: int | None = None, dry_run: bool = False) -> Path:
    from .runner import run_trials
    rd = run_dir(cfg)
    trials = pd.read_parquet(rd / "trials.parquet")
    lib = _cues(cfg)
    trials, n_drop = _reassign_failed_vignettes(trials, rd, model_key, lib, cfg.get("seed", 0))
    if n_drop:
        print(f"dropped {n_drop} trials: no passing vignette for their group ({model_key})")
    spec = get_model(model_key)
    reduced = (cfg.get("reduced_models") or [])
    if model_key in reduced or f"{model_key}:{fmt}" in reduced:  # e.g. OLMo base/SFT/DPO: labels only
        trials = trials[trials.cue_type == "label"]
    sub = ((cfg.get("elicit") or {}).get("subset_by_model") or {}).get(model_key)
    if sub:  # exploratory: fixed subset by trial-id hash, independent of --shard (index-based)
        i, n = map(int, sub.split("/"))
        h = trials.trial_id.map(lambda t: int(hashlib.md5(t.encode()).hexdigest(), 16) % n)
        trials = trials[h == i]
    ecfg = cfg.get("elicit") or {}
    if model_key in (ecfg.get("belief_trials_only") or []):  # belief-strength experiment: b != 0 only
        trials = trials[trials.b != 0]
    if model_key in (ecfg.get("no_belief_trials_only") or []):  # fresh no-belief baseline: b == 0 only
        trials = trials[trials.b == 0]
    ts = (ecfg.get("trial_subset_by_model") or {}).get(model_key)
    if ts:  # fixed random subset per tier (budget-limited API models, 2026-11-15); same draw every run
        rs = np.random.default_rng(int(ts.get("seed", 0)))
        parts = []
        for tier, n in ((2, ts.get("n_tier2")), (1, ts.get("n_tier1"))):
            pool = trials[trials.tier == tier]
            parts.append(pool.iloc[np.sort(rs.permutation(len(pool))[:int(n)])] if n is not None else pool)
        trials = pd.concat(parts).sort_index()
    rt = (ecfg.get("retest_subset_by_model") or {}).get(model_key)
    if rt:  # ask the paraphrased prior (test-retest) on a fixed subset only
        rs = np.random.default_rng(int(cfg.get("seed", 0)) + 99)
        keep = set(trials.trial_id.iloc[rs.permutation(len(trials))[:int(rt)]])
        trials = trials.assign(retest=trials.trial_id.isin(keep))
    if shard:
        i, n = map(int, shard.split("/"))
        trials = trials[trials.index % n == i]
    if limit:
        trials = trials.head(limit)
    cap_cfg = cfg.get("capture") or {}
    if capture is None:
        capture = model_key in cap_cfg.get("models", RQ3_MODELS) and fmt == "chat" and spec.open_weights
    over = {"capture_layers": cap_cfg.get("layers", list(range(8, 28)))} if capture else {}
    calibration = spec.open_weights and spec.readout == "dist" and model_key not in (ecfg.get("no_calibration") or [])
    if dry_run:
        return estimate(cfg, trials, model_key, fmt, rd, calibration=calibration)
    backend = _backend(cfg, model_key, **over)
    act_dir = rd / "acts" / f"{model_key}_{fmt}" if capture else None
    el = _elicitor(rd, backend, act_dir, chunk=(cfg.get("elicit") or {}).get("chunk", 256), shard=shard)
    from .elicit import DeadlineReached
    try:
        res = run_trials(trials, _topics(cfg), lib, el, fmt, capture=capture, calibration=calibration)
    except DeadlineReached as e:
        print(f"DEADLINE: {e}")
        raise
    (rd / "responses").mkdir(exist_ok=True)
    tag = f"{model_key}_{fmt}" + (f"_s{shard.replace('/', 'of')}" if shard else "")
    out = rd / "responses" / f"{tag}.parquet"
    res.to_parquet(out)
    if hasattr(backend, "usage"):
        print("API usage:", backend.usage)
    print(f"wrote {len(res)} rows -> {out}")
    return out


def estimate(cfg, trials, model_key, fmt, rd, calibration: bool | None = None):
    """Count unique model calls and estimate tokens / API cost (OpenRouter list prices)."""
    from .runner import calls_for, make_request
    spec = get_model(model_key)
    tmap = {t.topic_id: t for t in _topics(cfg)}
    lib = _cues(cfg)
    if calibration is None:
        calibration = spec.open_weights and spec.readout == "dist"
    seen, chars = set(), 0
    for row in trials.itertuples(index=False):
        for call in calls_for(row, calibration):
            r = make_request(row, tmap, lib, fmt, call, variant=spec.prompt_variant, system=spec.system_prompt,
                             belief_strength=spec.belief_strength)
            k = r.prompt_repr()
            if k not in seen:
                seen.add(k)
                chars += len(k)
    n = len(seen)
    k = sample_k_for(cfg, model_key) if spec.readout == "sample" else 1
    tokens_in = chars / 4 * k
    info = dict(model=model_key, fmt=fmt, trials=len(trials), unique_calls=n, api_requests=n * k,
                approx_input_tokens=int(tokens_in))
    if spec.backend == "openrouter":
        try:
            import httpx
            ms = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
            pr = next(m for m in ms if m["id"] == spec.openrouter_id)["pricing"]
            info["approx_cost_usd"] = round(tokens_in * float(pr["prompt"]) + n * k * 4 * float(pr["completion"]), 2)
        except Exception as e:  # noqa: BLE001
            info["price_lookup_error"] = repr(e)[:200]
    print(info)
    return info


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


def _write(results: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False)


def stage_pilot_report(cfg: dict) -> dict:
    from .controls import omega_histogram, pilot_report, prior_means
    from .stance import stance_spread_report
    rd = run_dir(cfg)
    df = load_responses(rd)
    rep = pilot_report(df)
    out = rd / "results" / "pilot"
    _write(rep, out)
    omega_histogram(df, out / "omega_hist.png")
    pm = prior_means(df)
    (rd / "prior_means.json").write_text(json.dumps({json.dumps(list(map(_jsonable, k))): v for k, v in pm.items()}))
    if (rd / "stance").exists():
        st = pd.concat([pd.read_parquet(f) for f in (rd / "stance").glob("*.parquet")])
        stance_spread_report(st).to_csv(out / "stance_spread.csv", index=False)
    for k, v in rep.items():
        print(f"== {k}\n{v.round(3).to_string() if isinstance(v, pd.DataFrame) else v}")
    return rep


def _jsonable(x):
    return x.item() if hasattr(x, "item") else x


def stage_fit(cfg: dict, engine: str = "ols", rq2_folds: int | None = None, n_boot: int = 1000) -> dict:
    from .analysis import rq1, rq2, rq2b, secondary
    from .controls import pilot_report
    rd = run_dir(cfg)
    df = load_responses(rd)
    df.to_parquet(rd / "analysis_frame.parquet")
    out = rd / "results"
    res = rq1.run_all(df, engine)
    _write(res, out / "rq1")
    _write(pilot_report(df), out / "controls")
    print(res["per_model"].round(3).to_string())
    summary = {"rq1": res["per_model"].to_dict("records")}
    for (model, fmt), _ in df.groupby(["model", "fmt"]):
        tag = f"{model}_{fmt}"
        try:
            sec = secondary.run_secondary(df, model, fmt)
            rows = [dict(spec="spline", f_stat=sec.get("spline", {}).get("f_stat"), f_p=sec.get("spline", {}).get("f_p"))]
            mix = sec["mixture"]
            rows.append(dict(spec="mixture", **{f"updated_{k}": v for k, v in mix["updated"].items()},
                             mean_p_ignore=mix["mean_p_ignore"]))
            (out / "secondary").mkdir(exist_ok=True, parents=True)
            pd.DataFrame(rows).to_csv(out / "secondary" / f"{tag}_summary.csv", index=False)
            mix["p_ignore_coefs"].to_csv(out / "secondary" / f"{tag}_mixture_p_ignore.csv")
            lp = sec["latent_prior"]
            pd.DataFrame(dict(est=lp["params"], se=lp["se"])).to_csv(out / "secondary" / f"{tag}_latent_prior.csv")
            if "spline" in sec:
                sec["spline"]["curves"].to_csv(out / "secondary" / f"{tag}_spline_curves.csv", index=False)
        except Exception as e:  # noqa: BLE001
            print(f"secondary failed for {tag}: {e!r}")
        if fmt != "chat":
            continue
        r2 = rq2.evaluate(df, model, fmt, max_folds=rq2_folds)
        (out / "rq2").mkdir(parents=True, exist_ok=True)
        r2["folds"].to_csv(out / "rq2" / f"{tag}_folds.csv", index=False)
        r2["summary"].to_csv(out / "rq2" / f"{tag}_summary.csv", index=False)
        summary.setdefault("rq2", {})[tag] = r2["success"]
        print(f"RQ2 {tag}: {r2['success'] or r2.get('note')}")
        b = rq2b.run(df, model, fmt, r2["success"], n_boot=n_boot)
        (out / "rq2b").mkdir(parents=True, exist_ok=True)
        if b.get("empty"):
            print(f"RQ2b {tag}: {b['note']}")
            continue
        if b.get("gated_out"):
            print(f"RQ2b gated out for {tag} (RQ2 did not beat baselines on held-out topics)")
            continue
        b["contrasts"].to_csv(out / "rq2b" / f"{tag}_contrasts.csv", index=False)
        b["table"].to_csv(out / "rq2b" / f"{tag}_table.csv")
        b["cv"].to_csv(out / "rq2b" / f"{tag}_cv_rank.csv", index=False)
        rq2b.signature_map(b["contrasts"], b["bootstrap"]["contrasts"], out / "rq2b" / f"{tag}_map.png", title=tag)
        if b["components"] is not None:
            b["components"][1].to_csv(out / "rq2b" / f"{tag}_residual_loadings.csv")
            summary.setdefault("rq2b", {})[tag] = dict(rank=b["rank"],
                                                       alignment=b["bootstrap"]["alignment_median"].tolist())
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def stage_steer(cfg: dict, model_key: str, fmt: str = "chat", max_eval: int | None = None) -> dict:
    """RQ3 + RQ3b on an open-weight model: masking probes, directions, sweep on validation
    topics, one evaluation on held-out test topics with controls, subspace comparisons,
    other-bias directions, signature-guided steering."""
    from .analysis import rq1
    from .mech import probe, steer, subspace
    from .mech.capture import ActStore
    rd = run_dir(cfg)
    scfg = cfg.get("steer") or {}
    df = pd.read_parquet(rd / "analysis_frame.parquet") if (rd / "analysis_frame.parquet").exists() else load_responses(rd)
    d = df[(df.model == model_key) & (df.fmt == fmt) & df.key_post.notna()].reset_index(drop=True)
    store = ActStore(rd / "acts" / f"{model_key}_{fmt}")
    layers = scfg.get("layers") or store.layers()
    cacheH = {}

    def get_H(l):
        if l not in cacheH:
            cacheH[l] = store.get(list(d.key_post), l)
        return cacheH[l]

    out_dir = rd / "results" / "rq3" / f"{model_key}_{fmt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # RQ1 gate: is there anything to fix?
    prim = rq1.primary(rq1.analysis_frame(df[df.fmt == fmt], 2, model_key))
    exploratory = not (prim["holm"]["delta_b"] or prim["holm"]["theta_cong"])
    sigs = None
    sig_path = rd / "results" / "rq2b" / f"{model_key}_{fmt}_contrasts.csv"
    if sig_path.exists():
        sigs = pd.read_csv(sig_path)
    # Masking probe
    mask = probe.run_masking(get_H, d, layers, seed=cfg.get("seed", 0), signatures=sigs)
    mask["sweep"].to_csv(out_dir / "probe_layer_sweep.csv", index=False)
    for k in ("overall", "by_dimension", "by_cue_type", "by_signature"):
        if k in mask:
            mask[k].to_csv(out_dir / f"masking_{k}.csv", index=False)
    print(f"probe layer {mask['layer']} test R2={mask['r2_test']:.3f}; masking={mask['overall'].masking_index.iloc[0]:.3f}")
    # Directions
    train_t, val_t, test_t = probe.split_topics(d.topic.unique(), scfg.get("test_frac", 0.3),
                                                scfg.get("val_frac", 0.2), cfg.get("seed", 0) + 1)
    dtrain = d[d.topic.isin(train_t)].reset_index(drop=True)
    H_index = d.index[d.topic.isin(train_t)]

    def get_H_train(l):
        return get_H(l)[H_index]

    ks = tuple(scfg.get("topic_k", (4, 8, 16)))
    # A resumed run (requeue after the deadline) reuses the directions saved by the first run, so
    # its steered requests hash to the same cache keys; set steer.recompute_directions to rebuild.
    dpath = out_dir / "directions"
    if Path(f"{dpath}.npz").exists() and Path(f"{dpath}.json").exists() and not scfg.get("recompute_directions"):
        dirs = steer.load_directions(dpath)
        print(f"loaded saved directions from {dpath}.npz (layers {sorted(dirs)})")
    else:
        dirs = steer.build_directions(dtrain, get_H_train, layers, ks, cfg.get("seed", 0))
        steer.save_directions(dirs, dpath)
    # per-dimension angles (theta + tau_k credibly nonzero only)
    thdim = rq1.theta_by_dimension(rq1.analysis_frame(df[df.fmt == fmt], 2, model_key))
    credible = set(thdim[thdim.p < 0.05].dimension)
    kmain = scfg.get("k_main", 8)
    ang_rows = []
    for l, rec in dirs.items():
        per = {k: v for k, v in rec[f"per_dim_hat_k{kmain}"].items() if k in credible}
        a = subspace.pairwise_directions(per)
        if not a.empty:
            ang_rows.append(a.assign(layer=l))
    if ang_rows:
        pd.concat(ang_rows).to_csv(out_dir / "per_dimension_angles.csv", index=False)
    # Steering sweep (needs the live model)
    backend = _backend(cfg, model_key)
    lib, topics = _cues(cfg), _topics(cfg)
    trials = pd.read_parquet(rd / "trials.parquet")
    trials = trials[trials.tier.isin([1, 2])]
    n_eval = max_eval or scfg.get("n_eval", 3000)
    rng = np.random.default_rng(cfg.get("seed", 0))
    t1 = trials[trials.tier == 1]

    def sample(sub, n):
        return sub.iloc[rng.permutation(len(sub))[:n]]

    val_trials = pd.concat([sample(trials[(trials.tier == 2) & trials.topic.isin(val_t)], n_eval // 2),
                            sample(t1, n_eval // 6)])
    test_trials = pd.concat([sample(trials[(trials.tier == 2) & trials.topic.isin(test_t)], n_eval),
                             sample(t1, n_eval // 3)])
    st_files = list((rd / "stance").glob(f"{model_key}_{fmt}.parquet"))
    stance = pd.read_parquet(st_files[0]) if st_files else None
    factory = lambda b: _elicitor(rd, b)  # noqa: E731
    sweep_layers = scfg.get("sweep_layers") or layers[::max(1, len(layers) // 5)]
    missing = [l for l in sweep_layers if l not in dirs]
    sweep_layers = [l for l in sweep_layers if l in dirs] or sorted(dirs)
    if not dirs:
        note = ("no steering directions: no training-topic stratum had both caved and resisted "
                "contradicted trials, so d_bias is undefined; steering skipped")
        (out_dir / "STEERING_SKIPPED.txt").write_text(note + "\n")
        print(note)
        return {"skipped": note}
    if missing:
        print(f"no directions at sweep layers {missing}; sweeping {sweep_layers} instead")
    cand = {}
    for name in [f"hat_k{kmain}", f"prompt_hat_k{kmain}", "random", f"topic_only_k{kmain}"]:
        cand[name] = {l: dict(vec=dirs[l][name], scale=dirs[l]["scale"][name]) for l in sweep_layers
                      if l in dirs and dirs[l].get(name) is not None}
    base_val = steer.evaluate_steering(backend, factory, val_trials, topics, lib, fmt, sweep_layers[0], np.ones(1), 0.0,
                                       stance=stance)
    sw = steer.sweep(backend, factory, val_trials, topics, lib, fmt, sweep_layers, {f"hat_k{kmain}": cand[f"hat_k{kmain}"]},
                     multiples=tuple(scfg.get("multiples", (0.5, 1, 2, 4, 8))),
                     modes=tuple(scfg.get("modes", ("all", "last"))), unsteered=base_val, stance=stance)
    sw.to_csv(out_dir / "sweep_validation.csv", index=False)
    ok = sw[sw.success] if sw.success.any() else sw
    gap_col = "t2_contra_gap_after"
    best = ok.loc[ok[gap_col].abs().idxmin()] if gap_col in ok else ok.iloc[0]
    L, psi_mult, mode = int(best.layer), float(best.multiple), best["mode"]
    print(f"chosen layer={L} multiple={psi_mult} mode={mode}")
    # Held-out test evaluation: main direction, controls, per-dimension, amplification
    base_test = steer.evaluate_steering(backend, factory, test_trials, topics, lib, fmt, L, np.ones(1), 0.0, stance=stance)
    s0 = steer.slopes(base_test)
    spread0 = steer.delta_j_spread(base_test)
    dims = sorted(set(base_test.dimension) - {"none"})
    rows = []
    tests = {f"hat_k{k}": dirs[L][f"hat_k{k}"] for k in ks}
    tests.update({"raw": dirs[L]["raw"], "random": dirs[L]["random"], f"topic_only_k{kmain}": dirs[L][f"topic_only_k{kmain}"]})
    if dirs[L].get(f"prompt_hat_k{kmain}") is not None:
        tests[f"prompt_hat_k{kmain}"] = dirs[L][f"prompt_hat_k{kmain}"]
    for name, vec in tests.items():
        for sign, tag in ((1, ""), (-1, "_amplify")):
            if tag and name != f"hat_k{kmain}":
                continue
            psi = sign * psi_mult * dirs[L]["scale"].get(name, steer.psi_scale(get_H(L), vec))
            res = steer.evaluate_steering(backend, factory, test_trials, topics, lib, fmt, L, vec, psi, mode, stance)
            s1 = steer.slopes(res)
            rep = steer.success(s0, s1, dims=dims, spread_before=spread0, spread_after=steer.delta_j_spread(res))
            rows.append(dict(direction=name + tag, layer=L, psi=psi, mode=mode,
                             **{k: v for k, v in rep.items() if not isinstance(v, dict)},
                             **{f"slope_{k}": v for k, v in s1.items()}))
    for k, vec in dirs[L][f"per_dim_hat_k{kmain}"].items():
        if k not in credible:
            continue
        psi = psi_mult * steer.psi_scale(get_H(L), vec)
        res = steer.evaluate_steering(backend, factory, test_trials, topics, lib, fmt, L, vec, psi, mode, stance)
        rep = steer.success(s0, steer.slopes(res), dims=dims)
        rows.append(dict(direction=f"per_dim_{k}", layer=L, psi=psi, mode=mode,
                         **{kk: v for kk, v in rep.items() if not isinstance(v, dict)}))
    test_rep = pd.DataFrame(rows)
    test_rep.to_csv(out_dir / "steering_test.csv", index=False)
    print(test_rep[["direction", "success"]].to_string())
    summary = dict(exploratory=exploratory, layer=L, multiple=psi_mult, mode=mode, probe_layer=mask["layer"],
                   probe_r2=mask["r2_test"], masking=float(mask["overall"].masking_index.iloc[0]))
    # Other-bias directions and subspace overlap
    if scfg.get("other_biases", True) and get_model(model_key).backend == "hf":
        from .mech.other_biases import bias_directions, run_contrast_sets
        ob_backend = _backend(cfg, model_key, capture_layers=[L])
        ob_store_dir = rd / "acts" / f"{model_key}_other_biases"
        el = _elicitor(rd, ob_backend, ob_store_dir)
        cs = run_contrast_sets(el, el.act_store, [L], n_mmlu=scfg.get("n_mmlu", 400), seed=cfg.get("seed", 0))
        cs.to_parquet(out_dir / "other_bias_contrast_sets.parquet")
        neutral = (dtrain.b == 0).values
        U = steer.topic_subspace(get_H_train(L)[neutral], kmain, cfg.get("seed", 0))
        bd = bias_directions(cs, el.act_store, L, U)
        bd["idea_pooled"] = dirs[L][f"hat_k{kmain}"]
        subspace.pairwise_directions(bd).to_csv(out_dir / "bias_angles.csv", index=False)
    # RQ3b
    if sigs is not None and scfg.get("rq3b", True):
        sd = steer.signature_directions(dtrain, get_H_train(L), sigs, train_t)
        spec_res = {}
        for name, rec in sd.items():
            psi = psi_mult * steer.psi_scale(get_H(L), rec["vec"])
            for target_sig, trows in sigs.groupby("signature"):
                sub = test_trials[test_trials.group.isin(set(trows.group))]
                if len(sub) < 100:
                    continue
                b0 = steer.slopes(base_test[base_test.group.isin(set(trows.group))])
                res = steer.evaluate_steering(backend, factory, sub, topics, lib, fmt, L, rec["vec"], psi, mode, stance)
                spec_res[(name, target_sig)] = steer.success(b0, steer.slopes(res), hedging=rec["kind"] == "hedging")
        steer.specificity_matrix(spec_res).to_csv(out_dir / "rq3b_specificity.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def stage_audit(cfg: dict, model_key: str, fmt: str = "chat", l2d_path: str | None = None,
                part: str | None = None) -> pd.DataFrame:
    """part (optional): run one condition x component only, e.g. 'steered/mmlu_pro:1of4' (see
    audit.parse_part), written to audit_parts/; `idea audit-merge` combines the parts."""
    from .audit import run_suite
    from .mech.other_biases import mmlu_items
    rd = run_dir(cfg)
    out_dir = rd / "results" / "rq3" / f"{model_key}_{fmt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = {"unsteered": []}
    if (out_dir / "summary.json").exists():
        summ = json.loads((out_dir / "summary.json").read_text())
        vecs = np.load(out_dir / "directions.npz")
        L, mult, mode = summ["layer"], summ["multiple"], summ["mode"]
        k = (cfg.get("steer") or {}).get("k_main", 8)
        vec, rnd = vecs[f"L{L}/hat_k{k}"], vecs[f"L{L}/random"]
        meta = json.loads((out_dir / "directions.json").read_text())
        scale = meta[f"L{L}/scale"][f"hat_k{k}"]
        conds.update({"steered": [(L, vec, mult * scale, mode)], "random": [(L, rnd, mult * scale, mode)]})
    else:
        print("no steering summary (steer was skipped or not run): auditing the unsteered model only")
    backend = _backend(cfg, model_key)
    acfg = cfg.get("audit") or {}
    items = mmlu_items(acfg.get("n_syceval", 300), seed=cfg.get("seed", 0) + 7, split="validation")
    if part:
        from .audit import parse_part, run_part
        cond = parse_part(part)[0]
        if cond not in conds:
            raise ValueError(f"condition {cond!r} not available (have {list(conds)})")
        pdir = out_dir / "audit_parts"
        pdir.mkdir(exist_ok=True)
        rec = run_part(backend, conds[cond], part, items, acfg.get("lm_eval_limit", 500), pdir)
        res = pd.DataFrame([rec])
        res.to_csv(pdir / f"{part.replace('/', '__').replace(':', '_')}.csv", index=False)
        print(res.T.to_string())
        return res
    res = run_suite(backend, conds, items, l2d_path or acfg.get("l2d_path"), acfg.get("lm_eval_limit", 500),
                    tuple(acfg.get("tasks", ("mmlu_pro", "bbq"))))
    res.to_csv(out_dir / "audit.csv", index=False)
    print(res.T.to_string())
    return res


def stage_audit_merge(cfg: dict, model_key: str, fmt: str = "chat") -> pd.DataFrame:
    from .audit import merge_parts
    out_dir = run_dir(cfg) / "results" / "rq3" / f"{model_key}_{fmt}"
    files = sorted((out_dir / "audit_parts").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no audit parts in {out_dir / 'audit_parts'}")
    res = merge_parts(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    res.to_csv(out_dir / "audit.csv", index=False)
    print(f"merged {len(files)} parts")
    print(res.T.to_string())
    return res


def stage_power(cfg: dict, model_key: str | None = None, reps: int = 200) -> pd.DataFrame:
    from .analysis import power
    rd = run_dir(cfg)
    kw = {}
    if model_key:
        kw = power.pilot_inputs(load_responses(rd), model_key)
        print("pilot inputs:", kw)
    g = power.grid(reps=reps, **kw)
    g.to_csv(rd / f"power_{model_key or 'default'}.csv", index=False)
    print(g.to_string())
    return g
