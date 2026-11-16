"""RQ2 cells and folds (cell definition revised 2026-11-10)."""
import numpy as np
import pandas as pd

from idea.analysis import rq2


def _trials(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    groups = ["democrat", "republican", "catholic", "atheist"]
    dims = {"democrat": "political", "republican": "political", "catholic": "religion", "atheist": "religion"}
    g = rng.choice(groups, n)
    b = rng.choice([-1, 0, 1], n)
    x = rng.uniform(-3, 3, n)
    d = np.sign(x).astype(int)
    contra = (b * d < 0).astype(float)
    shift = np.where(g == "catholic", -0.3, 0.0) * contra
    lp_b = rng.normal(0, 1, n)
    return pd.DataFrame(dict(group=g, dimension=[dims[v] for v in g], cue_type=rng.choice(["label", "vignette"], n),
                             source=rng.choice(["lab", "tank"], n), b=b, d=d, x=x, topic=rng.choice([f"t{i}" for i in range(12)], n),
                             lp_before=lp_b, lp_after=0.5 * lp_b + (0.7 + shift) * x + rng.normal(0, 0.5, n),
                             g=rng.normal(0, 0.5, n), v_k=0.0, has_v=0.0, heuristic_conflict=False, tier=2,
                             model="m", fmt="chat", excluded=False))


def test_cells_have_finite_se_even_when_some_are_dropped():
    t = _trials()
    t = pd.concat([t, t.iloc[:2].assign(group="lonely", dimension="age")])   # a cell with < 3 trials
    cells = rq2.cell_slopes(t, 0.0, 0.5)
    assert "topic" not in cells.columns and set(rq2.CELL_KEYS) <= set(cells.columns)
    assert np.isfinite(cells.se).all() and (cells.se > 0).all()
    assert "lonely" not in set(cells.group)
    # the planted contradicted-trial discount for catholic shows up in the unpooled cells
    c = cells[(cells.b * cells.d < 0)]
    assert c[c.group == "catholic"].beta.mean() < c[c.group != "catholic"].beta.mean() - 0.15


def test_topic_folds_partition_topics():
    topics = [f"t{i}" for i in range(35)]
    folds = rq2.topic_folds(topics)
    assert len(folds) == rq2.N_TOPIC_FOLDS
    flat = [t for f in folds for t in f]
    assert sorted(flat) == sorted(topics) and max(map(len, folds)) - min(map(len, folds)) <= 1


def test_evaluate_runs_both_splits():
    out = rq2.evaluate(_trials(6000), "m", "chat", ridge_grid=(1e-3, 1e-1))
    assert set(out["folds"].split) == {"held_out_topics", "held_out_dimension"}
    assert out["folds"].groupby("split").size()["held_out_topics"] == rq2.N_TOPIC_FOLDS
    assert {"full_rmse", "no_W_rmse", "mean_rmse"} <= set(out["summary"].columns)
