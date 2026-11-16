"""The standalone analysis scripts (analysis/) must reproduce the package's analysis exactly."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
DATASET = ROOT / "datasets" / "main_trials.csv.gz"
pytestmark = pytest.mark.skipif(not DATASET.exists(), reason="datasets not built")


def test_dimensions_match_package():
    from idea.cues import DIMENSIONS, UNSPECIFIED
    common = importlib.import_module("common")
    assert common.DIMENSIONS == DIMENSIONS and common.UNSPECIFIED == UNSPECIFIED


@pytest.fixture(scope="module")
def main_df():
    common = importlib.import_module("common")
    df = common.load_dataset("main")
    return df[df.study == "main"]


def test_null_label_survives_reading(main_df):
    assert (main_df.e == "null").sum() > 0


@pytest.mark.parametrize("model,fmt", [("gpt", "chat"), ("olmo_instruct", "completion"), ("llama", "chat")])
def test_rq1_matches_package(main_df, model, fmt):
    from idea.analysis import rq1
    script = importlib.import_module("02_rq1_primary")
    common = importlib.import_module("common")
    mine = script.primary(common.analysis_frame(main_df, 2, model, fmt))
    pkg = rq1.primary(rq1.analysis_frame(main_df[main_df.fmt == fmt], 2, model))
    assert mine["n"] == pkg["n"]
    assert np.isclose(mine["delta_b"], pkg["delta_b"].est, atol=1e-12)
    assert np.isclose(mine["theta_cong"], pkg["theta_cong"].est, atol=1e-12)
    assert np.isclose(mine["delta_b_se"], pkg["delta_b"].se, atol=1e-12)
    assert mine["holm_delta_b"] == pkg["holm"]["delta_b"]


def test_delta_j_matches_package(main_df):
    from idea.analysis import rq1
    sec = importlib.import_module("03_rq1_secondary")
    common = importlib.import_module("common")
    d_mine = common.analysis_frame(main_df, 2, "gpt", "chat")
    mine, _ = sec.unequal_deference(d_mine)
    pkg, _ = rq1.unequal_deference(rq1.analysis_frame(main_df[main_df.fmt == "chat"], 2, "gpt"))
    j = mine.merge(pkg, on="group", suffixes=("_m", "_p"))
    assert len(j) == 22 and np.allclose(j.delta_m, j.delta_p, atol=1e-12)
