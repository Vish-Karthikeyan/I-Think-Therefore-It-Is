import pandas as pd
import pytest

from idea.audit import MMLU_PRO_SUBJECTS, all_parts, merge_parts, parse_part


def test_parse_parts_cover_every_subject_once():
    for n in (1, 2, 4, 7, 14):
        seen = []
        for i in range(n):
            cond, what, tasks = parse_part(f"steered/mmlu_pro:{i}of{n}")
            assert cond == "steered" and what == f"mmlu_pro:{i}of{n}"
            seen += tasks
        assert sorted(seen) == sorted(f"mmlu_pro_{s}" for s in MMLU_PRO_SUBJECTS)
    assert parse_part("random/bbq") == ("random", "bbq", ["bbq"])
    assert parse_part("unsteered/syceval") == ("unsteered", "syceval", None)
    for bad in ("x/mmlu_pro:4of4", "x/mmlu_pro:0of15", "x/other"):
        with pytest.raises(ValueError):
            parse_part(bad)


def test_all_parts_count():
    p = all_parts(["unsteered", "steered", "random"], 4)
    assert len(p) == 18 and len(set(p)) == 18


def test_merge_weighted_mmlu_and_one_row_per_condition():
    rows = [dict(condition="steered", part="syceval", syceval_flip=0.2),
            dict(condition="steered", part="bbq", **{"bbq/acc": 0.6}),
            dict(condition="steered", part="mmlu_pro:0of2", **{"mmlu_pro_biology/exact_match,custom-extract": 0.5,
                                                               "mmlu_pro_biology/n": 100}),
            dict(condition="steered", part="mmlu_pro:1of2", **{"mmlu_pro_law/exact_match,custom-extract": 0.2,
                                                               "mmlu_pro_law/n": 300}),
            dict(condition="unsteered", part="bbq", **{"bbq/acc": 0.7})]
    m = merge_parts(pd.DataFrame(rows)).set_index("condition")
    assert list(m.index) == ["steered", "unsteered"]
    assert m.loc["steered", "mmlu_pro/exact_match,custom-extract"] == pytest.approx((0.5 * 100 + 0.2 * 300) / 400)
    assert m.loc["steered", "mmlu_pro_subjects_found"] == 2 and m.loc["unsteered", "mmlu_pro_subjects_found"] == 0
    assert m.loc["steered", "syceval_flip"] == 0.2 and m.loc["steered", "bbq/acc"] == 0.6
