import pandas as pd

from idea.runner import derive


def _frame(fmt, cov):
    return pd.DataFrame(dict(fmt=[fmt], readout=["dist"], cov_prior=[cov], cov_prior_rep=[cov], cov_post=[cov],
                             p_prior=[0.5], p_prior_rep=[0.5], p_post=[0.6], x=[1.0], e=["neutral"], b=[0]))


def test_completion_format_uses_lower_cutoff():
    assert not derive(_frame("completion", 0.6)).excluded.iloc[0]
    assert derive(_frame("completion", 0.4)).excluded.iloc[0]
    assert derive(_frame("completion", 0.6), coverage_min_completion=0.8).excluded.iloc[0]


def test_chat_format_keeps_0_9_rule():
    assert derive(_frame("chat", 0.85)).excluded.iloc[0]
    assert not derive(_frame("chat", 0.95)).excluded.iloc[0]
