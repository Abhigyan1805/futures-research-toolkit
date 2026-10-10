"""Unit tests for the statistics layer (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from candles.config import PATTERN_NAMES, PATTERN_SIDES
from candles.significance import (
    benjamini_hochberg,
    binom_directional_p,
    binom_test,
    evaluate,
    forward_returns,
    ttest_1samp,
)


def test_binom_test_known_values():
    assert binom_test(5, 10, 0.5) == pytest.approx(1.0, abs=1e-9)
    assert binom_test(10, 10, 0.5) == pytest.approx(2 / 1024, rel=1e-6)
    assert binom_test(0, 10, 0.5) == pytest.approx(2 / 1024, rel=1e-6)
    # 7 heads in 10 fair flips is not significant
    assert binom_test(7, 10, 0.5) > 0.1
    # 9 heads in 10 is
    assert binom_test(9, 10, 0.5) < 0.05


def test_binom_test_against_base_rate():
    # 60 hits in 100 when the base rate is 50% should be borderline
    assert 0.02 < binom_test(60, 100, 0.5) < 0.09
    # with a 60% base rate it should not be significant
    assert binom_test(60, 100, 0.6) == pytest.approx(1.0, abs=1e-9)


def test_binom_directional_one_sided():
    assert binom_directional_p(9, 10, 0.5) == pytest.approx(11 / 1024, rel=1e-6)
    # symmetric: 5/10 is not evidence in the claimed direction
    assert binom_directional_p(5, 10, 0.5) > 0.5
    # 9 up moves with a 50% up base rate is strong directional evidence
    assert binom_directional_p(90, 100, 0.5) < 1e-10


def test_ttest_known_value():
    t, p = ttest_1samp(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert t == pytest.approx(3.0 / (np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)))
    assert p == pytest.approx(0.01315, abs=2e-4)


def test_ttest_zero_variance():
    t, p = ttest_1samp(np.array([2.0, 2.0, 2.0]), 0.0)
    assert t == float("inf") and p == 0.0
    t2, p2 = ttest_1samp(np.array([2.0, 2.0, 2.0]), 2.0)
    assert t2 == 0.0 and np.isnan(p2)


def test_benjamini_hochberg_known():
    adj = benjamini_hochberg(np.array([0.01, 0.02, 0.03, 0.04, 0.05]))
    assert np.allclose(adj, 0.05)
    adj2 = benjamini_hochberg(np.array([0.001, 0.5]))
    assert np.allclose(adj2, [0.002, 0.5])
    # monotone in rank and never below the raw p
    raw = np.array([0.2, 0.01, 0.4, 0.03])
    adj3 = benjamini_hochberg(raw)
    assert np.all(adj3 >= raw - 1e-12) and np.all(adj3 <= 1.0)


def test_forward_returns_no_lookahead():
    close = pd.Series([1.0, 2.0, 4.0, 8.0],
                      index=pd.date_range("2020-01-01", periods=4, name="date"))
    fwd = forward_returns(close, (1, 2))
    assert fwd["fwd_ret_1"].tolist()[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert np.isnan(fwd["fwd_ret_1"].iloc[-1])
    assert fwd["fwd_ret_2"].tolist()[:2] == pytest.approx([3.0, 3.0])
    assert np.isnan(fwd["fwd_ret_2"].iloc[-1])


def _synthetic_candles(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)),
                      index=pd.date_range("2020-01-01", periods=n, name="date"))
    out = pd.DataFrame({"close": close})
    for name in PATTERN_NAMES:
        out[name] = False
    out.loc[out.index[10], "bullish_engulfing"] = True
    out.loc[out.index[11], "bullish_engulfing"] = True
    for h in (1, 5, 10):
        out[f"fwd_ret_{h}"] = forward_returns(close, (h,))[f"fwd_ret_{h}"]
    return out


def test_evaluate_shape_and_columns():
    candles = _synthetic_candles()
    summary = evaluate(candles, PATTERN_NAMES, PATTERN_SIDES, (1, 5, 10),
                       n_boot=200, seed=0)
    assert len(summary) == len(PATTERN_NAMES) * 3
    for col in ("pattern", "side", "horizon", "n_signals", "hit_rate", "base_rate",
                "edge", "beats_base", "binom_p", "binom_p_dir", "t_p", "boot_p",
                "primary_p", "primary_p_bh", "bh_significant"):
        assert col in summary.columns
    # doji is neutral -> no hit rate / binomial p
    doji = summary[summary["pattern"] == "doji"]
    assert doji["hit_rate"].isna().all() or (doji["n_signals"] == 0).all()
    # adjusted p is never smaller than raw for a finite test
    finite = summary.dropna(subset=["primary_p", "primary_p_bh"])
    assert (finite["primary_p_bh"] >= finite["primary_p"] - 1e-9).all()
