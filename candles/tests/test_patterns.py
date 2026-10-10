"""Unit tests for the pattern detectors on hand-built candles (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from candles.config import DEFAULT_CONFIG, PATTERN_NAMES
from candles.patterns import detect, label


def make(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (open, high, low, close) tuples with a date index."""
    idx = pd.date_range("2020-01-01", periods=len(bars), freq="D", name="date")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


CFG = DEFAULT_CONFIG.replace(trend_lookback=1)


def test_detect_returns_all_patterns_in_order():
    df = make([(10, 10.5, 9.5, 10.0)] * 3)
    hits = detect(df, CFG)
    assert list(hits.columns) == list(PATTERN_NAMES)
    assert hits.dtypes.eq(bool).all()


def test_plain_candle_matches_nothing():
    df = make([(10.0, 10.4, 9.7, 10.2), (10.2, 10.6, 9.9, 10.4), (10.4, 10.8, 10.1, 10.6)])
    hits = detect(df, CFG)
    assert not hits.to_numpy().any(), hits.sum()


def test_doji():
    df = make([(10.0, 10.5, 9.5, 10.02)])
    assert detect(df, CFG)["doji"].iloc[0]
    assert not detect(make([(10.0, 11.0, 9.9, 10.9)]), CFG)["doji"].iloc[0]


def test_hammer_requires_downtrend():
    up = make([(10.0, 10.2, 9.8, 10.0), (10.0, 10.2, 9.8, 10.5),
               (10.4, 10.55, 9.9, 10.5)])
    # hammer shape: small body near the top, long lower shadow; but prior move up
    assert not detect(up, CFG)["hammer"].iloc[2]
    down = make([(10.5, 10.7, 10.3, 10.5), (10.5, 10.7, 9.9, 10.0),
                 (9.9, 10.05, 9.4, 10.0)])
    assert detect(down, CFG)["hammer"].iloc[2]


def test_inverted_hammer_and_shooting_star_share_shape():
    down = make([(10.5, 10.7, 10.3, 10.5), (10.5, 10.7, 9.9, 10.0),
                 (10.0, 10.6, 9.95, 10.1)])
    up = make([(10.0, 10.2, 9.8, 10.0), (10.0, 10.2, 9.8, 10.6),
               (10.6, 11.2, 10.55, 10.7)])
    d = detect(down, CFG)
    u = detect(up, CFG)
    assert d["inverted_hammer"].iloc[2] and not d["shooting_star"].iloc[2]
    assert u["shooting_star"].iloc[2] and not u["inverted_hammer"].iloc[2]


def test_marubozu():
    bull = make([(10.0, 11.005, 9.995, 11.0)])
    bear = make([(11.0, 11.005, 9.995, 10.0)])
    assert detect(bull, CFG)["marubozu_bullish"].iloc[0]
    assert detect(bear, CFG)["marubozu_bearish"].iloc[0]
    assert not detect(bull, CFG)["marubozu_bearish"].iloc[0]


def test_bullish_and_bearish_engulfing():
    bull = make([(10.5, 10.6, 9.9, 10.0), (9.9, 10.7, 9.85, 10.6)])
    bear = make([(10.0, 10.6, 9.9, 10.5), (10.6, 10.65, 9.8, 9.9)])
    assert detect(bull, CFG)["bullish_engulfing"].iloc[1]
    assert detect(bear, CFG)["bearish_engulfing"].iloc[1]
    assert not detect(bull, CFG)["bearish_engulfing"].iloc[1]


def test_harami():
    bull = make([(11.0, 11.1, 9.95, 10.0), (10.2, 10.7, 10.1, 10.6)])
    bear = make([(10.0, 11.05, 9.9, 11.0), (10.8, 10.9, 10.3, 10.4)])
    assert detect(bull, CFG)["bullish_harami"].iloc[1]
    assert detect(bear, CFG)["bearish_harami"].iloc[1]


def test_morning_and_evening_star():
    morning = make([(11.0, 11.1, 9.95, 10.0), (9.8, 9.95, 9.7, 9.9),
                    (10.1, 10.75, 10.05, 10.7)])
    evening = make([(10.0, 11.05, 9.9, 11.0), (11.1, 11.25, 11.05, 11.15),
                    (10.9, 10.95, 10.25, 10.3)])
    assert detect(morning, CFG)["morning_star"].iloc[2]
    assert detect(evening, CFG)["evening_star"].iloc[2]


def test_three_white_soldiers_and_black_crows():
    soldiers = make([(10.0, 10.65, 9.95, 10.6), (10.5, 11.15, 10.45, 11.1),
                     (11.0, 11.75, 10.95, 11.7)])
    crows = make([(11.7, 11.75, 11.05, 11.1), (11.2, 11.25, 10.45, 10.5),
                  (10.6, 10.65, 9.85, 9.9)])
    assert detect(soldiers, CFG)["three_white_soldiers"].iloc[2]
    assert detect(crows, CFG)["three_black_crows"].iloc[2]


def test_label_every_candle():
    df = make([(10.0, 10.5, 9.5, 10.0), (10.0, 10.4, 9.6, 10.2),
               (10.3, 10.9, 10.2, 10.8)])
    out = label(df, CFG)
    for name in PATTERN_NAMES:
        assert name in out.columns and out[name].dtype == bool
    assert "patterns" in out.columns and "n_patterns" in out.columns
    assert len(out) == len(df)
    assert (out["n_patterns"] >= 0).all()
    # patterns string is consistent with the boolean columns
    for _, row in out.iterrows():
        expected = ",".join(n for n in PATTERN_NAMES if row[n])
        assert row["patterns"] == expected
        assert row["n_patterns"] == len([n for n in PATTERN_NAMES if row[n]])


def test_no_lookahead_trend_uses_past_only():
    # A hammer at the last bar must not be affected by bars after it (there are none),
    # and adding future bars must not change the label of the earlier hammer bar.
    base = make([(10.5, 10.7, 10.3, 10.5), (10.5, 10.7, 9.9, 10.0),
                 (9.9, 10.05, 9.4, 10.0)])
    extended = make([(10.5, 10.7, 10.3, 10.5), (10.5, 10.7, 9.9, 10.0),
                     (9.9, 10.05, 9.4, 10.0), (10.0, 12.0, 9.9, 11.9)])
    a = detect(base, CFG)["hammer"].iloc[2]
    b = detect(extended, CFG)["hammer"].iloc[2]
    assert a == b


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_zero_range_candles_never_match(name):
    # A flat bar (open=high=low=close) must not trigger any pattern.
    df = make([(10.0, 10.0, 10.0, 10.0)] * 4)
    assert not detect(df, CFG)[name].any()
