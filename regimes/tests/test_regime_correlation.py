import numpy as np
import pandas as pd
import pytest

from regimes import correlation


@pytest.fixture
def frame():
    idx = pd.date_range("2000-01-03", periods=200, freq="B")
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(0, 1, len(idx)))
    b = 2.0 * a + rng.normal(0, 0.1, len(idx))
    c = -a + rng.normal(0, 0.1, len(idx))
    return pd.DataFrame({"a": a, "b": b, "c": c}, index=idx)


def test_daily_changes(frame):
    ch = correlation.daily_changes(frame)
    assert ch.iloc[0].isna().all()
    assert np.allclose(ch["a"].iloc[1:].to_numpy(), np.diff(frame["a"].to_numpy()))


def test_rolling_correlation_of_series_with_itself_is_one(frame):
    rc = correlation.rolling_correlation(frame["a"], frame["a"], window=30)
    assert np.allclose(rc.dropna().to_numpy(), 1.0)


def test_regime_matrices_split_by_label(frame):
    idx = frame.index
    labels = pd.Series([0] * 100 + [1] * 100, index=idx)
    mats = correlation.regime_correlation_matrices(frame, labels)
    assert set(mats) == {0, 1}
    assert mats[0].shape == (3, 3)


def test_corr_stability_zero_for_identical():
    idx = ["a", "b", "c"]
    m = pd.DataFrame([[1, 0.5, 0.2], [0.5, 1, 0.1], [0.2, 0.1, 1]], index=idx, columns=idx)
    assert correlation.corr_stability(m, m) == pytest.approx(0.0)


def test_corr_stability_positive_when_flipped():
    idx = ["a", "b"]
    m1 = pd.DataFrame([[1, 0.8], [0.8, 1]], index=idx, columns=idx)
    m2 = pd.DataFrame([[1, -0.8], [-0.8, 1]], index=idx, columns=idx)
    assert correlation.corr_stability(m1, m2) == pytest.approx(1.6)
