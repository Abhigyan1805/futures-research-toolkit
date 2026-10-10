import pandas as pd
import pytest

from data import load_fred, load_ohlc, sample_fred, sample_ohlc


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("FRT_OFFLINE", "1")


def test_ohlc_fixture_shape():
    df = load_ohlc("ES=F")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.name == "date"
    assert (df["high"] >= df["low"]).all()
    assert df.index.is_monotonic_increasing


def test_ohlc_clip():
    df = load_ohlc("ES=F", "2023-06-01", "2023-06-30")
    assert df.index.min() >= pd.Timestamp("2023-06-01")
    assert df.index.max() <= pd.Timestamp("2023-06-30")


def test_fred_fixture_shape():
    df = load_fred("DGS2")
    assert list(df.columns) == ["DGS2"] and df["DGS2"].dtype == float
    assert sample_fred("DGS2").equals(df)


def test_sample_ohlc_matches_loader():
    assert sample_ohlc("ES=F").equals(load_ohlc("ES=F"))
