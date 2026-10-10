import pandas as pd
import pytest

from regimes import config, data


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("FRT_OFFLINE", "1")


def test_load_curve_offline_returns_all_tenors():
    curve = data.load_curve(start="1900-01-01")
    assert list(curve.columns) == config.CURVE_TENORS
    assert isinstance(curve.index, pd.DatetimeIndex)
    assert curve.index.name == "date"


def test_load_bundle_has_curve_and_policy():
    bundle = data.load_bundle(start="1900-01-01")
    assert set(bundle) == {"curve", "policy"}
    assert not bundle["curve"].empty
    assert set(bundle["policy"].columns).issubset(set(config.POLICY_SERIES))
