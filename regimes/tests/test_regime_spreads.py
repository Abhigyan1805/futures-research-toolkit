import numpy as np
import pandas as pd
import pytest

from regimes import config, spreads


@pytest.fixture
def levels():
    idx = pd.date_range("2000-01-03", periods=5, freq="B")
    return pd.DataFrame(
        {
            "DGS3MO": [4.0, 4.1, 4.2, 4.3, 4.4],
            "DGS1": [4.5, 4.6, 4.7, 4.8, 4.9],
            "DGS2": [5.0, 5.0, 5.0, 5.0, 5.0],
            "DGS5": [5.5, 5.4, 5.3, 5.2, 5.1],
            "DGS10": [6.0, 6.0, 6.0, 6.0, 6.0],
        },
        index=idx,
    )


def test_calendar_spread_in_basis_points(levels):
    s = spreads.calendar_spread(levels, "DGS1", "DGS2")
    assert s.name == "2y-1y"
    assert np.allclose(s.to_numpy(), [50.0, 40.0, 30.0, 20.0, 10.0])


def test_butterfly_is_second_difference(levels):
    f = spreads.butterfly(levels, "DGS3MO", "DGS1", "DGS2")
    assert f.name == "fly 2*1y-3m-2y"
    # 2*1y - 3m - 2y in bps
    expected = (2 * levels["DGS1"] - levels["DGS3MO"] - levels["DGS2"]) * 100
    assert np.allclose(f.to_numpy(), expected.to_numpy())


def test_build_spread_basket_columns(levels):
    basket = spreads.build_spread_basket(levels)
    assert list(basket.columns) == [
        config.spread_name(n, f) for n, f in config.CALENDAR_PAIRS
    ] + [config.butterfly_name(n, m, f) for n, m, f in config.BUTTERFLY_TRIPLES]
    assert basket.shape[1] == len(config.CALENDAR_PAIRS) + len(config.BUTTERFLY_TRIPLES)


def test_parallel_curve_has_zero_slope_and_curvature():
    idx = pd.date_range("2000-01-03", periods=3, freq="B")
    flat = pd.DataFrame({t: [3.0, 3.0, 3.0] for t in config.CURVE_TENORS}, index=idx)
    basket = spreads.build_spread_basket(flat)
    assert np.allclose(basket.to_numpy(), 0.0)
