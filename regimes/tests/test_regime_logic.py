import numpy as np
import pandas as pd
import pytest

from regimes import config, regimes


@pytest.fixture
def synthetic_basket():
    idx = pd.date_range("2000-01-03", periods=300, freq="B")
    rng = np.random.default_rng(0)
    # first 150 calm (low vol), last 150 stressed (high vol)
    calm = rng.normal(20.0, 0.5, 150)
    stress = rng.normal(20.0, 4.0, 150)
    front = np.concatenate([calm, stress])
    belly = np.concatenate([calm * 0.3, stress * 0.3])
    return pd.DataFrame(
        {"2y-1y": front, "fly 2*2y-1y-5y": belly}, index=idx
    )


def test_stress_features_global_standardisation(synthetic_basket):
    z = regimes.build_stress_features(synthetic_basket, window=21, standardize="global")
    assert list(z.columns) == ["z_vol_front", "z_vol_belly"]
    assert np.allclose(z.mean().to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(z.std(ddof=0).to_numpy(), 1.0, atol=1e-9)


def test_stress_features_expanding_uses_past_only(synthetic_basket):
    z = regimes.build_stress_features(
        synthetic_basket, window=21, standardize="expanding", min_periods=100
    )
    assert z.notna().all().all()
    # expanding standardisation needs history first: it starts later than the
    # rolling-volatility series and never re-standardises own past values.
    assert z.index.min() > synthetic_basket.index[100]


def test_standardize_rejects_unknown_method():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        regimes._standardize(df, "bogus", 1)


def test_threshold_regimes_extremes(synthetic_basket):
    z = regimes.build_stress_features(synthetic_basket, window=21, standardize="global")
    score = regimes.stress_score(z)
    labels = regimes.threshold_regimes(score, calm_q=0.5, stress_q=0.85, min_periods=10)
    assert set(labels.unique()).issubset({0, 1, 2})
    assert labels.loc[score.idxmax()] == 2
    top = score.nlargest(len(score) // 10).index
    assert (labels.loc[top] == 2).mean() > 0.8


def test_threshold_regimes_rejects_unsupported_state_count(synthetic_basket):
    z = regimes.build_stress_features(synthetic_basket, window=21, standardize="global")
    score = regimes.stress_score(z)
    with pytest.raises(ValueError):
        regimes.threshold_regimes(score, n_states=2)


def test_transition_matrix_rows_sum_to_one(synthetic_basket):
    z = regimes.build_stress_features(synthetic_basket, window=21, standardize="global")
    score = regimes.stress_score(z)
    labels = regimes.threshold_regimes(score, min_periods=10)
    tm = regimes.transition_matrix(labels, n_states=3)
    assert np.allclose(tm.sum(axis=1).to_numpy(), 1.0)


def test_episode_coverage_flags_known_window():
    idx = pd.date_range("2008-01-01", periods=400, freq="B")
    labels = pd.Series(0, index=idx)
    # make the GFC window stressed
    mask = (idx >= "2008-09-01") & (idx <= "2009-06-30")
    labels[mask] = 2
    cov = regimes.episode_coverage(labels, {"GFC 2008": ("2008-09-01", "2009-06-30")})
    assert cov.loc["GFC 2008", "detected"]
    assert cov.loc["GFC 2008", "elevated_share_pct"] > 90
    assert cov.loc["GFC 2008", "stressed_share_pct"] > 90
    assert cov.loc["GFC 2008", "in_scope"]


def test_episode_coverage_marks_out_of_scope():
    idx = pd.date_range("2013-01-01", periods=300, freq="B")
    labels = pd.Series(0, index=idx)
    cov = regimes.episode_coverage(
        labels, {"Taper tantrum 2013": ("2013-05-01", "2013-09-30")},
        out_of_scope={"Taper tantrum 2013"},
    )
    assert not cov.loc["Taper tantrum 2013", "in_scope"]
    assert not cov.loc["Taper tantrum 2013", "detected"]


def test_regime_summary_shares_sum_to_100(synthetic_basket):
    z = regimes.build_stress_features(synthetic_basket, window=21)
    score = regimes.stress_score(z)
    labels = regimes.threshold_regimes(score, min_periods=10)
    summary = regimes.regime_summary(labels, score)
    assert summary["share_pct"].sum() == pytest.approx(100.0, abs=0.2)
    assert summary["days"].sum() == len(labels)


def test_build_stress_features_uses_config_series_names():
    assert regimes.FRONT_SPREAD in [
        config.spread_name(n, f) for n, f in config.CALENDAR_PAIRS
    ]
    assert regimes.BELLY_BUTTERFLY in [
        config.butterfly_name(n, m, f) for n, m, f in config.BUTTERFLY_TRIPLES
    ]
