import numpy as np
import pandas as pd

from regimes.hmm import GaussianHMM, detect_regimes_hmm, logsumexp


def test_logsumexp_matches_naive():
    a = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 5.0]])
    naive = np.log(np.sum(np.exp(a), axis=1))
    assert np.allclose(logsumexp(a, axis=1), naive)
    assert np.allclose(logsumexp(a, axis=0), np.log(np.sum(np.exp(a), axis=0)))


def test_logsumexp_stable_for_large_values():
    a = np.array([1000.0, 1000.0])
    assert np.isfinite(logsumexp(a))


def _two_regime_data(seed=0):
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0, 0.3, size=(400, 2))
    stressed = rng.normal(3.0, 0.5, size=(400, 2))
    return np.vstack([calm, stressed]), np.array([0] * 400 + [1] * 400)


def test_hmm_recovers_two_separated_regimes():
    X, truth = _two_regime_data()
    model = GaussianHMM(n_states=2, seed=0).fit(X)
    pred = model.predict(X)
    acc = max((pred == truth).mean(), (pred == 1 - truth).mean())
    assert acc > 0.95


def test_hmm_mean_ordering_tracks_stress():
    X, _ = _two_regime_data()
    model = GaussianHMM(n_states=2, seed=0).fit(X)
    assert model.means_[1].mean() > model.means_[0].mean()


def test_detect_regimes_labels_sorted_calm_to_stressed():
    X, _ = _two_regime_data()
    idx = pd.date_range("2000-01-03", periods=len(X), freq="B")
    feats = pd.DataFrame(X, index=idx, columns=["z_a", "z_b"])
    labels, info = detect_regimes_hmm(feats, n_states=2, seed=0)
    assert set(labels.unique()).issubset({0, 1})
    # calm state (0) must have the smaller mean feature level than stressed (1)
    means = info["means"]
    assert means[0].mean() < means[1].mean()


def test_hmm_reproducible_with_seed():
    X, _ = _two_regime_data(seed=1)
    a = GaussianHMM(n_states=2, seed=7).fit(X).predict(X)
    b = GaussianHMM(n_states=2, seed=7).fit(X).predict(X)
    assert np.array_equal(a, b)


def test_hmm_loglik_improves_over_iterations():
    X, _ = _two_regime_data()
    model = GaussianHMM(n_states=2, seed=0).fit(X)
    hist = model.loglik_history_
    assert hist[-1] >= hist[0]
