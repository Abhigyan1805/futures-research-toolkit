"""A Gaussian hidden Markov model, implemented from scratch in NumPy.

Why from scratch
----------------
``hmmlearn`` is not a project dependency and CI is offline, so the regime model
is a self-contained diagonal-covariance Gaussian HMM fitted with the
Baum-Welch (EM) algorithm. Nothing is hidden behind a library: the forward-
backward recursions, the E/M updates and Viterbi decoding are all visible here,
which is exactly what an interviewer will want to probe.

Convention
----------
States are re-ordered after fitting so that state ``0`` is the *calm* state and
state ``n_states-1`` is the *stressed* state, ranked by the mean of the
(standardised) feature vector. All maths is done in log space for stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def logsumexp(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable ``log(sum(exp(a)))`` along an axis."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _log_gaussian_diag(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Log N(x | mu_k, diag(var_k)) for every t, k -> shape (T, K)."""
    var = np.maximum(variances, 1e-12)
    diff = X[:, None, :] - means[None, :, :]           # (T, K, D)
    log_det = np.sum(np.log(2.0 * np.pi * var), axis=1)  # (K,)
    maha = np.sum(diff * diff / var[None, :, :], axis=2)  # (T, K)
    return -0.5 * (log_det[None, :] + maha)


@dataclass
class GaussianHMM:
    """Diagonal-covariance Gaussian HMM fitted by Baum-Welch EM."""

    n_states: int = 3
    n_iter: int = 200
    tol: float = 1e-6
    seed: int = 0
    var_floor: float = 1e-4

    startprob_: np.ndarray = field(default=None, repr=False)
    transmat_: np.ndarray = field(default=None, repr=False)
    means_: np.ndarray = field(default=None, repr=False)
    variances_: np.ndarray = field(default=None, repr=False)
    loglik_: float = field(default=None, repr=False)
    loglik_history_: list = field(default_factory=list, repr=False)
    n_features_: int = field(default=0, repr=False)

    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        T, D = X.shape
        self.n_features_ = D
        # Rank observations by their mean feature (standardised -> stress level)
        # and split into contiguous chunks for a sensible, deterministic start.
        order = np.argsort(X.mean(axis=1))
        chunks = np.array_split(order, self.n_states)
        means = np.zeros((self.n_states, D))
        variances = np.zeros((self.n_states, D))
        for k, idx in enumerate(chunks):
            block = X[idx] if len(idx) else X
            means[k] = block.mean(axis=0)
            variances[k] = block.var(axis=0) + self.var_floor
        self.means_ = means
        self.variances_ = variances
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        # Persistence-friendly start away from the boundary.
        trans = np.full((self.n_states, self.n_states), 0.1 / max(self.n_states - 1, 1))
        np.fill_diagonal(trans, 0.9)
        self.transmat_ = trans + 1e-9 * rng.random((self.n_states, self.n_states))
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

    def _e_step(self, X: np.ndarray):
        logB = _log_gaussian_diag(X, self.means_, self.variances_)
        log_start = np.log(self.startprob_ + 1e-300)
        log_trans = np.log(self.transmat_ + 1e-300)
        T = X.shape[0]

        alpha = np.empty((T, self.n_states))
        alpha[0] = log_start + logB[0]
        for t in range(1, T):
            alpha[t] = logB[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
        loglik = float(logsumexp(alpha[-1]))

        beta = np.zeros((T, self.n_states))
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(log_trans + logB[t + 1] + beta[t + 1], axis=1)

        gamma = np.exp(alpha + beta - loglik)
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((self.n_states, self.n_states))
        for t in range(T - 1):
            log_xi = alpha[t][:, None] + log_trans + logB[t + 1] + beta[t + 1][None, :]
            log_xi -= logsumexp(log_xi)
            xi_sum += np.exp(log_xi)
        return loglik, gamma, xi_sum

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi_sum: np.ndarray) -> None:
        self.startprob_ = gamma[0] / gamma[0].sum()
        row = xi_sum.sum(axis=1, keepdims=True)
        self.transmat_ = xi_sum / np.maximum(row, 1e-300)
        for k in range(self.n_states):
            w = gamma[:, k]
            wsum = max(w.sum(), 1e-300)
            self.means_[k] = (w[:, None] * X).sum(axis=0) / wsum
            self.variances_[k] = (
                (w[:, None] * (X - self.means_[k]) ** 2).sum(axis=0) / wsum + self.var_floor
            )

    def fit(self, X) -> GaussianHMM:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        self._init_params(X)
        prev = -np.inf
        for _ in range(self.n_iter):
            loglik, gamma, xi_sum = self._e_step(X)
            self.loglik_history_.append(loglik)
            self._m_step(X, gamma, xi_sum)
            if np.isfinite(prev) and loglik - prev < self.tol:
                break
            prev = loglik
        self.loglik_ = float(self.loglik_history_[-1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Posterior state probabilities (T, K)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        _, gamma, _ = self._e_step(X)
        return gamma

    def predict(self, X) -> np.ndarray:
        """Viterbi (most likely state sequence), in log space."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        logB = _log_gaussian_diag(X, self.means_, self.variances_)
        log_start = np.log(self.startprob_ + 1e-300)
        log_trans = np.log(self.transmat_ + 1e-300)
        T = X.shape[0]
        delta = np.empty((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        delta[0] = log_start + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + scores[psi[t], np.arange(self.n_states)]
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


def detect_regimes_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    seed: int = 0,
    var_floor: float = 1e-4,
) -> tuple[pd.Series, dict]:
    """Fit a Gaussian HMM to standardised features and return regime labels.

    States are re-ordered so ``0`` = calm and ``n_states-1`` = stressed, ranked
    by the fitted state means of the (standardised) features.

    Returns
    -------
    labels : pd.Series
        Integer regime id per date (index preserved).
    info : dict
        ``means``, ``variances``, ``transmat``, ``startprob``, ``loglik``,
        ``order`` (stress ranking), ``n_states``.
    """
    X = features.to_numpy(dtype=float)
    model = GaussianHMM(n_states=n_states, seed=seed, var_floor=var_floor).fit(X)
    stress = model.means_.sum(axis=1)
    order = np.argsort(stress)  # calm -> stressed
    relabel = np.empty(n_states, dtype=int)
    relabel[order] = np.arange(n_states)

    raw = model.predict(X)
    labels = pd.Series(relabel[raw], index=features.index, name="regime")

    info = {
        "means": model.means_[order],
        "variances": model.variances_[order],
        "transmat": model.transmat_[np.ix_(order, order)],
        "startprob": model.startprob_[order],
        "loglik": model.loglik_,
        "order": order.tolist(),
        "n_states": n_states,
    }
    return labels, info
