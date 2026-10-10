"""Figures for the regime study (matplotlib, Agg backend, 150 dpi PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REGIME_COLORS = {0: "#d7f0d7", 1: "#fff2cc", 2: "#f6cfcf"}
REGIME_NAMES = {0: "calm", 1: "normal", 2: "stressed"}


def _shade(ax, labels: pd.Series) -> None:
    """Shade the axis background by contiguous regime runs."""
    if labels.empty:
        return
    values = labels.to_numpy()
    dates = labels.index
    start = dates[0]
    current = values[0]
    for i in range(1, len(values)):
        if values[i] != current:
            ax.axvspan(start, dates[i], color=REGIME_COLORS.get(int(current), "#eeeeee"),
                       alpha=0.6, lw=0)
            start = dates[i]
            current = values[i]
    ax.axvspan(start, dates[-1], color=REGIME_COLORS.get(int(current), "#eeeeee"),
               alpha=0.6, lw=0)


def plot_regime_timeline(
    series: pd.DataFrame,
    labels: pd.Series,
    score: pd.Series,
    path: Path,
    title: str = "Short-rate curve regimes",
) -> None:
    """Top: two proxy series shaded by regime. Bottom: stress score."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    _shade(ax1, labels)
    for col in series.columns:
        ax1.plot(series.index, series[col], lw=1.0, label=col)
    ax1.set_ylabel("basis points")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)

    _shade(ax2, labels)
    ax2.plot(score.index, score.to_numpy(), color="#333333", lw=1.0)
    ax2.set_ylabel("stress score")
    ax2.set_xlabel("date")

    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[s], alpha=0.6)
               for s in sorted(REGIME_NAMES)]
    ax2.legend(handles, [REGIME_NAMES[s] for s in sorted(REGIME_NAMES)],
               loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rolling_corr(rolling: pd.DataFrame, labels: pd.Series, path: Path) -> None:
    """Rolling pairwise correlations over time, shaded by regime."""
    fig, ax = plt.subplots(figsize=(13, 5))
    _shade(ax, labels)
    for col in rolling.columns:
        ax.plot(rolling.index, rolling[col], lw=1.0, label=col)
    ax.axhline(0.0, color="#888888", lw=0.8, ls="--")
    ax.set_ylabel("rolling correlation")
    ax.set_xlabel("date")
    ax.set_title("Rolling correlation of spread/butterfly daily changes")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(ax, mat: pd.DataFrame, title: str, vmin=-1, vmax=1) -> None:
    ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(mat.index, fontsize=7)
    ax.set_title(title, fontsize=10)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            val = mat.to_numpy()[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.6 else "black")


def plot_corr_heatmaps(matrices: dict, path: Path, title: str) -> None:
    """One correlation heatmap per regime, side by side."""
    states = sorted(matrices)
    fig, axes = plt.subplots(1, len(states), figsize=(5.2 * len(states), 5.0))
    if len(states) == 1:
        axes = [axes]
    for ax, state in zip(axes, states, strict=False):
        _heatmap(ax, matrices[state], f"{REGIME_NAMES.get(state, state)} (regime {state})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_corr_flip(calm: pd.DataFrame, stressed: pd.DataFrame, path: Path) -> None:
    """Difference heatmap: stressed minus calm correlation."""
    common = calm.index.intersection(stressed.index)
    diff = stressed.loc[common, common] - calm.loc[common, common]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _heatmap(ax, diff, "Correlation shift: stressed - calm", vmin=-1, vmax=1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_state_scatter(features: pd.DataFrame, labels: pd.Series, path: Path) -> None:
    """Scatter of the two stress features coloured by regime."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cols = list(features.columns)
    for state in sorted(labels.unique()):
        mask = (labels == state).reindex(features.index).fillna(False).to_numpy(dtype=bool)
        ax.scatter(features[cols[0]].to_numpy()[mask], features[cols[1]].to_numpy()[mask],
                   s=6, alpha=0.5, color=REGIME_COLORS.get(int(state), "#cccccc"),
                   edgecolors="none", label=f"{REGIME_NAMES.get(int(state), state)}")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title("Stress feature space by regime")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
