"""End-to-end regime study: ``python -m regimes.run``.

Pipeline
--------
1. Load the short-rate curve from free FRED history (proxy for STIR futures).
2. Build calendar-spread and butterfly proxies in basis points.
3. Build standardised rolling-volatility stress features.
4. Detect regimes two ways: a 3-state Gaussian HMM (from scratch) and a simple
   quantile threshold detector.
5. Report correlation structure per regime, rolling correlations, and whether
   known stress episodes (2008, 2020, 2022, taper tantrum) are flagged.
6. Write CSV tables, PNG figures, ``regimes.xlsx`` and ``headline.json``.

Outputs go to ``results/regimes/`` by default (contract: ``run(out_dir, offline)``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config, correlation, data, excel, plots, regimes, spreads

DEFAULT_OUT = Path("results/regimes")
ROLL_WINDOW = 63  # ~3 months for the displayed rolling correlations

# Pairs whose correlation we track explicitly (daily changes).
ROLLING_PAIRS = [
    ("2y-1y", "5y-2y"),
    ("2y-1y", "10y-5y"),
    ("2y-1y", "fly 2*2y-1y-5y"),
]


def _rolling_corr_table(changes: pd.DataFrame, window: int) -> pd.DataFrame:
    return pd.concat(
        [
            correlation.rolling_correlation(changes[a], changes[b], window)
            for a, b in ROLLING_PAIRS
        ],
        axis=1,
    )


def run(out_dir: Path = DEFAULT_OUT, offline: bool | None = None) -> dict:
    """Execute the full study and return JSON-serialisable headline numbers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Data and proxy baskets.
    bundle = data.load_bundle(offline=offline)
    curve, policy = bundle["curve"], bundle["policy"]
    basket = spreads.build_spread_basket(curve).dropna(how="all")
    policy.to_csv(out_dir / "policy_rates.csv")

    # 3. Stress features.
    z = regimes.build_stress_features(basket, window=config.FEATURE_WINDOW)
    score = regimes.stress_score(z)

    # 4. Two regime detectors.
    hmm_labels, hmm_info = regimes.fit_hmm_regimes(
        z, n_states=config.N_STATES, seed=config.RANDOM_SEED
    )
    thr_labels = regimes.threshold_regimes(score)
    labels = hmm_labels  # HMM is the primary; threshold is the cross-check

    # Correlation structure inside each regime (levels and daily changes).
    changes = correlation.daily_changes(basket.loc[labels.index])
    levels = basket.loc[labels.index]
    corr_levels = correlation.regime_correlation_matrices(levels, labels)
    corr_changes = correlation.regime_correlation_matrices(changes, labels)

    # Rolling correlations.
    rolling = _rolling_corr_table(changes, window=ROLL_WINDOW)

    # 5. Episode checks and summaries.
    episodes = regimes.episode_coverage(
        labels, config.KNOWN_EPISODES, out_of_scope=config.OUT_OF_SCOPE_EPISODES
    )
    thr_episodes = regimes.episode_coverage(
        thr_labels, config.KNOWN_EPISODES, out_of_scope=config.OUT_OF_SCOPE_EPISODES
    )
    summary = regimes.regime_summary(labels, score)
    trans_hmm = pd.DataFrame(hmm_info["transmat"])
    trans_thr = regimes.transition_matrix(thr_labels)

    # 6. Artifacts.
    basket.to_csv(out_dir / "spreads_bps.csv")
    z.to_csv(out_dir / "stress_features.csv")
    pd.DataFrame({"hmm": hmm_labels, "threshold": thr_labels, "stress_score": score}).to_csv(
        out_dir / "regime_labels.csv"
    )
    summary.to_csv(out_dir / "regime_summary.csv")
    episodes.to_csv(out_dir / "episode_coverage_hmm.csv")
    thr_episodes.to_csv(out_dir / "episode_coverage_threshold.csv")
    trans_hmm.to_csv(out_dir / "transition_hmm.csv")
    trans_thr.to_csv(out_dir / "transition_threshold.csv")
    rolling.to_csv(out_dir / "rolling_corr.csv")

    timeline_cols = [regimes.FRONT_SPREAD, regimes.BELLY_BUTTERFLY]
    plots.plot_regime_timeline(
        basket.loc[labels.index, timeline_cols], labels, score,
        out_dir / "regime_timeline.png",
    )
    plots.plot_rolling_corr(rolling, labels, out_dir / "rolling_corr.png")
    plots.plot_corr_heatmaps(corr_changes, out_dir / "corr_changes_heatmaps.png",
                             "Correlation of daily changes by regime")
    plots.plot_corr_heatmaps(corr_levels, out_dir / "corr_levels_heatmaps.png",
                             "Correlation of spread/butterfly levels by regime")
    plots.plot_state_scatter(z, labels, out_dir / "state_scatter.png")

    calm, stressed = 0, config.N_STATES - 1
    if calm in corr_changes and stressed in corr_changes:
        plots.plot_corr_flip(corr_changes[calm], corr_changes[stressed],
                             out_dir / "corr_flip.png")
        flip_metric = correlation.corr_stability(corr_changes[calm], corr_changes[stressed])
    else:
        flip_metric = float("nan")

    # Regime-average co-movement, and the single largest pairwise shift (labelled
    # honestly as the largest, not picked to flatter the story).
    mean_corr_changes = {
        s: round(correlation.mean_offdiag_correlation(m), 3) for s, m in corr_changes.items()
    }
    mean_corr_levels = {
        s: round(correlation.mean_offdiag_correlation(m), 3) for s, m in corr_levels.items()
    }
    largest_flip = {}
    if calm in corr_changes and stressed in corr_changes:
        names = list(corr_changes[calm].columns)
        best = (0.0, None, None)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                delta = corr_changes[stressed].iloc[i, j] - corr_changes[calm].iloc[i, j]
                if abs(delta) > best[0]:
                    best = (abs(delta), names[i], names[j])
        if best[1] is not None:
            largest_flip = {
                "pair": f"{best[1]} vs {best[2]}",
                "calm": round(float(corr_changes[calm].loc[best[1], best[2]]), 3),
                "stressed": round(float(corr_changes[stressed].loc[best[1], best[2]]), 3),
            }

    # Excel workbook.
    tables = {
        "regime_summary": summary.reset_index(),
        "episode_coverage": episodes.reset_index(),
        "transition_hmm": trans_hmm,
        "transition_threshold": trans_thr,
        "spread_stats": basket.describe().T.round(2),
        "rolling_corr_tail": rolling.tail(250),
    }
    for state, mat in corr_changes.items():
        tables[f"corr_changes_r{state}"] = mat.round(3)
    for state, mat in corr_levels.items():
        tables[f"corr_levels_r{state}"] = mat.round(3)
    provenance = {
        "source": "FRED constant-maturity Treasury yields (free CSV endpoint, no key)",
        "proxy_caveat": (
            "STIR futures contract-level history is not free; constant-maturity "
            "Treasury yields stand in as the short-rate curve proxy."
        ),
        "series": ", ".join(config.ALL_SERIES),
        "sample_start": curve.index.min().date().isoformat(),
        "sample_end": curve.index.max().date().isoformat(),
        "feature_window_days": config.FEATURE_WINDOW,
        "rolling_corr_window_days": ROLL_WINDOW,
        "n_states": config.N_STATES,
        "seed": config.RANDOM_SEED,
    }
    excel.write_workbook(out_dir / "regimes.xlsx", tables, provenance)

    # 7. Headline numbers (flat, JSON-serialisable, all from this real run).
    in_scope = episodes[episodes["in_scope"]]
    detected = int(in_scope["detected"].sum())
    headline = {
        "sample_start": provenance["sample_start"],
        "sample_end": provenance["sample_end"],
        "n_observations": int(len(z)),
        "n_spreads_and_flies": int(basket.shape[1]),
        "n_states": config.N_STATES,
        "hmm_loglik": round(float(hmm_info["loglik"]), 1),
        "calm_share_pct": float(summary.loc[calm, "share_pct"]),
        "stressed_share_pct": float(summary.loc[stressed, "share_pct"]),
        "episodes_detected": detected,
        "episodes_in_scope": int(len(in_scope)),
        "mean_change_corr_calm": mean_corr_changes.get(calm),
        "mean_change_corr_normal": mean_corr_changes.get(1),
        "mean_change_corr_stressed": mean_corr_changes.get(stressed),
        "mean_change_corr_lift_pct": round(
            100.0 * (
                mean_corr_changes.get(stressed, 0.0) / mean_corr_changes.get(calm, 1.0) - 1.0
            ),
            1,
        ),
        "mean_level_corr_calm": mean_corr_levels.get(calm),
        "mean_level_corr_stressed": mean_corr_levels.get(stressed),
        "corr_stability_calm_vs_stressed": round(flip_metric, 3),
        "largest_change_corr_flip_pair": largest_flip.get("pair"),
        "largest_change_corr_flip_calm": largest_flip.get("calm"),
        "largest_change_corr_flip_stressed": largest_flip.get("stressed"),
        "hmm_threshold_agreement_pct": round(
            100.0 * float((hmm_labels == thr_labels).mean()), 1
        ),
    }
    for name in config.KNOWN_EPISODES:
        key = name.lower().replace(" ", "_")
        headline[f"episode_{key}_elevated_pct"] = episodes.loc[name, "elevated_share_pct"]
        headline[f"episode_{key}_stressed_pct"] = episodes.loc[name, "stressed_share_pct"]
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2))
    return headline


def main() -> None:
    headline = run()
    print("regimes study complete -> results/regimes/")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
