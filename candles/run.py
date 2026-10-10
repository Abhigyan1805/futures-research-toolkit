"""Entry point for the candlestick pattern engine (``python -m candles.run``).

Pipeline
--------
1. Load daily OHLC for the liquid-futures universe via the shared loaders
   (:mod:`data`, Yahoo Finance; 10+ years).
2. Label every candle with the 14 patterns in :mod:`candles.patterns`.
3. Pool the events across instruments and test, for each pattern and each 1/5/10
   day horizon, whether it predicts the forward return better than a matched
   random baseline (:mod:`candles.significance`), with Benjamini-Hochberg
   correction.
4. Write labelled candles (CSV), a summary table (CSV + Excel), a figure and the
   headline numbers consumed by ``python -m data.gather``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from candles.config import DEFAULT_CONFIG, PatternConfig
from candles.pipeline import DEFAULT_TICKERS, HORIZONS, evaluate_universe, ticker_event_counts
from data import load_ohlc


def _load_frames(tickers: tuple[str, ...], start: str | None, end: str | None,
                 offline: bool | None) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = load_ohlc(ticker, start=start, end=end, offline=offline)
        except Exception as exc:  # noqa: BLE001 - report and keep the other instruments
            print(f"[candles] skipping {ticker}: {exc}")
            continue
        if df is None or len(df) < 60:
            print(f"[candles] skipping {ticker}: too few bars ({0 if df is None else len(df)})")
            continue
        frames[ticker] = df
    return frames


def _make_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directional = summary[summary["side"] != "neutral"].copy()
    patterns = list(dict.fromkeys(directional["pattern"]))
    horizons = sorted(directional["horizon"].unique())
    x = np.arange(len(patterns))
    width = 0.8 / max(len(horizons), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, h in enumerate(horizons):
        sub = directional[directional["horizon"] == h].set_index("pattern")
        vals = [sub["hit_rate"].get(p, np.nan) - sub["base_rate"].get(p, np.nan)
                for p in patterns]
        sig = [bool(sub["bh_significant"].get(p, False)) for p in patterns]
        bars = ax.bar(x + i * width, vals, width, label=f"{h}d")
        for bar, s in zip(bars, sig, strict=False):
            if s:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("hit rate - base rate")
    ax.set_title("Candlestick edge vs chance by pattern/horizon (black edge = BH-significant)")
    ax.set_xticks(x + width * (len(horizons) - 1) / 2)
    ax.set_xticklabels(patterns, rotation=40, ha="right")
    ax.legend(title="horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _headline(pooled: pd.DataFrame, summary: pd.DataFrame, tickers: list[str]) -> dict:
    directional = summary[summary["side"] != "neutral"]
    sig = directional[directional["bh_significant"]]
    best = summary.sort_values("primary_p_bh").iloc[0]
    index = pooled.index
    return {
        "n_tickers": len(tickers),
        "n_bars": int(len(pooled)),
        "date_start": str(index.min().date()),
        "date_end": str(index.max().date()),
        "n_pattern_events": int(summary["n_signals"].sum()),
        "n_tests": int(len(summary)),
        "n_significant_bh": int(summary["bh_significant"].sum()),
        "significant_patterns": ", ".join(sorted(sig["pattern"].unique())) or "none",
        "mean_hit_rate": round(float(directional["hit_rate"].mean()), 4),
        "mean_base_rate": round(float(directional["base_rate"].mean()), 4),
        "best_pattern": str(best["pattern"]),
        "best_horizon": int(best["horizon"]),
        "best_hit_rate": round(float(best["hit_rate"]), 4),
        "best_p_bh": round(float(best["primary_p_bh"]), 5),
    }


def run(out_dir: Path | str = Path("results/candles"), offline: bool | None = None,
        tickers: tuple[str, ...] = DEFAULT_TICKERS, start: str | None = "2009-01-01",
        end: str | None = None, cfg: PatternConfig = DEFAULT_CONFIG,
        horizons: tuple[int, ...] = HORIZONS, seed: int = 0,
        n_boot: int = 2000) -> dict:
    """Run the whole pipeline and write artifacts; return the headline dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_frames(tickers, start, end, offline)
    if not frames:
        raise RuntimeError("no instruments loaded; check tickers/network")

    pooled, summary = evaluate_universe(frames, cfg, horizons, seed=seed, n_boot=n_boot)

    summary.to_csv(out_dir / "summary.csv", index=False)
    counts = ticker_event_counts(pooled)
    counts.to_csv(out_dir / "event_counts_by_ticker.csv", index=False)
    for ticker in frames:
        sub = pooled[pooled["ticker"] == ticker].drop(columns=["ticker"])
        sub.to_csv(out_dir / f"labelled_{ticker.replace('=', '_').replace('^', '')}.csv")
    try:
        with pd.ExcelWriter(out_dir / "candles.xlsx", engine="openpyxl") as xl:
            summary.to_excel(xl, sheet_name="summary", index=False)
            counts.to_excel(xl, sheet_name="counts", index=False)
    except ImportError:
        print("[candles] openpyxl not available; skipping Excel export")
    try:
        _make_figure(summary, out_dir / "hit_rates.png")
    except ImportError:
        print("[candles] matplotlib not available; skipping figure")

    head = _headline(pooled, summary, list(frames))
    (out_dir / "headline.json").write_text(json.dumps(head, indent=2))
    return head


def main() -> None:
    head = run()
    print("candles:", json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
