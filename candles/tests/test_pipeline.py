"""End-to-end tests of the pipeline on committed fixtures (offline, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from candles.config import PATTERN_NAMES
from candles.pipeline import build_candles, evaluate_universe, ticker_event_counts
from candles.run import run
from data import sample_ohlc


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("FRT_OFFLINE", "1")


def test_build_candles_adds_patterns_and_forward_returns():
    df = sample_ohlc("ES=F")
    out = build_candles(df)
    for name in PATTERN_NAMES:
        assert name in out.columns and out[name].dtype == bool
    for h in (1, 5, 10):
        assert f"fwd_ret_{h}" in out.columns
    assert len(out) == len(df)


def test_evaluate_universe_pools_instruments():
    frames = {"ES=F": sample_ohlc("ES=F"), "GC=F": sample_ohlc("GC=F")}
    pooled, summary = evaluate_universe(frames, n_boot=200, seed=0)
    assert set(pooled["ticker"]) == {"ES=F", "GC=F"}
    assert len(summary) == len(PATTERN_NAMES) * 3
    counts = ticker_event_counts(pooled)
    assert set(counts["ticker"]) == {"ES=F", "GC=F"}
    assert (counts["n_bars"] > 50).all()


def test_run_writes_artifacts_offline(tmp_path: Path):
    out_dir = tmp_path / "candles"
    head = run(out_dir=out_dir, offline=True, tickers=("ES=F", "GC=F"),
               start=None, end=None, n_boot=200, seed=0)
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "headline.json").exists()
    assert (out_dir / "labelled_ES_F.csv").exists()
    disk = json.loads((out_dir / "headline.json").read_text())
    assert disk["n_tickers"] == 2
    assert set(head) == set(disk)
    assert head["n_bars"] > 100
    assert disk["n_tests"] == len(PATTERN_NAMES) * 3
    summary = pd.read_csv(out_dir / "summary.csv")
    assert {"pattern", "horizon", "n_signals", "primary_p_bh"} <= set(summary.columns)
