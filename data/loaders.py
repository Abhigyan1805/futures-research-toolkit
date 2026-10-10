"""Cached loaders for Yahoo Finance OHLC and FRED series.

Cache: Parquet files under ``.cache/`` (gitignored; override with FRT_CACHE_DIR).
Offline mode (``offline=True`` or env FRT_OFFLINE=1) never touches the network and
serves the small committed fixtures in ``data/fixtures`` instead (used by tests/CI).
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pandas as pd

FIXTURE_DIR = Path(__file__).parent / "fixtures"
OHLC_COLS = ["open", "high", "low", "close", "volume"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _cache_dir() -> Path:
    d = Path(os.environ.get("FRT_CACHE_DIR", Path(__file__).resolve().parents[1] / ".cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_offline(offline: bool | None) -> bool:
    if offline is None:
        return os.environ.get("FRT_OFFLINE", "") not in ("", "0")
    return offline


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _clip(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def _read_fixture(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"no offline fixture {path.name}; run `make fixtures` online")
    return pd.read_csv(path, index_col="date", parse_dates=True)


def sample_ohlc(ticker: str) -> pd.DataFrame:
    """Committed fixture for a ticker (e.g. ES=F, ZN=F); ~1 year of daily bars."""
    return _read_fixture(FIXTURE_DIR / f"ohlc_{_slug(ticker)}.csv")


def sample_fred(series: str) -> pd.DataFrame:
    """Committed fixture for a FRED series (e.g. DGS2, SOFR); ~1 year of daily values."""
    return _read_fixture(FIXTURE_DIR / f"fred_{_slug(series)}.csv")


def load_ohlc(ticker: str, start=None, end=None, *, offline: bool | None = None,
              refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance.

    Returns a DataFrame indexed by tz-naive DatetimeIndex named ``date`` with lowercase
    columns open, high, low, close, volume (prices are unadjusted; continuous futures
    such as ES=F are Yahoo's front-month series, rolled without back-adjustment).
    """
    if _is_offline(offline):
        return _clip(sample_ohlc(ticker), start, end)
    path = _cache_dir() / f"ohlc_{_slug(ticker)}_{start}_{end}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False,
                      multi_level_index=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    raw.columns = [str(c).lower() for c in raw.columns]
    df = raw[OHLC_COLS].copy()
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None)
    df.index.name = "date"
    df = df[~df.index.duplicated()].sort_index().dropna(subset=["open", "high", "low", "close"])
    df.to_parquet(path)
    return df


def load_fred(series: str, start=None, end=None, *, offline: bool | None = None,
              refresh: bool = False) -> pd.DataFrame:
    """A FRED series via the public CSV endpoint (no API key).

    Returns a one-column float DataFrame (column name = series id) indexed by ``date``.
    FRED's "." missing markers become NaN; values are in the series' own units
    (percent for rates such as DGS2, SOFR, DFF).
    """
    if _is_offline(offline):
        return _clip(sample_fred(series), start, end)
    path = _cache_dir() / f"fred_{_slug(series)}_{start}_{end}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import requests

    params = {"id": series}
    if start is not None:
        params["cosd"] = str(pd.Timestamp(start).date())
    if end is not None:
        params["coed"] = str(pd.Timestamp(end).date())
    resp = requests.get(FRED_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df = df.rename(columns={df.columns[0]: "date"}).set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = [series]
    df.to_parquet(path)
    return df
