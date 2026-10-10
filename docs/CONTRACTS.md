# Shared contracts

## Layout
```
data/        shared loaders, fixtures, gather.py (builds docs/RESULTS.md)   [lead]
fairvalue/   cost-of-carry engine                                           [builder 2]
regimes/     spread/butterfly correlation + regimes                          [builder 3]
candles/     candlestick patterns + significance tests                       [builder 4]
tests/       cross-module tests (modules may also keep tests in their own dir)
docs/        CONTRACTS.md, RESULTS.md (generated), WALKTHROUGH.md
results/<module>/   outputs of `make <module>` (CSV/Parquet/PNG/xlsx); small ones committed
```
Packages are declared explicitly in `pyproject.toml` (`packages = [...]`). New sub-packages must be added there.

## Data API (`from data import ...`)
```python
load_ohlc(ticker, start=None, end=None, *, offline=None, refresh=False) -> DataFrame
load_fred(series, start=None, end=None, *, offline=None, refresh=False) -> DataFrame
sample_ohlc(ticker) / sample_fred(series)        # committed fixtures (2023, ~250 rows)
```
- `load_ohlc`: Yahoo via yfinance, unadjusted. Index: tz-naive `DatetimeIndex` named `date`, sorted, unique.
  Columns exactly `open, high, low, close, volume` (lowercase). Tickers e.g. `ES=F ^GSPC GC=F CL=F ZN=F ZT=F ZF=F ZB=F`.
  Continuous futures are Yahoo's front-month series: rolls are NOT back-adjusted (gaps at roll dates).
- `load_fred`: public CSV endpoint, no key. One float column named by the series id, index `date`. `.` -> NaN.
  Rates are in percent. Series: `DGS3MO DGS1 DGS2 DGS5 DGS10 SOFR DFF`.
- Cache: Parquet in `.cache/` (gitignored; `FRT_CACHE_DIR` overrides). `refresh=True` bypasses it.
- Offline: `offline=True` or env `FRT_OFFLINE=1` serves fixtures from `data/fixtures/*.csv` and clips to start/end.
  **Tests and CI must run offline**; never hit the network in a test. Fixture filenames: `ohlc_<ticker with non-alnum -> _>.csv`,
  `fred_<series>.csv`. Regenerate with `make fixtures` (lead only).
  If your logic needs a fixture that is missing, build synthetic data inside the test instead.

## Module entry point
Each module provides `<module>/run.py` with
```python
def run(out_dir: Path = Path("results/<module>"), offline: bool | None = None) -> dict: ...
def main() -> None:  # called by `python -m <module>.run`; calls run() and prints headline
```
`run` writes its artifacts into `out_dir` and a flat JSON `out_dir/headline.json` of
**JSON-serialisable headline numbers** (key -> number/short string, e.g. `"mean_abs_basis_error_bps": 3.2`).
`python -m data.gather` turns these into `docs/RESULTS.md`. Only real-data runs may feed the README/resume.

## Output conventions
- Tables: CSV (small) or Parquet (large). Figures: PNG, 150 dpi, in `out_dir`. Excel export: `out_dir/<module>.xlsx` (openpyxl).
- Seeds fixed (`seed=0` default) so reruns reproduce.
- Dates ISO `YYYY-MM-DD`; rates in percent in tables unless a column name says `_bps` or `_dec`.

## Tooling
`make install | lint | test | e2e | all`; ruff (line length 100, rules E,F,I,B,UP) and pytest must pass.
CI runs both with `FRT_OFFLINE=1`.
