"""Regenerate the committed offline fixtures (needs network): `make fixtures`."""
from __future__ import annotations

from data.loaders import FIXTURE_DIR, _slug, load_fred, load_ohlc

OHLC = ["ES=F", "^GSPC", "GC=F", "CL=F", "ZN=F", "ZT=F", "ZF=F", "ZB=F"]
FRED = ["DGS2", "DGS5", "DGS10", "SOFR", "DFF", "DGS3MO", "DGS1"]
START, END = "2023-01-01", "2024-01-01"


def main() -> None:
    FIXTURE_DIR.mkdir(exist_ok=True)
    for t in OHLC:
        try:
            df = load_ohlc(t, START, END, offline=False, refresh=True)
        except Exception as e:  # some tickers may be unavailable on Yahoo
            print("skip", t, e)
            continue
        df.round(4).to_csv(FIXTURE_DIR / f"ohlc_{_slug(t)}.csv")
    for s in FRED:
        try:
            df = load_fred(s, START, END, offline=False, refresh=True)
        except Exception as e:
            print("skip", s, e)
            continue
        df.to_csv(FIXTURE_DIR / f"fred_{_slug(s)}.csv")


if __name__ == "__main__":
    main()
