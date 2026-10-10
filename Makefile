PY ?= .venv/bin/python
.PHONY: fixtures install lint test fairvalue regimes candles e2e all clean

install:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest

# Each module: `python -m <module>.run` writes results/<module>/ and headline.json
fairvalue regimes candles:
	$(PY) -m $@.run

# Runs all three modules on real (cached) data once, then gathers headline numbers.
e2e: fairvalue regimes candles
	$(PY) -m data.gather

all: lint test e2e

clean:
	rm -rf results/fairvalue results/regimes results/candles

fixtures:
	$(PY) -m data.make_fixtures
