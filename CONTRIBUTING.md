# Contribution Guide

Thanks for helping improve AtlasEconomica-FTSM. This repo focuses on data prep and time series forecasting experiments.

## Quick start
1. Create a virtual environment and install dependencies.
2. Run scripts from the repo root with `uv run python ...` or `python ...`.

See `README.md` for exact setup commands and data paths.

## Project layout
- `modules/`: data preparation utilities.
- `models/`: forecasting and modeling scripts.
- `tests/`: quick checks and examples.
- `notebooks/`: exploratory work.
- `data/`: local datasets (gitignored).

## Data expectations
- Raw CRSP CSVs go in `data/crsp/raw/`.
- Generated outputs go in `data/crsp/processed/`.
- Holiday outputs go in `data/holidays/raw/`.
- Do not commit raw or processed datasets.

## Code style
- Keep scripts runnable from the repo root.
- Use clear variable names.
- Add brief comments only when logic is non-obvious.
- Keep paths relative using `Path` or `os.path` helpers.

## Tests
- `tests/test_cuda.py` is a simple GPU check.
- `tests/energy-price-forecasting.py` downloads data and requires network access.

## Notebooks
- Keep notebooks focused on exploration.
- Port stable workflows into `modules/` or `models/` scripts.

## Pull requests
- Include a short description of what changed and why.
- Call out data assumptions and new outputs.
- Update `README.md` if user-facing behavior changes.
