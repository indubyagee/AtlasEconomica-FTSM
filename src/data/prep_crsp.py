import glob
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA_INPUT_PATH = Path("data") / "raw" / "crsp" / "dsfv2"
DATA_OUTPUT_PATH = Path("data") / "input" / "crsp" / "dsfv2"
SPLITS_CONFIG_PATH = BASE / "config" / "splits.yaml"

KEEP_COLUMNS = [
    "ticker",
    "permco",
    "dlycaldt",
    "dlyprc",
    "dlycap",
    "dlyret",
    "dlyvol",
    "shrout",
    "sprtrn",
]
VALUE_COLUMNS = ["dlyprc", "dlyret", "dlycap", "dlyvol", "shrout", "sprtrn"]


def _parse_splits_config_fallback(text: str) -> Dict[str, Dict[str, str]]:
    """
    Minimal parser for config/splits.yaml when PyYAML is unavailable.
    Expected shape:
    splits:
      <section>:
        begdt: "YYYY-MM-DD"
        enddt: "YYYY-MM-DD"
    """
    splits: Dict[str, Dict[str, str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line.startswith("  ") and not raw_line.startswith("    ") and stripped.endswith(":"):
            current = stripped[:-1]
            splits[current] = {}
            continue
        if raw_line.startswith("    ") and current and ":" in stripped:
            key, value = stripped.split(":", 1)
            splits[current][key.strip()] = value.strip().strip('"')
    return splits


def load_split_ranges(config_path: Path) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Split config not found at {config_path}")

    text = config_path.read_text()
    data: Dict[str, Dict[str, str]] = {}
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict) and isinstance(loaded.get("splits"), dict):
            data = loaded["splits"]  # type: ignore[assignment]
    except Exception:
        data = _parse_splits_config_fallback(text)

    if not data:
        raise ValueError(f"No splits found in {config_path}")

    def parse_date(label: str, raw_value: Any) -> pd.Timestamp:
        ts = pd.to_datetime(raw_value, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid date for {label} in {config_path}: {raw_value!r}")
        return ts

    parsed: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, values in data.items():
        if not isinstance(values, dict):
            raise ValueError(f"Split block {name} is malformed in {config_path}")
        beg = parse_date(f"{name}.begdt", values.get("begdt"))
        end = parse_date(f"{name}.enddt", values.get("enddt"))
        parsed[name] = (beg, end)

    return parsed


def _rename_columns_with_tickers(columns: list[str], permco_to_ticker: Dict[int, str]) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}
    for col in columns:
        if col == "timestamp":
            continue
        if "_" not in col:
            continue
        permco_part, metric = col.split("_", 1)
        try:
            permco = int(permco_part)
        except ValueError:
            continue
        ticker = permco_to_ticker.get(permco, "unknown")
        rename_map[col] = f"{ticker}_{metric}"
    return rename_map


def build_wide_dataframe(in_dir: Path) -> pd.DataFrame:
    patterns = [os.path.join(in_dir, ext) for ext in ("*.csv", "*.gz")]
    files = [fname for pattern in patterns for fname in glob.glob(pattern)]
    if not files:
        raise FileNotFoundError(f"No CSV/GZ files found in {in_dir}")

    con = duckdb.connect(database=":memory:")
    raw_rel = con.read_csv(files, header=True)
    raw_rel.create_view("raw_rel")
    raw_cols = con.sql("SELECT * FROM raw_rel LIMIT 0").columns
    lower_cols = [col.lower() for col in raw_cols]
    rename_expr = ", ".join([f'"{orig}" AS "{new}"' for orig, new in zip(raw_cols, lower_cols)])
    con.execute(f"CREATE OR REPLACE VIEW base_raw AS SELECT {rename_expr} FROM raw_rel")

    base_cols = con.sql("SELECT * FROM base_raw LIMIT 0").columns
    missing = [col for col in KEEP_COLUMNS if col not in base_cols]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    con.execute(
        """
        CREATE OR REPLACE VIEW base_rel AS
        SELECT
            ticker,
            permco,
            CAST(dlycaldt AS DATE) AS timestamp,
            dlyprc,
            dlyret,
            dlycap,
            dlyvol,
            shrout,
            sprtrn
        FROM base_raw
        """
    )

    permco_to_ticker = dict(
        con.sql(
            "SELECT permco, arg_max(ticker, timestamp) AS ticker FROM base_rel GROUP BY permco"
        ).fetchall()
    )

    long_rel = con.sql(
        """
        SELECT
            timestamp,
            CAST(permco AS VARCHAR) || '_' || metric AS col,
            value
        FROM (
            SELECT timestamp, permco, 'dlyprc' AS metric, dlyprc AS value FROM base_rel
            UNION ALL
            SELECT timestamp, permco, 'dlyret' AS metric, dlyret AS value FROM base_rel
            UNION ALL
            SELECT timestamp, permco, 'dlycap' AS metric, dlycap AS value FROM base_rel
            UNION ALL
            SELECT timestamp, permco, 'dlyvol' AS metric, dlyvol AS value FROM base_rel
            UNION ALL
            SELECT timestamp, permco, 'shrout' AS metric, shrout AS value FROM base_rel
            UNION ALL
            SELECT timestamp, permco, 'sprtrn' AS metric, sprtrn AS value FROM base_rel
        )
        """
    )

    pivot_rel = long_rel.pivot(
        values="value",
        index="timestamp",
        columns="col",
        agg_function="first",
    )

    date_span = con.sql(
        """
        SELECT *
        FROM generate_series(
            (SELECT min(timestamp) FROM base_rel),
            (SELECT max(timestamp) FROM base_rel),
            INTERVAL 1 DAY
        ) AS gs(timestamp)
        """
    )

    full_wide = date_span.join(pivot_rel, "timestamp", "left").order("timestamp").df()

    rename_map = _rename_columns_with_tickers(full_wide.columns, permco_to_ticker)
    full_wide = full_wide.rename(columns=rename_map)
    full_wide.insert(0, "id", 1)
    full_wide["timestamp"] = pd.to_datetime(full_wide["timestamp"], errors="coerce")
    if full_wide["timestamp"].isna().any():
        bad_count = int(full_wide["timestamp"].isna().sum())
        raise ValueError(f"Found {bad_count} rows with invalid timestamps after pivot.")

    return full_wide


def main() -> None:
    in_dir = BASE / DATA_INPUT_PATH
    out_dir = BASE / DATA_OUTPUT_PATH
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ranges = load_split_ranges(SPLITS_CONFIG_PATH)
    try:
        train_start, train_end = split_ranges["context"]
        val_start, val_end = split_ranges["validation"]
        test_start, test_end = split_ranges["test"]
    except KeyError as exc:
        raise KeyError(f"Missing split section in {SPLITS_CONFIG_PATH}: {exc}") from exc

    wide_df = build_wide_dataframe(in_dir)

    train_df = wide_df[(wide_df["timestamp"] >= train_start) & (wide_df["timestamp"] <= train_end)]
    val_df = wide_df[(wide_df["timestamp"] >= val_start) & (wide_df["timestamp"] <= val_end)]
    test_df = wide_df[(wide_df["timestamp"] >= test_start) & (wide_df["timestamp"] <= test_end)]
    full_df = wide_df[(wide_df["timestamp"] >= train_start) & (wide_df["timestamp"] <= test_end)]

    train_df.to_parquet(out_dir / "crsp_ftsm_context.parquet", index=False)
    train_df.to_csv(out_dir / "crsp_ftsm_context.csv", index=False)
    val_df.to_parquet(out_dir / "crsp_ftsm_validation.parquet", index=False)
    val_df.to_csv(out_dir / "crsp_ftsm_validation.csv", index=False)
    test_df.to_parquet(out_dir / "crsp_ftsm_test.parquet", index=False)
    test_df.to_csv(out_dir / "crsp_ftsm_test.csv", index=False)
    full_df.to_parquet(out_dir / "crsp_ftsm_all.parquet", index=False)
    full_df.to_csv(out_dir / "crsp_ftsm_all.csv", index=False)

    print(f"rows: {wide_df.shape[0]}, columns: {wide_df.shape[1]}")
    print(
        f"train rows: {train_df.shape[0]}, val rows: {val_df.shape[0]}, test rows: {test_df.shape[0]}, total rows: {full_df.shape[0]}"
    )
    print(f"date range: {train_start} -> {test_end}")


if __name__ == "__main__":
    main()
