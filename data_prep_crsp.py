import glob
import os
from typing import Any

import pandas as pd


def first_valid(series: pd.Series) -> object:
    first_idx = series.first_valid_index()
    if first_idx is None:
        return pd.NA
    return series.loc[first_idx]

def format_ticker(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    ticker = str(value).strip()
    return ticker if ticker else "unknown"

def main() -> None:
    data_dir = os.path.join(".", "data", "crsp")
    export_dir = os.path.join(".", "exports", "crsp")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.read_csv(csv_files[0])
    df.columns = [col.strip().lower() for col in df.columns]
    print(f"Original variables: {df.columns.tolist()}")
    
    keep_columns = [
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
    print(f"Preserving columns headers: {keep_columns}")
    
    missing = [col for col in keep_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")

    df = df[keep_columns].rename(columns={"dlycaldt": "timestamp"})
    df.insert(0, "id", 1)
    df = df[
        [
            "id",
            "timestamp",
            "permco",
            "ticker",
            "dlyprc",
            "dlyret",
            "dlycap",
            "dlyvol",
            "shrout",
            "sprtrn",
        ]
    ]

    permco_values = sorted(df["permco"].dropna().unique().tolist())
    #print(permco_values)
    print(f"Updating columns headers: {df.columns.tolist()}")
    print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")
    
    df_sorted = df.sort_values(["permco", "timestamp"], kind="mergesort")
    last_ticker_by_permco = df_sorted.groupby("permco", sort=False)["ticker"].last()
    permco_to_ticker = {
        permco: format_ticker(ticker)
        for permco, ticker in last_ticker_by_permco.items()
    }
    value_columns = ["dlyprc", "dlyret", "dlycap", "dlyvol", "shrout", "sprtrn"]

    wide = df.pivot_table(
        index="timestamp",
        columns="permco",
        values=value_columns,
        aggfunc=first_valid,
    )
    wide = wide.swaplevel(0, 1, axis=1)
    ordered_columns = [
        (permco, value) for permco in permco_values for value in value_columns
    ]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    wide.columns = [
        f"{permco_to_ticker.get(permco, 'unknown')}_{value}"
        for permco, value in wide.columns
    ]
    wide = wide.reset_index()
    wide["timestamp"] = pd.to_datetime(wide["timestamp"], errors="coerce")
    if wide["timestamp"].isna().any():
        bad_count = int(wide["timestamp"].isna().sum())
        raise ValueError(f"Found {bad_count} rows with invalid timestamps.")
    if wide["timestamp"].duplicated().any():
        dup_count = int(wide["timestamp"].duplicated().sum())
        raise ValueError(f"Found {dup_count} duplicate timestamps after pivot.")
    full_index = pd.date_range(
        start=wide["timestamp"].min(),
        end=wide["timestamp"].max(),
        freq="D",
    )
    wide = wide.set_index("timestamp").sort_index().reindex(full_index)
    wide = wide.reset_index().rename(columns={"index": "timestamp"})
    wide.insert(0, "id", 1)

    train_start = pd.Timestamp("2000-01-03")
    train_end = pd.Timestamp("2023-01-02")
    val_start = pd.Timestamp("2023-01-03")
    val_end = pd.Timestamp("2024-01-02")
    test_start = pd.Timestamp("2024-01-03")
    test_end = pd.Timestamp("2024-12-31")

    train_df = wide[(wide["timestamp"] >= train_start) & (wide["timestamp"] <= train_end)]
    val_df  = wide[(wide["timestamp"] >= val_start) & (wide["timestamp"] <= val_end)]
    test_df = wide[(wide["timestamp"] >= test_start) & (wide["timestamp"] <= test_end)]
    full_df = wide[(wide["timestamp"] >= train_start) & (wide["timestamp"] <= test_end)]

    train_path = os.path.join(export_dir, "crsp_chronos_train.parquet")
    train_csv_path = os.path.join(export_dir, "crsp_chronos_train.csv")
    val_path = os.path.join(export_dir, "crsp_chronos_val.parquet")
    val_csv_path = os.path.join(export_dir, "crsp_chronos_val.csv")
    test_path = os.path.join(export_dir, "crsp_chronos_test.parquet")
    test_csv_path = os.path.join(export_dir, "crsp_chronos_test.csv")
    full_path = os.path.join(export_dir, "crsp_chronos_full.parquet")
    full_csv_path = os.path.join(export_dir, "crsp_chronos_full.csv")

    train_df.to_parquet(train_path, index=False)
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_parquet(val_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_parquet(test_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    full_df.to_parquet(full_path, index=False)
    full_df.to_csv(full_csv_path, index=False)

    #print(permco_values)
    print(f"rows: {wide.shape[0]}, columns: {wide.shape[1]}")
    print(f"train rows: {train_df.shape[0]}, val rows: {val_df.shape[0]}, test rows: {test_df.shape[0]}, total rows: {full_df.shape[0]}")
    print(f"date range: {train_start} — {test_end}")
    #print(f"variables: {wide.columns.tolist()}")
    #print(wide.head())

if __name__ == "__main__":
    main()
