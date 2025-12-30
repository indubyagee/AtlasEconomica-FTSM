from __future__ import annotations

import argparse
import glob
import gzip
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

try:
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "pyarrow is required to run this script. Install it with: pip install pyarrow"
    ) from exc

CSV_GZ_SUFFIX = ".csv.gz"
SUPPORTED_PATTERNS = ("*.csv", "*.csv.gz", "*.gz")


def _is_supported_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(CSV_GZ_SUFFIX):
        return True
    return path.suffix.lower() in {".csv", ".gz"}


def _expand_inputs(raw_inputs: Sequence[str], recursive: bool) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_inputs:
        path = Path(raw)
        if path.is_dir():
            globber = path.rglob if recursive else path.glob
            for pattern in SUPPORTED_PATTERNS:
                for match in globber(pattern):
                    if match.is_file() and _is_supported_file(match):
                        paths.append(match)
            continue

        if path.exists():
            if path.is_file() and _is_supported_file(path):
                paths.append(path)
            else:
                raise ValueError(f"Unsupported input: {path}")
            continue

        if any(ch in raw for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(raw)]
            if not matches:
                raise FileNotFoundError(f"No matches for pattern: {raw}")
            for match in matches:
                if match.is_dir():
                    paths.extend(_expand_inputs([str(match)], recursive))
                elif match.is_file() and _is_supported_file(match):
                    paths.append(match)
            continue

        raise FileNotFoundError(f"Input not found: {path}")

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise FileNotFoundError("No supported CSV files found.")
    return unique_paths


def _output_path_for(input_path: Path, output_dir: Path | None) -> Path:
    name = input_path.name
    lower = name.lower()
    if lower.endswith(CSV_GZ_SUFFIX):
        stem = name[: -len(CSV_GZ_SUFFIX)]
    elif lower.endswith(".gz"):
        stem = name[:-3]
    elif lower.endswith(".csv"):
        stem = name[:-4]
    else:
        stem = input_path.stem

    target_dir = output_dir if output_dir is not None else input_path.parent
    return target_dir / f"{stem}.parquet"


def _input_stream_with_infer(path: Path) -> pa.NativeFile | None:
    try:
        return pa.input_stream(str(path), compression="infer")
    except TypeError:
        try:
            return pa.input_stream(str(path))
        except Exception:
            return None
    except Exception:
        return None


def _open_csv_reader(
    path: Path,
    read_options: csv.ReadOptions,
    parse_options: csv.ParseOptions,
    convert_options: csv.ConvertOptions,
) -> tuple[csv.CSVReader, pa.NativeFile | gzip.GzipFile | None]:
    if not path.name.lower().endswith(".gz"):
        return csv.open_csv(
            str(path),
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        ), None

    direct_error: Exception | None = None
    try:
        return csv.open_csv(
            str(path),
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        ), None
    except Exception as exc:
        direct_error = exc

    source = _input_stream_with_infer(path)
    if source is not None:
        try:
            return csv.open_csv(
                source,
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            ), source
        except Exception:
            source.close()

    source = gzip.open(path, "rb")
    try:
        return csv.open_csv(
            source,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options,
        ), source
    except Exception:
        source.close()
        if direct_error is not None:
            raise direct_error
        raise


def _format_rate(rows: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "0 rows/s"
    return f"{rows / elapsed:,.0f} rows/s"


def _convert_one(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    compression: str | None,
    compression_level: int | None,
    row_group_size: int,
    block_size_mb: int,
    delimiter: str,
    no_header: bool,
    use_dictionary: bool,
    write_statistics: bool,
    progress_rows: int,
    threads: int | None,
) -> None:
    if output_path.exists():
        if not overwrite:
            print(f"[Skip] {output_path} already exists.")
            return
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if threads is not None and threads > 0:
        pa.set_cpu_count(threads)

    read_options = csv.ReadOptions(
        use_threads=True,
        block_size=block_size_mb * 1024 * 1024,
        autogenerate_column_names=no_header,
    )
    parse_options = csv.ParseOptions(delimiter=delimiter)
    convert_options = csv.ConvertOptions(check_utf8=False)

    reader, source = _open_csv_reader(input_path, read_options, parse_options, convert_options)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    last_report = 0
    start_time = time.perf_counter()

    parquet_kwargs: dict[str, object] = {
        "compression": compression,
        "use_dictionary": use_dictionary,
        "write_statistics": write_statistics,
    }
    if compression is not None and compression_level is not None:
        parquet_kwargs["compression_level"] = compression_level

    try:
        for batch in reader:
            if writer is None:
                writer = pq.ParquetWriter(output_path, batch.schema, **parquet_kwargs)

            table = pa.Table.from_batches([batch])
            writer.write_table(table, row_group_size=row_group_size)
            total_rows += batch.num_rows

            if progress_rows > 0 and total_rows - last_report >= progress_rows:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[Progress] {input_path.name}: {total_rows:,} rows in {elapsed:,.1f}s "
                    f"({_format_rate(total_rows, elapsed)})"
                )
                last_report = total_rows

        if writer is None:
            empty_table = pa.Table.from_batches([], schema=reader.schema)
            pq.write_table(
                empty_table,
                output_path,
                row_group_size=row_group_size,
                **parquet_kwargs,
            )
    finally:
        if writer is not None:
            writer.close()
        if source is not None:
            source.close()

    elapsed = time.perf_counter() - start_time
    print(
        f"[Done] {input_path.name} -> {output_path} | rows: {total_rows:,} | "
        f"time: {elapsed:,.1f}s | {_format_rate(total_rows, elapsed)}"
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert large CSV or GZ files to Parquet using streaming Arrow IO. "
            "Optimized for large datasets with minimal memory usage."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input file(s), directory, or glob pattern(s) for .csv, .csv.gz, or .gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write parquet files. Defaults to each input file's directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single output file path (only valid when one input is provided).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet files.")
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "zstd", "gzip", "brotli", "lz4", "none"],
        help="Parquet compression codec. Use 'none' for fastest writes.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Optional compression level for codecs that support it (zstd, gzip, brotli).",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=1_000_000,
        help="Row group size for parquet writer.",
    )
    parser.add_argument(
        "--block-size-mb",
        type=int,
        default=64,
        help="CSV read block size in MB. Larger values can improve throughput.",
    )
    parser.add_argument("--delimiter", default=",", help="CSV delimiter.")
    parser.add_argument("--no-header", action="store_true", help="Treat CSV as headerless.")
    parser.add_argument(
        "--no-dictionary",
        action="store_true",
        help="Disable dictionary encoding for parquet output.",
    )
    parser.add_argument(
        "--disable-statistics",
        action="store_true",
        help="Disable writing column statistics for parquet output.",
    )
    parser.add_argument(
        "--progress-rows",
        type=int,
        default=5_000_000,
        help="Print progress every N rows. Use 0 to disable.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override Arrow CPU thread count.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if args.block_size_mb < 1:
        raise ValueError("--block-size-mb must be at least 1.")
    if args.row_group_size < 1:
        raise ValueError("--row-group-size must be at least 1.")
    if args.progress_rows < 0:
        raise ValueError("--progress-rows must be zero or a positive integer.")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be a positive integer.")

    inputs = _expand_inputs(args.inputs, args.recursive)
    if args.output is not None and len(inputs) != 1:
        raise ValueError("--output can only be used with a single input file.")

    compression = None if args.compression == "none" else args.compression
    output_dir = args.output_dir.resolve() if args.output_dir else None

    for input_path in inputs:
        output_path = args.output.resolve() if args.output else _output_path_for(input_path, output_dir)
        _convert_one(
            input_path=input_path,
            output_path=output_path,
            overwrite=args.overwrite,
            compression=compression,
            compression_level=args.compression_level,
            row_group_size=args.row_group_size,
            block_size_mb=args.block_size_mb,
            delimiter=args.delimiter,
            no_header=args.no_header,
            use_dictionary=not args.no_dictionary,
            write_statistics=not args.disable_statistics,
            progress_rows=args.progress_rows,
            threads=args.threads,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
