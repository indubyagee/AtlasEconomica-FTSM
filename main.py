from pathlib import Path
import runpy


def run_forecasting_engine() -> None:
    script_path = Path(__file__).resolve().parent / "models" / "forecasting-engine.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Forecasting engine not found at {script_path}")
    # Execute the Chronos forecasting script as if it were run directly.
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    run_forecasting_engine()


if __name__ == "__main__":
    main()
