# AtlasEconomica-FTSM

CRSP equity data preparation and Chronos-2 forecasting experiments.

## Introduction
- `modules/data_prep_crsp.py`: Convert CRSP daily CSVs to a wide daily panel and write train/val/test splits.
- `models/forecasting-engine.py`: Run Chronos-2 forecasts (no covariates, with covariates, cross-learning) and export metrics.
- `modules/data_prep_holidays.py`: Pull US public/unofficial and NYSE holiday calendars.
- `notebooks/`: Exploratory notebooks mirroring the scripts.
- `tests/`: Quick GPU check and an energy price forecasting example.
- `models/sentiment-analysis.py` and `models/regime-classifier.py`: Placeholder stubs for future work.

## Requirements
- Python 3.12+
- Recommended: `uv`
- Core packages: `chronos-forecasting[extras]`, `torch`, `pandas`, `numpy`, `pyarrow`, `matplotlib`, `plotly`, `holidays`

## Setup

```powershell
uv venv
.venv/Scripts/Activate.ps1
```

```powershell
# GPU (CUDA 12.4)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only
# uv pip install torch torchvision torchaudio

uv pip install "chronos-forecasting[extras]>=2.2" matplotlib numpy pandas pyarrow plotly holidays
```

```powershell
# Optional: Jupyter support
uv pip install jupyterlab ipykernel ipywidgets
uv run python -m ipykernel install --user --name atlaseconomica-ftsm --display-name "AtlasEconomica-FTSM (uv)"
```

## Data preparation (CRSP)

1. Place a CRSP CSV in `data/crsp/raw/` (the script uses the first `*.csv` file it finds).
2. Ensure the CSV includes the following columns (case-insensitive):
   - `ticker`, `permco`, `dlycaldt`, `dlyprc`, `dlycap`, `dlyret`, `dlyvol`, `shrout`, `sprtrn`
3. Run:

```powershell
uv run python modules/data_prep_crsp.py
```

Outputs are written to `data/crsp/processed/`:
- `crsp_ftsm_context.parquet` (2000-01-03 to 2023-01-02)
- `crsp_ftsm_validation.parquet` (2023-01-03 to 2024-01-02)
- `crsp_ftsm_test.parquet` (2024-01-03 to 2024-12-31)
- `crsp_ftsm_all.parquet`
- CSV versions of each file

Notes:
- Column headers are normalized to lowercase.
- Output columns are named `<ticker>_<field>` for each permco.
- The timeline is reindexed to a daily calendar; non-trading days remain NaN.

## Data preparation (Holidays)

```powershell
uv run python modules/data_prep_holidays.py
```

Outputs are written to `data/holidays/raw/`:
- `us_holidays.csv` / `us_holidays.parquet`
- `nyse_holidays.csv` / `nyse_holidays.parquet`

## Forecasting engine (Chronos-2)

1. Update the paths and target in `models/forecasting-engine.py` to match your local data and desired target column.
2. If you do not have a GPU, change `device_map="cuda"` to `device_map="cpu"`.

Example configuration:

```python
context_input_path = base / "data" / "crsp" / "processed" / "crsp_ftsm_context.parquet"
validation_input_path = base / "data" / "crsp" / "processed" / "crsp_ftsm_validation.parquet"
target = "MSFT_dlyret"
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
```

Run:

```powershell
uv run python models/forecasting-engine.py
# or via the main entrypoint
uv run python main.py
```

Outputs are written to `data/crsp/`:
- `crsp_ftsm_results.csv`
- `crsp_ftsm_analysis.csv`

The script also displays a Plotly bar chart of metrics and Matplotlib forecast plots.

## Metrics

MSE, MAE, RMSE, MAPE (lower is better)
- MSE: Average of squared errors; emphasizes large errors.
- MAE: Average of absolute errors; robust to outliers.
- RMSE: Square root of MSE; same units as data.
- MAPE: Average percentage error; useful for forecasting.

R2 (higher is better)
- R2: Proportion of variance explained (0 to 1).

## Tests and examples

```powershell
uv run python tests/test_cuda.py
```

```powershell
# Network access required to download a public dataset
uv run python tests/energy-price-forecasting.py
```

## Citations

- Center for Research in Security Prices (CRSP) data retrieved from Wharton Research Data Services. "WRDS" wrds.wharton.upenn.edu, accessed 2025-11-20.
- Macroeconomic data retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/
- Woo, Gerald, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. “Unified Training of Universal Time Series Forecasting Transformers.” arXiv:2402.02592. Preprint, arXiv, May 22, 2024. https://doi.org/10.48550/arXiv.2402.02592.
- Ansari, Abdul Fatir, Lorenzo Stella, Caner Turkmen, et al. “Chronos: Learning the Language of Time Series.” arXiv:2403.07815. Preprint, arXiv, November 4, 2024. https://doi.org/10.48550/arXiv.2403.07815.
- Ansari, Abdul Fatir, Oleksandr Shchur, Jaris Küken, et al. “Chronos-2: From Univariate to Universal Forecasting.” arXiv:2510.15821. Version 1. Preprint, arXiv, October 17, 2025. https://doi.org/10.48550/arXiv.2510.15821.
- Goswami, Mononito, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. “MOMENT: A Family of Open Time-Series Foundation Models.” arXiv:2402.03885. Preprint, arXiv, October 10, 2024. https://doi.org/10.48550/arXiv.2402.03885.
- Sonkavde, Gaurang, Deepak Sudhakar Dharrao, Anupkumar M. Bongale, Sarika T. Deokate, Deepak Doreswamy, and Subraya Krishna Bhat. “Forecasting Stock Market Prices Using Machine Learning and Deep Learning Models: A Systematic Review, Performance Analysis and Discussion of Implications.” International Journal of Financial Studies 11, no. 3 (2023): 94. https://doi.org/10.3390/ijfs11030094.
- Bi, Ziqian, Keyu Chen, Chiung-Yi Tseng, et al. “Is GPT-OSS Good? A Comprehensive Evaluation of OpenAI’s Latest Open Source Models.” arXiv:2508.12461. Version 1. Preprint, arXiv, August 17, 2025. https://doi.org/10.48550/arXiv.2508.12461.
- Ekambaram, Vijay, Arindam Jati, Pankaj Dayama, et al. “Tiny Time Mixers (TTMs): Fast Pre-Trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series.” arXiv:2401.03955. Preprint, arXiv, November 7, 2024. https://doi.org/10.48550/arXiv.2401.03955.
- Ekambaram, Vijay, Subodh Kumar, Arindam Jati, et al. “TSPulse: Dual Space Tiny Pre-Trained Models for Rapid Time-Series Analysis.” arXiv:2505.13033. Preprint, arXiv, June 25, 2025. https://doi.org/10.48550/arXiv.2505.13033.

- Holiday data retrieved from Azure Open Datasets via PyPI package `holidays`.

## Links

- https://huggingface.co/amazon/chronos-2
- https://huggingface.co/google/timesfm-2.5-200m-pytorch
- https://huggingface.co/AutonLab/MOMENT-1-large
- https://huggingface.co/Prior-Labs/tabpfn_2_5
- https://huggingface.co/ProsusAI/finbert
- https://huggingface.co/openai/gpt-oss-20b
- https://huggingface.co/openai/gpt-oss-120b
- https://huggingface.co/moonshotai/Kimi-K2-Thinking
