import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Use only 1 GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA available: {torch.cuda.is_available()}")

# Load the Chronos-2 pipeline
# GPU recommended for faster inference, but CPU is also supported using device_map="cpu"
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")


# Return forecasting configuration
target = "MSFT_dlyret"  # Column name containing values to forecast
prediction_length = 30  # Number of days to forecast ahead
id_column = "id"        # Column identifying different
timestamp_column = "timestamp"  # Column containing datetime information
timeseries_id = "1"     # Specific time series to visualize
cross_learning = True
batch_size = 100

context_data_input_path = "./exports/crsp/crsp_chronos_train.parquet"
test_data_input_path    = "./exports/crsp/crsp_chronos_val.parquet"

# Visualization configuration
plot_viewport_width = 24
plot_viewport_height = 8
history_length = 356 - prediction_length


# Load historical data and past values of covariates
context_df = pd.read_parquet(context_data_input_path)

context_df[timestamp_column] = pd.to_datetime(context_df[timestamp_column])
context_df[id_column] = context_df[id_column].astype(str)

context_timestamp_range = context_df[timestamp_column].sort_index()
context_start = context_timestamp_range.iloc[0]
context_end = context_timestamp_range.iloc[-1]

print(f"Target: {target}")
print(f"[Context] Dataframe shape: {context_df.shape}; {context_start} — {context_end}")
#print(f"Variables: {context_df.columns.tolist()}")
#display(context_df.head()) # Uncomment to view dataframe in notebook


# Load future values of covariates
test_df = pd.read_parquet(test_data_input_path)

test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column])
test_df[id_column] = test_df[id_column].astype(str)
future_df = test_df.drop(columns=target).iloc[:prediction_length].reset_index(drop=True)

test_ts_range = test_df[timestamp_column].sort_index()
future_ts_range = future_df[timestamp_column].sort_index()

test_tsr_start = test_ts_range.iloc[0]
test_tsr_end = test_ts_range.iloc[-1]
future_tsr_start = future_ts_range.iloc[0]
future_tsr_end = future_ts_range.iloc[-1]

print(f"[Full Dataset] Dataframe shape: {test_df.shape}; {test_tsr_start} — {test_tsr_end}")
print(f"[Testing] Dataframe shape: {future_df.shape}; {future_tsr_start} — {future_tsr_end}")
#print(f"Variables: {future_df.columns.tolist()}")
#display(future_df.head()) # Uncomment to view dataframe in notebook


# Comparison: forecast without covariates
pred_no_cov_df = pipeline.predict_df(
    context_df[[id_column, timestamp_column, target]],
    future_df=None,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)
print("[Prediction (without covariates)] Dataframe shape:", pred_no_cov_df.shape)
#display(pred_no_cov_df.head()) # Uncomment to view dataframe in notebook

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
)
print("[Prediction (with covariates)] Dataframe shape:", pred_df.shape)
#display(pred_df.head()) # Uncomment to view dataframe in notebook

# Cross-learning enabled for joint prediction
# This assigns the same group ID to all time series, allowing information sharing
joint_pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=id_column,
    timestamp_column=timestamp_column,
    target=target,
    cross_learning=cross_learning,
    batch_size=batch_size,
)
print("[Prediction (with covariates and cross-learning enabled)] Dataframe shape:", joint_pred_df.shape)
#display(joint_pred_df.head()) # Uncomment to view dataframe in notebook

output_dir = os.path.dirname(context_data_input_path)
os.makedirs(output_dir, exist_ok=True)

combined_pred_df = pd.concat(
    [
        pred_no_cov_df.assign(model="no_covariates"),
        pred_df.assign(model="with_covariates"),
        joint_pred_df.assign(model="with_covariates_cross_learning"),
    ],
    ignore_index=True,
)
combined_pred_path = os.path.join(output_dir, "crsp_chronos_results.csv")
combined_pred_df.to_csv(combined_pred_path, index=False)
print(f"[Prediction results] Exported combined results to: {combined_pred_path}")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if total == 0 else 1 - np.sum((y_true - y_pred) ** 2) / total
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def _build_analysis_row(prediction_df: pd.DataFrame, label: str) -> dict:
    pred_filtered = prediction_df.query("target_name == @target")
    merged = pred_filtered[[id_column, timestamp_column, "predictions"]].merge(
        test_df[[id_column, timestamp_column, target]],
        on=[id_column, timestamp_column],
        how="inner",
    )
    y_true = merged[target].to_numpy()
    y_pred = merged["predictions"].to_numpy()
    metrics = _compute_metrics(y_true, y_pred)
    metrics["model"] = label
    metrics["n"] = len(merged)
    return metrics


analysis_rows = [
    _build_analysis_row(pred_no_cov_df, "no_covariates"),
    _build_analysis_row(pred_df, "with_covariates"),
    _build_analysis_row(joint_pred_df, "with_covariates_cross_learning"),
]
analysis_df = pd.DataFrame(analysis_rows)
analysis_df = analysis_df[["model", "n", "MSE", "MAE", "RMSE", "MAPE", "R2"]]

analysis_path = os.path.join(output_dir, "crsp_chronos_analysis.csv")
analysis_df.to_csv(analysis_path, index=False)
print(f"[Prediction analysis] Exported analysis to: {analysis_path}")

analysis_long_df = analysis_df.melt(
    id_vars=["model"],
    value_vars=["MSE", "MAE", "RMSE", "MAPE", "R2"],
    var_name="metric",
    value_name="value",
)
fig = px.bar(
    analysis_long_df,
    x="metric",
    y="value",
    color="model",
    barmode="group",
    title="Chronos Forecast Error Metrics",
)
fig.update_layout(legend_title_text="Model")
fig.show()

# Visualization helper function
def plot_forecast(
    context_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    timeseries_id: str,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    history_length: int = history_length,
    title_suffix: str = "",
):
    ts_context = context_df.query(f"{id_column} == @timeseries_id").set_index(timestamp_column)[target_column]
    ts_pred = pred_df.query(f"{id_column} == @timeseries_id and target_name == @target_column").set_index(
        timestamp_column
    )[["0.1", "predictions", "0.9"]]
    ts_ground_truth = test_df.query(f"{id_column} == @timeseries_id").set_index(timestamp_column)[target_column]

    last_date = ts_context.index.max()
    start_idx = max(0, len(ts_context) - history_length)
    plot_cutoff = ts_context.index[start_idx]
    ts_context = ts_context[ts_context.index >= plot_cutoff]
    ts_pred = ts_pred[ts_pred.index >= plot_cutoff]
    ts_ground_truth = ts_ground_truth[ts_ground_truth.index >= plot_cutoff]
    ts_ground_truth = ts_ground_truth.loc[ts_pred.index] # Trim ground truth to forecast horizon
    
    fig = plt.figure(figsize=(plot_viewport_width, plot_viewport_height))
    ax = fig.gca()
    ts_context.plot(ax=ax, label=f"historical {target_column}", color="xkcd:azure")
    ts_ground_truth.plot(ax=ax, label=f"future {target_column} (ground truth)", color="xkcd:grass green")
    ts_pred["predictions"].plot(ax=ax, label="forecast", color="xkcd:violet")
    ax.fill_between(
        ts_pred.index,
        ts_pred["0.1"],
        ts_pred["0.9"],
        alpha=0.7,
        label="prediction interval",
        color="xkcd:light lavender",
    )
    ax.axvline(x=last_date, color="black", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_title(f"{target_column} forecast for {timeseries_id} {title_suffix}")
    plt.show()
    
# Visualize forecast with covariates
plot_forecast(
    context_df,
    pred_no_cov_df,
    test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix="(without covariates)",
)
plot_forecast(
    context_df,
    pred_df,
    test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix="(with covariates)",
)
plot_forecast(
    context_df,
    joint_pred_df,
    test_df,
    target_column=target,
    timeseries_id=timeseries_id,
    title_suffix=f"(with covariates and cross-learning enabled [batch_size = {batch_size}])",
)