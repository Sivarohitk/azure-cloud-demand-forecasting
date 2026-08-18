"""Export validated, business-ready CSV tables for Power BI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
POWERBI_ROOT = PROJECT_ROOT / "powerbi"

DEMAND_HISTORY_SOURCE = PROCESSED_ROOT / "hourly_demand_by_trigger.parquet"
MODEL_COMPARISON_SOURCE = OUTPUTS_ROOT / "model_comparison.csv"
MODEL_COMPARISON_BY_TRIGGER_SOURCE = (
    OUTPUTS_ROOT / "model_comparison_by_trigger.csv"
)
SELECTED_MODEL_SOURCE = OUTPUTS_ROOT / "selected_model.json"
SELECTED_PREDICTIONS_SOURCE = OUTPUTS_ROOT / "selected_model_test_predictions.csv"
BASELINE_PREDICTIONS_SOURCE = OUTPUTS_ROOT / "baseline_test_predictions.csv"
SARIMAX_PREDICTIONS_SOURCE = OUTPUTS_ROOT / "sarimax_test_predictions.csv"
LIGHTGBM_PREDICTIONS_SOURCE = OUTPUTS_ROOT / "lightgbm_test_predictions.csv"
FUTURE_FORECAST_SOURCE = OUTPUTS_ROOT / "future_24h_forecast.csv"
FORECAST_INTERVALS_SOURCE = OUTPUTS_ROOT / "forecast_intervals.csv"
CAPACITY_SCENARIOS_SOURCE = OUTPUTS_ROOT / "capacity_scenarios.csv"
CAPACITY_RECOMMENDATION_SOURCE = OUTPUTS_ROOT / "capacity_recommendation.csv"

DEMAND_HISTORY_OUTPUT = POWERBI_ROOT / "demand_history.csv"
MODEL_PERFORMANCE_OUTPUT = POWERBI_ROOT / "model_performance.csv"
FORECAST_VS_ACTUAL_OUTPUT = POWERBI_ROOT / "forecast_vs_actual.csv"
FUTURE_FORECAST_OUTPUT = POWERBI_ROOT / "future_forecast.csv"
CAPACITY_SCENARIOS_OUTPUT = POWERBI_ROOT / "capacity_scenarios.csv"
CAPACITY_RECOMMENDATION_OUTPUT = POWERBI_ROOT / "capacity_recommendation.csv"

OUTPUT_PATHS = {
    "demand_history.csv": DEMAND_HISTORY_OUTPUT,
    "model_performance.csv": MODEL_PERFORMANCE_OUTPUT,
    "forecast_vs_actual.csv": FORECAST_VS_ACTUAL_OUTPUT,
    "future_forecast.csv": FUTURE_FORECAST_OUTPUT,
    "capacity_scenarios.csv": CAPACITY_SCENARIOS_OUTPUT,
    "capacity_recommendation.csv": CAPACITY_RECOMMENDATION_OUTPUT,
}


def require_inputs() -> None:
    """Fail clearly if an upstream final artifact is unavailable."""
    required = (
        DEMAND_HISTORY_SOURCE,
        MODEL_COMPARISON_SOURCE,
        MODEL_COMPARISON_BY_TRIGGER_SOURCE,
        SELECTED_MODEL_SOURCE,
        SELECTED_PREDICTIONS_SOURCE,
        BASELINE_PREDICTIONS_SOURCE,
        SARIMAX_PREDICTIONS_SOURCE,
        LIGHTGBM_PREDICTIONS_SOURCE,
        FUTURE_FORECAST_SOURCE,
        FORECAST_INTERVALS_SOURCE,
        CAPACITY_SCENARIOS_SOURCE,
        CAPACITY_RECOMMENDATION_SOURCE,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required Power BI export inputs are missing:\n{details}")
    if not POWERBI_ROOT.is_dir():
        raise FileNotFoundError(f"Power BI directory does not exist: {POWERBI_ROOT}")


def require_columns(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    """Require the exact upstream fields needed for one transformation."""
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def validate_business_table(
    name: str,
    frame: pd.DataFrame,
    business_keys: list[str],
    numeric_columns: list[str],
) -> None:
    """Validate keys, numeric parsing, finite values, and clean CSV columns."""
    if frame.empty:
        raise ValueError(f"{name} is empty.")
    accidental_index_columns = [
        column
        for column in frame.columns
        if str(column).lower() in {"index", "level_0"}
        or str(column).lower().startswith("unnamed:")
    ]
    if accidental_index_columns:
        raise ValueError(
            f"{name} has accidental index columns: {accidental_index_columns}"
        )
    if frame[business_keys].isna().any().any():
        raise ValueError(f"{name} has missing business-key values.")
    if frame.duplicated(business_keys).any():
        duplicates = int(frame.duplicated(business_keys, keep=False).sum())
        raise ValueError(
            f"{name} has {duplicates} rows with duplicate business keys "
            f"{business_keys}."
        )
    if frame.isna().any().any():
        missing = frame.isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"{name} has unjustified missing values: {missing}")

    for column in numeric_columns:
        parsed = pd.to_numeric(frame[column], errors="raise")
        values = parsed.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} contains NaN or infinite values.")
        frame[column] = parsed


def build_demand_history() -> pd.DataFrame:
    """Create the historical hourly Trigger workload table."""
    source_columns = [
        "hour_index",
        "trace_day",
        "hour_of_day",
        "Trigger",
        "invocations",
        "compute_seconds",
    ]
    frame = pd.read_parquet(
        DEMAND_HISTORY_SOURCE, columns=source_columns, engine="pyarrow"
    )
    frame.rename(columns={"Trigger": "trigger"}, inplace=True)
    frame = frame[
        [
            "hour_index",
            "trace_day",
            "hour_of_day",
            "trigger",
            "invocations",
            "compute_seconds",
        ]
    ].sort_values(["hour_index", "trigger"], kind="stable")
    frame.reset_index(drop=True, inplace=True)
    validate_business_table(
        "demand_history.csv",
        frame,
        ["hour_index", "trigger"],
        ["hour_index", "trace_day", "hour_of_day", "invocations", "compute_seconds"],
    )
    return frame


def build_model_performance() -> pd.DataFrame:
    """Combine aggregate and Trigger-level validation/test metrics."""
    total = pd.read_csv(MODEL_COMPARISON_SOURCE)
    trigger = pd.read_csv(MODEL_COMPARISON_BY_TRIGGER_SOURCE)
    metric_columns = ["MAE", "RMSE", "WAPE"]
    require_columns(
        total,
        ["split", "model", *metric_columns],
        MODEL_COMPARISON_SOURCE,
    )
    require_columns(
        trigger,
        ["split", "model", "Trigger", *metric_columns],
        MODEL_COMPARISON_BY_TRIGGER_SOURCE,
    )

    total_rows = total[["model", "split", *metric_columns]].copy()
    total_rows["trigger"] = "TOTAL"
    trigger_rows = trigger[["model", "split", "Trigger", *metric_columns]].rename(
        columns={"Trigger": "trigger"}
    )
    frame = pd.concat([trigger_rows, total_rows], ignore_index=True)
    frame.rename(columns={"split": "dataset_split"}, inplace=True)
    frame = frame[
        ["model", "dataset_split", "trigger", "MAE", "RMSE", "WAPE"]
    ].sort_values(["dataset_split", "model", "trigger"], kind="stable")
    frame.reset_index(drop=True, inplace=True)
    validate_business_table(
        "model_performance.csv",
        frame,
        ["model", "dataset_split", "trigger"],
        metric_columns,
    )
    return frame


def standardize_prediction_source(
    path: Path,
    label_column: str,
    label: str,
    forecast_column: str,
) -> pd.DataFrame:
    """Filter and standardize one test prediction source for joining."""
    source = pd.read_csv(path)
    required = [
        "split",
        label_column,
        "hour_index",
        "Trigger",
        "actual_compute_seconds",
        "predicted_compute_seconds",
    ]
    require_columns(source, required, path)
    frame = source.loc[
        (source["split"] == "test") & (source[label_column] == label),
        [
            "hour_index",
            "Trigger",
            "actual_compute_seconds",
            "predicted_compute_seconds",
        ],
    ].copy()
    if frame.empty:
        raise ValueError(f"{path} has no test predictions for {label!r}.")
    frame.rename(
        columns={"Trigger": "trigger", "predicted_compute_seconds": forecast_column},
        inplace=True,
    )
    if frame.duplicated(["hour_index", "trigger"]).any():
        raise ValueError(f"{path} has duplicate test Trigger/hour rows.")
    return frame


def merge_forecast(
    base: pd.DataFrame, candidate: pd.DataFrame, forecast_column: str
) -> pd.DataFrame:
    """Join one model forecast after proving actual targets are identical."""
    merged = base.merge(
        candidate,
        on=["hour_index", "trigger"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_candidate"),
    )
    if len(merged) != len(base) or len(candidate) != len(base):
        raise ValueError(f"{forecast_column} does not share the complete test grid.")
    if not np.allclose(
        merged["actual_compute_seconds"],
        merged["actual_compute_seconds_candidate"],
    ):
        raise ValueError(f"{forecast_column} uses different test actual values.")
    merged.drop(columns=["actual_compute_seconds_candidate"], inplace=True)
    return merged


def build_forecast_vs_actual() -> pd.DataFrame:
    """Create one aligned test-period forecast comparison table."""
    selected_metadata = json.loads(SELECTED_MODEL_SOURCE.read_text(encoding="utf-8"))
    selected_model = str(selected_metadata.get("selected_model", ""))
    if not selected_model:
        raise ValueError("selected_model.json has no selected_model value.")

    selected = standardize_prediction_source(
        SELECTED_PREDICTIONS_SOURCE,
        "model",
        selected_model,
        "selected_forecast",
    )
    seasonal = standardize_prediction_source(
        BASELINE_PREDICTIONS_SOURCE,
        "baseline",
        "seasonal_naive",
        "baseline_forecast",
    )
    sarimax = standardize_prediction_source(
        SARIMAX_PREDICTIONS_SOURCE,
        "model",
        "SARIMAX",
        "SARIMAX_forecast",
    )
    lightgbm = standardize_prediction_source(
        LIGHTGBM_PREDICTIONS_SOURCE,
        "model",
        "LightGBM",
        "LightGBM_forecast",
    )

    frame = selected
    for candidate, column in (
        (seasonal, "baseline_forecast"),
        (sarimax, "SARIMAX_forecast"),
        (lightgbm, "LightGBM_forecast"),
    ):
        frame = merge_forecast(frame, candidate, column)
    frame["selected_model"] = selected_model
    frame = frame[
        [
            "hour_index",
            "trigger",
            "actual_compute_seconds",
            "baseline_forecast",
            "SARIMAX_forecast",
            "LightGBM_forecast",
            "selected_model",
            "selected_forecast",
        ]
    ].sort_values(["hour_index", "trigger"], kind="stable")
    frame.reset_index(drop=True, inplace=True)
    validate_business_table(
        "forecast_vs_actual.csv",
        frame,
        ["hour_index", "trigger"],
        [
            "hour_index",
            "actual_compute_seconds",
            "baseline_forecast",
            "SARIMAX_forecast",
            "LightGBM_forecast",
            "selected_forecast",
        ],
    )
    return frame


def build_future_forecast() -> pd.DataFrame:
    """Create the future point and uncertainty percentile table."""
    intervals = pd.read_csv(FORECAST_INTERVALS_SOURCE)
    point_source = pd.read_csv(FUTURE_FORECAST_SOURCE)
    interval_columns = [
        "forecast_horizon_hour",
        "aggregation_level",
        "Trigger",
        "point_forecast_compute_seconds",
        "simulated_P50_compute_seconds",
        "simulated_P80_compute_seconds",
        "simulated_P90_compute_seconds",
        "simulated_P95_compute_seconds",
        "simulated_P99_compute_seconds",
    ]
    require_columns(intervals, interval_columns, FORECAST_INTERVALS_SOURCE)
    require_columns(
        point_source,
        [
            "forecast_horizon_hour",
            "aggregation_level",
            "Trigger",
            "point_forecast_compute_seconds",
        ],
        FUTURE_FORECAST_SOURCE,
    )

    source_keys = ["forecast_horizon_hour", "aggregation_level", "Trigger"]
    reconciliation = intervals[source_keys + ["point_forecast_compute_seconds"]].merge(
        point_source[source_keys + ["point_forecast_compute_seconds"]],
        on=source_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_interval", "_point"),
    )
    if len(reconciliation) != len(intervals) or len(point_source) != len(intervals):
        raise ValueError("Future point and interval forecast grids differ.")
    if not np.allclose(
        reconciliation["point_forecast_compute_seconds_interval"],
        reconciliation["point_forecast_compute_seconds_point"],
    ):
        raise ValueError("Future point forecasts do not reconcile across outputs.")

    frame = intervals[interval_columns].copy()
    frame["trigger"] = frame["Trigger"].replace({"ALL_TRIGGERS": "TOTAL"})
    frame.rename(
        columns={
            "forecast_horizon_hour": "forecast_hour",
            "point_forecast_compute_seconds": "point_forecast",
            "simulated_P50_compute_seconds": "P50",
            "simulated_P80_compute_seconds": "P80",
            "simulated_P90_compute_seconds": "P90",
            "simulated_P95_compute_seconds": "P95",
            "simulated_P99_compute_seconds": "P99",
        },
        inplace=True,
    )
    frame = frame[
        ["forecast_hour", "trigger", "point_forecast", "P50", "P80", "P90", "P95", "P99"]
    ].sort_values(["forecast_hour", "trigger"], kind="stable")
    frame.reset_index(drop=True, inplace=True)
    validate_business_table(
        "future_forecast.csv",
        frame,
        ["forecast_hour", "trigger"],
        ["forecast_hour", "point_forecast", "P50", "P80", "P90", "P95", "P99"],
    )
    percentile_values = frame[["P50", "P80", "P90", "P95", "P99"]].to_numpy()
    if (np.diff(percentile_values, axis=1) < -1e-9).any():
        raise ValueError("future_forecast.csv percentiles are not ordered.")
    return frame


def build_capacity_scenarios() -> pd.DataFrame:
    """Create the hourly aggregate capacity-risk table."""
    source = pd.read_csv(CAPACITY_SCENARIOS_SOURCE)
    required = [
        "forecast_horizon_hour",
        "headroom_percent",
        "point_forecast_compute_seconds",
        "required_capacity_compute_seconds",
        "shortage_probability",
        "expected_shortage_compute_seconds",
        "expected_unused_capacity",
    ]
    require_columns(source, required, CAPACITY_SCENARIOS_SOURCE)
    frame = source[required].rename(
        columns={
            "forecast_horizon_hour": "forecast_hour",
            "headroom_percent": "headroom_pct",
            "point_forecast_compute_seconds": "point_forecast",
            "required_capacity_compute_seconds": "capacity",
        }
    )
    frame.sort_values(["forecast_hour", "headroom_pct"], inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    validate_business_table(
        "capacity_scenarios.csv",
        frame,
        ["forecast_hour", "headroom_pct"],
        list(frame.columns),
    )
    if not frame["shortage_probability"].between(0, 1).all():
        raise ValueError("capacity_scenarios.csv shortage_probability is outside [0, 1].")
    expected_capacity = frame["point_forecast"] * (1 + frame["headroom_pct"] / 100)
    if not np.allclose(frame["capacity"], expected_capacity):
        raise ValueError("capacity_scenarios.csv capacity arithmetic does not reconcile.")
    return frame


def build_capacity_recommendation() -> pd.DataFrame:
    """Copy the validated final recommendation without changing its conclusions."""
    frame = pd.read_csv(CAPACITY_RECOMMENDATION_SOURCE)
    numeric_columns = [
        "headroom_percent",
        "maximum_hourly_shortage_probability",
        "average_shortage_probability",
        "expected_total_shortage",
        "average_unused_capacity",
        "service_target_maximum_hourly_shortage_probability",
        "minimum_recommended_headroom_percent",
    ]
    require_columns(
        frame,
        [
            *numeric_columns,
            "meets_service_target",
            "is_minimum_recommended_headroom",
            "planning_service_target",
            "recommendation_status",
            "capacity_measure",
            "methodology_note",
        ],
        CAPACITY_RECOMMENDATION_SOURCE,
    )
    validate_business_table(
        "capacity_recommendation.csv",
        frame,
        ["headroom_percent"],
        numeric_columns,
    )
    if not frame["maximum_hourly_shortage_probability"].between(0, 1).all():
        raise ValueError("Recommendation shortage probabilities are outside [0, 1].")
    return frame


def main() -> int:
    """Build, validate, and write exactly six Power BI source tables."""
    require_inputs()
    tables = {
        "demand_history.csv": build_demand_history(),
        "model_performance.csv": build_model_performance(),
        "forecast_vs_actual.csv": build_forecast_vs_actual(),
        "future_forecast.csv": build_future_forecast(),
        "capacity_scenarios.csv": build_capacity_scenarios(),
        "capacity_recommendation.csv": build_capacity_recommendation(),
    }
    if set(tables) != set(OUTPUT_PATHS):
        raise AssertionError("Power BI output set does not match the authorized files.")

    for name, frame in tables.items():
        frame.to_csv(OUTPUT_PATHS[name], index=False)

    print("POWER BI EXPORT COMPLETE")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
