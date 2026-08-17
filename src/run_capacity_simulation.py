"""Simulate future workload uncertainty and normalized capacity headroom."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT


OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEMAND_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_demand_by_trigger.parquet"
SELECTED_MODEL_PATH = OUTPUTS_ROOT / "selected_model.json"
BASELINE_VALIDATION_PREDICTIONS_PATH = (
    OUTPUTS_ROOT / "baseline_validation_predictions.csv"
)
BASELINE_TEST_PREDICTIONS_PATH = OUTPUTS_ROOT / "baseline_test_predictions.csv"

FUTURE_FORECAST_PATH = OUTPUTS_ROOT / "future_24h_forecast.csv"
FORECAST_INTERVALS_PATH = OUTPUTS_ROOT / "forecast_intervals.csv"
CAPACITY_SCENARIOS_PATH = OUTPUTS_ROOT / "capacity_scenarios.csv"
CAPACITY_RECOMMENDATION_PATH = OUTPUTS_ROOT / "capacity_recommendation.csv"

SELECTED_MODEL = "Historical Hour-of-Day Mean"
SELECTED_BASELINE_LABEL = "historical_hour_of_day_mean"
HORIZON_HOURS = 24
SIMULATION_COUNT = 10_000
RANDOM_SEED = 42
# P99 estimation needs roughly 100 observations for one expected tail observation.
MIN_HOUR_OF_DAY_RESIDUAL_VECTORS = 100
HEADROOM_PERCENTAGES = (0, 5, 10, 15, 20, 25, 30, 35, 40, 50)
SERVICE_TARGET_MAX_SHORTAGE_PROBABILITY = 0.05
INTERVAL_PERCENTILES = (50, 80, 90, 95, 99)

DEMAND_MEASURE = "workload demand proxy in compute-seconds per hour"
CAPACITY_MEASURE = (
    "required workload-serving capacity in normalized compute-seconds per hour"
)
METHODOLOGY_NOTE = (
    "This analysis demonstrates the workload-serving capacity decision methodology "
    "and does not represent Microsoft's real internal Azure capacity numbers."
)
POOLED_RESIDUAL_LIMITATION = (
    "Only five complete residual vectors were available per hour-of-day, below the "
    "100 needed for one expected observation in the P99 tail; all 120 out-of-sample "
    "residual-hour vectors were pooled instead. This does not preserve hour-of-day-"
    "specific residual behavior, but it preserves contemporaneous cross-Trigger "
    "residual dependence."
)
HOUR_OF_DAY_RESIDUAL_METHOD = (
    "Empirical residual vectors sampled within matching hour-of-day groups."
)
RESIDUAL_SOURCE = "validation and test out-of-sample forecast errors"


def require_inputs() -> None:
    """Fail clearly when a required real-data input is unavailable."""
    required = (
        DEMAND_PATH,
        SELECTED_MODEL_PATH,
        BASELINE_VALIDATION_PREDICTIONS_PATH,
        BASELINE_TEST_PREDICTIONS_PATH,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required simulation inputs are missing:\n{details}")


def load_selected_model() -> str:
    """Read and validate the validation-selected model metadata."""
    metadata = json.loads(SELECTED_MODEL_PATH.read_text(encoding="utf-8"))
    required_fields = {
        "selected_model",
        "validation_MAE",
        "validation_RMSE",
        "validation_WAPE",
        "selection_criterion",
        "test_set_excluded_from_selection",
    }
    missing_fields = sorted(required_fields.difference(metadata))
    if missing_fields:
        raise ValueError(
            f"{SELECTED_MODEL_PATH} is missing fields: {missing_fields}"
        )
    if metadata["selection_criterion"] != "Lowest aggregated validation WAPE":
        raise ValueError(
            "selected_model.json does not document validation-WAPE selection."
        )

    selected_model = str(metadata["selected_model"])
    if selected_model != SELECTED_MODEL:
        raise ValueError(
            "This simulation stage implements the currently selected "
            f"{SELECTED_MODEL!r} forecast, but selected_model.json contains "
            f"{selected_model!r}. Re-run this stage only with matching selected-model "
            "forecast logic."
        )
    return selected_model


def load_observed_demand() -> pd.DataFrame:
    """Load the complete observed hourly Trigger workload history."""
    columns = ["hour_index", "hour_of_day", "Trigger", "compute_seconds"]
    frame = pd.read_parquet(DEMAND_PATH, columns=columns, engine="pyarrow")
    if frame.empty:
        raise ValueError(f"Observed demand dataset is empty: {DEMAND_PATH}")

    frame["hour_index"] = pd.to_numeric(frame["hour_index"], errors="raise").astype(
        np.int64
    )
    frame["hour_of_day"] = pd.to_numeric(
        frame["hour_of_day"], errors="raise"
    ).astype(np.int8)
    frame["Trigger"] = frame["Trigger"].astype("string")
    frame["compute_seconds"] = pd.to_numeric(
        frame["compute_seconds"], errors="coerce"
    )
    frame.sort_values(["hour_index", "Trigger"], inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)

    if frame.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError("Observed demand contains duplicate Trigger/hour rows.")
    values = frame["compute_seconds"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Observed compute_seconds must be finite and nonnegative.")
    if not frame["hour_of_day"].between(0, 23).all():
        raise ValueError("Observed demand contains invalid hour_of_day values.")
    if not np.array_equal(
        frame["hour_of_day"].to_numpy(),
        (frame["hour_index"] % HORIZON_HOURS).to_numpy(),
    ):
        raise ValueError("hour_of_day does not match the continuous hour_index.")

    hours = np.sort(frame["hour_index"].unique())
    if hours[0] != 0 or not np.all(np.diff(hours) == 1):
        raise ValueError("Observed hour_index must be continuous and begin at zero.")
    triggers = sorted(frame["Trigger"].astype(str).unique())
    expected_rows = len(hours) * len(triggers)
    if len(frame) != expected_rows:
        raise ValueError("Observed demand does not contain a complete Trigger/hour grid.")
    return frame


def build_point_forecasts(
    observed: pd.DataFrame, selected_model: str
) -> tuple[pd.DataFrame, list[str]]:
    """Refit the selected hour-of-day mean on all observed trace hours."""
    triggers = sorted(observed["Trigger"].astype(str).unique())
    means = observed.groupby(["Trigger", "hour_of_day"], sort=True)[
        "compute_seconds"
    ].mean()
    expected_pairs = len(triggers) * HORIZON_HOURS
    if len(means) != expected_pairs or not np.isfinite(means.to_numpy()).all():
        raise ValueError("Every Trigger/hour-of-day pair needs a finite historical mean.")

    first_future_hour = int(observed["hour_index"].max()) + 1
    records: list[dict[str, str | int | float]] = []
    for horizon_hour in range(1, HORIZON_HOURS + 1):
        hour_index = first_future_hour + horizon_hour - 1
        hour_of_day = hour_index % HORIZON_HOURS
        for trigger in triggers:
            records.append(
                {
                    "forecast_horizon_hour": horizon_hour,
                    "hour_index": hour_index,
                    "hour_of_day": hour_of_day,
                    "aggregation_level": "Trigger",
                    "Trigger": trigger,
                    "selected_model": selected_model,
                    "point_forecast_compute_seconds": float(
                        means.loc[(trigger, hour_of_day)]
                    ),
                    "demand_measure": DEMAND_MEASURE,
                }
            )

    trigger_forecasts = pd.DataFrame.from_records(records)
    total_forecasts = (
        trigger_forecasts.groupby(
            ["forecast_horizon_hour", "hour_index", "hour_of_day"],
            as_index=False,
            sort=True,
        )["point_forecast_compute_seconds"]
        .sum()
        .assign(
            aggregation_level="total_demand",
            Trigger="ALL_TRIGGERS",
            selected_model=selected_model,
            demand_measure=DEMAND_MEASURE,
        )
    )
    forecasts = pd.concat([trigger_forecasts, total_forecasts], ignore_index=True)
    forecasts.sort_values(
        ["hour_index", "aggregation_level", "Trigger"],
        inplace=True,
        kind="stable",
    )
    forecasts.reset_index(drop=True, inplace=True)

    if (forecasts["point_forecast_compute_seconds"] < 0).any():
        raise AssertionError("Point forecasts must be nonnegative.")
    return forecasts, triggers


def load_residual_vectors(
    triggers: list[str], selected_model: str
) -> tuple[np.ndarray, np.ndarray, str, str, dict[int, int]]:
    """Load real out-of-sample residual vectors for empirical resampling."""
    frames: list[pd.DataFrame] = []
    for split, path in (
        ("validation", BASELINE_VALIDATION_PREDICTIONS_PATH),
        ("test", BASELINE_TEST_PREDICTIONS_PATH),
    ):
        source = pd.read_csv(path)
        required = {
            "split",
            "baseline",
            "hour_index",
            "hour_of_day",
            "Trigger",
            "actual_compute_seconds",
            "predicted_compute_seconds",
        }
        missing = sorted(required.difference(source.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        selected = source.loc[
            (source["baseline"] == SELECTED_BASELINE_LABEL)
            & (source["split"] == split)
        ].copy()
        if selected.empty:
            raise ValueError(f"{path} has no residual source rows for {selected_model}.")
        frames.append(selected)

    residuals = pd.concat(frames, ignore_index=True)
    residuals["residual"] = (
        residuals["actual_compute_seconds"]
        - residuals["predicted_compute_seconds"]
    )
    if set(residuals["Trigger"].astype(str)) != set(triggers):
        raise ValueError("Residual and observed-demand Trigger groups differ.")
    if residuals.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError("Residual history contains duplicate Trigger/hour rows.")
    if not np.isfinite(residuals["residual"].to_numpy(dtype=float)).all():
        raise ValueError("Residual history contains non-finite values.")

    hour_metadata = residuals[["hour_index", "hour_of_day"]].drop_duplicates()
    if hour_metadata.duplicated("hour_index").any():
        raise ValueError("A residual hour maps to multiple hour-of-day values.")
    pivot = residuals.pivot(index="hour_index", columns="Trigger", values="residual")
    pivot = pivot.reindex(columns=triggers).sort_index()
    if pivot.isna().any().any():
        raise ValueError("Residual history does not form complete hourly vectors.")

    hour_of_day_by_hour = hour_metadata.set_index("hour_index")["hour_of_day"]
    residual_hour_of_day = (
        hour_of_day_by_hour.reindex(pivot.index).to_numpy(dtype=np.int8)
    )
    counts = pd.Series(residual_hour_of_day).value_counts().sort_index()
    counts_by_hour = {int(hour): int(counts.get(hour, 0)) for hour in range(24)}
    use_hour_of_day = min(counts_by_hour.values()) >= MIN_HOUR_OF_DAY_RESIDUAL_VECTORS
    if use_hour_of_day:
        method = "hour_of_day_empirical_residual_vector_sampling"
        limitation = HOUR_OF_DAY_RESIDUAL_METHOD
    else:
        method = "pooled_empirical_residual_vector_sampling"
        limitation = POOLED_RESIDUAL_LIMITATION
    return (
        pivot.to_numpy(dtype=float),
        residual_hour_of_day,
        method,
        limitation,
        counts_by_hour,
    )


def simulate_demand(
    point_forecasts: pd.DataFrame,
    triggers: list[str],
    residual_vectors: np.ndarray,
    residual_hour_of_day: np.ndarray,
    residual_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate empirical Trigger and aggregate demand simulations."""
    trigger_points = (
        point_forecasts.loc[point_forecasts["aggregation_level"] == "Trigger"]
        .pivot(
            index="forecast_horizon_hour",
            columns="Trigger",
            values="point_forecast_compute_seconds",
        )
        .reindex(index=range(1, HORIZON_HOURS + 1), columns=triggers)
    )
    if trigger_points.isna().any().any():
        raise ValueError("Future Trigger point-forecast grid is incomplete.")

    rng = np.random.default_rng(RANDOM_SEED)
    future_hour_of_day = (
        point_forecasts.loc[
            point_forecasts["aggregation_level"] == "total_demand"
        ]
        .sort_values("forecast_horizon_hour", kind="stable")["hour_of_day"]
        .to_numpy(dtype=np.int8)
    )
    sampled = np.empty(
        (HORIZON_HOURS, SIMULATION_COUNT, len(triggers)), dtype=np.float64
    )
    for future_offset in range(HORIZON_HOURS):
        if residual_method == "hour_of_day_empirical_residual_vector_sampling":
            candidates = np.flatnonzero(
                residual_hour_of_day == future_hour_of_day[future_offset]
            )
        else:
            candidates = np.arange(len(residual_vectors))
        if not len(candidates):
            raise ValueError("No empirical residual vectors are available for sampling.")
        sampled_indices = rng.choice(candidates, size=SIMULATION_COUNT, replace=True)
        sampled[future_offset] = residual_vectors[sampled_indices]

    trigger_simulations = np.maximum(
        trigger_points.to_numpy(dtype=float)[:, np.newaxis, :] + sampled,
        0.0,
    )
    total_simulations = trigger_simulations.sum(axis=2)
    if not np.isfinite(trigger_simulations).all():
        raise AssertionError("Monte Carlo demand simulations must be finite.")
    return trigger_simulations, total_simulations


def build_forecast_intervals(
    point_forecasts: pd.DataFrame,
    triggers: list[str],
    trigger_simulations: np.ndarray,
    total_simulations: np.ndarray,
    residual_method: str,
    residual_limitation: str,
    residual_observation_hours: int,
) -> pd.DataFrame:
    """Summarize empirical simulation percentiles for every workload series."""
    trigger_index = {trigger: index for index, trigger in enumerate(triggers)}
    records: list[dict[str, str | int | float]] = []
    for row in point_forecasts.itertuples(index=False):
        horizon_index = int(row.forecast_horizon_hour) - 1
        if row.aggregation_level == "Trigger":
            simulations = trigger_simulations[
                horizon_index, :, trigger_index[str(row.Trigger)]
            ]
        else:
            simulations = total_simulations[horizon_index]
        quantiles = np.percentile(simulations, INTERVAL_PERCENTILES)
        record: dict[str, str | int | float] = {
            "forecast_horizon_hour": int(row.forecast_horizon_hour),
            "hour_index": int(row.hour_index),
            "hour_of_day": int(row.hour_of_day),
            "aggregation_level": str(row.aggregation_level),
            "Trigger": str(row.Trigger),
            "selected_model": str(row.selected_model),
            "point_forecast_compute_seconds": float(
                row.point_forecast_compute_seconds
            ),
        }
        record.update(
            {
                f"simulated_P{percentile}_compute_seconds": float(value)
                for percentile, value in zip(INTERVAL_PERCENTILES, quantiles)
            }
        )
        record.update(
            {
                "simulation_count": SIMULATION_COUNT,
                "random_seed": RANDOM_SEED,
                "residual_observation_hours": residual_observation_hours,
                "residual_source": RESIDUAL_SOURCE,
                "residual_sampling_method": residual_method,
                "residual_sampling_limitation": residual_limitation,
                "demand_measure": DEMAND_MEASURE,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_capacity_scenarios(
    point_forecasts: pd.DataFrame, total_simulations: np.ndarray
) -> pd.DataFrame:
    """Evaluate hourly shortage and unused-capacity outcomes by headroom."""
    total_points = (
        point_forecasts.loc[
            point_forecasts["aggregation_level"] == "total_demand"
        ]
        .sort_values("forecast_horizon_hour", kind="stable")
        .reset_index(drop=True)
    )
    if len(total_points) != HORIZON_HOURS:
        raise ValueError("Expected one aggregate point forecast per future hour.")

    records: list[dict[str, str | int | float]] = []
    for row in total_points.itertuples(index=False):
        horizon_index = int(row.forecast_horizon_hour) - 1
        simulated_demand = total_simulations[horizon_index]
        point_forecast = float(row.point_forecast_compute_seconds)
        for headroom_percent in HEADROOM_PERCENTAGES:
            capacity = point_forecast * (1.0 + headroom_percent / 100.0)
            shortage = np.maximum(simulated_demand - capacity, 0.0)
            unused_capacity = np.maximum(capacity - simulated_demand, 0.0)
            records.append(
                {
                    "forecast_horizon_hour": int(row.forecast_horizon_hour),
                    "hour_index": int(row.hour_index),
                    "hour_of_day": int(row.hour_of_day),
                    "headroom_percent": headroom_percent,
                    "point_forecast_compute_seconds": point_forecast,
                    "required_capacity_compute_seconds": capacity,
                    "shortage_probability": float(np.mean(shortage > 0.0)),
                    "expected_shortage_compute_seconds": float(shortage.mean()),
                    "p95_shortage_compute_seconds": float(
                        np.percentile(shortage, 95)
                    ),
                    "expected_unused_capacity": float(unused_capacity.mean()),
                    "simulation_count": SIMULATION_COUNT,
                    "capacity_measure": CAPACITY_MEASURE,
                    "methodology_note": METHODOLOGY_NOTE,
                }
            )
    return pd.DataFrame.from_records(records)


def build_capacity_recommendation(capacity_scenarios: pd.DataFrame) -> pd.DataFrame:
    """Summarize each headroom and flag the minimum passing service target."""
    summary = (
        capacity_scenarios.groupby("headroom_percent", as_index=False, sort=True)
        .agg(
            maximum_hourly_shortage_probability=("shortage_probability", "max"),
            average_shortage_probability=("shortage_probability", "mean"),
            expected_total_shortage=(
                "expected_shortage_compute_seconds",
                "sum",
            ),
            average_unused_capacity=(
                "expected_unused_capacity",
                "mean",
            ),
        )
        .sort_values("headroom_percent", kind="stable")
        .reset_index(drop=True)
    )
    summary["service_target_maximum_hourly_shortage_probability"] = (
        SERVICE_TARGET_MAX_SHORTAGE_PROBABILITY
    )
    summary["meets_service_target"] = (
        summary["maximum_hourly_shortage_probability"]
        <= SERVICE_TARGET_MAX_SHORTAGE_PROBABILITY
    )

    passing = summary.loc[summary["meets_service_target"], "headroom_percent"]
    if passing.empty:
        recommended_headroom: float | None = None
        recommendation_status = (
            "No tested headroom satisfies maximum hourly shortage probability <= 5%."
        )
    else:
        recommended_headroom = float(passing.min())
        recommendation_status = (
            f"Minimum tested headroom satisfying the service target: "
            f"{recommended_headroom:g}%."
        )

    summary["is_minimum_recommended_headroom"] = False
    if recommended_headroom is not None:
        summary.loc[
            summary["headroom_percent"] == recommended_headroom,
            "is_minimum_recommended_headroom",
        ] = True
    summary["minimum_recommended_headroom_percent"] = recommended_headroom
    summary["planning_service_target"] = (
        "maximum hourly shortage probability <= 5%"
    )
    summary["recommendation_status"] = recommendation_status
    summary["capacity_measure"] = CAPACITY_MEASURE
    summary["methodology_note"] = METHODOLOGY_NOTE
    return summary


def validate_outputs(
    forecasts: pd.DataFrame,
    intervals: pd.DataFrame,
    scenarios: pd.DataFrame,
    recommendation: pd.DataFrame,
    triggers: list[str],
) -> None:
    """Assert output grids, nonnegativity, and capacity arithmetic."""
    expected_forecast_rows = HORIZON_HOURS * (len(triggers) + 1)
    if len(forecasts) != expected_forecast_rows or len(intervals) != expected_forecast_rows:
        raise AssertionError("Forecast output row counts are incorrect.")
    if len(scenarios) != HORIZON_HOURS * len(HEADROOM_PERCENTAGES):
        raise AssertionError("Capacity scenario row count is incorrect.")
    if len(recommendation) != len(HEADROOM_PERCENTAGES):
        raise AssertionError("Capacity recommendation row count is incorrect.")

    numeric_forecast_columns = [
        "point_forecast_compute_seconds",
        *[
            f"simulated_P{percentile}_compute_seconds"
            for percentile in INTERVAL_PERCENTILES
        ],
    ]
    if (intervals[numeric_forecast_columns] < 0).any().any():
        raise AssertionError("Forecast outputs cannot contain negative demand.")
    interval_values = intervals[
        [
            f"simulated_P{percentile}_compute_seconds"
            for percentile in INTERVAL_PERCENTILES
        ]
    ].to_numpy()
    if (np.diff(interval_values, axis=1) < -1e-9).any():
        raise AssertionError("Simulated percentiles must be monotonically ordered.")
    if not scenarios["shortage_probability"].between(0, 1).all():
        raise AssertionError("Shortage probabilities must be between zero and one.")
    if (
        scenarios[
            [
                "required_capacity_compute_seconds",
                "expected_shortage_compute_seconds",
                "p95_shortage_compute_seconds",
                "expected_unused_capacity",
            ]
        ]
        < 0
    ).any().any():
        raise AssertionError("Capacity outcome values cannot be negative.")

    expected_capacity = scenarios["point_forecast_compute_seconds"] * (
        1.0 + scenarios["headroom_percent"] / 100.0
    )
    if not np.allclose(
        scenarios["required_capacity_compute_seconds"], expected_capacity
    ):
        raise AssertionError("Capacity does not reconcile to point forecast and headroom.")
    passing = recommendation.loc[recommendation["meets_service_target"]]
    flagged = recommendation.loc[
        recommendation["is_minimum_recommended_headroom"]
    ]
    if passing.empty and not flagged.empty:
        raise AssertionError("No recommendation should be flagged when none passes.")
    if not passing.empty:
        if len(flagged) != 1 or flagged.iloc[0]["headroom_percent"] != passing[
            "headroom_percent"
        ].min():
            raise AssertionError("The minimum passing headroom was not flagged.")


def main() -> int:
    """Run the selected-model uncertainty and normalized capacity analysis."""
    require_inputs()
    selected_model = load_selected_model()
    observed = load_observed_demand()
    forecasts, triggers = build_point_forecasts(observed, selected_model)
    (
        residual_vectors,
        residual_hour_of_day,
        residual_method,
        residual_limitation,
        counts_by_hour,
    ) = load_residual_vectors(triggers, selected_model)
    trigger_simulations, total_simulations = simulate_demand(
        forecasts,
        triggers,
        residual_vectors,
        residual_hour_of_day,
        residual_method,
    )
    intervals = build_forecast_intervals(
        forecasts,
        triggers,
        trigger_simulations,
        total_simulations,
        residual_method,
        residual_limitation,
        len(residual_vectors),
    )
    scenarios = build_capacity_scenarios(forecasts, total_simulations)
    recommendation = build_capacity_recommendation(scenarios)
    validate_outputs(forecasts, intervals, scenarios, recommendation, triggers)

    forecasts.to_csv(FUTURE_FORECAST_PATH, index=False)
    intervals.to_csv(FORECAST_INTERVALS_PATH, index=False)
    scenarios.to_csv(CAPACITY_SCENARIOS_PATH, index=False)
    recommendation.to_csv(CAPACITY_RECOMMENDATION_PATH, index=False)

    recommended = recommendation.loc[
        recommendation["is_minimum_recommended_headroom"]
    ]
    print("CAPACITY SIMULATION COMPLETE")
    print(f"Selected model: {selected_model}")
    print(
        f"Future horizon: hour_index {forecasts['hour_index'].min()} through "
        f"{forecasts['hour_index'].max()}"
    )
    print(f"Monte Carlo simulations: {SIMULATION_COUNT}")
    print(f"Residual observation hours: {len(residual_vectors)}")
    print(f"Residual vectors per hour-of-day: {sorted(set(counts_by_hour.values()))}")
    print(f"Residual sampling method: {residual_method}")
    print(f"Residual sampling limitation: {residual_limitation}")
    print(f"Capacity measure: {CAPACITY_MEASURE}")
    print(METHODOLOGY_NOTE)
    print("\nHorizon-level capacity results:")
    print(
        recommendation[
            [
                "headroom_percent",
                "maximum_hourly_shortage_probability",
                "average_shortage_probability",
                "expected_total_shortage",
                "average_unused_capacity",
                "meets_service_target",
            ]
        ].to_string(index=False)
    )
    if recommended.empty:
        print(
            "\nRecommendation: no tested headroom satisfies maximum hourly "
            "shortage probability <= 5%."
        )
    else:
        headroom = float(recommended.iloc[0]["headroom_percent"])
        print(f"\nRecommendation: minimum tested headroom is {headroom:g}%.")
    print("\nOutput shapes:")
    print(f"  future_24h_forecast.csv: {forecasts.shape}")
    print(f"  forecast_intervals.csv: {intervals.shape}")
    print(f"  capacity_scenarios.csv: {scenarios.shape}")
    print(f"  capacity_recommendation.csv: {recommendation.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
