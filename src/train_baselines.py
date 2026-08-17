"""Evaluate two leakage-safe forecasting baselines on chronological splits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
VALIDATION_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "baseline_validation_predictions.csv"
)
VALIDATION_METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "baseline_validation_metrics.csv"
)
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "baseline_test_predictions.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "outputs" / "baseline_test_metrics.csv"

TARGET_COLUMN = "compute_seconds"
SEASONAL_PERIOD = 24
SEASONAL_NAIVE = "seasonal_naive"
HOUR_OF_DAY_MEAN = "historical_hour_of_day_mean"
BASELINES = (SEASONAL_NAIVE, HOUR_OF_DAY_MEAN)
SOURCE_COLUMNS = ("hour_index", "hour_of_day", "Trigger", TARGET_COLUMN, "split")
PREDICTION_COLUMNS = (
    "split",
    "baseline",
    "hour_index",
    "hour_of_day",
    "Trigger",
    "actual_compute_seconds",
    "predicted_compute_seconds",
)


def read_split(split_name: str) -> pd.DataFrame:
    """Read one split without loading later splits into the evaluation step."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Model feature dataset does not exist: {INPUT_PATH}")

    frame = pd.read_parquet(
        INPUT_PATH,
        columns=list(SOURCE_COLUMNS),
        filters=[("split", "==", split_name)],
        engine="pyarrow",
    )
    if frame.empty:
        raise ValueError(f"Model feature dataset has no {split_name} observations.")
    observed_splits = set(frame["split"].astype(str).unique())
    if observed_splits != {split_name}:
        raise ValueError(
            f"Expected only the {split_name} split, found {sorted(observed_splits)}."
        )

    frame["Trigger"] = frame["Trigger"].astype("string")
    frame["split"] = frame["split"].astype("string")
    frame["hour_index"] = pd.to_numeric(frame["hour_index"], errors="raise").astype(
        np.int64
    )
    frame["hour_of_day"] = pd.to_numeric(
        frame["hour_of_day"], errors="raise"
    ).astype(np.int8)
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.sort_values(["hour_index", "Trigger"], kind="stable").reset_index(
        drop=True
    )
    validate_split_grid(frame, split_name)
    return frame


def validate_split_grid(frame: pd.DataFrame, split_name: str) -> None:
    """Require finite targets and the same continuous hour grid for every Trigger."""

    if frame.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError(f"The {split_name} split has duplicate Trigger/hour rows.")
    target_values = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError(f"The {split_name} split has missing or non-finite targets.")
    if (target_values < 0).any():
        raise ValueError(f"The {split_name} split has negative targets.")
    if not frame["hour_of_day"].between(0, 23).all():
        raise ValueError(f"The {split_name} split has invalid hour_of_day values.")
    if not np.array_equal(
        frame["hour_of_day"].to_numpy(),
        (frame["hour_index"] % SEASONAL_PERIOD).to_numpy(),
    ):
        raise ValueError(
            f"The {split_name} split hour_of_day does not match hour_index."
        )

    global_hours = np.sort(frame["hour_index"].unique())
    if not np.all(np.diff(global_hours) == 1):
        raise ValueError(f"The {split_name} split hour_index is not continuous.")
    for trigger, trigger_frame in frame.groupby("Trigger", sort=True):
        if not np.array_equal(trigger_frame["hour_index"].to_numpy(), global_hours):
            raise ValueError(
                f"Trigger {trigger!r} does not have the complete {split_name} hour grid."
            )


def assert_split_order(train: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Prove that reference data ends before the evaluation horizon starts."""

    if int(train["hour_index"].max()) >= int(evaluation["hour_index"].min()):
        raise AssertionError("Reference history overlaps the evaluation horizon.")
    if set(train["Trigger"].unique()) != set(evaluation["Trigger"].unique()):
        raise AssertionError("Reference and evaluation Trigger groups differ.")


def actual_history(frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Index observed target values by Trigger and global hour."""

    return {
        (str(row.Trigger), int(row.hour_index)): float(row.compute_seconds)
        for row in frame.itertuples(index=False)
    }


def fixed_origin_seasonal_naive(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> np.ndarray:
    """Forecast from TRAIN only, recursively repeating its last 24-hour cycle.

    The first validation day uses observed TRAIN values at t-24. For later
    validation hours, the unavailable t-24 value is the corresponding prior
    forecast, so no validation actual enters the fixed-origin forecast.
    """

    reference_values = actual_history(train)
    predictions: dict[tuple[str, int], float] = {}

    for hour_index, hour_frame in evaluation.groupby("hour_index", sort=True):
        for row in hour_frame.itertuples(index=False):
            trigger = str(row.Trigger)
            current_hour = int(hour_index)
            reference_hour = current_hour - SEASONAL_PERIOD
            reference_key = (trigger, reference_hour)
            if reference_key in reference_values:
                prediction = reference_values[reference_key]
            elif reference_key in predictions:
                prediction = predictions[reference_key]
            else:
                raise ValueError(
                    "Seasonal-naive validation reference is unavailable for "
                    f"Trigger={trigger}, hour_index={current_hour}."
                )
            predictions[(trigger, current_hour)] = prediction

    return np.array(
        [
            predictions[(str(row.Trigger), int(row.hour_index))]
            for row in evaluation.itertuples(index=False)
        ],
        dtype=np.float64,
    )


def walk_forward_seasonal_naive(
    pre_test_history: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    """Apply actual(t-24) while revealing test actuals only after each origin."""

    history_values = actual_history(pre_test_history)
    predictions: dict[tuple[str, int], float] = {}

    for hour_index, hour_frame in test.groupby("hour_index", sort=True):
        current_hour = int(hour_index)
        reference_hour = current_hour - SEASONAL_PERIOD
        if reference_hour >= current_hour:
            raise AssertionError("Seasonal-naive reference is not strictly historical.")

        for row in hour_frame.itertuples(index=False):
            trigger = str(row.Trigger)
            reference_key = (trigger, reference_hour)
            if reference_key not in history_values:
                raise ValueError(
                    "Seasonal-naive test reference is unavailable for "
                    f"Trigger={trigger}, hour_index={current_hour}."
                )
            predictions[(trigger, current_hour)] = history_values[reference_key]

        # Reveal the current hour only after all predictions at that origin exist.
        for row in hour_frame.itertuples(index=False):
            history_values[(str(row.Trigger), current_hour)] = float(
                row.compute_seconds
            )

    return np.array(
        [
            predictions[(str(row.Trigger), int(row.hour_index))]
            for row in test.itertuples(index=False)
        ],
        dtype=np.float64,
    )


def fit_hour_of_day_means(train: pd.DataFrame) -> pd.Series:
    """Fit one fixed mean per Trigger/hour_of_day using TRAIN observations only."""

    means = train.groupby(["Trigger", "hour_of_day"], sort=True)[
        TARGET_COLUMN
    ].mean()
    if not np.isfinite(means.to_numpy(dtype=np.float64)).all():
        raise ValueError("TRAIN hour-of-day means contain non-finite values.")
    return means


def predict_hour_of_day_mean(
    evaluation: pd.DataFrame, train_means: pd.Series
) -> np.ndarray:
    """Apply the fixed TRAIN-only Trigger/hour_of_day mapping."""

    keys = pd.MultiIndex.from_frame(evaluation[["Trigger", "hour_of_day"]])
    predictions = train_means.reindex(keys).to_numpy(dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("A Trigger/hour_of_day pair has no TRAIN mean.")
    return predictions


def format_predictions(
    evaluation: pd.DataFrame,
    split_name: str,
    baseline_predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Create a tidy prediction table with one row per baseline/Trigger/hour."""

    frames: list[pd.DataFrame] = []
    for baseline in BASELINES:
        predictions = np.asarray(baseline_predictions[baseline], dtype=np.float64)
        if len(predictions) != len(evaluation):
            raise AssertionError(f"{baseline} prediction row count is incorrect.")
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError(f"{baseline} produced invalid predictions.")

        frame = evaluation[["hour_index", "hour_of_day", "Trigger"]].copy()
        frame.insert(0, "baseline", baseline)
        frame.insert(0, "split", split_name)
        frame["actual_compute_seconds"] = evaluation[TARGET_COLUMN].to_numpy(
            dtype=np.float64
        )
        frame["predicted_compute_seconds"] = predictions
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result = result[list(PREDICTION_COLUMNS)].sort_values(
        ["baseline", "hour_index", "Trigger"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = len(evaluation) * len(BASELINES)
    if len(result) != expected_rows:
        raise AssertionError("Prediction output row count is incorrect.")
    return result


def calculate_metrics(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE, and denominator-safe WAPE as a ratio."""

    actual_values = actual.to_numpy(dtype=np.float64)
    prediction_values = prediction.to_numpy(dtype=np.float64)
    errors = actual_values - prediction_values
    absolute_errors = np.abs(errors)
    denominator = float(np.abs(actual_values).sum())
    return {
        "MAE": float(absolute_errors.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "WAPE": float(absolute_errors.sum() / denominator)
        if denominator > 0
        else np.nan,
    }


def evaluate_metrics(predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Calculate metrics by Trigger and after summing total demand by hour."""

    records: list[dict[str, str | float]] = []
    for baseline in BASELINES:
        baseline_frame = predictions.loc[predictions["baseline"] == baseline]
        for trigger, trigger_frame in baseline_frame.groupby("Trigger", sort=True):
            records.append(
                {
                    "split": split_name,
                    "baseline": baseline,
                    "aggregation_level": "Trigger",
                    "Trigger": str(trigger),
                    **calculate_metrics(
                        trigger_frame["actual_compute_seconds"],
                        trigger_frame["predicted_compute_seconds"],
                    ),
                }
            )

        # Total-demand metrics must be calculated after summing all Triggers.
        hourly_totals = baseline_frame.groupby("hour_index", as_index=False).agg(
            actual_compute_seconds=("actual_compute_seconds", "sum"),
            predicted_compute_seconds=("predicted_compute_seconds", "sum"),
        )
        records.append(
            {
                "split": split_name,
                "baseline": baseline,
                "aggregation_level": "total_demand",
                "Trigger": "ALL_TRIGGERS",
                **calculate_metrics(
                    hourly_totals["actual_compute_seconds"],
                    hourly_totals["predicted_compute_seconds"],
                ),
            }
        )

    metrics = pd.DataFrame.from_records(records)
    metrics = metrics.sort_values(
        ["baseline", "aggregation_level", "Trigger"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = len(BASELINES) * (
        predictions["Trigger"].nunique() + 1
    )
    if len(metrics) != expected_rows:
        raise AssertionError("Metric output row count is incorrect.")
    return metrics


def write_outputs(
    validation_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> None:
    """Write exactly the four authorized baseline evaluation outputs."""

    output_pairs = (
        (VALIDATION_PREDICTIONS_PATH, validation_predictions),
        (VALIDATION_METRICS_PATH, validation_metrics),
        (TEST_PREDICTIONS_PATH, test_predictions),
        (TEST_METRICS_PATH, test_metrics),
    )
    for output_path, _ in output_pairs:
        if not output_path.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_path.parent}"
            )
    for output_path, frame in output_pairs:
        frame.to_csv(output_path, index=False)


def print_metrics(split_name: str, metrics: pd.DataFrame) -> None:
    """Print measured metrics without substituting or hard-coding results."""

    print(f"\n{split_name.upper()} measured metrics:")
    print(
        metrics.to_string(
            index=False,
            columns=[
                "baseline",
                "aggregation_level",
                "Trigger",
                "MAE",
                "RMSE",
                "WAPE",
            ],
            float_format=lambda value: f"{value:.6f}",
        )
    )


def main() -> int:
    """Evaluate validation first, then perform the final test evaluation."""

    train = read_split("train")
    validation = read_split("validation")
    assert_split_order(train, validation)

    train_hour_of_day_means = fit_hour_of_day_means(train)
    validation_predictions = format_predictions(
        validation,
        "validation",
        {
            SEASONAL_NAIVE: fixed_origin_seasonal_naive(train, validation),
            HOUR_OF_DAY_MEAN: predict_hour_of_day_mean(
                validation, train_hour_of_day_means
            ),
        },
    )
    validation_metrics = evaluate_metrics(validation_predictions, "validation")

    # Test is read and evaluated only after validation evaluation is complete.
    test = read_split("test")
    assert_split_order(validation, test)
    pre_test_history = pd.concat([train, validation], ignore_index=True)
    test_predictions = format_predictions(
        test,
        "test",
        {
            SEASONAL_NAIVE: walk_forward_seasonal_naive(pre_test_history, test),
            HOUR_OF_DAY_MEAN: predict_hour_of_day_mean(
                test, train_hour_of_day_means
            ),
        },
    )
    test_metrics = evaluate_metrics(test_predictions, "test")

    write_outputs(
        validation_predictions,
        validation_metrics,
        test_predictions,
        test_metrics,
    )

    print("Baseline forecasting evaluation completed.")
    print(f"  validation prediction rows: {len(validation_predictions)}")
    print(f"  validation metric rows: {len(validation_metrics)}")
    print(f"  test prediction rows: {len(test_predictions)}")
    print(f"  test metric rows: {len(test_metrics)}")
    print_metrics("validation", validation_metrics)
    print_metrics("test", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
