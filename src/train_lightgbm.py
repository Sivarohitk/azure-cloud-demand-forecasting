"""Select and evaluate one global leakage-safe LightGBM demand model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from build_features import (
    FEATURE_COLUMNS,
    FEATURE_OUTPUT_PATH,
    LAG_HOURS,
    ROLLING_MEAN_HOURS,
    ROLLING_STD_HOURS,
)
from config import PROJECT_ROOT


TARGET_HISTORY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand_by_trigger.parquet"
)
VALIDATION_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "lightgbm_validation_predictions.csv"
)
VALIDATION_METRICS_PATH = (
    PROJECT_ROOT / "outputs" / "lightgbm_validation_metrics.csv"
)
TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "lightgbm_test_predictions.csv"
)
TEST_METRICS_PATH = PROJECT_ROOT / "outputs" / "lightgbm_test_metrics.csv"
FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT / "outputs" / "lightgbm_feature_importance.csv"
)

TARGET_COLUMN = "compute_seconds"
MODEL_NAME = "LightGBM"
FORECAST_WINDOW_HOURS = 24
RANDOM_STATE = 42
PREDICTOR_COLUMNS = ("Trigger", *FEATURE_COLUMNS)

# Four declared configurations form a compact, deterministic comparison. No
# automatic search or test-driven tuning is performed.
CANDIDATE_CONFIGS = (
    {
        "candidate_id": "lgbm_1",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 20,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    {
        "candidate_id": "lgbm_2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 20,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    },
    {
        "candidate_id": "lgbm_3",
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "min_child_samples": 10,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
    },
    {
        "candidate_id": "lgbm_4",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "n_estimators": 400,
        "min_child_samples": 30,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
    },
)


@dataclass
class ForecastResult:
    """Predictions and fitted-model facts from a rolling evaluation."""

    predictions: pd.DataFrame
    first_window_model: LGBMRegressor


def read_model_split(split_name: str) -> pd.DataFrame:
    """Read one feature split without loading later targets prematurely."""

    if not FEATURE_OUTPUT_PATH.is_file():
        raise FileNotFoundError(f"Feature dataset does not exist: {FEATURE_OUTPUT_PATH}")
    required_columns = [
        "hour_index",
        "Trigger",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
        "split",
    ]
    frame = pd.read_parquet(
        FEATURE_OUTPUT_PATH,
        columns=required_columns,
        filters=[("split", "==", split_name)],
        engine="pyarrow",
    )
    if frame.empty:
        raise ValueError(f"Feature dataset has no {split_name} observations.")
    frame["Trigger"] = frame["Trigger"].astype("string")
    frame["split"] = frame["split"].astype("string")
    frame = frame.sort_values(["hour_index", "Trigger"], kind="stable").reset_index(
        drop=True
    )
    validate_model_split(frame, split_name)
    return frame


def validate_model_split(frame: pd.DataFrame, split_name: str) -> None:
    """Validate allowed columns, target values, and shared chronological grids."""

    if set(frame["split"].unique()) != {split_name}:
        raise ValueError(f"Unexpected split labels while reading {split_name}.")
    if frame.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError(f"The {split_name} split has duplicate Trigger/hour rows.")
    values = frame[[TARGET_COLUMN, *FEATURE_COLUMNS]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"The {split_name} target or features are non-finite.")
    if (frame[TARGET_COLUMN] < 0).any():
        raise ValueError(f"The {split_name} target contains negative demand.")

    global_hours = np.sort(frame["hour_index"].unique())
    if not np.all(np.diff(global_hours) == 1):
        raise ValueError(f"The {split_name} hour_index is not continuous.")
    for trigger, trigger_frame in frame.groupby("Trigger", sort=True):
        if not np.array_equal(trigger_frame["hour_index"].to_numpy(), global_hours):
            raise ValueError(
                f"Trigger {trigger!r} does not have the full {split_name} hour grid."
            )


def assert_split_order(earlier: pd.DataFrame, later: pd.DataFrame) -> None:
    """Prove that the later evaluation split starts after all earlier rows."""

    if int(earlier["hour_index"].max()) >= int(later["hour_index"].min()):
        raise AssertionError("Chronological splits overlap or are out of order.")
    if set(earlier["Trigger"].unique()) != set(later["Trigger"].unique()):
        raise AssertionError("Chronological splits contain different Triggers.")


def read_target_history(
    end_hour_index: int, trigger_categories: tuple[str, ...]
) -> dict[str, list[float]]:
    """Read actual target history only through the requested forecast origin."""

    if not TARGET_HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Hourly demand history does not exist: {TARGET_HISTORY_PATH}"
        )
    frame = pd.read_parquet(
        TARGET_HISTORY_PATH,
        columns=["hour_index", "Trigger", TARGET_COLUMN],
        filters=[("hour_index", "<=", end_hour_index)],
        engine="pyarrow",
    )
    frame["Trigger"] = frame["Trigger"].astype("string")
    frame = frame.sort_values(["Trigger", "hour_index"], kind="stable")
    if frame.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError("Target history has duplicate Trigger/hour rows.")

    expected_hours = np.arange(end_hour_index + 1, dtype=np.int64)
    history: dict[str, list[float]] = {}
    for trigger in trigger_categories:
        trigger_frame = frame.loc[frame["Trigger"] == trigger]
        if not np.array_equal(trigger_frame["hour_index"].to_numpy(), expected_hours):
            raise ValueError(
                f"Target history for Trigger={trigger} is incomplete through "
                f"hour_index={end_hour_index}."
            )
        values = trigger_frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Target history for Trigger={trigger} is invalid.")
        history[trigger] = values.tolist()
    return history


def clone_history(history: dict[str, list[float]]) -> dict[str, list[float]]:
    """Copy recursive target histories so candidate evaluations are independent."""

    return {trigger: values.copy() for trigger, values in history.items()}


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio only when its historical denominator is positive."""

    return float(numerator / denominator) if denominator > 0 else np.nan


def recursive_feature_rows(
    hour_index: int,
    trigger_categories: tuple[str, ...],
    history: dict[str, list[float]],
) -> pd.DataFrame:
    """Rebuild the approved features using only history before ``hour_index``."""

    for trigger in trigger_categories:
        if len(history[trigger]) != hour_index:
            raise AssertionError(
                f"History for Trigger={trigger} does not end at forecast origin "
                f"hour_index={hour_index}."
            )

    lag24_total = sum(
        history[trigger][hour_index - 24] for trigger in trigger_categories
    )
    records: list[dict[str, float | int | str]] = []
    hour_of_day = hour_index % 24
    angle = 2.0 * np.pi * hour_of_day / 24.0

    for trigger in trigger_categories:
        values = history[trigger]
        record: dict[str, float | int | str] = {
            "Trigger": trigger,
            "hour_of_day": hour_of_day,
            "hour_sin": float(np.sin(angle)),
            "hour_cos": float(np.cos(angle)),
            "trace_day_index": hour_index // 24,
        }
        for lag in LAG_HOURS:
            record[f"lag_{lag}"] = values[hour_index - lag]
        for window in ROLLING_MEAN_HOURS:
            record[f"rolling_mean_{window}"] = float(
                np.mean(values[-window:])
            )
        for window in ROLLING_STD_HOURS:
            record[f"rolling_std_{window}"] = float(
                np.std(values[-window:], ddof=1)
            )
        record["rolling_max_24"] = float(np.max(values[-24:]))
        record["rolling_min_24"] = float(np.min(values[-24:]))
        historical_p95 = float(np.quantile(values, 0.95))
        record["historical_load_ratio"] = safe_ratio(values[-1], historical_p95)
        record["lag24_trigger_share"] = safe_ratio(
            values[hour_index - 24], lag24_total
        )
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    return frame[[*PREDICTOR_COLUMNS]]


def encode_predictors(
    frame: pd.DataFrame, trigger_categories: tuple[str, ...]
) -> pd.DataFrame:
    """Apply one stable categorical encoding used by every global model fit."""

    predictors = frame[list(PREDICTOR_COLUMNS)].copy()
    predictors["Trigger"] = pd.Categorical(
        predictors["Trigger"], categories=list(trigger_categories)
    )
    if predictors["Trigger"].isna().any():
        raise ValueError("A Trigger is missing from the TRAIN categorical vocabulary.")
    return predictors


def candidate_parameters(config: dict[str, object]) -> dict[str, object]:
    """Return only LightGBM estimator parameters from a declared candidate."""

    return {key: value for key, value in config.items() if key != "candidate_id"}


def fit_global_model(
    training_rows: pd.DataFrame,
    trigger_categories: tuple[str, ...],
    config: dict[str, object],
) -> LGBMRegressor:
    """Fit one global model across all Trigger workload observations."""

    model = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        **candidate_parameters(config),
    )
    model.fit(
        encode_predictors(training_rows, trigger_categories),
        training_rows[TARGET_COLUMN].to_numpy(dtype=np.float64),
        categorical_feature=["Trigger"],
    )
    return model


def validate_first_hour_features(
    recursive_rows: pd.DataFrame,
    evaluation: pd.DataFrame,
    hour_index: int,
) -> None:
    """Reconcile each window's first features with build_features.py output."""

    expected = evaluation.loc[evaluation["hour_index"] == hour_index].sort_values(
        "Trigger"
    )
    observed = recursive_rows.sort_values("Trigger")
    if not np.array_equal(
        observed["Trigger"].astype(str).to_numpy(),
        expected["Trigger"].astype(str).to_numpy(),
    ):
        raise AssertionError("Recursive and stored Trigger rows do not align.")
    if not np.allclose(
        observed[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64),
        expected[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64),
        # NumPy and pandas use slightly different floating algorithms for
        # sample standard deviation; the formulas still agree numerically.
        rtol=1e-9,
        atol=1e-5,
        equal_nan=True,
    ):
        raise AssertionError(
            f"Recursive features do not match stored historical features at "
            f"hour_index={hour_index}."
        )


def rolling_recursive_forecast(
    initial_training_rows: pd.DataFrame,
    evaluation: pd.DataFrame,
    initial_history: dict[str, list[float]],
    trigger_categories: tuple[str, ...],
    config: dict[str, object],
    split_name: str,
) -> ForecastResult:
    """Fit and forecast in 24-hour windows with recursive within-window features."""

    training_rows = initial_training_rows.copy()
    history = clone_history(initial_history)
    evaluation_lookup = evaluation.set_index(["hour_index", "Trigger"])
    evaluation_hours = np.sort(evaluation["hour_index"].unique())
    prediction_records: list[dict[str, object]] = []
    first_window_model: LGBMRegressor | None = None

    for start in range(0, len(evaluation_hours), FORECAST_WINDOW_HOURS):
        window_hours = evaluation_hours[start : start + FORECAST_WINDOW_HOURS]
        window_number = start // FORECAST_WINDOW_HOURS + 1
        model = fit_global_model(training_rows, trigger_categories, config)
        if first_window_model is None:
            first_window_model = model

        for offset, hour_index_value in enumerate(window_hours):
            hour_index = int(hour_index_value)
            recursive_rows = recursive_feature_rows(
                hour_index, trigger_categories, history
            )
            if offset == 0:
                validate_first_hour_features(
                    recursive_rows, evaluation, hour_index
                )
            raw_predictions = np.asarray(
                model.predict(
                    encode_predictors(recursive_rows, trigger_categories)
                ),
                dtype=np.float64,
            )
            if not np.isfinite(raw_predictions).all():
                raise ValueError("LightGBM produced non-finite predictions.")
            clipped_mask = raw_predictions < 0
            predictions = np.clip(raw_predictions, 0.0, None)

            for trigger, prediction, was_clipped in zip(
                trigger_categories, predictions, clipped_mask, strict=True
            ):
                actual = float(
                    evaluation_lookup.loc[(hour_index, trigger), TARGET_COLUMN]
                )
                prediction_records.append(
                    {
                        "split": split_name,
                        "model": MODEL_NAME,
                        "candidate_id": str(config["candidate_id"]),
                        "hour_index": hour_index,
                        "hour_of_day": hour_index % 24,
                        "forecast_window": window_number,
                        "Trigger": trigger,
                        "actual_compute_seconds": actual,
                        "predicted_compute_seconds": float(prediction),
                        "was_clipped_to_zero": bool(was_clipped),
                    }
                )
                history[trigger].append(float(prediction))

        # The completed window is now observed. Replace recursive predictions
        # with actuals and make those actual feature rows available for refit.
        completed_rows = evaluation.loc[
            evaluation["hour_index"].isin(window_hours)
        ].copy()
        for row in completed_rows.itertuples(index=False):
            history[str(row.Trigger)][int(row.hour_index)] = float(
                row.compute_seconds
            )
        training_rows = pd.concat(
            [training_rows, completed_rows], ignore_index=True
        )

    if first_window_model is None:
        raise AssertionError("No LightGBM forecast window was fitted.")
    predictions = pd.DataFrame.from_records(prediction_records).sort_values(
        ["hour_index", "Trigger"], kind="stable"
    ).reset_index(drop=True)
    if len(predictions) != len(evaluation):
        raise AssertionError(f"{split_name} prediction row count is incorrect.")
    if (predictions["predicted_compute_seconds"] < 0).any():
        raise AssertionError(f"{split_name} contains negative final predictions.")
    return ForecastResult(predictions, first_window_model)


def calculate_metrics(
    actual: pd.Series, prediction: pd.Series
) -> dict[str, float]:
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


def aggregated_wape(predictions: pd.DataFrame) -> float:
    """Calculate validation WAPE after summing Trigger demand by hour."""

    totals = predictions.groupby("hour_index", as_index=False).agg(
        actual_compute_seconds=("actual_compute_seconds", "sum"),
        predicted_compute_seconds=("predicted_compute_seconds", "sum"),
    )
    return calculate_metrics(
        totals["actual_compute_seconds"], totals["predicted_compute_seconds"]
    )["WAPE"]


def evaluate_metrics(predictions: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Calculate metrics by Trigger and after summing total demand by hour."""

    records: list[dict[str, str | float]] = []
    for trigger, trigger_frame in predictions.groupby("Trigger", sort=True):
        records.append(
            {
                "split": split_name,
                "model": MODEL_NAME,
                "aggregation_level": "Trigger",
                "Trigger": str(trigger),
                **calculate_metrics(
                    trigger_frame["actual_compute_seconds"],
                    trigger_frame["predicted_compute_seconds"],
                ),
            }
        )
    totals = predictions.groupby("hour_index", as_index=False).agg(
        actual_compute_seconds=("actual_compute_seconds", "sum"),
        predicted_compute_seconds=("predicted_compute_seconds", "sum"),
    )
    records.append(
        {
            "split": split_name,
            "model": MODEL_NAME,
            "aggregation_level": "total_demand",
            "Trigger": "ALL_TRIGGERS",
            **calculate_metrics(
                totals["actual_compute_seconds"],
                totals["predicted_compute_seconds"],
            ),
        }
    )
    return pd.DataFrame.from_records(records).sort_values(
        ["aggregation_level", "Trigger"], kind="stable"
    ).reset_index(drop=True)


def select_configuration(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_history: dict[str, list[float]],
    trigger_categories: tuple[str, ...],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Select one configuration using aggregated validation WAPE only."""

    candidate_results: list[tuple[float, dict[str, object], pd.DataFrame]] = []
    for config in CANDIDATE_CONFIGS:
        result = rolling_recursive_forecast(
            train,
            validation,
            train_history,
            trigger_categories,
            config,
            "validation",
        )
        wape = aggregated_wape(result.predictions)
        if not np.isfinite(wape):
            raise ValueError(
                f"Candidate {config['candidate_id']} produced non-finite validation WAPE."
            )
        candidate_results.append((wape, config, result.predictions))
        print(
            f"CANDIDATE {config['candidate_id']} "
            f"aggregated_validation_WAPE={wape:.12f} "
            f"clipped={int(result.predictions['was_clipped_to_zero'].sum())} "
            f"parameters={candidate_parameters(config)}"
        )

    selected_wape, selected_config, selected_predictions = min(
        candidate_results, key=lambda result: result[0]
    )
    print(
        f"SELECTED {selected_config['candidate_id']} "
        f"aggregated_validation_WAPE={selected_wape:.12f} "
        f"parameters={candidate_parameters(selected_config)}"
    )
    return selected_config, selected_predictions


def build_feature_importance(
    model: LGBMRegressor,
    selected_config: dict[str, object],
    training_end_hour_index: int,
) -> pd.DataFrame:
    """Export LightGBM gain importance from the pre-test global model."""

    feature_names = model.booster_.feature_name()
    gains = model.booster_.feature_importance(importance_type="gain")
    total_gain = float(gains.sum())
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_type": "gain",
            "gain": gains.astype(np.float64),
            "gain_percentage": (
                100.0 * gains / total_gain
                if total_gain > 0
                else np.zeros_like(gains)
            ),
            "selected_candidate_id": str(selected_config["candidate_id"]),
            "training_end_hour_index": training_end_hour_index,
        }
    )
    importance = importance.sort_values("gain", ascending=False, kind="stable").reset_index(
        drop=True
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    if set(importance["feature"]) != set(PREDICTOR_COLUMNS):
        raise AssertionError("Feature importance does not cover all approved predictors.")
    return importance


def write_outputs(
    validation_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    """Write exactly the five authorized LightGBM outputs."""

    output_pairs = (
        (VALIDATION_PREDICTIONS_PATH, validation_predictions),
        (VALIDATION_METRICS_PATH, validation_metrics),
        (TEST_PREDICTIONS_PATH, test_predictions),
        (TEST_METRICS_PATH, test_metrics),
        (FEATURE_IMPORTANCE_PATH, feature_importance),
    )
    for output_path, _ in output_pairs:
        if not output_path.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_path.parent}"
            )
    for output_path, frame in output_pairs:
        frame.to_csv(output_path, index=False)


def print_metrics(split_name: str, metrics: pd.DataFrame) -> None:
    """Print real measured metrics from the completed evaluation."""

    print(f"\n{split_name.upper()} measured LightGBM metrics:")
    print(
        metrics.to_string(
            index=False,
            columns=["aggregation_level", "Trigger", "MAE", "RMSE", "WAPE"],
            float_format=lambda value: f"{value:.6f}",
        )
    )


def main() -> int:
    """Tune on validation, then open and evaluate the untouched test split."""

    if len(CANDIDATE_CONFIGS) > 6:
        raise AssertionError("LightGBM candidate count exceeds the allowed maximum.")

    train = read_model_split("train")
    validation = read_model_split("validation")
    assert_split_order(train, validation)
    trigger_categories = tuple(sorted(train["Trigger"].unique()))
    train_history = read_target_history(
        int(train["hour_index"].max()), trigger_categories
    )

    selected_config, validation_predictions = select_configuration(
        train,
        validation,
        train_history,
        trigger_categories,
    )
    validation_metrics = evaluate_metrics(validation_predictions, "validation")

    # Test data is not read until the validation-only configuration is fixed.
    test = read_model_split("test")
    assert_split_order(validation, test)
    pre_test_training_rows = pd.concat([train, validation], ignore_index=True)
    pre_test_history = read_target_history(
        int(validation["hour_index"].max()), trigger_categories
    )
    test_result = rolling_recursive_forecast(
        pre_test_training_rows,
        test,
        pre_test_history,
        trigger_categories,
        selected_config,
        "test",
    )
    test_metrics = evaluate_metrics(test_result.predictions, "test")
    feature_importance = build_feature_importance(
        test_result.first_window_model,
        selected_config,
        int(validation["hour_index"].max()),
    )

    write_outputs(
        validation_predictions,
        validation_metrics,
        test_result.predictions,
        test_metrics,
        feature_importance,
    )

    print("\nLightGBM forecasting completed.")
    print(f"  selected candidate: {selected_config['candidate_id']}")
    print(f"  validation prediction rows: {len(validation_predictions)}")
    print(f"  validation metric rows: {len(validation_metrics)}")
    print(f"  test prediction rows: {len(test_result.predictions)}")
    print(f"  test metric rows: {len(test_metrics)}")
    print(f"  feature-importance rows: {len(feature_importance)}")
    print(
        "  validation predictions clipped to zero: "
        f"{int(validation_predictions['was_clipped_to_zero'].sum())}"
    )
    print(
        "  test predictions clipped to zero: "
        f"{int(test_result.predictions['was_clipped_to_zero'].sum())}"
    )
    print_metrics("validation", validation_metrics)
    print_metrics("test", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
