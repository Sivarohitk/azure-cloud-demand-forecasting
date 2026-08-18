"""Select and evaluate per-Trigger SARIMAX workload forecasts.

SARIMAX can produce negative unconstrained forecasts even though workload demand
cannot be negative. Final ``compute_seconds`` predictions are therefore clipped
to zero before export and metric calculation; clipping counts are reported.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
VALIDATION_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "sarimax_validation_predictions.csv"
)
VALIDATION_METRICS_PATH = PROJECT_ROOT / "outputs" / "sarimax_validation_metrics.csv"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "sarimax_test_predictions.csv"
TEST_METRICS_PATH = PROJECT_ROOT / "outputs" / "sarimax_test_metrics.csv"
SELECTED_ORDERS_PATH = PROJECT_ROOT / "outputs" / "sarimax_selected_orders.csv"

TARGET_COLUMN = "compute_seconds"
MODEL_NAME = "SARIMAX"
SEASONAL_PERIOD = 24
FORECAST_WINDOW_HOURS = 24
MAX_ITERATIONS = 200

NON_SEASONAL_ORDERS = ((1, 0, 0), (1, 0, 1), (2, 0, 1))
SEASONAL_ORDERS = ((1, 0, 0, 24), (1, 0, 1, 24))
CANDIDATES = tuple(
    (order, seasonal_order)
    for order in NON_SEASONAL_ORDERS
    for seasonal_order in SEASONAL_ORDERS
)
SOURCE_COLUMNS = ("hour_index", "hour_of_day", "Trigger", TARGET_COLUMN, "split")


@dataclass
class ForecastDiagnostics:
    """Warnings and post-processing facts collected across forecast windows."""

    warning_messages: list[str] = field(default_factory=list)
    convergence_warning_messages: list[str] = field(default_factory=list)
    nonconverged_fit_count: int = 0
    clipped_prediction_count: int = 0

    def extend(self, other: "ForecastDiagnostics") -> None:
        """Accumulate diagnostics from one fitted forecast window."""

        self.warning_messages.extend(other.warning_messages)
        self.convergence_warning_messages.extend(
            other.convergence_warning_messages
        )
        self.nonconverged_fit_count += other.nonconverged_fit_count
        self.clipped_prediction_count += other.clipped_prediction_count


@dataclass
class CandidateResult:
    """Validation result for one SARIMAX specification."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    predictions: np.ndarray
    window_numbers: np.ndarray
    clipped_mask: np.ndarray
    metrics: dict[str, float]
    diagnostics: ForecastDiagnostics


def order_label(order: tuple[int, ...]) -> str:
    """Serialize an order tuple without spaces for stable CSV output."""

    return "(" + ",".join(str(value) for value in order) + ")"


def read_split(split_name: str) -> pd.DataFrame:
    """Read one chronological split without loading later splits early."""

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
    validate_split(frame, split_name)
    return frame


def validate_split(frame: pd.DataFrame, split_name: str) -> None:
    """Validate target values and the common continuous Trigger/hour grid."""

    if set(frame["split"].unique()) != {split_name}:
        raise ValueError(f"Unexpected split labels while reading {split_name}.")
    if frame.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError(f"The {split_name} split has duplicate Trigger/hour rows.")
    target = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError(f"The {split_name} target contains invalid demand values.")
    if not np.array_equal(
        frame["hour_of_day"].to_numpy(),
        (frame["hour_index"] % SEASONAL_PERIOD).to_numpy(),
    ):
        raise ValueError(f"The {split_name} hour_of_day does not match hour_index.")

    global_hours = np.sort(frame["hour_index"].unique())
    if not np.all(np.diff(global_hours) == 1):
        raise ValueError(f"The {split_name} hour_index is not continuous.")
    for trigger, trigger_frame in frame.groupby("Trigger", sort=True):
        if not np.array_equal(trigger_frame["hour_index"].to_numpy(), global_hours):
            raise ValueError(
                f"Trigger {trigger!r} does not have the full {split_name} hour grid."
            )


def assert_split_order(earlier: pd.DataFrame, later: pd.DataFrame) -> None:
    """Assert that all reference observations precede the evaluation split."""

    if int(earlier["hour_index"].max()) >= int(later["hour_index"].min()):
        raise AssertionError("Chronological splits overlap or are out of order.")
    if set(earlier["Trigger"].unique()) != set(later["Trigger"].unique()):
        raise AssertionError("Chronological splits contain different Triggers.")


def calculate_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and denominator-safe WAPE as a ratio."""

    actual_values = np.asarray(actual, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
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


def fit_forecast_window(
    history: np.ndarray,
    steps: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    context: str,
) -> tuple[np.ndarray, np.ndarray, ForecastDiagnostics]:
    """Fit one SARIMAX window, logging every warning and convergence result."""

    if len(history) <= SEASONAL_PERIOD:
        raise ValueError("SARIMAX history is too short for daily seasonality.")

    scale = max(float(np.mean(np.abs(history))), 1.0)
    scaled_history = np.asarray(history, dtype=np.float64) / scale
    diagnostics = ForecastDiagnostics()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model = SARIMAX(
            scaled_history,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(maxiter=MAX_ITERATIONS, disp=False)
        raw_prediction = np.asarray(fitted.forecast(steps=steps), dtype=np.float64)

    for captured in caught_warnings:
        message = f"{captured.category.__name__}: {captured.message}"
        diagnostics.warning_messages.append(message)
        if issubclass(captured.category, ConvergenceWarning):
            diagnostics.convergence_warning_messages.append(message)
        print(f"WARNING [{context}] {message}")

    converged = bool(getattr(fitted, "mle_retvals", {}).get("converged", True))
    if not converged:
        message = "Optimizer reported converged=False"
        diagnostics.nonconverged_fit_count = 1
        if message not in diagnostics.convergence_warning_messages:
            diagnostics.convergence_warning_messages.append(message)
            diagnostics.warning_messages.append(message)
            print(f"WARNING [{context}] {message}")

    raw_prediction = raw_prediction * scale
    if not np.isfinite(raw_prediction).all():
        raise ValueError("SARIMAX produced missing or non-finite forecasts.")
    negative_mask = raw_prediction < 0
    diagnostics.clipped_prediction_count = int(negative_mask.sum())
    prediction = np.clip(raw_prediction, 0.0, None)
    return prediction, negative_mask, diagnostics


def rolling_window_forecast(
    history: np.ndarray,
    evaluation: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    context: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ForecastDiagnostics]:
    """Forecast in non-overlapping 24-hour windows with strictly prior history."""

    observed_history = np.asarray(history, dtype=np.float64).copy()
    evaluation_values = np.asarray(evaluation, dtype=np.float64)
    prediction_parts: list[np.ndarray] = []
    window_number_parts: list[np.ndarray] = []
    clipped_mask_parts: list[np.ndarray] = []
    diagnostics = ForecastDiagnostics()

    for start in range(0, len(evaluation_values), FORECAST_WINDOW_HOURS):
        stop = min(start + FORECAST_WINDOW_HOURS, len(evaluation_values))
        actual_window = evaluation_values[start:stop]
        window_number = start // FORECAST_WINDOW_HOURS + 1
        window_context = f"{context}, forecast_window={window_number}"
        prediction, clipped_mask, window_diagnostics = fit_forecast_window(
            observed_history,
            len(actual_window),
            order,
            seasonal_order,
            window_context,
        )
        prediction_parts.append(prediction)
        clipped_mask_parts.append(clipped_mask)
        window_number_parts.append(
            np.full(len(actual_window), window_number, dtype=np.int8)
        )
        diagnostics.extend(window_diagnostics)

        # Actuals are appended only after the entire next-day window is forecast.
        observed_history = np.concatenate([observed_history, actual_window])

    predictions = np.concatenate(prediction_parts)
    window_numbers = np.concatenate(window_number_parts)
    clipped_mask = np.concatenate(clipped_mask_parts)
    if len(predictions) != len(evaluation_values):
        raise AssertionError("SARIMAX forecast length does not match evaluation data.")
    return predictions, window_numbers, clipped_mask, diagnostics


def select_orders(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[
    dict[str, CandidateResult],
    pd.DataFrame,
]:
    """Select each Trigger specification using validation WAPE only."""

    selections: dict[str, CandidateResult] = {}
    selection_records: list[dict[str, object]] = []

    for trigger in sorted(train["Trigger"].unique()):
        train_values = (
            train.loc[train["Trigger"] == trigger]
            .sort_values("hour_index")[TARGET_COLUMN]
            .to_numpy(dtype=np.float64)
        )
        validation_values = (
            validation.loc[validation["Trigger"] == trigger]
            .sort_values("hour_index")[TARGET_COLUMN]
            .to_numpy(dtype=np.float64)
        )
        successful_results: list[CandidateResult] = []
        failed_specifications: list[str] = []
        all_warning_messages: list[str] = []
        all_convergence_messages: list[str] = []

        for order, seasonal_order in CANDIDATES:
            candidate_context = (
                f"validation, Trigger={trigger}, order={order_label(order)}, "
                f"seasonal_order={order_label(seasonal_order)}"
            )
            try:
                (
                    predictions,
                    window_numbers,
                    clipped_mask,
                    diagnostics,
                ) = rolling_window_forecast(
                    train_values,
                    validation_values,
                    order,
                    seasonal_order,
                    candidate_context,
                )
                metrics = calculate_metrics(validation_values, predictions)
                result = CandidateResult(
                    order=order,
                    seasonal_order=seasonal_order,
                    predictions=predictions,
                    window_numbers=window_numbers,
                    clipped_mask=clipped_mask,
                    metrics=metrics,
                    diagnostics=diagnostics,
                )
                successful_results.append(result)
                all_warning_messages.extend(
                    f"{order_label(order)}+{order_label(seasonal_order)}: {message}"
                    for message in diagnostics.warning_messages
                )
                all_convergence_messages.extend(
                    f"{order_label(order)}+{order_label(seasonal_order)}: {message}"
                    for message in diagnostics.convergence_warning_messages
                )
                print(
                    "CANDIDATE PASS "
                    f"Trigger={trigger} order={order_label(order)} "
                    f"seasonal_order={order_label(seasonal_order)} "
                    f"validation_WAPE={metrics['WAPE']:.12f} "
                    f"warnings={len(diagnostics.warning_messages)} "
                    f"clipped={diagnostics.clipped_prediction_count}"
                )
            except Exception as exc:
                failure = (
                    f"order={order_label(order)},"
                    f"seasonal_order={order_label(seasonal_order)},"
                    f"error={type(exc).__name__}: {exc}"
                )
                failed_specifications.append(failure)
                print(f"CANDIDATE FAIL Trigger={trigger} {failure}")

        if not successful_results:
            raise RuntimeError(f"All SARIMAX candidates failed for Trigger={trigger}.")

        # Candidate declaration order is the deterministic tie-breaker only.
        selected = min(successful_results, key=lambda result: result.metrics["WAPE"])
        selections[str(trigger)] = selected
        print(
            "SELECTED "
            f"Trigger={trigger} order={order_label(selected.order)} "
            f"seasonal_order={order_label(selected.seasonal_order)} "
            f"validation_WAPE={selected.metrics['WAPE']:.12f}"
        )
        selection_records.append(
            {
                "Trigger": str(trigger),
                "selected_order": order_label(selected.order),
                "selected_seasonal_order": order_label(selected.seasonal_order),
                "selection_metric": "validation_WAPE",
                "validation_MAE": selected.metrics["MAE"],
                "validation_RMSE": selected.metrics["RMSE"],
                "validation_WAPE": selected.metrics["WAPE"],
                "candidate_count": len(CANDIDATES),
                "successful_candidate_count": len(successful_results),
                "failed_candidate_count": len(failed_specifications),
                "failed_specifications": " | ".join(failed_specifications),
                "candidate_warning_count": len(all_warning_messages),
                "candidate_warnings": " | ".join(all_warning_messages),
                "candidate_convergence_warning_count": len(
                    all_convergence_messages
                ),
                "candidate_convergence_warnings": " | ".join(
                    all_convergence_messages
                ),
                "selected_nonconverged_fit_count": (
                    selected.diagnostics.nonconverged_fit_count
                ),
                "selected_clipped_validation_prediction_count": (
                    selected.diagnostics.clipped_prediction_count
                ),
            }
        )

    return selections, pd.DataFrame.from_records(selection_records)


def build_prediction_frame(
    evaluation: pd.DataFrame,
    split_name: str,
    selections: dict[str, CandidateResult],
    pre_split_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, ForecastDiagnostics]]:
    """Build selected validation predictions or refitted test predictions."""

    prediction_frames: list[pd.DataFrame] = []
    test_diagnostics: dict[str, ForecastDiagnostics] = {}

    for trigger in sorted(evaluation["Trigger"].unique()):
        trigger_name = str(trigger)
        selected = selections[trigger_name]
        trigger_evaluation = evaluation.loc[
            evaluation["Trigger"] == trigger
        ].sort_values("hour_index")

        if pre_split_history is None:
            predictions = selected.predictions
            window_numbers = selected.window_numbers
            clipped_mask = selected.clipped_mask
            diagnostics = selected.diagnostics
        else:
            history_values = (
                pre_split_history.loc[pre_split_history["Trigger"] == trigger]
                .sort_values("hour_index")[TARGET_COLUMN]
                .to_numpy(dtype=np.float64)
            )
            evaluation_values = trigger_evaluation[TARGET_COLUMN].to_numpy(
                dtype=np.float64
            )
            context = (
                f"test, Trigger={trigger_name}, "
                f"order={order_label(selected.order)}, "
                f"seasonal_order={order_label(selected.seasonal_order)}"
            )
            (
                predictions,
                window_numbers,
                clipped_mask,
                diagnostics,
            ) = rolling_window_forecast(
                history_values,
                evaluation_values,
                selected.order,
                selected.seasonal_order,
                context,
            )
            test_diagnostics[trigger_name] = diagnostics
            print(
                "TEST PASS "
                f"Trigger={trigger_name} order={order_label(selected.order)} "
                f"seasonal_order={order_label(selected.seasonal_order)} "
                f"warnings={len(diagnostics.warning_messages)} "
                f"clipped={diagnostics.clipped_prediction_count}"
            )

        frame = trigger_evaluation[
            ["hour_index", "hour_of_day", "Trigger", TARGET_COLUMN]
        ].copy()
        frame.insert(0, "model", MODEL_NAME)
        frame.insert(0, "split", split_name)
        frame["order"] = order_label(selected.order)
        frame["seasonal_order"] = order_label(selected.seasonal_order)
        frame["forecast_window"] = window_numbers
        frame["actual_compute_seconds"] = frame.pop(TARGET_COLUMN)
        frame["predicted_compute_seconds"] = predictions
        frame["was_clipped_to_zero"] = clipped_mask
        prediction_frames.append(frame)

    result = pd.concat(prediction_frames, ignore_index=True)
    result = result.sort_values(["hour_index", "Trigger"], kind="stable").reset_index(
        drop=True
    )
    if len(result) != len(evaluation):
        raise AssertionError(f"{split_name} prediction row count is incorrect.")
    if not np.isfinite(
        result[["actual_compute_seconds", "predicted_compute_seconds"]].to_numpy()
    ).all():
        raise ValueError(f"{split_name} predictions contain non-finite values.")
    if (result["predicted_compute_seconds"] < 0).any():
        raise AssertionError(f"{split_name} contains negative final predictions.")
    return result, test_diagnostics


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
                    trigger_frame["actual_compute_seconds"].to_numpy(),
                    trigger_frame["predicted_compute_seconds"].to_numpy(),
                ),
            }
        )

    hourly_totals = predictions.groupby("hour_index", as_index=False).agg(
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
                hourly_totals["actual_compute_seconds"].to_numpy(),
                hourly_totals["predicted_compute_seconds"].to_numpy(),
            ),
        }
    )
    return pd.DataFrame.from_records(records).sort_values(
        ["aggregation_level", "Trigger"], kind="stable"
    ).reset_index(drop=True)


def add_test_diagnostics(
    selected_orders: pd.DataFrame,
    test_diagnostics: dict[str, ForecastDiagnostics],
) -> pd.DataFrame:
    """Attach test execution diagnostics without changing selected orders."""

    result = selected_orders.copy()
    result["test_warning_count"] = result["Trigger"].map(
        lambda trigger: len(test_diagnostics[trigger].warning_messages)
    )
    result["test_warnings"] = result["Trigger"].map(
        lambda trigger: " | ".join(test_diagnostics[trigger].warning_messages)
    )
    result["test_convergence_warning_count"] = result["Trigger"].map(
        lambda trigger: len(
            test_diagnostics[trigger].convergence_warning_messages
        )
    )
    result["test_convergence_warnings"] = result["Trigger"].map(
        lambda trigger: " | ".join(
            test_diagnostics[trigger].convergence_warning_messages
        )
    )
    result["test_nonconverged_fit_count"] = result["Trigger"].map(
        lambda trigger: test_diagnostics[trigger].nonconverged_fit_count
    )
    result["test_clipped_prediction_count"] = result["Trigger"].map(
        lambda trigger: test_diagnostics[trigger].clipped_prediction_count
    )
    return result


def write_outputs(
    validation_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_metrics: pd.DataFrame,
    selected_orders: pd.DataFrame,
) -> None:
    """Write exactly the five authorized SARIMAX CSV outputs."""

    output_pairs = (
        (VALIDATION_PREDICTIONS_PATH, validation_predictions),
        (VALIDATION_METRICS_PATH, validation_metrics),
        (TEST_PREDICTIONS_PATH, test_predictions),
        (TEST_METRICS_PATH, test_metrics),
        (SELECTED_ORDERS_PATH, selected_orders),
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

    print(f"\n{split_name.upper()} measured SARIMAX metrics:")
    print(
        metrics.to_string(
            index=False,
            columns=["aggregation_level", "Trigger", "MAE", "RMSE", "WAPE"],
            float_format=lambda value: f"{value:.6f}",
        )
    )


def main() -> int:
    """Select on validation, then read and evaluate the untouched test split."""

    train = read_split("train")
    validation = read_split("validation")
    assert_split_order(train, validation)

    selections, selected_orders = select_orders(train, validation)
    validation_predictions, _ = build_prediction_frame(
        validation, "validation", selections
    )
    validation_metrics = evaluate_metrics(validation_predictions, "validation")

    # Test is not read until every Trigger order has been selected on validation.
    test = read_split("test")
    assert_split_order(validation, test)
    pre_test_history = pd.concat([train, validation], ignore_index=True)
    test_predictions, test_diagnostics = build_prediction_frame(
        test,
        "test",
        selections,
        pre_split_history=pre_test_history,
    )
    test_metrics = evaluate_metrics(test_predictions, "test")
    selected_orders = add_test_diagnostics(selected_orders, test_diagnostics)

    write_outputs(
        validation_predictions,
        validation_metrics,
        test_predictions,
        test_metrics,
        selected_orders,
    )

    print("\nSARIMAX forecasting completed.")
    print(f"  validation prediction rows: {len(validation_predictions)}")
    print(f"  validation metric rows: {len(validation_metrics)}")
    print(f"  test prediction rows: {len(test_predictions)}")
    print(f"  test metric rows: {len(test_metrics)}")
    print(f"  selected-order rows: {len(selected_orders)}")
    print(
        "  validation predictions clipped to zero: "
        f"{int(validation_predictions['was_clipped_to_zero'].sum())}"
    )
    print(
        "  test predictions clipped to zero: "
        f"{int(test_predictions['was_clipped_to_zero'].sum())}"
    )
    print_metrics("validation", validation_metrics)
    print_metrics("test", test_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
