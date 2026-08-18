"""Compare forecast models and select one using validation performance only."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT


OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

MODEL_COMPARISON_PATH = OUTPUTS_ROOT / "model_comparison.csv"
MODEL_COMPARISON_BY_TRIGGER_PATH = (
    OUTPUTS_ROOT / "model_comparison_by_trigger.csv"
)
SELECTED_MODEL_PATH = OUTPUTS_ROOT / "selected_model.json"
SELECTED_MODEL_TEST_PREDICTIONS_PATH = (
    OUTPUTS_ROOT / "selected_model_test_predictions.csv"
)

MODEL_ORDER = (
    "Seasonal Naive",
    "Historical Hour-of-Day Mean",
    "SARIMAX",
    "LightGBM",
)

BASELINE_LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "historical_hour_of_day_mean": "Historical Hour-of-Day Mean",
}

MODEL_SOURCES = {
    "Seasonal Naive": {
        "family": "baseline",
        "source_label": "seasonal_naive",
    },
    "Historical Hour-of-Day Mean": {
        "family": "baseline",
        "source_label": "historical_hour_of_day_mean",
    },
    "SARIMAX": {"family": "sarimax", "source_label": "SARIMAX"},
    "LightGBM": {"family": "lightgbm", "source_label": "LightGBM"},
}

PREDICTION_VALUE_COLUMNS = (
    "actual_compute_seconds",
    "predicted_compute_seconds",
)
PREDICTION_OUTPUT_COLUMNS = (
    "split",
    "model",
    "hour_index",
    "hour_of_day",
    "Trigger",
    *PREDICTION_VALUE_COLUMNS,
)
METRIC_COLUMNS = ("MAE", "RMSE", "WAPE")
SELECTION_CRITERION = "Lowest aggregated validation WAPE"


def source_path(family: str, split: str, artifact: str) -> Path:
    """Return the expected path for one source model artifact."""
    return OUTPUTS_ROOT / f"{family}_{split}_{artifact}.csv"


def require_source_files() -> None:
    """Fail clearly if any model output needed for comparison is absent."""
    required_paths = {
        source_path(source["family"], split, artifact)
        for source in MODEL_SOURCES.values()
        for split in ("validation", "test")
        for artifact in ("predictions", "metrics")
    }
    missing = sorted(path for path in required_paths if not path.is_file())
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required model comparison inputs are missing:\n" + missing_text
        )


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Calculate MAE, RMSE, and safe WAPE for a prediction frame."""
    actual = frame["actual_compute_seconds"].to_numpy(dtype=float)
    prediction = frame["predicted_compute_seconds"].to_numpy(dtype=float)
    error = actual - prediction
    absolute_error = np.abs(error)
    denominator = np.abs(actual).sum()
    wape = float(absolute_error.sum() / denominator) if denominator else np.nan
    return {
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "WAPE": wape,
    }


def load_model_predictions(
    model: str,
    split: str,
    cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    """Load and standardize one model's prediction rows."""
    source = MODEL_SOURCES[model]
    path = source_path(source["family"], split, "predictions")
    if path not in cache:
        cache[path] = pd.read_csv(path)
    frame = cache[path]

    required = {
        "split",
        "hour_index",
        "hour_of_day",
        "Trigger",
        *PREDICTION_VALUE_COLUMNS,
    }
    label_column = "baseline" if source["family"] == "baseline" else "model"
    required.add(label_column)
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    selected = frame.loc[frame[label_column] == source["source_label"]].copy()
    if selected.empty:
        raise ValueError(
            f"{path} has no prediction rows for {source['source_label']!r}."
        )
    if set(selected["split"]) != {split}:
        raise ValueError(f"{path} contains unexpected split labels.")

    selected["model"] = model
    selected = selected.loc[:, PREDICTION_OUTPUT_COLUMNS].copy()
    selected.sort_values(["hour_index", "Trigger"], inplace=True, kind="stable")
    selected.reset_index(drop=True, inplace=True)

    if selected.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError(f"{path} contains duplicate hour/Trigger predictions.")
    if selected[list(PREDICTION_VALUE_COLUMNS)].isna().any().any():
        raise ValueError(f"{path} contains missing actual or predicted values.")
    return selected


def load_model_metrics(
    model: str,
    split: str,
    cache: dict[Path, pd.DataFrame],
) -> pd.DataFrame:
    """Load and standardize one model's published metrics."""
    source = MODEL_SOURCES[model]
    path = source_path(source["family"], split, "metrics")
    if path not in cache:
        cache[path] = pd.read_csv(path)
    frame = cache[path]

    required = {"split", "aggregation_level", "Trigger", *METRIC_COLUMNS}
    label_column = "baseline" if source["family"] == "baseline" else "model"
    required.add(label_column)
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {missing_columns}")

    selected = frame.loc[frame[label_column] == source["source_label"]].copy()
    if selected.empty:
        raise ValueError(f"{path} has no metric rows for {source['source_label']!r}.")
    if set(selected["split"]) != {split}:
        raise ValueError(f"{path} contains unexpected split labels.")

    selected["model"] = model
    columns = ["split", "model", "aggregation_level", "Trigger", *METRIC_COLUMNS]
    return selected.loc[:, columns].reset_index(drop=True)


def validate_published_metrics(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    model: str,
    split: str,
) -> None:
    """Recalculate every source metric from predictions before comparison."""
    trigger_rows = metrics.loc[metrics["aggregation_level"] == "Trigger"]
    total_rows = metrics.loc[metrics["aggregation_level"] == "total_demand"]
    expected_triggers = set(predictions["Trigger"])

    if set(trigger_rows["Trigger"]) != expected_triggers:
        raise ValueError(f"{model} {split} Trigger metrics do not match predictions.")
    if len(total_rows) != 1 or total_rows.iloc[0]["Trigger"] != "ALL_TRIGGERS":
        raise ValueError(f"{model} {split} must have one total-demand metric row.")

    for row in trigger_rows.itertuples(index=False):
        recalculated = calculate_metrics(
            predictions.loc[predictions["Trigger"] == row.Trigger]
        )
        published = {metric: getattr(row, metric) for metric in METRIC_COLUMNS}
        if not all(
            np.isclose(recalculated[metric], published[metric], rtol=1e-10, atol=1e-6)
            for metric in METRIC_COLUMNS
        ):
            raise ValueError(
                f"{model} {split} metrics do not reconcile for Trigger {row.Trigger}."
            )

    hourly_total = predictions.groupby("hour_index", as_index=False)[
        list(PREDICTION_VALUE_COLUMNS)
    ].sum()
    recalculated_total = calculate_metrics(hourly_total)
    published_total = total_rows.iloc[0]
    if not all(
        np.isclose(
            recalculated_total[metric],
            published_total[metric],
            rtol=1e-10,
            atol=1e-6,
        )
        for metric in METRIC_COLUMNS
    ):
        raise ValueError(f"{model} {split} total-demand metrics do not reconcile.")


def load_and_validate_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one split and confirm all models use identical evaluation rows."""
    prediction_cache: dict[Path, pd.DataFrame] = {}
    metric_cache: dict[Path, pd.DataFrame] = {}
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    reference: pd.DataFrame | None = None

    for model in MODEL_ORDER:
        predictions = load_model_predictions(model, split, prediction_cache)
        metrics = load_model_metrics(model, split, metric_cache)
        validate_published_metrics(predictions, metrics, model, split)

        evaluation_rows = predictions.loc[
            :, ["hour_index", "Trigger", "actual_compute_seconds"]
        ]
        if reference is None:
            reference = evaluation_rows.reset_index(drop=True)
        else:
            candidate = evaluation_rows.reset_index(drop=True)
            same_keys = candidate[["hour_index", "Trigger"]].equals(
                reference[["hour_index", "Trigger"]]
            )
            same_actuals = np.allclose(
                candidate["actual_compute_seconds"],
                reference["actual_compute_seconds"],
            )
            if not same_keys or not same_actuals:
                raise ValueError(
                    f"{model} {split} predictions do not share the common evaluation set."
                )

        prediction_frames.append(predictions)
        metric_frames.append(metrics)

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
    )


def aggregated_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return exactly one total-demand metric row for each model."""
    result = metrics.loc[metrics["aggregation_level"] == "total_demand"].copy()
    counts = result.groupby("model").size()
    if set(counts.index) != set(MODEL_ORDER) or not counts.eq(1).all():
        raise ValueError("Expected exactly one aggregated metric row per model.")
    return result


def build_comparison(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    selected_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build total-demand and Trigger-level comparison tables."""
    validation_totals = aggregated_metrics(validation_metrics)
    test_totals = aggregated_metrics(test_metrics)

    ranks = (
        validation_totals.set_index("model")["WAPE"]
        .rank(method="min", ascending=True)
        .astype(int)
        .to_dict()
    )

    comparison = pd.concat([validation_totals, test_totals], ignore_index=True)
    comparison["validation_wape_rank"] = comparison["model"].map(ranks)
    comparison["selected_on_validation"] = comparison["model"].eq(selected_model)
    split_order = pd.Categorical(
        comparison["split"], categories=["validation", "test"], ordered=True
    )
    comparison = comparison.assign(_split_order=split_order).sort_values(
        ["_split_order", "validation_wape_rank"], kind="stable"
    )
    comparison.drop(columns=["_split_order", "aggregation_level", "Trigger"], inplace=True)
    comparison = comparison.loc[
        :,
        [
            "split",
            "model",
            *METRIC_COLUMNS,
            "validation_wape_rank",
            "selected_on_validation",
        ],
    ].reset_index(drop=True)

    by_trigger = pd.concat(
        [
            validation_metrics.loc[
                validation_metrics["aggregation_level"] == "Trigger"
            ],
            test_metrics.loc[test_metrics["aggregation_level"] == "Trigger"],
        ],
        ignore_index=True,
    )
    by_trigger["validation_wape_rank"] = by_trigger["model"].map(ranks)
    by_trigger["selected_on_validation"] = by_trigger["model"].eq(selected_model)
    by_trigger["_split_order"] = pd.Categorical(
        by_trigger["split"], categories=["validation", "test"], ordered=True
    )
    by_trigger.sort_values(
        ["_split_order", "Trigger", "validation_wape_rank"],
        inplace=True,
        kind="stable",
    )
    by_trigger.drop(columns=["_split_order", "aggregation_level"], inplace=True)
    by_trigger = by_trigger.loc[
        :,
        [
            "split",
            "model",
            "Trigger",
            *METRIC_COLUMNS,
            "validation_wape_rank",
            "selected_on_validation",
        ],
    ].reset_index(drop=True)
    return comparison, by_trigger


def write_outputs(
    comparison: pd.DataFrame,
    comparison_by_trigger: pd.DataFrame,
    selected_metadata: dict[str, str | float],
    selected_test_predictions: pd.DataFrame,
) -> None:
    """Write only the four authorized model-selection artifacts."""
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    comparison_by_trigger.to_csv(MODEL_COMPARISON_BY_TRIGGER_PATH, index=False)
    SELECTED_MODEL_PATH.write_text(
        json.dumps(selected_metadata, indent=2) + "\n", encoding="utf-8"
    )
    selected_test_predictions.to_csv(
        SELECTED_MODEL_TEST_PREDICTIONS_PATH, index=False
    )


def main() -> int:
    """Select the best validation model and report its untouched test results."""
    require_source_files()

    # The test artifacts are deliberately not loaded until this selection is fixed.
    validation_predictions, validation_metrics = load_and_validate_split("validation")
    validation_totals = aggregated_metrics(validation_metrics).sort_values(
        ["WAPE", "MAE", "RMSE", "model"], kind="stable"
    )
    selected_validation = validation_totals.iloc[0]
    selected_model = str(selected_validation["model"])

    test_predictions, test_metrics = load_and_validate_split("test")
    comparison, comparison_by_trigger = build_comparison(
        validation_metrics, test_metrics, selected_model
    )

    selected_test_predictions = test_predictions.loc[
        test_predictions["model"] == selected_model,
        PREDICTION_OUTPUT_COLUMNS,
    ].copy()
    selected_test_predictions.sort_values(
        ["hour_index", "Trigger"], inplace=True, kind="stable"
    )
    selected_test_predictions.reset_index(drop=True, inplace=True)

    execution_timestamp = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    selected_metadata: dict[str, str | float] = {
        "selected_model": selected_model,
        "validation_MAE": float(selected_validation["MAE"]),
        "validation_RMSE": float(selected_validation["RMSE"]),
        "validation_WAPE": float(selected_validation["WAPE"]),
        "selection_criterion": SELECTION_CRITERION,
        "pipeline_execution_timestamp": execution_timestamp,
        "test_set_excluded_from_selection": (
            "The test set was excluded from model selection and was loaded only "
            "after the validation winner was fixed."
        ),
    }
    write_outputs(
        comparison,
        comparison_by_trigger,
        selected_metadata,
        selected_test_predictions,
    )

    selected_test = aggregated_metrics(test_metrics).loc[
        lambda frame: frame["model"] == selected_model
    ].iloc[0]
    seasonal_test = aggregated_metrics(test_metrics).loc[
        lambda frame: frame["model"] == "Seasonal Naive"
    ].iloc[0]
    outperformed_seasonal = bool(selected_test["WAPE"] < seasonal_test["WAPE"])

    print("MODEL COMPARISON COMPLETE")
    print("Selection criterion: lowest aggregated validation WAPE")
    print("Test artifacts were excluded from selection.")
    print("\nAggregated validation metrics:")
    print(
        validation_totals[["model", *METRIC_COLUMNS]].to_string(
            index=False,
            formatters={metric: "{:.12f}".format for metric in METRIC_COLUMNS},
        )
    )
    print(f"\nSelected model: {selected_model}")
    print("\nSelected model unbiased test metrics:")
    print(
        selected_test[["model", *METRIC_COLUMNS]].to_frame().T.to_string(
            index=False
        )
    )
    print("\nSeasonal Naive test metrics:")
    print(
        seasonal_test[["model", *METRIC_COLUMNS]].to_frame().T.to_string(
            index=False
        )
    )
    if outperformed_seasonal:
        print("\nThe selected model outperformed Seasonal Naive on test WAPE.")
    else:
        print("\nThe selected model did NOT outperform Seasonal Naive on test WAPE.")

    print("\nOutput shapes:")
    print(f"  model_comparison.csv: {comparison.shape}")
    print(f"  model_comparison_by_trigger.csv: {comparison_by_trigger.shape}")
    print(f"  selected_model_test_predictions.csv: {selected_test_predictions.shape}")
    print(f"  selected_model.json: {len(selected_metadata)} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
