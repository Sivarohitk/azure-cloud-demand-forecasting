"""Build leakage-safe trigger-level features and chronological data splits.

``historical_load_ratio`` is a normalized historical workload/load proxy. It
is derived only from prior compute demand in this public workload trace and is
not measured Azure datacenter CPU utilization or physical capacity.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT


INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand_by_trigger.parquet"
)
FEATURE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.parquet"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "split_summary.csv"

TARGET_COLUMN = "compute_seconds"
LAG_HOURS = (1, 2, 3, 6, 12, 24, 48, 72)
ROLLING_MEAN_HOURS = (3, 6, 12, 24)
ROLLING_STD_HOURS = (6, 24)
TEST_HOURS = 72
VALIDATION_HOURS = 48
WARMUP_HOURS = max(LAG_HOURS)

KEY_COLUMNS = ("trace_day", "hour_of_day", "hour_index", "Trigger")
TEMPORAL_FEATURE_COLUMNS = (
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "trace_day_index",
)
LAG_FEATURE_COLUMNS = tuple(f"lag_{lag}" for lag in LAG_HOURS)
ROLLING_FEATURE_COLUMNS = (
    *(f"rolling_mean_{window}" for window in ROLLING_MEAN_HOURS),
    *(f"rolling_std_{window}" for window in ROLLING_STD_HOURS),
    "rolling_max_24",
    "rolling_min_24",
)
HISTORICAL_FEATURE_COLUMNS = (
    "historical_load_ratio",
    "lag24_trigger_share",
)
FEATURE_COLUMNS = (
    *TEMPORAL_FEATURE_COLUMNS,
    *LAG_FEATURE_COLUMNS,
    *ROLLING_FEATURE_COLUMNS,
    *HISTORICAL_FEATURE_COLUMNS,
)


def read_and_validate_source() -> pd.DataFrame:
    """Read the hourly trigger dataset and validate its shared time grid."""

    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Processed demand dataset does not exist: {INPUT_PATH}")

    frame = pd.read_parquet(INPUT_PATH, columns=[*KEY_COLUMNS, TARGET_COLUMN])
    if frame.empty:
        raise ValueError(f"Processed demand dataset is empty: {INPUT_PATH}")
    if frame[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("Processed demand dataset contains missing key values.")
    if frame.duplicated(["hour_index", "Trigger"]).any():
        raise ValueError("Processed demand dataset has duplicate Trigger/hour_index rows.")

    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    target_values = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise ValueError("compute_seconds contains missing or non-finite values.")
    if (target_values < 0).any():
        raise ValueError("compute_seconds contains negative values.")

    frame["hour_index"] = pd.to_numeric(frame["hour_index"], errors="raise").astype(
        np.int64
    )
    frame["hour_of_day"] = pd.to_numeric(
        frame["hour_of_day"], errors="raise"
    ).astype(np.int8)
    frame["trace_day"] = pd.to_numeric(frame["trace_day"], errors="raise").astype(
        np.int16
    )
    frame["Trigger"] = frame["Trigger"].astype("string")
    frame = frame.sort_values(["Trigger", "hour_index"], kind="stable").reset_index(
        drop=True
    )

    global_hours = np.sort(frame["hour_index"].unique())
    if len(global_hours) <= WARMUP_HOURS + VALIDATION_HOURS + TEST_HOURS:
        raise ValueError(
            "Not enough global hours for the 72-hour warm-up, training data, "
            "48-hour validation split, and 72-hour test split."
        )
    if not np.all(np.diff(global_hours) == 1):
        raise ValueError("Global hour_index must be continuous.")
    if not frame["hour_of_day"].between(0, 23).all():
        raise ValueError("hour_of_day must be between 0 and 23.")
    if not np.array_equal(
        frame["hour_of_day"].to_numpy(),
        (frame["hour_index"] % 24).to_numpy(),
    ):
        raise ValueError("hour_of_day does not reconcile with hour_index.")

    for trigger, trigger_frame in frame.groupby("Trigger", sort=True):
        trigger_hours = trigger_frame["hour_index"].to_numpy()
        if not np.array_equal(trigger_hours, global_hours):
            raise ValueError(
                f"Trigger {trigger!r} does not contain the complete global hour grid."
            )

    return frame


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Divide nonnegative series, leaving a missing value for zero denominators."""

    numerator_values = numerator.to_numpy(dtype=np.float64)
    denominator_values = denominator.to_numpy(dtype=np.float64)
    ratio = np.full(len(numerator), np.nan, dtype=np.float64)
    np.divide(
        numerator_values,
        denominator_values,
        out=ratio,
        where=np.isfinite(denominator_values) & (denominator_values > 0),
    )
    return ratio


def engineer_features(source: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and strictly historical features independently by Trigger."""

    frame = source.copy()
    minimum_trace_day = int(frame["trace_day"].min())
    frame["trace_day_index"] = (frame["trace_day"] - minimum_trace_day).astype(
        np.int16
    )
    hour_angle = 2.0 * np.pi * frame["hour_of_day"] / 24.0
    frame["hour_sin"] = np.sin(hour_angle)
    frame["hour_cos"] = np.cos(hour_angle)

    grouped_target = frame.groupby("Trigger", sort=False)[TARGET_COLUMN]
    for lag in LAG_HOURS:
        frame[f"lag_{lag}"] = grouped_target.shift(lag)

    # Every rolling and expanding statistic starts from shift(1), so the current
    # target and all future observations are excluded from its window.
    prior_target = grouped_target.shift(1)
    grouped_prior_target = prior_target.groupby(frame["Trigger"], sort=False)
    for window in ROLLING_MEAN_HOURS:
        frame[f"rolling_mean_{window}"] = grouped_prior_target.transform(
            lambda values, size=window: values.rolling(
                size, min_periods=size
            ).mean()
        )
    for window in ROLLING_STD_HOURS:
        frame[f"rolling_std_{window}"] = grouped_prior_target.transform(
            lambda values, size=window: values.rolling(
                size, min_periods=size
            ).std()
        )
    frame["rolling_max_24"] = grouped_prior_target.transform(
        lambda values: values.rolling(24, min_periods=24).max()
    )
    frame["rolling_min_24"] = grouped_prior_target.transform(
        lambda values: values.rolling(24, min_periods=24).min()
    )

    historical_p95 = grouped_prior_target.transform(
        lambda values: values.expanding(min_periods=1).quantile(0.95)
    )
    frame["historical_load_ratio"] = safe_ratio(frame["lag_1"], historical_p95)

    # At hour t, lag_24 and its across-trigger total both describe hour t-24.
    # Their ratio therefore uses only demand already observed before hour t.
    lag24_total = frame.groupby("hour_index", sort=False)["lag_24"].transform(
        lambda values: values.sum(min_count=1)
    )
    frame["lag24_trigger_share"] = safe_ratio(frame["lag_24"], lag24_total)

    return frame


def assign_chronological_splits(feature_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove lag warm-up rows and assign shared global chronological splits."""

    global_hours = np.sort(feature_frame["hour_index"].unique())
    warmup_start_hour = int(global_hours[WARMUP_HOURS])
    validation_start_hour = int(global_hours[-(TEST_HOURS + VALIDATION_HOURS)])
    test_start_hour = int(global_hours[-TEST_HOURS])

    model_frame = feature_frame.loc[
        feature_frame["hour_index"] >= warmup_start_hour
    ].copy()
    model_frame["split"] = np.select(
        [
            model_frame["hour_index"] >= test_start_hour,
            model_frame["hour_index"] >= validation_start_hour,
        ],
        ["test", "validation"],
        default="train",
    )
    model_frame = model_frame.sort_values(
        ["hour_index", "Trigger"], kind="stable"
    ).reset_index(drop=True)

    split_bounds = model_frame.groupby("split")["hour_index"].agg(["min", "max"])
    required_splits = {"train", "validation", "test"}
    if set(split_bounds.index) != required_splits:
        raise AssertionError("Train, validation, and test splits must all be non-empty.")
    assert split_bounds.loc["train", "max"] < split_bounds.loc["validation", "min"]
    assert split_bounds.loc["validation", "max"] < split_bounds.loc["test", "min"]

    unique_hours_by_split = model_frame.groupby("split")["hour_index"].nunique()
    if int(unique_hours_by_split["validation"]) != VALIDATION_HOURS:
        raise AssertionError("Validation split does not contain exactly 48 global hours.")
    if int(unique_hours_by_split["test"]) != TEST_HOURS:
        raise AssertionError("Test split does not contain exactly 72 global hours.")

    trigger_bounds = model_frame.groupby(["split", "Trigger"])["hour_index"].agg(
        ["min", "max", "nunique"]
    )
    for split_name, split_frame in trigger_bounds.groupby(level="split"):
        if (split_frame[["min", "max", "nunique"]].nunique() != 1).any():
            raise AssertionError(
                f"Triggers do not share identical {split_name} chronological bounds."
            )

    if not model_frame["lag24_trigger_share"].dropna().between(0.0, 1.0).all():
        raise AssertionError("lag24_trigger_share falls outside [0, 1].")
    finite_load_ratio = model_frame["historical_load_ratio"].dropna()
    if not np.isfinite(finite_load_ratio.to_numpy()).all():
        raise AssertionError("historical_load_ratio contains infinite values.")
    if (finite_load_ratio < 0).any():
        raise AssertionError("historical_load_ratio contains negative values.")

    return model_frame


def summary_key(value: str) -> str:
    """Convert observed categorical labels into stable CSV column fragments."""

    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def build_split_summary(model_frame: pd.DataFrame) -> pd.DataFrame:
    """Create one wide row with split, Trigger, and post-warm-up missingness facts."""

    summary: dict[str, int | float] = {
        "warmup_hours": WARMUP_HOURS,
        "observations_after_warmup": len(model_frame),
    }
    split_order = ("train", "validation", "test")
    triggers = sorted(model_frame["Trigger"].astype(str).unique())

    for split_name in split_order:
        split_frame = model_frame.loc[model_frame["split"] == split_name]
        summary[f"{split_name}_start_hour_index"] = int(
            split_frame["hour_index"].min()
        )
        summary[f"{split_name}_end_hour_index"] = int(
            split_frame["hour_index"].max()
        )
        summary[f"{split_name}_observations"] = len(split_frame)

        trigger_counts = split_frame.groupby("Trigger", observed=True).size()
        for trigger in triggers:
            trigger_name = summary_key(trigger)
            summary[f"{split_name}_{trigger_name}_observations"] = int(
                trigger_counts.get(trigger, 0)
            )

    total_trigger_counts = model_frame.groupby("Trigger", observed=True).size()
    for trigger in triggers:
        summary[f"{summary_key(trigger)}_observations"] = int(
            total_trigger_counts.get(trigger, 0)
        )

    feature_missing = model_frame[list(FEATURE_COLUMNS)].isna().sum()
    total_feature_values = len(model_frame) * len(FEATURE_COLUMNS)
    total_missing = int(feature_missing.sum())
    summary["feature_missing_values_after_warmup"] = total_missing
    summary["feature_missing_percentage_after_warmup"] = (
        100.0 * total_missing / total_feature_values if total_feature_values else np.nan
    )
    for feature in FEATURE_COLUMNS:
        missing_count = int(feature_missing[feature])
        summary[f"{feature}_missing_values_after_warmup"] = missing_count
        summary[f"{feature}_missing_percentage_after_warmup"] = (
            100.0 * missing_count / len(model_frame) if len(model_frame) else np.nan
        )

    return pd.DataFrame([summary])


def validate_historical_features(
    source: pd.DataFrame, engineered: pd.DataFrame, model_frame: pd.DataFrame
) -> None:
    """Assert lag alignment and post-warm-up feature completeness where defined."""

    expected_lag_1 = source.groupby("Trigger", sort=False)[TARGET_COLUMN].shift(1)
    if not np.allclose(
        engineered["lag_1"].to_numpy(dtype=np.float64),
        expected_lag_1.to_numpy(dtype=np.float64),
        equal_nan=True,
    ):
        raise AssertionError("lag_1 does not equal the prior Trigger observation.")

    required_warmup_features = [
        *LAG_FEATURE_COLUMNS,
        *ROLLING_FEATURE_COLUMNS,
    ]
    if model_frame[required_warmup_features].isna().any().any():
        raise AssertionError(
            "Lag or rolling features remain missing after the required 72-hour warm-up."
        )


def write_outputs(model_frame: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    """Write exactly the authorized feature and split-summary outputs."""

    for output_path in (FEATURE_OUTPUT_PATH, SPLIT_SUMMARY_PATH):
        if not output_path.parent.is_dir():
            raise FileNotFoundError(
                f"Output directory does not exist: {output_path.parent}"
            )

    output_columns = [
        "hour_index",
        "Trigger",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
        "split",
    ]
    output_frame = model_frame[output_columns].copy()
    output_frame["Trigger"] = output_frame["Trigger"].astype("category")
    output_frame["split"] = pd.Categorical(
        output_frame["split"], categories=["train", "validation", "test"], ordered=True
    )
    output_frame.to_parquet(FEATURE_OUTPUT_PATH, index=False, engine="pyarrow")
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)


def main() -> int:
    """Build, validate, write, and report the requested modeling dataset."""

    source = read_and_validate_source()
    engineered = engineer_features(source)
    model_frame = assign_chronological_splits(engineered)
    validate_historical_features(source, engineered, model_frame)
    split_summary = build_split_summary(model_frame)
    write_outputs(model_frame, split_summary)

    split_sizes = model_frame.groupby("split", sort=False).size()
    print("Feature engineering and chronological splitting completed.")
    print(f"  model_features shape: ({len(model_frame)}, {4 + len(FEATURE_COLUMNS)})")
    print(f"  split_summary shape: {split_summary.shape}")
    print("\nChronological split sizes:")
    for split_name in ("train", "validation", "test"):
        split_frame = model_frame.loc[model_frame["split"] == split_name]
        print(
            f"  {split_name}: {int(split_sizes[split_name])} observations "
            f"(hour_index {int(split_frame['hour_index'].min())}-"
            f"{int(split_frame['hour_index'].max())})"
        )
    missing_values = int(
        model_frame[list(FEATURE_COLUMNS)].isna().sum().sum()
    )
    print(f"\nFeature missing values after warm-up: {missing_values}")
    print("Chronology assertions: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
