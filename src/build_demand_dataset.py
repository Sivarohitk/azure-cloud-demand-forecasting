"""Build leakage-safe hourly workload-demand datasets from the raw trace."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset

from config import (
    DURATION_PATTERN,
    INVOCATION_MINUTE_COLUMNS,
    INVOCATION_PATTERN,
    MEMORY_PATTERN,
    PROJECT_ROOT,
    RAW_DATA_ROOT,
)


JOIN_COLUMNS = ("HashOwner", "HashApp", "HashFunction")
TRIGGER_ORDER = (
    "http",
    "timer",
    "event",
    "queue",
    "storage",
    "orchestration",
    "others",
)
INVOCATION_DAYS = tuple(range(1, 15))
MEMORY_DAYS = tuple(range(1, 13))
INVOCATION_BATCH_ROWS = 2_048

INVOCATION_FEATHER_NAME = "invocations_per_function_md.feather"
DURATION_FEATHER_NAME = "function_durations_percentiles.feather"
MEMORY_FEATHER_NAME = "app_memory_percentiles.feather"

HOURLY_TRIGGER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_demand_by_trigger.parquet"
)
HOURLY_TOTAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "hourly_total_demand.parquet"
)
DAILY_MEMORY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "daily_memory_summary.parquet"
)
QUALITY_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "data_quality_summary.csv"


@dataclass(frozen=True)
class SourceSet:
    """One logical trace table stored as daily CSVs or one Feather file."""

    layout: str
    paths: tuple[Path, ...]
    files_by_day: dict[int, Path] | None = None


@dataclass
class DayAccumulator:
    """Bounded hourly accumulators for one trace day."""

    invocations: np.ndarray = field(
        default_factory=lambda: np.zeros((24, len(TRIGGER_ORDER)), dtype=np.float64)
    )
    compute_seconds: np.ndarray = field(
        default_factory=lambda: np.zeros((24, len(TRIGGER_ORDER)), dtype=np.float64)
    )
    active_functions: np.ndarray = field(
        default_factory=lambda: np.zeros((24, len(TRIGGER_ORDER)), dtype=np.int64)
    )
    matched_invocations: np.ndarray = field(
        default_factory=lambda: np.zeros((24, len(TRIGGER_ORDER)), dtype=np.float64)
    )
    source_rows: int = 0
    missing_duration_rows: int = 0
    missing_duration_invocations: float = 0.0


def parse_trace_day(path: Path) -> int:
    """Extract a dNN trace-day identifier from an official CSV filename."""

    match = re.search(r"\.d(\d{1,2})\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot identify trace day from {path.name}")
    return int(match.group(1))


def discover_source(
    csv_pattern: str,
    feather_name: str,
    expected_days: tuple[int, ...],
    label: str,
) -> SourceSet:
    """Find exactly one unambiguous physical layout for a logical source."""

    csv_paths = tuple(sorted(path for path in RAW_DATA_ROOT.rglob(csv_pattern) if path.is_file()))
    feather_paths = tuple(
        sorted(path for path in RAW_DATA_ROOT.rglob(feather_name) if path.is_file())
    )

    if csv_paths and feather_paths:
        raise RuntimeError(
            f"Ambiguous {label} source: both daily CSV and consolidated Feather "
            "files are present."
        )

    if csv_paths:
        files_by_day: dict[int, Path] = {}
        for path in csv_paths:
            day = parse_trace_day(path)
            if day in files_by_day:
                raise RuntimeError(f"Duplicate {label} source for trace day {day}.")
            files_by_day[day] = path
        validate_day_set(files_by_day, expected_days, label)
        return SourceSet("daily_csv", csv_paths, files_by_day)

    if len(feather_paths) == 1:
        return SourceSet("consolidated_feather", feather_paths)
    if len(feather_paths) > 1:
        raise RuntimeError(f"Multiple consolidated {label} Feather files found.")

    raise FileNotFoundError(
        f"No {label} source found under {RAW_DATA_ROOT}. Expected {csv_pattern} "
        f"or {feather_name}."
    )


def validate_day_set(
    observed_days: set[int] | dict[int, object],
    expected_days: tuple[int, ...],
    label: str,
) -> None:
    """Require exactly the trace days published for a logical source."""

    observed = set(observed_days)
    expected = set(expected_days)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} trace days are incomplete. Missing={missing}; "
            f"unexpected={unexpected}."
        )


def read_small_source(
    source: SourceSet,
    columns: list[str],
    expected_days: tuple[int, ...],
    label: str,
) -> pd.DataFrame:
    """Read projected duration or memory columns and attach trace_day."""

    if source.layout == "daily_csv":
        assert source.files_by_day is not None
        frames = []
        for day, path in sorted(source.files_by_day.items()):
            frame = pd.read_csv(path, usecols=columns)
            frame.insert(0, "trace_day", day)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.read_feather(source.paths[0], columns=[*columns, "day"])
        result = result.rename(columns={"day": "trace_day"})

    observed_days = set(result["trace_day"].dropna().astype(int).unique())
    validate_day_set(observed_days, expected_days, label)
    result["trace_day"] = result["trace_day"].astype(np.int16)
    return result


def build_duration_indexes(
    duration_frame: pd.DataFrame,
) -> tuple[dict[int, pd.Series], dict[str, int]]:
    """Create one Count-weighted duration lookup per function and trace day."""

    duplicated = duration_frame.duplicated(["trace_day", *JOIN_COLUMNS], keep=False)
    duration_frame["Average"] = pd.to_numeric(
        duration_frame["Average"], errors="coerce"
    )
    duration_frame["Count"] = pd.to_numeric(duration_frame["Count"], errors="coerce")
    if not np.isfinite(duration_frame["Count"].to_numpy()).all():
        raise ValueError("Duration data contains missing or non-finite Count values.")
    if (duration_frame["Count"] <= 0).any():
        raise ValueError("Duration data contains non-positive Count values.")

    valid_average = np.isfinite(duration_frame["Average"].to_numpy()) & (
        duration_frame["Average"].to_numpy() >= 0
    )
    invalid_average_row_count = int((~valid_average).sum())
    valid_duration_frame = duration_frame.loc[valid_average].copy()

    group_columns = ["trace_day", *JOIN_COLUMNS]
    weighted_rows = valid_duration_frame.assign(
        weighted_duration_ms=(
            valid_duration_frame["Average"] * valid_duration_frame["Count"]
        )
    )
    consolidated = (
        weighted_rows.groupby(group_columns, as_index=False, sort=True)
        .agg(
            weighted_duration_ms=("weighted_duration_ms", "sum"),
            total_duration_count=("Count", "sum"),
        )
    )
    consolidated["Average"] = (
        consolidated["weighted_duration_ms"]
        / consolidated["total_duration_count"]
    )

    duplicate_group_count = int(
        duration_frame.loc[duplicated].groupby(group_columns, sort=False).ngroups
    )
    duplicate_metrics = {
        "duration_duplicate_row_count": int(duplicated.sum()),
        "duration_duplicate_key_group_count": duplicate_group_count,
        "duration_invalid_average_row_count": invalid_average_row_count,
    }

    indexes = {
        day: day_frame.set_index(list(JOIN_COLUMNS))["Average"]
        for day, day_frame in consolidated.groupby("trace_day", sort=True)
    }
    return indexes, duplicate_metrics


def iter_invocation_batches(source: SourceSet) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield bounded invocation batches, separated by trace day."""

    projected_columns = [*JOIN_COLUMNS, "Trigger", *INVOCATION_MINUTE_COLUMNS]

    if source.layout == "daily_csv":
        assert source.files_by_day is not None
        for day, path in sorted(source.files_by_day.items()):
            for frame in pd.read_csv(
                path,
                usecols=projected_columns,
                chunksize=INVOCATION_BATCH_ROWS,
            ):
                yield day, frame
        return

    dataset = arrow_dataset.dataset(str(source.paths[0]), format="ipc")
    scanner = dataset.scanner(
        columns=[*projected_columns, "day"],
        batch_size=INVOCATION_BATCH_ROWS,
        use_threads=True,
    )
    for record_batch in scanner.to_batches():
        frame = record_batch.to_pandas()
        for raw_day, day_frame in frame.groupby("day", sort=False):
            day = int(raw_day)
            yield day, day_frame.drop(columns="day").reset_index(drop=True)


def process_invocation_batch(
    day: int,
    frame: pd.DataFrame,
    duration_index: pd.Series,
    accumulator: DayAccumulator,
) -> None:
    """Aggregate one function-row batch directly from minutes to hours."""

    normalized_triggers = frame["Trigger"].astype("string").str.strip().str.casefold()
    trigger_codes = pd.Categorical(
        normalized_triggers, categories=TRIGGER_ORDER
    ).codes
    if (trigger_codes < 0).any():
        invalid = sorted(
            str(value)
            for value in normalized_triggers[trigger_codes < 0].drop_duplicates()
        )
        raise ValueError(f"Trace day {day} has unsupported trigger values: {invalid}")

    minute_counts = frame[list(INVOCATION_MINUTE_COLUMNS)].to_numpy(
        dtype=np.float64, copy=False
    )
    if not np.isfinite(minute_counts).all():
        raise ValueError(f"Trace day {day} contains missing or non-finite invocations.")
    if (minute_counts < 0).any():
        raise ValueError(f"Trace day {day} contains negative invocation counts.")
    if not np.equal(minute_counts, np.floor(minute_counts)).all():
        raise ValueError(f"Trace day {day} contains non-integer invocation counts.")

    hourly_counts = minute_counts.reshape(-1, 24, 60).sum(axis=2)
    function_index = pd.MultiIndex.from_frame(frame[list(JOIN_COLUMNS)])
    average_duration_ms = duration_index.reindex(function_index).to_numpy(
        dtype=np.float64
    )
    matched = np.isfinite(average_duration_ms)

    accumulator.source_rows += len(frame)
    accumulator.missing_duration_rows += int((~matched).sum())
    accumulator.missing_duration_invocations += float(hourly_counts[~matched].sum())

    for trigger_code in range(len(TRIGGER_ORDER)):
        trigger_mask = trigger_codes == trigger_code
        if not trigger_mask.any():
            continue

        trigger_counts = hourly_counts[trigger_mask]
        accumulator.invocations[:, trigger_code] += trigger_counts.sum(axis=0)
        accumulator.active_functions[:, trigger_code] += (
            trigger_counts > 0
        ).sum(axis=0)

        matched_trigger_mask = trigger_mask & matched
        if not matched_trigger_mask.any():
            continue
        matched_counts = hourly_counts[matched_trigger_mask]
        matched_durations = average_duration_ms[matched_trigger_mask]
        accumulator.matched_invocations[:, trigger_code] += matched_counts.sum(
            axis=0
        )
        accumulator.compute_seconds[:, trigger_code] += (
            matched_counts * matched_durations[:, np.newaxis] / 1000.0
        ).sum(axis=0)


def build_hourly_trigger_dataset(
    invocation_source: SourceSet,
    duration_indexes: dict[int, pd.Series],
) -> tuple[pd.DataFrame, dict[int, DayAccumulator]]:
    """Build a complete hourly grid for every observed official trigger."""

    accumulators = {day: DayAccumulator() for day in INVOCATION_DAYS}
    for day, frame in iter_invocation_batches(invocation_source):
        if day not in accumulators:
            raise ValueError(f"Unexpected invocation trace day: {day}")
        process_invocation_batch(
            day,
            frame,
            duration_indexes[day],
            accumulators[day],
        )

    observed_days = {day for day, value in accumulators.items() if value.source_rows}
    validate_day_set(observed_days, INVOCATION_DAYS, "Invocation")

    records: list[dict[str, int | float | str]] = []
    for day in INVOCATION_DAYS:
        accumulator = accumulators[day]
        for hour_of_day in range(24):
            hour_index = (day - 1) * 24 + hour_of_day
            for trigger_code, trigger in enumerate(TRIGGER_ORDER):
                invocations = accumulator.invocations[hour_of_day, trigger_code]
                compute_seconds = accumulator.compute_seconds[
                    hour_of_day, trigger_code
                ]
                matched_invocations = accumulator.matched_invocations[
                    hour_of_day, trigger_code
                ]
                weighted_duration = (
                    compute_seconds * 1000.0 / matched_invocations
                    if matched_invocations > 0
                    else np.nan
                )
                records.append(
                    {
                        "trace_day": day,
                        "hour_of_day": hour_of_day,
                        "hour_index": hour_index,
                        "Trigger": trigger,
                        "invocations": int(round(invocations)),
                        "compute_seconds": float(compute_seconds),
                        "active_functions": int(
                            accumulator.active_functions[
                                hour_of_day, trigger_code
                            ]
                        ),
                        "invocation_weighted_average_duration_ms": float(
                            weighted_duration
                        ),
                    }
                )

    result = pd.DataFrame.from_records(records)
    result = result.sort_values(
        ["hour_index", "Trigger"], kind="stable"
    ).reset_index(drop=True)
    return result, accumulators


def build_hourly_total_dataset(hourly_trigger: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trigger demand and add invocation-share columns."""

    key_columns = ["trace_day", "hour_of_day", "hour_index"]
    hourly_total = (
        hourly_trigger.groupby(key_columns, as_index=False, sort=True)
        .agg(
            total_invocations=("invocations", "sum"),
            total_compute_seconds=("compute_seconds", "sum"),
            total_active_functions=("active_functions", "sum"),
        )
        .sort_values("hour_index", kind="stable")
        .reset_index(drop=True)
    )

    trigger_pivot = hourly_trigger.pivot(
        index="hour_index", columns="Trigger", values="invocations"
    )
    observed_triggers = [
        trigger
        for trigger in TRIGGER_ORDER
        if trigger in trigger_pivot and trigger_pivot[trigger].sum() > 0
    ]
    denominators = hourly_total.set_index("hour_index")["total_invocations"]
    denominator_values = denominators.to_numpy(dtype=np.float64)
    for trigger in observed_triggers:
        numerators = trigger_pivot[trigger].reindex(denominators.index)
        shares = np.full(len(denominators), np.nan, dtype=np.float64)
        np.divide(
            numerators.to_numpy(dtype=np.float64),
            denominator_values,
            out=shares,
            where=denominator_values > 0,
        )
        hourly_total[f"invocation_share_{trigger}"] = shares

    return hourly_total


def build_memory_summary(
    memory_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create daily application-level memory statistics without capacity claims."""

    duplicated = memory_frame.duplicated(
        ["trace_day", "HashOwner", "HashApp"], keep=False
    )
    memory_frame["AverageAllocatedMb"] = pd.to_numeric(
        memory_frame["AverageAllocatedMb"], errors="coerce"
    )
    memory_frame["SampleCount"] = pd.to_numeric(
        memory_frame["SampleCount"], errors="coerce"
    )
    if not np.isfinite(
        memory_frame[["AverageAllocatedMb", "SampleCount"]].to_numpy()
    ).all():
        raise ValueError("Memory data contains missing or non-finite values.")
    if (memory_frame["AverageAllocatedMb"] < 0).any():
        raise ValueError("Memory data contains negative AverageAllocatedMb values.")
    if (memory_frame["SampleCount"] <= 0).any():
        raise ValueError("Memory data contains non-positive SampleCount values.")

    application_keys = ["trace_day", "HashOwner", "HashApp"]
    weighted_rows = memory_frame.assign(
        allocated_mb_samples=(
            memory_frame["AverageAllocatedMb"] * memory_frame["SampleCount"]
        )
    )
    applications = (
        weighted_rows.groupby(application_keys, as_index=False, sort=True)
        .agg(
            allocated_mb_samples=("allocated_mb_samples", "sum"),
            total_sample_count=("SampleCount", "sum"),
        )
    )
    applications["AverageAllocatedMb"] = (
        applications["allocated_mb_samples"] / applications["total_sample_count"]
    )

    summary = (
        applications.groupby("trace_day", as_index=False, sort=True)
        .agg(
            application_count=("HashApp", "size"),
            mean_allocated_mb=("AverageAllocatedMb", "mean"),
            median_allocated_mb=("AverageAllocatedMb", "median"),
            p95_allocated_mb=("AverageAllocatedMb", lambda values: values.quantile(0.95)),
        )
        .sort_values("trace_day", kind="stable")
        .reset_index(drop=True)
    )
    summary["application_count"] = summary["application_count"].astype(np.int64)
    duplicate_group_count = int(
        memory_frame.loc[duplicated]
        .groupby(application_keys, sort=False)
        .ngroups
    )
    duplicate_metrics = {
        "memory_duplicate_row_count": int(duplicated.sum()),
        "memory_duplicate_key_group_count": duplicate_group_count,
    }
    return summary, duplicate_metrics


def validate_outputs(
    hourly_trigger: pd.DataFrame,
    hourly_total: pd.DataFrame,
    accumulators: dict[int, DayAccumulator],
) -> dict[str, float | bool]:
    """Assert ordering, non-negativity, completeness, and reconciliation."""

    if (hourly_trigger["invocations"] < 0).any():
        raise AssertionError("Negative invocation counts found after aggregation.")
    if (hourly_trigger["compute_seconds"] < 0).any():
        raise AssertionError("Negative compute_seconds found after aggregation.")

    hours_per_day = hourly_total.groupby("trace_day")["hour_of_day"].nunique()
    if (hours_per_day > 24).any():
        raise AssertionError("A trace day contains more than 24 hourly rows.")
    if not (hours_per_day == 24).all():
        raise AssertionError("One or more trace days do not contain all 24 hours.")

    expected_hour_index = np.arange(len(INVOCATION_DAYS) * 24)
    if not np.array_equal(hourly_total["hour_index"], expected_hour_index):
        raise AssertionError("hour_index is not continuous and ordered from zero.")

    regrouped = hourly_trigger.groupby("hour_index", sort=True).agg(
        invocations=("invocations", "sum"),
        compute_seconds=("compute_seconds", "sum"),
        active_functions=("active_functions", "sum"),
    )
    total_indexed = hourly_total.set_index("hour_index")
    invocation_difference = float(
        np.abs(
            regrouped["invocations"].to_numpy()
            - total_indexed["total_invocations"].to_numpy()
        ).max()
    )
    compute_difference = float(
        np.abs(
            regrouped["compute_seconds"].to_numpy()
            - total_indexed["total_compute_seconds"].to_numpy()
        ).max()
    )
    active_difference = float(
        np.abs(
            regrouped["active_functions"].to_numpy()
            - total_indexed["total_active_functions"].to_numpy()
        ).max()
    )

    raw_invocations = sum(value.invocations.sum() for value in accumulators.values())
    raw_compute_seconds = sum(
        value.compute_seconds.sum() for value in accumulators.values()
    )
    raw_invocation_difference = float(
        abs(float(hourly_total["total_invocations"].sum()) - raw_invocations)
    )
    raw_compute_difference = float(
        abs(float(hourly_total["total_compute_seconds"].sum()) - raw_compute_seconds)
    )

    reconciliation_passed = bool(
        invocation_difference == 0.0
        and active_difference == 0.0
        and raw_invocation_difference == 0.0
        and np.isclose(compute_difference, 0.0, atol=1e-9, rtol=1e-12)
        and np.isclose(raw_compute_difference, 0.0, atol=1e-6, rtol=1e-12)
    )
    if not reconciliation_passed:
        raise AssertionError("Trigger, total, or raw accumulator sums do not reconcile.")

    return {
        "reconciliation_passed": reconciliation_passed,
        "max_hourly_invocation_difference": invocation_difference,
        "max_hourly_compute_seconds_difference": compute_difference,
        "max_hourly_active_function_difference": active_difference,
        "raw_total_invocation_difference": raw_invocation_difference,
        "raw_total_compute_seconds_difference": raw_compute_difference,
    }


def build_quality_summary(
    invocation_source: SourceSet,
    duration_source: SourceSet,
    memory_source: SourceSet,
    duration_frame: pd.DataFrame,
    memory_frame: pd.DataFrame,
    hourly_trigger: pd.DataFrame,
    hourly_total: pd.DataFrame,
    daily_memory: pd.DataFrame,
    accumulators: dict[int, DayAccumulator],
    reconciliation: dict[str, float | bool],
    duration_duplicate_metrics: dict[str, int],
    memory_duplicate_metrics: dict[str, int],
) -> pd.DataFrame:
    """Create a one-row record of observed source and output quality metrics."""

    invocation_rows = sum(value.source_rows for value in accumulators.values())
    missing_duration_rows = sum(
        value.missing_duration_rows for value in accumulators.values()
    )
    missing_duration_invocations = sum(
        value.missing_duration_invocations for value in accumulators.values()
    )
    total_invocations = int(hourly_total["total_invocations"].sum())
    total_compute_seconds = float(hourly_total["total_compute_seconds"].sum())
    missing_function_percentage = (
        100.0 * missing_duration_rows / invocation_rows if invocation_rows else np.nan
    )
    missing_invocation_percentage = (
        100.0 * missing_duration_invocations / total_invocations
        if total_invocations
        else np.nan
    )
    memory_days = set(daily_memory["trace_day"].astype(int))
    memory_unavailable_days = sorted(set(INVOCATION_DAYS) - memory_days)

    values: dict[str, object] = {
        "invocation_source_layout": invocation_source.layout,
        "duration_source_layout": duration_source.layout,
        "memory_source_layout": memory_source.layout,
        "invocation_source_file_count": len(invocation_source.paths),
        "duration_source_file_count": len(duration_source.paths),
        "memory_source_file_count": len(memory_source.paths),
        "invocation_processed_day_count": len(INVOCATION_DAYS),
        "duration_processed_day_count": int(duration_frame["trace_day"].nunique()),
        "memory_processed_day_count": int(memory_frame["trace_day"].nunique()),
        "invocation_source_row_count": invocation_rows,
        "duration_source_row_count": len(duration_frame),
        "memory_source_row_count": len(memory_frame),
        "hourly_trigger_row_count": len(hourly_trigger),
        "hourly_total_row_count": len(hourly_total),
        "daily_memory_row_count": len(daily_memory),
        "missing_duration_join_function_count": missing_duration_rows,
        "missing_duration_join_function_percentage": missing_function_percentage,
        "missing_duration_join_invocation_count": missing_duration_invocations,
        "missing_duration_join_invocation_percentage": missing_invocation_percentage,
        **duration_duplicate_metrics,
        **memory_duplicate_metrics,
        "memory_unavailable_trace_days": "|".join(
            str(day) for day in memory_unavailable_days
        ),
        "total_invocations": total_invocations,
        "total_compute_seconds": total_compute_seconds,
        **reconciliation,
    }
    return pd.DataFrame([values])


def write_outputs(
    hourly_trigger: pd.DataFrame,
    hourly_total: pd.DataFrame,
    daily_memory: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> None:
    """Write exactly the four authorized curated outputs."""

    for path in (
        HOURLY_TRIGGER_PATH,
        HOURLY_TOTAL_PATH,
        DAILY_MEMORY_PATH,
        QUALITY_SUMMARY_PATH,
    ):
        if not path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {path.parent}")

    hourly_trigger.to_parquet(HOURLY_TRIGGER_PATH, index=False, engine="pyarrow")
    hourly_total.to_parquet(HOURLY_TOTAL_PATH, index=False, engine="pyarrow")
    daily_memory.to_parquet(DAILY_MEMORY_PATH, index=False, engine="pyarrow")
    quality_summary.to_csv(QUALITY_SUMMARY_PATH, index=False)


def main() -> int:
    """Build, validate, write, and report the requested demand datasets."""

    if not RAW_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Raw data directory does not exist: {RAW_DATA_ROOT}")

    invocation_source = discover_source(
        INVOCATION_PATTERN,
        INVOCATION_FEATHER_NAME,
        INVOCATION_DAYS,
        "invocation",
    )
    duration_source = discover_source(
        DURATION_PATTERN,
        DURATION_FEATHER_NAME,
        INVOCATION_DAYS,
        "duration",
    )
    memory_source = discover_source(
        MEMORY_PATTERN,
        MEMORY_FEATHER_NAME,
        MEMORY_DAYS,
        "memory",
    )

    duration_frame = read_small_source(
        duration_source,
        [*JOIN_COLUMNS, "Average", "Count"],
        INVOCATION_DAYS,
        "Duration",
    )
    duration_indexes, duration_duplicate_metrics = build_duration_indexes(
        duration_frame
    )
    validate_day_set(duration_indexes, INVOCATION_DAYS, "Duration")

    hourly_trigger, accumulators = build_hourly_trigger_dataset(
        invocation_source, duration_indexes
    )
    hourly_total = build_hourly_total_dataset(hourly_trigger)

    memory_frame = read_small_source(
        memory_source,
        ["HashOwner", "HashApp", "SampleCount", "AverageAllocatedMb"],
        MEMORY_DAYS,
        "Memory",
    )
    daily_memory, memory_duplicate_metrics = build_memory_summary(memory_frame)
    reconciliation = validate_outputs(hourly_trigger, hourly_total, accumulators)
    quality_summary = build_quality_summary(
        invocation_source,
        duration_source,
        memory_source,
        duration_frame,
        memory_frame,
        hourly_trigger,
        hourly_total,
        daily_memory,
        accumulators,
        reconciliation,
        duration_duplicate_metrics,
        memory_duplicate_metrics,
    )

    write_outputs(hourly_trigger, hourly_total, daily_memory, quality_summary)

    missing_function_count = int(
        quality_summary.loc[0, "missing_duration_join_function_count"]
    )
    missing_function_percentage = float(
        quality_summary.loc[0, "missing_duration_join_function_percentage"]
    )
    missing_invocation_count = float(
        quality_summary.loc[0, "missing_duration_join_invocation_count"]
    )
    missing_invocation_percentage = float(
        quality_summary.loc[0, "missing_duration_join_invocation_percentage"]
    )

    print("Demand aggregation completed.")
    print("\nOutput shapes:")
    print(f"  hourly_demand_by_trigger: {hourly_trigger.shape}")
    print(f"  hourly_total_demand: {hourly_total.shape}")
    print(f"  daily_memory_summary: {daily_memory.shape}")
    print(f"  data_quality_summary: {quality_summary.shape}")
    print("\nReconciliation checks:")
    print(f"  passed: {reconciliation['reconciliation_passed']}")
    print(
        "  max hourly invocation difference: "
        f"{reconciliation['max_hourly_invocation_difference']}"
    )
    print(
        "  max hourly compute-seconds difference: "
        f"{reconciliation['max_hourly_compute_seconds_difference']}"
    )
    print("\nMissing-data findings:")
    print(
        "  duration duplicate key groups combined with Count weighting: "
        f"{duration_duplicate_metrics['duration_duplicate_key_group_count']} "
        f"groups across {duration_duplicate_metrics['duration_duplicate_row_count']} rows"
    )
    print(
        "  invalid duration Average rows excluded without substitution: "
        f"{duration_duplicate_metrics['duration_invalid_average_row_count']}"
    )
    print(
        "  memory duplicate key groups combined with SampleCount weighting: "
        f"{memory_duplicate_metrics['memory_duplicate_key_group_count']} "
        f"groups across {memory_duplicate_metrics['memory_duplicate_row_count']} rows"
    )
    print(
        "  invocation rows without a duration join: "
        f"{missing_function_count} ({missing_function_percentage:.6f}%)"
    )
    print(
        "  invocations attached to missing duration joins: "
        f"{missing_invocation_count:.0f} ({missing_invocation_percentage:.6f}%)"
    )
    print(
        "  memory-unavailable trace days (not substituted): "
        f"{quality_summary.loc[0, 'memory_unavailable_trace_days']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
