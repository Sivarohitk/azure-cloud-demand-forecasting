"""Validate the raw Microsoft Azure Functions Trace 2019 files in place."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

from config import (
    AZURE_FUNCTIONS_TRIGGER_GROUPS,
    DURATION_PATTERN,
    DURATION_REQUIRED_COLUMNS,
    EXPECTED_DURATION_FILES,
    EXPECTED_INVOCATION_FILES,
    EXPECTED_MEMORY_FILES,
    INVOCATION_ID_COLUMNS,
    INVOCATION_MINUTE_COLUMNS,
    INVOCATION_PATTERN,
    MEMORY_PATTERN,
    MEMORY_REQUIRED_COLUMNS,
    RAW_DATA_ROOT,
    TRIGGER_SAMPLE_ROWS_PER_FILE,
)


def display_path(path: Path) -> str:
    """Return a stable path relative to the configured raw-data root."""

    return path.relative_to(RAW_DATA_ROOT).as_posix()


def find_files(pattern: str) -> list[Path]:
    """Recursively find files matching a dataset filename pattern."""

    return sorted(
        (path for path in RAW_DATA_ROOT.rglob(pattern) if path.is_file()),
        key=lambda path: display_path(path).casefold(),
    )


def report_file_set(
    label: str,
    files: list[Path],
    expected_names: tuple[str, ...],
    *,
    required: bool,
) -> bool:
    """Report discovered files and return whether the file set is valid."""

    expected = set(expected_names)
    basename_counts = Counter(path.name for path in files)
    actual = set(basename_counts)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    duplicates = sorted(name for name, count in basename_counts.items() if count > 1)
    valid = (
        len(files) == len(expected_names)
        and not missing
        and not unexpected
        and not duplicates
    )

    print(f"\n{label}: found {len(files)}; expected {len(expected_names)}")
    if files:
        for path in files:
            print(f"  - {display_path(path)}")
    else:
        print("  - none")

    severity = "FAIL" if required else "WARN"
    if valid:
        print(f"[PASS] {label} file set is complete.")
        return True

    print(f"[{severity}] {label} file set is incomplete or ambiguous.")
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    if unexpected:
        print(f"  Unexpected matches: {', '.join(unexpected)}")
    if duplicates:
        print(f"  Duplicate filenames: {', '.join(duplicates)}")
    if not required:
        print("  Missing memory days will not be substituted with other data.")
    return False


def read_header(path: Path) -> list[str]:
    """Read a CSV header without loading data rows."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as error:
            raise ValueError("file is empty") from error


def validate_schema(
    label: str,
    files: Iterable[Path],
    required_columns: Iterable[str],
) -> bool:
    """Validate required columns in each file by reading headers only."""

    required = set(required_columns)
    valid = True
    print(f"\n{label} schema validation:")

    for path in files:
        try:
            header = read_header(path)
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            print(f"[FAIL] {display_path(path)}: could not read header ({error}).")
            valid = False
            continue

        missing = sorted(required - set(header))
        if missing:
            print(
                f"[FAIL] {display_path(path)}: missing columns "
                f"{', '.join(missing)}"
            )
            valid = False
        else:
            print(f"[PASS] {display_path(path)}")

    return valid


def validate_trigger_samples(files: Iterable[Path]) -> bool:
    """Validate bounded Trigger-column samples against the documented groups."""

    valid = True
    print("\nInvocation trigger validation:")

    for path in files:
        sampled_rows = 0
        invalid_values: Counter[str] = Counter()

        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "Trigger" not in reader.fieldnames:
                    print(
                        f"[FAIL] {display_path(path)}: Trigger column is unavailable."
                    )
                    valid = False
                    continue

                for row in reader:
                    if sampled_rows >= TRIGGER_SAMPLE_ROWS_PER_FILE:
                        break
                    raw_value = row.get("Trigger")
                    normalized = raw_value.strip().casefold() if raw_value else ""
                    if normalized not in AZURE_FUNCTIONS_TRIGGER_GROUPS:
                        invalid_values[raw_value or "<blank>"] += 1
                    sampled_rows += 1
        except (OSError, UnicodeError, csv.Error) as error:
            print(f"[FAIL] {display_path(path)}: could not sample triggers ({error}).")
            valid = False
            continue

        if sampled_rows == 0:
            print(f"[FAIL] {display_path(path)}: no rows available to sample.")
            valid = False
        elif invalid_values:
            values = ", ".join(repr(value) for value in sorted(invalid_values))
            print(
                f"[FAIL] {display_path(path)}: incompatible trigger value(s) "
                f"in {sampled_rows} sampled rows: {values}"
            )
            valid = False
        else:
            print(
                f"[PASS] {display_path(path)}: {sampled_rows} sampled rows use "
                "documented trigger groups."
            )

    return valid


def main() -> int:
    """Run raw-data validation and return a process exit status."""

    print("Azure Functions Trace 2019 raw-data validation")
    print(f"Raw root: {RAW_DATA_ROOT}")

    if not RAW_DATA_ROOT.is_dir():
        print(f"[FAIL] Raw directory does not exist: {RAW_DATA_ROOT}")
        print("\n=== Validation summary ===")
        print("RESULT: FAIL")
        print("Download and extract the official dataset into the raw root above.")
        return 1

    print("[PASS] Raw directory exists.")

    invocation_files = find_files(INVOCATION_PATTERN)
    duration_files = find_files(DURATION_PATTERN)
    memory_files = find_files(MEMORY_PATTERN)

    invocation_set_valid = report_file_set(
        "Invocation files",
        invocation_files,
        EXPECTED_INVOCATION_FILES,
        required=True,
    )
    duration_set_valid = report_file_set(
        "Duration files",
        duration_files,
        EXPECTED_DURATION_FILES,
        required=True,
    )
    memory_set_valid = report_file_set(
        "Memory files",
        memory_files,
        EXPECTED_MEMORY_FILES,
        required=False,
    )

    invocation_schema_valid = validate_schema(
        "Invocation",
        invocation_files,
        (*INVOCATION_ID_COLUMNS, *INVOCATION_MINUTE_COLUMNS),
    )
    duration_schema_valid = validate_schema(
        "Duration", duration_files, DURATION_REQUIRED_COLUMNS
    )
    memory_schema_valid = validate_schema(
        "Memory", memory_files, MEMORY_REQUIRED_COLUMNS
    )
    triggers_valid = validate_trigger_samples(invocation_files)

    required_valid = all(
        (
            invocation_set_valid,
            duration_set_valid,
            invocation_schema_valid,
            duration_schema_valid,
            memory_schema_valid,
            triggers_valid,
        )
    )

    print("\n=== Validation summary ===")
    print(f"Invocation files: {'PASS' if invocation_set_valid else 'FAIL'}")
    print(f"Duration files: {'PASS' if duration_set_valid else 'FAIL'}")
    print(f"Memory files: {'PASS' if memory_set_valid else 'WARN'}")
    schemas_valid = all(
        (invocation_schema_valid, duration_schema_valid, memory_schema_valid)
    )
    print(f"Schemas: {'PASS' if schemas_valid else 'FAIL'}")
    print(f"Trigger groups: {'PASS' if triggers_valid else 'FAIL'}")
    print(f"RESULT: {'PASS' if required_valid else 'FAIL'}")
    return 0 if required_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
