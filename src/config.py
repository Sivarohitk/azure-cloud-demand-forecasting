"""Configuration for validating the Azure Functions Trace 2019 raw files."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "azurefunctions2019"

INVOCATION_PATTERN = "invocations_per_function_md.anon.d*.csv"
DURATION_PATTERN = "function_durations_percentiles.anon.d*.csv"
MEMORY_PATTERN = "app_memory_percentiles.anon.d*.csv"

EXPECTED_INVOCATION_FILES = tuple(
    f"invocations_per_function_md.anon.d{day:02d}.csv" for day in range(1, 15)
)
EXPECTED_DURATION_FILES = tuple(
    f"function_durations_percentiles.anon.d{day:02d}.csv"
    for day in range(1, 15)
)
EXPECTED_MEMORY_FILES = tuple(
    f"app_memory_percentiles.anon.d{day:02d}.csv" for day in range(1, 13)
)

INVOCATION_ID_COLUMNS = ("HashOwner", "HashApp", "HashFunction", "Trigger")
INVOCATION_MINUTE_COLUMNS = tuple(str(minute) for minute in range(1, 1441))
DURATION_REQUIRED_COLUMNS = (
    "HashOwner",
    "HashApp",
    "HashFunction",
    "Average",
    "Count",
)
MEMORY_REQUIRED_COLUMNS = (
    "HashOwner",
    "HashApp",
    "SampleCount",
    "AverageAllocatedMb",
)

AZURE_FUNCTIONS_TRIGGER_GROUPS = frozenset(
    {"http", "timer", "event", "queue", "storage", "orchestration", "others"}
)
TRIGGER_SAMPLE_ROWS_PER_FILE = 10_000
