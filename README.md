# Azure Cloud Demand Forecasting & Capacity Planning

## Objective

This portfolio project will analyze the official Microsoft Azure Functions public workload trace, forecast future workload demand, quantify forecast uncertainty, estimate capacity headroom and shortage risk using clearly labeled workload/capacity proxies, and export curated datasets for a Power BI dashboard.

## Planned pipeline

1. Download and preserve the official raw workload trace.
2. Validate, clean, and aggregate the workload data into curated time-series datasets.
3. Create leakage-safe time and lag features using only information available before each prediction timestamp.
4. Split the series chronologically into training, validation, and untouched test periods.
5. Train and select statistically defensible baseline, LightGBM, and SARIMAX forecasts using validation data only.
6. Quantify forecast uncertainty and estimate proxy capacity headroom and shortage risk.
7. Evaluate the selected approach once on the test period and export curated Parquet datasets for Power BI.

## Dataset

Dataset placeholder: official Microsoft Azure Functions public workload trace. Source details and download instructions will be added in a later authorized step.

## Results

Results will be added only after the pipeline has been executed against the real downloaded dataset.
