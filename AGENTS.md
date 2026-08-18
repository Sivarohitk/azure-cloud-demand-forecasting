# Azure Cloud Demand Forecasting & Capacity Planning Guardrails

## Project purpose

Analyze the official Microsoft Azure Functions public workload trace, forecast future workload demand, quantify forecast uncertainty, estimate capacity headroom and shortage risk, and export curated datasets for a Power BI dashboard.

## Approved technology scope

Use only:

- Python
- Pandas
- NumPy
- scikit-learn
- LightGBM
- statsmodels SARIMAX
- SciPy only if mathematically necessary
- PyArrow for Parquet
- Power BI as the final visualization layer

Do not add:

- Prophet
- XGBoost
- CatBoost
- TensorFlow
- PyTorch
- neural networks
- FastAPI
- Flask
- Streamlit
- Dash
- Docker
- Kubernetes
- Azure Machine Learning
- Azure Functions deployment
- Databricks
- Spark
- databases
- APIs
- GenAI
- RAG
- CI/CD
- web applications
- frontend code
- cloud infrastructure
- synthetic production data
- any technology or feature not explicitly requested in a later prompt

## Development rules

1. Never create a file unless the current prompt explicitly authorizes that file or directory.
2. Never create additional features because they seem useful.
3. Never fabricate model results, metrics, dataset characteristics, dates, or business conclusions.
4. All metrics must come from executing code against the real downloaded dataset.
5. Never hard-code favorable model results.
6. Raw source data must never be modified.
7. Prevent time-series leakage.
8. All rolling features must use only information available before the prediction timestamp.
9. Model selection must use validation data, never test data.
10. Test data must remain untouched until final evaluation.
11. If required data is missing, stop and report what is missing rather than generating replacement data.
12. Do not create fake Azure regions, datacenters, customers, subscriptions, or workloads.
13. The public dataset does not expose Microsoft's real datacenter capacity. Any capacity metric constructed by this project must be clearly labeled as a workload/capacity proxy rather than actual Azure physical capacity.
14. Keep the implementation understandable enough to explain during a Microsoft Data & Applied Scientist interview.
15. Prefer simple, statistically defensible methods over unnecessary complexity.

Do not implement data processing, feature engineering, forecasting, Monte Carlo simulations, or Power BI work until a later prompt explicitly authorizes that work.
