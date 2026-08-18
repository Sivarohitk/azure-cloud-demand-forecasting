# Power BI Dashboard Build Guide

Build exactly three report pages in Power BI Desktop. Do not add a fourth page, maps, decomposition trees, AI visuals, Copilot visuals, geographic analysis, cost estimates, or invented Azure infrastructure fields.

The dashboard reports workload demand in compute-seconds. Capacity means **required workload-serving capacity in normalized compute-seconds per hour**. It does not represent Microsoft's real servers, CPUs, datacenters, or installed capacity.

## 1. Load the source tables

1. Open Power BI Desktop.
2. Select **Home > Get data > Text/CSV**.
3. Browse to the project `powerbi` folder, select `demand_history.csv`, and select **Open**.
4. In the preview dialog, select **Load**.
5. Repeat **Home > Get data > Text/CSV > Open > Load** for:
   - `model_performance.csv`
   - `forecast_vs_actual.csv`
   - `future_forecast.csv`
   - `capacity_scenarios.csv`
   - `capacity_recommendation.csv`
6. The existing `outputs/lightgbm_feature_importance.csv` file is present, so import it for the required Page 2 feature-importance visual:
   1. Select **Home > Get data > Text/CSV**.
   2. Browse to the project `outputs` folder.
   3. Select `lightgbm_feature_importance.csv` and select **Open**.
   4. Select **Load**.
7. In the **Data** pane, verify these seven table names:
   - `demand_history`
   - `model_performance`
   - `forecast_vs_actual`
   - `future_forecast`
   - `capacity_scenarios`
   - `capacity_recommendation`
   - `lightgbm_feature_importance`

## 2. Verify data types and model behavior

1. Select the **Table view** icon on the left.
2. Select each table in the **Data** pane, then select each column and use **Column tools > Data type** to verify:
   - identifiers and labels such as `trigger`, `model`, `dataset_split`, and `selected_model`: **Text**;
   - `hour_index`, `trace_day`, `hour_of_day`, `forecast_hour`, and rank fields: **Whole number**;
   - demand, forecast, error, capacity, gain, and probability fields: **Decimal number**;
   - `meets_service_target` and `is_minimum_recommended_headroom`: **True/False**.
3. Keep `headroom_pct`, `headroom_percent`, and `minimum_recommended_headroom_percent` as numeric whole-percent values. Do not apply Power BI's Percentage format to these columns because their stored values use 10 to mean 10%, not 0.10.
4. Format ratio fields as percentages:
   1. Select `model_performance[WAPE]`.
   2. Select **Column tools > Format > Percentage** and set two decimal places.
   3. Repeat for `capacity_scenarios[shortage_probability]`, `capacity_recommendation[maximum_hourly_shortage_probability]`, `capacity_recommendation[average_shortage_probability]`, and `capacity_recommendation[service_target_maximum_hourly_shortage_probability]`.
5. Select the **Model view** icon on the left.
6. These are curated fact tables and should remain disconnected. If relationship lines were created automatically:
   1. Select **Home > Manage relationships**.
   2. Select each auto-created relationship.
   3. Select **Delete**, confirm the deletion, and repeat until no relationships remain.
   4. Select **Close**.

## 3. Create the required measures

Open `powerbi/measures.dax` beside Power BI Desktop and create only the six measures listed there.

### Measures stored under `forecast_vs_actual`

For each of the first four definitions in `measures.dax`:

1. In the **Data** pane, right-click `forecast_vs_actual`.
2. Select **New measure**.
3. Replace the formula-bar contents with one complete measure definition.
4. Press **Enter**.
5. Repeat for:
   - `Total Actual Demand`
   - `Selected Model`
   - `Test WAPE`
   - `Recommended Capacity Headroom`

Apply these formats from **Measure tools**:

- `Total Actual Demand`: decimal number with thousands separator and two decimal places.
- `Test WAPE`: percentage with two decimal places.
- `Recommended Capacity Headroom`: percentage with zero decimal places.
- `Selected Model`: leave as text.

### Measures stored under `capacity_scenarios`

1. In the **Data** pane, right-click `capacity_scenarios`.
2. Select **New measure**.
3. Paste the complete `Capacity View Point Forecast` definition and press **Enter**.
4. Right-click `capacity_scenarios` again, select **New measure**, paste the complete `Recommended Scenario Capacity` definition, and press **Enter**.
5. From **Measure tools**, format both measures as decimal numbers with thousands separators and two decimal places.

The DAX reads the selected model and recommended headroom from the exported tables. It contains no hard-coded model result or headroom recommendation.

## 4. Create exactly three report pages

1. Select the **Report view** icon on the left.
2. Double-click the existing page tab, type `Demand & Forecast Overview`, and press **Enter**.
3. Select the **+** button beside the page tabs once. Double-click the new tab, type `Model Performance`, and press **Enter**.
4. Select the **+** button once more. Double-click the new tab, type `Capacity & Risk`, and press **Enter**.
5. Confirm that there are exactly three page tabs and no blank fourth page.

If the **Visualizations** or **Filters** pane is hidden, select **View > Panes** and enable the missing pane before following the visual steps.

## Page 1 — Demand & Forecast Overview

Open the `Demand & Forecast Overview` page.

### KPI card: Total Actual Demand

1. Click a blank area of the canvas.
2. In **Visualizations**, select **Card**.
3. Drag the `Total Actual Demand` measure into the card's **Data** field.
4. Open **Format visual > General > Title**, turn **Title** on, and enter `Total Actual Demand`.
5. Position the card at the upper left.

### KPI card: Selected Model

1. Click a blank area and select **Card**.
2. Drag the `Selected Model` measure into **Data**.
3. Turn on the title and enter `Selected Model`.
4. Place the card to the right of Total Actual Demand.

### KPI card: Test WAPE

1. Click a blank area and select **Card**.
2. Drag the `Test WAPE` measure into **Data**.
3. Turn on the title and enter `Test WAPE`.
4. Place the card to the right of Selected Model.

### KPI card: Recommended Capacity Headroom

1. Click a blank area and select **Card**.
2. Drag the `Recommended Capacity Headroom` measure into **Data**.
3. Turn on the title and enter `Recommended Capacity Headroom`.
4. Place the card at the upper right.

### Line chart: actual demand versus selected forecast

1. Click a blank area and select **Line chart**.
2. From `forecast_vs_actual`, drag `hour_index` to **X-axis**.
3. Drag `actual_compute_seconds` and `selected_forecast` to **Y-axis**.
4. For both Y-axis fields, open the field menu and select **Sum**.
5. Open **Format visual > General > Title**, turn it on, and enter `Actual Demand vs Selected Forecast`.
6. Resize the chart across the middle-left portion of the page.

### Grouped demand visual by Trigger

1. Click a blank area and select **Clustered column chart**.
2. From `forecast_vs_actual`, drag `trigger` to **X-axis**.
3. Drag `actual_compute_seconds` to **Y-axis** and set its aggregation to **Sum**.
4. Turn on the title and enter `Actual Demand by Trigger`.
5. Place the chart to the right of the line chart.

### Trigger slicer

1. Click a blank area and select **Slicer**.
2. Drag `forecast_vs_actual[trigger]` into the slicer's **Field**.
3. Open **Format visual > Visual > Slicer settings > Options** and set **Style** to **Dropdown**.
4. Turn on the title and enter `Trigger`.
5. Place the slicer above the two charts.
6. Select a Trigger and confirm that the Total Actual Demand card and both Page 1 charts respond. Clear the slicer afterward.

## Page 2 — Model Performance

Open the `Model Performance` page.

For each of the three comparison visuals below, apply the same visual-level filters:

1. Select the visual.
2. Drag `model_performance[dataset_split]` to **Filters on this visual** and select only `test`.
3. Drag `model_performance[trigger]` to **Filters on this visual** and select only `TOTAL`.

### Comparison visual: MAE by model

1. Click a blank area and select **Clustered column chart**.
2. Drag `model_performance[model]` to **X-axis**.
3. Drag `model_performance[MAE]` to **Y-axis** and set it to **Max**.
4. Apply the `test` and `TOTAL` visual filters above.
5. Turn on the title and enter `Test MAE by Model`.

### Comparison visual: RMSE by model

1. Click a blank area and select **Clustered column chart**.
2. Drag `model_performance[model]` to **X-axis**.
3. Drag `model_performance[RMSE]` to **Y-axis** and set it to **Max**.
4. Apply the `test` and `TOTAL` visual filters.
5. Turn on the title and enter `Test RMSE by Model`.

### Comparison visual: WAPE by model

1. Click a blank area and select **Clustered column chart**.
2. Drag `model_performance[model]` to **X-axis**.
3. Drag `model_performance[WAPE]` to **Y-axis** and set it to **Max**.
4. Apply the `test` and `TOTAL` visual filters.
5. Turn on the title and enter `Test WAPE by Model`.

Place the MAE, RMSE, and WAPE charts in one row across the top of the page.

### Actual versus predicted line chart

1. Click a blank area and select **Line chart**.
2. From `forecast_vs_actual`, drag `hour_index` to **X-axis**.
3. Add these fields to **Y-axis**, setting each to **Sum**:
   - `actual_compute_seconds`
   - `baseline_forecast`
   - `SARIMAX_forecast`
   - `LightGBM_forecast`
   - `selected_forecast`
4. Turn on the title and enter `Actual vs Model Forecasts`.
5. Place the chart across the middle-left area.

### Model performance table

1. Click a blank area and select **Table**.
2. Add these `model_performance` fields in order:
   - `model`
   - `dataset_split`
   - `trigger`
   - `MAE`
   - `RMSE`
   - `WAPE`
3. Turn on the title and enter `Model Performance Detail`.
4. Place the table across the bottom of the page.

### LightGBM gain feature importance

The feature-importance output exists, so include this visual.

1. Click a blank area and select **Clustered bar chart**.
2. From `lightgbm_feature_importance`, drag `feature` to **Y-axis**.
3. Drag `gain_percentage` to **X-axis** and set it to **Max**.
4. Select the visual's **More options (...) > Sort axis > gain_percentage > Sort descending**.
5. Keep `gain_percentage` as a decimal number because the source already stores whole-percent values.
6. Turn on the title and enter `LightGBM Gain Feature Importance (%)`.
7. Place the bar chart to the right of the actual-versus-predicted line chart.

## Page 3 — Capacity & Risk

Open the `Capacity & Risk` page.

### Point forecast with P50, P90, and P95 uncertainty

1. Click a blank area and select **Line chart**.
2. From `future_forecast`, drag `forecast_hour` to **X-axis**.
3. Add `point_forecast`, `P50`, `P90`, and `P95` to **Y-axis**, setting each to **Sum**.
4. Drag `future_forecast[trigger]` to **Filters on this visual** and select only `TOTAL`.
5. Turn on the title and enter `Total Workload Forecast and Uncertainty`.
6. Place the chart across the upper-left area.

### Capacity line compared with forecast demand

1. Click a blank area and select **Line chart**.
2. Drag `capacity_scenarios[forecast_hour]` to **X-axis**.
3. Drag the `Capacity View Point Forecast` and `Recommended Scenario Capacity` measures to **Y-axis**.
4. Turn on the title and enter `Forecast Demand vs Recommended Capacity`.
5. Place the chart across the upper-right area.

The capacity measure dynamically selects the recommendation flagged in `capacity_recommendation`; do not type a headroom value into a visual filter.

### Shortage probability by headroom percentage

1. Click a blank area and select **Line chart**.
2. Drag `capacity_recommendation[headroom_percent]` to **X-axis**.
3. Drag `maximum_hourly_shortage_probability` to **Y-axis** and set it to **Max**.
4. Set the X-axis title to `Headroom (%)`.
5. Turn on the visual title and enter `Maximum Shortage Probability by Headroom`.

### Expected shortage by headroom percentage

1. Click a blank area and select **Clustered column chart**.
2. Drag `capacity_recommendation[headroom_percent]` to **X-axis**.
3. Drag `expected_total_shortage` to **Y-axis** and set it to **Max**.
4. Set the X-axis title to `Headroom (%)`.
5. Turn on the visual title and enter `Expected Total Shortage by Headroom`.

Place the shortage-probability and expected-shortage visuals side by side in the middle row.

### Recommended headroom KPI card

1. Click a blank area and select **Card**.
2. Drag the `Recommended Capacity Headroom` measure into **Data**.
3. Turn on the title and enter `Recommended Capacity Headroom`.
4. Place the card at the left of the bottom row.

### Scenario table

1. Click a blank area and select **Table**.
2. Add these fields from `capacity_recommendation` in order:
   - `headroom_percent`
   - `maximum_hourly_shortage_probability`
   - `average_shortage_probability`
   - `expected_total_shortage`
   - `average_unused_capacity`
   - `meets_service_target`
   - `is_minimum_recommended_headroom`
3. In the table's field list, rename `headroom_percent` for this visual to `Headroom (%)`.
4. Turn on the title and enter `Capacity Scenario Summary`.
5. Place the table across the remainder of the bottom row.

## 5. Final verification

1. Confirm the report has exactly these three page tabs:
   - `Demand & Forecast Overview`
   - `Model Performance`
   - `Capacity & Risk`
2. Confirm Page 1 contains four KPI cards, two demand visuals, and one Trigger slicer.
3. Confirm Page 2 contains three model-metric comparisons, one actual-versus-predicted line chart, one performance table, and the existing LightGBM gain-importance visual.
4. Confirm Page 3 contains two forecast/capacity line charts, two headroom-risk visuals, one recommended-headroom card, and one scenario table.
5. Confirm percentage displays are formatted from numeric fields rather than text containing `%` symbols.
6. Confirm no visual uses invented calendar dates, Azure regions, servers, CPUs, datacenters, costs, revenue, or customer data.
7. Confirm no fourth report page, map, decomposition tree, AI visual, or Copilot visual exists.

Do not create or export a PBIX as part of this repository stage; this guide is the manual construction specification for Power BI Desktop.
