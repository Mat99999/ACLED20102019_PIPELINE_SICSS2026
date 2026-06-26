# ACLED Weekly Protest Diffusion and Riot Escalation Pipeline

This repository contains a weekly spatiotemporal prediction pipeline for ACLED protest and riot events in Africa, 2010-2019. It is a separate local adaptation of the earlier ACLED grid model, using a new dataset and a different hypothesis structure.

## Research Focus

The project studies whether recent protest activity improves prediction of future protest occurrence over time and space, whether peaceful protest histories have stronger predictive value than non-peaceful protest histories, and whether non-peaceful protest events improve prediction of subsequent riots in the focal or neighboring 100 x 100 km grid cells. The revised H3 grid specification does not impose a same-country restriction.

## Hypotheses

- **H1A Spatial diffusion:** recent protest activity in neighboring 100 x 100 km grid cells is expected to improve prediction of protest occurrence in the focal grid cell, beyond the focal cell's own recent protest history, measured over 1-week, 2-week, and 4-week windows.
- **H1B Temporal diffusion:** recent protest activity in the focal grid cell is expected to improve prediction of protest occurrence in the same grid cell.
- **H2 Peaceful versus non-peaceful diffusion:** recent peaceful protest histories are expected to have stronger predictive value for future protest occurrence than recent non-peaceful protest histories, both within the focal grid cell and in neighboring grid cells. Non-peaceful protests are defined as ACLED protest sub-event types `Protest with intervention` and `Excessive force against protesters`.
- **H3 Protest escalation:** higher levels of non-peaceful protest events are expected to improve prediction of riots in the focal or neighboring 100 x 100 km grid cells in subsequent periods, without restricting the outcome to the same country.

## Main Setup

- Source data: `data/new data/ACLED_protests_riots_2010_2019.csv`
- Event types included: `Protests` and `Riots`
- Geographic scope: Africa
- Time period: 2010-2019
- Main temporal unit: week
- Main spatial unit: 100 x 100 km grid cell
- Train period: 2010-2017
- Test period: 2018-2019
- H1/H2 unit of analysis: grid-cell x week
- H3 revised inferential unit of analysis: grid-cell x week
- Main H1/H2 target: whether a grid cell has at least one protest in the next 1, 2, or 4 weeks
- Revised H3 target: whether the focal or neighboring grid cells have at least one riot in the next 1, 2, or 4 weeks, with no same-country filter

## Repository Structure

```text
acled_protests_riots_2010_2019_pipeline/
  README.md
  requirements.txt
  data/
    new data/
      ACLED_protests_riots_2010_2019.csv
  notebooks/
    acled_protests_riots_weekly_pipeline.ipynb
    acled_h1a_h1b_h2_h3_inferential_support.ipynb
    acled_h3_grid_no_country_predictive_100km.ipynb
    acled_200km_predictive_robustness.ipynb
  scripts/
    inferential_tests_h1a_h1b_h2_h3.py
    predictive_h3_grid_no_country_100km.py
    robustness_200km_predictive_models.py
  outputs/
    tables/
    figures/
    panels/
```

## Quick Start: Local Computer

1. Create and activate a Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the notebook.

```bash
jupyter notebook notebooks/acled_protests_riots_weekly_pipeline.ipynb
```

The notebook avoids hard-coded local paths and expects the dataset to be inside `data/new data/`.

## Notebook

`notebooks/acled_protests_riots_weekly_pipeline.ipynb` does the following:

1. Loads and validates ACLED protests and riots for 2010-2019.
2. Creates event flags for protests, riots, peaceful protests, and non-peaceful protests.
3. Builds a 100 km weekly grid-cell panel for H1A, H1B, and H2.
4. Creates 1-week, 2-week, and 4-week history windows for focal and neighboring grid cells.
5. Trains and tests logistic regression and random forest models for next-week protest risk.
6. Compares temporal-history models with spatial-neighbor models.
7. Estimates standardized coefficients for the predictive value of peaceful versus non-peaceful protest histories.
8. Builds a country-week panel for the original H3 predictive model.
9. Saves tables, panels, and figures under `outputs/`.

The companion inferential script/notebook tests H1A, H1B, H2, and the revised grid-based H3 across matched 1-week, 2-week, and 4-week future windows.

## Revised H3 Predictive Model: 100 x 100 km Grid Cells, No Country Restriction

The original main notebook includes a country-week H3 predictive baseline. The revised main H3 predictive specification is implemented separately so it matches the updated hypothesis:

```bash
python scripts/predictive_h3_grid_no_country_100km.py
```

The Colab version is `notebooks/acled_h3_grid_no_country_predictive_100km.ipynb`.

This revised H3 predictive model uses:

- Unit: 100 x 100 km grid-cell x week
- Outcome: at least one riot in the focal cell or eight neighboring cells in the next 1, 2, or 4 weeks
- No same-country restriction
- Train period: 2010-2017
- Test period: 2018-2019
- Models: logistic regression and random forest

Current revised H3 predictive result: supported. Adding non-peaceful protest histories improves average precision in all 3 logistic-regression horizons and all 3 random-forest horizons.

## Revised Inferential Tests

Run the supporting inferential analysis after the main notebook has generated `outputs/panels/panel_100km_week.csv.gz`.

```bash
python scripts/inferential_tests_h1a_h1b_h2_h3.py
```

The inferential tests are intended as supporting evidence for the main predictive analysis. They estimate logistic models with two-way cluster-robust standard errors by grid cell and week. The revised H3 model uses the focal cell plus the eight neighboring 100 x 100 km cells and does not restrict neighboring cells by country.

## Predictive Robustness Check: 200 x 200 km Grid Cells

The main predictive analysis uses 100 x 100 km grid cells. As a spatial-scale robustness check, run:

```bash
python scripts/robustness_200km_predictive_models.py
```

The robustness check keeps the same historical split as the main pipeline: train on 2010-2017 and test on 2018-2019. It reruns the predictive H1A, H1B, H2, and revised no-country H3 analyses using 200 x 200 km grid cells. For H3, escalation is defined as riot occurrence in the focal 200 x 200 km grid cell or one of the eight neighboring 200 x 200 km cells over subsequent 1-week, 2-week, and 4-week windows.

The Colab version is `notebooks/acled_200km_predictive_robustness.ipynb`.

Current 200 km robustness results:

- H1A spatial diffusion: supported. Neighboring protest histories improve average precision over focal temporal histories.
- H1B temporal diffusion: supported. Focal protest histories improve average precision over place/time features.
- H2 peaceful diffusion: partially supported. Peaceful coefficients exceed non-peaceful coefficients in 4 of 6 focal/neighbor history contrasts.
- H3 protest escalation: partially supported. Adding non-peaceful histories improves average precision in all 3 logistic horizons and 1 of 3 random-forest horizons.

## Key Generated Outputs

### `outputs/tables/`

- `01_weekly_data_summary.csv`: dataset summary.
- `02_event_types.csv`: counts for protests and riots.
- `03_sub_event_types.csv`: ACLED sub-event type counts.
- `04_events_by_year_type.csv`: yearly protest and riot counts.
- `05_panel_100km_week_summary.csv`: grid-cell x week panel summary.
- `06_h1_h2_cell_model_results_100km_week.csv`: main H1/H2 test-period model results.
- `07_h1_temporal_spatial_uplift_100km_week.csv`: direct H1B and H1A uplift comparisons.
- `08_h2_peaceful_nonpeaceful_diffusion_coefficients.csv`: H2 coefficient table.
- `09_f1_optimal_thresholds_100km_week.csv`: F1-optimal thresholds for selected models.
- `10_precision_at_top_k_100km_week.csv`: precision and lift among highest-risk cell-weeks.
- `11_h3_country_week_riot_model_results.csv`: H3 country-week model results.
- `12_h3_escalation_coefficients.csv`: H3 peaceful/non-peaceful protest history coefficients.
- `13_inferential_summary_1_2_4_week.csv`: summary of H1A, H1B, H2, and revised H3 inferential support.
- `14_h1a_h1b_inferential_1_2_4_week.csv`: H1A/H1B grid-cell-week inferential results.
- `15_h2_inferential_1_2_4_week.csv`: H2 peaceful-minus-non-peaceful inferential contrasts.
- `16_h3_grid_no_country_inferential_1_2_4_week.csv`: revised H3 grid-based inferential results with no same-country restriction.
- `17_inferential_model_formulas_1_2_4_week.csv`: model formulas used for the inferential tests.
- `18_robustness_200km_panel_summary.csv`: summary of the 200 km grid-cell-week robustness panel.
- `19_robustness_200km_h1_h2_model_results.csv`: H1/H2 200 km test-period predictive model results.
- `20_robustness_200km_h1_uplift.csv`: H1A/H1B average-precision and ROC-AUC uplift under 200 km cells.
- `21_robustness_200km_h2_coefficients.csv`: standardized H2 logistic coefficients under 200 km cells.
- `22_robustness_200km_h2_direction_summary.csv`: H2 peaceful-vs-non-peaceful coefficient contrasts under 200 km cells.
- `23_robustness_200km_h3_no_country_model_results.csv`: revised no-country H3 200 km predictive model results.
- `24_robustness_200km_h3_nonpeaceful_uplift.csv`: predictive gain from adding non-peaceful histories for H3 under 200 km cells.
- `25_robustness_200km_h3_nonpeaceful_coefficients.csv`: standardized H3 logistic coefficients under 200 km cells.
- `26_robustness_200km_predictive_summary.csv`: compact robustness conclusion table for H1A, H1B, H2, and H3.
- `27_h3_grid_no_country_predictive_100km_panel_summary.csv`: revised 100 km H3 grid-cell-week panel summary.
- `28_h3_grid_no_country_predictive_100km_model_results.csv`: revised 100 km no-country H3 predictive model results.
- `29_h3_grid_no_country_predictive_100km_nonpeaceful_uplift.csv`: predictive gain from adding non-peaceful histories for revised 100 km H3.
- `30_h3_grid_no_country_predictive_100km_nonpeaceful_coefficients.csv`: standardized revised 100 km H3 logistic coefficients for non-peaceful histories.
- `31_h3_grid_no_country_predictive_100km_summary.csv`: compact revised 100 km H3 predictive conclusion.

### `outputs/figures/`

- `weekly_events_by_type.png`
- `panel_diagnostics_100km_week.png`
- `h1_h2_model_average_precision_100km_week.png`
- `h2_peaceful_nonpeaceful_coefficients.png`
- `precision_recall_curves_100km_week.png`
- `h3_riot_escalation_coefficients.png`

### `outputs/panels/`

- `panel_100km_week.csv.gz`: generated grid-cell x week panel, gzip-compressed because the uncompressed CSV exceeds GitHub's single-file size limit.
- `predictions_100km_week_test.csv`: H1/H2 test predictions.
- `country_week_riot_escalation_panel.csv`: original H3 country-week panel.
- `country_week_riot_predictions_test.csv`: original H3 test predictions.
- `predictions_100km_h3_grid_no_country_test.csv`: revised 100 km no-country H3 test predictions.
- `panel_200km_week.csv.gz`: 200 km grid-cell-week robustness panel.
- `predictions_200km_week_test.csv`: H1/H2 200 km test predictions.
- `predictions_200km_h3_no_country_test.csv`: revised no-country H3 200 km test predictions.

## Notes

This is a predictive and descriptive design, not a causal identification strategy. Evidence that history or neighboring-cell features improve prediction should be interpreted as evidence of predictive temporal or spatial clustering in ACLED event data, not proof that one protest causes another event.

The original predictive notebook includes a country-week H3 model as a legacy baseline. The revised H3 predictive and inferential analyses use a grid-cell-week local-neighborhood definition with no same-country restriction. Results should be interpreted as observational associations rather than causal effects.
