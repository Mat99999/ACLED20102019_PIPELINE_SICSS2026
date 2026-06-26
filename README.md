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
  scripts/
    inferential_tests_h1a_h1b_h2_h3.py
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

## Revised Inferential Tests

Run the supporting inferential analysis after the main notebook has generated `outputs/panels/panel_100km_week.csv.gz`.

```bash
python scripts/inferential_tests_h1a_h1b_h2_h3.py
```

The inferential tests are intended as supporting evidence for the main predictive analysis. They estimate logistic models with two-way cluster-robust standard errors by grid cell and week. The revised H3 model uses the focal cell plus the eight neighboring 100 x 100 km cells and does not restrict neighboring cells by country.

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

## Notes

This is a predictive and descriptive design, not a causal identification strategy. Evidence that history or neighboring-cell features improve prediction should be interpreted as evidence of predictive temporal or spatial clustering in ACLED event data, not proof that one protest causes another event.

The original predictive notebook includes a country-week H3 model. The revised inferential H3 analysis instead uses a grid-cell-week local-neighborhood definition with no same-country restriction. Results should be interpreted as observational associations rather than causal effects.
