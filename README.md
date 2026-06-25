# ACLED Weekly Protest Diffusion and Riot Escalation Pipeline

This repository contains a weekly spatiotemporal prediction pipeline for ACLED protest and riot events in Africa, 2010-2019. It is a separate local adaptation of the earlier ACLED grid model, using a new dataset and a different hypothesis structure.

## Research Focus

The project studies whether protests diffuse over time and space, whether peaceful and non-peaceful protests diffuse differently, and whether non-peaceful protests are followed by riots in the same country.

## Hypotheses

- **H1A Spatial diffusion:** protests in nearby grid cells raise protest risk in the focal cell, even after accounting for the focal cell's own 1-week, 2-week, and 1-month history.
- **H1B Temporal diffusion:** protests in the focal cell in recent periods predict protest risk in the same cell in the next period.
- **H2 Peaceful diffusion:** non-peaceful protests are less likely to diffuse than peaceful protests over space and time. Non-peaceful protests are defined as ACLED protest sub-event types `Protest with intervention` and `Excessive force against protesters`.
- **H3 Protest escalation:** non-peaceful protest events are associated with a higher probability of riots in the same country in subsequent periods.

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
- H3 unit of analysis: country x week
- Main H1/H2 target: whether a grid cell has at least one protest in the next week
- Main H3 target: whether a country has at least one riot in the next week

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
7. Estimates standardized coefficients for peaceful versus non-peaceful diffusion.
8. Builds a country-week panel for H3 and models next-week riot risk.
9. Saves tables, panels, and figures under `outputs/`.

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
- `country_week_riot_escalation_panel.csv`: H3 country-week panel.
- `country_week_riot_predictions_test.csv`: H3 test predictions.

## Notes

This is a predictive and descriptive design, not a causal identification strategy. Evidence that history or neighboring-cell features improve prediction should be interpreted as evidence of temporal or spatial clustering in ACLED event data, not proof that one protest causes another event.

For H3, the country-week model is intentionally separate from the grid-cell model because the hypothesis concerns riots in the same country after non-peaceful protests.
