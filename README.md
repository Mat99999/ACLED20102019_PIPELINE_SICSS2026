# ACLED Spatiotemporal Protest Diffusion Model

This repository contains a reproducible first-stage spatiotemporal prediction pipeline for studying protest diffusion in Africa using protest-only ACLED event data from January 2000 through December 2020.

## Research question

How do protests spread over time and space in Africa?

## Hypotheses

- **H1 Spatial diffusion:** protests in nearby grid cells raise the risk of protest in the focal cell, even after accounting for the focal cell's own 1-month, 2-month, 6-month, and 1-year protest history.
- **H2 Scale robustness:** the same broad pattern should be visible when the grid is made smaller, 25 x 25 km, or larger, 100 x 100 km.
- **H3 Fatality heterogeneity:** nonfatal protests produce stronger spillover effects than fatal protests. The event-level variable `fatal_protest` is coded as `1` when an ACLED protest event has one or more fatalities and `0` otherwise.

## Main setup

- Unit of analysis: grid-cell x month
- Main grid: independent African land grid at 50 x 50 km, built from Natural Earth country boundaries
- Robustness grids: 25 x 25 km and 100 x 100 km
- Training target period: 2000-2012
- Validation target period: 2013-2015, used for model and threshold choices
- Final test target period: 2016-2020
- Main target: whether a grid cell has at least one protest event next month
- Main model family: logistic regression, with random forest used as a predictive benchmark

## Repository structure

```text
acled-spatiotemporal-conflict-model/
  README.md
  requirements.txt
  data/
    new data/
      comprehensive protest data.csv
      Protest_2000_2015_train.csv
      Protest_2016_2020_test.csv
    old data/
      ACLED Data_2026-06-17.csv
  notebooks/
    acled_spatiotemporal_grid_model.ipynb
  outputs/
    tables/
    figures/
    panels/
```

## Quick start: local computer

1. Clone the repository.

```bash
git clone https://github.com/Mat99999/acled-spatiotemporal-conflict-model.git
cd acled-spatiotemporal-conflict-model
```

2. Create and activate a Python environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Open and run the notebook.

```bash
jupyter notebook notebooks/acled_spatiotemporal_grid_model.ipynb
```

If you use VSCode, open the repository folder and run the notebook with the environment created above.

## Quick start: Google Colab

Open the notebook in Colab: https://colab.research.google.com/github/Mat99999/acled-spatiotemporal-conflict-model/blob/main/notebooks/acled_spatiotemporal_grid_model.ipynb

The notebook is designed to avoid hard-coded local file paths. Once this repository is on GitHub, you can open `notebooks/acled_spatiotemporal_grid_model.ipynb` in Colab. If the notebook is opened directly from GitHub and the project files are not present in the Colab runtime, the setup cell clones this repository automatically and reads `data/new data/comprehensive protest data.csv`.

The default Colab run uses `ROBUSTNESS_MODE = "quick"`. This still runs the 25, 50, and 100 km robustness check, but only for the core random forest B/C/D feature sets and it reuses the already fitted 50 km validation results. Set `ROBUSTNESS_MODE = "full"` for the complete Logit plus random forest robustness table, or `ROBUSTNESS_MODE = "skip"` to run only the main 50 km analysis.

Large generated panel and prediction CSVs are not written by default in Colab. Set `SAVE_LARGE_OUTPUTS = True` only when those files are needed locally.

## Notebook

`notebooks/acled_spatiotemporal_grid_model.ipynb` is the main research notebook. It:

1. Loads and validates the comprehensive ACLED protest CSV.
2. Builds an independent African land grid using Natural Earth boundaries.
3. Aggregates ACLED protest events into a cell-month panel.
4. Creates local lag features and neighboring-cell protest features.
5. Uses 2000-2012 for training, 2013-2015 for validation, and 2016-2020 for the final test.
6. Trains and tests logistic regression and random forest models.
7. Compares local-history models against local-plus-neighbor models.
8. Adds formal fatal versus nonfatal neighboring-protest diagnostics for H3.
9. Adds ranking, PR curve, calibration, map-like, and robustness diagnostics.
10. Saves outputs and displays the main charts inline.

## Generated outputs

### `outputs/tables/`

- `01_data_summary.csv`: basic dataset summary.
- `02_event_types.csv`: event counts by ACLED event type.
- `02b_protest_sub_event_types.csv`: event counts by protest subtype.
- `02c_analysis_periods.csv`: training, validation, and final-test event counts.
- `03_panel_50km_summary.csv`: summary of the 50 km cell-month panel.
- `04a_model_results_50km_validation.csv`: validation metrics for model development.
- `04b_model_results_50km_final_test.csv`: final test metrics for the 50 km models.
- `05_feature_set_comparison_50km.csv`: comparison of feature sets A-E.
- `07_robustness_25_50_100km_validation.csv`: validation-period robustness results across grid sizes. By default this is the faster random forest robustness table; set `ROBUSTNESS_MODE = "full"` in the notebook for the complete Logit plus random forest version.
- `08_validation_selected_thresholds_final_test.csv`: final-test threshold metrics using validation-selected thresholds.
- `09_precision_at_top_k_final_test.csv`: precision and lift among top-risk cell-months.
- `10_neighbor_uplift_B_vs_C_final_test.csv`: direct comparison of local-history versus local-plus-neighbor models.
- `11_calibration_bins_50km.csv`: calibration-bin table used for calibration curves.
- `12a_h3_coefficients_50km.csv`: H3 coefficients comparing neighboring nonfatal and fatal protest spillovers.
- `12b_h3_wald_test_50km.csv`: formal Wald test for the H3 neighbor-coefficient contrast.

### `outputs/figures/`

- `events_per_month.png`: monthly ACLED protest event counts.
- `africa_land_grid_50km.png`: map-like diagnostic of the 50 km African land grid.
- `panel_diagnostics_50km.png`: panel activity and event distribution diagnostics.
- `model_average_precision_50km.png`: model comparison by average precision.
- `observed_vs_predicted_risk_map_50km.png`: map-like test-period sanity check.
- `precision_recall_curves_50km.png`: PR curves for selected models.
- `calibration_curves_50km.png`: calibration curves for selected models.
- `precision_at_top_k_50km.png`: precision among highest-risk cell-months.
- `robustness_grid_sizes.png`: robustness across 25, 50, and 100 km grids.

### `outputs/panels/`

- `panel_50km_month.csv`: generated 50 km cell-month modeling panel.
- `predictions_50km_validation.csv`: validation predictions for selected 50 km models.
- `predictions_50km_final_test.csv`: final-test predictions for selected 50 km models.
- `06_test_risk_map_50km.csv`: observed and predicted test risk by grid cell.

## Notes

This is a first-stage predictive design, not a causal model. Evidence that neighboring-cell features improve prediction should be interpreted as evidence of spatiotemporal clustering or diffusion patterns in the ACLED protest data, not proof that protests in one cell cause protests in another.

The repository includes generated outputs for convenience, but they can be recreated by running the notebook from the repository root.
