# ACLED Spatiotemporal Protest Diffusion Model

This repository contains a reproducible first-stage spatiotemporal prediction pipeline for studying protest diffusion in Africa using protest-only ACLED event data from January 2000 through December 2020.

## Research question

How do protests spread over time and space in Africa?

## Hypotheses

- **H1 Spatial diffusion:** protests in nearby grid cells raise the risk of protest in the focal cell, even after accounting for the focal cell's own 1-month, 2-month, 6-month, and 1-year protest history.
- **H2 Scale robustness:** the same broad pattern should be visible when the grid is made smaller, 50 x 50 km, or larger, 200 x 200 km.
- **H3 Fatality heterogeneity:** nonfatal protests produce stronger spillover effects than fatal protests. The event-level variable `fatal_protest` is coded as `1` when an ACLED protest event has one or more fatalities and `0` otherwise.

## Main setup

- Unit of analysis: grid-cell x month
- Main grid: 100 x 100 km
- Robustness grids: 50 x 50 km and 200 x 200 km
- Training period: 2000-2015
- Test period: 2016-2020
- Main target: whether a grid cell has at least one protest event next month
- Main model family: logistic regression, with random forest used as a predictive benchmark

## Repository structure

```text
acled-spatiotemporal-conflict-model/
  README.md
  requirements.txt
  data/
    new data/
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

The notebook is designed to avoid hard-coded local file paths. Once this repository is on GitHub, you can open `notebooks/acled_spatiotemporal_grid_model.ipynb` in Colab. If the notebook is opened directly from GitHub and the project files are not present in the Colab runtime, the setup cell clones this repository automatically and reads the two split CSV files in `data/new data/`.

## Notebook

`notebooks/acled_spatiotemporal_grid_model.ipynb` is the main research notebook. It:

1. Loads and validates the ACLED protest training and test CSV files.
2. Builds approximate kilometer-based grid cells.
3. Aggregates ACLED protest events into a cell-month panel.
4. Creates local lag features and neighboring-cell protest features.
5. Trains and tests logistic regression and random forest models.
6. Compares local-history models against local-plus-neighbor models.
7. Adds fatal versus nonfatal neighboring-protest features for H3.
8. Adds threshold, precision-at-top-k, PR curve, calibration, and robustness diagnostics.
9. Saves outputs and displays the main charts inline.

## Generated outputs

### `outputs/tables/`

- `01_data_summary.csv`: basic dataset summary.
- `02_event_types.csv`: event counts by ACLED event type.
- `02b_protest_sub_event_types.csv`: event counts by protest subtype.
- `02c_train_test_split_summary.csv`: row counts and basic split diagnostics.
- `03_panel_100km_summary.csv`: summary of the 100 km cell-month panel.
- `04_model_results_100km.csv`: main test metrics for the 100 km models.
- `05_feature_set_comparison_100km.csv`: comparison of feature sets A-E.
- `07_robustness_50_100_200km.csv`: robustness results across grid sizes.
- `08_f1_optimal_thresholds_100km.csv`: F1-optimal thresholds and associated metrics.
- `09_precision_at_top_k_100km.csv`: precision and lift among top-risk cell-months.
- `10_neighbor_uplift_B_vs_C_100km.csv`: direct comparison of local-history versus local-plus-neighbor models.
- `11_calibration_bins_100km.csv`: calibration-bin table used for calibration curves.
- `12_h3_neighbor_fatality_spillover_100km.csv`: standardized logit coefficients comparing neighboring nonfatal and fatal protest spillovers.

### `outputs/figures/`

- `events_per_month.png`: monthly ACLED protest event counts.
- `panel_diagnostics_100km.png`: panel activity and event distribution diagnostics.
- `model_average_precision_100km.png`: model comparison by average precision.
- `observed_vs_predicted_risk_map_100km.png`: map-like test-period sanity check.
- `precision_recall_curves_100km.png`: PR curves for selected models.
- `calibration_curves_100km.png`: calibration curves for selected models.
- `precision_at_top_k_100km.png`: precision among highest-risk cell-months.
- `robustness_grid_sizes.png`: robustness across 50, 100, and 200 km grids.

### `outputs/panels/`

- `panel_100km_month.csv`: generated 100 km cell-month modeling panel.
- `predictions_100km_test.csv`: test predictions for selected 100 km models.
- `06_test_risk_map_100km.csv`: observed and predicted test risk by grid cell.

## Notes

This is a first-stage predictive design, not a causal model. Evidence that neighboring-cell features improve prediction should be interpreted as evidence of spatiotemporal clustering or diffusion patterns in the ACLED protest data, not proof that protests in one cell cause protests in another.

The repository includes generated outputs for convenience, but they can be recreated by running the notebook from the repository root.
