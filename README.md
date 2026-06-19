# ACLED Spatiotemporal Conflict Model

This repository contains a reproducible first-stage spatiotemporal prediction pipeline for studying conflict clustering in Africa using ACLED event data from 2000-2010.

## Research question

Do conflict clusters spread over time? More specifically, does recent conflict in neighboring 100 x 100 km grid cells improve prediction of conflict in a focal cell next month, beyond that cell's own conflict history?

## Main setup

- Unit of analysis: grid-cell x month
- Main grid: 100 x 100 km
- Robustness grids: 50 x 50 km and 200 x 200 km
- Training period: 2000-2005
- Validation period: 2006-2010
- Reserved later test period: 2011-2020, not used in this notebook
- Main target: whether a grid cell has at least one conflict event next month

## Repository structure

```text
acled_spatiotemporal_project/
  README.md
  requirements.txt
  data/
    ACLED Data_2026-06-17.csv
  notebooks/
    acled_spatiotemporal_grid_model.ipynb
  outputs/
    tables/
    figures/
    panels/
  docs/
    ACLED_Spatiotemporal_Conflict_Model_Methods_and_Results_Brief.docx
```

## Quick start: local computer

1. Clone the repository.

```bash
git clone https://github.com/Mat99999/acled-spatiotemporal-conflict-model.git
cd acled_spatiotemporal_project
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

The notebook is designed to avoid hard-coded local file paths. Once this repository is on GitHub, you can open `notebooks/acled_spatiotemporal_grid_model.ipynb` in Colab. If the notebook is opened directly from GitHub and the project files are not present in the Colab runtime, the setup cell clones this repository automatically.

## Notebook

`notebooks/acled_spatiotemporal_grid_model.ipynb` is the main research notebook. It:

1. Loads and validates the ACLED CSV.
2. Builds approximate kilometer-based grid cells.
3. Aggregates ACLED events into a cell-month panel.
4. Creates local lag features and neighboring-cell conflict features.
5. Trains and validates logistic regression and random forest models.
6. Compares local-history models against local-plus-neighbor models.
7. Adds threshold, precision-at-top-k, PR curve, calibration, and robustness diagnostics.
8. Saves outputs and displays the main charts inline.

## Generated outputs

### `outputs/tables/`

- `01_data_summary.csv`: basic dataset summary.
- `02_event_types.csv`: event counts by ACLED event type.
- `03_panel_100km_summary.csv`: summary of the 100 km cell-month panel.
- `04_model_results_100km.csv`: main validation metrics for the 100 km models.
- `05_feature_set_comparison_100km.csv`: comparison of feature sets A-D.
- `07_robustness_50_100_200km.csv`: robustness results across grid sizes.
- `08_f1_optimal_thresholds_100km.csv`: F1-optimal thresholds and associated metrics.
- `09_precision_at_top_k_100km.csv`: precision and lift among top-risk cell-months.
- `10_neighbor_uplift_B_vs_C_100km.csv`: direct comparison of local-history versus local-plus-neighbor models.
- `11_calibration_bins_100km.csv`: calibration-bin table used for calibration curves.

### `outputs/figures/`

- `events_per_month.png`: monthly ACLED event counts.
- `panel_diagnostics_100km.png`: panel activity and event distribution diagnostics.
- `model_average_precision_100km.png`: model comparison by average precision.
- `observed_vs_predicted_risk_map_100km.png`: map-like validation sanity check.
- `precision_recall_curves_100km.png`: PR curves for selected models.
- `calibration_curves_100km.png`: calibration curves for selected models.
- `precision_at_top_k_100km.png`: precision among highest-risk cell-months.
- `robustness_grid_sizes.png`: robustness across 50, 100, and 200 km grids.

### `outputs/panels/`

- `panel_100km_month.csv`: generated 100 km cell-month modeling panel.
- `predictions_100km_validation.csv`: validation predictions for selected 100 km models.
- `06_validation_risk_map_100km.csv`: observed and predicted risk by grid cell.

## Research brief

The `docs/` folder contains `ACLED_Spatiotemporal_Conflict_Model_Methods_and_Results_Brief.docx`, a detailed English methods and preliminary-results brief explaining the research question, data, hypotheses, feature engineering, models, evaluation metrics, robustness checks, limitations, and next steps.

## Notes

This is a first-stage predictive design, not a causal model. Evidence that neighboring-cell features improve prediction should be interpreted as evidence of spatiotemporal clustering or diffusion patterns in the ACLED event data, not proof that conflict in one cell causes conflict in another.

The repository includes generated outputs for convenience, but they can be recreated by running the notebook from the repository root.
