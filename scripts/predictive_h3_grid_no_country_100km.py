"""Revised H3 predictive model at the 100 km grid-cell-week level.

This script fills the revised main-predictive H3 specification:

    Non-peaceful protest histories are expected to improve prediction of
    subsequent riots in the focal or neighboring 100 x 100 km grid cells,
    without imposing a same-country restriction.

The original main notebook kept an older country-week H3 predictive baseline.
This script uses the same 100 km grid panel as H1/H2 and tests the revised
grid-neighborhood H3 outcome over 1-, 2-, and 4-week horizons.

Outputs are written under outputs/tables/ and outputs/panels/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs/panels/panel_100km_week.csv.gz"
TABLES = ROOT / "outputs/tables"
PANELS = ROOT / "outputs/panels"
TABLES.mkdir(parents=True, exist_ok=True)
PANELS.mkdir(parents=True, exist_ok=True)

GRID_KM = 100
RANDOM_STATE = 42
TRAIN_END = pd.Timestamp("2017-12-31")
TEST_START = pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2019-12-31")

BASE_FEATURES = ["grid_x", "grid_y", "year", "sin_week", "cos_week"]


def load_panel() -> pd.DataFrame:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PANEL_PATH}. Run notebooks/acled_protests_riots_weekly_pipeline.ipynb first."
        )
    panel = pd.read_csv(PANEL_PATH)
    panel["week"] = pd.to_datetime(panel["week"])
    return panel


def add_revised_h3_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["cell_id", "week"]).copy()
    if "neighbor_riot_cells_hist_1w" not in panel.columns:
        for window in (1, 2, 4):
            panel[f"neighbor_riot_cells_hist_{window}w"] = (
                panel.groupby("cell_id")["neighbor_riot_cells"]
                .transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
                .fillna(0)
            )
    panel["local_riot_now"] = (
        (panel["riot_flag"] > 0) | (panel["neighbor_riot_cells"] > 0)
    ).astype(int)
    max_week = panel["week"].max()
    for horizon in (1, 2, 4):
        future_local_riots = sum(
            panel.groupby("cell_id")["local_riot_now"].shift(-i)
            for i in range(1, horizon + 1)
        )
        target = f"target_local_riot_{horizon}w"
        panel[target] = (future_local_riots > 0).astype(float)
        panel.loc[panel["week"] > max_week - pd.Timedelta(weeks=horizon), target] = np.nan
    return panel


def make_masks(panel: pd.DataFrame):
    train = panel["week"] <= TRAIN_END
    test = (panel["week"] >= TEST_START) & (panel["week"] <= TEST_END)
    return train, test


def evaluate_predictions(y_true, y_prob, threshold=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if threshold is None:
        threshold = y_true.mean()
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate_pred": y_pred.mean(),
    }


def h3_feature_sets():
    base = BASE_FEATURES + [
        "riots_hist_1w",
        "riots_hist_2w",
        "riots_hist_4w",
        "neighbor_riot_cells_hist_1w",
        "neighbor_riot_cells_hist_2w",
        "neighbor_riot_cells_hist_4w",
    ]
    peaceful = base + [
        "peaceful_protests_hist_1w",
        "peaceful_protests_hist_2w",
        "peaceful_protests_hist_4w",
        "neighbor_peaceful_protests_hist_1w",
        "neighbor_peaceful_protests_hist_2w",
        "neighbor_peaceful_protests_hist_4w",
    ]
    nonpeaceful = peaceful + [
        "nonpeaceful_protests_hist_1w",
        "nonpeaceful_protests_hist_2w",
        "nonpeaceful_protests_hist_4w",
        "neighbor_nonpeaceful_protests_hist_1w",
        "neighbor_nonpeaceful_protests_hist_2w",
        "neighbor_nonpeaceful_protests_hist_4w",
    ]
    return {
        "A_riot_history": base,
        "B_add_peaceful_history": peaceful,
        "C_add_nonpeaceful_history": nonpeaceful,
    }


def fit_binary_models(panel, feature_sets, outcome, prediction_id_cols):
    train_mask, test_mask = make_masks(panel)
    y_train = panel.loc[train_mask, outcome].astype(int)
    y_test = panel.loc[test_mask, outcome].astype(int)
    predictions = panel.loc[test_mask, prediction_id_cols + [outcome]].copy()
    rows = []
    fitted = {}

    for name, features in feature_sets.items():
        X_train = panel.loc[train_mask, features].fillna(0)
        X_test = panel.loc[test_mask, features].fillna(0)

        logit = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
        )
        logit.fit(X_train, y_train)
        p_logit = logit.predict_proba(X_test)[:, 1]
        row = {
            "grid_km": GRID_KM,
            "outcome": outcome,
            "model": "Logistic regression",
            "feature_set": name,
            "n_features": len(features),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "test_base_rate": float(y_test.mean()),
        }
        row.update(evaluate_predictions(y_test, p_logit))
        rows.append(row)
        predictions[f"p_logit_{name}"] = p_logit
        fitted[f"Logit_{name}"] = (logit, features)

        rf = RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(X_train, y_train)
        p_rf = rf.predict_proba(X_test)[:, 1]
        row = {
            "grid_km": GRID_KM,
            "outcome": outcome,
            "model": "Random forest",
            "feature_set": name,
            "n_features": len(features),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "test_base_rate": float(y_test.mean()),
        }
        row.update(evaluate_predictions(y_test, p_rf))
        rows.append(row)
        predictions[f"p_rf_{name}"] = p_rf
        fitted[f"RF_{name}"] = (rf, features)

    return pd.DataFrame(rows), predictions, fitted


def h3_uplift(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in sorted(perf["horizon_weeks"].unique()):
        for model in ["Logistic regression", "Random forest"]:
            b = perf[
                (perf["horizon_weeks"] == horizon)
                & (perf["model"] == model)
                & (perf["feature_set"] == "B_add_peaceful_history")
            ].iloc[0]
            c = perf[
                (perf["horizon_weeks"] == horizon)
                & (perf["model"] == model)
                & (perf["feature_set"] == "C_add_nonpeaceful_history")
            ].iloc[0]
            rows.append(
                {
                    "grid_km": GRID_KM,
                    "horizon_weeks": int(horizon),
                    "model": model,
                    "delta_average_precision_nonpeaceful": float(
                        c["average_precision"] - b["average_precision"]
                    ),
                    "delta_roc_auc_nonpeaceful": float(c["roc_auc"] - b["roc_auc"]),
                    "delta_brier_nonpeaceful": float(c["brier"] - b["brier"]),
                }
            )
    return pd.DataFrame(rows)


def h3_nonpeaceful_coefficients(h3_fitted_by_horizon):
    rows = []
    for horizon, fitted in h3_fitted_by_horizon.items():
        model, features = fitted["Logit_C_add_nonpeaceful_history"]
        coefs = model.named_steps["logisticregression"].coef_[0]
        for feature, coef in zip(features, coefs):
            if "nonpeaceful" not in feature:
                continue
            rows.append(
                {
                    "grid_km": GRID_KM,
                    "horizon_weeks": horizon,
                    "feature": feature,
                    "standardized_logit_coefficient": float(coef),
                    "odds_ratio_per_1sd": float(np.exp(coef)),
                    "scope": "neighbor" if feature.startswith("neighbor_") else "focal",
                    "history_window": feature.rsplit("_", 1)[-1],
                }
            )
    return pd.DataFrame(rows)


def h3_summary(uplift: pd.DataFrame) -> pd.DataFrame:
    logit_positive = int(
        (
            uplift[uplift["model"] == "Logistic regression"][
                "delta_average_precision_nonpeaceful"
            ]
            > 0
        ).sum()
    )
    rf_positive = int(
        (
            uplift[uplift["model"] == "Random forest"][
                "delta_average_precision_nonpeaceful"
            ]
            > 0
        ).sum()
    )
    if logit_positive == 3 and rf_positive == 3:
        result = "Supported"
    elif logit_positive > 0 or rf_positive > 0:
        result = "Partially supported"
    else:
        result = "Not supported"
    return pd.DataFrame(
        [
            {
                "grid_km": GRID_KM,
                "hypothesis": "H3 protest escalation",
                "predictive_result": result,
                "evidence": f"Adding non-peaceful histories improves average precision in {logit_positive}/3 logistic and {rf_positive}/3 random-forest horizons.",
            }
        ]
    )


def main():
    print("Loading 100 km grid-cell-week panel")
    panel = load_panel()
    print("Creating revised no-country H3 targets")
    panel = add_revised_h3_targets(panel)

    panel_summary = pd.DataFrame(
        {
            "metric": [
                "grid_km",
                "cell_weeks",
                "cells",
                "weeks",
                "target_local_riot_1w_share",
                "target_local_riot_2w_share",
                "target_local_riot_4w_share",
                "train_period",
                "test_period",
            ],
            "value": [
                GRID_KM,
                len(panel),
                panel["cell_id"].nunique(),
                panel["week"].nunique(),
                panel["target_local_riot_1w"].mean(),
                panel["target_local_riot_2w"].mean(),
                panel["target_local_riot_4w"].mean(),
                "2010-2017",
                "2018-2019",
            ],
        }
    )
    panel_summary.to_csv(
        TABLES / "27_h3_grid_no_country_predictive_100km_panel_summary.csv",
        index=False,
    )

    print("Fitting revised H3 no-country 100 km predictive models")
    h3_perf_rows = []
    h3_prediction_frames = []
    h3_fitted_by_horizon = {}
    feature_sets = h3_feature_sets()
    for horizon in (1, 2, 4):
        outcome = f"target_local_riot_{horizon}w"
        subset = panel.dropna(subset=[outcome]).copy()
        perf, preds, fitted = fit_binary_models(
            subset,
            feature_sets,
            outcome,
            ["cell_id", "grid_x", "grid_y", "week", "riots", "neighbor_riot_cells"],
        )
        perf["horizon_weeks"] = horizon
        h3_perf_rows.append(perf)
        preds["horizon_weeks"] = horizon
        h3_prediction_frames.append(preds)
        h3_fitted_by_horizon[horizon] = fitted

    h3_perf = pd.concat(h3_perf_rows, ignore_index=True)
    h3_perf.to_csv(
        TABLES / "28_h3_grid_no_country_predictive_100km_model_results.csv",
        index=False,
    )
    pd.concat(h3_prediction_frames, ignore_index=True).to_csv(
        PANELS / "predictions_100km_h3_grid_no_country_test.csv", index=False
    )

    uplift = h3_uplift(h3_perf)
    uplift.to_csv(
        TABLES / "29_h3_grid_no_country_predictive_100km_nonpeaceful_uplift.csv",
        index=False,
    )

    coefs = h3_nonpeaceful_coefficients(h3_fitted_by_horizon)
    coefs.to_csv(
        TABLES / "30_h3_grid_no_country_predictive_100km_nonpeaceful_coefficients.csv",
        index=False,
    )

    summary = h3_summary(uplift)
    summary.to_csv(
        TABLES / "31_h3_grid_no_country_predictive_100km_summary.csv", index=False
    )

    print("\nRevised H3 predictive summary")
    print(summary.to_string(index=False))
    print("\nH3 non-peaceful uplift")
    print(uplift.to_string(index=False))


if __name__ == "__main__":
    main()
