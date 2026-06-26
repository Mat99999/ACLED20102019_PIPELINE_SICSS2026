"""200 km grid-cell robustness checks for the predictive analyses.

The main pipeline uses 100 x 100 km grid cells. This script reruns the core
predictive analyses using 200 x 200 km cells as a spatial-scale robustness
check. It keeps the same train/test split and modeling logic as the main
pipeline where possible.

Outputs are written under outputs/tables/ and outputs/panels/.
"""

from __future__ import annotations

import math
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
DATA_PATH = ROOT / "data/new data/ACLED_protests_riots_2010_2019.csv"
TABLES = ROOT / "outputs/tables"
PANELS = ROOT / "outputs/panels"
TABLES.mkdir(parents=True, exist_ok=True)
PANELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ROBUST_GRID_KM = 200
TRAIN_END = pd.Timestamp("2017-12-31")
TEST_START = pd.Timestamp("2018-01-01")
TEST_END = pd.Timestamp("2019-12-31")
NONPEACEFUL_PROTEST_SUBTYPES = [
    "Protest with intervention",
    "Excessive force against protesters",
]

BASE_FEATURES = ["grid_x", "grid_y", "year", "sin_week", "cos_week"]
TEMPORAL_FEATURES = BASE_FEATURES + [
    "protests_hist_1w",
    "protests_hist_2w",
    "protests_hist_4w",
]
SPATIOTEMPORAL_FEATURES = TEMPORAL_FEATURES + [
    "neighbor_protests_hist_1w",
    "neighbor_protests_hist_2w",
    "neighbor_protests_hist_4w",
]
PEACEFUL_DIFFUSION_FEATURES = BASE_FEATURES + [
    "peaceful_protests_hist_1w",
    "peaceful_protests_hist_2w",
    "peaceful_protests_hist_4w",
    "nonpeaceful_protests_hist_1w",
    "nonpeaceful_protests_hist_2w",
    "nonpeaceful_protests_hist_4w",
    "neighbor_peaceful_protests_hist_1w",
    "neighbor_peaceful_protests_hist_2w",
    "neighbor_peaceful_protests_hist_4w",
    "neighbor_nonpeaceful_protests_hist_1w",
    "neighbor_nonpeaceful_protests_hist_2w",
    "neighbor_nonpeaceful_protests_hist_4w",
]
FEATURE_SETS = {
    "A_place_time": BASE_FEATURES,
    "B_temporal_focal_history": TEMPORAL_FEATURES,
    "C_spatiotemporal_neighbors": SPATIOTEMPORAL_FEATURES,
    "D_peaceful_nonpeaceful_diffusion": PEACEFUL_DIFFUSION_FEATURES,
}


def load_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["week"] = df["event_date"].dt.to_period("W-SUN").dt.start_time
    df["is_protest"] = (df["event_type"] == "Protests").astype(int)
    df["is_riot"] = (df["event_type"] == "Riots").astype(int)
    df["peaceful_protest"] = (
        (df["event_type"] == "Protests")
        & (df["sub_event_type"] == "Peaceful protest")
    ).astype(int)
    df["nonpeaceful_protest"] = (
        (df["event_type"] == "Protests")
        & (df["sub_event_type"].isin(NONPEACEFUL_PROTEST_SUBTYPES))
    ).astype(int)
    return df


def add_grid_columns(events: pd.DataFrame, grid_km: int) -> pd.DataFrame:
    out = events.copy()
    lat0 = out["latitude"].median()
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * math.cos(math.radians(lat0))
    out["x_km"] = out["longitude"] * km_per_degree_lon
    out["y_km"] = out["latitude"] * km_per_degree_lat
    out["grid_x"] = np.floor(out["x_km"] / grid_km).astype(int)
    out["grid_y"] = np.floor(out["y_km"] / grid_km).astype(int)
    out["cell_id"] = out["grid_x"].astype(str) + "_" + out["grid_y"].astype(str)
    out["grid_km"] = grid_km
    return out


def add_history_windows(
    panel: pd.DataFrame, group_col: str, columns, windows=(1, 2, 4)
) -> pd.DataFrame:
    panel = panel.sort_values([group_col, "week"]).copy()
    for col in columns:
        for window in windows:
            panel[f"{col}_hist_{window}w"] = (
                panel.groupby(group_col)[col]
                .transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
                .fillna(0)
            )
    return panel


def add_future_targets(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["cell_id", "week"]).copy()
    max_week = panel["week"].max()
    for horizon in (1, 2, 4):
        future_protests = sum(
            panel.groupby("cell_id")["protest_flag"].shift(-i)
            for i in range(1, horizon + 1)
        )
        panel[f"target_protest_{horizon}w"] = (future_protests > 0).astype(float)
        panel.loc[
            panel["week"] > max_week - pd.Timedelta(weeks=horizon),
            f"target_protest_{horizon}w",
        ] = np.nan

        future_riots = sum(
            panel.groupby("cell_id")["local_riot_now"].shift(-i)
            for i in range(1, horizon + 1)
        )
        panel[f"target_local_riot_{horizon}w"] = (future_riots > 0).astype(float)
        panel.loc[
            panel["week"] > max_week - pd.Timedelta(weeks=horizon),
            f"target_local_riot_{horizon}w",
        ] = np.nan
    return panel


def build_cell_week_panel(events: pd.DataFrame, grid_km: int) -> pd.DataFrame:
    e = add_grid_columns(events.dropna(subset=["latitude", "longitude", "week"]), grid_km)
    agg = (
        e.groupby(["cell_id", "grid_x", "grid_y", "week"])
        .agg(
            protests=("is_protest", "sum"),
            riots=("is_riot", "sum"),
            peaceful_protests=("peaceful_protest", "sum"),
            nonpeaceful_protests=("nonpeaceful_protest", "sum"),
            fatalities=("fatalities", "sum"),
            modal_country=(
                "country",
                lambda s: s.mode().iat[0] if not s.mode().empty else np.nan,
            ),
        )
        .reset_index()
    )
    cells = agg[["cell_id", "grid_x", "grid_y"]].drop_duplicates()
    weeks = pd.date_range(agg["week"].min(), agg["week"].max(), freq="W-MON")
    panel = cells.merge(pd.DataFrame({"week": weeks}), how="cross")
    panel = panel.merge(agg, on=["cell_id", "grid_x", "grid_y", "week"], how="left")
    count_cols = [
        "protests",
        "riots",
        "peaceful_protests",
        "nonpeaceful_protests",
        "fatalities",
    ]
    panel[count_cols] = panel[count_cols].fillna(0)
    panel["modal_country"] = panel.groupby("cell_id")["modal_country"].ffill().bfill()
    panel["protest_flag"] = (panel["protests"] > 0).astype(int)
    panel["riot_flag"] = (panel["riots"] > 0).astype(int)

    base = panel[
        [
            "grid_x",
            "grid_y",
            "week",
            "protests",
            "peaceful_protests",
            "nonpeaceful_protests",
            "riot_flag",
        ]
    ].copy()
    shifted = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tmp = base.copy()
            tmp["grid_x"] = tmp["grid_x"] - dx
            tmp["grid_y"] = tmp["grid_y"] - dy
            shifted.append(tmp)
    neigh = pd.concat(shifted, ignore_index=True)
    neigh_agg = (
        neigh.groupby(["grid_x", "grid_y", "week"])
        .agg(
            neighbor_protests=("protests", "sum"),
            neighbor_peaceful_protests=("peaceful_protests", "sum"),
            neighbor_nonpeaceful_protests=("nonpeaceful_protests", "sum"),
            neighbor_riot_cells=("riot_flag", "sum"),
        )
        .reset_index()
    )
    panel = panel.merge(neigh_agg, on=["grid_x", "grid_y", "week"], how="left")
    for col in [
        "neighbor_protests",
        "neighbor_peaceful_protests",
        "neighbor_nonpeaceful_protests",
        "neighbor_riot_cells",
    ]:
        panel[col] = panel[col].fillna(0)
    panel["local_riot_now"] = (
        (panel["riot_flag"] > 0) | (panel["neighbor_riot_cells"] > 0)
    ).astype(int)

    panel = add_history_windows(
        panel,
        "cell_id",
        [
            "protests",
            "peaceful_protests",
            "nonpeaceful_protests",
            "riots",
            "neighbor_protests",
            "neighbor_peaceful_protests",
            "neighbor_nonpeaceful_protests",
            "neighbor_riot_cells",
        ],
        windows=(1, 2, 4),
    )
    panel["target_protest_next_week"] = panel.groupby("cell_id")["protest_flag"].shift(-1)
    panel["target_riot_next_week"] = panel.groupby("cell_id")["riot_flag"].shift(-1)
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype(int)
    panel["year"] = panel["week"].dt.year
    panel["sin_week"] = np.sin(2 * np.pi * panel["week_of_year"] / 52)
    panel["cos_week"] = np.cos(2 * np.pi * panel["week_of_year"] / 52)
    panel = add_future_targets(panel)
    return panel.dropna(subset=["target_protest_next_week"]).reset_index(drop=True)


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


def fit_binary_models(panel, feature_sets, outcome, prediction_id_cols, prefix=""):
    train_mask, test_mask = make_masks(panel)
    y_train = panel.loc[train_mask, outcome].astype(int)
    y_test = panel.loc[test_mask, outcome].astype(int)
    predictions = panel.loc[test_mask, prediction_id_cols + [outcome]].copy()
    rows = []
    fitted = {}
    for name, features in feature_sets.items():
        features = [f for f in features if f in panel.columns]
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
            "grid_km": ROBUST_GRID_KM,
            "outcome": outcome,
            "model": f"{prefix}Logistic regression",
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
            "grid_km": ROBUST_GRID_KM,
            "outcome": outcome,
            "model": f"{prefix}Random forest",
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


def h2_coefficients(fitted):
    model, features = fitted["Logit_D_peaceful_nonpeaceful_diffusion"]
    coefs = model.named_steps["logisticregression"].coef_[0]
    rows = []
    for feature, coef in zip(features, coefs):
        if "peaceful" not in feature:
            continue
        scope = (
            "spatial_neighbor_history"
            if feature.startswith("neighbor_")
            else "temporal_focal_history"
        )
        protest_type = "nonpeaceful" if "nonpeaceful" in feature else "peaceful"
        window = feature.rsplit("_", 1)[-1]
        rows.append(
            {
                "grid_km": ROBUST_GRID_KM,
                "feature": feature,
                "standardized_logit_coefficient": float(coef),
                "odds_ratio_per_1sd": float(np.exp(coef)),
                "scope": scope,
                "protest_type": protest_type,
                "window": window,
            }
        )
    return pd.DataFrame(rows)


def h1_uplift(perf):
    pivot = perf.pivot_table(
        index="model", columns="feature_set", values=["average_precision", "roc_auc", "brier"]
    )
    rows = []
    for model in ["Logistic regression", "Random forest"]:
        rows.append(
            {
                "grid_km": ROBUST_GRID_KM,
                "model": model,
                "delta_ap_temporal_minus_place_time": float(
                    pivot.loc[model, ("average_precision", "B_temporal_focal_history")]
                    - pivot.loc[model, ("average_precision", "A_place_time")]
                ),
                "delta_ap_spatial_minus_temporal": float(
                    pivot.loc[model, ("average_precision", "C_spatiotemporal_neighbors")]
                    - pivot.loc[model, ("average_precision", "B_temporal_focal_history")]
                ),
                "delta_auc_temporal_minus_place_time": float(
                    pivot.loc[model, ("roc_auc", "B_temporal_focal_history")]
                    - pivot.loc[model, ("roc_auc", "A_place_time")]
                ),
                "delta_auc_spatial_minus_temporal": float(
                    pivot.loc[model, ("roc_auc", "C_spatiotemporal_neighbors")]
                    - pivot.loc[model, ("roc_auc", "B_temporal_focal_history")]
                ),
            }
        )
    return pd.DataFrame(rows)


def h2_direction_summary(coefs):
    rows = []
    for scope in ["temporal_focal_history", "spatial_neighbor_history"]:
        for window in ["1w", "2w", "4w"]:
            peaceful = coefs[
                (coefs["scope"] == scope)
                & (coefs["window"] == window)
                & (coefs["protest_type"] == "peaceful")
            ]["standardized_logit_coefficient"].iloc[0]
            nonpeaceful = coefs[
                (coefs["scope"] == scope)
                & (coefs["window"] == window)
                & (coefs["protest_type"] == "nonpeaceful")
            ]["standardized_logit_coefficient"].iloc[0]
            rows.append(
                {
                    "grid_km": ROBUST_GRID_KM,
                    "scope": scope,
                    "window": window,
                    "peaceful_coefficient": float(peaceful),
                    "nonpeaceful_coefficient": float(nonpeaceful),
                    "peaceful_minus_nonpeaceful": float(peaceful - nonpeaceful),
                    "supports_h2_direction": bool(peaceful > nonpeaceful),
                }
            )
    return pd.DataFrame(rows)


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


def h3_uplift(perf):
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
                    "grid_km": ROBUST_GRID_KM,
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
                    "grid_km": ROBUST_GRID_KM,
                    "horizon_weeks": horizon,
                    "feature": feature,
                    "standardized_logit_coefficient": float(coef),
                    "odds_ratio_per_1sd": float(np.exp(coef)),
                    "scope": "neighbor"
                    if feature.startswith("neighbor_")
                    else "focal",
                    "history_window": feature.rsplit("_", 1)[-1],
                }
            )
    return pd.DataFrame(rows)


def result_summary(h1, h2, h3):
    h1_rf = h1[h1["model"] == "Random forest"].iloc[0]
    h2_support = int(h2["supports_h2_direction"].sum())
    h3_logit_positive = int(
        (
            h3[h3["model"] == "Logistic regression"][
                "delta_average_precision_nonpeaceful"
            ]
            > 0
        ).sum()
    )
    h3_rf_positive = int(
        (
            h3[h3["model"] == "Random forest"][
                "delta_average_precision_nonpeaceful"
            ]
            > 0
        ).sum()
    )
    return pd.DataFrame(
        [
            {
                "grid_km": ROBUST_GRID_KM,
                "hypothesis": "H1A spatial diffusion",
                "robustness_result": "Supported"
                if h1_rf["delta_ap_spatial_minus_temporal"] > 0
                else "Not supported",
                "evidence": "Adding neighboring protest histories improves average precision over focal temporal history in the random forest.",
            },
            {
                "grid_km": ROBUST_GRID_KM,
                "hypothesis": "H1B temporal diffusion",
                "robustness_result": "Supported"
                if h1_rf["delta_ap_temporal_minus_place_time"] > 0
                else "Not supported",
                "evidence": "Adding focal protest histories improves average precision over place/time features in the random forest.",
            },
            {
                "grid_km": ROBUST_GRID_KM,
                "hypothesis": "H2 peaceful diffusion",
                "robustness_result": "Partially supported"
                if 0 < h2_support < len(h2)
                else ("Supported" if h2_support == len(h2) else "Not supported"),
                "evidence": f"Peaceful coefficient exceeds non-peaceful coefficient in {h2_support} of {len(h2)} focal/neighbor history contrasts.",
            },
            {
                "grid_km": ROBUST_GRID_KM,
                "hypothesis": "H3 protest escalation",
                "robustness_result": "Supported"
                if h3_logit_positive == 3 and h3_rf_positive == 3
                else "Partially supported",
                "evidence": f"Adding non-peaceful histories improves average precision in {h3_logit_positive}/3 logistic and {h3_rf_positive}/3 random-forest horizons.",
            },
        ]
    )


def main():
    print("Loading events")
    events = load_events()
    print("Building 200 km panel")
    panel = build_cell_week_panel(events, ROBUST_GRID_KM)
    panel.to_csv(PANELS / "panel_200km_week.csv.gz", index=False, compression="gzip")

    panel_summary = pd.DataFrame(
        {
            "metric": [
                "grid_km",
                "cell_weeks",
                "cells",
                "weeks",
                "target_protest_next_week_share",
                "target_local_riot_1w_share",
                "target_local_riot_2w_share",
                "target_local_riot_4w_share",
            ],
            "value": [
                ROBUST_GRID_KM,
                len(panel),
                panel["cell_id"].nunique(),
                panel["week"].nunique(),
                panel["target_protest_next_week"].mean(),
                panel["target_local_riot_1w"].mean(),
                panel["target_local_riot_2w"].mean(),
                panel["target_local_riot_4w"].mean(),
            ],
        }
    )
    panel_summary.to_csv(TABLES / "18_robustness_200km_panel_summary.csv", index=False)

    print("Fitting H1/H2 protest-risk robustness models")
    h1h2_perf, h1h2_predictions, h1h2_fitted = fit_binary_models(
        panel.dropna(subset=["target_protest_next_week"]),
        FEATURE_SETS,
        "target_protest_next_week",
        ["cell_id", "grid_x", "grid_y", "week", "protests", "neighbor_protests"],
    )
    h1h2_perf.to_csv(
        TABLES / "19_robustness_200km_h1_h2_model_results.csv", index=False
    )
    h1h2_predictions.to_csv(
        PANELS / "predictions_200km_week_test.csv", index=False
    )
    h1 = h1_uplift(h1h2_perf)
    h1.to_csv(TABLES / "20_robustness_200km_h1_uplift.csv", index=False)
    h2_coef = h2_coefficients(h1h2_fitted)
    h2_coef.to_csv(
        TABLES / "21_robustness_200km_h2_coefficients.csv", index=False
    )
    h2 = h2_direction_summary(h2_coef)
    h2.to_csv(
        TABLES / "22_robustness_200km_h2_direction_summary.csv", index=False
    )

    print("Fitting revised H3 no-country local-grid robustness models")
    h3_perf_rows = []
    h3_fitted_by_horizon = {}
    h3_prediction_frames = []
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
        TABLES / "23_robustness_200km_h3_no_country_model_results.csv", index=False
    )
    pd.concat(h3_prediction_frames, ignore_index=True).to_csv(
        PANELS / "predictions_200km_h3_no_country_test.csv", index=False
    )
    h3 = h3_uplift(h3_perf)
    h3.to_csv(
        TABLES / "24_robustness_200km_h3_nonpeaceful_uplift.csv", index=False
    )
    h3_coefs = h3_nonpeaceful_coefficients(h3_fitted_by_horizon)
    h3_coefs.to_csv(
        TABLES / "25_robustness_200km_h3_nonpeaceful_coefficients.csv",
        index=False,
    )

    summary = result_summary(h1, h2, h3)
    summary.to_csv(TABLES / "26_robustness_200km_predictive_summary.csv", index=False)

    print("\nRobustness summary")
    print(summary.to_string(index=False))
    print("\nH3 non-peaceful uplift")
    print(h3.to_string(index=False))


if __name__ == "__main__":
    main()
