"""Inferential support tests for H1A, H1B, H2, and revised H3.

This script is a companion to the main predictive pipeline. The predictive
models remain the primary analysis; these models provide conventional
inferential supporting evidence with coefficients, odds ratios, confidence
intervals, and p-values.

Key revision for H3
-------------------
H3 is operationalized at the 100 km grid-cell-week level, without a same-country
restriction. The outcome is at least one riot in the focal cell or any of the
eight queen-adjacent neighboring cells during the next 1, 2, or 4 weeks.

Outputs are written to outputs/tables/.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs/panels/panel_100km_week.csv.gz"
OUT = ROOT / "outputs/tables"
OUT.mkdir(parents=True, exist_ok=True)


def fit_clustered_glm(formula: str, data: pd.DataFrame):
    """Fit binomial GLM and return two-way clustered covariance by cell/week."""
    model = smf.glm(
        formula=formula,
        data=data,
        family=sm.families.Binomial(),
    ).fit(maxiter=100)
    cell_codes = pd.factorize(data["cell_id"])[0]
    week_codes = pd.factorize(data["week"])[0]
    covariance, _, _ = cov_cluster_2groups(model, cell_codes, week_codes)
    return model, covariance


def term_row(
    model,
    covariance: np.ndarray,
    data: pd.DataFrame,
    hypothesis: str,
    horizon: int,
    term: str,
    test: str,
):
    names = list(model.params.index)
    idx = names.index(term)
    beta = float(model.params[term])
    se = float(math.sqrt(covariance[idx, idx]))
    p_value = float(2 * norm.sf(abs(beta / se)))
    return {
        "hypothesis": hypothesis,
        "future_window_weeks": horizon,
        "test": test,
        "estimate_log_odds": beta,
        "cluster_se": se,
        "odds_ratio": float(math.exp(beta)),
        "ci_low": float(math.exp(beta - 1.96 * se)),
        "ci_high": float(math.exp(beta + 1.96 * se)),
        "p_value_two_sided": p_value,
        "supported_at_5pct": bool(beta > 0 and p_value < 0.05),
        "n_cell_weeks": int(model.nobs),
        "positive_outcomes": int(data[model.model.endog_names].sum()),
        "cell_clusters": int(data["cell_id"].nunique()),
        "week_clusters": int(data["week"].nunique()),
    }


def contrast_row(
    model,
    covariance: np.ndarray,
    data: pd.DataFrame,
    hypothesis: str,
    horizon: int,
    positive_term: str,
    negative_term: str,
    test: str,
):
    names = list(model.params.index)
    contrast = np.zeros(len(names))
    contrast[names.index(positive_term)] = 1
    contrast[names.index(negative_term)] = -1
    beta = float(contrast @ model.params.to_numpy())
    se = float(math.sqrt(contrast @ covariance @ contrast))
    p_value = float(2 * norm.sf(abs(beta / se)))
    return {
        "hypothesis": hypothesis,
        "future_window_weeks": horizon,
        "test": test,
        "estimate_log_odds_difference": beta,
        "cluster_se": se,
        "ci_low_log_odds_difference": beta - 1.96 * se,
        "ci_high_log_odds_difference": beta + 1.96 * se,
        "p_value_two_sided": p_value,
        "supported_at_5pct": bool(beta > 0 and p_value < 0.05),
        "n_cell_weeks": int(model.nobs),
        "positive_outcomes": int(data[model.model.endog_names].sum()),
        "cell_clusters": int(data["cell_id"].nunique()),
        "week_clusters": int(data["week"].nunique()),
    }


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["week"])
    panel = panel.sort_values(["cell_id", "week"]).reset_index(drop=True)
    panel["year_cat"] = panel["year"].astype("category")
    max_week = panel["week"].max()

    # H1A/H1B/H2 outcomes: focal protest in the next h weeks.
    for h in (1, 2, 4):
        future_protests = sum(
            panel.groupby("cell_id")["protests"].shift(-i)
            for i in range(1, h + 1)
        )
        panel[f"target_protest_{h}w"] = (future_protests > 0).astype(float)
        panel.loc[
            panel["week"] > max_week - pd.Timedelta(weeks=h),
            f"target_protest_{h}w",
        ] = np.nan

        panel[f"neighbor_riot_cells_hist_{h}w"] = (
            panel.groupby("cell_id")["neighbor_riot_cells"]
            .transform(lambda s, w=h: s.rolling(w, min_periods=1).sum())
            .fillna(0)
        )

    # Revised H3 outcome: riot in focal or neighboring cells, no country filter.
    panel["local_riot_now"] = (
        (panel["riots"] > 0) | (panel["neighbor_riot_cells"] > 0)
    ).astype(int)
    for h in (1, 2, 4):
        future_local_riots = sum(
            panel.groupby("cell_id")["local_riot_now"].shift(-i)
            for i in range(1, h + 1)
        )
        panel[f"target_local_riot_{h}w"] = (future_local_riots > 0).astype(float)
        panel.loc[
            panel["week"] > max_week - pd.Timedelta(weeks=h),
            f"target_local_riot_{h}w",
        ] = np.nan

    h1_rows = []
    h2_rows = []
    h3_rows = []
    formula_rows = []

    for h in (1, 2, 4):
        df = panel.dropna(subset=[f"target_protest_{h}w"]).copy()
        for raw in [
            f"protests_hist_{h}w",
            f"neighbor_protests_hist_{h}w",
            f"peaceful_protests_hist_{h}w",
            f"nonpeaceful_protests_hist_{h}w",
            f"neighbor_peaceful_protests_hist_{h}w",
            f"neighbor_nonpeaceful_protests_hist_{h}w",
            f"riots_hist_{h}w",
            f"neighbor_riot_cells_hist_{h}w",
        ]:
            df[f"log_{raw}"] = np.log1p(df[raw])

        h1_formula = (
            f"target_protest_{h}w ~ log_protests_hist_{h}w "
            f"+ log_neighbor_protests_hist_{h}w "
            f"+ log_riots_hist_{h}w + log_neighbor_riot_cells_hist_{h}w "
            "+ grid_x + grid_y + sin_week + cos_week + C(year_cat)"
        )
        h1_model, h1_cov = fit_clustered_glm(h1_formula, df)
        h1_rows.append(
            term_row(
                h1_model,
                h1_cov,
                df,
                "H1B temporal diffusion",
                h,
                f"log_protests_hist_{h}w",
                f"Focal-cell protest history in previous {h} week(s) predicts focal protest in next {h} week(s)",
            )
        )
        h1_rows.append(
            term_row(
                h1_model,
                h1_cov,
                df,
                "H1A spatial diffusion",
                h,
                f"log_neighbor_protests_hist_{h}w",
                f"Neighboring-cell protest history in previous {h} week(s) predicts focal protest in next {h} week(s), controlling focal history",
            )
        )
        formula_rows.append({"horizon": h, "model": "H1A/H1B", "formula": h1_formula})

        h2_formula = (
            f"target_protest_{h}w ~ log_peaceful_protests_hist_{h}w "
            f"+ log_nonpeaceful_protests_hist_{h}w "
            f"+ log_neighbor_peaceful_protests_hist_{h}w "
            f"+ log_neighbor_nonpeaceful_protests_hist_{h}w "
            f"+ log_riots_hist_{h}w + log_neighbor_riot_cells_hist_{h}w "
            "+ grid_x + grid_y + sin_week + cos_week + C(year_cat)"
        )
        h2_model, h2_cov = fit_clustered_glm(h2_formula, df)
        h2_rows.append(
            contrast_row(
                h2_model,
                h2_cov,
                df,
                "H2 peaceful diffusion",
                h,
                f"log_peaceful_protests_hist_{h}w",
                f"log_nonpeaceful_protests_hist_{h}w",
                f"Focal peaceful minus focal non-peaceful history, previous {h} week(s), predicting next {h} week(s)",
            )
        )
        h2_rows.append(
            contrast_row(
                h2_model,
                h2_cov,
                df,
                "H2 peaceful diffusion",
                h,
                f"log_neighbor_peaceful_protests_hist_{h}w",
                f"log_neighbor_nonpeaceful_protests_hist_{h}w",
                f"Neighbor peaceful minus neighbor non-peaceful history, previous {h} week(s), predicting next {h} week(s)",
            )
        )
        formula_rows.append({"horizon": h, "model": "H2", "formula": h2_formula})

        riot_df = panel.dropna(subset=[f"target_local_riot_{h}w"]).copy()
        riot_df["log_local_nonpeaceful"] = np.log1p(
            riot_df["nonpeaceful_protests"]
            + riot_df["neighbor_nonpeaceful_protests"]
        )
        riot_df["log_local_peaceful"] = np.log1p(
            riot_df["peaceful_protests"]
            + riot_df["neighbor_peaceful_protests"]
        )
        riot_df["log_local_riot_history_4w"] = np.log1p(
            riot_df["riots_hist_4w"] + riot_df["neighbor_riot_cells_hist_4w"]
        )
        h3_formula = (
            f"target_local_riot_{h}w ~ log_local_nonpeaceful "
            "+ log_local_peaceful + local_riot_now + log_local_riot_history_4w "
            "+ grid_x + grid_y + sin_week + cos_week + C(year_cat)"
        )
        h3_model, h3_cov = fit_clustered_glm(h3_formula, riot_df)
        h3_rows.append(
            term_row(
                h3_model,
                h3_cov,
                riot_df,
                "H3 protest escalation",
                h,
                "log_local_nonpeaceful",
                f"Non-peaceful protests in focal or neighboring cells predict riots in same local area in next {h} week(s), with no country restriction",
            )
        )
        formula_rows.append({"horizon": h, "model": "H3", "formula": h3_formula})

    summary_rows = []
    for hypothesis, rows in [
        ("H1A spatial diffusion", [r for r in h1_rows if r["hypothesis"].startswith("H1A")]),
        ("H1B temporal diffusion", [r for r in h1_rows if r["hypothesis"].startswith("H1B")]),
        ("H2 peaceful diffusion", h2_rows),
        ("H3 protest escalation", h3_rows),
    ]:
        supported = sum(r["supported_at_5pct"] for r in rows)
        total = len(rows)
        summary_rows.append(
            {
                "hypothesis": hypothesis,
                "supported_tests": supported,
                "total_tests": total,
                "overall_inferential_conclusion": (
                    "Supported"
                    if supported == total
                    else "Partially supported"
                    if supported
                    else "Not supported"
                ),
            }
        )

    outputs = {
        "13_inferential_summary_1_2_4_week.csv": pd.DataFrame(summary_rows),
        "14_h1a_h1b_inferential_1_2_4_week.csv": pd.DataFrame(h1_rows),
        "15_h2_inferential_1_2_4_week.csv": pd.DataFrame(h2_rows),
        "16_h3_grid_no_country_inferential_1_2_4_week.csv": pd.DataFrame(h3_rows),
        "17_inferential_model_formulas_1_2_4_week.csv": pd.DataFrame(formula_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False)
        print(f"Wrote {OUT / filename}")

    print("\nSummary")
    print(outputs["13_inferential_summary_1_2_4_week.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
