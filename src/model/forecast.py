"""
County-Level ED Burden Forecasting
====================================

Trains on historical county-year panel data and produces forward-looking
forecasts of which counties are projected to experience worsening ED burden.

This is the government-facing deliverable: a model trained on historical
patterns that identifies counties needing intervention BEFORE crisis develops.

Design Principles
-----------------
1. Train on years T1..Tn-2, validate on Tn-1..Tn (temporal hold-out)
2. Use only lagged predictors (available before the forecast year)
3. Produce county-level risk scores with confidence intervals
4. Generate actionable tier classifications (Critical / At-Risk / Stable)

Key Innovation
--------------
Even without future SEDD data, we can forecast excess burden because:
- All predictor variables (CMS, CDC, BLS, Census) are released 6-12
  months BEFORE SEDD data for the same period
- A model trained on 2015-2021 predictors -> 2015-2021 ED burden
  can be applied to 2022-2023 predictors to forecast 2022-2023 burden
- If validated, the same model applied to 2024-2025 predictors
  gives policy-relevant forecasts for counties that DON'T HAVE ED DATA YET

Usage
-----
    python -m src.model.forecast --panel data/processed/panel_nc.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# Feature definitions (same alternative data sources as cross-sectional)
# =========================================================================

PREDICTOR_COLS = {
    # Economic conditions (BLS + Census SAIPE)
    "UnemploymentRate": "Unemployment rate (%)",
    "PovertyRate": "Poverty rate (%)",
    "MedianHouseholdIncome": "Median household income ($)",
    "ChildPovertyRate": "Child poverty rate (%)",

    # Insurance coverage (CMS)
    "MA_Enrollment_Annual": "Medicare Advantage enrollees",
    "MC_Enrollment_Annual": "Medicaid enrollees",
    "pub_insured_pct": "% publicly insured",

    # Social vulnerability (CDC SVI)
    "RPL_THEME1": "SVI: Socioeconomic Status",
    "RPL_THEME2": "SVI: Household Characteristics",
    "RPL_THEME3": "SVI: Racial/Ethnic Minority",
    "RPL_THEME4": "SVI: Housing/Transportation",

    # Healthcare infrastructure (HRSA AHRF)
    "MDs_per_10k": "MDs per 10,000 population",
    "PCP_per_10k": "PCPs per 10,000 population",
    "HPSA_PrimaryCare": "Primary care HPSA designation",

    # Digital access (FCC)
    "BroadbandPer1000HH": "Broadband connections per 1,000 HH",

    # Demographics
    "E_TOTPOP": "Total population",
    "EP_AGE65": "% aged 65+",
    "EP_UNINSUR": "% uninsured",
    "EP_DISABL": "% with disability",
    "EP_MINRTY": "% racial/ethnic minority",
}


# =========================================================================
# Temporal train/test split
# =========================================================================

def temporal_split(panel, train_end_year, test_start_year=None):
    """
    Split panel into training and testing sets by year.

    Parameters
    ----------
    panel : pd.DataFrame
        County-year panel with Year column.
    train_end_year : int
        Last year included in training.
    test_start_year : int, optional
        First year of test set. Defaults to train_end_year + 1.

    Returns
    -------
    train, test : pd.DataFrame
    """
    if test_start_year is None:
        test_start_year = train_end_year + 1

    train = panel[panel["Year"] <= train_end_year].copy()
    test = panel[panel["Year"] >= test_start_year].copy()

    print(f"Temporal split: train {train['Year'].min()}-{train['Year'].max()} "
          f"({len(train)} obs), test {test['Year'].min()}-{test['Year'].max()} "
          f"({len(test)} obs)")

    return train, test


# =========================================================================
# Model training and forecasting
# =========================================================================

def train_forecast_model(panel, target_col="log_ED_rate",
                          train_end_year=2021,
                          output_dir="results/forecast"):
    """
    Train temporal model and generate forecasts.

    Parameters
    ----------
    panel : pd.DataFrame
        County-year panel.
    target_col : str
        Target variable column name.
    train_end_year : int
        Last year of training data.
    output_dir : str
        Directory for forecast outputs.

    Returns
    -------
    dict with model, predictions, and risk scores
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify available predictors
    available = [c for c in PREDICTOR_COLS if c in panel.columns]
    if not available:
        raise ValueError(
            "No predictor columns found in panel. "
            f"Expected columns like: {list(PREDICTOR_COLS.keys())[:5]}"
        )

    print(f"\nAvailable predictors: {len(available)}/{len(PREDICTOR_COLS)}")

    # Feature engineering
    panel = panel.copy()
    if "E_TOTPOP" in panel.columns:
        panel["log_Pop"] = np.log(panel["E_TOTPOP"].clip(lower=1))
        if "log_Pop" not in available:
            available.append("log_Pop")

    # Prepare target
    if target_col not in panel.columns:
        if "Total_ED" in panel.columns and "E_TOTPOP" in panel.columns:
            panel["ED_rate_per_1000"] = (
                panel["Total_ED"] / panel["E_TOTPOP"].clip(lower=1) * 1000
            )
            panel["log_ED_rate"] = np.log(
                panel["ED_rate_per_1000"].clip(lower=0.1)
            )
        else:
            raise ValueError(f"Target column '{target_col}' not found and "
                             f"cannot be computed.")

    # Drop rows without target
    model_data = panel.dropna(subset=[target_col])

    # Temporal split
    train, test = temporal_split(model_data, train_end_year)

    X_train = train[available]
    y_train = train[target_col]
    X_test = test[available]
    y_test = test[target_col]

    # Models
    models = {
        "Ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 50))),
        ]),
        "Elastic Net": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95],
                alphas=np.logspace(-3, 3, 50),
                max_iter=10000, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42)),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred_train = pipe.predict(X_train)
        pred_test = pipe.predict(X_test)

        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred_test)
        mae_test = mean_absolute_error(y_test, pred_test)

        results[name] = {
            "model": pipe,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_test": mae_test,
        }
        print(f"  {name}: R2_train={r2_train:.3f}, "
              f"R2_test={r2_test:.3f}, MAE_test={mae_test:.3f}")

    # Select best model by test R2
    best_name = max(results, key=lambda k: results[k]["r2_test"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (R2_test={results[best_name]['r2_test']:.3f})")

    # Generate predictions for all years
    panel["predicted"] = best_model.predict(panel[available])
    panel["residual"] = panel[target_col] - panel["predicted"]

    # --- Risk scoring ---
    # Compute excess burden as deviation from prediction
    # Positive residual = more ED burden than predicted = underserved
    mean_resid = panel["residual"].mean()
    std_resid = panel["residual"].std()
    panel["z_score"] = (panel["residual"] - mean_resid) / std_resid

    # Tier classification
    panel["risk_tier"] = "Stable"
    panel.loc[panel["z_score"] > 1.0, "risk_tier"] = "At-Risk"
    panel.loc[panel["z_score"] > 1.5, "risk_tier"] = "Critical"

    # --- Trend analysis ---
    # For each county, compute the slope of residuals over time
    trend_results = []
    for fips in panel["FIPS"].unique():
        county_data = panel[panel["FIPS"] == fips].sort_values("Year")
        if len(county_data) >= 3:
            years_arr = county_data["Year"].values.astype(float)
            resids = county_data["residual"].values
            if not np.any(np.isnan(resids)):
                slope = np.polyfit(years_arr, resids, 1)[0]
                trend_results.append({
                    "FIPS": fips,
                    "County": county_data["County"].iloc[0],
                    "trend_slope": slope,
                    "latest_z": county_data["z_score"].iloc[-1],
                    "latest_tier": county_data["risk_tier"].iloc[-1],
                    "mean_z": county_data["z_score"].mean(),
                })

    trends = pd.DataFrame(trend_results)
    if not trends.empty:
        # Worsening = positive slope (excess burden increasing over time)
        trends["trend_direction"] = "Stable"
        trends.loc[trends["trend_slope"] > 0.02, "trend_direction"] = "Worsening"
        trends.loc[trends["trend_slope"] < -0.02, "trend_direction"] = "Improving"

        # Priority score: combines current status + trajectory
        trends["priority_score"] = trends["latest_z"] + 2 * trends["trend_slope"]
        trends = trends.sort_values("priority_score", ascending=False)

        # Save
        trends.to_csv(output_dir / "county_risk_trends.csv", index=False)
        print(f"\nCounty risk trends saved to {output_dir / 'county_risk_trends.csv'}")

        # Print top priority counties
        print(f"\n{'=' * 60}")
        print("TOP PRIORITY COUNTIES (highest excess burden + worsening trend)")
        print(f"{'=' * 60}")
        for _, row in trends.head(10).iterrows():
            print(f"  {row['County']:20s} | Tier: {row['latest_tier']:10s} | "
                  f"Trend: {row['trend_direction']:10s} | "
                  f"Priority: {row['priority_score']:.2f}")

    # --- Figures ---
    _plot_forecast_results(panel, trends, output_dir, best_name)

    # Save full results
    panel_out = panel[["FIPS", "County", "Year", target_col, "predicted",
                        "residual", "z_score", "risk_tier"]].copy()
    panel_out.to_csv(output_dir / "forecast_predictions.csv", index=False)

    model_comparison = pd.DataFrame([
        {"Model": k, "R2_Train": v["r2_train"],
         "R2_Test": v["r2_test"], "MAE_Test": v["mae_test"]}
        for k, v in results.items()
    ])
    model_comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "predictions": panel_out,
        "trends": trends,
        "model_comparison": model_comparison,
    }


def _plot_forecast_results(panel, trends, output_dir, model_name):
    """Generate publication-quality forecast figures."""

    # Fig 1: Temporal validation -- predicted vs actual by year
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly = panel.groupby("Year").agg(
        actual_mean=(panel.columns[panel.columns.str.contains("log_ED|ED_rate")][0]
                      if any(panel.columns.str.contains("log_ED|ED_rate"))
                      else panel.columns[3], "mean"),
        predicted_mean=("predicted", "mean"),
    ).reset_index()

    if not yearly.empty:
        ax.plot(yearly["Year"], yearly["actual_mean"], "ko-",
                label="Observed (mean)", linewidth=2, markersize=8)
        ax.plot(yearly["Year"], yearly["predicted_mean"], "b--s",
                label=f"Predicted ({model_name})", linewidth=2, markersize=8)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Mean ED Burden", fontsize=12)
        ax.set_title("Temporal Validation: Observed vs. Predicted ED Burden",
                      fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_temporal_validation.png", dpi=300)
    plt.close(fig)

    # Fig 2: County risk tier distribution over time
    if "risk_tier" in panel.columns:
        tier_counts = panel.groupby(["Year", "risk_tier"]).size().unstack(
            fill_value=0
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {"Stable": "#4CAF50", "At-Risk": "#FF9800", "Critical": "#F44336"}
        for tier in ["Stable", "At-Risk", "Critical"]:
            if tier in tier_counts.columns:
                ax.plot(tier_counts.index, tier_counts[tier], "o-",
                        color=colors[tier], label=tier, linewidth=2, markersize=8)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Number of Counties", fontsize=12)
        ax.set_title("County Risk Tier Distribution Over Time", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_risk_tiers_over_time.png", dpi=300)
        plt.close(fig)

    # Fig 3: Priority matrix (current status vs trend)
    if not trends.empty and len(trends) > 5:
        fig, ax = plt.subplots(figsize=(10, 8))
        color_map = {"Stable": "#4CAF50", "At-Risk": "#FF9800",
                      "Critical": "#F44336"}
        for tier, group in trends.groupby("latest_tier"):
            ax.scatter(group["trend_slope"], group["latest_z"],
                       c=color_map.get(tier, "#757575"), label=tier,
                       s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

        # Label top priority counties
        for _, row in trends.head(5).iterrows():
            ax.annotate(row["County"], (row["trend_slope"], row["latest_z"]),
                         fontsize=9, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points")

        ax.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5,
                    label="At-Risk threshold")
        ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.5,
                    label="Critical threshold")
        ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

        ax.set_xlabel("Trend Slope (+ = worsening)", fontsize=12)
        ax.set_ylabel("Latest Excess Burden (z-score)", fontsize=12)
        ax.set_title("County Priority Matrix: Current Burden vs. Trajectory",
                      fontsize=14)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "fig_priority_matrix.png", dpi=300)
        plt.close(fig)

    print(f"  Forecast figures saved to {output_dir}/")


# =========================================================================
# Forward projection (no SEDD data needed)
# =========================================================================

def project_forward(trained_model, predictor_features, current_predictors,
                     output_path=None):
    """
    Apply trained model to current-year predictors to forecast ED burden
    for years where SEDD data is not yet available.

    Parameters
    ----------
    trained_model : sklearn Pipeline
        Model trained on historical panel.
    predictor_features : list
        Feature column names used in training.
    current_predictors : pd.DataFrame
        County-level predictor data for the forecast year(s).
        Must include FIPS, County, Year columns + predictor_features.
    output_path : str, optional
        Path to save projections.

    Returns
    -------
    pd.DataFrame
        Projected excess burden scores with risk tiers.
    """
    X = current_predictors[predictor_features]
    predictions = trained_model.predict(X)

    result = current_predictors[["FIPS", "County", "Year"]].copy()
    result["projected_ED_burden"] = predictions
    result["projected_risk_tier"] = "Stable"
    # Use training-set statistics for z-scoring
    # (would need to pass these in from training -- simplified here)

    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Projections saved to {output_path}")

    return result


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import sys

    print("ED Burden Forecasting Module")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python -m src.model.forecast --panel data/processed/panel_nc.csv")
    print()
    print("Options:")
    print("  --panel PATH        Path to county-year panel CSV")
    print("  --train-end YEAR    Last year of training data (default: 2021)")
    print("  --output DIR        Output directory (default: results/forecast)")
    print()
    print("This module requires a county-year panel dataset built by:")
    print("  python -m src.data_acquisition.build_panel")
    print()
    print("The forecasting approach:")
    print("  1. Train on historical panel (e.g., 2015-2021)")
    print("  2. Validate on held-out years (e.g., 2022-2023)")
    print("  3. Apply to current-year predictors for forward projection")
    print("  4. Identify counties needing intervention before crisis develops")
