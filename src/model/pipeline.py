"""
Excess Emergency Department Burden Pipeline
=============================================
Identifies underserved counties using alternative (non-traditional) data
sources to predict and map excess ED utilization.

Research Design
---------------
Two complementary models:

Model A — "Expected Utilization Model"
    Target:     log(Total_ED)
    Key feature: log(Population) as population offset
    Insight:     After controlling for population, which alternative data
                 features explain remaining ED volume variation?
    Excess burden = exp(residual) → ratio of observed to expected ED visits

Model B — "Rate-Based Structural Model"
    Target:     ED visits per 1,000 population
    No population feature (already normalized)
    Insight:     Can sociodemographic/infrastructure features predict the
                 ED utilization RATE? Low R² here means local unmeasured
                 factors (catchment areas, primary care access) dominate.

Predictors (nationally available — generalizable to any US state):
    - CMS Medicare/Medicaid enrollment
    - CDC Social Vulnerability Index
    - CDC PLACES (chronic disease prevalence)
    - HRSA AHRF (healthcare infrastructure/workforce)

Key Methodological Notes:
    - Current ED data (NC DETECT/SHEPS) is by HOSPITAL LOCATION, not patient
      residence. Counties with regional medical centers may show rates >1,000
      per 1,000 pop because they serve neighboring counties.
    - HCUP SEDD upgrade: When SEDD data is available (via Duke DUA), use
      src.database_read.load_sedd to get residence-based counts (PSTCO FIPS).
      This resolves the catchment-area problem. NC: 2007-2023, SC: 2006-2023.
    - 18 of 100 NC counties have no ED facility -- their residents appear
      in neighboring county data. These are flagged as "healthcare deserts."
    - All preprocessing is inside sklearn Pipelines (no data leakage).

Usage
-----
    python -m src.model.pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, cross_validate,
    LeaveOneOut,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, RidgeCV, LassoCV, ElasticNetCV,
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 10
N_BOOTSTRAP = 1000
EXCESS_BURDEN_THRESHOLD_Z = 1.0

DATA_PATH = Path("data/final/merged_county_data.csv")
OUTPUT_DIR = Path("results")

# ─── Feature Definitions ─────────────────────────────────────────────────────
# Only nationally-available data sources for generalizability.

FEATURES = {
    "Insurance Coverage (CMS)": {
        "pub_insured_pct": "% publicly insured (MA + MC / pop)",
        "log_MA": "log(Medicare Advantage enrollees)",
        "log_MC": "log(Medicare enrollees)",
    },
    "Social Vulnerability (CDC SVI)": {
        "RPL_THEME1": "SVI: Socioeconomic Status",
        "RPL_THEME2": "SVI: Household Characteristics",
        "RPL_THEME3": "SVI: Racial/Ethnic Minority Status",
        "RPL_THEME4": "SVI: Housing Type / Transportation",
    },
    "Demographics (ACS via SVI)": {
        "EP_POV150": "% below 150% poverty line",
        "EP_UNEMP": "% unemployed",
        "EP_UNINSUR": "% uninsured",
        "EP_AGE65": "% aged 65+",
        "EP_AGE17": "% aged under 17",
        "EP_DISABL": "% with disability",
        "EP_MINRTY": "% racial/ethnic minority",
        "EP_MOBILE": "% living in mobile homes",
        "EP_NOVEH": "% households without vehicle",
        "EP_NOINT": "% without internet",
    },
    "Healthcare Infrastructure (HRSA)": {
        "HPSA_PrimaryCare_2023": "Primary care HPSA designation",
        "HPSA_Dental_2023": "Dental care HPSA designation",
        "HPSA_Mental_2023": "Mental health HPSA designation",
        "MDs_per_10k": "MDs per 10,000 population",
        "PCP_per_10k": "Primary care MDs per 10,000 pop",
    },
    "Chronic Disease (CDC PLACES)": {
        "OBESITY_age-adjusted_prevalence": "Obesity prevalence",
        "DIABETES_age-adjusted_prevalence": "Diabetes prevalence",
        "STROKE_age-adjusted_prevalence": "Stroke prevalence",
        "CHD_age-adjusted_prevalence": "Coronary heart disease prevalence",
        "CASTHMA_age-adjusted_prevalence": "Asthma prevalence",
        "CSMOKING_age-adjusted_prevalence": "Smoking prevalence",
        "GHLTH_age-adjusted_prevalence": "Poor general health prevalence",
        "MHLTH_age-adjusted_prevalence": "Poor mental health prevalence",
        "PHLTH_age-adjusted_prevalence": "Poor physical health prevalence",
        "ACCESS2_age-adjusted_prevalence": "Lack of health insurance prev.",
        "LPA_age-adjusted_prevalence": "Physical inactivity prevalence",
    },
    "Rurality (USDA)": {
        "RUCC_2013": "Rural-Urban Continuum Code",
        "UIC_2013": "Urban Influence Code",
    },
}

# Flatten
ALL_FEATURE_COLS = []
FEATURE_TO_GROUP = {}
FEATURE_TO_LABEL = {}
for group, cols in FEATURES.items():
    for col, label in cols.items():
        ALL_FEATURE_COLS.append(col)
        FEATURE_TO_GROUP[col] = group
        FEATURE_TO_LABEL[col] = label

# Additional feature for Model A (population offset)
FEATURE_TO_GROUP["log_Pop"] = "Population (offset)"
FEATURE_TO_LABEL["log_Pop"] = "log(Total population)"

GROUP_COLORS = {
    "Insurance Coverage (CMS)": "#1976D2",
    "Social Vulnerability (CDC SVI)": "#F57C00",
    "Demographics (ACS via SVI)": "#388E3C",
    "Healthcare Infrastructure (HRSA)": "#D32F2F",
    "Chronic Disease (CDC PLACES)": "#7B1FA2",
    "Rurality (USDA)": "#00796B",
    "Population (offset)": "#455A64",
    "Other": "#757575",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(path=DATA_PATH):
    """
    Load merged county data. Engineer features. Split into modeling
    and healthcare-desert subsets.
    """
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} counties, {df.shape[1]} raw columns")

    for col in df.columns:
        if col not in ["County", "Region"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Feature Engineering ──
    df["log_Pop"] = np.log(df["E_TOTPOP"].clip(lower=1))
    df["log_MA"] = np.log1p(df["AVG_MA_Enrollments_2023_FY"])
    df["log_MC"] = np.log1p(df["AVG_MC_Enrollees"])
    df["pub_insured_pct"] = (
        (df["AVG_MA_Enrollments_2023_FY"] + df["AVG_MC_Enrollees"])
        / df["E_TOTPOP"].clip(lower=1) * 100
    )
    df["MDs_per_10k"] = (
        df["MDs_All_2021"] / df["E_TOTPOP"].clip(lower=1) * 10000
    )
    df["PCP_per_10k"] = (
        df["MDs_PrimaryCare_perCap_2021"]
        / df["E_TOTPOP"].clip(lower=1) * 10000
    )

    # ── Targets ──
    df["ED_rate_per_1000"] = (df["Total_ED"] / df["E_TOTPOP"]) * 1000
    df["log_Total_ED"] = np.log(df["Total_ED"].clip(lower=1))

    # ── Split ──
    df_no_ed = df[df["Total_ED"] == 0].copy()
    df_model = df[df["Total_ED"] > 0].copy()

    print(f"Counties WITH ED facilities: {len(df_model)} (for modeling)")
    print(f"Counties WITHOUT ED facilities: {len(df_no_ed)} "
          f"(healthcare deserts)")

    # ── Feature matrices ──
    available_a = ["log_Pop"] + [c for c in ALL_FEATURE_COLS
                                  if c in df_model.columns]
    available_b = [c for c in ALL_FEATURE_COLS if c in df_model.columns]

    X_a = df_model[available_a].copy()  # Model A: includes log_Pop
    X_b = df_model[available_b].copy()  # Model B: rate features only
    y_a = df_model["log_Total_ED"].copy()
    y_b = df_model["ED_rate_per_1000"].copy()

    print(f"\nModel A features: {X_a.shape[1]} (incl. log_Pop offset)")
    print(f"Model B features: {X_b.shape[1]} (rate-only, no population)")
    print(f"\nTarget A — log(Total_ED): mean={y_a.mean():.2f}, "
          f"std={y_a.std():.2f}")
    print(f"Target B — ED/1000 pop:   mean={y_b.mean():.1f}, "
          f"std={y_b.std():.1f}")

    return df_model, df_no_ed, X_a, y_a, X_b, y_b


# ═══════════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════════════

def get_models():
    """Models from simple to complex, all with imputation inside pipeline."""
    alphas = np.logspace(-3, 3, 50)
    return {
        "OLS": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=alphas)),
        ]),
        "Lasso": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LassoCV(alphas=alphas, max_iter=10000,
                              random_state=RANDOM_STATE)),
        ]),
        "Elastic Net": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95],
                alphas=alphas, max_iter=10000,
                random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
        "SVR (RBF)": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, gamma="scale")),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=3,
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=3,
                random_state=RANDOM_STATE)),
        ]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Validation & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def cross_validate_models(X, y, models, label="", cv_folds=N_CV_FOLDS):
    """K-fold CV with train/test reporting."""
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, pipeline in models.items():
        print(f"  {name}...", end=" ", flush=True)
        scores = cross_validate(
            pipeline, X, y, cv=cv,
            scoring=["r2", "neg_mean_squared_error",
                     "neg_mean_absolute_error"],
            return_train_score=True, n_jobs=-1,
        )
        r2 = scores["test_r2"]
        rmse = np.sqrt(-scores["test_neg_mean_squared_error"])
        mae = -scores["test_neg_mean_absolute_error"]
        train_r2 = scores["train_r2"].mean()

        print(f"R² = {r2.mean():.4f} +/- {r2.std():.4f}")
        results.append({
            "Model": name, "Target": label,
            "CV_R2_mean": r2.mean(), "CV_R2_std": r2.std(),
            "CV_RMSE_mean": rmse.mean(), "CV_RMSE_std": rmse.std(),
            "CV_MAE_mean": mae.mean(), "CV_MAE_std": mae.std(),
            "Train_R2": train_r2,
            "Overfit_Gap": train_r2 - r2.mean(),
        })

    return pd.DataFrame(results).sort_values("CV_R2_mean", ascending=False)


def holdout_evaluation(X_train, X_test, y_train, y_test, models):
    """Train & evaluate on holdout test set."""
    results = []
    fitted = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results.append({
            "Model": name,
            "Test_R2": r2_score(y_test, y_pred),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Test_MAE": mean_absolute_error(y_test, y_pred),
        })
        fitted[name] = pipeline
    return pd.DataFrame(results).sort_values("Test_R2", ascending=False), fitted


def leave_one_out_predictions(X, y, pipeline):
    """LOO CV returning per-sample predictions (unbiased)."""
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    for tr, te in loo.split(X):
        pipeline.fit(X.iloc[tr], y.iloc[tr])
        y_pred[te] = pipeline.predict(X.iloc[te])
    return y_pred


def bootstrap_ci(model, X_test, y_test, n=N_BOOTSTRAP, ci=0.95):
    """Bootstrap confidence intervals."""
    rng = np.random.RandomState(RANDOM_STATE)
    ns = len(y_test)
    r2s, rmses = [], []
    Xa, ya = np.array(X_test), np.array(y_test)
    for _ in range(n):
        idx = rng.choice(ns, size=ns, replace=True)
        pred = model.predict(Xa[idx])
        r2s.append(r2_score(ya[idx], pred))
        rmses.append(np.sqrt(mean_squared_error(ya[idx], pred)))
    a = (1 - ci) / 2
    return {
        "R2_mean": np.mean(r2s),
        "R2_lo": np.percentile(r2s, 100 * a),
        "R2_hi": np.percentile(r2s, 100 * (1 - a)),
        "RMSE_mean": np.mean(rmses),
        "RMSE_lo": np.percentile(rmses, 100 * a),
        "RMSE_hi": np.percentile(rmses, 100 * (1 - a)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Excess Burden Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_excess_burden(df_model, X, y_log, pipeline):
    """
    Compute excess ED burden using Model A residuals.

    Excess = observed_ED / expected_ED  (ratio on original scale)
    Values > 1 mean more ED visits than expected.

    Uses LOO predictions for unbiased per-county estimates.
    """
    print("  Computing Leave-One-Out predictions...")
    y_pred_log = leave_one_out_predictions(X, y_log, pipeline)

    results = df_model[["County", "Region", "E_TOTPOP", "Total_ED"]].copy()
    results["ED_rate_per_1000"] = (
        results["Total_ED"] / results["E_TOTPOP"] * 1000
    )
    results["Observed_log_ED"] = y_log.values
    results["Predicted_log_ED"] = y_pred_log
    results["Residual"] = results["Observed_log_ED"] - results["Predicted_log_ED"]

    # Excess ratio: exp(residual) = observed / expected
    results["Excess_ratio"] = np.exp(results["Residual"])
    results["Expected_ED"] = np.exp(results["Predicted_log_ED"])
    results["Excess_ED_visits"] = results["Total_ED"] - results["Expected_ED"]

    # Z-score classification
    res_std = results["Residual"].std()
    res_mean = results["Residual"].mean()
    results["Excess_z"] = (results["Residual"] - res_mean) / res_std

    results["Classification"] = "Expected"
    results.loc[
        results["Excess_z"] > EXCESS_BURDEN_THRESHOLD_Z,
        "Classification",
    ] = "Excess Burden"
    results.loc[
        results["Excess_z"] < -EXCESS_BURDEN_THRESHOLD_Z,
        "Classification",
    ] = "Below Expected"

    return results.sort_values("Excess_ratio", ascending=False)


def build_underserved_map(excess_df, df_no_ed):
    """Combine excess-burden + healthcare-desert counties."""
    deserts = df_no_ed[["County", "Region", "E_TOTPOP"]].copy()
    deserts["Total_ED"] = 0
    deserts["ED_rate_per_1000"] = 0
    deserts["Classification"] = "No ED Facility"
    deserts["Excess_ratio"] = np.nan
    deserts["Excess_z"] = np.nan

    full = pd.concat([excess_df, deserts], ignore_index=True)
    full = full.sort_values("County")

    print("\nUnderserved Map Summary:")
    for cls, n in full["Classification"].value_counts().items():
        print(f"  {cls}: {n} counties")
    return full


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Importance & Parsimony
# ═══════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(model, X, y, feature_names):
    """Permutation importance."""
    result = permutation_importance(
        model, X, y, n_repeats=30,
        random_state=RANDOM_STATE, scoring="r2", n_jobs=-1,
    )
    return pd.DataFrame({
        "Feature": feature_names,
        "Label": [FEATURE_TO_LABEL.get(f, f) for f in feature_names],
        "Group": [FEATURE_TO_GROUP.get(f, "Other") for f in feature_names],
        "Importance_mean": result.importances_mean,
        "Importance_std": result.importances_std,
    }).sort_values("Importance_mean", ascending=False)


def parsimony_analysis(X, y, label=""):
    """Test fewer features: is a 5-feature model enough?"""
    feature_sets = {
        "3-feature (pop + SVI + uninsured)": [
            "log_Pop", "RPL_THEME1", "EP_UNINSUR",
        ],
        "5-feature (+ PCP + diabetes)": [
            "log_Pop", "RPL_THEME1", "EP_UNINSUR",
            "PCP_per_10k", "DIABETES_age-adjusted_prevalence",
        ],
        "10-feature": [
            "log_Pop", "RPL_THEME1", "RPL_THEME4",
            "EP_POV150", "EP_UNINSUR", "EP_AGE65",
            "PCP_per_10k", "HPSA_PrimaryCare_2023",
            "DIABETES_age-adjusted_prevalence",
            "pub_insured_pct",
        ],
        f"Full ({len(X.columns)} features)": list(X.columns),
    }

    cv = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, cols in feature_sets.items():
        avail = [c for c in cols if c in X.columns]
        if not avail:
            continue
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 50))),
        ])
        scores = cross_val_score(pipe, X[avail], y, cv=cv, scoring="r2",
                                 n_jobs=-1)
        rows.append({
            "Feature_Set": name, "N_Features": len(avail),
            "CV_R2_mean": scores.mean(), "CV_R2_std": scores.std(),
        })
        print(f"  {name}: R² = {scores.mean():.4f} +/- {scores.std():.4f}")

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(cv_a, cv_b, out):
    """Side-by-side model comparison for both targets."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, cv, title in [
        (axes[0], cv_a, "Model A: log(Total ED)"),
        (axes[1], cv_b, "Model B: ED Rate per 1,000"),
    ]:
        data = cv.sort_values("CV_R2_mean")
        colors = ["#2196F3" if r > 0 else "#F44336"
                  for r in data["CV_R2_mean"]]
        bars = ax.barh(data["Model"], data["CV_R2_mean"],
                       xerr=data["CV_R2_std"],
                       color=colors, alpha=0.85, edgecolor="white")
        ax.set_xlabel("R² (10-fold CV)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.axvline(x=0, color="black", linewidth=0.5)
        for bar, val in zip(bars, data["CV_R2_mean"]):
            ax.text(max(val, 0) + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out / "fig_model_comparison.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(y_true, y_pred, model_name, r2, out,
                             xlabel="Observed", ylabel="Predicted",
                             filename="fig_pred_vs_actual.png"):
    """Scatter with 45-degree reference."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, s=60, c="#2196F3",
               edgecolors="white", linewidth=0.5)
    lims = [min(min(y_true), min(y_pred)) * 0.9,
            max(max(y_true), max(y_pred)) * 1.1]
    ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{model_name} — R² = {r2:.4f}", fontsize=13)
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_residuals(y_true, y_pred, model_name, out,
                   filename="fig_residuals.png"):
    """Residual diagnostics."""
    res = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, res, alpha=0.6, s=50, c="#FF5722",
                    edgecolors="white", linewidth=0.5)
    axes[0].axhline(y=0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Residual", fontsize=11)
    axes[0].set_title(f"Residuals vs Predicted ({model_name})", fontsize=12)

    axes[1].hist(res, bins=15, color="#4CAF50", edgecolor="white", alpha=0.85)
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=12)

    plt.tight_layout()
    plt.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importance(imp_df, top_n, out,
                            filename="fig_feature_importance.png"):
    """Top-N features colored by data source."""
    top = imp_df.head(top_n).iloc[::-1]
    colors = [GROUP_COLORS.get(FEATURE_TO_GROUP.get(f, "Other"), "#757575")
              for f in top["Feature"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["Label"], top["Importance_mean"],
            xerr=top["Importance_std"],
            color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Permutation Importance (decrease in R²)", fontsize=11)
    ax.set_title(f"Top {top_n} Predictors of ED Utilization", fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    seen = []
    handles = []
    for f in top["Feature"]:
        g = FEATURE_TO_GROUP.get(f, "Other")
        if g not in seen:
            seen.append(g)
            handles.append(Patch(color=GROUP_COLORS.get(g, "#757575"),
                                 label=g))
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_group_importance(imp_df, out):
    """Aggregate importance by data source."""
    group_imp = (imp_df.groupby("Group")["Importance_mean"]
                 .sum().sort_values())
    colors = [GROUP_COLORS.get(g, "#757575") for g in group_imp.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    group_imp.plot(kind="barh", ax=ax, color=colors, alpha=0.85,
                   edgecolor="white")
    ax.set_xlabel("Cumulative Permutation Importance", fontsize=11)
    ax.set_title("Predictive Value by Alternative Data Source", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "fig_data_source_importance.png", dpi=200,
                bbox_inches="tight")
    plt.close()


def plot_excess_burden(burden_df, out):
    """County-level excess burden bar chart."""
    df = (burden_df.dropna(subset=["Excess_ratio"])
          .sort_values("Excess_ratio"))

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.18)))
    # Color: ratio > 1 = excess (red), < 1 = below expected (blue)
    colors = ["#D32F2F" if x > 1 else "#1976D2"
              for x in df["Excess_ratio"]]

    ax.barh(df["County"], df["Excess_ratio"] - 1, left=1,
            color=colors, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(x=1, color="black", linewidth=0.8)
    ax.set_xlabel("Excess Ratio (1.0 = expected; >1.0 = excess burden)",
                  fontsize=11)
    ax.set_title(
        "Excess ED Burden by County\n"
        "Red = more ED visits than expected | Blue = fewer than expected",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out / "fig_excess_burden.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_parsimony(df, out):
    """Features vs R²."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df["N_Features"], df["CV_R2_mean"], yerr=df["CV_R2_std"],
                fmt="o-", color="#1976D2", linewidth=2, markersize=8,
                capsize=5)
    for _, row in df.iterrows():
        ax.annotate(row["Feature_Set"],
                    (row["N_Features"], row["CV_R2_mean"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("Cross-Validated R²", fontsize=12)
    ax.set_title("Parsimony: Model Performance vs Feature Count", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig_parsimony.png", dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path=None):
    """Execute full dual-model analysis pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Data
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("STEP 1: Data Loading & Feature Engineering")
    print("=" * 65)
    df_model, df_no_ed, X_a, y_a, X_b, y_b = load_and_prepare(
        data_path or DATA_PATH
    )

    regions = df_model["Region"].fillna("Unknown")
    rc = regions.value_counts()
    strat = regions.replace({r: rc.idxmax() for r in rc[rc < 2].index})

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Train/Test Split
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2: Stratified Train/Test Split (80/20)")
    print("=" * 65)
    idx_train, idx_test = train_test_split(
        df_model.index, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=strat,
    )
    Xa_tr, Xa_te = X_a.loc[idx_train], X_a.loc[idx_test]
    ya_tr, ya_te = y_a.loc[idx_train], y_a.loc[idx_test]
    Xb_tr, Xb_te = X_b.loc[idx_train], X_b.loc[idx_test]
    yb_tr, yb_te = y_b.loc[idx_train], y_b.loc[idx_test]
    print(f"Train: {len(idx_train)} counties  |  Test: {len(idx_test)}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Cross-Validation — Both Models
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3a: Cross-Validation — Model A: log(Total ED)")
    print("=" * 65)
    cv_a = cross_validate_models(Xa_tr, ya_tr, get_models(), "log(Total_ED)")

    print("\n" + "=" * 65)
    print("STEP 3b: Cross-Validation — Model B: ED Rate per 1,000")
    print("=" * 65)
    cv_b = cross_validate_models(Xb_tr, yb_tr, get_models(), "ED_rate")

    all_cv = pd.concat([cv_a, cv_b], ignore_index=True)
    all_cv.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

    print("\n-- Model A (log ED) Top 3 --")
    print(cv_a.head(3)[["Model", "CV_R2_mean", "CV_R2_std",
                         "Train_R2", "Overfit_Gap"]].to_string(index=False))
    print("\n-- Model B (rate) Top 3 --")
    print(cv_b.head(3)[["Model", "CV_R2_mean", "CV_R2_std",
                         "Train_R2", "Overfit_Gap"]].to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Holdout — Model A (primary model for excess burden)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4: Holdout Evaluation — Model A")
    print("=" * 65)
    holdout_a, fitted_a = holdout_evaluation(
        Xa_tr, Xa_te, ya_tr, ya_te, get_models()
    )
    print(holdout_a.to_string(index=False))
    holdout_a.to_csv(OUTPUT_DIR / "holdout_model_a.csv", index=False)

    best_name = holdout_a.iloc[0]["Model"]
    best_model = fitted_a[best_name]
    best_r2 = holdout_a.iloc[0]["Test_R2"]
    y_pred_a = best_model.predict(Xa_te)
    print(f"\nBest Model A: {best_name} (Test R² = {best_r2:.4f})")

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Bootstrap CI
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5: Bootstrap 95% Confidence Intervals")
    print("=" * 65)
    ci = bootstrap_ci(best_model, Xa_te, ya_te)
    print(f"R²:   {ci['R2_mean']:.4f} [{ci['R2_lo']:.4f}, {ci['R2_hi']:.4f}]")
    print(f"RMSE: {ci['RMSE_mean']:.4f} [{ci['RMSE_lo']:.4f}, "
          f"{ci['RMSE_hi']:.4f}]")
    pd.DataFrame([{**ci, "Model": best_name}]).to_csv(
        OUTPUT_DIR / "bootstrap_ci.csv", index=False
    )

    # ──────────────────────────────────────────────────────────────────────
    # Step 6: Excess Burden
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 6: Excess ED Burden (LOO predictions)")
    print("=" * 65)
    fresh_pipe = get_models()[best_name]
    excess_df = compute_excess_burden(df_model, X_a, y_a, fresh_pipe)
    underserved = build_underserved_map(excess_df, df_no_ed)
    excess_df.to_csv(OUTPUT_DIR / "excess_burden_scores.csv", index=False)
    underserved.to_csv(OUTPUT_DIR / "underserved_map.csv", index=False)

    print("\nTop 10 Excess Burden Counties:")
    cols = ["County", "Region", "Total_ED", "ED_rate_per_1000",
            "Excess_ratio", "Classification"]
    print(excess_df.head(10)[[c for c in cols if c in excess_df.columns]]
          .to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────
    # Step 7: Feature Importance
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 7: Feature Importance (Model A)")
    print("=" * 65)
    full_model = get_models()[best_name]
    full_model.fit(X_a, y_a)
    imp = compute_feature_importance(full_model, X_a, y_a, list(X_a.columns))
    print(imp.head(15).to_string(index=False))
    imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────
    # Step 8: Parsimony
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 8: Parsimony Analysis — How Few Features Suffice?")
    print("=" * 65)
    pars = parsimony_analysis(X_a, y_a)
    pars.to_csv(OUTPUT_DIR / "parsimony.csv", index=False)

    # ──────────────────────────────────────────────────────────────────────
    # Step 9: Figures
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 9: Publication Figures")
    print("=" * 65)

    plot_model_comparison(cv_a, cv_b, OUTPUT_DIR)
    print("  -> fig_model_comparison.png")

    plot_predicted_vs_actual(
        np.array(ya_te), y_pred_a, best_name, best_r2, OUTPUT_DIR,
        xlabel="Observed log(Total ED)", ylabel="Predicted log(Total ED)",
        filename="fig_pred_vs_actual_model_a.png",
    )
    print("  -> fig_pred_vs_actual_model_a.png")

    plot_residuals(np.array(ya_te), y_pred_a, best_name, OUTPUT_DIR,
                   filename="fig_residuals_model_a.png")
    print("  -> fig_residuals_model_a.png")

    plot_feature_importance(imp, 20, OUTPUT_DIR)
    print("  -> fig_feature_importance.png")

    plot_group_importance(imp, OUTPUT_DIR)
    print("  -> fig_data_source_importance.png")

    plot_excess_burden(excess_df, OUTPUT_DIR)
    print("  -> fig_excess_burden.png")

    plot_parsimony(pars, OUTPUT_DIR)
    print("  -> fig_parsimony.png")

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────
    n_excess = (excess_df["Classification"] == "Excess Burden").sum()
    n_desert = len(df_no_ed)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
    print(f"\nBest model (A): {best_name}")
    print(f"  Test R² = {best_r2:.4f}  "
          f"[{ci['R2_lo']:.4f}, {ci['R2_hi']:.4f}] 95% CI")
    print(f"\nUnderserved counties:")
    print(f"  {n_excess} with excess ED burden (z > {EXCESS_BURDEN_THRESHOLD_Z})")
    print(f"  {n_desert} healthcare deserts (no ED facility)")
    print(f"  {n_excess + n_desert} / {len(df_model) + n_desert} total")

    return {
        "cv_model_a": cv_a, "cv_model_b": cv_b,
        "holdout_a": holdout_a,
        "ci": ci, "feature_importance": imp,
        "excess_burden": excess_df, "underserved_map": underserved,
        "parsimony": pars, "fitted": fitted_a,
    }


def load_panel_year(panel_path, year):
    """
    Load a single year slice from a panel CSV and prepare it for the
    cross-sectional pipeline.

    Maps panel column names to the feature names expected by the pipeline.
    Returns a DataFrame that can replace the standard merged_county_data.csv.
    """
    df = pd.read_csv(panel_path, low_memory=False)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df = df[df["Year"] == year].copy()

    if len(df) == 0:
        raise ValueError(f"No data for year {year} in {panel_path}")

    print(f"Panel slice: {len(df)} counties for year {year}")

    # Convert all non-ID columns to numeric
    for col in df.columns:
        if col not in ("FIPS", "County", "Year", "Region"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Year Panel Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def select_panel_features(df, year_range, min_coverage=0.80):
    """
    Automatically select features that have sufficient coverage across the
    requested year range.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel with FIPS, County, Year, and feature columns.
    year_range : tuple
        (start_year, end_year) inclusive.
    min_coverage : float
        Minimum fraction of non-null values required for a feature to be
        included (across all county-years in the range). Default 0.80.

    Returns
    -------
    tuple
        (selected_features, coverage_report) where selected_features is a
        list of column names and coverage_report is a DataFrame.
    """
    scaffold = {"FIPS", "County", "Year", "Region"}
    candidates = [c for c in df.columns if c not in scaffold]

    report_rows = []
    selected = []

    for col in candidates:
        n_total = len(df)
        n_avail = df[col].notna().sum()
        coverage = n_avail / n_total if n_total > 0 else 0
        years_with = sorted(df.loc[df[col].notna(), "Year"].unique())

        report_rows.append({
            "Feature": col,
            "Coverage": coverage,
            "N_Available": n_avail,
            "N_Total": n_total,
            "Years": f"{min(years_with)}-{max(years_with)}" if years_with
                     else "none",
            "N_Years": len(years_with),
            "Selected": coverage >= min_coverage,
        })

        if coverage >= min_coverage:
            selected.append(col)

    report = pd.DataFrame(report_rows).sort_values("Coverage", ascending=False)
    return selected, report


def analyze_year_ranges(panel_path):
    """
    Analyze all possible year ranges and report feature x observation
    tradeoffs to help choose the optimal window for modeling.

    Prints a summary table and returns the analysis DataFrame.
    """
    df = pd.read_csv(panel_path, low_memory=False)
    for col in df.columns:
        if col not in ("FIPS", "County", "Year", "Region"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    scaffold = {"FIPS", "County", "Year", "Region"}
    all_years = sorted(df["Year"].unique())
    candidates = [c for c in df.columns if c not in scaffold]

    print("=" * 70)
    print("YEAR RANGE ANALYSIS: Feature Coverage vs Temporal Depth")
    print("=" * 70)
    print()

    rows = []
    for start in all_years:
        for end in all_years:
            if end < start:
                continue
            mask = (df["Year"] >= start) & (df["Year"] <= end)
            subset = df[mask]
            n_years = end - start + 1
            n_obs = len(subset)

            # Count features with >= 80% coverage in this window
            n_feat_80 = 0
            n_feat_100 = 0
            for col in candidates:
                cov = subset[col].notna().mean()
                if cov >= 0.80:
                    n_feat_80 += 1
                if cov >= 0.99:
                    n_feat_100 += 1

            rows.append({
                "Start": start, "End": end,
                "N_Years": n_years, "N_Obs": n_obs,
                "Features_80pct": n_feat_80,
                "Features_100pct": n_feat_100,
            })

    analysis = pd.DataFrame(rows)

    # Print key configurations
    print(f"{'Range':<12} {'Years':>5} {'Obs':>5} "
          f"{'Feat(>=80%)':>10} {'Feat(100%)':>10}")
    print("-" * 50)
    key_ranges = [
        (2015, 2023), (2017, 2023), (2018, 2023), (2019, 2023),
        (2020, 2023), (2018, 2021), (2019, 2022),
    ]
    for s, e in key_ranges:
        row = analysis[(analysis["Start"] == s) & (analysis["End"] == e)]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {s}-{e}    {r['N_Years']:>5} {r['N_Obs']:>5} "
                  f"{r['Features_80pct']:>10} {r['Features_100pct']:>10}")

    # Identify the "sweet spot" -- maximize n_years * n_features
    analysis["Score"] = analysis["N_Years"] * analysis["Features_80pct"]
    best = analysis.loc[analysis["Score"].idxmax()]
    print(f"\nRecommended range (max years x features): "
          f"{int(best['Start'])}-{int(best['End'])} "
          f"({int(best['N_Years'])} years, "
          f"{int(best['Features_80pct'])} features, "
          f"{int(best['N_Obs'])} obs)")

    # Also recommend for different priorities
    deep = analysis[analysis["N_Years"] == analysis["N_Years"].max()]
    deep_best = deep.loc[deep["Features_80pct"].idxmax()]
    print(f"  Max temporal depth: "
          f"{int(deep_best['Start'])}-{int(deep_best['End'])} "
          f"({int(deep_best['N_Years'])} years, "
          f"{int(deep_best['Features_80pct'])} features)")

    rich = analysis.loc[analysis["Features_80pct"].idxmax()]
    print(f"  Max feature richness: "
          f"{int(rich['Start'])}-{int(rich['End'])} "
          f"({int(rich['N_Years'])} years, "
          f"{int(rich['Features_80pct'])} features)")

    return analysis


def run_panel_pipeline(panel_path, start_year=None, end_year=None,
                       min_coverage=0.80, target_col=None,
                       output_dir=None):
    """
    Run the modeling pipeline on a multi-year panel.

    Automatically selects features based on coverage in the specified year
    range. Supports three modes:

    1. Pooled cross-section: All years pooled together with year fixed effects
    2. Per-year loop: Run the cross-sectional pipeline separately for each year
    3. Coverage report only: When no target variable is available (awaiting SEDD)

    Parameters
    ----------
    panel_path : str
        Path to panel CSV (e.g., data/processed/panel_nc.csv).
    start_year, end_year : int, optional
        Year range to use. If None, auto-detected from data.
    min_coverage : float
        Minimum feature coverage to include (default 0.80).
    target_col : str, optional
        Name of the target column (ED visits). If None and no ED column
        found, runs in coverage-report-only mode.
    output_dir : str, optional
        Output directory (default: results/panel/).
    """
    out = Path(output_dir or "results/panel")
    out.mkdir(parents=True, exist_ok=True)

    # Load panel
    df = pd.read_csv(panel_path, low_memory=False)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    for col in df.columns:
        if col not in ("FIPS", "County", "Year", "Region"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_years = sorted(df["Year"].unique())
    if start_year is None:
        start_year = min(all_years)
    if end_year is None:
        end_year = max(all_years)

    mask = (df["Year"] >= start_year) & (df["Year"] <= end_year)
    df = df[mask].copy()
    n_counties = df["FIPS"].nunique()
    n_years = df["Year"].nunique()

    print("=" * 70)
    print(f"PANEL PIPELINE: {n_counties} counties x {n_years} years "
          f"({start_year}-{end_year})")
    print(f"Observations: {len(df)}  |  "
          f"Min coverage threshold: {min_coverage:.0%}")
    print("=" * 70)

    # Feature selection
    selected, report = select_panel_features(df, (start_year, end_year),
                                              min_coverage)
    report.to_csv(out / "feature_coverage.csv", index=False)

    n_selected = len(selected)
    n_total = len(report)
    print(f"\nFeature selection: {n_selected} / {n_total} features "
          f"meet {min_coverage:.0%} coverage")

    # Categorize selected features
    categories = {
        "Demographic/Economic": [f for f in selected
                                  if any(f.startswith(p) for p in
                                         ["Poverty", "Median", "Child",
                                          "Unemployment", "EP_", "RPL_",
                                          "E_TOTPOP"])],
        "Health (PLACES)": [f for f in selected
                            if any(f.endswith(s) for s in
                                   ["_prevalence"])],
        "Health (CHR)": [f for f in selected if f.startswith("chr_")],
        "Insurance/Enrollment": [f for f in selected
                                  if any(f.startswith(p) for p in
                                         ["MA_", "Medicaid"])],
    }
    for cat, feats in categories.items():
        if feats:
            print(f"  {cat}: {len(feats)}")
            for f in feats[:5]:
                cov = report.loc[report["Feature"] == f, "Coverage"].values
                print(f"    {f} ({cov[0]:.0%})" if len(cov) > 0 else
                      f"    {f}")
            if len(feats) > 5:
                print(f"    ... and {len(feats) - 5} more")

    # Check for target variable
    target_found = False
    if target_col and target_col in df.columns:
        target_found = True
    else:
        # Look for common ED visit column names
        for candidate in ["Total_ED", "ED_visits", "ed_visits",
                          "SEDD_visits", "sedd_total"]:
            if candidate in df.columns and df[candidate].notna().any():
                target_col = candidate
                target_found = True
                break

    if not target_found:
        print(f"\n{'=' * 70}")
        print("TARGET VARIABLE NOT FOUND")
        print("=" * 70)
        print("No ED visit column detected in the panel. The panel has "
              "all predictor")
        print("variables ready. Once HCUP SEDD data is available:")
        print(f"  1. Place SEDD files in data/raw/sedd/")
        print(f"  2. Rebuild: python -m src.data_acquisition.build_panel")
        print(f"  3. Re-run:  python -m src.model.pipeline --panel "
              f"{panel_path} --year-range {start_year}-{end_year}")
        print(f"\nFeature coverage report saved to: "
              f"{out / 'feature_coverage.csv'}")

        # Still produce the year-range analysis
        print()
        analyze_year_ranges(panel_path)
        return {"features": selected, "coverage": report}

    # ── With target variable: run full modeling ──

    # Feature engineering
    if "E_TOTPOP" in df.columns:
        df["log_Pop"] = np.log(df["E_TOTPOP"].clip(lower=1))
    if "MA_Enrollment_Annual" in df.columns and "E_TOTPOP" in df.columns:
        df["log_MA"] = np.log1p(df["MA_Enrollment_Annual"])
    if "Medicaid_Enrollment" in df.columns and "E_TOTPOP" in df.columns:
        df["log_Medicaid"] = np.log1p(df["Medicaid_Enrollment"])
        df["pub_insured_pct"] = (
            (df.get("MA_Enrollment_Annual", 0) +
             df["Medicaid_Enrollment"])
            / df["E_TOTPOP"].clip(lower=1) * 100
        )

    df["log_ED"] = np.log(df[target_col].clip(lower=1))
    df["ED_rate_per_1000"] = df[target_col] / df["E_TOTPOP"].clip(lower=1) * 1000

    # Build feature matrix
    engineered = ["log_Pop", "log_MA", "log_Medicaid", "pub_insured_pct"]
    feature_cols = selected + [f for f in engineered if f in df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))  # dedupe, preserve order

    # Split: counties with ED > 0 vs healthcare deserts
    df_model = df[df[target_col] > 0].copy()
    df_no_ed = df[df[target_col] == 0].copy()

    avail_features = [c for c in feature_cols if c in df_model.columns]
    X = df_model[avail_features]
    y_log = df_model["log_ED"]
    y_rate = df_model["ED_rate_per_1000"]

    print(f"\nModeling: {len(df_model)} obs with ED > 0, "
          f"{len(df_no_ed)} healthcare deserts")
    print(f"Features: {len(avail_features)}  |  Target: {target_col}")

    # Run cross-validation
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION: Model A — log(ED visits)")
    print("=" * 70)
    cv_a = cross_validate_models(X, y_log, get_models(), "log(ED)")
    cv_a.to_csv(out / "cv_results_panel.csv", index=False)

    print(f"\nTop 3 models:")
    print(cv_a.head(3)[["Model", "CV_R2_mean", "CV_R2_std",
                         "Train_R2", "Overfit_Gap"]].to_string(index=False))

    # Feature importance with best model
    best_name = cv_a.iloc[0]["Model"]
    best_pipe = get_models()[best_name]
    best_pipe.fit(X, y_log)
    imp = compute_feature_importance(best_pipe, X, y_log, avail_features)
    imp.to_csv(out / "feature_importance_panel.csv", index=False)

    print(f"\n{'=' * 70}")
    print(f"TOP PREDICTORS ({best_name})")
    print("=" * 70)
    print(imp.head(15).to_string(index=False))

    # Per-year cross-validation (how stable is performance over time?)
    print(f"\n{'=' * 70}")
    print("PER-YEAR MODEL STABILITY")
    print("=" * 70)
    year_results = []
    for yr in sorted(df_model["Year"].unique()):
        yr_mask = df_model["Year"] == yr
        X_yr = X[yr_mask]
        y_yr = y_log[yr_mask]
        if len(X_yr) < 20:
            continue
        cv = KFold(n_splits=min(5, len(X_yr) // 5), shuffle=True,
                   random_state=RANDOM_STATE)
        pipe = get_models()[best_name]
        scores = cross_val_score(pipe, X_yr, y_yr, cv=cv, scoring="r2")
        year_results.append({
            "Year": yr, "N_Counties": len(X_yr),
            "CV_R2_mean": scores.mean(), "CV_R2_std": scores.std(),
        })
        print(f"  {yr}: n={len(X_yr)}, "
              f"R² = {scores.mean():.3f} +/- {scores.std():.3f}")

    if year_results:
        pd.DataFrame(year_results).to_csv(
            out / "per_year_stability.csv", index=False)

    print(f"\n{'=' * 70}")
    print("PANEL PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {out.resolve()}")

    return {
        "cv": cv_a, "feature_importance": imp,
        "features": avail_features, "coverage": report,
        "per_year": year_results,
    }


if __name__ == "__main__":
    import sys

    panel_path = None
    year = None
    year_range = None
    min_cov = 0.80
    analyze_only = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--panel" and i + 1 < len(sys.argv):
            panel_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--year" and i + 1 < len(sys.argv):
            year = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--year-range" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split("-")
            year_range = (int(parts[0]), int(parts[1]))
            i += 2
        elif sys.argv[i] == "--min-coverage" and i + 1 < len(sys.argv):
            min_cov = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--analyze":
            analyze_only = True
            i += 1
        elif sys.argv[i] == "--help":
            print("Excess ED Burden Pipeline")
            print("=" * 50)
            print()
            print("Usage:")
            print("  python -m src.model.pipeline"
                  "                    # cross-section (default)")
            print("  python -m src.model.pipeline --panel PATH "
                  "--year YYYY    # single year from panel")
            print("  python -m src.model.pipeline --panel PATH "
                  "--year-range 2018-2023")
            print("  python -m src.model.pipeline --panel PATH "
                  "--analyze      # coverage analysis only")
            print()
            print("Options:")
            print("  --panel PATH        Panel CSV path")
            print("  --year YYYY         Single year (cross-section)")
            print("  --year-range S-E    Year range (multi-year pipeline)")
            print("  --min-coverage F    Min feature coverage, 0-1 "
                  "(default: 0.80)")
            print("  --analyze           Print year-range analysis only")
            sys.exit(0)
        else:
            i += 1

    if panel_path and analyze_only:
        analyze_year_ranges(panel_path)
    elif panel_path and year_range:
        run_panel_pipeline(panel_path,
                           start_year=year_range[0],
                           end_year=year_range[1],
                           min_coverage=min_cov)
    elif panel_path and year:
        print(f"Running pipeline on panel year {year} from {panel_path}")
        panel_df = load_panel_year(panel_path, year)
        tmp_path = Path(f"data/processed/panel_year_{year}.csv")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        panel_df.to_csv(tmp_path, index=False)
        run_pipeline(data_path=tmp_path)
    elif panel_path:
        # Default: run with full year range
        run_panel_pipeline(panel_path, min_coverage=min_cov)
    else:
        run_pipeline()
