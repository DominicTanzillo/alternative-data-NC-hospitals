"""
Exploratory Data Analysis (EDA) Module
=======================================

Comprehensive EDA on the county-level panel/cross-sectional data.
Produces summary statistics, correlation analysis, missing data patterns,
distribution plots, and bivariate relationship plots.

Usage
-----
    python -m src.model.eda
    python -m src.model.eda --panel data/processed/panel_nc.csv --year 2021
    python -m src.model.eda --data data/final/merged_county_data.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("results/eda")

# Potential target columns (checked in order)
TARGET_CANDIDATES = [
    "Total_ED", "ED_visits", "ed_visits", "SEDD_visits",
    "chr_preventable_hosp", "chr_premature_mort",
]

# Feature grouping patterns for organized analysis
FEATURE_GROUPS = {
    "Demographics (SVI)": lambda c: c.startswith(("EP_", "RPL_", "E_TOTPOP")),
    "Economic": lambda c: c in ("PovertyRate", "MedianHouseholdIncome",
                                 "ChildPovertyRate", "UnemploymentRate"),
    "Chronic Disease (PLACES)": lambda c: c.endswith("_prevalence"),
    "Health Outcomes (CHR)": lambda c: c.startswith("chr_"),
    "Infrastructure (AHRF)": lambda c: c.startswith("AHRF_"),
    "Insurance/Enrollment": lambda c: c.startswith(("MA_", "Medicaid",
                                                     "pub_insured", "log_MA",
                                                     "log_MC")),
}


def classify_feature(col):
    """Assign a feature to a group based on naming patterns."""
    for group, test in FEATURE_GROUPS.items():
        if test(col):
            return group
    return "Other"


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path=None, panel_path=None, year=None):
    """
    Load data for EDA. Supports cross-sectional CSV or a panel year slice.

    Returns (df, target_col) where target_col may be None.
    """
    if panel_path:
        df = pd.read_csv(panel_path, low_memory=False)
        df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
        if year:
            df = df[df["Year"] == year].copy()
            print(f"Panel slice: {len(df)} counties for year {year}")
        else:
            print(f"Full panel: {len(df)} county-year observations")
    elif path:
        df = pd.read_csv(path, low_memory=False)
        print(f"Cross-sectional data: {len(df)} rows x {df.shape[1]} columns")
    else:
        # Try default paths
        for p in ["data/processed/panel_nc.csv",
                   "data/final/merged_county_data.csv"]:
            if Path(p).exists():
                df = pd.read_csv(p, low_memory=False)
                if "FIPS" in df.columns:
                    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
                # Use most recent year if panel
                if "Year" in df.columns:
                    latest = df["Year"].max()
                    df = df[df["Year"] == latest].copy()
                    print(f"Auto-loaded panel, using year {latest}: "
                          f"{len(df)} counties")
                else:
                    print(f"Auto-loaded: {len(df)} rows from {p}")
                break
        else:
            raise FileNotFoundError(
                "No data file found. Use --data or --panel to specify path."
            )

    # Convert to numeric
    id_cols = {"FIPS", "County", "Year", "Region"}
    for col in df.columns:
        if col not in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect target variable
    target_col = None
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns and df[candidate].notna().sum() > 10:
            target_col = candidate
            break

    return df, target_col


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data Summary
# ═══════════════════════════════════════════════════════════════════════════════

def data_summary(df, target_col, out):
    """Generate descriptive statistics and save summary CSV."""
    id_cols = {"FIPS", "County", "Year", "Region"}
    feature_cols = [c for c in df.columns if c not in id_cols]

    rows = []
    for col in feature_cols:
        s = df[col]
        rows.append({
            "Feature": col,
            "Group": classify_feature(col),
            "N_NonNull": s.notna().sum(),
            "Pct_Missing": s.isna().mean() * 100,
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "Q25": s.quantile(0.25) if s.notna().sum() > 0 else np.nan,
            "Median": s.median(),
            "Q75": s.quantile(0.75) if s.notna().sum() > 0 else np.nan,
            "Max": s.max(),
            "Skewness": s.skew() if s.notna().sum() > 2 else np.nan,
        })

    summary = pd.DataFrame(rows).sort_values(["Group", "Feature"])
    summary.to_csv(out / "eda_summary.csv", index=False)

    print(f"\n{'='*65}")
    print("DATA SUMMARY")
    print(f"{'='*65}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features: {len(feature_cols)} "
          f"(target: {target_col or 'not available'})")

    # Group counts
    groups = summary.groupby("Group").size()
    print(f"\nFeatures by group:")
    for g, n in groups.items():
        print(f"  {g}: {n}")

    # Missing data summary
    high_missing = summary[summary["Pct_Missing"] > 50]
    if len(high_missing) > 0:
        print(f"\nFeatures with >50% missing: {len(high_missing)}")
        for _, r in high_missing.head(10).iterrows():
            print(f"  {r['Feature']}: {r['Pct_Missing']:.0f}% missing")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Target Variable Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_target(df, target_col, out):
    """Distribution and outlier analysis of target variable."""
    if target_col is None:
        print("\n[EDA] No target variable available -- skipping target analysis")
        return

    y = df[target_col].dropna()
    print(f"\n{'='*65}")
    print(f"TARGET: {target_col}")
    print(f"{'='*65}")
    print(f"  N = {len(y)}")
    print(f"  Mean = {y.mean():.1f}, Median = {y.median():.1f}")
    print(f"  Std = {y.std():.1f}")
    print(f"  Range = [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Skewness = {y.skew():.2f}")

    # Check for population column to compute rate
    pop_col = None
    for c in ["E_TOTPOP", "AHRF_population", "chr_population"]:
        if c in df.columns and df[c].notna().sum() > 10:
            pop_col = c
            break

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Raw distribution
    axes[0, 0].hist(y, bins=30, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0, 0].set_xlabel(target_col)
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"Distribution of {target_col}")

    # Log transform
    y_log = np.log(y.clip(lower=1))
    axes[0, 1].hist(y_log, bins=30, color="#4CAF50", alpha=0.8,
                     edgecolor="white")
    axes[0, 1].set_xlabel(f"log({target_col})")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Log-Transformed {target_col}")

    # Rate per 1000 (if population available)
    if pop_col:
        rate = y / df.loc[y.index, pop_col].clip(lower=1) * 1000
        axes[1, 0].hist(rate, bins=30, color="#FF9800", alpha=0.8,
                         edgecolor="white")
        axes[1, 0].set_xlabel(f"{target_col} per 1,000 pop")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Rate per 1,000 Population")

        # Flag outliers (>3 SD from mean)
        z = (rate - rate.mean()) / rate.std()
        outliers = df.loc[rate.index[z.abs() > 3]]
        if len(outliers) > 0 and "County" in outliers.columns:
            print(f"\n  Outlier counties (|z| > 3 on rate):")
            for _, row in outliers.iterrows():
                print(f"    {row.get('County', '?')}: "
                      f"rate = {rate.loc[row.name]:.0f}/1000")
    else:
        axes[1, 0].text(0.5, 0.5, "No population column\navailable",
                         ha="center", va="center", fontsize=12)
        axes[1, 0].set_title("Rate per 1,000 Population")

    # QQ plot
    from scipy import stats
    stats.probplot(y_log, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"Q-Q Plot: log({target_col})")

    plt.tight_layout()
    plt.savefig(out / "fig_target_distribution.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"  -> fig_target_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_analysis(df, target_col, out):
    """Correlation heatmap, top correlates, and multicollinearity detection."""
    id_cols = {"FIPS", "County", "Year", "Region"}
    numeric_cols = [c for c in df.columns if c not in id_cols
                    and df[c].notna().sum() > len(df) * 0.3]

    if len(numeric_cols) < 3:
        print("\n[EDA] Too few numeric columns for correlation analysis")
        return

    corr = df[numeric_cols].corr()

    # Save pairwise correlations with target
    if target_col and target_col in numeric_cols:
        target_corr = corr[target_col].drop(target_col).sort_values(
            key=abs, ascending=False
        )
        corr_df = pd.DataFrame({
            "Feature": target_corr.index,
            "Correlation": target_corr.values,
            "Abs_Correlation": target_corr.abs().values,
            "Group": [classify_feature(c) for c in target_corr.index],
        })
        corr_df.to_csv(out / "eda_correlations.csv", index=False)

        print(f"\n{'='*65}")
        print(f"TOP CORRELATES WITH {target_col}")
        print(f"{'='*65}")
        for _, r in corr_df.head(20).iterrows():
            print(f"  {r['Feature']:<45s} r = {r['Correlation']:+.3f}  "
                  f"({r['Group']})")

    # Correlation heatmap (grouped)
    # Select top features by variance to keep heatmap readable
    if len(numeric_cols) > 40:
        # Keep top 40 by variance
        variances = df[numeric_cols].var().sort_values(ascending=False)
        plot_cols = list(variances.head(40).index)
    else:
        plot_cols = numeric_cols

    # Sort by group for visual clustering
    plot_cols_sorted = sorted(plot_cols, key=classify_feature)
    corr_plot = df[plot_cols_sorted].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr_plot.values, cmap="RdBu_r", vmin=-1, vmax=1,
                    aspect="auto")
    ax.set_xticks(range(len(plot_cols_sorted)))
    ax.set_yticks(range(len(plot_cols_sorted)))
    ax.set_xticklabels(plot_cols_sorted, rotation=90, fontsize=6)
    ax.set_yticklabels(plot_cols_sorted, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(out / "fig_correlation_heatmap.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print(f"  -> fig_correlation_heatmap.png")

    # Multicollinearity: flag pairs with |r| > 0.85
    high_corr_pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                high_corr_pairs.append({
                    "Feature_1": cols[i],
                    "Feature_2": cols[j],
                    "Correlation": r,
                })
    if high_corr_pairs:
        hc_df = pd.DataFrame(high_corr_pairs).sort_values(
            "Correlation", key=abs, ascending=False
        )
        print(f"\nHighly correlated pairs (|r| > 0.85): {len(hc_df)}")
        for _, r in hc_df.head(15).iterrows():
            print(f"  {r['Feature_1']:<35s} <-> "
                  f"{r['Feature_2']:<35s}  r={r['Correlation']:+.3f}")

    return corr


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VIF (Variance Inflation Factor)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vif(df, target_col, out, max_features=40):
    """Compute VIF for numeric features to detect multicollinearity."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    id_cols = {"FIPS", "County", "Year", "Region"}
    targets = set(TARGET_CANDIDATES)
    feature_cols = [c for c in df.columns
                    if c not in id_cols and c not in targets
                    and df[c].notna().sum() > len(df) * 0.5]

    if len(feature_cols) < 3:
        print("\n[EDA] Too few features for VIF computation")
        return

    # Limit to manageable number
    if len(feature_cols) > max_features:
        # Prioritize by variance
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = list(variances.head(max_features).index)

    X = df[feature_cols].copy()
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = pd.DataFrame(
        scaler.fit_transform(imp.fit_transform(X)),
        columns=feature_cols
    )

    vif_rows = []
    for col in feature_cols:
        y_vif = X_clean[col]
        X_vif = X_clean.drop(columns=[col])
        lr = LinearRegression()
        lr.fit(X_vif, y_vif)
        r2 = lr.score(X_vif, y_vif)
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_rows.append({
            "Feature": col,
            "Group": classify_feature(col),
            "VIF": vif,
            "R2_vs_others": r2,
        })

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(out / "eda_vif.csv", index=False)

    print(f"\n{'='*65}")
    print("VARIANCE INFLATION FACTORS (top 20)")
    print(f"{'='*65}")
    print(f"{'Feature':<45s} {'VIF':>8s}  {'Group'}")
    print("-" * 75)
    for _, r in vif_df.head(20).iterrows():
        flag = " ***" if r["VIF"] > 10 else ""
        print(f"  {r['Feature']:<43s} {r['VIF']:>8.1f}  {r['Group']}{flag}")

    n_high = (vif_df["VIF"] > 10).sum()
    if n_high:
        print(f"\n  *** {n_high} features with VIF > 10 "
              f"(severe multicollinearity)")

    return vif_df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Feature Distributions
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_distributions(df, out, max_features=36):
    """Histogram grid of all numeric features."""
    id_cols = {"FIPS", "County", "Year", "Region"}
    feature_cols = [c for c in df.columns if c not in id_cols
                    and df[c].notna().sum() > 5]

    if len(feature_cols) > max_features:
        # Keep those with highest variance
        variances = df[feature_cols].var().sort_values(ascending=False)
        feature_cols = list(variances.head(max_features).index)

    n = len(feature_cols)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if n == 1 else axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna()
        color = {"Demographics (SVI)": "#F57C00",
                 "Economic": "#1976D2",
                 "Chronic Disease (PLACES)": "#7B1FA2",
                 "Health Outcomes (CHR)": "#D32F2F",
                 "Infrastructure (AHRF)": "#388E3C",
                 "Insurance/Enrollment": "#00796B",
                 }.get(classify_feature(col), "#757575")
        ax.hist(data, bins=20, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(col, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out / "fig_feature_distributions.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  -> fig_feature_distributions.png ({n} features)")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Missing Data Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def plot_missing_patterns(df, out):
    """Heatmap of missingness by feature x county."""
    id_cols = {"FIPS", "County", "Year", "Region"}
    feature_cols = [c for c in df.columns if c not in id_cols]

    if not feature_cols:
        return

    # Compute missing fraction per feature
    missing_pct = df[feature_cols].isna().mean().sort_values(ascending=False)

    # Only show features with some missingness
    missing_features = missing_pct[missing_pct > 0]
    if len(missing_features) == 0:
        print("  No missing data -- skipping missing pattern plot")
        return

    # Limit to top 50 most-missing features
    plot_features = list(missing_features.head(50).index)

    # Create binary missing matrix
    missing_mat = df[plot_features].isna().astype(int)

    fig, ax = plt.subplots(figsize=(16, max(6, len(plot_features) * 0.25)))
    im = ax.imshow(missing_mat.T.values, cmap="YlOrRd", aspect="auto",
                    interpolation="nearest")
    ax.set_yticks(range(len(plot_features)))
    ax.set_yticklabels(plot_features, fontsize=7)
    ax.set_xlabel("County index")
    ax.set_title("Missing Data Pattern (yellow=present, red=missing)")
    fig.colorbar(im, ax=ax, shrink=0.5, label="Missing (1) / Present (0)")
    plt.tight_layout()
    plt.savefig(out / "fig_missing_patterns.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  -> fig_missing_patterns.png ({len(plot_features)} features)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Bivariate Relationships
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bivariate(df, target_col, out, top_n=10):
    """Scatter plots of top-correlated features vs target."""
    if target_col is None:
        print("\n[EDA] No target variable -- skipping bivariate plots")
        return

    id_cols = {"FIPS", "County", "Year", "Region"}
    feature_cols = [c for c in df.columns if c not in id_cols
                    and c != target_col
                    and df[c].notna().sum() > len(df) * 0.3]

    if not feature_cols:
        return

    # Get top correlates
    corrs = df[feature_cols].corrwith(df[target_col]).abs().sort_values(
        ascending=False
    )
    top_features = list(corrs.head(top_n).index)

    ncols = min(5, top_n)
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        mask = df[[feat, target_col]].notna().all(axis=1)
        ax.scatter(df.loc[mask, feat], df.loc[mask, target_col],
                   alpha=0.5, s=30, c="#2196F3", edgecolors="white",
                   linewidth=0.3)
        r = df[[feat, target_col]].dropna().corr().iloc[0, 1]
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel(target_col, fontsize=8)
        ax.set_title(f"r = {r:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)

    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Top {top_n} Bivariate Relationships with {target_col}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out / "fig_bivariate_top10.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  -> fig_bivariate_top10.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main EDA Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_eda(path=None, panel_path=None, year=None, output_dir=None):
    """Run full EDA pipeline."""
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    df, target_col = load_data(path=path, panel_path=panel_path, year=year)

    # 1. Data summary
    summary = data_summary(df, target_col, out)

    # 2. Target analysis
    analyze_target(df, target_col, out)

    # 3. Correlation analysis
    correlation_analysis(df, target_col, out)

    # 4. VIF
    compute_vif(df, target_col, out)

    # 5. Feature distributions
    print(f"\n{'='*65}")
    print("FEATURE DISTRIBUTIONS")
    print(f"{'='*65}")
    plot_feature_distributions(df, out)

    # 6. Missing data patterns
    print(f"\n{'='*65}")
    print("MISSING DATA PATTERNS")
    print(f"{'='*65}")
    plot_missing_patterns(df, out)

    # 7. Bivariate relationships
    print(f"\n{'='*65}")
    print("BIVARIATE RELATIONSHIPS")
    print(f"{'='*65}")
    plot_bivariate(df, target_col, out)

    print(f"\n{'='*65}")
    print("EDA COMPLETE")
    print(f"{'='*65}")
    print(f"Outputs saved to: {out.resolve()}")
    return {"summary": summary, "target_col": target_col}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    path = None
    panel_path = None
    year = None
    output_dir = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--data" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--panel" and i + 1 < len(sys.argv):
            panel_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--year" and i + 1 < len(sys.argv):
            year = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--help":
            print("EDA Module")
            print("=" * 40)
            print()
            print("Usage:")
            print("  python -m src.model.eda")
            print("  python -m src.model.eda --panel data/processed/panel_nc.csv --year 2021")
            print("  python -m src.model.eda --data data/final/merged_county_data.csv")
            print()
            print("Options:")
            print("  --data PATH      Cross-sectional CSV")
            print("  --panel PATH     Panel CSV")
            print("  --year YYYY      Year to slice from panel")
            print("  --output DIR     Output directory (default: results/eda/)")
            sys.exit(0)
        else:
            i += 1

    run_eda(path=path, panel_path=panel_path, year=year,
            output_dir=output_dir)
