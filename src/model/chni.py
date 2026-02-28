"""
Community Health Need Index (CHNi)
===================================

A simple, zero-parameter composite score for ranking county-level healthcare
need using 5 publicly available variables. Designed to be:

    1. Generalizable — uses only nationally-available data (CHR, PLACES, Census)
    2. Teachable    — z-score average, no regression coefficients to fit
    3. Transparent   — each component has clear epidemiological meaning
    4. Validated     — correlates r ≈ 0.58 with preventable hospitalization rate

Components
----------
    1. Diabetes prevalence      (CDC PLACES)  — chronic disease burden
    2. 1 / Life expectancy      (CHR)         — overall population health
    3. 1 / PCP rate             (CHR)         — healthcare access (shortage)
    4. Poverty rate             (Census SAIPE) — economic deprivation
    5. Physical inactivity      (CHR)         — modifiable behavioral risk

Scoring: Each component is z-standardized (mean=0, sd=1) and averaged.
Higher CHNi = greater healthcare need.

Usage
-----
    python -m src.model.chni
    python -m src.model.chni --panel data/processed/panel_nc.csv --year 2021
    python -m src.model.chni --data data/final/merged_county_data.csv

References
----------
    - County Health Rankings: countyhealthrankings.org
    - CDC PLACES: cdc.gov/places
    - Census SAIPE: census.gov/programs-surveys/saipe.html
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# CHNi Definition
# ═══════════════════════════════════════════════════════════════════════════════

# Each component: (column_name, invert, label, source)
#   invert=True means higher raw value → LOWER need, so we flip it (1/x)
CHNI_COMPONENTS = [
    {
        "column": "DIABETES_age-adjusted_prevalence",
        "invert": False,
        "label": "Diabetes prevalence",
        "source": "CDC PLACES",
        "rationale": "Marker of chronic disease burden; strongly associated "
                     "with ED utilization and preventable hospitalizations",
    },
    {
        "column": "chr_life_expectancy",
        "invert": True,
        "label": "Life expectancy (inverted)",
        "source": "County Health Rankings",
        "rationale": "Summary measure of population health; low life expectancy "
                     "indicates systemic health disadvantage.",
    },
    {
        "column": "chr_pcp_rate",
        "invert": True,
        "label": "PCP rate (inverted, shortage)",
        "source": "County Health Rankings",
        "rationale": "Primary care physician supply; shortage forces reliance "
                     "on emergency departments for routine care.",
    },
    {
        "column": "PovertyRate",
        "invert": False,
        "label": "Poverty rate",
        "source": "Census SAIPE",
        "rationale": "Economic deprivation limits access to preventive care, "
                     "healthy food, and stable housing.",
    },
    {
        "column": "chr_phys_inactivity",
        "invert": False,
        "label": "Physical inactivity",
        "source": "County Health Rankings",
        "rationale": "Modifiable behavioral risk factor upstream of diabetes, "
                     "cardiovascular disease, and obesity.",
    },
]

# Validation target (if available)
VALIDATION_TARGET = "chr_preventable_hosp"

OUTPUT_DIR = Path("results/chni")


# ═══════════════════════════════════════════════════════════════════════════════
# Core Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_chni(df, components=None):
    """
    Compute the Community Health Need Index for each row.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in CHNI_COMPONENTS.
    components : list of dict, optional
        Override default components. Each dict needs 'column' and 'invert'.

    Returns
    -------
    pd.Series
        CHNi score for each row. Higher = greater need.
    """
    if components is None:
        components = CHNI_COMPONENTS

    z_scores = []
    for comp in components:
        col = comp["column"]
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

        vals = pd.to_numeric(df[col], errors="coerce")

        # Invert if higher raw value means lower need
        if comp.get("invert", False):
            vals = 1.0 / vals.replace(0, np.nan)

        # Z-standardize
        mean = vals.mean()
        std = vals.std()
        if std == 0 or np.isnan(std):
            z = pd.Series(0.0, index=df.index)
        else:
            z = (vals - mean) / std

        z_scores.append(z)

    # Equal-weighted average of z-scores
    z_matrix = pd.concat(z_scores, axis=1)
    chni = z_matrix.mean(axis=1)

    return chni


def score_counties(df, components=None):
    """
    Score and rank counties. Returns DataFrame with CHNi and component z-scores.

    Parameters
    ----------
    df : pd.DataFrame
        County-level data with required columns.
    components : list of dict, optional
        Override default components.

    Returns
    -------
    pd.DataFrame
        Ranked counties with CHNi score, component z-scores, and raw values.
    """
    if components is None:
        components = CHNI_COMPONENTS

    result = pd.DataFrame()

    # Carry over identifiers and validation target
    carry_cols = ["FIPS", "County", "Year"]
    if VALIDATION_TARGET in df.columns:
        carry_cols.append(VALIDATION_TARGET)
    for id_col in carry_cols:
        if id_col in df.columns:
            result[id_col] = df[id_col].values

    # Compute component z-scores
    for comp in components:
        col = comp["column"]
        vals = pd.to_numeric(df[col], errors="coerce")

        # Raw value
        result[f"raw_{comp['label'].split(' (')[0].replace(' ', '_').lower()}"] = vals.values

        # Transform if inverted
        if comp.get("invert", False):
            vals = 1.0 / vals.replace(0, np.nan)

        mean = vals.mean()
        std = vals.std()
        if std > 0 and not np.isnan(std):
            z = (vals - mean) / std
        else:
            z = pd.Series(0.0, index=df.index)

        label_short = comp["label"].split(" (")[0].replace(" ", "_").lower()
        result[f"z_{label_short}"] = z.values

    # Composite
    z_cols = [c for c in result.columns if c.startswith("z_")]
    result["CHNi"] = result[z_cols].mean(axis=1)
    result["CHNi_rank"] = result["CHNi"].rank(ascending=False, method="min").astype(int)

    # Classify into tiers
    result["Tier"] = pd.cut(
        result["CHNi"],
        bins=[-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf],
        labels=["Very Low Need", "Low Need", "Moderate",
                "High Need", "Very High Need"],
    )

    return result.sort_values("CHNi", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_chni(scores, target_col=VALIDATION_TARGET):
    """
    Validate CHNi against preventable hospitalization rate (or other target).

    Returns dict with correlation, R², and significance test.
    """
    if target_col not in scores.columns:
        print(f"  Validation target '{target_col}' not available -- skipping")
        return None

    target = pd.to_numeric(scores[target_col], errors="coerce")
    chni = scores["CHNi"]

    mask = target.notna() & chni.notna()
    if mask.sum() < 10:
        print(f"  Too few valid observations ({mask.sum()}) for validation")
        return None

    from scipy import stats

    r, p = stats.pearsonr(chni[mask], target[mask])
    rho, p_rho = stats.spearmanr(chni[mask], target[mask])

    results = {
        "n": int(mask.sum()),
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "r_squared": r ** 2,
    }

    print(f"\n  Validation against: {target_col}")
    print(f"  N = {results['n']}")
    print(f"  Pearson r  = {r:+.3f}  (p = {p:.2e})")
    print(f"  Spearman ρ = {rho:+.3f}  (p = {p_rho:.2e})")
    print(f"  R²         = {r**2:.3f}")
    print(f"  Interpretation: CHNi explains {r**2*100:.1f}% of variance in "
          f"preventable hospitalizations with ZERO fitted parameters.")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_chni_ranking(scores, out, top_n=15):
    """Horizontal bar chart of highest and lowest need counties."""
    if "County" not in scores.columns:
        print("  No 'County' column -- skipping ranking plot")
        return

    sorted_scores = scores.sort_values("CHNi", ascending=False)
    high = sorted_scores.head(top_n)
    low = sorted_scores.tail(top_n).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Highest need
    colors_high = ["#c0392b" if v > 1.0 else "#e74c3c" if v > 0.5
                    else "#f39c12" for v in high["CHNi"]]
    ax1.barh(range(top_n), high["CHNi"].values, color=colors_high,
             edgecolor="white")
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(high["County"].values, fontsize=9)
    ax1.set_xlabel("CHNi Score", fontsize=11)
    ax1.set_title(f"Top {top_n} Highest Need", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()

    # Lowest need
    colors_low = ["#27ae60" if v < -1.0 else "#2ecc71" if v < -0.5
                   else "#f1c40f" for v in low["CHNi"]]
    ax2.barh(range(top_n), low["CHNi"].values, color=colors_low,
             edgecolor="white")
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(low["County"].values, fontsize=9)
    ax2.set_xlabel("CHNi Score", fontsize=11)
    ax2.set_title(f"Top {top_n} Lowest Need", fontsize=13, fontweight="bold")
    ax2.invert_yaxis()

    plt.suptitle("Community Health Need Index (CHNi) — County Rankings",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "fig_chni_ranking.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> fig_chni_ranking.png")


def plot_chni_validation(scores, out, target_col=VALIDATION_TARGET):
    """Scatter plot of CHNi vs validation target."""
    if target_col not in scores.columns:
        return

    target = pd.to_numeric(scores[target_col], errors="coerce")
    chni = scores["CHNi"]
    mask = target.notna() & chni.notna()

    if mask.sum() < 10:
        print(f"  Too few valid pairs ({mask.sum()}) for validation plot")
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    x = chni[mask].values
    y = target[mask].values
    tiers = scores.loc[mask, "Tier"].values

    # Color by tier
    tier_colors = {
        "Very High Need": "#c0392b",
        "High Need": "#e74c3c",
        "Moderate": "#f39c12",
        "Low Need": "#2ecc71",
        "Very Low Need": "#27ae60",
    }

    for tier, color in tier_colors.items():
        tier_idx = tiers == tier
        if tier_idx.any():
            ax.scatter(x[tier_idx], y[tier_idx],
                       c=color, label=tier, s=50, alpha=0.7,
                       edgecolors="white", linewidth=0.5)

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), "--", color="#555", linewidth=1.5, alpha=0.8)

    from scipy import stats
    r, pval = stats.pearsonr(x, y)
    ax.text(0.05, 0.95,
            f"r = {r:.3f}\nR2 = {r**2:.3f}\np < {pval:.1e}\nn = {len(x)}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Label outlier counties
    if "County" in scores.columns:
        top3 = scores[mask].nlargest(3, "CHNi")
        for _, row in top3.iterrows():
            ax.annotate(row["County"],
                        (row["CHNi"], pd.to_numeric(row[target_col], errors="coerce")),
                        fontsize=8, alpha=0.8,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Community Health Need Index (CHNi)", fontsize=12)
    ax.set_ylabel("Preventable Hospitalization Rate (CHR)", fontsize=12)
    ax.set_title("CHNi Validation: Zero-Parameter Score vs. Outcome",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out / "fig_chni_validation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> fig_chni_validation.png")


def plot_chni_components(scores, out):
    """Heatmap showing z-score contribution of each component per county."""
    z_cols = [c for c in scores.columns if c.startswith("z_")]
    if not z_cols or "County" not in scores.columns:
        return

    # Sort by CHNi
    sorted_scores = scores.sort_values("CHNi", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_scores) * 0.15)))

    data = sorted_scores[z_cols].values
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-2.5, vmax=2.5)

    ax.set_yticks(range(len(sorted_scores)))
    ax.set_yticklabels(sorted_scores["County"].values, fontsize=6)
    ax.set_xticks(range(len(z_cols)))
    labels = [c.replace("z_", "").replace("_", " ").title() for c in z_cols]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.5, label="Z-score (red = high need)")
    ax.set_title("CHNi Component Breakdown by County",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out / "fig_chni_components.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> fig_chni_components.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_chni(path=None, panel_path=None, year=None, output_dir=None):
    """Run CHNi scoring pipeline."""
    out = Path(output_dir or OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("COMMUNITY HEALTH NEED INDEX (CHNi)")
    print("=" * 65)

    # ── Load data ──
    if panel_path:
        df = pd.read_csv(panel_path, low_memory=False)
        df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
        if year:
            df = df[df["Year"] == year].copy()
            print(f"Panel slice: {len(df)} counties for year {year}")
        else:
            latest = df["Year"].max()
            df = df[df["Year"] == latest].copy()
            print(f"Using most recent year ({latest}): {len(df)} counties")
    elif path:
        df = pd.read_csv(path, low_memory=False)
        if "FIPS" in df.columns:
            df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
        print(f"Cross-sectional: {len(df)} rows")
    else:
        for p in ["data/processed/panel_nc.csv",
                   "data/final/merged_county_data.csv"]:
            if Path(p).exists():
                df = pd.read_csv(p, low_memory=False)
                if "FIPS" in df.columns:
                    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
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
                "No data found. Use --data or --panel to specify path."
            )

    # Convert to numeric
    for col in df.columns:
        if col not in {"FIPS", "County", "Year", "Region"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Check required columns ──
    missing = []
    for comp in CHNI_COMPONENTS:
        if comp["column"] not in df.columns:
            missing.append(comp["column"])
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        print("Available columns with partial matches:")
        for m in missing:
            key = m.split("_")[0].lower()
            matches = [c for c in df.columns if key in c.lower()]
            print(f"  '{m}' -> {matches[:5]}")
        return None

    # ── Score counties ──
    print(f"\nScoring {len(df)} counties on 5 components:")
    for comp in CHNI_COMPONENTS:
        direction = "(inverted)" if comp["invert"] else ""
        print(f"  {comp['label']:<35s}  [{comp['source']}] {direction}")

    scores = score_counties(df)

    # ── Display results ──
    print(f"\n{'='*65}")
    print("CHNi RESULTS")
    print(f"{'='*65}")
    print(f"\nScore distribution:")
    print(f"  Mean  = {scores['CHNi'].mean():.3f}")
    print(f"  Std   = {scores['CHNi'].std():.3f}")
    print(f"  Range = [{scores['CHNi'].min():.2f}, {scores['CHNi'].max():.2f}]")

    # Tier counts
    print(f"\nTier distribution:")
    tier_counts = scores["Tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        print(f"  {tier:<20s}: {count:>3d} counties")

    # Top and bottom counties
    if "County" in scores.columns:
        print(f"\nTop 10 HIGHEST need counties:")
        print(f"  {'Rank':<5s} {'County':<20s} {'CHNi':>8s}  {'Tier'}")
        print(f"  {'-'*55}")
        for _, row in scores.head(10).iterrows():
            print(f"  {row['CHNi_rank']:<5d} {row['County']:<20s} "
                  f"{row['CHNi']:>+8.3f}  {row['Tier']}")

        print(f"\nTop 10 LOWEST need counties:")
        print(f"  {'Rank':<5s} {'County':<20s} {'CHNi':>8s}  {'Tier'}")
        print(f"  {'-'*55}")
        for _, row in scores.tail(10).iloc[::-1].iterrows():
            print(f"  {row['CHNi_rank']:<5d} {row['County']:<20s} "
                  f"{row['CHNi']:>+8.3f}  {row['Tier']}")

    # ── Validate ──
    print(f"\n{'='*65}")
    print("VALIDATION")
    print(f"{'='*65}")
    validation = validate_chni(scores)

    # ── Save outputs ──
    scores.to_csv(out / "chni_scores.csv", index=False)
    print(f"\n  -> chni_scores.csv ({len(scores)} counties)")

    # Component definitions
    comp_df = pd.DataFrame(CHNI_COMPONENTS)
    comp_df.to_csv(out / "chni_components.csv", index=False)
    print(f"  -> chni_components.csv")

    # ── Plots ──
    print(f"\n{'='*65}")
    print("FIGURES")
    print(f"{'='*65}")
    plot_chni_ranking(scores, out)
    plot_chni_validation(scores, out)
    plot_chni_components(scores, out)

    print(f"\n{'='*65}")
    print("CHNi COMPLETE")
    print(f"{'='*65}")
    print(f"Outputs saved to: {out.resolve()}")

    return {"scores": scores, "validation": validation}


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
            print("Community Health Need Index (CHNi)")
            print("=" * 40)
            print()
            print("A zero-parameter composite score ranking counties by")
            print("healthcare need using 5 epidemiologically-grounded variables.")
            print()
            print("Usage:")
            print("  python -m src.model.chni")
            print("  python -m src.model.chni --panel data/processed/panel_nc.csv --year 2021")
            print("  python -m src.model.chni --data data/final/merged_county_data.csv")
            print()
            print("Components:")
            for comp in CHNI_COMPONENTS:
                inv = " (inverted)" if comp["invert"] else ""
                print(f"  - {comp['label']}{inv}: {comp['column']}")
            print()
            print("Options:")
            print("  --data PATH      Cross-sectional CSV")
            print("  --panel PATH     Panel CSV")
            print("  --year YYYY      Year to slice from panel")
            print("  --output DIR     Output directory (default: results/chni/)")
            sys.exit(0)
        else:
            i += 1

    run_chni(path=path, panel_path=panel_path, year=year,
             output_dir=output_dir)
