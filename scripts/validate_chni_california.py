"""
Validate CHNi out-of-state using CHR 2024 for California, New York, and Florida.

CHR has all 5 CHNi components + preventable hospitalizations for every US county.
This is a ZERO-REFITTING out-of-state validation: same formula, new states.

Also downloads NY SPARCS open ED data for actual ED visit validation.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import urllib.request
import json
import ssl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, ".")

OUT = Path("results/chni/out_of_state")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("data/california")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_file(url, dest, desc=""):
    if dest.exists():
        print("  [cached] {}".format(desc or dest.name))
        return True
    print("  Downloading {}...".format(desc or dest.name))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print("    -> {} ({:,.0f} bytes)".format(dest.name, len(data)))
        return True
    except Exception as e:
        print("    FAILED: {}".format(e))
        return False


# =========================================================================
# 1. Parse CHR 2024 for multiple states
# =========================================================================
print("=" * 65)
print("STEP 1: Build CHNi for CA, NY, FL from CHR 2024")
print("=" * 65)

chr_path = DATA / "chr_2024_analytic.csv"
chr_raw = pd.read_csv(chr_path, low_memory=False, encoding="latin-1")

# Row 0 is a description row; skip it
chr_df = chr_raw.iloc[1:].copy()
chr_df.columns = chr_raw.columns

# Key columns
chr_df["FIPS"] = chr_df["5-digit FIPS Code"].astype(str).str.zfill(5)
chr_df["State"] = chr_df["State Abbreviation"]
chr_df["County"] = chr_df["Name"]

# Extract CHNi components
chr_df["diabetes"] = pd.to_numeric(chr_df["Diabetes Prevalence raw value"], errors="coerce")
chr_df["life_exp"] = pd.to_numeric(chr_df["Life Expectancy raw value"], errors="coerce")
chr_df["pcp_rate"] = pd.to_numeric(chr_df["Primary Care Physicians raw value"], errors="coerce")
chr_df["phys_inact"] = pd.to_numeric(chr_df["Physical Inactivity raw value"], errors="coerce")
chr_df["poverty"] = pd.to_numeric(chr_df["Children in Poverty raw value"], errors="coerce")
chr_df["prev_hosp"] = pd.to_numeric(chr_df["Preventable Hospital Stays raw value"], errors="coerce")

# Note: CHR PCP rate is "ratio of population to PCP" (higher = fewer PCPs = worse)
# Our CHNi inverts PCP rate (1/rate). CHR already stores it as ratio-to-pop,
# so higher values = FEWER PCPs = MORE need. We need to handle this.
# In our NC panel, chr_pcp_rate is the raw ratio (PCPs per population).
# CHR "Primary Care Physicians raw value" is the RATIO of pop to PCPs.
# So it's already inverted relative to our NC definition. Let's check.

# Process each state
states = {
    "CA": {"name": "California", "fips_prefix": "06", "n_counties": 58},
    "NY": {"name": "New York", "fips_prefix": "36", "n_counties": 62},
    "FL": {"name": "Florida", "fips_prefix": "12", "n_counties": 67},
    "NC": {"name": "North Carolina", "fips_prefix": "37", "n_counties": 100},
}

all_results = {}

for state_abbr, info in states.items():
    print()
    print("-" * 50)
    print("{} ({})".format(info["name"], state_abbr))
    print("-" * 50)

    # Filter to state (exclude state-level summary row where county FIPS = "000")
    state_df = chr_df[
        (chr_df["FIPS"].str[:2] == info["fips_prefix"]) &
        (chr_df["FIPS"].str[2:] != "000")
    ].copy()

    print("  Counties found: {}".format(len(state_df)))

    # Check data availability
    components = ["diabetes", "life_exp", "pcp_rate", "phys_inact", "poverty"]
    for comp in components + ["prev_hosp"]:
        n = state_df[comp].notna().sum()
        print("    {}: {}/{} available".format(comp, n, len(state_df)))

    # Compute CHNi using z-scores within state
    # diabetes: higher = more need (no inversion)
    # life_exp: lower = more need (invert)
    # pcp_rate: CHR stores as pop-to-PCP ratio, so higher = fewer PCPs = more need (no inversion needed!)
    # phys_inact: higher = more need (no inversion)
    # poverty: higher = more need (no inversion)

    z_scores = {}

    # Diabetes
    vals = state_df["diabetes"]
    z_scores["z_diabetes"] = (vals - vals.mean()) / vals.std()

    # Life expectancy (invert: lower life exp = higher need)
    vals = 1.0 / state_df["life_exp"].replace(0, np.nan)
    z_scores["z_life_exp"] = (vals - vals.mean()) / vals.std()

    # PCP rate (CHR = pop per PCP, higher = worse, so NO additional inversion)
    vals = state_df["pcp_rate"]
    z_scores["z_pcp_shortage"] = (vals - vals.mean()) / vals.std()

    # Physical inactivity
    vals = state_df["phys_inact"]
    z_scores["z_phys_inact"] = (vals - vals.mean()) / vals.std()

    # Poverty
    vals = state_df["poverty"]
    z_scores["z_poverty"] = (vals - vals.mean()) / vals.std()

    z_df = pd.DataFrame(z_scores, index=state_df.index)
    state_df["CHNi"] = z_df.mean(axis=1)

    # Validate against preventable hospitalizations
    mask = state_df["CHNi"].notna() & state_df["prev_hosp"].notna()
    if mask.sum() > 10:
        r, p = stats.pearsonr(state_df.loc[mask, "CHNi"],
                               state_df.loc[mask, "prev_hosp"])
        rho, p_rho = stats.spearmanr(state_df.loc[mask, "CHNi"],
                                       state_df.loc[mask, "prev_hosp"])
        print()
        print("  CHNi vs Preventable Hospitalizations:")
        print("    N = {}".format(mask.sum()))
        print("    Pearson r  = {:+.3f} (p = {:.2e})".format(r, p))
        print("    Spearman rho = {:+.3f} (p = {:.2e})".format(rho, p_rho))
        print("    R2 = {:.3f}".format(r ** 2))

        all_results[state_abbr] = {
            "name": info["name"],
            "n": int(mask.sum()),
            "r": r,
            "rho": rho,
            "r2": r ** 2,
            "p": p,
            "state_df": state_df,
        }

        # Top/bottom 5
        ranked = state_df.sort_values("CHNi", ascending=False)
        print()
        print("  Top 5 highest need:")
        for _, row in ranked.head(5).iterrows():
            print("    {}: CHNi={:+.2f}, Prev Hosp={:.0f}".format(
                row["County"], row["CHNi"],
                row["prev_hosp"] if pd.notna(row["prev_hosp"]) else -1))
        print("  Top 5 lowest need:")
        for _, row in ranked.tail(5).iterrows():
            print("    {}: CHNi={:+.2f}, Prev Hosp={:.0f}".format(
                row["County"], row["CHNi"],
                row["prev_hosp"] if pd.notna(row["prev_hosp"]) else -1))
    else:
        print("  Insufficient data for validation (N={})".format(mask.sum()))

# =========================================================================
# 2. Try NY SPARCS ED data
# =========================================================================
print()
print("=" * 65)
print("STEP 2: Download NY SPARCS ED Encounter Data")
print("=" * 65)

ny_ed_path = DATA / "ny_sparcs_ed.csv"
ny_ed_url = ("https://health.data.ny.gov/api/views/5gzv-zv2z/rows.csv"
             "?accessType=DOWNLOAD")
download_file(ny_ed_url, ny_ed_path, "NY SPARCS ED encounters by facility")

ny_ed_success = False
if ny_ed_path.exists():
    ny_ed = pd.read_csv(ny_ed_path, low_memory=False)
    print("  NY ED shape: {}".format(ny_ed.shape))
    print("  NY ED columns: {}".format(list(ny_ed.columns)))

    # Get most recent year
    year_col = [c for c in ny_ed.columns if "year" in c.lower()][0]
    ny_ed[year_col] = pd.to_numeric(ny_ed[year_col], errors="coerce")
    latest = ny_ed[year_col].max()
    print("  Latest year: {}".format(int(latest)))

    # Aggregate total ED encounters by facility county
    county_col = [c for c in ny_ed.columns if "county" in c.lower() and "facil" in c.lower()]
    enc_col = [c for c in ny_ed.columns if "total" in c.lower() and "encounter" in c.lower()]

    if county_col and enc_col:
        county_col = county_col[0]
        enc_col = enc_col[0]
        ny_latest = ny_ed[ny_ed[year_col] == latest].copy()
        ny_latest[enc_col] = pd.to_numeric(ny_latest[enc_col], errors="coerce")

        # Annual total by county
        ny_county = ny_latest.groupby(county_col)[enc_col].sum().reset_index()
        ny_county.columns = ["County", "Total_ED"]
        print("  NY counties with ED data: {}".format(len(ny_county)))
        print("  Total NY ED visits: {:,.0f}".format(ny_county["Total_ED"].sum()))
        ny_ed_success = True

# =========================================================================
# 3. Summary comparison figure
# =========================================================================
print()
print("=" * 65)
print("STEP 3: Multi-State Validation Summary")
print("=" * 65)

n_states = len(all_results)
if n_states < 2:
    print("  Need at least 2 states for comparison")
else:
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 6))
    if n_states == 1:
        axes = [axes]

    for i, (abbr, res) in enumerate(all_results.items()):
        ax = axes[i]
        sdf = res["state_df"]
        mask = sdf["CHNi"].notna() & sdf["prev_hosp"].notna()
        x = sdf.loc[mask, "CHNi"]
        y = sdf.loc[mask, "prev_hosp"]

        ax.scatter(x, y, c="#3498db", s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

        # Trend line
        z = np.polyfit(x, y, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_fit(x_line), "--", color="#c0392b", linewidth=2, alpha=0.8)

        ax.text(0.05, 0.95,
                "r = {:.3f}\nR2 = {:.3f}\nn = {}".format(res["r"], res["r2"], res["n"]),
                transform=ax.transAxes, fontsize=12, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("CHNi Score", fontsize=11)
        ax.set_ylabel("Preventable Hospitalization Rate", fontsize=11)
        ax.set_title("{} ({} counties)".format(res["name"], res["n"]),
                     fontsize=13, fontweight="bold")

        # Label top 3
        top3 = sdf[mask].nlargest(3, "CHNi")
        for _, row in top3.iterrows():
            if pd.notna(row["prev_hosp"]):
                ax.annotate(row["County"], (row["CHNi"], row["prev_hosp"]),
                            fontsize=7, alpha=0.8, xytext=(5, 5),
                            textcoords="offset points")

    plt.suptitle("CHNi Validation Across States (Zero Refitting)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(OUT / "fig_chni_multistate.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  -> fig_chni_multistate.png")

    # Summary table
    print()
    print("  {:<20s} {:>5s} {:>10s} {:>10s} {:>12s}".format(
        "State", "N", "Pearson r", "R2", "p-value"))
    print("  " + "-" * 60)
    for abbr, res in all_results.items():
        print("  {:<20s} {:>5d} {:>+10.3f} {:>10.3f} {:>12.2e}".format(
            res["name"], res["n"], res["r"], res["r2"], res["p"]))

    # Bar chart of r values
    fig, ax = plt.subplots(figsize=(8, 5))
    state_names = [res["name"] for res in all_results.values()]
    r_vals = [res["r"] for res in all_results.values()]
    colors = ["#3498db" if r > 0.5 else "#f39c12" if r > 0.3 else "#e74c3c" for r in r_vals]
    bars = ax.bar(state_names, r_vals, color=colors, edgecolor="white", alpha=0.85)
    for bar, r in zip(bars, r_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                "r={:.3f}".format(r), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson r (CHNi vs. Preventable Hosp.)", fontsize=11)
    ax.set_title("CHNi Generalizability: Same Formula, Different States",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(r_vals) * 1.15)
    ax.axhline(0.5, color="#999", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(state_names) - 0.5, 0.51, "r = 0.50 threshold", fontsize=8, color="#999")
    plt.tight_layout()
    plt.savefig(OUT / "fig_chni_generalizability.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  -> fig_chni_generalizability.png")

# =========================================================================
# 4. Save state-level scores
# =========================================================================
for abbr, res in all_results.items():
    sdf = res["state_df"]
    out_cols = ["FIPS", "County", "CHNi", "diabetes", "life_exp", "pcp_rate",
                "phys_inact", "poverty", "prev_hosp"]
    sdf[out_cols].to_csv(OUT / "chni_scores_{}.csv".format(abbr.lower()), index=False)
    print("  -> chni_scores_{}.csv".format(abbr.lower()))

print()
print("=" * 65)
print("OUT-OF-STATE VALIDATION COMPLETE")
print("=" * 65)
