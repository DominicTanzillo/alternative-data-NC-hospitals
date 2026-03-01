"""Validate CHNi against REAL NC ED visit data (2021 SHEPS) and compute excess burden."""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, ".")
from src.model.chni import compute_chni

OUT = Path("results/chni")
OUT.mkdir(parents=True, exist_ok=True)

# Load the merged county data with REAL ED visits
df = pd.read_csv("data/final/merged_county_data.csv", low_memory=False)
for col in df.columns:
    if col not in {"FIPS", "County", "Region"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# The merged data predates CHR expansion -- merge in CHR variables from panel
panel = pd.read_csv("data/processed/panel_nc.csv", low_memory=False)
panel["FIPS"] = panel["FIPS"].astype(str).str.zfill(5)
chr_cols_needed = ["chr_life_expectancy", "chr_pcp_rate", "chr_phys_inactivity",
                   "chr_preventable_hosp"]
panel_2021 = panel[panel["Year"] == 2021][["FIPS"] + chr_cols_needed].copy()
for col in chr_cols_needed:
    panel_2021[col] = pd.to_numeric(panel_2021[col], errors="coerce")

# Merged data has County but no FIPS; panel has both. Join via County.
panel_2021_by_county = panel[panel["Year"] == 2021][["County"] + chr_cols_needed].copy()
for col in chr_cols_needed:
    panel_2021_by_county[col] = pd.to_numeric(panel_2021_by_county[col], errors="coerce")

for col in chr_cols_needed:
    if col not in df.columns:
        mapping = panel_2021_by_county.set_index("County")[col]
        df[col] = df["County"].map(mapping)
        n = df[col].notna().sum()
        print("  Merged {}: {}/100 values".format(col, n))

# Also bring in PovertyRate from panel if missing
if "PovertyRate" not in df.columns:
    pov_map = panel[panel["Year"] == 2021].set_index("County")["PovertyRate"]
    df["PovertyRate"] = df["County"].map(pov_map)
    print("  Merged PovertyRate: {}/100".format(df["PovertyRate"].notna().sum()))

print("=" * 65)
print("CHNi vs. REAL ED VISITS (NC 2021 SHEPS Data)")
print("=" * 65)
print("Counties: {}".format(len(df)))
print("Total ED visits across NC: {:,.0f}".format(df["Total_ED"].sum()))
print()

# Compute CHNi
df["CHNi"] = compute_chni(df)

# Compute ED rate per 1000 population
df["ED_rate_per_1k"] = df["Total_ED"] / df["E_TOTPOP"] * 1000

# Counties with no ED facility
no_ed = df[df["Total_ED"] == 0]
has_ed = df[df["Total_ED"] > 0]
print("Counties WITH ED facility: {}".format(len(has_ed)))
print("Counties WITHOUT ED facility: {} (healthcare deserts)".format(len(no_ed)))
print("  -> {}".format(", ".join(no_ed["County"].tolist())))
print()

# Validation 1: CHNi vs Total ED visits (all counties)
mask = df["CHNi"].notna() & df["Total_ED"].notna()
r_total, p_total = stats.pearsonr(df.loc[mask, "CHNi"], df.loc[mask, "Total_ED"])
print("CHNi vs Total_ED (raw counts, all 100 counties):")
print("  r = {:+.3f}, p = {:.2e}".format(r_total, p_total))
print("  (Low r expected -- raw counts dominated by population size)")
print()

# Validation 2: CHNi vs log(Total_ED)
df_ed = df[df["Total_ED"] > 0].copy()
r_log, p_log = stats.pearsonr(df_ed["CHNi"], np.log(df_ed["Total_ED"]))
print("CHNi vs log(Total_ED) ({} counties with EDs):".format(len(df_ed)))
print("  r = {:+.3f}, p = {:.2e}".format(r_log, p_log))
print()

# Validation 3: CHNi vs ED rate per 1000
rate_mask = df_ed["ED_rate_per_1k"].notna() & df_ed["CHNi"].notna()
r_rate, p_rate = stats.pearsonr(df_ed.loc[rate_mask, "CHNi"],
                                 df_ed.loc[rate_mask, "ED_rate_per_1k"])
print("CHNi vs ED_rate_per_1k ({} counties with EDs):".format(rate_mask.sum()))
print("  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(r_rate, r_rate**2, p_rate))
print()

# Validation 4: vs preventable hosp (comparison)
prev_mask = df["CHNi"].notna() & df["chr_preventable_hosp"].notna()
r_prev, p_prev = stats.pearsonr(df.loc[prev_mask, "CHNi"],
                                  df.loc[prev_mask, "chr_preventable_hosp"])
print("CHNi vs chr_preventable_hosp (100 counties, comparison):")
print("  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(r_prev, r_prev**2, p_prev))
print()

# Catchment area analysis
print("=" * 65)
print("CATCHMENT AREA ANALYSIS")
print("=" * 65)
print()
catchment = df_ed[df_ed["ED_rate_per_1k"] > 500].sort_values("ED_rate_per_1k", ascending=False)
print("Catchment-area counties (ED rate > 500/1k pop):")
for _, r in catchment.iterrows():
    print("  {}: {:.0f}/1k pop, {:,.0f} visits, pop {:,.0f}".format(
        r["County"], r["ED_rate_per_1k"], r["Total_ED"], r["E_TOTPOP"]))

non_catchment = df_ed[df_ed["ED_rate_per_1k"] <= 500].copy()
r_clean, p_clean = stats.pearsonr(non_catchment["CHNi"],
                                    non_catchment["ED_rate_per_1k"])
print()
print("CHNi vs ED_rate (excluding catchment outliers, {} counties):".format(len(non_catchment)))
print("  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(r_clean, r_clean**2, p_clean))

# Excess ED Burden
print()
print("=" * 65)
print("EXCESS ED BURDEN (CHNi-BASED)")
print("=" * 65)
print()

X = non_catchment[["CHNi"]].values
y = non_catchment["ED_rate_per_1k"].values
lr = LinearRegression().fit(X, y)
non_catchment["expected_ED_rate"] = lr.predict(X)
non_catchment["excess_ratio"] = non_catchment["ED_rate_per_1k"] / non_catchment["expected_ED_rate"]
non_catchment["excess_pct"] = (non_catchment["excess_ratio"] - 1) * 100

print("Counties with MOST EXCESS ED utilization (actual >> expected from need):")
over = non_catchment.nlargest(10, "excess_ratio")
for _, r in over.iterrows():
    print("  {:<20s} actual={:.0f}/1k expected={:.0f}/1k excess={:+.0f}%  CHNi={:+.2f}".format(
        r["County"], r["ED_rate_per_1k"], r["expected_ED_rate"], r["excess_pct"], r["CHNi"]))

print()
print("Counties with LEAST ED utilization relative to need:")
under = non_catchment.nsmallest(10, "excess_ratio")
for _, r in under.iterrows():
    print("  {:<20s} actual={:.0f}/1k expected={:.0f}/1k excess={:+.0f}%  CHNi={:+.2f}".format(
        r["County"], r["ED_rate_per_1k"], r["expected_ED_rate"], r["excess_pct"], r["CHNi"]))

# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: All counties with EDs
ax = axes[0]
ax.scatter(df_ed["CHNi"], df_ed["ED_rate_per_1k"], c="#3498db", s=40, alpha=0.7,
           edgecolors="white", linewidth=0.5)
for _, r in catchment.iterrows():
    ax.annotate(r["County"], (r["CHNi"], r["ED_rate_per_1k"]),
                fontsize=7, alpha=0.8, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("ED Visits per 1,000 Pop", fontsize=11)
ax.set_title("All Counties with EDs (n={})\nr={:.3f}".format(len(df_ed), r_rate),
             fontsize=11, fontweight="bold")

# Panel 2: Non-catchment only
ax = axes[1]
colors = ["#c0392b" if x > 1 else "#e74c3c" if x > 0.5 else "#f39c12" if x > -0.3
          else "#2ecc71" if x > -1 else "#27ae60" for x in non_catchment["CHNi"]]
ax.scatter(non_catchment["CHNi"], non_catchment["ED_rate_per_1k"],
           c=colors, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
x_line = np.linspace(non_catchment["CHNi"].min(), non_catchment["CHNi"].max(), 100)
ax.plot(x_line, lr.predict(x_line.reshape(-1, 1)), "--", color="#555",
        linewidth=1.5, alpha=0.8)
ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}\np = {:.1e}".format(r_clean, r_clean**2, p_clean),
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("ED Visits per 1,000 Pop", fontsize=11)
ax.set_title("Excluding Catchment Outliers (n={})\nr={:.3f}".format(len(non_catchment), r_clean),
             fontsize=11, fontweight="bold")

# Panel 3: Excess burden
ax = axes[2]
sorted_excess = non_catchment.sort_values("excess_pct")
colors_bar = ["#c0392b" if x > 50 else "#e74c3c" if x > 20 else "#f39c12" if x > -20
              else "#2ecc71" if x > -50 else "#27ae60" for x in sorted_excess["excess_pct"]]
ax.barh(range(len(sorted_excess)), sorted_excess["excess_pct"].values,
        color=colors_bar, height=0.8, edgecolor="none")
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks([])
ax.set_xlabel("Excess ED Utilization (%)", fontsize=11)
ax.set_title("Excess ED Burden\n(Actual vs CHNi-Expected)", fontsize=11, fontweight="bold")
for i, (_, r) in enumerate(sorted_excess.head(3).iterrows()):
    ax.text(r["excess_pct"] - 2, i, r["County"], fontsize=7, va="center", ha="right",
            fontweight="bold")
for i_off, (_, r) in enumerate(sorted_excess.tail(3).iterrows()):
    idx = len(sorted_excess) - 3 + i_off
    ax.text(r["excess_pct"] + 2, idx, r["County"], fontsize=7, va="center", ha="left",
            fontweight="bold")

plt.suptitle("CHNi vs. REAL Emergency Department Visits (NC 2021)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig_chni_vs_real_ed.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print()
print("-> fig_chni_vs_real_ed.png")

# Save excess burden table
excess_out = non_catchment[["County", "CHNi", "Total_ED", "E_TOTPOP",
                             "ED_rate_per_1k", "expected_ED_rate",
                             "excess_ratio", "excess_pct"]].copy()
excess_out = excess_out.sort_values("excess_ratio", ascending=False)
excess_out.to_csv(OUT / "chni_excess_ed_burden.csv", index=False)
print("-> chni_excess_ed_burden.csv")
