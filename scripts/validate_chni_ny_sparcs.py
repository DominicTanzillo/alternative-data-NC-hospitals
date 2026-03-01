"""Validate CHNi against NY SPARCS data: PPV rates (residence-based) + inpatient admissions.

PPV = Potentially Preventable Visit rates by PATIENT COUNTY (residence-based).
Also validates against inpatient emergency admissions by hospital county.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/chni/out_of_state")
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================================
# 1. Build CHNi for NY from CHR 2024
# =========================================================================
print("=" * 65)
print("STEP 1: Compute CHNi for New York counties")
print("=" * 65)

chr_raw = pd.read_csv("data/california/chr_2024_analytic.csv",
                       low_memory=False, encoding="latin-1")
chr_df = chr_raw.iloc[1:].copy()
chr_df.columns = chr_raw.columns

chr_df["FIPS"] = chr_df["5-digit FIPS Code"].astype(str).str.zfill(5)

# NY = FIPS prefix 36, exclude state summary (FIPS xxx000)
ny_chr = chr_df[
    (chr_df["FIPS"].str[:2] == "36") &
    (chr_df["FIPS"].str[2:] != "000")
].copy().reset_index(drop=True)

ny_chr["County"] = ny_chr["Name"].str.replace(" County", "").str.strip()
ny_chr = ny_chr[ny_chr["County"] != "New York"].copy()
# Fix name mismatch: CHR has "St. Lawrence", PPV has "St Lawrence"
ny_chr["County"] = ny_chr["County"].str.replace("St.", "St", regex=False)

# Extract CHNi components
ny_chr["diabetes"] = pd.to_numeric(ny_chr["Diabetes Prevalence raw value"], errors="coerce")
ny_chr["life_exp"] = pd.to_numeric(ny_chr["Life Expectancy raw value"], errors="coerce")
ny_chr["pcp_rate"] = pd.to_numeric(ny_chr["Primary Care Physicians raw value"], errors="coerce")
ny_chr["phys_inact"] = pd.to_numeric(ny_chr["Physical Inactivity raw value"], errors="coerce")
ny_chr["poverty"] = pd.to_numeric(ny_chr["Children in Poverty raw value"], errors="coerce")
ny_chr["prev_hosp"] = pd.to_numeric(ny_chr["Preventable Hospital Stays raw value"], errors="coerce")
ny_chr["population"] = pd.to_numeric(ny_chr["Population raw value"], errors="coerce")

# Compute CHNi z-scores within NY
z = {}
vals = ny_chr["diabetes"]; z["z_diab"] = (vals - vals.mean()) / vals.std()
vals = 1.0 / ny_chr["life_exp"].replace(0, np.nan); z["z_life"] = (vals - vals.mean()) / vals.std()
vals = ny_chr["pcp_rate"]; z["z_pcp"] = (vals - vals.mean()) / vals.std()
vals = ny_chr["phys_inact"]; z["z_inact"] = (vals - vals.mean()) / vals.std()
vals = ny_chr["poverty"]; z["z_pov"] = (vals - vals.mean()) / vals.std()

z_df = pd.DataFrame(z)
ny_chr["CHNi"] = z_df.mean(axis=1)

print("NY counties: {}".format(len(ny_chr)))
print("CHNi range: [{:.2f}, {:.2f}]".format(ny_chr["CHNi"].min(), ny_chr["CHNi"].max()))
print()

# =========================================================================
# 2. Validate against PPV rates (RESIDENCE-BASED)
# =========================================================================
print("=" * 65)
print("STEP 2: CHNi vs PPV Rates (Residence-Based, 2011-2023)")
print("=" * 65)

ppv = pd.read_csv("data/new_york/ny_ppv_by_patient_county.csv")
ppv["discharge_year"] = pd.to_numeric(ppv["discharge_year"], errors="coerce")
for col in ["observed_rate_per_100", "expected_rate_per_100",
            "adjusted_rate_per_100", "difference_in_rates"]:
    ppv[col] = pd.to_numeric(ppv[col], errors="coerce")

# County name mapping: PPV uses bare county names, some may need fixing
# NYC boroughs in PPV: Bronx, Kings, New York, Queens, Richmond
# CHR uses "New York" for Manhattan, "Kings" for Brooklyn, etc.

# Check overlap
ppv_counties = set(ppv["patient_county_name"].unique())
chr_counties = set(ny_chr["County"].unique())
overlap = ppv_counties & chr_counties
only_ppv = ppv_counties - chr_counties
only_chr = chr_counties - ppv_counties
print("PPV counties: {}".format(len(ppv_counties)))
print("CHR counties: {}".format(len(chr_counties)))
print("Overlap: {}".format(len(overlap)))
if only_ppv:
    print("Only in PPV: {}".format(sorted(only_ppv)))
if only_chr:
    print("Only in CHR: {}".format(sorted(only_chr)))
print()

# Temporal validation: CHNi vs adjusted PPV rate for each year
print("Temporal validation (CHNi vs Adjusted PPV Rate):")
print("-" * 55)
years = []
r_ppv_values = []

for yr in range(2011, 2024):
    ppv_yr = ppv[ppv["discharge_year"] == yr].copy()
    ppv_yr = ppv_yr.rename(columns={"patient_county_name": "County"})
    merged = ny_chr.merge(ppv_yr[["County", "adjusted_rate_per_100", "observed_rate_per_100"]],
                          on="County", how="inner")
    mask = merged["CHNi"].notna() & merged["adjusted_rate_per_100"].notna()
    if mask.sum() > 10:
        r, p = stats.pearsonr(merged.loc[mask, "CHNi"],
                               merged.loc[mask, "adjusted_rate_per_100"])
        years.append(yr)
        r_ppv_values.append(r)
        print("  {}: r = {:+.3f}, R2 = {:.3f}, n = {} (p = {:.2e})".format(
            yr, r, r**2, mask.sum(), p))

print()
if r_ppv_values:
    print("Mean r across years: {:.3f}".format(np.mean(r_ppv_values)))
    print("Std r: {:.3f}".format(np.std(r_ppv_values)))
    print()

# Detailed 2023 analysis
ppv_2023 = ppv[ppv["discharge_year"] == 2023].copy()
ppv_2023 = ppv_2023.rename(columns={"patient_county_name": "County"})
merged_2023 = ny_chr.merge(ppv_2023, on="County", how="inner")
mask_2023 = merged_2023["CHNi"].notna() & merged_2023["adjusted_rate_per_100"].notna()

r_adj, p_adj = stats.pearsonr(merged_2023.loc[mask_2023, "CHNi"],
                                merged_2023.loc[mask_2023, "adjusted_rate_per_100"])
r_obs, p_obs = stats.pearsonr(merged_2023.loc[mask_2023, "CHNi"],
                                merged_2023.loc[mask_2023, "observed_rate_per_100"])
rho, p_rho = stats.spearmanr(merged_2023.loc[mask_2023, "CHNi"],
                               merged_2023.loc[mask_2023, "adjusted_rate_per_100"])

print("2023 Detailed Results:")
print("  CHNi vs Adjusted PPV Rate:  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(
    r_adj, r_adj**2, p_adj))
print("  CHNi vs Observed PPV Rate:  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(
    r_obs, r_obs**2, p_obs))
print("  Spearman rho (adjusted):    {:+.3f}, p = {:.2e}".format(rho, p_rho))
print("  *** PPV = Potentially Preventable Visit rate (RESIDENCE-BASED) ***")
print()

# Sanity: CHNi vs preventable hospitalizations (CHR)
r_prev, p_prev = stats.pearsonr(ny_chr.loc[ny_chr["CHNi"].notna() & ny_chr["prev_hosp"].notna(), "CHNi"],
                                  ny_chr.loc[ny_chr["CHNi"].notna() & ny_chr["prev_hosp"].notna(), "prev_hosp"])
print("Sanity: CHNi vs CHR Preventable Hosp: r = {:+.3f}".format(r_prev))
print()

# Top/bottom counties
print("Top 10 highest PPV rate (2023):")
for _, r in merged_2023.sort_values("adjusted_rate_per_100", ascending=False).head(10).iterrows():
    print("  {:<20s} PPV={:.1f}/100  CHNi={:+.2f}  prev_hosp={:.0f}".format(
        r["County"], r["adjusted_rate_per_100"], r["CHNi"],
        r["prev_hosp"] if pd.notna(r["prev_hosp"]) else -1))

print()
print("Top 10 lowest PPV rate (2023):")
for _, r in merged_2023.sort_values("adjusted_rate_per_100").head(10).iterrows():
    print("  {:<20s} PPV={:.1f}/100  CHNi={:+.2f}  prev_hosp={:.0f}".format(
        r["County"], r["adjusted_rate_per_100"], r["CHNi"],
        r["prev_hosp"] if pd.notna(r["prev_hosp"]) else -1))

# =========================================================================
# 3. Validate against inpatient emergency admissions (facility-based)
# =========================================================================
print()
print("=" * 65)
print("STEP 3: CHNi vs Inpatient Emergency Admissions (Facility-Based)")
print("=" * 65)

admissions = pd.read_csv("data/new_york/ny_sparcs_admission_type_by_county.csv")
admissions["n"] = pd.to_numeric(admissions["n"], errors="coerce")

# Emergency admissions by county per year
print()
print("Temporal validation (CHNi vs Emergency Admission Rate per 1k):")
print("-" * 55)
years_adm = []
r_adm_values = []

for yr in range(2015, 2024):
    yr_data = admissions[admissions["year"] == yr]
    emerg = yr_data[yr_data["type_of_admission"] == "Emergency"].groupby(
        "hospital_county")["n"].sum().reset_index()
    emerg.columns = ["County", "Emergency_Admissions"]

    merged_adm = ny_chr.merge(emerg, on="County", how="inner")
    merged_adm["emerg_rate"] = merged_adm["Emergency_Admissions"] / merged_adm["population"] * 1000
    mask_a = merged_adm["CHNi"].notna() & merged_adm["emerg_rate"].notna()

    if mask_a.sum() > 10:
        r, p = stats.pearsonr(merged_adm.loc[mask_a, "CHNi"],
                               merged_adm.loc[mask_a, "emerg_rate"])
        years_adm.append(yr)
        r_adm_values.append(r)
        print("  {}: r = {:+.3f}, R2 = {:.3f}, n = {} (p = {:.2e})".format(
            yr, r, r**2, mask_a.sum(), p))

if r_adm_values:
    print()
    print("Mean r (emergency admissions): {:.3f}".format(np.mean(r_adm_values)))
    print("  Note: FACILITY-BASED -- expect catchment contamination")

# =========================================================================
# 4. Publication figures
# =========================================================================
print()
print("=" * 65)
print("STEP 4: Generating Figures")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Panel 1: CHNi vs adjusted PPV rate (2023)
ax = axes[0, 0]
m = merged_2023[mask_2023]
ax.scatter(m["CHNi"], m["adjusted_rate_per_100"], c="#e74c3c", s=50, alpha=0.7,
           edgecolors="white", linewidth=0.5)
z_fit = np.polyfit(m["CHNi"].values, m["adjusted_rate_per_100"].values, 1)
x_line = np.linspace(m["CHNi"].min(), m["CHNi"].max(), 100)
ax.plot(x_line, np.poly1d(z_fit)(x_line), "--", color="#333", linewidth=2)
ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}\np = {:.1e}".format(r_adj, r_adj**2, p_adj),
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
for _, r in m.nlargest(3, "adjusted_rate_per_100").iterrows():
    ax.annotate(r["County"], (r["CHNi"], r["adjusted_rate_per_100"]),
                fontsize=7, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("Adjusted PPV Rate (per 100 ED visits)", fontsize=11)
ax.set_title("CHNi vs Preventable ED Visit Rate\n(NY 2023, Residence-Based, n={})".format(
    mask_2023.sum()), fontsize=12, fontweight="bold")

# Panel 2: Temporal stability of PPV validation
ax = axes[0, 1]
if len(years) > 3:
    ax.bar(years, r_ppv_values, color="#e74c3c", edgecolor="white", alpha=0.85)
    mean_r = np.mean(r_ppv_values)
    ax.axhline(mean_r, color="#333", linewidth=2, linestyle="--",
               label="Mean r = {:.3f}".format(mean_r))
    for yr, rv in zip(years, r_ppv_values):
        ax.text(yr, rv + 0.01, "{:.2f}".format(rv), ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Pearson r (CHNi vs PPV Rate)", fontsize=11)
    ax.set_title("NY Temporal Stability: CHNi vs PPV Rate\n(Residence-Based, 2011-2023)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(r_ppv_values) * 1.2)
    ax.legend(fontsize=10)

# Panel 3: CHNi vs emergency admission rate (2023, facility-based)
ax = axes[1, 0]
yr_2023 = admissions[admissions["year"] == 2023]
emerg_2023 = yr_2023[yr_2023["type_of_admission"] == "Emergency"].groupby(
    "hospital_county")["n"].sum().reset_index()
emerg_2023.columns = ["County", "Emergency_Admissions"]
m_adm = ny_chr.merge(emerg_2023, on="County", how="inner")
m_adm["emerg_rate"] = m_adm["Emergency_Admissions"] / m_adm["population"] * 1000
mask_adm = m_adm["CHNi"].notna() & m_adm["emerg_rate"].notna()
r_e, p_e = stats.pearsonr(m_adm.loc[mask_adm, "CHNi"], m_adm.loc[mask_adm, "emerg_rate"])

ax.scatter(m_adm.loc[mask_adm, "CHNi"], m_adm.loc[mask_adm, "emerg_rate"],
           c="#3498db", s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
z_fit3 = np.polyfit(m_adm.loc[mask_adm, "CHNi"].values,
                     m_adm.loc[mask_adm, "emerg_rate"].values, 1)
ax.plot(x_line, np.poly1d(z_fit3)(x_line), "--", color="#333", linewidth=2)
ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}".format(r_e, r_e**2),
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
# Label catchment outliers
for _, r in m_adm[mask_adm].nlargest(3, "emerg_rate").iterrows():
    ax.annotate(r["County"], (r["CHNi"], r["emerg_rate"]),
                fontsize=7, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("Emergency Admissions per 1,000 Pop", fontsize=11)
ax.set_title("CHNi vs Emergency Admission Rate\n(NY 2023, Facility-Based, n={})".format(
    mask_adm.sum()), fontsize=12, fontweight="bold")

# Panel 4: Comparison bar chart -- different validation targets
ax = axes[1, 1]
targets = ["PPV Rate\n(Residence)", "Prev Hosp\n(CHR)", "Emerg Admit\n(Facility)"]
r_vals = [r_adj, r_prev, r_e]
colors = ["#e74c3c", "#3498db", "#f39c12"]
bars = ax.bar(targets, r_vals, color=colors, edgecolor="white", alpha=0.85)
for bar, rv in zip(bars, r_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            "r={:.3f}".format(rv), ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Pearson r with CHNi", fontsize=11)
ax.set_title("NY Validation: CHNi vs Different Targets", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(r_vals) * 1.25)

plt.suptitle("New York: CHNi Validation with SPARCS Data (Zero Refitting)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig_ny_sparcs_validation.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("  -> fig_ny_sparcs_validation.png")

# =========================================================================
# 5. Combined CA + NY residence-based summary
# =========================================================================
print()
print("=" * 65)
print("COMBINED RESIDENCE-BASED VALIDATION SUMMARY")
print("=" * 65)

# CA results (from previous validation)
ca_ed_path = Path("data/california/ca_ed_by_residence.csv")
if ca_ed_path.exists():
    ca_ed = pd.read_csv(ca_ed_path)
    ca_ed["Encounters"] = pd.to_numeric(ca_ed["Encounters"], errors="coerce")

    ca_chr = chr_df[
        (chr_df["FIPS"].str[:2] == "06") &
        (chr_df["FIPS"].str[2:] != "000")
    ].copy().reset_index(drop=True)
    ca_chr["County"] = ca_chr["Name"].str.replace(" County", "").str.strip()
    ca_chr = ca_chr[ca_chr["County"] != "California"].copy()
    ca_chr["diabetes"] = pd.to_numeric(ca_chr["Diabetes Prevalence raw value"], errors="coerce")
    ca_chr["life_exp"] = pd.to_numeric(ca_chr["Life Expectancy raw value"], errors="coerce")
    ca_chr["pcp_rate"] = pd.to_numeric(ca_chr["Primary Care Physicians raw value"], errors="coerce")
    ca_chr["phys_inact"] = pd.to_numeric(ca_chr["Physical Inactivity raw value"], errors="coerce")
    ca_chr["poverty"] = pd.to_numeric(ca_chr["Children in Poverty raw value"], errors="coerce")
    ca_chr["population"] = pd.to_numeric(ca_chr["Population raw value"], errors="coerce")

    z_ca = {}
    v = ca_chr["diabetes"]; z_ca["z_d"] = (v - v.mean()) / v.std()
    v = 1.0 / ca_chr["life_exp"].replace(0, np.nan); z_ca["z_l"] = (v - v.mean()) / v.std()
    v = ca_chr["pcp_rate"]; z_ca["z_p"] = (v - v.mean()) / v.std()
    v = ca_chr["phys_inact"]; z_ca["z_i"] = (v - v.mean()) / v.std()
    v = ca_chr["poverty"]; z_ca["z_pv"] = (v - v.mean()) / v.std()
    ca_chr["CHNi"] = pd.DataFrame(z_ca).mean(axis=1)

    ca_2023 = ca_ed[ca_ed["Service year"] == 2023].groupby("Patient County")["Encounters"].sum().reset_index()
    ca_2023.columns = ["County", "ED_Visits"]
    ca_m = ca_chr.merge(ca_2023, on="County", how="inner")
    ca_m["ED_rate"] = ca_m["ED_Visits"] / ca_m["population"] * 1000
    ca_mask = ca_m["CHNi"].notna() & ca_m["ED_rate"].notna()
    r_ca, p_ca = stats.pearsonr(ca_m.loc[ca_mask, "CHNi"], ca_m.loc[ca_mask, "ED_rate"])

    print()
    print("{:<25s} {:>8s} {:>10s} {:>10s} {:>12s} {:>15s}".format(
        "State", "N", "r", "R2", "p", "Data Source"))
    print("-" * 85)
    print("{:<25s} {:>8d} {:>+10.3f} {:>10.3f} {:>12.2e} {:>15s}".format(
        "California (2023)", int(ca_mask.sum()), r_ca, r_ca**2, p_ca, "HCAI ED Visits"))
    print("{:<25s} {:>8d} {:>+10.3f} {:>10.3f} {:>12.2e} {:>15s}".format(
        "New York (2023)", int(mask_2023.sum()), r_adj, r_adj**2, p_adj, "SPARCS PPV Rate"))
    print()
    print("Both use RESIDENCE-BASED data (by patient county of residence)")
    print("Both use ZERO REFITTING (same CHNi formula, different states)")

print()
print("=" * 65)
print("NY SPARCS VALIDATION COMPLETE")
print("=" * 65)
