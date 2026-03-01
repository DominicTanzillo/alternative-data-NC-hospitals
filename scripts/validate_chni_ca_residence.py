"""Validate CHNi against California RESIDENCE-BASED ED visits (HCAI open data)."""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/chni/out_of_state")
OUT.mkdir(parents=True, exist_ok=True)

# Load CA residence-based ED data
ed = pd.read_csv("data/california/ca_ed_by_residence.csv")
ed["Encounters"] = pd.to_numeric(ed["Encounters"], errors="coerce")

# Aggregate total per county for 2023
ed_2023 = ed[ed["Service year"] == 2023].groupby("Patient County")["Encounters"].sum().reset_index()
ed_2023.columns = ["County", "Total_ED_Residence"]

# Load CHR 2024 for CA
chr_raw = pd.read_csv("data/california/chr_2024_analytic.csv", low_memory=False, encoding="latin-1")
chr_df = chr_raw.iloc[1:].copy()
chr_df.columns = chr_raw.columns

ca_chr = chr_df[
    (chr_df["5-digit FIPS Code"].astype(str).str.zfill(5).str[:2] == "06") &
    (chr_df["5-digit FIPS Code"].astype(str).str.zfill(5).str[2:] != "000")
].copy().reset_index(drop=True)

ca_chr["County"] = ca_chr["Name"].str.replace(" County", "").str.strip()
# Remove state-level summary row
ca_chr = ca_chr[ca_chr["County"] != "California"].copy()
ca_chr["diabetes"] = pd.to_numeric(ca_chr["Diabetes Prevalence raw value"], errors="coerce")
ca_chr["life_exp"] = pd.to_numeric(ca_chr["Life Expectancy raw value"], errors="coerce")
ca_chr["pcp_rate"] = pd.to_numeric(ca_chr["Primary Care Physicians raw value"], errors="coerce")
ca_chr["phys_inact"] = pd.to_numeric(ca_chr["Physical Inactivity raw value"], errors="coerce")
ca_chr["poverty"] = pd.to_numeric(ca_chr["Children in Poverty raw value"], errors="coerce")
ca_chr["prev_hosp"] = pd.to_numeric(ca_chr["Preventable Hospital Stays raw value"], errors="coerce")
ca_chr["population"] = pd.to_numeric(ca_chr["Population raw value"], errors="coerce")

# Compute CHNi
z = {}
vals = ca_chr["diabetes"]; z["z_diab"] = (vals - vals.mean()) / vals.std()
vals = 1.0 / ca_chr["life_exp"].replace(0, np.nan); z["z_life"] = (vals - vals.mean()) / vals.std()
vals = ca_chr["pcp_rate"]; z["z_pcp"] = (vals - vals.mean()) / vals.std()
vals = ca_chr["phys_inact"]; z["z_inact"] = (vals - vals.mean()) / vals.std()
vals = ca_chr["poverty"]; z["z_pov"] = (vals - vals.mean()) / vals.std()
z_df = pd.DataFrame(z)
ca_chr["CHNi"] = z_df.mean(axis=1)

# Merge
merged = ca_chr.merge(ed_2023, on="County", how="inner")
merged["ED_rate_per_1k"] = merged["Total_ED_Residence"] / merged["population"] * 1000

print("=" * 65)
print("CALIFORNIA: CHNi vs RESIDENCE-BASED ED VISITS")
print("=" * 65)
print("Counties matched: {}".format(len(merged)))
print("Total CA ED visits (residence-based, 2023): {:,.0f}".format(
    merged["Total_ED_Residence"].sum()))
print()

mask = merged["CHNi"].notna() & merged["ED_rate_per_1k"].notna()

r1, p1 = stats.pearsonr(merged.loc[mask, "CHNi"], merged.loc[mask, "ED_rate_per_1k"])
print("CHNi vs ED_rate_per_1k (RESIDENCE-BASED):")
print("  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(r1, r1**2, p1))
print("  *** RESIDENCE-BASED -- no catchment area contamination ***")
print()

r2, p2 = stats.pearsonr(merged.loc[mask, "CHNi"], merged.loc[mask, "prev_hosp"])
print("CHNi vs Preventable Hosp Rate (CHR):")
print("  r = {:+.3f}, R2 = {:.3f}, p = {:.2e}".format(r2, r2**2, p2))
print()

rho, p_rho = stats.spearmanr(merged.loc[mask, "CHNi"], merged.loc[mask, "ED_rate_per_1k"])
print("Spearman rho (CHNi vs ED rate): {:+.3f}, p = {:.2e}".format(rho, p_rho))
print()

# Sanity check: ED rate vs prev hosp
r_sanity, _ = stats.pearsonr(merged.loc[mask, "ED_rate_per_1k"], merged.loc[mask, "prev_hosp"])
print("Sanity: ED_rate vs prev_hosp: r = {:+.3f}".format(r_sanity))
print()

# Top/bottom counties
print("Top 10 highest ED rate (per 1k residents):")
for _, r in merged.sort_values("ED_rate_per_1k", ascending=False).head(10).iterrows():
    print("  {:<25s} rate={:.0f}/1k  CHNi={:+.2f}  prev_hosp={:.0f}".format(
        r["County"], r["ED_rate_per_1k"], r["CHNi"], r["prev_hosp"]))

print()
print("Top 10 lowest ED rate:")
for _, r in merged.sort_values("ED_rate_per_1k").head(10).iterrows():
    print("  {:<25s} rate={:.0f}/1k  CHNi={:+.2f}  prev_hosp={:.0f}".format(
        r["County"], r["ED_rate_per_1k"], r["CHNi"], r["prev_hosp"]))

# ---- Publication figure ----
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.scatter(merged["CHNi"], merged["ED_rate_per_1k"], c="#e74c3c", s=50, alpha=0.7,
           edgecolors="white", linewidth=0.5)
z_fit = np.polyfit(merged.loc[mask, "CHNi"], merged.loc[mask, "ED_rate_per_1k"], 1)
x_line = np.linspace(merged["CHNi"].min(), merged["CHNi"].max(), 100)
ax.plot(x_line, np.poly1d(z_fit)(x_line), "--", color="#333", linewidth=2)
ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}\np = {:.1e}".format(r1, r1**2, p1),
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
for _, r in merged.nlargest(3, "ED_rate_per_1k").iterrows():
    ax.annotate(r["County"], (r["CHNi"], r["ED_rate_per_1k"]),
                fontsize=7, xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("ED Visits per 1,000 Residents", fontsize=11)
ax.set_title("CHNi vs Residence-Based ED Rate\n(California 2023, n={})".format(mask.sum()),
             fontsize=12, fontweight="bold")

ax = axes[1]
ax.scatter(merged["CHNi"], merged["prev_hosp"], c="#3498db", s=50, alpha=0.7,
           edgecolors="white", linewidth=0.5)
z_fit2 = np.polyfit(merged.loc[mask, "CHNi"], merged.loc[mask, "prev_hosp"], 1)
ax.plot(x_line, np.poly1d(z_fit2)(x_line), "--", color="#333", linewidth=2)
ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}".format(r2, r2**2),
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("Preventable Hosp Rate (CHR)", fontsize=11)
ax.set_title("CHNi vs Preventable Hospitalizations\n(California, n={})".format(mask.sum()),
             fontsize=12, fontweight="bold")

ax = axes[2]
ax.scatter(merged["ED_rate_per_1k"], merged["prev_hosp"], c="#27ae60", s=50, alpha=0.7,
           edgecolors="white", linewidth=0.5)
ax.text(0.05, 0.95, "r = {:.3f}".format(r_sanity),
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
ax.set_xlabel("ED Visits per 1,000 Residents", fontsize=11)
ax.set_ylabel("Preventable Hosp Rate (CHR)", fontsize=11)
ax.set_title("Residence ED Rate vs Prev Hosp\n(Sanity Check)", fontsize=12, fontweight="bold")

plt.suptitle("California: CHNi vs RESIDENCE-BASED ED Visits (Zero Refitting)",
             fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(OUT / "fig_ca_residence_ed_validation.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print()
print("-> fig_ca_residence_ed_validation.png")

# ---- Multi-year temporal validation ----
print()
print("=" * 65)
print("CA TEMPORAL VALIDATION (2015-2024)")
print("=" * 65)

years = []
r_values = []
for yr in range(2015, 2025):
    ed_yr = ed[ed["Service year"] == yr].groupby("Patient County")["Encounters"].sum().reset_index()
    ed_yr.columns = ["County", "Total_ED"]
    m = ca_chr.merge(ed_yr, on="County", how="inner")
    m["rate"] = m["Total_ED"] / m["population"] * 1000
    valid = m["CHNi"].notna() & m["rate"].notna()
    if valid.sum() > 10:
        r, p = stats.pearsonr(m.loc[valid, "CHNi"], m.loc[valid, "rate"])
        years.append(yr)
        r_values.append(r)
        print("  {}: r = {:+.3f}, R2 = {:.3f}, n = {} (p = {:.2e})".format(
            yr, r, r**2, valid.sum(), p))

# Temporal stability figure
if len(years) > 3:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, r_values, color="#e74c3c", edgecolor="white", alpha=0.85)
    mean_r = np.mean(r_values)
    ax.axhline(mean_r, color="#333", linewidth=2, linestyle="--",
               label="Mean r = {:.3f}".format(mean_r))
    for yr, rv in zip(years, r_values):
        ax.text(yr, rv + 0.01, "{:.2f}".format(rv), ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Pearson r (CHNi vs Residence-Based ED Rate)", fontsize=11)
    ax.set_title("California: CHNi Temporal Stability Against REAL ED Visits",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(r_values) * 1.15)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig_ca_temporal_ed.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print()
    print("-> fig_ca_temporal_ed.png")

print()
print("=" * 65)
print("VALIDATION COMPLETE")
print("=" * 65)
