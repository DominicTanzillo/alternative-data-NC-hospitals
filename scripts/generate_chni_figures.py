"""Generate additional CHNi visualizations: choropleth map, distribution, radar, temporal."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path("results/chni")
OUT.mkdir(parents=True, exist_ok=True)
scores = pd.read_csv(OUT / "chni_scores.csv")
scores["FIPS"] = scores["FIPS"].astype(str).str.zfill(5)

# =========================================================================
# 1. Choropleth Map of NC counties
# =========================================================================
print("1. Generating choropleth map...")

try:
    import geopandas as gpd

    url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_500k.zip"
    gdf = gpd.read_file(url)
    nc = gdf[gdf["STATEFP"] == "37"].copy()
    nc["FIPS"] = nc["GEOID"]
    nc = nc.merge(scores[["FIPS", "CHNi", "Tier"]], on="FIPS", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    cmap = plt.cm.RdYlGn_r
    vmin, vmax = -2.0, 2.5
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    nc.plot(column="CHNi", cmap=cmap, norm=norm, linewidth=0.5,
            edgecolor="0.4", ax=ax, legend=False,
            missing_kwds={"color": "lightgrey"})

    for _, row in nc.iterrows():
        if row["CHNi"] is not None and (row["CHNi"] > 1.2 or row["CHNi"] < -1.2):
            centroid = row.geometry.centroid
            ax.annotate(row["NAME"], (centroid.x, centroid.y),
                        fontsize=6, ha="center", va="center",
                        fontweight="bold", color="black",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  alpha=0.7, linewidth=0))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("CHNi Score (higher = greater need)", fontsize=11)

    ax.set_title("Community Health Need Index (CHNi) -- North Carolina Counties, 2021",
                 fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "fig_chni_map.png", dpi=250, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  -> fig_chni_map.png")

except Exception as e:
    print(f"  Map generation issue: {e}")

# =========================================================================
# 2. Distribution + tier histogram
# =========================================================================
print("2. Generating distribution figure...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

tier_colors = {
    "Very High Need": "#c0392b",
    "High Need": "#e74c3c",
    "Moderate": "#f39c12",
    "Low Need": "#2ecc71",
    "Very Low Need": "#27ae60",
}

# 2a. Histogram with tier coloring
ax = axes[0]
bins = np.linspace(-2.0, 2.5, 25)
for tier in ["Very Low Need", "Low Need", "Moderate", "High Need", "Very High Need"]:
    subset = scores[scores["Tier"] == tier]["CHNi"]
    ax.hist(subset, bins=bins, color=tier_colors[tier], alpha=0.85,
            label="{} (n={})".format(tier, len(subset)),
            edgecolor="white", linewidth=0.5)

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_ylabel("Number of Counties", fontsize=11)
ax.set_title("CHNi Score Distribution", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")

# 2b. Component box plots
ax = axes[1]
z_cols = [c for c in scores.columns if c.startswith("z_")]
z_labels = [c.replace("z_", "").replace("_", " ").title() for c in z_cols]
bp = ax.boxplot([scores[c].dropna() for c in z_cols],
                labels=z_labels, patch_artist=True, vert=True)
comp_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
for patch, color in zip(bp["boxes"], comp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Z-score", fontsize=11)
ax.set_title("Component Distributions", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=30, labelsize=9)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

# 2c. Lollipop chart -- all 100 counties sorted
ax = axes[2]
sorted_s = scores.sort_values("CHNi")
colors = [tier_colors.get(str(t), "#999") for t in sorted_s["Tier"]]
ax.barh(range(len(sorted_s)), sorted_s["CHNi"].values, color=colors,
        height=0.8, edgecolor="none")
ax.axvline(0, color="black", linewidth=1, linestyle="-")
ax.set_yticks([])
ax.set_xlabel("CHNi Score", fontsize=11)
ax.set_title("All 100 Counties Ranked", fontsize=12, fontweight="bold")

# Label extremes
for i, (_, row) in enumerate(sorted_s.head(3).iterrows()):
    ax.text(row["CHNi"] - 0.05, i, row["County"], fontsize=7,
            va="center", ha="right", color="white", fontweight="bold")
for i_off, (_, row) in enumerate(sorted_s.tail(3).iterrows()):
    idx = len(sorted_s) - 3 + i_off
    ax.text(row["CHNi"] + 0.05, idx, row["County"], fontsize=7,
            va="center", ha="left", color="white", fontweight="bold")

plt.suptitle("Community Health Need Index (CHNi) -- Overview",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig_chni_distribution.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("  -> fig_chni_distribution.png")

# =========================================================================
# 3. Radar/spider chart -- top 5 highest vs top 5 lowest
# =========================================================================
print("3. Generating radar comparison...")

z_cols = [c for c in scores.columns if c.startswith("z_")]
labels = [c.replace("z_", "").replace("_", " ").title() for c in z_cols]
N = len(labels)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                subplot_kw=dict(polar=True))

top5 = scores.nlargest(5, "CHNi")
radar_colors_high = ["#c0392b", "#e74c3c", "#e67e22", "#d35400", "#a93226"]
for (_, row), color in zip(top5.iterrows(), radar_colors_high):
    values = [row[c] for c in z_cols]
    values += values[:1]
    ax1.plot(angles, values, linewidth=1.5, alpha=0.7, label=row["County"], color=color)
    ax1.fill(angles, values, alpha=0.05, color=color)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_title("Top 5 Highest Need", fontsize=12, fontweight="bold", pad=20)
ax1.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
ax1.set_ylim(-2, 3)

bot5 = scores.nsmallest(5, "CHNi")
radar_colors_low = ["#27ae60", "#2ecc71", "#1abc9c", "#16a085", "#0e6655"]
for (_, row), color in zip(bot5.iterrows(), radar_colors_low):
    values = [row[c] for c in z_cols]
    values += values[:1]
    ax2.plot(angles, values, linewidth=1.5, alpha=0.7, label=row["County"], color=color)
    ax2.fill(angles, values, alpha=0.05, color=color)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_title("Top 5 Lowest Need", fontsize=12, fontweight="bold", pad=20)
ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
ax2.set_ylim(-2, 3)

plt.suptitle("CHNi Component Profiles: Highest vs. Lowest Need Counties",
             fontsize=14, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(OUT / "fig_chni_radar.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("  -> fig_chni_radar.png")

# =========================================================================
# 4. Temporal stability plot
# =========================================================================
print("4. Generating temporal stability figure...")

import sys
sys.path.insert(0, ".")
from src.model.chni import compute_chni
from scipy import stats

panel = pd.read_csv("data/processed/panel_nc.csv", low_memory=False)
panel["FIPS"] = panel["FIPS"].astype(str).str.zfill(5)
for col in panel.columns:
    if col not in {"FIPS", "County", "Year", "Region"}:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

# Compute CHNi per year
for year in sorted(panel["Year"].unique()):
    mask = panel["Year"] == year
    sub = panel[mask].copy()
    panel.loc[mask, "CHNi"] = compute_chni(sub).values

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 4a. CHNi trajectories for select counties
ax = axes[0]
highlight = {
    "Robeson": "#c0392b", "Northampton": "#e74c3c", "Halifax": "#f39c12",
    "Wake": "#27ae60", "Orange": "#2ecc71", "Mecklenburg": "#3498db",
}

for fips in panel["FIPS"].unique():
    county_data = panel[panel["FIPS"] == fips].sort_values("Year")
    cname = county_data["County"].iloc[0]
    if cname not in highlight:
        ax.plot(county_data["Year"], county_data["CHNi"], color="#ddd",
                linewidth=0.3, alpha=0.5)

for cname, color in highlight.items():
    county_data = panel[panel["County"] == cname].sort_values("Year")
    ax.plot(county_data["Year"], county_data["CHNi"], "o-", color=color,
            linewidth=2, markersize=5, label=cname, alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("CHNi Score", fontsize=11)
ax.set_title("CHNi Trajectories (2015-2023)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")

# 4b. Validation r across years
ax = axes[1]
years_list = sorted(panel["Year"].unique())
r_values = []
for year in years_list:
    sub = panel[panel["Year"] == year]
    m = sub["CHNi"].notna() & sub["chr_preventable_hosp"].notna()
    if m.sum() > 10:
        r, _ = stats.pearsonr(sub.loc[m, "CHNi"], sub.loc[m, "chr_preventable_hosp"])
        r_values.append(r)
    else:
        r_values.append(np.nan)

ax.bar(years_list, r_values, color="#3498db", edgecolor="white", alpha=0.85)
mean_r = np.nanmean(r_values)
ax.axhline(mean_r, color="#e74c3c", linewidth=2, linestyle="--",
           label="Mean r = {:.3f}".format(mean_r))
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Pearson r (CHNi vs. Preventable Hosp.)", fontsize=11)
ax.set_title("Validation Stability Across Years", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.75)
ax.legend(fontsize=10)

for yr, rv in zip(years_list, r_values):
    if not np.isnan(rv):
        ax.text(yr, rv + 0.01, "{:.2f}".format(rv), ha="center", fontsize=8,
                fontweight="bold")

plt.suptitle("CHNi Temporal Consistency (2015-2023)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig_chni_temporal.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("  -> fig_chni_temporal.png")

print("\nAll figures generated successfully!")
print("Files in {}:".format(OUT))
for f in sorted(OUT.iterdir()):
    print("  {}".format(f.name))
