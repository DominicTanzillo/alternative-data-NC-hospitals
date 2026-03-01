"""Generate interactive US county choropleth map of CHNi v2 scores.

Reads CHR 2024 raw values, computes CHNi v2 for all ~3100 US counties,
and produces an interactive HTML map using Plotly.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import urllib.request
import ssl

import plotly.express as px

OUT = Path("results/chni/data_driven")
OUT.mkdir(parents=True, exist_ok=True)

# ── Load v2 definition ──────────────────────────────────────────────────
v2_def = pd.read_csv(OUT / "chni_v2_definition.csv")
consensus_cols = v2_def["column"].tolist()
directions = dict(zip(v2_def["column"], v2_def["direction"]))

print("CHNi v2 components:")
for _, row in v2_def.iterrows():
    print("  {} ({})".format(row["feature"], row["direction"]))
print()

# ── Load CHR 2024 national ──────────────────────────────────────────────
print("Loading CHR 2024 national data...")
raw = pd.read_csv("data/california/chr_2024_analytic.csv",
                   low_memory=False, encoding="latin-1")
df = raw.iloc[1:].copy()
df.columns = raw.columns
df["FIPS"] = df["5-digit FIPS Code"].astype(str).str.zfill(5)
# Drop state-level rows
df = df[df["FIPS"].str[2:] != "000"].copy().reset_index(drop=True)

# State FIPS lookup
state_fips = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}
df["State"] = df["FIPS"].str[:2].map(state_fips)
df["County_Name"] = df["Name"].str.replace(" County", "").str.strip()

print("Counties loaded: {}".format(len(df)))

# ── Compute CHNi v2 ─────────────────────────────────────────────────────
print("Computing CHNi v2 for all counties...")

z_parts = []
for col in consensus_cols:
    vals = pd.to_numeric(df[col], errors="coerce")
    if directions[col] == "invert":
        vals = -vals
    z = (vals - vals.mean()) / vals.std()
    z_parts.append(z)

z_matrix = pd.concat(z_parts, axis=1)
df["CHNi_v2"] = z_matrix.mean(axis=1)

# Also compute v1 for comparison tooltip
v1_cols = [
    "Diabetes Prevalence raw value",
    "Life Expectancy raw value",
    "Primary Care Physicians raw value",
    "Physical Inactivity raw value",
    "Children in Poverty raw value",
]
z_v1 = []
for col in v1_cols:
    vals = pd.to_numeric(df[col], errors="coerce")
    if col in ("Life Expectancy raw value", "Primary Care Physicians raw value"):
        vals = 1.0 / vals.replace(0, np.nan)
    z = (vals - vals.mean()) / vals.std()
    z_v1.append(z)
df["CHNi_v1"] = pd.concat(z_v1, axis=1).mean(axis=1)

scored = df[df["CHNi_v2"].notna()].copy()
print("Counties with v2 score: {} / {}".format(len(scored), len(df)))
print("v2 range: [{:.2f}, {:.2f}]".format(scored["CHNi_v2"].min(),
                                            scored["CHNi_v2"].max()))

# Clip extreme outliers for color scale
scored["v2_display"] = scored["CHNi_v2"].clip(-2.5, 2.5)

# ── Need tier ────────────────────────────────────────────────────────────
scored["Tier"] = pd.cut(
    scored["CHNi_v2"],
    bins=[-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf],
    labels=["Very Low", "Low", "Moderate", "High", "Very High"],
)

# Add component raw values for tooltip
for col in consensus_cols:
    short = col.replace(" raw value", "")
    scored[short] = pd.to_numeric(scored[col], errors="coerce")

# Population for tooltip
scored["Population"] = pd.to_numeric(scored["Population raw value"], errors="coerce")

# Prev hosp for tooltip
scored["Prev Hosp Rate"] = pd.to_numeric(
    scored["Preventable Hospital Stays raw value"], errors="coerce")

# ── Download GeoJSON ─────────────────────────────────────────────────────
geojson_path = Path("data/us_counties_geojson.json")
if not geojson_path.exists():
    print("Downloading US county GeoJSON...")
    url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
           "geojson-counties-fips.json")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        data = resp.read()
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with open(geojson_path, "wb") as f:
        f.write(data)
    print("  Saved to {}".format(geojson_path))
else:
    print("Using cached GeoJSON: {}".format(geojson_path))

with open(geojson_path) as f:
    counties_geojson = json.load(f)

# ── Build hover text ─────────────────────────────────────────────────────
hover_parts = []
for _, row in scored.iterrows():
    parts = [
        "<b>{}, {}</b>".format(row["County_Name"], row["State"]),
        "FIPS: {}".format(row["FIPS"]),
        "",
        "<b>CHNi v2: {:+.2f}</b>  (Tier: {})".format(row["CHNi_v2"], row["Tier"]),
        "CHNi v1: {:+.2f}".format(row["CHNi_v1"]) if pd.notna(row["CHNi_v1"]) else "",
        "",
        "--- v2 Components ---",
    ]
    for col in consensus_cols:
        short = col.replace(" raw value", "")
        val = row.get(short, np.nan)
        direction = directions[col]
        arrow = "+" if direction == "positive" else "-"
        if pd.notna(val):
            parts.append("  {} {}: {:.1f}".format(arrow, short, val))
    if pd.notna(row.get("Population", np.nan)):
        parts.append("")
        parts.append("Pop: {:,.0f}".format(row["Population"]))
    if pd.notna(row.get("Prev Hosp Rate", np.nan)):
        parts.append("Prev Hosp Rate: {:.0f}".format(row["Prev Hosp Rate"]))
    hover_parts.append("<br>".join(parts))

scored["hover_text"] = hover_parts

# ── Create choropleth ────────────────────────────────────────────────────
print("Building interactive map...")

fig = px.choropleth(
    scored,
    geojson=counties_geojson,
    locations="FIPS",
    color="v2_display",
    color_continuous_scale=[
        [0.0, "#1a9641"],    # dark green (very low need)
        [0.2, "#a6d96a"],    # light green
        [0.4, "#ffffbf"],    # yellow (moderate)
        [0.6, "#fdae61"],    # orange
        [0.8, "#d73027"],    # red
        [1.0, "#67001f"],    # dark red (very high need)
    ],
    range_color=[-2.5, 2.5],
    scope="usa",
    labels={"v2_display": "CHNi v2"},
    custom_data=["hover_text"],
)

fig.update_traces(
    hovertemplate="%{customdata[0]}<extra></extra>",
    marker_line_width=0.2,
    marker_line_color="rgba(100,100,100,0.3)",
)

fig.update_coloraxes(
    colorbar_title_text="CHNi v2<br>Score",
    colorbar_tickvals=[-2, -1, 0, 1, 2],
    colorbar_ticktext=["Very Low<br>Need (-2)", "Low (-1)",
                       "Moderate (0)", "High (+1)", "Very High<br>Need (+2)"],
    colorbar_len=0.6,
)

fig.update_layout(
    title={
        "text": ("Community Health Need Index v2 (Data-Driven)<br>"
                 "<sup>6 variables selected by LASSO + Elastic Net + Random Forest "
                 "against real ED data from CA & NY</sup>"),
        "x": 0.5,
        "xanchor": "center",
        "font": {"size": 18},
    },
    geo=dict(
        lakecolor="rgb(220, 230, 240)",
        landcolor="rgb(245, 245, 245)",
        showlakes=True,
    ),
    margin={"r": 10, "t": 80, "l": 10, "b": 30},
    width=1200,
    height=700,
    annotations=[
        dict(
            text=("Components: Food Insecurity (+) | Teen Births (+) | "
                  "Median Household Income (-) | Frequent Mental Distress (+) | "
                  "Voter Turnout (-) | Suicides (+)"),
            xref="paper", yref="paper",
            x=0.5, y=-0.02,
            showarrow=False,
            font=dict(size=11, color="grey"),
            xanchor="center",
        ),
    ],
)

# ── Save ─────────────────────────────────────────────────────────────────
html_path = OUT / "map_chni_v2_usa.html"
fig.write_html(
    str(html_path),
    include_plotlyjs=True,
    full_html=True,
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    },
)

print()
print("Interactive map saved: {}".format(html_path))
print("File size: {:.1f} MB".format(html_path.stat().st_size / 1e6))

# ── Summary stats ────────────────────────────────────────────────────────
print()
print("Tier distribution:")
tier_counts = scored["Tier"].value_counts().sort_index()
for tier, count in tier_counts.items():
    print("  {:<15s} {:>5d} counties ({:.1f}%)".format(
        tier, count, count / len(scored) * 100))

# Top/bottom states by mean v2
state_means = scored.groupby("State")["CHNi_v2"].mean().sort_values(ascending=False)
print()
print("Top 10 states by mean CHNi v2 (highest need):")
for st, val in state_means.head(10).items():
    n = (scored["State"] == st).sum()
    print("  {} {:>+8.3f}  (n={})".format(st, val, n))
print()
print("Bottom 10 states (lowest need):")
for st, val in state_means.tail(10).items():
    n = (scored["State"] == st).sum()
    print("  {} {:>+8.3f}  (n={})".format(st, val, n))

print()
print("Done. Open the HTML file in a browser to explore.")
