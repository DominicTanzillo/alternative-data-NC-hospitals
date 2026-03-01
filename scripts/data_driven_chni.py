"""Data-Driven CHNi v2 — Let Real ED Data Pick the Variables.

Uses CA residence-based ED visits + NY SPARCS PPV rates as ground truth,
CHR 2024 raw values as candidate features.  Six phases:

1. Univariate screening (Pearson r against CA & NY targets)
2. LASSO / Elastic Net on pooled CA+NY (~118 counties)
3. Random Forest permutation importance
4. Build CHNi v2 from consensus features
5. Head-to-head validation (v1 vs v2 vs v2w)
6. National model (n~3000 counties, CHR preventable hosp as target)
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/chni/data_driven")
OUT.mkdir(parents=True, exist_ok=True)

# ── Column lists ────────────────────────────────────────────────────────
# Exclude demographics, population, and preventable hosp (Phase 6 target)
EXCLUDE_COLS = {
    "Population raw value",
    "Preventable Hospital Stays raw value",
    "% Below 18 Years of Age raw value",
    "% 65 and Older raw value",
    "% Non-Hispanic Black raw value",
    "% American Indian or Alaska Native raw value",
    "% Asian raw value",
    "% Native Hawaiian or Other Pacific Islander raw value",
    "% Hispanic raw value",
    "% Non-Hispanic White raw value",
    "% Not Proficient in English raw value",
    "% Female raw value",
    "% Rural raw value",
}

# CHNi v1 components (for comparison)
V1_COLS = [
    "Diabetes Prevalence raw value",
    "Life Expectancy raw value",
    "Primary Care Physicians raw value",
    "Physical Inactivity raw value",
    "Children in Poverty raw value",
]


# ── Helpers ─────────────────────────────────────────────────────────────
def load_chr_national():
    """Load CHR 2024 analytic file, return county-level rows with FIPS."""
    raw = pd.read_csv("data/california/chr_2024_analytic.csv",
                       low_memory=False, encoding="latin-1")
    df = raw.iloc[1:].copy()
    df.columns = raw.columns
    df["FIPS"] = df["5-digit FIPS Code"].astype(str).str.zfill(5)
    # Drop state-level summaries (FIPS ending in 000)
    df = df[df["FIPS"].str[2:] != "000"].copy().reset_index(drop=True)
    return df


def get_feature_cols(df):
    """Return list of CHR raw-value columns eligible as features."""
    raw_cols = [c for c in df.columns if c.endswith("raw value")]
    return [c for c in raw_cols if c not in EXCLUDE_COLS]


def extract_features(df, feature_cols):
    """Convert feature columns to numeric, return DataFrame."""
    X = pd.DataFrame()
    for col in feature_cols:
        X[col] = pd.to_numeric(df[col], errors="coerce")
    return X


def short_name(col):
    """Shorten 'Foo raw value' → 'Foo'."""
    return col.replace(" raw value", "")


def compute_chni_v1(df):
    """Compute CHNi v1 (hand-picked, equal-weight z-score average)."""
    z_parts = []
    for col in V1_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        if col == "Life Expectancy raw value":
            vals = 1.0 / vals.replace(0, np.nan)
        z = (vals - vals.mean()) / vals.std()
        z_parts.append(z)
    return pd.concat(z_parts, axis=1).mean(axis=1)


# ═════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("DATA-DRIVEN CHNi v2: LOADING DATA")
print("=" * 70)

chr_all = load_chr_national()
feature_cols = get_feature_cols(chr_all)
print("CHR counties loaded: {}".format(len(chr_all)))
print("Candidate features: {} (85 total - {} excluded)".format(
    len(feature_cols), len(EXCLUDE_COLS)))

# ── CA data ──
ca_chr = chr_all[chr_all["FIPS"].str[:2] == "06"].copy().reset_index(drop=True)
ca_chr["County"] = ca_chr["Name"].str.replace(" County", "").str.strip()

ca_ed = pd.read_csv("data/california/ca_ed_by_residence.csv")
ca_ed["Encounters"] = pd.to_numeric(ca_ed["Encounters"], errors="coerce")
ca_2023 = (ca_ed[ca_ed["Service year"] == 2023]
           .groupby("Patient County")["Encounters"].sum().reset_index())
ca_2023.columns = ["County", "ED_Visits"]

ca_pop = pd.to_numeric(ca_chr["Population raw value"], errors="coerce")
ca_chr["population"] = ca_pop

ca = ca_chr.merge(ca_2023, on="County", how="inner")
ca["target"] = ca["ED_Visits"] / ca["population"] * 1000  # ED rate per 1k
print("CA counties matched: {} (target: ED rate/1k)".format(len(ca)))

# ── NY data ──
ny_chr = chr_all[chr_all["FIPS"].str[:2] == "36"].copy().reset_index(drop=True)
ny_chr["County"] = ny_chr["Name"].str.replace(" County", "").str.strip()
ny_chr["County"] = ny_chr["County"].str.replace("St.", "St", regex=False)

ppv = pd.read_csv("data/new_york/ny_ppv_by_patient_county.csv")
ppv["discharge_year"] = pd.to_numeric(ppv["discharge_year"], errors="coerce")
ppv["observed_rate_per_100"] = pd.to_numeric(ppv["observed_rate_per_100"], errors="coerce")
ppv_2023 = (ppv[ppv["discharge_year"] == 2023]
            .rename(columns={"patient_county_name": "County"})
            [["County", "observed_rate_per_100"]])

ny = ny_chr.merge(ppv_2023, on="County", how="inner")
ny["target"] = ny["observed_rate_per_100"]  # PPV rate per 100
print("NY counties matched: {} (target: observed PPV rate/100)".format(len(ny)))
print()

# ═════════════════════════════════════════════════════════════════════════
# PHASE 1: UNIVARIATE SCREENING
# ═════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: UNIVARIATE SCREENING")
print("=" * 70)

ca_X = extract_features(ca, feature_cols)
ny_X = extract_features(ny, feature_cols)

screening = []
for col in feature_cols:
    row = {"feature": short_name(col), "column": col}
    # CA
    mask_ca = ca_X[col].notna() & ca["target"].notna()
    if mask_ca.sum() > 10:
        r_ca, p_ca = stats.pearsonr(ca_X.loc[mask_ca, col], ca.loc[mask_ca, "target"])
        row["r_ca"] = r_ca
        row["p_ca"] = p_ca
    else:
        row["r_ca"] = np.nan
        row["p_ca"] = np.nan
    # NY
    mask_ny = ny_X[col].notna() & ny["target"].notna()
    if mask_ny.sum() > 10:
        r_ny, p_ny = stats.pearsonr(ny_X.loc[mask_ny, col], ny.loc[mask_ny, "target"])
        row["r_ny"] = r_ny
        row["p_ny"] = p_ny
    else:
        row["r_ny"] = np.nan
        row["p_ny"] = np.nan
    # Average |r|
    abs_rs = [abs(row["r_ca"]), abs(row["r_ny"])]
    abs_rs = [x for x in abs_rs if not np.isnan(x)]
    row["avg_abs_r"] = np.mean(abs_rs) if abs_rs else np.nan
    screening.append(row)

screen_df = pd.DataFrame(screening).sort_values("avg_abs_r", ascending=False)
screen_df.to_csv(OUT / "univariate_screening.csv", index=False)

print("Top 25 features by average |r| across CA + NY:")
print("{:<45s} {:>8s} {:>8s} {:>10s}".format("Feature", "r(CA)", "r(NY)", "avg|r|"))
print("-" * 75)
for _, row in screen_df.head(25).iterrows():
    print("{:<45s} {:>+8.3f} {:>+8.3f} {:>10.3f}".format(
        row["feature"][:45],
        row["r_ca"] if pd.notna(row["r_ca"]) else 0,
        row["r_ny"] if pd.notna(row["r_ny"]) else 0,
        row["avg_abs_r"]))

# Check where v1 variables rank
print()
print("CHNi v1 variable rankings:")
for v1col in V1_COLS:
    name = short_name(v1col)
    rank_row = screen_df[screen_df["feature"] == name]
    if not rank_row.empty:
        idx = screen_df.index.get_loc(rank_row.index[0]) + 1
        print("  #{:>3d}  {:<40s} avg|r| = {:.3f}".format(
            idx, name, rank_row["avg_abs_r"].values[0]))

# ── Univariate screening figure ──
top_n = min(30, len(screen_df.dropna(subset=["avg_abs_r"])))
plot_df = screen_df.head(top_n).iloc[::-1]  # reverse for horizontal bar

fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.35)))
y_pos = range(len(plot_df))
bars_ca = ax.barh(y_pos, plot_df["r_ca"].fillna(0), height=0.4,
                  align="edge", color="#e74c3c", alpha=0.8, label="CA ED rate")
bars_ny = ax.barh([y + 0.4 for y in y_pos], plot_df["r_ny"].fillna(0), height=0.4,
                  align="edge", color="#3498db", alpha=0.8, label="NY PPV rate")

# Highlight v1 variables
v1_names = {short_name(c) for c in V1_COLS}
labels = []
for _, row in plot_df.iterrows():
    lbl = row["feature"][:40]
    if row["feature"] in v1_names:
        lbl = "* " + lbl
    labels.append(lbl)

ax.set_yticks([y + 0.4 for y in y_pos])
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Pearson r with ED Outcome", fontsize=11)
ax.set_title("Univariate Screening: Top {} CHR Features\n(* = CHNi v1 variable)".format(top_n),
             fontsize=13, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.5)
ax.legend(fontsize=10, loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "fig_univariate_screening.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print()
print("-> fig_univariate_screening.png")

# ═════════════════════════════════════════════════════════════════════════
# PHASE 2: LASSO / ELASTIC NET (POOLED CA + NY)
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PHASE 2: LASSO / ELASTIC NET (POOLED CA + NY)")
print("=" * 70)

# Pool datasets: z-score targets within state so scales are comparable
ca_z_target = (ca["target"] - ca["target"].mean()) / ca["target"].std()
ny_z_target = (ny["target"] - ny["target"].mean()) / ny["target"].std()

ca_pool = extract_features(ca, feature_cols).copy()
ca_pool["z_target"] = ca_z_target.values
ca_pool["state"] = "CA"

ny_pool = extract_features(ny, feature_cols).copy()
ny_pool["z_target"] = ny_z_target.values
ny_pool["state"] = "NY"

pooled = pd.concat([ca_pool, ny_pool], ignore_index=True)

# Remove features with >20% missing
miss_pct = pooled[feature_cols].isnull().mean()
keep_cols = miss_pct[miss_pct < 0.20].index.tolist()
dropped = [c for c in feature_cols if c not in keep_cols]
if dropped:
    print("Dropped {} features with >20% missing: {}".format(
        len(dropped), [short_name(c) for c in dropped]))

X_raw = pooled[keep_cols].copy()
y = pooled["z_target"].values

# Impute remaining NaN with median, then standardize
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_imp = imputer.fit_transform(X_raw)
X_std = scaler.fit_transform(X_imp)

# ── LassoCV ──
print()
print("Running LassoCV (100 alphas, 10-fold)...")
lasso = LassoCV(n_alphas=100, cv=10, max_iter=10000, random_state=42)
lasso.fit(X_std, y)
lasso_coefs = pd.Series(lasso.coef_, index=keep_cols)
lasso_survivors = lasso_coefs[lasso_coefs.abs() > 1e-6].sort_values(key=abs, ascending=False)

print("LassoCV alpha: {:.4f}".format(lasso.alpha_))
print("Cross-validated R2: {:.3f}".format(lasso.score(X_std, y)))
print("Features surviving LASSO: {}/{}".format(len(lasso_survivors), len(keep_cols)))
print()
print("{:<45s} {:>10s}".format("Feature", "Coef"))
print("-" * 58)
for col, coef in lasso_survivors.items():
    print("{:<45s} {:>+10.4f}".format(short_name(col)[:45], coef))

lasso_df = pd.DataFrame({
    "feature": [short_name(c) for c in lasso_survivors.index],
    "column": lasso_survivors.index,
    "lasso_coef": lasso_survivors.values,
})
lasso_df.to_csv(OUT / "lasso_coefficients.csv", index=False)

# ── ElasticNetCV ──
print()
print("Running ElasticNetCV (7 l1_ratios, 10-fold)...")
enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
                    n_alphas=100, cv=10, max_iter=10000, random_state=42)
enet.fit(X_std, y)
enet_coefs = pd.Series(enet.coef_, index=keep_cols)
enet_survivors = enet_coefs[enet_coefs.abs() > 1e-6].sort_values(key=abs, ascending=False)

print("ElasticNetCV alpha: {:.4f}, l1_ratio: {:.2f}".format(enet.alpha_, enet.l1_ratio_))
print("Cross-validated R2: {:.3f}".format(enet.score(X_std, y)))
print("Features surviving Elastic Net: {}/{}".format(len(enet_survivors), len(keep_cols)))
print()
print("{:<45s} {:>10s}".format("Feature", "Coef"))
print("-" * 58)
for col, coef in enet_survivors.items():
    print("{:<45s} {:>+10.4f}".format(short_name(col)[:45], coef))

enet_df = pd.DataFrame({
    "feature": [short_name(c) for c in enet_survivors.index],
    "column": enet_survivors.index,
    "enet_coef": enet_survivors.values,
})
enet_df.to_csv(OUT / "enet_coefficients.csv", index=False)

# ── LASSO path figure ──
fig, ax = plt.subplots(figsize=(8, 5))
alpha_path = lasso.alphas_
coef_path = []
from sklearn.linear_model import Lasso
for a in alpha_path:
    m = Lasso(alpha=a, max_iter=10000)
    m.fit(X_std, y)
    coef_path.append(m.coef_.copy())
coef_path = np.array(coef_path)  # (n_alphas, n_features)

for i, col in enumerate(keep_cols):
    if col in lasso_survivors.index:
        ax.plot(-np.log10(alpha_path), coef_path[:, i],
                label=short_name(col)[:30], linewidth=1.5)
    else:
        ax.plot(-np.log10(alpha_path), coef_path[:, i],
                color="grey", alpha=0.15, linewidth=0.5)

ax.axvline(-np.log10(lasso.alpha_), color="black", linestyle="--",
           linewidth=1, label="Optimal alpha")
ax.set_xlabel("-log10(alpha)", fontsize=11)
ax.set_ylabel("Coefficient", fontsize=11)
ax.set_title("LASSO Path: Feature Selection", fontsize=13, fontweight="bold")
ax.legend(fontsize=7, loc="best", ncol=2)
plt.tight_layout()
plt.savefig(OUT / "fig_lasso_path.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print()
print("-> fig_lasso_path.png")

# ═════════════════════════════════════════════════════════════════════════
# PHASE 3: RANDOM FOREST PERMUTATION IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PHASE 3: RANDOM FOREST PERMUTATION IMPORTANCE")
print("=" * 70)

rf = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5,
                            max_features="sqrt", random_state=42, n_jobs=-1)
rf.fit(X_std, y)
rf_r2 = rf.score(X_std, y)
print("RF in-sample R2: {:.3f}".format(rf_r2))

print("Computing permutation importance (30 repeats)...")
perm = permutation_importance(rf, X_std, y, n_repeats=30,
                               random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=keep_cols)
perm_std = pd.Series(perm.importances_std, index=keep_cols)

# Top features (importance > 0)
rf_top = perm_imp[perm_imp > 0.005].sort_values(ascending=False)
print()
print("Top RF features (permutation importance > 0.005):")
print("{:<45s} {:>12s} {:>10s}".format("Feature", "Importance", "Std"))
print("-" * 70)
for col in rf_top.index:
    print("{:<45s} {:>12.4f} {:>10.4f}".format(
        short_name(col)[:45], perm_imp[col], perm_std[col]))

rf_df = pd.DataFrame({
    "feature": [short_name(c) for c in rf_top.index],
    "column": rf_top.index,
    "rf_importance": rf_top.values,
    "rf_importance_std": [perm_std[c] for c in rf_top.index],
})
rf_df.to_csv(OUT / "rf_permutation_importance.csv", index=False)

# ═════════════════════════════════════════════════════════════════════════
# PHASE 4: BUILD CHNi v2 (CONSENSUS FEATURES)
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PHASE 4: BUILD CHNi v2 (CONSENSUS FEATURES)")
print("=" * 70)

# Consensus = selected by >= 2 of 3 methods
lasso_set = set(lasso_survivors.index)
enet_set = set(enet_survivors.index)
rf_set = set(rf_top.index)

all_selected = lasso_set | enet_set | rf_set
consensus = []
for col in all_selected:
    count = sum([col in lasso_set, col in enet_set, col in rf_set])
    if count >= 2:
        consensus.append(col)

# Sort by average LASSO abs(coef) for stable ordering
consensus_order = []
for col in consensus:
    lc = abs(lasso_coefs.get(col, 0))
    consensus_order.append((col, lc))
consensus_order.sort(key=lambda x: x[1], reverse=True)
consensus_cols = [c for c, _ in consensus_order]

# Determine directionality from LASSO sign
# Positive coef → higher raw value → more need → positive direction
# Negative coef → higher raw value → less need → invert
directions = {}
for col in consensus_cols:
    lc = lasso_coefs.get(col, 0)
    ec = enet_coefs.get(col, 0)
    # Use LASSO sign if available, else Elastic Net
    if abs(lc) > 1e-6:
        directions[col] = "positive" if lc > 0 else "invert"
    elif abs(ec) > 1e-6:
        directions[col] = "positive" if ec > 0 else "invert"
    else:
        # Default: check univariate correlation sign (average of CA+NY)
        srow = screen_df[screen_df["column"] == col]
        if not srow.empty:
            avg_r = np.nanmean([srow["r_ca"].values[0], srow["r_ny"].values[0]])
            directions[col] = "positive" if avg_r > 0 else "invert"
        else:
            directions[col] = "positive"

print("Consensus features (selected by >= 2 of 3 methods):")
print("{:<45s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
    "Feature", "LASSO", "ENet", "RF", "Direction"))
print("-" * 85)
for col in consensus_cols:
    in_l = "Y" if col in lasso_set else ""
    in_e = "Y" if col in enet_set else ""
    in_r = "Y" if col in rf_set else ""
    print("{:<45s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
        short_name(col)[:45], in_l, in_e, in_r, directions[col]))

print()
print("Total consensus features: {}".format(len(consensus_cols)))


def compute_chni_v2(df, consensus_cols, directions, weighted=False, lasso_coefs=None):
    """Compute CHNi v2 for a DataFrame with CHR raw value columns.

    If weighted=False: equal-weight z-score average (v2).
    If weighted=True: LASSO-coefficient-weighted z-score average (v2w).
    """
    z_parts = []
    weights = []
    for col in consensus_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if directions[col] == "invert":
            vals = -vals  # negate so higher = more need
        z = (vals - vals.mean()) / vals.std()
        z_parts.append(z)
        if weighted and lasso_coefs is not None:
            weights.append(abs(lasso_coefs.get(col, 0)))
        else:
            weights.append(1.0)

    z_matrix = pd.concat(z_parts, axis=1)
    w = np.array(weights)
    w = w / w.sum()  # normalize
    score = z_matrix.values @ w
    return pd.Series(score, index=df.index)


# Save CHNi v2 definition
v2_def = pd.DataFrame({
    "feature": [short_name(c) for c in consensus_cols],
    "column": consensus_cols,
    "direction": [directions[c] for c in consensus_cols],
    "lasso_coef": [lasso_coefs.get(c, 0) for c in consensus_cols],
    "enet_coef": [enet_coefs.get(c, 0) for c in consensus_cols],
    "rf_importance": [perm_imp.get(c, 0) for c in consensus_cols],
    "in_lasso": [c in lasso_set for c in consensus_cols],
    "in_enet": [c in enet_set for c in consensus_cols],
    "in_rf": [c in rf_set for c in consensus_cols],
})
v2_def.to_csv(OUT / "chni_v2_definition.csv", index=False)
print("-> chni_v2_definition.csv")

# Feature importance comparison figure
fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(all_selected) * 0.35)))

# Collect all features for comparison
all_feat_names = sorted(all_selected, key=lambda c: abs(lasso_coefs.get(c, 0)), reverse=True)
plot_names = [short_name(c)[:35] for c in all_feat_names]

# LASSO
ax = axes[0]
vals = [lasso_coefs.get(c, 0) for c in all_feat_names]
colors = ["#e74c3c" if c in consensus_cols else "#bdc3c7" for c in all_feat_names]
ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(plot_names, fontsize=7)
ax.set_xlabel("LASSO Coefficient")
ax.set_title("LASSO", fontweight="bold")
ax.invert_yaxis()

# Elastic Net
ax = axes[1]
vals = [enet_coefs.get(c, 0) for c in all_feat_names]
ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(plot_names, fontsize=7)
ax.set_xlabel("Elastic Net Coefficient")
ax.set_title("Elastic Net", fontweight="bold")
ax.invert_yaxis()

# RF permutation importance
ax = axes[2]
vals = [perm_imp.get(c, 0) for c in all_feat_names]
ax.barh(range(len(vals)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(plot_names, fontsize=7)
ax.set_xlabel("Permutation Importance")
ax.set_title("Random Forest", fontweight="bold")
ax.invert_yaxis()

plt.suptitle("Feature Importance Comparison\n(Red = consensus, Grey = single-method only)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "fig_feature_importance_comparison.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print("-> fig_feature_importance_comparison.png")

# ═════════════════════════════════════════════════════════════════════════
# PHASE 5: HEAD-TO-HEAD VALIDATION (v1 vs v2 vs v2w)
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PHASE 5: HEAD-TO-HEAD VALIDATION")
print("=" * 70)

# Compute all index versions for CA and NY
ca["CHNi_v1"] = compute_chni_v1(ca)
ca["CHNi_v2"] = compute_chni_v2(ca, consensus_cols, directions)
ca["CHNi_v2w"] = compute_chni_v2(ca, consensus_cols, directions,
                                  weighted=True, lasso_coefs=lasso_coefs)

ny["CHNi_v1"] = compute_chni_v1(ny)
ny["CHNi_v2"] = compute_chni_v2(ny, consensus_cols, directions)
ny["CHNi_v2w"] = compute_chni_v2(ny, consensus_cols, directions,
                                  weighted=True, lasso_coefs=lasso_coefs)

# ── Correlations ──
results = []
for label, df, target_name in [("CA (ED rate/1k)", ca, "target"),
                                ("NY (PPV rate/100)", ny, "target")]:
    for version in ["CHNi_v1", "CHNi_v2", "CHNi_v2w"]:
        mask = df[version].notna() & df[target_name].notna()
        if mask.sum() > 5:
            r, p = stats.pearsonr(df.loc[mask, version], df.loc[mask, target_name])
            rho, _ = stats.spearmanr(df.loc[mask, version], df.loc[mask, target_name])
            results.append({
                "state": label, "index": version,
                "pearson_r": r, "R2": r**2, "spearman_rho": rho,
                "p_value": p, "n": int(mask.sum()),
            })

# ── Leave-one-state-out CV ──
# Train on CA → predict NY (and vice versa)
for train_label, train_df, test_label, test_df in [
    ("CA", ca, "NY", ny), ("NY", ny, "CA", ca)
]:
    for version in ["CHNi_v2", "CHNi_v2w"]:
        # Compute scores using TRAINING state's mean/std (out-of-sample z-scoring)
        # For v2, re-z-score consensus features within training state
        z_parts_train = []
        z_parts_test = []
        weights = []
        for col in consensus_cols:
            tr_vals = pd.to_numeric(train_df[col], errors="coerce")
            te_vals = pd.to_numeric(test_df[col], errors="coerce")
            if directions[col] == "invert":
                tr_vals = -tr_vals
                te_vals = -te_vals
            tr_mean, tr_std = tr_vals.mean(), tr_vals.std()
            z_tr = (tr_vals - tr_mean) / tr_std
            z_te = (te_vals - tr_mean) / tr_std  # use train stats
            z_parts_train.append(z_tr)
            z_parts_test.append(z_te)
            if version == "CHNi_v2w":
                weights.append(abs(lasso_coefs.get(col, 0)))
            else:
                weights.append(1.0)

        w = np.array(weights)
        w = w / w.sum()

        test_z = pd.concat(z_parts_test, axis=1)
        test_score = test_z.values @ w

        mask = np.isfinite(test_score) & test_df["target"].notna().values
        if mask.sum() > 5:
            r, p = stats.pearsonr(test_score[mask], test_df.loc[mask, "target"])
            results.append({
                "state": "LOSO: train {} -> test {}".format(train_label, test_label),
                "index": version,
                "pearson_r": r, "R2": r**2,
                "spearman_rho": np.nan, "p_value": p, "n": int(mask.sum()),
            })

# ── CHR Preventable Hosp (national) ──
chr_all["prev_hosp"] = pd.to_numeric(chr_all["Preventable Hospital Stays raw value"],
                                      errors="coerce")
chr_all["CHNi_v1"] = compute_chni_v1(chr_all)
chr_all["CHNi_v2"] = compute_chni_v2(chr_all, consensus_cols, directions)
chr_all["CHNi_v2w"] = compute_chni_v2(chr_all, consensus_cols, directions,
                                       weighted=True, lasso_coefs=lasso_coefs)

for version in ["CHNi_v1", "CHNi_v2", "CHNi_v2w"]:
    mask = chr_all[version].notna() & chr_all["prev_hosp"].notna()
    if mask.sum() > 10:
        r, p = stats.pearsonr(chr_all.loc[mask, version], chr_all.loc[mask, "prev_hosp"])
        results.append({
            "state": "National (prev hosp, n~3k)",
            "index": version,
            "pearson_r": r, "R2": r**2,
            "spearman_rho": np.nan, "p_value": p, "n": int(mask.sum()),
        })

val_df = pd.DataFrame(results)
val_df.to_csv(OUT / "validation_comparison.csv", index=False)

print()
print("Head-to-Head Validation Results:")
print("{:<35s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s}".format(
    "Comparison", "Index", "r", "R2", "p", "n"))
print("-" * 85)
for _, row in val_df.iterrows():
    print("{:<35s} {:>10s} {:>+10.3f} {:>10.3f} {:>10.2e} {:>6d}".format(
        row["state"][:35], row["index"], row["pearson_r"],
        row["R2"], row["p_value"], row["n"]))

# ── 2x2 scatter: v1 vs v2 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax_idx, (state_label, df_plot, target_label) in enumerate([
    ("CA", ca, "ED rate/1k"),
    ("NY", ny, "Observed PPV rate/100"),
]):
    for ver_idx, (version, color) in enumerate([
        ("CHNi_v1", "#e74c3c"), ("CHNi_v2", "#3498db"),
    ]):
        ax = axes[ax_idx, ver_idx]
        mask = df_plot[version].notna() & df_plot["target"].notna()
        x, y_vals = df_plot.loc[mask, version], df_plot.loc[mask, "target"]
        r, p = stats.pearsonr(x, y_vals)

        ax.scatter(x, y_vals, c=color, s=50, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
        z_fit = np.polyfit(x.values, y_vals.values, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.poly1d(z_fit)(x_line), "--", color="#333", linewidth=2)
        ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}\np = {:.1e}".format(r, r**2, p),
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("{} Score".format(version), fontsize=11)
        ax.set_ylabel(target_label, fontsize=11)
        ax.set_title("{}: {} vs {}\n(n={})".format(
            state_label, version, target_label, mask.sum()),
            fontsize=12, fontweight="bold")

plt.suptitle("CHNi v1 (Hand-Picked) vs v2 (Data-Driven)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig_chni_v1_v2_validation.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print("-> fig_chni_v1_v2_validation.png")

# ── Validation summary bar chart ──
fig, ax = plt.subplots(figsize=(12, 6))

# Filter to the 4 main comparisons (CA, NY, National prev hosp)
main_comps = val_df[val_df["state"].isin([
    "CA (ED rate/1k)", "NY (PPV rate/100)", "National (prev hosp, n~3k)"
])].copy()

if not main_comps.empty:
    states = main_comps["state"].unique()
    versions = ["CHNi_v1", "CHNi_v2", "CHNi_v2w"]
    x = np.arange(len(states))
    width = 0.25
    colors = {"CHNi_v1": "#e74c3c", "CHNi_v2": "#3498db", "CHNi_v2w": "#2ecc71"}

    for i, ver in enumerate(versions):
        r_vals = []
        for s in states:
            row = main_comps[(main_comps["state"] == s) & (main_comps["index"] == ver)]
            r_vals.append(row["pearson_r"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, r_vals, width, label=ver, color=colors[ver],
                      edgecolor="white", alpha=0.85)
        for bar, rv in zip(bars, r_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    "{:.3f}".format(rv), ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in states], fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Validation: CHNi v1 vs v2 vs v2w", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(OUT / "fig_validation_summary.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.close()
print("-> fig_validation_summary.png")

# ═════════════════════════════════════════════════════════════════════════
# PHASE 6: NATIONAL MODEL (n~3000)
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PHASE 6: NATIONAL MODEL (CHR Preventable Hosp, n~3000)")
print("=" * 70)

# Use all US counties with non-null preventable hosp as target
nat_target = chr_all["prev_hosp"].copy()
nat_X = extract_features(chr_all, [c for c in feature_cols
                                    if c != "Preventable Hospital Stays raw value"])
# Keep only feature cols that exist after exclusions
nat_feature_cols = [c for c in nat_X.columns if c in keep_cols
                    and c != "Preventable Hospital Stays raw value"]
nat_X = nat_X[nat_feature_cols]

# Drop rows with missing target
valid_mask = nat_target.notna()
nat_X_valid = nat_X.loc[valid_mask].copy()
nat_y = nat_target.loc[valid_mask].values

# Drop features with >20% missing in national data
nat_miss = nat_X_valid.isnull().mean()
nat_keep = nat_miss[nat_miss < 0.20].index.tolist()
nat_X_valid = nat_X_valid[nat_keep]

print("National counties with target: {}".format(len(nat_y)))
print("National features after missingness filter: {}".format(len(nat_keep)))

# Impute + standardize
nat_imputer = SimpleImputer(strategy="median")
nat_scaler = StandardScaler()
nat_X_imp = nat_imputer.fit_transform(nat_X_valid)
nat_X_std = nat_scaler.fit_transform(nat_X_imp)

# ── National LassoCV ──
print()
print("Running National LassoCV...")
nat_lasso = LassoCV(n_alphas=100, cv=10, max_iter=10000, random_state=42)
nat_lasso.fit(nat_X_std, nat_y)
nat_lasso_coefs = pd.Series(nat_lasso.coef_, index=nat_keep)
nat_survivors = nat_lasso_coefs[nat_lasso_coefs.abs() > 1e-6].sort_values(
    key=abs, ascending=False)

print("National LassoCV alpha: {:.4f}".format(nat_lasso.alpha_))
print("National CV R2: {:.3f}".format(nat_lasso.score(nat_X_std, nat_y)))
print("National features surviving: {}/{}".format(len(nat_survivors), len(nat_keep)))
print()
print("{:<45s} {:>10s}".format("Feature", "Coef"))
print("-" * 58)
for col, coef in nat_survivors.head(25).items():
    print("{:<45s} {:>+10.4f}".format(short_name(col)[:45], coef))
if len(nat_survivors) > 25:
    print("  ... ({} more features)".format(len(nat_survivors) - 25))

nat_lasso_df = pd.DataFrame({
    "feature": [short_name(c) for c in nat_survivors.index],
    "column": nat_survivors.index,
    "national_lasso_coef": nat_survivors.values,
})
nat_lasso_df.to_csv(OUT / "national_lasso_coefficients.csv", index=False)

# ── Validate national model on CA ED and NY PPV (out-of-sample) ──
print()
print("Validating national model on CA and NY (truly out-of-sample)...")

# Score using national LASSO coefficients
nat_surv_cols = list(nat_survivors.index)

for label, df_test in [("CA (ED rate/1k)", ca), ("NY (PPV rate/100)", ny)]:
    test_X = extract_features(df_test, [c for c in nat_surv_cols if c in df_test.columns])
    available = [c for c in nat_surv_cols if c in test_X.columns]
    if not available:
        print("  {}: no overlapping features".format(label))
        continue

    # Use national mean/std for z-scoring (from nat_scaler)
    test_vals = pd.DataFrame()
    for col in available:
        test_vals[col] = pd.to_numeric(df_test[col], errors="coerce")

    # Impute with national medians, standardize with national params
    col_indices = [nat_keep.index(c) for c in available if c in nat_keep]
    test_imp = np.full((len(df_test), len(available)), np.nan)
    for j, col in enumerate(available):
        v = test_vals[col].values
        if col in nat_keep:
            idx = nat_keep.index(col)
            med = nat_imputer.statistics_[idx]
            v = np.where(np.isnan(v), med, v)
            v = (v - nat_scaler.mean_[idx]) / nat_scaler.scale_[idx]
        test_imp[:, j] = v

    # Predict using LASSO coefs
    coefs = np.array([nat_lasso_coefs[c] for c in available])
    nat_pred = test_imp @ coefs + nat_lasso.intercept_

    mask = np.isfinite(nat_pred) & df_test["target"].notna().values
    if mask.sum() > 5:
        r, p = stats.pearsonr(nat_pred[mask], df_test.loc[mask, "target"])
        print("  {}: r = {:+.3f}, R2 = {:.3f}, p = {:.2e}, n = {}".format(
            label, r, r**2, p, mask.sum()))

# ── National vs state comparison figure ──
# Compare which features rank high nationally vs in state-level analysis
state_ranks = lasso_coefs.abs().rank(ascending=False)
nat_ranks = nat_lasso_coefs.abs().rank(ascending=False)

common_feats = set(state_ranks.index) & set(nat_ranks.index)
if len(common_feats) > 5:
    fig, ax = plt.subplots(figsize=(8, 8))
    x_vals, y_vals, labels = [], [], []
    for c in common_feats:
        x_vals.append(state_ranks[c])
        y_vals.append(nat_ranks[c])
        labels.append(short_name(c))

    ax.scatter(x_vals, y_vals, s=50, c="#3498db", alpha=0.7,
               edgecolors="white", linewidth=0.5)

    # Label top features
    for i, (xv, yv, lbl) in enumerate(zip(x_vals, y_vals, labels)):
        if xv <= 10 or yv <= 10:
            ax.annotate(lbl[:25], (xv, yv), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")

    rho, p_rho = stats.spearmanr(x_vals, y_vals)
    ax.text(0.05, 0.95, "Spearman rho = {:.3f}\np = {:.2e}".format(rho, p_rho),
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    max_rank = max(max(x_vals), max(y_vals))
    ax.plot([1, max_rank], [1, max_rank], "--", color="grey", alpha=0.5)
    ax.set_xlabel("State-Level LASSO Rank (CA+NY)", fontsize=11)
    ax.set_ylabel("National LASSO Rank (n~3000)", fontsize=11)
    ax.set_title("Feature Ranking: State vs National", fontsize=13, fontweight="bold")
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT / "fig_national_vs_state.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("-> fig_national_vs_state.png")

# Save consensus features summary
consensus_df = pd.DataFrame({
    "feature": [short_name(c) for c in consensus_cols],
    "column": consensus_cols,
    "direction": [directions[c] for c in consensus_cols],
    "lasso_coef_state": [lasso_coefs.get(c, 0) for c in consensus_cols],
    "enet_coef_state": [enet_coefs.get(c, 0) for c in consensus_cols],
    "rf_importance": [perm_imp.get(c, 0) for c in consensus_cols],
    "national_lasso_coef": [nat_lasso_coefs.get(c, 0) for c in consensus_cols],
    "in_national_lasso": [c in nat_survivors.index for c in consensus_cols],
})
consensus_df.to_csv(OUT / "consensus_features.csv", index=False)

# ═════════════════════════════════════════════════════════════════════════
# PHASE 7: NORTH CAROLINA APPLICATION
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("PHASE 7: NORTH CAROLINA — CHNi v2 vs REAL ED VISITS (SHEPS 2021)")
print("=" * 70)

# ── Load NC data ──
nc_merged_path = Path("data/final/merged_county_data.csv")
if nc_merged_path.exists():
    nc_ed = pd.read_csv(nc_merged_path)
    nc_ed["Total_ED"] = pd.to_numeric(nc_ed["Total_ED"], errors="coerce")
    nc_ed["E_TOTPOP"] = pd.to_numeric(nc_ed["E_TOTPOP"], errors="coerce")
    nc_ed["ED_rate_per_1k"] = nc_ed["Total_ED"] / nc_ed["E_TOTPOP"] * 1000
    print("NC counties loaded: {}".format(len(nc_ed)))
    print("Counties with ED > 0: {}".format((nc_ed["Total_ED"] > 0).sum()))

    # Load CHR 2024 for NC (FIPS prefix "37")
    nc_chr = chr_all[chr_all["FIPS"].str[:2] == "37"].copy().reset_index(drop=True)
    nc_chr["County"] = nc_chr["Name"].str.replace(" County", "").str.strip()
    print("NC CHR counties: {}".format(len(nc_chr)))

    # Score NC with v1, v2, v2w
    nc_chr["CHNi_v1"] = compute_chni_v1(nc_chr)
    nc_chr["CHNi_v2"] = compute_chni_v2(nc_chr, consensus_cols, directions)
    nc_chr["CHNi_v2w"] = compute_chni_v2(nc_chr, consensus_cols, directions,
                                          weighted=True, lasso_coefs=lasso_coefs)

    # CHR preventable hosp for NC
    nc_chr["prev_hosp"] = pd.to_numeric(
        nc_chr["Preventable Hospital Stays raw value"], errors="coerce")

    # Merge with SHEPS ED data
    nc = nc_chr.merge(nc_ed[["County", "Total_ED", "E_TOTPOP", "ED_rate_per_1k"]],
                       on="County", how="inner")
    print("NC counties matched (CHR + SHEPS): {}".format(len(nc)))
    print()

    # ── Validate against real ED rate ──
    # Facility-based data has catchment issues: exclude counties with no ED
    # and extreme outliers (ED rate > 500/1k)
    nc_has_ed = nc[nc["Total_ED"] > 0].copy()
    nc_clean = nc_has_ed[nc_has_ed["ED_rate_per_1k"] <= 500].copy()
    print("Counties with ED facilities: {}".format(len(nc_has_ed)))
    print("Counties after outlier removal (rate <= 500/1k): {}".format(len(nc_clean)))
    print()

    nc_results = []
    for label, nc_sub in [("NC all with ED", nc_has_ed),
                           ("NC clean (rate<=500)", nc_clean)]:
        for version in ["CHNi_v1", "CHNi_v2", "CHNi_v2w"]:
            mask = nc_sub[version].notna() & nc_sub["ED_rate_per_1k"].notna()
            if mask.sum() > 5:
                r, p = stats.pearsonr(nc_sub.loc[mask, version],
                                       nc_sub.loc[mask, "ED_rate_per_1k"])
                rho, _ = stats.spearmanr(nc_sub.loc[mask, version],
                                          nc_sub.loc[mask, "ED_rate_per_1k"])
                nc_results.append({
                    "comparison": label, "index": version,
                    "pearson_r": r, "R2": r**2, "spearman_rho": rho,
                    "p_value": p, "n": int(mask.sum()),
                })

    # Also validate against CHR preventable hosp
    for version in ["CHNi_v1", "CHNi_v2", "CHNi_v2w"]:
        mask = nc_chr[version].notna() & nc_chr["prev_hosp"].notna()
        if mask.sum() > 5:
            r, p = stats.pearsonr(nc_chr.loc[mask, version],
                                   nc_chr.loc[mask, "prev_hosp"])
            nc_results.append({
                "comparison": "NC prev hosp (CHR)", "index": version,
                "pearson_r": r, "R2": r**2, "spearman_rho": np.nan,
                "p_value": p, "n": int(mask.sum()),
            })

    nc_val_df = pd.DataFrame(nc_results)
    nc_val_df.to_csv(OUT / "nc_validation.csv", index=False)

    print("NC Validation Results:")
    print("{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s}".format(
        "Comparison", "Index", "r", "R2", "p", "n"))
    print("-" * 75)
    for _, row in nc_val_df.iterrows():
        print("{:<25s} {:>10s} {:>+10.3f} {:>10.3f} {:>10.2e} {:>6d}".format(
            row["comparison"][:25], row["index"], row["pearson_r"],
            row["R2"], row["p_value"], row["n"]))

    # ── NC county scores table ──
    nc_scores = nc_chr[["FIPS", "County", "CHNi_v1", "CHNi_v2", "CHNi_v2w",
                         "prev_hosp"]].copy()

    # Add ED data where available
    ed_map = nc_ed.set_index("County")[["Total_ED", "E_TOTPOP", "ED_rate_per_1k"]]
    nc_scores = nc_scores.merge(ed_map, left_on="County", right_index=True, how="left")

    # Rank and tier by v2
    nc_scores["v2_rank"] = nc_scores["CHNi_v2"].rank(ascending=False).astype("Int64")
    nc_scores["v1_rank"] = nc_scores["CHNi_v1"].rank(ascending=False).astype("Int64")
    nc_scores["rank_change"] = nc_scores["v1_rank"] - nc_scores["v2_rank"]

    nc_scores = nc_scores.sort_values("CHNi_v2", ascending=False)
    nc_scores.to_csv(OUT / "nc_chni_v2_scores.csv", index=False)

    print()
    print("Top 15 NC counties by CHNi v2 (highest need):")
    print("{:<20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
        "County", "v2", "v1", "v2_rank", "v1_rank", "rank_chg", "ED rate"))
    print("-" * 80)
    for _, row in nc_scores.head(15).iterrows():
        ed_str = "{:.0f}".format(row["ED_rate_per_1k"]) if pd.notna(row["ED_rate_per_1k"]) else "N/A"
        print("{:<20s} {:>+8.2f} {:>+8.2f} {:>8d} {:>8d} {:>+10d} {:>10s}".format(
            row["County"][:20], row["CHNi_v2"], row["CHNi_v1"],
            row["v2_rank"], row["v1_rank"],
            row["rank_change"], ed_str))

    print()
    print("Bottom 15 NC counties by CHNi v2 (lowest need):")
    print("{:<20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
        "County", "v2", "v1", "v2_rank", "v1_rank", "rank_chg", "ED rate"))
    print("-" * 80)
    for _, row in nc_scores.tail(15).iterrows():
        ed_str = "{:.0f}".format(row["ED_rate_per_1k"]) if pd.notna(row["ED_rate_per_1k"]) else "N/A"
        print("{:<20s} {:>+8.2f} {:>+8.2f} {:>8d} {:>8d} {:>+10d} {:>10s}".format(
            row["County"][:20], row["CHNi_v2"], row["CHNi_v1"],
            row["v2_rank"], row["v1_rank"],
            row["rank_change"], ed_str))

    # ── NC figure: 2x2 (v1 vs v2 scatter against ED rate + prev hosp) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ver_idx, (version, color, ver_label) in enumerate([
        ("CHNi_v1", "#e74c3c", "v1 (Hand-Picked)"),
        ("CHNi_v2", "#3498db", "v2 (Data-Driven)"),
    ]):
        # Top row: vs SHEPS ED rate (clean)
        ax = axes[0, ver_idx]
        mask = nc_clean[version].notna() & nc_clean["ED_rate_per_1k"].notna()
        x_p, y_p = nc_clean.loc[mask, version], nc_clean.loc[mask, "ED_rate_per_1k"]
        r, p = stats.pearsonr(x_p, y_p)
        ax.scatter(x_p, y_p, c=color, s=50, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
        z_fit = np.polyfit(x_p.values, y_p.values, 1)
        x_line = np.linspace(x_p.min(), x_p.max(), 100)
        ax.plot(x_line, np.poly1d(z_fit)(x_line), "--", color="#333", linewidth=2)
        ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}\np = {:.1e}".format(r, r**2, p),
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        for _, rr in nc_clean.nlargest(3, "ED_rate_per_1k").iterrows():
            ax.annotate(rr["County"], (rr[version], rr["ED_rate_per_1k"]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("{} Score".format(ver_label), fontsize=11)
        ax.set_ylabel("SHEPS ED Rate per 1,000 Pop", fontsize=11)
        ax.set_title("NC: {} vs Real ED Rate\n(n={}, facility-based, outliers removed)".format(
            ver_label, mask.sum()), fontsize=11, fontweight="bold")

        # Bottom row: vs CHR preventable hosp
        ax = axes[1, ver_idx]
        mask = nc_chr[version].notna() & nc_chr["prev_hosp"].notna()
        x_p = nc_chr.loc[mask, version]
        y_p = nc_chr.loc[mask, "prev_hosp"]
        r, p = stats.pearsonr(x_p, y_p)
        ax.scatter(x_p, y_p, c=color, s=50, alpha=0.7,
                   edgecolors="white", linewidth=0.5)
        z_fit = np.polyfit(x_p.values, y_p.values, 1)
        x_line = np.linspace(x_p.min(), x_p.max(), 100)
        ax.plot(x_line, np.poly1d(z_fit)(x_line), "--", color="#333", linewidth=2)
        ax.text(0.05, 0.95, "r = {:.3f}\nR2 = {:.3f}\np = {:.1e}".format(r, r**2, p),
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("{} Score".format(ver_label), fontsize=11)
        ax.set_ylabel("CHR Preventable Hosp Rate", fontsize=11)
        ax.set_title("NC: {} vs Preventable Hosp\n(n={})".format(
            ver_label, mask.sum()), fontsize=11, fontweight="bold")

    plt.suptitle("North Carolina: CHNi v1 vs v2 Against Real Outcomes",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig_nc_v1_v2_validation.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print()
    print("-> fig_nc_v1_v2_validation.png")
    print("-> nc_chni_v2_scores.csv")
    print("-> nc_validation.csv")

    # ── Biggest rank movers ──
    print()
    print("Biggest rank changes (v1 -> v2):")
    movers = nc_scores.dropna(subset=["rank_change"])
    top_up = movers.nlargest(5, "rank_change")
    top_down = movers.nsmallest(5, "rank_change")
    print("  Moved UP in need ranking (v2 says higher need than v1):")
    for _, row in top_up.iterrows():
        print("    {:<20s} v1_rank={:>3d} -> v2_rank={:>3d} (change {:+d})".format(
            row["County"][:20], row["v1_rank"], row["v2_rank"], row["rank_change"]))
    print("  Moved DOWN in need ranking (v2 says lower need than v1):")
    for _, row in top_down.iterrows():
        print("    {:<20s} v1_rank={:>3d} -> v2_rank={:>3d} (change {:+d})".format(
            row["County"][:20], row["v1_rank"], row["v2_rank"], row["rank_change"]))

    # ═════════════════════════════════════════════════════════════════════
    # REGIONAL ANALYSIS + IMPUTATION FOR MISSING COUNTIES
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("NC REGIONAL ANALYSIS")
    print("=" * 70)

    # Import region mappings
    import sys
    sys.path.insert(0, str(Path("helpers")))
    from dictionaries import county_to_region

    nc_scores["Region"] = nc_scores["County"].map(county_to_region)

    # Identify missing counties
    missing = nc_scores[nc_scores["CHNi_v2"].isna()].copy()
    scored = nc_scores[nc_scores["CHNi_v2"].notna()].copy()

    if len(missing) > 0:
        print()
        print("Counties missing CHNi v2 ({}) — imputing from regional median:".format(
            len(missing)))
        for _, row in missing.iterrows():
            region = row["Region"]
            region_median = scored.loc[scored["Region"] == region, "CHNi_v2"].median()
            region_median_w = scored.loc[scored["Region"] == region, "CHNi_v2w"].median()
            nc_scores.loc[nc_scores["County"] == row["County"], "CHNi_v2"] = region_median
            nc_scores.loc[nc_scores["County"] == row["County"], "CHNi_v2w"] = region_median_w
            nc_scores.loc[nc_scores["County"] == row["County"], "v2_imputed"] = True
            print("  {:<20s} ({:<15s}) -> v2 = {:+.2f} (regional median)".format(
                row["County"], region, region_median))

        # Re-rank after imputation
        nc_scores["v2_rank"] = nc_scores["CHNi_v2"].rank(ascending=False).astype(int)
        nc_scores["rank_change"] = nc_scores["v1_rank"] - nc_scores["v2_rank"]
    else:
        nc_scores["v2_imputed"] = False

    nc_scores["v2_imputed"] = nc_scores["v2_imputed"].fillna(False)

    # ── Regional summary ──
    print()
    print("Regional Summary (mean CHNi v2, scored counties only):")
    print("{:<18s} {:>5s} {:>10s} {:>10s} {:>10s} {:>12s} {:>10s}".format(
        "Region", "n", "v2 mean", "v2 std", "v1 mean", "v2-v1 shift", "ED rate"))
    print("-" * 80)

    region_order = ["West", "North Central", "Piedmont", "South Central", "East"]
    region_stats = []
    for region in region_order:
        r_data = nc_scores[nc_scores["Region"] == region]
        r_scored = r_data[~r_data["v2_imputed"]]
        n = len(r_data)
        v2_mean = r_data["CHNi_v2"].mean()
        v2_std = r_data["CHNi_v2"].std()
        v1_mean = r_data["CHNi_v1"].mean()
        ed_rate = r_data["ED_rate_per_1k"].replace(0, np.nan).mean()
        shift = v2_mean - v1_mean
        region_stats.append({
            "Region": region, "n": n,
            "v2_mean": v2_mean, "v2_std": v2_std,
            "v1_mean": v1_mean, "shift": shift,
            "ed_rate_mean": ed_rate,
        })
        ed_str = "{:.0f}".format(ed_rate) if pd.notna(ed_rate) else "N/A"
        print("{:<18s} {:>5d} {:>+10.3f} {:>10.3f} {:>+10.3f} {:>+12.3f} {:>10s}".format(
            region, n, v2_mean, v2_std, v1_mean, shift, ed_str))

    region_df = pd.DataFrame(region_stats)
    region_df.to_csv(OUT / "nc_regional_summary.csv", index=False)

    # Re-save with imputed values and region
    nc_scores = nc_scores.sort_values("CHNi_v2", ascending=False)
    nc_scores.to_csv(OUT / "nc_chni_v2_scores.csv", index=False)

    # ── Regional figure: grouped bar (v1 vs v2 by region) + component heatmap ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: v1 vs v2 by region
    ax = axes[0]
    x_pos = np.arange(len(region_order))
    width = 0.35
    v1_means = [region_df.loc[region_df["Region"] == r, "v1_mean"].values[0]
                for r in region_order]
    v2_means = [region_df.loc[region_df["Region"] == r, "v2_mean"].values[0]
                for r in region_order]

    bars1 = ax.bar(x_pos - width/2, v1_means, width, label="CHNi v1",
                   color="#e74c3c", edgecolor="white", alpha=0.85)
    bars2 = ax.bar(x_pos + width/2, v2_means, width, label="CHNi v2",
                   color="#3498db", edgecolor="white", alpha=0.85)

    for bar, val in zip(bars1, v1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                "{:+.2f}".format(val), ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, v2_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                "{:+.2f}".format(val), ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(region_order, fontsize=10)
    ax.set_ylabel("Mean CHNi Score", fontsize=11)
    ax.set_title("NC Regional Need: v1 vs v2", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=10)

    # Panel 2: v2 component heatmap by region
    ax = axes[1]
    comp_means = []
    comp_labels = []
    for col in consensus_cols:
        vals = pd.to_numeric(nc_chr[col], errors="coerce")
        if directions[col] == "invert":
            vals = -vals
        z = (vals - vals.mean()) / vals.std()
        nc_chr["_z_" + col] = z
        comp_labels.append(short_name(col))

    heatmap_data = []
    for region in region_order:
        region_counties = [c for c, r in county_to_region.items() if r == region]
        region_chr = nc_chr[nc_chr["County"].isin(region_counties)]
        row = []
        for col in consensus_cols:
            row.append(region_chr["_z_" + col].mean())
        heatmap_data.append(row)

    hm = np.array(heatmap_data)
    im = ax.imshow(hm, cmap="RdYlGn_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_xticks(range(len(comp_labels)))
    ax.set_xticklabels(comp_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(region_order, fontsize=10)

    # Annotate cells
    for i in range(len(region_order)):
        for j in range(len(comp_labels)):
            val = hm[i, j]
            color = "white" if abs(val) > 0.8 else "black"
            ax.text(j, i, "{:+.2f}".format(val), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean z-score (higher = more need)", shrink=0.8)
    ax.set_title("v2 Component Profile by Region", fontsize=13, fontweight="bold")

    plt.suptitle("North Carolina: Regional Patterns in CHNi v2",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "fig_nc_regional.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print()
    print("-> fig_nc_regional.png")
    print("-> nc_regional_summary.csv")

    # Clean up temp columns
    for col in consensus_cols:
        nc_chr.drop("_z_" + col, axis=1, inplace=True, errors="ignore")

else:
    print("NC merged data not found at {}".format(nc_merged_path))
    print("Skipping NC application phase.")

# ═════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print("CHNi v1 (hand-picked): {} variables".format(len(V1_COLS)))
for c in V1_COLS:
    print("  - {}".format(short_name(c)))
print()
print("CHNi v2 (data-driven): {} consensus variables".format(len(consensus_cols)))
for c in consensus_cols:
    print("  - {} ({})".format(short_name(c), directions[c]))
print()

# Quick summary of r values
print("Validation (Pearson r):")
for _, row in val_df[val_df["state"].isin([
    "CA (ED rate/1k)", "NY (PPV rate/100)"])].iterrows():
    print("  {:<35s} {} : r = {:+.3f}".format(
        row["state"], row["index"], row["pearson_r"]))

print()
print("Output files saved to: {}".format(OUT))
print()
csv_files = sorted(OUT.glob("*.csv"))
fig_files = sorted(OUT.glob("*.png"))
print("CSVs ({})".format(len(csv_files)))
for f in csv_files:
    print("  {}".format(f.name))
print("Figures ({})".format(len(fig_files)))
for f in fig_files:
    print("  {}".format(f.name))
print()
print("=" * 70)
print("DATA-DRIVEN CHNi v2 COMPLETE")
print("=" * 70)
