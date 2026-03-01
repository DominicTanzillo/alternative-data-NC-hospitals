# Alternative Data as Predictors of Healthcare Quality

A reproducible pipeline that uses publicly available "alternative" data to predict county-level emergency department burden, identify underserved communities, and forecast which counties need intervention -- without requiring proprietary claims data. Applied to North Carolina's 100 counties with a design generalizable to any US state.

## Research Question

**Can publicly available alternative data sources serve as leading indicators of healthcare quality at the county level, enabling proactive identification of communities trending toward crisis?**

## Why This Matters

Healthcare quality assessment traditionally requires expensive proprietary data (claims, EMRs, provider surveys) that is slow to obtain, limited in geographic scope, and difficult to replicate. By the time conventional analyses identify underserved communities, the crisis is already entrenched.

This project demonstrates that **freely available federal datasets** -- updated monthly to annually -- can predict ED utilization patterns, identify excess burden, and forecast future deterioration. The same predictors are available for every US county, making this a scalable, reproducible framework for any state or region.

## What Makes This "Alternative Data"?

We use the term "alternative data" in the same sense as finance: non-traditional data sources that provide actionable signals ahead of conventional indicators. Every predictor in this model is publicly available, nationally standardized, and updated regularly:

| Data Source | What It Captures | Update Frequency | Years Available |
|---|---|---|---|
| **CMS Medicare/Medicaid Enrollment** | Insurance coverage landscape | Monthly | 2007-present |
| **CDC Social Vulnerability Index (SVI)** | Socioeconomic risk factors (4 themes) | Biennial | 2010-2022 |
| **CDC PLACES** | Chronic disease prevalence (40+ indicators) | Annual | 2020-2025 releases (county data years 2017-2023) |
| **HRSA Area Health Resource File** | Healthcare workforce & infrastructure | Annual | 2010-present |
| **BLS Local Area Unemployment** | Economic conditions | Monthly | 1990-present |
| **Census SAIPE** | Poverty and income | Annual | 1995-present |
| **FCC Broadband Deployment** | Digital/telehealth access | Semi-annual | 2009-present |
| **County Health Rankings** | Composite health outcomes | Annual | 2010-present |
| **USDA Rural-Urban Codes** | Rurality classification | Periodic | 2003-2023 |

The ED visit data (target variable) uses **HCUP SEDD** (State Emergency Department Databases), which provides residence-based visit counts by patient county of residence. For the current proof-of-concept, facility-based data from NC DETECT is used.

## Methodology

### Dual-Model Design

**Model A — Expected Utilization Model** (primary)
- Target: `log(Total ED visits)`
- Includes `log(Population)` as an offset variable
- Answers: *After controlling for population, which alternative data features explain remaining ED volume variation?*
- Best model: Elastic Net (CV R-squared = 0.61)

**Model B — Rate-Based Structural Model** (diagnostic)
- Target: ED visits per 1,000 population
- No population feature (already normalized)
- Answers: *Can sociodemographic features alone predict ED utilization rates?*
- Result: All models show negative CV R-squared, meaning structural factors alone cannot predict ED rates

The negative R-squared in Model B is itself the key finding: it demonstrates that **local healthcare system factors** (hospital catchment areas, proximity to urgent care, primary care access) dominate ED utilization patterns more than demographics or disease burden.

### Excess Burden Identification

Excess ED burden is computed from Model A residuals using Leave-One-Out cross-validation:

```
Excess Ratio = exp(Observed log(ED) - Predicted log(ED))
             = Observed ED visits / Expected ED visits
```

Counties are classified as:
- **Excess Burden**: Excess ratio z-score > 1.0 (significantly more ED visits than expected)
- **Expected**: Within 1 standard deviation of expected
- **Healthcare Desert**: No ED facility in the county (18 of 100 NC counties)

### Validation

- 80/20 stratified train/test split (stratified by region)
- 10-fold cross-validation on training set
- Leave-One-Out CV for county-level excess burden (unbiased per-county estimates)
- Bootstrap confidence intervals (1,000 resamples)
- 8 models compared: OLS, Ridge, Lasso, Elastic Net, KNN, SVR, Random Forest, Gradient Boosting

## Community Health Need Index (CHNi)

A zero-parameter composite index that quantifies county-level healthcare need using only publicly available data. Two versions exist:

### CHNi v1 (Hand-Picked)

Five epidemiologically-motivated variables, equal-weight z-score average:

| Component | Source | Direction |
|-----------|--------|-----------|
| Diabetes prevalence | CDC PLACES | Higher = more need |
| Life expectancy | County Health Rankings | Lower = more need |
| PCP rate (pop-to-provider) | County Health Rankings | Higher = more need (shortage) |
| Physical inactivity | County Health Rankings | Higher = more need |
| Poverty rate | Census SAIPE | Higher = more need |

### CHNi v2 (Data-Driven)

Rather than hand-picking variables, we let real ED outcome data select them. Using residence-based ED visits from California (HCAI) and Potentially Preventable Visit rates from New York (SPARCS) as ground truth, three methods (LASSO, Elastic Net, Random Forest) screened 72 CHR candidate features. Six consensus variables emerged:

| Component | Methods Selected By | Direction |
|-----------|-------------------|-----------|
| Food Insecurity | LASSO + Elastic Net + RF | Higher = more need |
| Teen Births | LASSO + Elastic Net + RF | Higher = more need |
| Median Household Income | LASSO + Elastic Net + RF | Lower = more need |
| Frequent Mental Distress | Elastic Net + RF | Higher = more need |
| Voter Turnout | Elastic Net + RF | Lower = more need |
| Suicides | Elastic Net + RF | Higher = more need |

### Validation Results

| Target | v1 (hand-picked) | v2 (data-driven) |
|--------|-----------------|------------------|
| CA residence-based ED rate | r = 0.35 | **r = 0.60** |
| NY residence-based PPV rate | r = 0.60 | r = 0.59 / **0.65** (weighted) |
| Leave-one-state-out (CA->NY) | -- | r = 0.60 |
| Leave-one-state-out (NY->CA) | -- | r = 0.62 |

v2 nearly doubles v1's correlation with California ED visits while matching performance on New York. Leave-one-state-out cross-validation confirms generalization.

### Policy Implications

The v2 variables are all actionable policy levers -- food insecurity, teen births, mental distress, and suicide rates are targets for intervention programs, while voter turnout and median income reflect civic engagement and economic opportunity. In North Carolina, v2 reveals a Western mountain "despair" pattern (high suicides, mental distress, low voter turnout) invisible to v1, while confirming the Eastern rural and South Central corridors as high-need.

### Running the Analysis

```bash
# Full data-driven CHNi pipeline (6 phases + NC regional analysis)
python scripts/data_driven_chni.py

# Interactive US county choropleth map (requires plotly)
python scripts/map_chni_v2.py
```

Outputs are saved to `results/chni/data_driven/` including CSVs, validation figures, and an interactive HTML map of all ~3,100 US counties.

## Key Findings

### Model Performance

| Model | CV R-squared (Model A) | Overfit Gap |
|---|---|---|
| Elastic Net | 0.608 +/- 0.296 | 0.227 |
| Lasso | 0.607 +/- 0.302 | 0.228 |
| Ridge | 0.544 +/- 0.307 | 0.342 |
| Random Forest | 0.510 +/- 0.300 | 0.407 |

Linear regularized models outperform tree-based models, consistent with the small sample size (82 counties with ED facilities).

### Top Predictors (Permutation Importance)

1. **log(Population)** — dominant predictor (0.416)
2. **Poor mental health prevalence** (CDC PLACES) — 0.053
3. **Urban Influence Code** (USDA) — 0.041
4. **% mobile homes** (ACS/SVI) — 0.038
5. **log(Medicare enrollees)** (CMS) — 0.033

### Underserved Counties Identified

- **3 counties** with statistically significant excess ED burden (Scotland, Hoke, Mecklenburg)
- **18 counties** classified as healthcare deserts (no ED facility)
- **21 of 100** NC counties flagged as underserved

### Important Methodological Note

ED visit data from NC DETECT counts visits **at hospitals in a county**, not visits **by residents of a county**. Counties with regional medical centers (e.g., Hoke with 1,354 ED visits per 1,000 population) serve patients from surrounding counties without hospitals. This catchment-area effect is an inherent feature of facility-level data and should be interpreted accordingly.

### Planned Upgrade: HCUP SEDD (Residence-Based ED Counts)

The pipeline includes a data loader (`src/database_read/load_sedd.py`) for HCUP State Emergency Department Databases (SEDD), which provide ED visits by **patient county of residence** (PSTCO FIPS code) rather than hospital location. This resolves the catchment-area problem entirely.

- **Data access**: HCUP SEDD requires a Data Use Agreement through AHRQ (available via Duke University)
- **State coverage**: NC (2007-2023), SC (2006-2023); VA is not available in SEDD
- **Usage**: `python -m src.database_read.load_sedd data/SEDD/nc_sedd.csv --year 2021`

## Data Coverage Matrix (2015-2023)

The panel builder downloads and assembles data from multiple public sources. Coverage varies by source:

| Data Source | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | Download |
|---|---|---|---|---|---|---|---|---|---|---|
| Census SAIPE (poverty, income) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Auto (API) |
| BLS LAUS (unemployment) | Y | Y | Y | Y | Y | Y | Y | Y | Y | Auto (API) |
| CDC SVI (social vulnerability) | cf | Y | cf | Y | cf | Y | cf | Y | cf | Auto (CSV) |
| CDC PLACES (chronic disease) | - | - | Y* | Y | Y | Y | Y | Y | Y | Auto (SODA) |
| County Health Rankings | - | - | Y | Y | Y | Y | Y | Y | Y | Auto (CSV) |
| CMS MA Enrollment | Y | Y | Y | Y | Y | Y | Y | Y | Y | Auto (ZIP) |
| NC Medicaid | Y | Y | Y | Y | Y | Y | Y | Y | Y | Auto (XLSX) |
| HCUP SEDD (target: ED visits) | D | D | D | D | D | D | D | D | D | DUA |

**Legend:** Y = available, D = DUA required, cf = carry-forward from prior SVI release, - = not available for this year, Y* = partial measures only (subset of BRFSS indicators)

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── pipeline.py              # Cross-sectional analysis pipeline
│   │   └── forecast.py              # Temporal forecasting & risk tiers
│   ├── data_acquisition/
│   │   ├── download_public_data.py  # Automated data downloader
│   │   ├── build_panel.py           # Multi-year county panel builder
│   │   └── README_DATA_SOURCES.md   # Data procurement guide
│   ├── database_read/
│   │   ├── ED_Visits.py             # PDF extraction for NC DETECT data
│   │   ├── load_sedd.py             # HCUP SEDD county aggregation
│   │   └── read_cdc.py              # CDC data processing
│   └── graph_making/
│       └── nc_map.py                # Choropleth map generation
├── data/                            # NOT tracked in git (see .gitignore)
│   ├── raw/                         # Untouched downloads by source
│   │   ├── saipe/                   #   Census SAIPE
│   │   ├── bls_laus/                #   BLS unemployment
│   │   ├── places/                  #   CDC PLACES
│   │   ├── chr/                     #   County Health Rankings
│   │   ├── svi/                     #   CDC SVI
│   │   ├── ma_enrollment/           #   CMS MA enrollment
│   │   └── medicaid_nc/             #   NC Medicaid
│   ├── processed/                   # Cleaned county-year panels
│   └── final/                       # Pipeline-ready merged datasets
├── helpers/
│   └── dictionaries.py              # County-to-region mappings
├── scripts/
│   ├── data_driven_chni.py          # Data-driven CHNi v2 (6-phase pipeline)
│   ├── map_chni_v2.py               # Interactive US county choropleth
│   ├── validate_chni_ca_residence.py # CA residence-based ED validation
│   ├── validate_chni_ny_sparcs.py   # NY SPARCS PPV validation
│   └── validate_chni_real_ed.py     # NC SHEPS ED validation
├── results/                         # Cross-sectional pipeline outputs
│   ├── forecast/                    # Temporal forecast outputs
│   └── chni/                        # CHNi scores, figures, validation
│       ├── out_of_state/            #   Multi-state validation (CA, NY, FL)
│       └── data_driven/             #   CHNi v2 analysis & interactive map
├── notebooks/
│   ├── analysis.ipynb               # Publication walkthrough
│   ├── dominicworkbook.ipynb        # Data merging & need scores
│   └── andrewworkbook.ipynb         # EDA & initial model comparison
├── requirements.txt
└── README.md
```

**Note:** Raw data files are not included in this repository due to size
and licensing constraints. See `src/data_acquisition/README_DATA_SOURCES.md`
for download instructions and URLs for every data source.

## Reproducing the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full cross-sectional pipeline (Phase 1)
python -m src.model.pipeline
```

Outputs are saved to `results/`:
- `cv_results.csv` -- Cross-validation results for all models
- `holdout_model_a.csv` -- Holdout test set performance
- `bootstrap_ci.csv` -- 95% confidence intervals
- `excess_burden_scores.csv` -- Per-county excess burden scores
- `underserved_map.csv` -- Full underserved classification (all 100 counties)
- `feature_importance.csv` -- Permutation importance rankings
- `parsimony.csv` -- Feature count vs. model performance
- `fig_*.png` -- Publication-ready figures

### Phase 2: Multi-Year Panel Analysis

```bash
# Step 1: Download all public data (fully automated)
python -m src.data_acquisition.download_public_data

# Step 2: Build county x year panel (100 counties x 9 years = 900 obs)
python -m src.data_acquisition.build_panel --state NC --years 2015-2023

# Step 3a: Analyze optimal year ranges given feature coverage
python -m src.model.pipeline --panel data/processed/panel_nc.csv --analyze

# Step 3b: Run multi-year pipeline (auto-selects features by coverage)
python -m src.model.pipeline --panel data/processed/panel_nc.csv --year-range 2018-2023

# Step 3c: Run single-year cross-section from the panel
python -m src.model.pipeline --panel data/processed/panel_nc.csv --year 2021

# Step 4: Train temporal model and generate forecasts
python -m src.model.forecast --panel data/processed/panel_nc.csv
```

**Year Range Tradeoffs:**

| Range | Years | Obs | Features (>=80%) | Notes |
|---|---|---|---|---|
| 2015-2023 | 9 | 900 | 24 | Max depth, structural features only |
| 2017-2023 | 7 | 700 | 54 | **Recommended** (max years x features) |
| 2018-2023 | 6 | 600 | 58 | Best balance with full PLACES coverage |
| 2020-2023 | 4 | 400 | 60 | Max features, fewer observations |

### Phase 3: Forward Projection (Government Deliverable)

The forecasting module trains on historical data (2015-2021) and validates
on held-out years (2022-2023). Because all predictor data is released
6-12 months before ED visit data, the model can project excess burden
forward to identify counties trending toward crisis -- without waiting
for the next year's ED data.

Outputs:
- `results/forecast/county_risk_trends.csv` -- Priority-ranked counties
- `results/forecast/fig_priority_matrix.png` -- Current burden vs. trajectory
- `results/forecast/fig_risk_tiers_over_time.png` -- Tier shifts over time

## Phased Research Design

| Phase | Data | Output | Status |
|-------|------|--------|--------|
| **1. Cross-Section** | Single year (2021), 100 NC counties | Excess burden map, model comparison | Complete |
| **2. Panel** | 2015-2023, 100 counties x 9 years | Temporal trends, year-over-year changes | Awaiting SEDD |
| **3. Forecast** | Train 2015-2021, validate 2022-2023 | Forward projections, county risk tiers | Awaiting SEDD |
| **4. Replication** | SC panel (2015-2023, 46 counties) | Cross-state validation | Awaiting SEDD |
| **5. CHNi** | CA + NY residence-based ED data | Data-driven need index, interactive map | Complete |

The key insight of Phase 3: **even without future ED data, we can identify
which counties are projected to worsen** based on the trajectory of their
publicly available predictor variables. This makes the model a leading
indicator of healthcare quality rather than a lagging one.

## Generalizability

This pipeline is designed to be state-agnostic. Every predictor is a
nationally standardized federal dataset available for all US counties.

To apply to another state:
1. Obtain county-level ED visit counts (HCUP SEDD recommended, or state
   health department data)
2. Download predictor datasets (all publicly available -- see
   `src/data_acquisition/README_DATA_SOURCES.md` for URLs)
3. Build the panel: `python -m src.data_acquisition.build_panel --state SC`
4. Run the pipeline: `python -m src.model.pipeline`

**HCUP SEDD availability:** NC (2007-2023), SC (2006-2023). VA does not
participate in SEDD. Data requires a DUA through
[AHRQ](https://hcup-us.ahrq.gov/tech_assist/dua.jsp).

## Data Sources

| Source | URL | Coverage |
|--------|-----|----------|
| CMS Medicare Advantage | [cms.gov](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/monthly-enrollment-contract/plan/state/county) | 2007-present, monthly |
| CMS Medicare Geographic Variation | [data.cms.gov](https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county) | 2007-present, annual |
| NC Medicaid Enrollment | [medicaid.ncdhhs.gov](https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports) | SFY 1998-2026 |
| CDC Social Vulnerability Index | [atsdr.cdc.gov](https://atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html) | 2010-2022, biennial |
| CDC PLACES | [data.cdc.gov](https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb) | 2020-2025 releases |
| HRSA AHRF | [data.hrsa.gov](https://data.hrsa.gov/data/download) | 2019-2024 releases |
| BLS Unemployment (LAUS) | [bls.gov](https://www.bls.gov/lau/data.htm) | 1990-present, monthly |
| Census SAIPE | [census.gov](https://www.census.gov/programs-surveys/saipe/data/datasets.html) | 1995-present, annual |
| County Health Rankings | [countyhealthrankings.org](https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation) | 2010-2025, annual |
| FCC Broadband | [fcc.gov](https://www.fcc.gov/form-477-county-data-internet-access-services) | 2009-2024, semi-annual |
| CA HCAI ED Visits | [data.chhs.ca.gov](https://data.chhs.ca.gov/) | 2008-2024, by patient county of residence |
| NY SPARCS PPV Rates | [health.data.ny.gov](https://health.data.ny.gov/) | 2011-2023, by patient county of residence |
| HCUP SEDD | [hcup-us.ahrq.gov](https://hcup-us.ahrq.gov/seddoverview.jsp) | NC: 2007-2023 (DUA required) |

Full data procurement guide: `src/data_acquisition/README_DATA_SOURCES.md`

## Requirements

Python 3.9+ with scikit-learn, pandas, numpy, matplotlib. See `requirements.txt` for pinned versions.
