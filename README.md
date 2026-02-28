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
| **CDC PLACES** | Chronic disease prevalence (40 indicators) | Annual | 2020-2025 releases |
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

## Project Structure

```
.
├── src/
│   ├── model/
│   │   ├── pipeline.py          # Cross-sectional analysis pipeline
│   │   └── forecast.py          # Temporal forecasting & risk tiers
│   ├── data_acquisition/
│   │   ├── build_panel.py       # Multi-year county panel builder
│   │   └── README_DATA_SOURCES.md  # Data procurement guide
│   ├── database_read/
│   │   ├── ED_Visits.py         # PDF extraction for NC DETECT data
│   │   ├── load_sedd.py         # HCUP SEDD county aggregation
│   │   └── read_cdc.py          # CDC data processing
│   └── graph_making/
│       └── nc_map.py            # Choropleth map generation
├── data/                        # NOT tracked in git (see .gitignore)
│   ├── raw/                     # Untouched downloads by source
│   ├── processed/               # Cleaned county-year panels
│   └── final/                   # Pipeline-ready merged datasets
├── helpers/
│   └── dictionaries.py          # County-to-region mappings
├── results/                     # Cross-sectional pipeline outputs
│   └── forecast/                # Temporal forecast outputs
├── notebooks/
│   ├── analysis.ipynb           # Publication walkthrough
│   ├── dominicworkbook.ipynb    # Data merging & need scores
│   └── andrewworkbook.ipynb     # EDA & initial model comparison
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

# Run the full pipeline
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
# Build county x year panel from downloaded data sources
python -m src.data_acquisition.build_panel --state NC --years 2015-2023

# Train temporal model and generate forecasts
python -m src.model.forecast --panel data/processed/panel_nc.csv
```

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
| **1. Cross-Section** | Single year (2021), 100 NC counties | Excess burden map, model comparison | Ready |
| **2. Panel** | 2015-2023, 100 counties x 9 years | Temporal trends, year-over-year changes | Awaiting SEDD |
| **3. Forecast** | Train 2015-2021, validate 2022-2023 | Forward projections, county risk tiers | Awaiting SEDD |
| **4. Replication** | SC panel (2015-2023, 46 counties) | Cross-state validation | Awaiting SEDD |

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
| HCUP SEDD | [hcup-us.ahrq.gov](https://hcup-us.ahrq.gov/seddoverview.jsp) | NC: 2007-2023 (DUA required) |

Full data procurement guide: `src/data_acquisition/README_DATA_SOURCES.md`

## Requirements

Python 3.9+ with scikit-learn, pandas, numpy, matplotlib. See `requirements.txt` for pinned versions.
