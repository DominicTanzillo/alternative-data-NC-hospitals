# Identifying Underserved Healthcare Counties Using Alternative Data Sources

A reproducible pipeline for identifying counties with excess emergency department (ED) burden using publicly available, non-traditional ("alternative") data sources. Applied to North Carolina's 100 counties as a proof of concept, with a design generalizable to any US state.

## Research Question

**Can publicly available alternative data sources predict county-level emergency department utilization, and can the model residuals identify systematically underserved communities?**

## What Makes This "Alternative Data"?

Traditional healthcare needs assessments rely on proprietary claims data, provider surveys, and patient-reported outcomes — all expensive, slow, and difficult to access. This project uses exclusively **publicly available, nationally standardized datasets** that are updated regularly and cover every US county:

| Data Source | What It Captures | Update Frequency |
|---|---|---|
| **CMS Medicare/Medicaid Enrollment** | Insurance coverage landscape | Monthly |
| **CDC Social Vulnerability Index (SVI)** | Socioeconomic risk factors (4 themes) | Biennial |
| **CDC PLACES** | Chronic disease prevalence (age-adjusted) | Annual |
| **HRSA Area Health Resource File** | Healthcare workforce & infrastructure | Annual |
| **USDA Rural-Urban Codes** | Rurality classification | Periodic |

The ED visit data (target variable) comes from **NC DETECT**, North Carolina's syndromic surveillance system. For other states, equivalent data is available through state health departments or HCUP's State Inpatient/ED Databases.

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
│   │   └── pipeline.py          # Main analysis pipeline
│   ├── database_read/
│   │   ├── ED_Visits.py         # PDF extraction for NC DETECT data
│   │   ├── load_sedd.py         # HCUP SEDD county aggregation
│   │   └── read_cdc.py          # CDC data processing
│   └── graph_making/
│       └── nc_map.py            # Choropleth map generation
├── data/
│   ├── final/
│   │   ├── merged_county_data.csv  # Primary merged dataset (100 x 300)
│   │   ├── need_scores/            # Composite need score outputs
│   │   └── model/                  # Legacy model outputs
│   ├── AHRF/                       # HRSA Area Health Resource File
│   ├── MA/                         # CMS Medicare Advantage enrollment
│   ├── ED_visits/                  # NC DETECT ED visit PDFs
│   ├── SVI/                        # CDC Social Vulnerability Index
│   ├── places/                     # CDC PLACES chronic disease data
│   └── NC Medicaid Reports/        # State Medicaid enrollment
├── helpers/
│   └── dictionaries.py             # County-to-region mappings
├── results/                        # Pipeline outputs (figures, CSVs)
├── notebooks/
│   ├── dominicworkbook.ipynb       # Data merging & need scores
│   └── andrewworkbook.ipynb        # EDA & initial model comparison
├── requirements.txt
└── README.md
```

## Reproducing the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.model.pipeline
```

Outputs are saved to `results/`:
- `cv_results.csv` — Cross-validation results for all models
- `holdout_model_a.csv` — Holdout test set performance
- `bootstrap_ci.csv` — 95% confidence intervals
- `excess_burden_scores.csv` — Per-county excess burden scores
- `underserved_map.csv` — Full underserved classification (all 100 counties)
- `feature_importance.csv` — Permutation importance rankings
- `parsimony.csv` — Feature count vs. model performance
- `fig_*.png` — Publication-ready figures

## Generalizability

This pipeline is designed to be state-agnostic. To apply it to another state:

**Option A: Using HCUP SEDD (recommended for publication)**
1. Obtain HCUP SEDD data through a Data Use Agreement ([AHRQ HCUP](https://hcup-us.ahrq.gov/tech_assist/dua.jsp))
2. Export to CSV and aggregate by patient residence county:
   ```bash
   python -m src.database_read.load_sedd data/SEDD/nc_sedd.csv \
       --year 2021 --merge data/final/merged_county_data.csv
   ```
3. Download predictor datasets (all nationally available) and merge
4. Run `python -m src.model.pipeline`

**Option B: Using state syndromic surveillance data**
1. Obtain county-level ED visit counts (state health department or HCUP SID)
2. Download the predictor datasets (all nationally available):
   - CMS MA/MC enrollment by county
   - CDC SVI by county
   - CDC PLACES by county
   - HRSA AHRF by county
3. Merge into the same schema as `merged_county_data.csv`
4. Run `python -m src.model.pipeline`

## Data Sources

- **CMS Medicare Advantage Enrollment**: [data.cms.gov](https://data.cms.gov)
- **CMS Medicare Geographic Variation**: [data.cms.gov](https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons)
- **CDC Social Vulnerability Index**: [atsdr.cdc.gov](https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html)
- **CDC PLACES**: [cdc.gov/places](https://www.cdc.gov/places/)
- **HRSA Area Health Resource File**: [data.hrsa.gov](https://data.hrsa.gov/data/download)
- **NC DETECT**: [ncdetect.org](https://ncdetect.org/)
- **NC Medicaid Enrollment**: [medicaid.ncdhhs.gov](https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports)

## Requirements

Python 3.9+ with scikit-learn, pandas, numpy, matplotlib. See `requirements.txt` for pinned versions.
