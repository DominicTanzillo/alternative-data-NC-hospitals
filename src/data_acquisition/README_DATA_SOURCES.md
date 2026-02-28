# Data Acquisition Guide
## Alternative Data for Predicting Healthcare Quality

This document specifies every data source, its year coverage, download URL,
and the exact steps to acquire it. The goal: build a **county x year panel
dataset** (2010-2023) that predicts ED burden and healthcare quality using
only publicly available, nationally standardized "alternative" data.

---

## Panel Design

**Target window:** 2010-2023 (aligned to HCUP SEDD availability for NC)
**Geographic unit:** County (FIPS code)
**States:** NC (primary), SC (replication)
**Observations:** 100 counties x 14 years = 1,400 county-year panels (NC)

---

## Data Sources

### 1. ED Visits -- HCUP SEDD (Target Variable)

| Field | Value |
|-------|-------|
| Source | AHRQ Healthcare Cost and Utilization Project |
| URL | https://hcup-us.ahrq.gov/seddoverview.jsp |
| Years | NC: 2007-2023, SC: 2006-2023 |
| Access | Data Use Agreement (DUA) required |
| Key variable | `PSTCO` -- patient county of residence FIPS |
| Format | SAS transport (.asc) or CSV export |
| Cost | ~$200-400 per state-year |
| Turnaround | 3-6 weeks from application |

**Why SEDD:** Residence-based counts eliminate the catchment-area problem
that plagues facility-based data. This is the single biggest methodological
upgrade for the project.

**Aggregation:** Visit-level records -> county-year totals using PSTCO.
Loader: `src/database_read/load_sedd.py`

---

### 2. CMS Medicare Advantage Enrollment (Insurance Coverage)

| Field | Value |
|-------|-------|
| Source | Centers for Medicare & Medicaid Services |
| URL | https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/monthly-enrollment-contract/plan/state/county |
| Years | Dec 2006 - present (monthly) |
| Access | Public download, no registration |
| Key variable | Monthly county-level MA enrollees |
| Format | Excel (.xlsx) |

**Panel strategy:** Download monthly files for 2010-2023. Compute annual
average enrollment per county. Already have 2022-2023 files in `data/MA/`.

---

### 3. CMS Medicare Geographic Variation (Medicare FFS Utilization)

| Field | Value |
|-------|-------|
| Source | Centers for Medicare & Medicaid Services |
| URL | https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county |
| Years | 2007-present (annual) |
| Access | Public download |
| Key variables | Per-capita Medicare spending, ED visits, hospitalizations, readmissions, chronic conditions |
| Format | CSV |

**Why this matters:** Medicare FFS ED visit rates are a direct proxy for
ED utilization among the 65+ population. This is a free, public,
county-level ED utilization signal that complements SEDD.

---

### 4. NC Medicaid Enrollment (Insurance Coverage)

| Field | Value |
|-------|-------|
| Source | NC Department of Health and Human Services |
| URL | https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports |
| Years | SFY 1998-2026 (gaps: SFY 2011, SFY 2014) |
| Access | Public download |
| Key variable | Monthly unduplicated enrollment by county |
| Format | Excel (.xlsx) |

**Panel strategy:** Download annual reports for SFY 2010-2023. Interpolate
SFY 2011 and 2014 gaps.

**Note:** NC Medicaid expansion (Dec 2023) creates a structural break.
Model should include a post-expansion indicator for 2024+ data.

---

### 5. CDC Social Vulnerability Index (Social Determinants)

| Field | Value |
|-------|-------|
| Source | CDC/ATSDR |
| URL | https://atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html |
| Years | 2000, 2010, 2014, 2016, 2018, 2020, 2022 |
| Access | Public download |
| Key variables | 4 theme percentile rankings, 17 component variables |
| Format | CSV |

**Panel strategy:** Biennial data. For odd years, carry forward from prior
release (e.g., 2015 uses 2014 SVI, 2017 uses 2016 SVI). For 2010-2013,
use 2010 release as baseline.

---

### 6. CDC PLACES (Chronic Disease Prevalence)

| Field | Value |
|-------|-------|
| Source | CDC Division of Population Health |
| URL | https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb |
| Years | Release 2020-2025 (underlying BRFSS data ~2017-2023) |
| Access | Public download |
| Key variables | 40 health indicators, age-adjusted and crude prevalence |
| Format | CSV |

**Panel strategy:** Only covers ~2017-2023 effectively. For 2010-2016,
use County Health Rankings (see below) for comparable health behavior
and outcome measures.

---

### 7. HRSA Area Health Resources File (Healthcare Workforce)

| Field | Value |
|-------|-------|
| Source | HRSA National Center for Health Workforce Analysis |
| URL | https://data.hrsa.gov/data/download |
| Years | Releases 2019-2024 (each contains multi-year historical data) |
| Access | Public download |
| Key variables | MDs per capita, HPSA designations, hospital beds, FQHCs |
| Format | Flat file (.csv) with data dictionary |

**Panel strategy:** Each AHRF release is a cross-section but contains
historical data columns. The 2024 release includes physician counts back
to ~2010. Download multiple release years to maximize historical coverage.

---

### 8. BLS Local Area Unemployment Statistics (Economic Conditions)

| Field | Value |
|-------|-------|
| Source | Bureau of Labor Statistics |
| URL | https://www.bls.gov/lau/data.htm |
| Years | 1990-present (monthly and annual) |
| Access | Public download, bulk files |
| Key variables | Unemployment rate, labor force, employment |
| Format | Text/CSV |

**Panel strategy:** Annual averages for 2010-2023. Full county coverage,
no gaps. One of the strongest annually-updated predictors available.

---

### 9. Census SAIPE (Poverty and Income)

| Field | Value |
|-------|-------|
| Source | U.S. Census Bureau |
| URL | https://www.census.gov/programs-surveys/saipe/data/datasets.html |
| Years | 1989, 1993, 1995-2024 (annual) |
| Access | Public download and API |
| Key variables | Median household income, poverty rate, child poverty rate |
| Format | CSV |

**Panel strategy:** Annual single-year county estimates. Full 2010-2023
coverage with no gaps. More current than ACS 5-year overlapping windows.

---

### 10. County Health Rankings & Roadmaps (Meta-Source)

| Field | Value |
|-------|-------|
| Source | University of Wisconsin Population Health Institute |
| URL | https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation |
| Years | 2010-2025 (annual) |
| Access | Public download |
| Key variables | Premature death, health behaviors, clinical care, social factors |
| Format | Excel/CSV |

**Panel strategy:** This fills the pre-2017 gap for health behavior data
that PLACES cannot cover. Use CHR&R for 2010-2016, PLACES for 2017-2023,
with cross-walk validation in overlap years.

---

### 11. FCC Broadband Deployment (Digital Access)

| Field | Value |
|-------|-------|
| Source | Federal Communications Commission |
| URL | https://www.fcc.gov/form-477-county-data-internet-access-services |
| Years | June 2009 - June 2024 (semi-annual) |
| Access | Public download |
| Key variables | Residential broadband connections per 1,000 households |
| Format | CSV |

**Why this matters:** Broadband access affects telehealth availability.
Counties with low broadband may have higher ED use for conditions
manageable remotely. Novel predictor for the alternative data framing.

---

### 12. USDA Rural-Urban Classifications

| Field | Value |
|-------|-------|
| Source | USDA Economic Research Service |
| URL | https://www.ers.usda.gov/data-products/county-level-data-sets/ |
| Years | RUCC: 2003, 2013, 2023. UIC: 2003, 2013. |
| Access | Public download |
| Key variables | Rural-Urban Continuum Code, Urban Influence Code |
| Format | Excel |

**Panel strategy:** These change slowly. Use 2013 codes for 2010-2017,
2023 codes for 2018-2023.

---

## Year Coverage Matrix

```
Source                    2010  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2023
---------------------------------------------------------------------------------------------------------
HCUP SEDD (target)         Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
CMS MA Enrollment          Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
CMS Medicare Geo Var       Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
NC Medicaid                Y     -     Y     Y     -     Y     Y     Y     Y     Y     Y     Y     Y     Y
CDC SVI                    Y     .     .     .     Y     .     Y     .     Y     .     Y     .     Y     .
CDC PLACES                 .     .     .     .     .     .     .     Y     Y     Y     Y     Y     Y     Y
County Health Rankings     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
HRSA AHRF                  Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
BLS Unemployment           Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
Census SAIPE               Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
FCC Broadband              Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y     Y
USDA Rural-Urban           .     .     .     Y     .     .     .     .     .     .     .     .     .     Y

Y = annual data available
. = carry-forward or interpolate from nearest available year
- = missing (NC Medicaid gaps in SFY 2011 and 2014)
```

---

## Pragmatic Recommendation: Phased Approach

### Phase 1: Cross-Section (Current -- Ready Now)
- Single year (2021), 100 NC counties
- Current merged dataset + pipeline
- Paper 1: "Identifying Excess ED Burden Using Alternative Data"

### Phase 2: Panel (When SEDD Arrives -- ~2 months)
- **Recommended window: 2015-2023** (9 years)
  - All sources have strong coverage
  - PLACES begins 2017, CHR&R fills 2015-2016
  - Post-ACA stabilization, pre-Medicaid-expansion
  - 100 counties x 9 years = 900 county-year observations
- Paper 2: "Temporal Trends in Healthcare Underservice"

### Phase 3: Forecasting (After Panel Model is Validated)
- Train on 2015-2021, validate on 2022-2023
- Forecast 2024-2026 using lagged predictors (all public, available now)
- **Key output:** "These 15 counties are projected to experience
  worsening ED burden by 2026 based on current trends"
- This is the government-facing deliverable

---

## Data Directory Structure

```
data/
  raw/                    # Untouched downloads (never modify)
    sedd/                 # HCUP SEDD files (DUA required)
    ma_enrollment/        # CMS monthly MA files
    medicare_geo_var/     # CMS Geographic Variation
    medicaid_nc/          # NC DHHS enrollment reports
    svi/                  # CDC SVI releases
    places/               # CDC PLACES releases
    ahrf/                 # HRSA AHRF releases
    bls_laus/             # BLS unemployment
    saipe/                # Census poverty/income
    chr/                  # County Health Rankings
    fcc_broadband/        # FCC Form 477
    usda/                 # USDA rural-urban codes
  processed/              # Cleaned, standardized county-year panels
    panel_nc.csv          # Master NC panel (county x year)
    panel_sc.csv          # SC panel (for replication)
  final/                  # Pipeline-ready merged datasets
    merged_county_data.csv
```
