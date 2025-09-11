---
aliases:
---
## **Step 1. Identify Data Releases for 2023**

- **CMS Medicare Geographic Variation (GV PUF)** → Annual update, usually spring. Download the 2024 _county-level_ file.
- https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county/data?query=%7B%22filters%22%3A%7B%22list%22%3A%5B%7B%22conjunction%22%3A%7B%22value%22%3A%22AND%22%7D%2C%22conditions%22%3A%5B%7B%22column%22%3A%7B%22value%22%3A%22BENE_GEO_DESC%22%7D%2C%22comparator%22%3A%7B%22value%22%3A%22STARTS_WITH%22%7D%2C%22filterValue%22%3A%5B%22NC%22%5D%7D%2C%7B%22column%22%3A%7B%22value%22%3A%22YEAR%22%7D%2C%22comparator%22%3A%7B%22value%22%3A%22%3D%22%7D%2C%22filterValue%22%3A%5B%222023%22%5D%7D%5D%7D%5D%2C%22rootConjunction%22%3A%7B%22value%22%3A%22AND%22%7D%7D%2C%22keywords%22%3A%222023%22%2C%22offset%22%3A0%2C%22limit%22%3A10%2C%22sort%22%3A%7B%22sortBy%22%3A%22YEAR%22%2C%22sortOrder%22%3A%22ASC%22%7D%2C%22columns%22%3A%5B%5D%7D


    
- **CMS Monthly Enrollment** → Download Jan–Dec 2024 CSVs, save as a folder.
	https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/mcradvpartdenroldata/monthly-ma/ma-enrollment-scc-2023-01
    
- **NC Medicaid Enrollment** → Grab monthly county enrollment PDFs/CSVs from NC DHHS dashboard for 2024.
https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports#FiscalYears2025-2021EnrollmentReports-5593

https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports#FiscalYears2025-2021EnrollmentReports-5593

- **HRSA AHRF (Area Health Resource File)** → They release one file/year (the 2024 edition).

https://data.hrsa.gov/data/download
    
- **CDC PLACES & SVI** → PLACES (2023 release has 2021 data; check for 2024 update). SVI is updated ~every 2 years, so you may only have 2022/2023.

https://www.atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html

    
- **NC DETECT Public Dashboards** → ED visits; may not have a raw 2024 CSV, but you can snapshot public tables or request export.


Your NC DETECT data request has been submitted successfully. 

Your request ID is tanzillo20250904170040942.

A confirmation email will be sent to you shortly. Please send questions to ncdetect@listserv.med.unc.edu.

[Go to NC DETECT Home](https://ncdetect.org/)


---
Awesome—North Carolina is perfect for this. Here’s a tight, NC-focused “data map” of free, reputable datasets you can pull to 1) prove need (shortages, avoidable use), 2) show demand (Medicare/Medicaid enrollment), and 3) pinpoint where scholarships + new primary-care clinics would do the most good.

# Core supply/shortage datasets (NC-ready)

- **HRSA HPSA (Primary Care) & MUA/P** – official shortage designations + HPSA scores; searchable, downloadable, and API-accessible. Use to flag high-need counties/tracts and to score “how urgent.” ([HRSA Data](https://data.hrsa.gov/topics/health-workforce/health-workforce-shortage-areas?utm_source=chatgpt.com "Health Workforce Shortage Areas"))
    
- **HRSA AHRF (Area Health Resources Files)** – county-level counts of primary-care physicians/NPs/PAs, training programs, socioeconomic context; long time series. Great for “providers per 10k” and trends by county. ([HRSA Data](https://data.hrsa.gov/data/download?utm_source=chatgpt.com "Data Downloads"))
    
- **NC Office of Rural Health / State health stats** – NC-specific shortage one-pager + indicators (e.g., primary-care clinicians per 10k). Use to anchor state narrative and validate HRSA numbers. ([NC DHHS](https://www.ncdhhs.gov/nc-dhhs-orh-hpsa-one-pager/open?utm_source=chatgpt.com "North Carolina Health Professional Shortage Area Primary ..."), [NC State Center for Health Statistics](https://schs.dph.ncdhhs.gov/units/ldas/docs/17-PrimaryCareCliniciansIndicator2023NCSHIP.pdf?utm_source=chatgpt.com "Primary Care Clinicians - NC State Center for Health Statistics"))
    
- **Care Compare – Doctors & Clinicians National Downloadable File** – clinician records with specialties and practice locations; filter to PCP specialties to map actual sites of care. (Pairs well with NPI taxonomy filters.) ([CMS Data](https://data.cms.gov/provider-data/dataset/mj5m-pzi6?utm_source=chatgpt.com "National Downloadable File | Provider Data Catalog"))
    
- **NPPES (NPI) Downloadable File** – the master provider roster with practice addresses; use for custom primary-care panels (family med, general internal med, general pediatrics, FNP/ANP, PA) when you need the rawest view. ([CMS Download](https://download.cms.gov/nppes/NPI_Files.html?utm_source=chatgpt.com "NPI Files"), [CMS](https://www.cms.gov/medicare/regulations-guidance/administrative-simplification/data-dissemination?utm_source=chatgpt.com "Data Dissemination"))
    

# Demand & avoidable-use (policy punch)

- **CMS Medicare Geographic Variation PUF (County)** – county-level metrics for FFS beneficiaries including ED visits and **AHRQ Prevention Quality Indicators (PQIs)** for ambulatory-care-sensitive hospitalizations; perfect for “avoidable use falls when primary care supply rises” arguments. ([CMS Data](https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county?utm_source=chatgpt.com "Medicare Geographic Variation - by National, State & County"), [CMS](https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/medicare-geographic-variation/downloads/geo_var_puf_methods_paper.pdf?utm_source=chatgpt.com "Medicare Data for the Geographic Variation Public Use File"))
    
- **CMS Medicare Monthly Enrollment** – current monthly counts (and dashboards) to size the older-adult demand by county/state; add MA penetration if you want contract/plan granularity. ([CMS Data](https://data.cms.gov/summary-statistics-on-beneficiary-enrollment/medicare-and-medicaid-reports/medicare-monthly-enrollment?utm_source=chatgpt.com "Medicare Monthly Enrollment"), [CMS](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data?utm_source=chatgpt.com "Medicare Advantage/Part D Contract and Enrollment Data"))
    
- **NC Medicaid Enrollment (by county, monthly; expansion dashboards)** – shows where lower-income adult demand is concentrated, including post-expansion spikes in rural counties. ([NC Medicaid](https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports?utm_source=chatgpt.com "NC Medicaid Enrollment Reports"), [NC DHHS](https://www.ncdhhs.gov/news/press-releases/2023/12/20/new-medicaid-expansion-enrollment-dashboard-updated-monthly-enrollee-data?utm_source=chatgpt.com "New Medicaid Expansion Enrollment Dashboard, Updated ..."))
    
- **NC DETECT public dashboards (ED)** – county-level ED visit rates by condition (asthma/COPD, mental health, falls, overall). Raw line-level data require a request, but the public dashboards are enough for county comparisons. ([ncdetect.org](https://ncdetect.org/dashboard/?utm_source=chatgpt.com "Dashboards"))
    

# Facility & access context (where to place clinics)

- **HRSA Health Center Program Sites (FQHCs) & NHSC sites** – locations of existing safety-net primary-care access; use to find geographic gaps. ([HRSA Data](https://data.hrsa.gov/data/download?utm_source=chatgpt.com "Data Downloads"))
    
- **HIFLD hospitals / urgent care** – national geocoded facility points; helps ensure you’re not duplicating proximate capacity and to identify true primary-care deserts. ([hifld-geoplatform.hub.arcgis.com](https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform%3A%3Ahospitals-1?layer=0&utm_source=chatgpt.com "Hospitals"), [Data.gov](https://catalog.data.gov/dataset/urgent-care-facilities?utm_source=chatgpt.com "Urgent Care Facilities - Dataset - Catalog - Data.gov"))
    

# Social risk & rurality overlays (targeting)

- **CDC PLACES** – small-area (county/tract) estimates of chronic disease and preventive-care indicators; great for layering morbidity and screening gaps. ([CMS](https://www.cms.gov/medicare/regulations-guidance/administrative-simplification/data-dissemination?utm_source=chatgpt.com "Data Dissemination"))
    
- **CDC Social Vulnerability Index (SVI)** – county/tract social risk composite to prioritize equitable placement. ([NPI Registry](https://npiregistry.cms.hhs.gov/help/taxonomy?utm_source=chatgpt.com "NPPES NPI Registry"))
    
- **USDA Rural–Urban Continuum/RUCA codes** – standard rurality flags (county & ZIP/tract) to define your rural cohorts.
    

# Extra Medicare provider utilization (optional, nice-to-have)

- **Medicare Physician & Other Practitioners PUF** – service counts & payments by NPI/HCPCS or by geography/service; lets you see actual primary-care throughput patterns. ([CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service?utm_source=chatgpt.com "Medicare Physician & Other Practitioners - by Provider and ..."))
    

---

## How these fit together (quick join plan)

- **Geography keys:** County FIPS for AHRF, CMS GV PUF, SVI, PLACES, NC Medicaid; ZIP/tract for PLACES/SVI; latitude/longitude for provider/facility points.
    
- **Provider keys:** NPI from Care Compare/NPPES; use taxonomy to flag _primary care_ (family medicine, general internal medicine, general pediatrics, geriatrics; FNP/ANP; PA). ([CMS Data](https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/DOC_Data_Dictionary.pdf?utm_source=chatgpt.com "Provider Data Catalog: Doctors & Clinicians Data Dictionary"))
    
- **Core stack for NC:** HPSA/MUA-P (need), AHRF (supply per 10k), NC Medicaid enrollment (demand), CMS GV PUF (ED + PQIs), Care Compare/NPPES (practice locations), PLACES + SVI (risk), USDA rurality (rural targeting). ([HRSA Data](https://data.hrsa.gov/topics/health-workforce/health-workforce-shortage-areas?utm_source=chatgpt.com "Health Workforce Shortage Areas"), [NC Medicaid](https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports?utm_source=chatgpt.com "NC Medicaid Enrollment Reports"), [CMS Data](https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county?utm_source=chatgpt.com "Medicare Geographic Variation - by National, State & County"), [CMS](https://www.cms.gov/medicare/regulations-guidance/administrative-simplification/data-dissemination?utm_source=chatgpt.com "Data Dissemination"), [NPI Registry](https://npiregistry.cms.hhs.gov/help/taxonomy?utm_source=chatgpt.com "NPPES NPI Registry"))
    

## What you can argue with these

1. **Scholarships → more PCPs in rural HPSAs:** Show counties with high HPSA scores, low PCPs per 10k (AHRF), and high Medicaid/Medicare density. ([HRSA Data](https://data.hrsa.gov/topics/health-workforce/health-workforce-shortage-areas?utm_source=chatgpt.com "Health Workforce Shortage Areas"), [NC Medicaid](https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports?utm_source=chatgpt.com "NC Medicaid Enrollment Reports"))
    
2. **Clinics where need is sharpest:** Prioritize counties/tracts with high ED visit and PQI rates (GV PUF/NC DETECT), high SVI, and limited nearby primary-care sites (Care Compare + FQHC/NHSC). ([CMS](https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/medicare-geographic-variation/downloads/geo_var_puf_methods_paper.pdf?utm_source=chatgpt.com "Medicare Data for the Geographic Variation Public Use File"), [ncdetect.org](https://ncdetect.org/dashboard/?utm_source=chatgpt.com "Dashboards"), [NPI Registry](https://npiregistry.cms.hhs.gov/help/taxonomy?utm_source=chatgpt.com "NPPES NPI Registry"), [CMS Data](https://data.cms.gov/provider-data/dataset/mj5m-pzi6?utm_source=chatgpt.com "National Downloadable File | Provider Data Catalog"))
    

If you want, I can pull these for **North Carolina only** and hand you a clean starter folder (county crosswalk, HPSA snapshot, AHRF provider rates, latest Medicaid enrollment, GV PUF county table, PLACES/SVI, provider points).

----

Here’s a **clean set of copy-paste links** (all free/public) you can drop into your project doc for North Carolina. I grouped them by theme so you can grab what you need quickly:

---

### **Provider Supply & Shortages**

- HRSA HPSA (Primary Care) Find Tool: [https://data.hrsa.gov/tools/shortage-area/hpsa-find](https://data.hrsa.gov/tools/shortage-area/hpsa-find)
    
- HRSA MUA/P (Medically Underserved Areas/Populations): [https://data.hrsa.gov/tools/shortage-area/mua-find](https://data.hrsa.gov/tools/shortage-area/mua-find)
    
- HRSA Area Health Resource Files (AHRF): [https://data.hrsa.gov/data/download](https://data.hrsa.gov/data/download)
    

---

### **Medicare Data (Demand & Utilization)**

- Medicare Geographic Variation Public Use File (County): [https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-geographic-variation](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-geographic-variation)
    
- Medicare Monthly Enrollment Data (by state/county): [https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-monthly-enrollment](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-monthly-enrollment)
    
- Medicare Physician & Other Practitioners PUF: [https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-provider-utilization-and-payment-data/physician-other-practitioners](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-provider-utilization-and-payment-data/physician-other-practitioners)
    

---

### **Medicaid Data (NC Focused)**

- NC Medicaid County Enrollment (Dashboard & Reports): [https://medicaid.ncdhhs.gov/reports/enrollment](https://medicaid.ncdhhs.gov/reports/enrollment)
    
- NC Medicaid Expansion Data (Monthly): [https://medicaid.ncdhhs.gov/nc-medicaid-expansion](https://medicaid.ncdhhs.gov/nc-medicaid-expansion)
    
- CMS Medicaid Enrollment & Expenditure Reports: [https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data/index.html](https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data/index.html)
    

---

### **Facilities & Access Points**

- HRSA Health Center Program Data (FQHC Sites): [https://data.hrsa.gov/data/download](https://data.hrsa.gov/data/download) (Health Center tab)
    
- NHSC (National Health Service Corps) Sites: [https://data.hrsa.gov/data/download](https://data.hrsa.gov/data/download) (NHSC Sites tab)
    
- HIFLD Hospital & Emergency Care Facilities: [https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals](https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals)
    

---

### **Provider Rosters**

- Care Compare: Doctors & Clinicians National Downloadable File: [https://data.cms.gov/provider-data/dataset/mj5m-pzi6](https://data.cms.gov/provider-data/dataset/mj5m-pzi6)
    
- NPPES NPI Downloadable File (weekly): [https://download.cms.gov/nppes/NPI_Files.html](https://download.cms.gov/nppes/NPI_Files.html)
    

---

### **Avoidable Use & Health Outcomes**

- NC DETECT (ED Visit Public Dashboard): [https://ncdetect.org/public-data/](https://ncdetect.org/public-data/)
    
- AHRQ Prevention Quality Indicators (overview): [https://qualityindicators.ahrq.gov/Modules/pqi_resources.aspx](https://qualityindicators.ahrq.gov/Modules/pqi_resources.aspx)
    

---

### **Social Risk & Rurality**

- CDC PLACES (County/Tract Preventive Health & Chronic Disease Indicators): [https://www.cdc.gov/places](https://www.cdc.gov/places)
    
- CDC Social Vulnerability Index (SVI): [https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html](https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html)
    
- USDA Rural–Urban Continuum Codes: [https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/](https://www.ers.usda.gov/data-products/rural-urban-continuum-codes/)
    
- USDA RUCA Codes (ZIP/tract rurality): [https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes/](https://www.ers.usda.gov/data-products/rural-urban-commuting-area-codes/)
    

---

Do you want me to also **bundle these into an Excel/CSV starter file** (with columns = dataset name, description, link) so you can keep your project organized from the start?!

Yes — Obsidian will render these cleanly if you use a LaTeX plugin (or if you wrap them in `$$ ... $$` for block math and `$ ... $` for inline). Here’s your pipeline’s equations ready to paste:

---

### **Metric 1: Provider Availability Gap**


$$
\text{Availability Gap} = 
\frac{\text{Medicare Enrollees} + \text{Medicaid Enrollees}}
     {\text{Primary Care Providers}}
$$




---

### **Metric 2: Avoidable Utilization Burden**


$$
\text{Avoidable Utilization Burden} = 
\frac{\text{ED Visits for Ambulatory-Care-Sensitive Conditions (PQIs)}}
     {\text{1000 Medicare/Medicaid Enrollees}}
$$


---

### **Metric 3: Socio-Demographic Risk Index**


$$
\text{Risk Index} = 
\text{SVI Percentile} + \text{Chronic Disease Prevalence (PLACES)}
$$

---

### **Composite Need Score**


$$
\text{County Need Score} = 
\text{Availability Gap} + 
\text{Avoidable Utilization Burden} + 
\text{Risk Index}
$$ 

---



---


[[output-2.png]]

Advances.parquet

![[output-3.png]]