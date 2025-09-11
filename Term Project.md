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

"Your request ID is tanzillo20250904170040942.

A confirmation email will be sent to you shortly. Please send questions to ncdetect@listserv.med.unc.edu."

[Go to NC DETECT Home](https://ncdetect.org/)


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
