"""
Download & Stage Public Data Sources (2015-2023)
=================================================

Automated downloads:
    1. Census SAIPE  (poverty, income)          -- 2015-2023 via API
    2. BLS LAUS      (unemployment)             -- 2015-2023 via API
    3. CDC PLACES    (chronic disease)           -- 2015-2023 via SODA API (10 releases)
    4. County Health Rankings                    -- 2019-2025 via bulk CSV

Manual-download instructions printed for:
    A. CMS MA Enrollment   (2015-2021, monthly)
    B. CDC SVI             (5 biennial releases)
    C. NC Medicaid         (SFY 2015-2023)
    D. HCUP SEDD           (DUA required)

Usage
-----
    python -m src.data_acquisition.download_public_data
"""

import csv
import io
import json
import os
import shutil
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_FIPS = "37"
STATE_ABBR = "NC"
YEARS = range(2015, 2024)  # 2015-2023
RAW_DIR = Path("data/raw")

# County FIPS codes by state
NC_COUNTY_FIPS = [f"37{i:03d}" for i in range(1, 200, 2)]  # 100 counties
SC_COUNTY_FIPS = [f"45{i:03d}" for i in range(1, 92, 2)]   # 46 counties

STATE_COUNTY_FIPS_MAP = {
    "NC": NC_COUNTY_FIPS,
    "SC": SC_COUNTY_FIPS,
}

STATE_FIPS_MAP = {"NC": "37", "SC": "45"}
STATE_FULLNAME = {"NC": "North Carolina", "SC": "South Carolina"}


def _ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _fetch_json(url, retries=3, delay=2):
    """Fetch JSON from a URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NC-Hospital-Research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}/{retries} after error: {e}")
                time.sleep(delay * (attempt + 1))
            else:
                raise


def _fetch_csv_text(url, retries=3, delay=2):
    """Fetch raw text from a CSV URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NC-Hospital-Research/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if attempt < retries - 1:
                print(f"    Retry {attempt + 1}/{retries} after error: {e}")
                time.sleep(delay * (attempt + 1))
            else:
                raise


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Census SAIPE
# ═══════════════════════════════════════════════════════════════════════════════

def download_saipe():
    """
    Download Census SAIPE poverty/income estimates for NC counties, 2015-2023.

    API: https://api.census.gov/data/timeseries/poverty/saipe
    Variables (current API names use _PT suffix for point estimates):
        SAEPOVRTALL_PT  = poverty rate (all ages)
        SAEMHI_PT       = median household income
        SAEPOVRT0_17_PT = child poverty rate (0-17)
        NAME            = county name
    """
    out_path = RAW_DIR / "saipe" / f"saipe_{STATE_ABBR.lower()}_2015_2023.csv"
    if out_path.exists():
        print(f"  [SAIPE] Already exists: {out_path}")
        return True

    print("  [SAIPE] Downloading from Census API...")
    _ensure_dir(out_path)

    rows = []
    for year in YEARS:
        url = (
            f"https://api.census.gov/data/timeseries/poverty/saipe"
            f"?get=NAME,SAEPOVRTALL_PT,SAEMHI_PT,SAEPOVRT0_17_PT"
            f"&for=county:*&in=state:{STATE_FIPS}&time={year}"
        )
        try:
            data = _fetch_json(url)
        except Exception as e:
            print(f"    [SAIPE] Failed for {year}: {e}")
            continue

        # First row is header
        header = data[0]
        for row in data[1:]:
            rec = dict(zip(header, row))
            fips = rec.get("state", "") + rec.get("county", "")
            rows.append({
                "FIPS": fips,
                "Year": year,
                "County": rec.get("NAME", "").replace(" County", ""),
                "PovertyRate": rec.get("SAEPOVRTALL_PT", ""),
                "MedianHouseholdIncome": rec.get("SAEMHI_PT", ""),
                "ChildPovertyRate": rec.get("SAEPOVRT0_17_PT", ""),
            })
        print(f"    {year}: {len(data) - 1} counties")
        time.sleep(0.5)  # Be kind to the API

    if not rows:
        print("  [SAIPE] No data retrieved!")
        return False

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FIPS", "Year", "County",
                                                "PovertyRate",
                                                "MedianHouseholdIncome",
                                                "ChildPovertyRate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [SAIPE] Saved {len(rows)} records to {out_path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BLS LAUS (Local Area Unemployment Statistics)
# ═══════════════════════════════════════════════════════════════════════════════

def download_bls_laus():
    """
    Download BLS LAUS unemployment rates for NC counties, 2015-2023.

    API: https://api.bls.gov/publicAPI/v2/timeseries/data/
    Series ID pattern: LAUCN37XXXXX0000000003
      - CN = county
      - 37XXXXX = state + county FIPS (7 digits total, 2+5)
      - 03 = unemployment rate

    BLS limits (no API key): 25 series per request, 20 years per series.
    100 NC counties -> 4 batches of 25.
    """
    out_path = RAW_DIR / "bls_laus" / f"laus_annual_{STATE_ABBR.lower()}.csv"
    if out_path.exists():
        print(f"  [BLS] Already exists: {out_path}")
        return True

    _ensure_dir(out_path)

    # Build series IDs for unemployment rate (measure 03)
    county_fips_list = STATE_COUNTY_FIPS_MAP.get(STATE_ABBR, NC_COUNTY_FIPS)
    series_ids = []
    fips_for_series = {}
    for fips in county_fips_list:
        # LAUS series: LAUCN + 5-digit county FIPS + 0000000003
        sid = f"LAUCN{fips}0000000003"
        series_ids.append(sid)
        fips_for_series[sid] = fips

    all_rows = []
    batch_size = 25  # BLS limit without API key
    n_batches = (len(series_ids) + batch_size - 1) // batch_size
    print(f"  [BLS] Downloading from BLS API ({n_batches} batches)...")
    for batch_start in range(0, len(series_ids), batch_size):
        batch = series_ids[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"    Batch {batch_num}: series {batch_start + 1}-"
              f"{min(batch_start + batch_size, len(series_ids))}")

        payload = json.dumps({
            "seriesid": batch,
            "startyear": str(min(YEARS)),
            "endyear": str(max(YEARS)),
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "NC-Hospital-Research/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"    [BLS] Batch {batch_num} failed: {e}")
            continue

        if result.get("status") != "REQUEST_SUCCEEDED":
            print(f"    [BLS] API error: {result.get('message', 'unknown')}")
            continue

        # Collect monthly values per FIPS+Year, compute annual average
        for series in result.get("Results", {}).get("series", []):
            sid = series["seriesID"]
            fips = fips_for_series.get(sid, "")
            monthly = {}  # {year: [values]}
            for obs in series.get("data", []):
                period = obs.get("period", "")
                # M13 = annual average (if available), M01-M12 = monthly
                if period == "M13":
                    all_rows.append({
                        "FIPS": fips,
                        "Year": int(obs["year"]),
                        "UnemploymentRate": obs.get("value", ""),
                    })
                elif period.startswith("M"):
                    yr = int(obs["year"])
                    try:
                        val = float(obs["value"])
                        monthly.setdefault(yr, []).append(val)
                    except (ValueError, TypeError):
                        pass
            # If M13 wasn't available, compute from monthly
            years_done = {r["Year"] for r in all_rows if r["FIPS"] == fips}
            for yr, vals in monthly.items():
                if yr not in years_done and vals:
                    all_rows.append({
                        "FIPS": fips,
                        "Year": yr,
                        "UnemploymentRate": f"{sum(vals) / len(vals):.1f}",
                    })

        time.sleep(2)  # BLS rate limit

    if not all_rows:
        print("  [BLS] No data retrieved!")
        return False

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FIPS", "Year",
                                                "UnemploymentRate"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  [BLS] Saved {len(all_rows)} records to {out_path}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CDC PLACES
# ═══════════════════════════════════════════════════════════════════════════════

def download_places():
    """
    Download CDC PLACES county-level health data for NC from all releases.

    Uses the Socrata Open Data API (SODA) for each PLACES release.
    Each release contains 2 BRFSS data years (the Year column reflects the
    actual BRFSS survey year, not the release year):

        2020 release (dv4u-3x3q): data years 2017-2018  (no locationid -- uses name matching)
        2021 release (pqpp-u99h): data years 2018-2019
        2022 release (duw2-7jbt): data years 2019-2020
        2023 release (h3ej-a9ec): data years 2020-2021
        2024 release (fu4u-a9bh): data years 2021-2022
        2025 release (swc5-untb): data years 2022-2023

    Note: The 2016-2019 releases (dataset IDs 9z78-nsfp, vurf-k5wr, rja3-32tc,
    6vp6-wxuq, q8xq-ygsk) are city/place-level "500 Cities" data, NOT county-
    level. County-level PLACES data only begins with the 2020 release.

    For overlapping years, later releases overwrite earlier ones (more
    current methodology).

    Output is pivoted wide: one row per (FIPS, Year), measures as columns.
    """
    out_path = RAW_DIR / "places" / f"places_{STATE_ABBR.lower()}.csv"
    if out_path.exists():
        print(f"  [PLACES] Already exists: {out_path}")
        return True

    _ensure_dir(out_path)

    # 6 PLACES county-level dataset IDs in release order.
    # County-level PLACES only started with the 2020 release.
    # The 2016-2019 releases are city/place-level ("500 Cities") -- excluded.
    places_datasets = [
        ("dv4u-3x3q", "2020 release"),   # data years 2017-2018 (no locationid)
        ("pqpp-u99h", "2021 release"),    # data years 2018-2019
        ("duw2-7jbt", "2022 release"),    # data years 2019-2020
        ("h3ej-a9ec", "2023 release"),    # data years 2020-2021
        ("fu4u-a9bh", "2024 release"),    # data years 2021-2022
        ("swc5-untb", "2025 release"),    # data years 2022-2023
    ]

    # County FIPS pattern for filtering (5-digit, starting with state FIPS)
    import re
    county_fips_re = re.compile(rf"^{STATE_FIPS}\d{{3}}$")

    # Reverse lookup: county name -> FIPS for datasets without locationid
    # Build name-to-FIPS mapping for the current state
    _nc_names = {
        "ALAMANCE": "37001", "ALEXANDER": "37003", "ALLEGHANY": "37005",
        "ANSON": "37007", "ASHE": "37009", "AVERY": "37011",
        "BEAUFORT": "37013", "BERTIE": "37015", "BLADEN": "37017",
        "BRUNSWICK": "37019", "BUNCOMBE": "37021", "BURKE": "37023",
        "CABARRUS": "37025", "CALDWELL": "37027", "CAMDEN": "37029",
        "CARTERET": "37031", "CASWELL": "37033", "CATAWBA": "37035",
        "CHATHAM": "37037", "CHEROKEE": "37039", "CHOWAN": "37041",
        "CLAY": "37043", "CLEVELAND": "37045", "COLUMBUS": "37047",
        "CRAVEN": "37049", "CUMBERLAND": "37051", "CURRITUCK": "37053",
        "DARE": "37055", "DAVIDSON": "37057", "DAVIE": "37059",
        "DUPLIN": "37061", "DURHAM": "37063", "EDGECOMBE": "37065",
        "FORSYTH": "37067", "FRANKLIN": "37069", "GASTON": "37071",
        "GATES": "37073", "GRAHAM": "37075", "GRANVILLE": "37077",
        "GREENE": "37079", "GUILFORD": "37081", "HALIFAX": "37083",
        "HARNETT": "37085", "HAYWOOD": "37087", "HENDERSON": "37089",
        "HERTFORD": "37091", "HOKE": "37093", "HYDE": "37095",
        "IREDELL": "37097", "JACKSON": "37099", "JOHNSTON": "37101",
        "JONES": "37103", "LEE": "37105", "LENOIR": "37107",
        "LINCOLN": "37109", "MACON": "37111", "MADISON": "37113",
        "MARTIN": "37115", "MCDOWELL": "37117", "MECKLENBURG": "37119",
        "MITCHELL": "37121", "MONTGOMERY": "37123", "MOORE": "37125",
        "NASH": "37127", "NEW HANOVER": "37129", "NORTHAMPTON": "37131",
        "ONSLOW": "37133", "ORANGE": "37135", "PAMLICO": "37137",
        "PASQUOTANK": "37139", "PENDER": "37141", "PERQUIMANS": "37143",
        "PERSON": "37145", "PITT": "37147", "POLK": "37149",
        "RANDOLPH": "37151", "RICHMOND": "37153", "ROBESON": "37155",
        "ROCKINGHAM": "37157", "ROWAN": "37159", "RUTHERFORD": "37161",
        "SAMPSON": "37163", "SCOTLAND": "37165", "STANLY": "37167",
        "STOKES": "37169", "SURRY": "37171", "SWAIN": "37173",
        "TRANSYLVANIA": "37175", "TYRRELL": "37177", "UNION": "37179",
        "VANCE": "37181", "WAKE": "37183", "WARREN": "37185",
        "WASHINGTON": "37187", "WATAUGA": "37189", "WAYNE": "37191",
        "WILKES": "37193", "WILSON": "37195", "YADKIN": "37197",
        "YANCEY": "37199",
    }

    # SC county name -> FIPS mapping (for 2020 release without locationid)
    _sc_names = {
        "ABBEVILLE": "45001", "AIKEN": "45003", "ALLENDALE": "45005",
        "ANDERSON": "45007", "BAMBERG": "45009", "BARNWELL": "45011",
        "BEAUFORT": "45013", "BERKELEY": "45015", "CALHOUN": "45017",
        "CHARLESTON": "45019", "CHEROKEE": "45021", "CHESTER": "45023",
        "CHESTERFIELD": "45025", "CLARENDON": "45027", "COLLETON": "45029",
        "DARLINGTON": "45031", "DILLON": "45033", "DORCHESTER": "45035",
        "EDGEFIELD": "45037", "FAIRFIELD": "45039", "FLORENCE": "45041",
        "GEORGETOWN": "45043", "GREENVILLE": "45045", "GREENWOOD": "45047",
        "HAMPTON": "45049", "HORRY": "45051", "JASPER": "45053",
        "KERSHAW": "45055", "LANCASTER": "45057", "LAURENS": "45059",
        "LEE": "45061", "LEXINGTON": "45063", "MCCORMICK": "45065",
        "MARION": "45067", "MARLBORO": "45069", "NEWBERRY": "45071",
        "OCONEE": "45073", "ORANGEBURG": "45075", "PICKENS": "45077",
        "RICHLAND": "45079", "SALUDA": "45081", "SPARTANBURG": "45083",
        "SUMTER": "45085", "UNION": "45087", "WILLIAMSBURG": "45089",
        "YORK": "45091",
    }

    # Use appropriate name lookup for current state
    _name_lookup = _nc_names if STATE_ABBR == "NC" else _sc_names

    all_records = []
    for uid, label in places_datasets:
        release_count = 0
        offset = 0
        limit = 50000
        while True:
            url = (
                f"https://data.cdc.gov/resource/{uid}.json"
                f"?%24where=stateabbr%3D%27{STATE_ABBR}%27"
                f"&%24limit={limit}&%24offset={offset}"
            )
            try:
                data = _fetch_json(url)
            except Exception as e:
                print(f"    [PLACES] {label} failed: {e}")
                break
            if not data:
                break

            # Some releases (2020) don't have locationid -- use county name
            for rec in data:
                loc_id = rec.get("locationid", "")
                if not loc_id or not county_fips_re.match(loc_id):
                    # Try to resolve FIPS from county name
                    cname = rec.get("locationname", "").strip().upper()
                    resolved = _name_lookup.get(cname, "")
                    if resolved:
                        rec["locationid"] = resolved
                        loc_id = resolved

                if county_fips_re.match(loc_id):
                    all_records.append(rec)
                    release_count += 1

            print(f"    [PLACES] {label}: fetched {len(data)}, "
                  f"kept {release_count} county records so far")
            if len(data) < limit:
                break
            offset += limit
        if release_count > 0:
            print(f"    [PLACES] {label} total: {release_count} county records")
        time.sleep(0.5)

    if not all_records:
        print("  [PLACES] No data retrieved!")
        return False

    print(f"  [PLACES] Total {STATE_ABBR} county records: {len(all_records)}")

    # Pivot: later releases overwrite earlier for overlapping year+measure.
    # We process datasets in release order (earliest first), so when a later
    # release has the same (FIPS, year, measure), it overwrites.
    pivot = {}
    for rec in all_records:
        loc_id = rec.get("locationid", "")
        year = rec.get("year", "")
        measure_id = rec.get("measureid", "")
        data_val_type = rec.get("datavaluetypeid", "")
        data_val = rec.get("data_value", "")
        county_name = rec.get("locationname", "")

        if not loc_id or not year or not measure_id:
            continue

        # Skip years outside our 2015-2023 panel window
        try:
            year_int = int(year)
            if year_int < 2015 or year_int > 2023:
                continue
        except (ValueError, TypeError):
            continue

        key = (loc_id, year)
        if key not in pivot:
            pivot[key] = {"FIPS": loc_id, "Year": year, "County": county_name}

        col_suffix = ("age-adjusted" if "age" in data_val_type.lower()
                       else "crude")
        col_name = f"{measure_id}_{col_suffix}_prevalence"
        pivot[key][col_name] = data_val

    rows = list(pivot.values())
    if not rows:
        print("  [PLACES] Pivot produced no rows!")
        return False

    all_cols = set()
    for r in rows:
        all_cols.update(r.keys())
    col_order = ["FIPS", "Year", "County"] + sorted(
        c for c in all_cols if c not in ("FIPS", "Year", "County")
    )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_order, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  [PLACES] Saved {len(rows)} county-year records to {out_path}")

    # Show year coverage
    year_counts = {}
    for r in rows:
        y = r.get("Year", "?")
        year_counts[y] = year_counts.get(y, 0) + 1
    for y in sorted(year_counts):
        print(f"    Year {y}: {year_counts[y]} counties")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 4. County Health Rankings
# ═══════════════════════════════════════════════════════════════════════════════

def download_chr():
    """
    Download County Health Rankings analytic data files for 2019-2025.

    URL pattern:
      https://www.countyhealthrankings.org/sites/default/files/media/document/
      analytic_data{year}.csv

    Filter to NC (stateFIPS = 37), extract key health measures.
    CHR column naming varies by year -- we extract by column position and
    known variable codes where possible.
    """
    chr_dir = RAW_DIR / "chr"
    chr_dir.mkdir(parents=True, exist_ok=True)

    # CHR releases: the 2025 release contains data reflecting ~2021-2023
    # The year in the filename is the RELEASE year, not the data year.
    chr_years = range(2017, 2026)  # 2017-2025 releases

    any_success = False
    for year in chr_years:
        out_path = chr_dir / f"chr_{STATE_ABBR.lower()}_{year}.csv"
        if out_path.exists():
            print(f"  [CHR] Already exists: {out_path}")
            any_success = True
            continue

        # CHR has changed URL patterns over time; try several
        # 2017-2018 used an older path (no /media/document/ prefix)
        # 2018 has a _0 suffix
        urls = [
            f"https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data{year}.csv",
            f"https://www.countyhealthrankings.org/sites/default/files/analytic_data{year}.csv",
            f"https://www.countyhealthrankings.org/sites/default/files/analytic_data{year}_0.csv",
            f"https://www.countyhealthrankings.org/sites/default/files/media/document/{year}%20County%20Health%20Rankings%20Data%20-%20v1.csv",
        ]

        downloaded = False
        for url in urls:
            try:
                print(f"  [CHR] Trying {year}: {url[:80]}...")
                raw_text = _fetch_csv_text(url)
                if len(raw_text) < 500:
                    continue

                # Parse and filter to NC
                # CHR files often have header rows -- detect the actual header
                lines = raw_text.split("\n")
                header_idx = 0
                for i, line in enumerate(lines[:10]):
                    if "FIPS" in line.upper() or "stateFIPS" in line.lower():
                        header_idx = i
                        break

                text_from_header = "\n".join(lines[header_idx:])
                reader = csv.DictReader(io.StringIO(text_from_header))

                state_rows = []
                for row in reader:
                    # Find state FIPS -- could be "stateFIPS", "State FIPS Code",
                    # "5-digit FIPS Code" starting with state prefix, etc.
                    row_st_fips = ""
                    fips_5 = ""
                    for key, val in row.items():
                        if key and "statefips" in key.lower().replace(" ", ""):
                            row_st_fips = str(val).strip()
                        if key and "fips" in key.lower() and "5" in key.lower():
                            fips_5 = str(val).strip().zfill(5)
                        if key and key.lower() == "fipscode":
                            fips_5 = str(val).strip().zfill(5)

                    is_target = (row_st_fips == STATE_FIPS or
                                 (fips_5 and fips_5.startswith(STATE_FIPS)))
                    if is_target:
                        state_rows.append(row)

                if state_rows:
                    fieldnames = list(state_rows[0].keys())
                    with open(out_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(state_rows)
                    print(f"    Saved {len(state_rows)} {STATE_ABBR} records for {year}")
                    any_success = True
                    downloaded = True
                    break
                else:
                    print(f"    No NC records found in {year} file")
            except Exception as e:
                print(f"    Failed: {e}")
                continue

        if not downloaded:
            print(f"  [CHR] Could not download {year} -- manual download may be needed")
            print(f"    URL: https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation")

        time.sleep(1)

    return any_success


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Stage existing local data
# ═══════════════════════════════════════════════════════════════════════════════

def stage_existing_data():
    """
    Copy existing data files from their current locations into data/raw/.

    - data/MA/2023_fy/ -> data/raw/ma_enrollment/
    - data/SVI/NC_SVI.csv -> data/raw/svi/svi_37_2020.csv
    - data/NC Medicaid Reports/ -> data/raw/medicaid_nc/
    - data/places/places.csv -> used by download_places()
    """
    print("\n  Staging existing local data...")

    # MA enrollment files (2022-2023)
    ma_src = Path("data/MA/2023_fy")
    ma_dst = RAW_DIR / "ma_enrollment"
    ma_dst.mkdir(parents=True, exist_ok=True)
    if ma_src.exists():
        copied = 0
        for f in ma_src.glob("SCC_Enrollment_MA_*.csv"):
            dst = ma_dst / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        # Also check extra_2023
        extra = Path("data/MA/extra_2023")
        if extra.exists():
            for f in extra.glob("*.csv"):
                dst = ma_dst / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
        print(f"    [MA] Staged {copied} enrollment files to {ma_dst}")
    else:
        print(f"    [MA] Source not found: {ma_src}")

    # SVI (existing is 2020 release)
    svi_src = Path("data/SVI/NC_SVI.csv")
    svi_dst = RAW_DIR / "svi" / "svi_37_2020.csv"
    if svi_src.exists() and not svi_dst.exists():
        svi_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(svi_src, svi_dst)
        print(f"    [SVI] Staged 2020 SVI to {svi_dst}")
    elif svi_dst.exists():
        print(f"    [SVI] Already staged: {svi_dst}")

    # NC Medicaid
    med_src = Path("data/NC Medicaid Reports")
    med_dst = RAW_DIR / "medicaid_nc"
    med_dst.mkdir(parents=True, exist_ok=True)
    if med_src.exists():
        copied = 0
        for f in med_src.glob("*.xlsx"):
            dst = med_dst / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        print(f"    [Medicaid] Staged {copied} files to {med_dst}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CDC SVI (direct CSV download)
# ═══════════════════════════════════════════════════════════════════════════════

def download_svi():
    """
    Download CDC SVI county-level data for 2014, 2016, 2018, 2022.

    Direct CSV URL pattern:
        https://svi.cdc.gov/Documents/Data/{YEAR}/csv/states_counties/
        SVI_{YEAR}_US_county.csv

    Downloads national file, filters to NC (STATE = 37), saves per-year.
    2020 is handled by stage_existing_data() from local NC_SVI.csv.
    """
    svi_dir = RAW_DIR / "svi"
    svi_dir.mkdir(parents=True, exist_ok=True)

    svi_years = [2014, 2016, 2018, 2022]
    any_success = False

    for year in svi_years:
        out_path = svi_dir / f"svi_{STATE_FIPS}_{year}.csv"
        if out_path.exists():
            print(f"  [SVI] Already exists: {out_path}")
            any_success = True
            continue

        url = (f"https://svi.cdc.gov/Documents/Data/{year}/csv/"
               f"states_counties/SVI_{year}_US_county.csv")
        print(f"  [SVI] Downloading {year} from {url[:60]}...")

        try:
            raw_text = _fetch_csv_text(url)
        except Exception as e:
            print(f"    [SVI] Failed for {year}: {e}")
            continue

        # Parse and filter to NC
        # Handle BOM if present
        if raw_text.startswith("\ufeff"):
            raw_text = raw_text[1:]
        reader = csv.DictReader(io.StringIO(raw_text))
        state_fullname = STATE_FULLNAME.get(STATE_ABBR, "")
        svi_rows = []
        fieldnames = None
        for row in reader:
            if fieldnames is None:
                fieldnames = list(row.keys())
            # SVI uses different column names across years
            is_target = False
            if row.get("ST_ABBR", "").strip() == STATE_ABBR:
                is_target = True
            elif state_fullname and row.get("STATE", "").strip() == state_fullname:
                is_target = True
            elif row.get("STATEFP", "").strip() == STATE_FIPS:
                is_target = True
            elif row.get("FIPS", "").strip().startswith(STATE_FIPS):
                is_target = True
            elif row.get("STCNTY", "").strip().startswith(STATE_FIPS):
                is_target = True

            if is_target:
                svi_rows.append(row)

        if svi_rows and fieldnames:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(svi_rows)
            print(f"    [SVI] Saved {len(svi_rows)} {STATE_ABBR} counties for {year}")
            any_success = True
        else:
            print(f"    [SVI] No {STATE_ABBR} records found in {year} file")

        time.sleep(1)

    return any_success


# ═══════════════════════════════════════════════════════════════════════════════
# 7. NC Medicaid enrollment (direct XLSX download)
# ═══════════════════════════════════════════════════════════════════════════════

def download_medicaid():
    """
    Download NC Medicaid annual unduplicated enrollment by county, SFY 2015-2022.

    Direct XLSX URLs from medicaid.ncdhhs.gov (found via page inspection).
    URL pattern varies slightly by year (plural/singular, -0 suffix).
    """
    med_dir = RAW_DIR / "medicaid_nc"
    med_dir.mkdir(parents=True, exist_ok=True)

    # Direct download URLs per SFY
    medicaid_urls = {
        2015: "https://medicaid.ncdhhs.gov/sfy2015annualunduplicatedenrollmentcountsbycountyandbudgetgroups/download?attachment",
        2016: "https://medicaid.ncdhhs.gov/sfy2016annualunduplicatedenrollmentcountsbycountyandbudgetgroups/download?attachment",
        2017: "https://medicaid.ncdhhs.gov/sfy2017annualunduplicatedenrollmentcountsbycountyandbudgetgroups/download?attachment",
        2018: "https://medicaid.ncdhhs.gov/sfy2018annualunduplicatedenrollmentcountsbycountyandbudgetgroups/download?attachment",
        2019: "https://medicaid.ncdhhs.gov/sfy2019annualunduplicatedenrollmentcountsbycountyandbudgetgroups/download?attachment",
        2020: "https://medicaid.ncdhhs.gov/sfy2020annualunduplicatedenrollmentcountsbycountyandbudgetgroups/download?attachment",
        2021: "https://medicaid.ncdhhs.gov/sfy2021annualunduplicatedenrollmentcountsbycountyandbudgetgroup-0/download?attachment",
        2022: "https://medicaid.ncdhhs.gov/sfy2022annualunduplicatedenrollmentcountsbycountyandbudgetgroup-0/download?attachment",
    }

    any_success = False
    for sfy, url in medicaid_urls.items():
        out_path = med_dir / f"medicaid_nc_sfy{sfy}.xlsx"
        if out_path.exists():
            print(f"  [Medicaid] Already exists: {out_path}")
            any_success = True
            continue

        print(f"  [Medicaid] Downloading SFY {sfy}...")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "NC-Hospital-Research/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if len(data) < 1000:
                    print(f"    [Medicaid] SFY {sfy}: file too small "
                          f"({len(data)} bytes), skipping")
                    continue
                with open(out_path, "wb") as f:
                    f.write(data)
                print(f"    [Medicaid] SFY {sfy}: saved ({len(data):,} bytes)")
                any_success = True
        except Exception as e:
            print(f"    [Medicaid] SFY {sfy} failed: {e}")

        time.sleep(0.5)

    return any_success


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CMS MA Enrollment (direct ZIP download, 2015-2021)
# ═══════════════════════════════════════════════════════════════════════════════

def download_ma_enrollment():
    """
    Download CMS Medicare Advantage monthly enrollment files (SCC level).

    Two URL patterns due to CMS website migration:
    - Old (2015 through ~Sept 2019):
      https://www.cms.gov/research-statistics-data-and-systems/.../downloads/
      {YYYY}/{month_abbrev}/scc-enrollment-ma-{YYYY}-{MM}.zip
    - New (~Oct 2019 through 2021):
      https://www.cms.gov/files/zip/
      ma-enrollment-statecountycontract-{month_full}-{YYYY}-full-version.zip

    Each ZIP contains a CSV with all-state enrollment at the contract level.
    """
    import zipfile

    ma_dir = RAW_DIR / "ma_enrollment"
    ma_dir.mkdir(parents=True, exist_ok=True)

    month_abbrev = ["jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec"]
    month_full = ["january", "february", "march", "april", "may", "june",
                  "july", "august", "september", "october", "november",
                  "december"]

    old_base = ("https://www.cms.gov/research-statistics-data-and-systems/"
                "statistics-trends-and-reports/mcradvpartdenroldata/downloads")
    new_base = "https://www.cms.gov/files/zip"

    # Which year-months do we already have?
    existing = set()
    for f in ma_dir.glob("SCC_Enrollment_MA_*.csv"):
        parts = f.stem.split("_")
        if len(parts) >= 5:
            try:
                existing.add((int(parts[3]), int(parts[4])))
            except ValueError:
                pass

    downloaded = 0
    failed = 0
    for year in range(2015, 2022):  # 2015-2021
        for month_idx in range(12):
            month_num = month_idx + 1
            if (year, month_num) in existing:
                continue

            mm = f"{month_num:02d}"
            out_csv = ma_dir / f"SCC_Enrollment_MA_{year}_{mm}.csv"
            if out_csv.exists():
                continue

            # Try URL patterns in order of likelihood
            urls_to_try = []

            if year < 2020 or (year == 2019 and month_num <= 9):
                # Old URL pattern first
                urls_to_try.append(
                    f"{old_base}/{year}/{month_abbrev[month_idx]}/"
                    f"scc-enrollment-ma-{year}-{mm}.zip"
                )
                # Try full month name variant
                urls_to_try.append(
                    f"{old_base}/{year}/{month_full[month_idx]}/"
                    f"scc-enrollment-ma-{year}-{mm}.zip"
                )
            # New URL pattern
            urls_to_try.append(
                f"{new_base}/ma-enrollment-statecountycontract-"
                f"{month_full[month_idx]}-{year}-full-version.zip"
            )
            # Old pattern as fallback for transition months
            if year >= 2019:
                urls_to_try.append(
                    f"{old_base}/{year}/{month_abbrev[month_idx]}/"
                    f"scc-enrollment-ma-{year}-{mm}.zip"
                )

            success = False
            for url in urls_to_try:
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "NC-Hospital-Research/1.0"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        zip_data = resp.read()

                    # Extract CSV from ZIP
                    zip_buf = io.BytesIO(zip_data)
                    with zipfile.ZipFile(zip_buf) as zf:
                        csv_names = [n for n in zf.namelist()
                                     if n.lower().endswith(".csv")]
                        if not csv_names:
                            continue
                        with zf.open(csv_names[0]) as cf:
                            with open(out_csv, "wb") as outf:
                                outf.write(cf.read())

                    downloaded += 1
                    success = True
                    break
                except (urllib.error.HTTPError, urllib.error.URLError):
                    continue
                except Exception as e:
                    print(f"    [MA] {year}-{mm} error: {e}")
                    continue

            if success:
                if downloaded % 12 == 0 or month_num == 12:
                    print(f"    [MA] Downloaded through {year}-{mm} "
                          f"({downloaded} files so far)")
            else:
                failed += 1
                if failed <= 5:
                    print(f"    [MA] {year}-{mm}: could not download")

            time.sleep(0.3)  # Be kind to CMS servers

    if downloaded > 0:
        print(f"  [MA] Downloaded {downloaded} new files "
              f"({failed} failed)")
    elif failed > 0:
        print(f"  [MA] All {failed} downloads failed")
    else:
        print(f"  [MA] All files already present")

    return downloaded > 0 or len(existing) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Manual download instructions (only for what's still missing)
# ═══════════════════════════════════════════════════════════════════════════════

def print_manual_instructions():
    """Print instructions for datasets that cannot be auto-downloaded."""

    # Check what's missing
    missing = []

    # MA enrollment: check how many year-months we have
    ma_dir = RAW_DIR / "ma_enrollment"
    ma_files = list(ma_dir.glob("SCC_Enrollment_MA_*.csv")) if ma_dir.exists() else []
    ma_years_have = set()
    for f in ma_files:
        parts = f.stem.split("_")
        # SCC_Enrollment_MA_YYYY_MM -> parts[3] is the year
        if len(parts) >= 5:
            try:
                yr = int(parts[3])
                if 2000 <= yr <= 2099:
                    ma_years_have.add(yr)
            except ValueError:
                pass
    ma_years_need = set(YEARS) - ma_years_have
    if ma_years_need:
        missing.append("MA")

    # SVI: check which releases we have
    svi_dir = RAW_DIR / "svi"
    svi_releases = [2014, 2016, 2018, 2020, 2022]
    svi_have = []
    svi_need = []
    for yr in svi_releases:
        if (svi_dir / f"svi_37_{yr}.csv").exists():
            svi_have.append(yr)
        else:
            svi_need.append(yr)
    if svi_need:
        missing.append("SVI")

    # Medicaid
    med_dir = RAW_DIR / "medicaid_nc"
    med_files = list(med_dir.glob("*.xlsx")) if med_dir.exists() else []
    if len(med_files) < 9:
        missing.append("Medicaid")

    # SEDD always needs manual
    missing.append("SEDD")

    if not missing:
        print("\n  All downloadable data is staged!")
        return

    print("\n" + "=" * 65)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 65)

    if "MA" in missing:
        print(f"""
A. CMS MEDICARE ADVANTAGE ENROLLMENT (need years: {sorted(ma_years_need)})
   -----------------------------------------------------------------------
   URL: https://www.cms.gov/data-research/statistics-trends-and-reports/
        medicare-advantagepart-d-contract-and-enrollment-data/
        monthly-ma-enrollment-state/county/contract

   Steps:
   1. Navigate to the page above
   2. For each month in {min(ma_years_need)}-{max(ma_years_need)}:
      - Find the "State/County/Contract" enrollment file
      - Download the CSV/ZIP for that month
   3. Save as: data/raw/ma_enrollment/SCC_Enrollment_MA_YYYY_MM.csv

   Note: You already have files for years: {sorted(ma_years_have)}
   Total files needed: ~{len(ma_years_need) * 12} (12 months x {len(ma_years_need)} years)
""")

    if "SVI" in missing:
        print(f"""
B. CDC SOCIAL VULNERABILITY INDEX (need releases: {svi_need})
   -----------------------------------------------------------
   URL: https://atsdr.cdc.gov/place-health/php/svi/
        svi-data-documentation-download.html

   Steps:
   1. Navigate to the page above
   2. For each release year ({', '.join(str(y) for y in svi_need)}):
      - Download the COUNTY-LEVEL CSV for that year
      - Filter to North Carolina (STATE = 37 or ST_ABBR = NC)
      - Save as: data/raw/svi/svi_37_YYYY.csv
   3. Required columns: FIPS, RPL_THEME1-4, EP_POV150, EP_UNEMP,
      EP_UNINSUR, EP_AGE65, EP_AGE17, EP_DISABL, EP_MINRTY,
      EP_MOBILE, EP_NOVEH, EP_NOINT, E_TOTPOP

   Note: You already have releases: {svi_have}
""")

    if "Medicaid" in missing:
        print(f"""
C. NC MEDICAID ENROLLMENT (need SFY 2015-2023, 9 annual files)
   ------------------------------------------------------------
   URL: https://medicaid.ncdhhs.gov/reports/nc-medicaid-enrollment-reports

   Steps:
   1. Navigate to the page above
   2. For each State Fiscal Year (SFY) 2015 through 2023:
      - Download "Annual Unduplicated Enrollment Counts by County"
      - Save as: data/raw/medicaid_nc/medicaid_nc_sfyYYYY.xlsx
   3. You already have: {len(med_files)} file(s) in {med_dir}

   Note: SFY 2011 and SFY 2014 may have gaps -- interpolate if needed.
""")

    if "SEDD" in missing:
        print("""
D. HCUP SEDD (State Emergency Department Databases)
   --------------------------------------------------
   Status: Requires Data Use Agreement (DUA) through AHRQ.
   URL: https://hcup-us.ahrq.gov/tech_assist/dua.jsp

   This is the TARGET VARIABLE for the panel model. Without SEDD,
   the panel can still be built with all predictor variables, but
   the target (residence-based ED visits) will be missing.

   NC SEDD available years: 2007-2023
   Estimated cost: $200-400 per state-year
""")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(state=None):
    global STATE_FIPS, STATE_ABBR

    if state:
        STATE_ABBR = state.upper()
        STATE_FIPS = STATE_FIPS_MAP.get(STATE_ABBR, "37")
    else:
        STATE_ABBR = "NC"
        STATE_FIPS = "37"

    print("=" * 65)
    print(f"Download & Stage Public Data Sources ({STATE_ABBR}, 2015-2023)")
    print("=" * 65)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Stage existing local data first ---
    if STATE_ABBR == "NC":
        print("\n--- STAGING EXISTING DATA ---")
        stage_existing_data()

    # --- Automated downloads ---
    print("\n--- AUTOMATED DOWNLOADS (APIs) ---\n")

    results = {}

    print("[1/7] Census SAIPE (poverty, income)")
    results["SAIPE"] = download_saipe()

    print("\n[2/7] BLS LAUS (unemployment)")
    results["BLS"] = download_bls_laus()

    print("\n[3/7] CDC PLACES (chronic disease)")
    results["PLACES"] = download_places()

    print("\n[4/7] County Health Rankings")
    results["CHR"] = download_chr()

    print("\n--- AUTOMATED DOWNLOADS (direct URLs) ---\n")

    print("[5/7] CDC SVI (social vulnerability)")
    results["SVI"] = download_svi()

    if STATE_ABBR == "NC":
        print("\n[6/7] NC Medicaid (enrollment)")
        results["Medicaid"] = download_medicaid()
    else:
        print(f"\n[6/7] {STATE_ABBR} Medicaid (enrollment)")
        print(f"  [Medicaid] {STATE_ABBR} Medicaid requires state-specific "
              f"data source -- see manual instructions")
        results["Medicaid"] = False

    print(f"\n[7/7] CMS MA Enrollment (monthly files, 2015-2021)")
    print("  (This may take a while — up to 84 ZIP files)")
    results["MA"] = download_ma_enrollment()

    # --- Manual instructions for anything still missing ---
    print_manual_instructions()

    # --- Summary ---
    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    for source, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {source:12s}  {status}")

    # Count what's available
    print(f"\nFiles in data/raw/:")
    for subdir in sorted(RAW_DIR.iterdir()):
        if subdir.is_dir():
            n = sum(1 for _ in subdir.glob("*") if _.is_file())
            print(f"  {subdir.name:20s}  {n} files")

    print(f"\nNext step: python -m src.data_acquisition.build_panel "
          f"--state {STATE_ABBR} --years 2015-2023")


# ═══════════════════════════════════════════════════════════════════════════════
# State Accessibility Assessment
# ═══════════════════════════════════════════════════════════════════════════════

def print_state_assessment():
    """Print summary of data accessibility by state for replication."""
    print("\n" + "=" * 80)
    print("STATE ACCESSIBILITY ASSESSMENT FOR REPLICATION")
    print("=" * 80)
    print(f"\n{'State':<6} {'SEDD/ED Data':<25} {'Predictors':<12} "
          f"{'Ease':<8} {'Notes'}")
    print("-" * 80)
    states = [
        ("NC", "SEDD 2007-2023 (DUA)", "Full", "High",
         "Current study"),
        ("SC", "SEDD 2000-2023 (DUA)", "Full", "High",
         "Best replication target"),
        ("CA", "Open HCAI data", "Full", "Medium",
         "Large N (58 counties)"),
        ("NY", "SPARCS open data", "Full", "Medium",
         "62 counties"),
        ("FL", "FloridaHealthFinder", "Full", "Medium",
         "67 counties"),
        ("VA", "No SEDD", "Full", "Low",
         "Ruled out -- no ED visit data"),
    ]
    for st, ed, pred, ease, notes in states:
        print(f"  {st:<4} {ed:<25} {pred:<12} {ease:<8} {notes}")
    print()


if __name__ == "__main__":
    import sys

    state = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--state" and i + 1 < len(sys.argv):
            state = sys.argv[i + 1].upper()
            i += 2
        elif sys.argv[i] == "--assess-states":
            print_state_assessment()
            sys.exit(0)
        elif sys.argv[i] == "--help":
            print("Download & Stage Public Data Sources")
            print("=" * 50)
            print()
            print("Usage:")
            print("  python -m src.data_acquisition.download_public_data")
            print("  python -m src.data_acquisition.download_public_data --state SC")
            print("  python -m src.data_acquisition.download_public_data --assess-states")
            print()
            print("Options:")
            print("  --state STATE     State abbreviation (default: NC)")
            print("  --assess-states   Print state data accessibility assessment")
            sys.exit(0)
        else:
            i += 1

    main(state=state)
