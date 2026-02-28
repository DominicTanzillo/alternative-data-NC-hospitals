"""
Multi-Year County Panel Builder
================================

Constructs a county x year panel dataset (100 NC counties x 9 years = 900 obs)
from downloaded public data sources for temporal ED burden modeling.

Data sources loaded from data/raw/:
    - Census SAIPE (poverty, income)             -- 2015-2023
    - BLS LAUS (unemployment)                     -- 2015-2023
    - CDC SVI (social vulnerability, biennial)    -- carry-forward interpolation
    - CDC PLACES (chronic disease)                -- 2015-2023 (10 SODA releases)
    - County Health Rankings (health measures)    -- 2019-2025
    - CMS MA Enrollment (monthly -> annual avg)   -- varies
    - NC Medicaid (annual enrollment)             -- varies
    - HCUP SEDD (target: residence-based ED)      -- when available via DUA

Usage
-----
    python -m src.data_acquisition.build_panel --state NC --years 2015-2023
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import csv
import io

warnings.filterwarnings("ignore")

# NC county FIPS codes (100 counties)
NC_COUNTY_FIPS = {
    "37001": "Alamance", "37003": "Alexander", "37005": "Alleghany",
    "37007": "Anson", "37009": "Ashe", "37011": "Avery",
    "37013": "Beaufort", "37015": "Bertie", "37017": "Bladen",
    "37019": "Brunswick", "37021": "Buncombe", "37023": "Burke",
    "37025": "Cabarrus", "37027": "Caldwell", "37029": "Camden",
    "37031": "Carteret", "37033": "Caswell", "37035": "Catawba",
    "37037": "Chatham", "37039": "Cherokee", "37041": "Chowan",
    "37043": "Clay", "37045": "Cleveland", "37047": "Columbus",
    "37049": "Craven", "37051": "Cumberland", "37053": "Currituck",
    "37055": "Dare", "37057": "Davidson", "37059": "Davie",
    "37061": "Duplin", "37063": "Durham", "37065": "Edgecombe",
    "37067": "Forsyth", "37069": "Franklin", "37071": "Gaston",
    "37073": "Gates", "37075": "Graham", "37077": "Granville",
    "37079": "Greene", "37081": "Guilford", "37083": "Halifax",
    "37085": "Harnett", "37087": "Haywood", "37089": "Henderson",
    "37091": "Hertford", "37093": "Hoke", "37095": "Hyde",
    "37097": "Iredell", "37099": "Jackson", "37101": "Johnston",
    "37103": "Jones", "37105": "Lee", "37107": "Lenoir",
    "37109": "Lincoln", "37111": "Macon", "37113": "Madison",
    "37115": "Martin", "37117": "McDowell", "37119": "Mecklenburg",
    "37121": "Mitchell", "37123": "Montgomery", "37125": "Moore",
    "37127": "Nash", "37129": "New Hanover", "37131": "Northampton",
    "37133": "Onslow", "37135": "Orange", "37137": "Pamlico",
    "37139": "Pasquotank", "37141": "Pender", "37143": "Perquimans",
    "37145": "Person", "37147": "Pitt", "37149": "Polk",
    "37151": "Randolph", "37153": "Richmond", "37155": "Robeson",
    "37157": "Rockingham", "37159": "Rowan", "37161": "Rutherford",
    "37163": "Sampson", "37165": "Scotland", "37167": "Stanly",
    "37169": "Stokes", "37171": "Surry", "37173": "Swain",
    "37175": "Transylvania", "37177": "Tyrrell", "37179": "Union",
    "37181": "Vance", "37183": "Wake", "37185": "Warren",
    "37187": "Washington", "37189": "Watauga", "37191": "Wayne",
    "37193": "Wilkes", "37195": "Wilson", "37197": "Yadkin",
    "37199": "Yancey",
}

STATE_FIPS = {"NC": "37", "SC": "45"}

# ---------------------------------------------------------------------------
# Individual data source loaders
# ---------------------------------------------------------------------------


def load_saipe(data_dir, state_fips="37", years=range(2015, 2024)):
    """
    Load Census SAIPE poverty and income estimates.

    Expected file: data_dir/saipe/saipe_nc_2015_2023.csv
    Columns: FIPS, Year, PovertyRate, MedianHouseholdIncome, ChildPovertyRate
    """
    path = Path(data_dir) / "saipe" / "saipe_nc_2015_2023.csv"
    if not path.exists():
        # Try generic naming
        path = Path(data_dir) / "saipe" / f"saipe_{state_fips}.csv"
    if not path.exists():
        print(f"  [SAIPE] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["Year"] = df["Year"].astype(int)
    for col in ["PovertyRate", "MedianHouseholdIncome", "ChildPovertyRate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Year"].isin(years)]
    print(f"  [SAIPE] Loaded {len(df)} county-year records "
          f"({df['Year'].nunique()} years)")
    return df[["FIPS", "Year"] + [c for c in
              ["PovertyRate", "MedianHouseholdIncome", "ChildPovertyRate"]
              if c in df.columns]]


def load_bls_unemployment(data_dir, state="NC", years=range(2015, 2024)):
    """
    Load BLS LAUS annual average unemployment by county.

    Expected file: data_dir/bls_laus/laus_annual_nc.csv
    Columns: FIPS, Year, UnemploymentRate
    """
    path = Path(data_dir) / "bls_laus" / f"laus_annual_{state.lower()}.csv"
    if not path.exists():
        print(f"  [BLS] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["Year"] = df["Year"].astype(int)
    df["UnemploymentRate"] = pd.to_numeric(df["UnemploymentRate"],
                                            errors="coerce")
    df = df[df["Year"].isin(years)]
    print(f"  [BLS] Loaded {len(df)} county-year records "
          f"({df['Year'].nunique()} years)")
    return df[["FIPS", "Year", "UnemploymentRate"]]


def load_svi_panel(data_dir, state_fips="37", years=range(2015, 2024)):
    """
    Load CDC SVI data and carry-forward interpolate for missing years.

    SVI is available for: 2014, 2016, 2018, 2020, 2022
    For other years, carries forward from the most recent available release.
    """
    svi_years = [2014, 2016, 2018, 2020, 2022]
    svi_dir = Path(data_dir) / "svi"

    # SVI columns we care about
    svi_theme_cols = ["RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4"]
    svi_demo_cols = [
        "EP_POV150", "EP_UNEMP", "EP_UNINSUR", "EP_AGE65", "EP_AGE17",
        "EP_DISABL", "EP_MINRTY", "EP_MOBILE", "EP_NOVEH", "EP_NOINT",
        "E_TOTPOP",
    ]
    svi_want = svi_theme_cols + svi_demo_cols

    all_svi = []
    for sy in svi_years:
        path = svi_dir / f"svi_{state_fips}_{sy}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)

        # Standardize FIPS column
        fips_col = None
        for candidate in ["FIPS", "fips", "STCNTY", "ST_CNTY", "FIPS_CODE"]:
            if candidate in df.columns:
                fips_col = candidate
                break
        if fips_col is None:
            # Try to find any column with 5-digit FIPS-like values
            for col in df.columns:
                sample = df[col].dropna().astype(str).str.strip()
                if sample.str.match(r"^37\d{3}$").mean() > 0.5:
                    fips_col = col
                    break
        if fips_col is None:
            print(f"    [SVI] Warning: no FIPS column found in {path.name}")
            continue

        df["FIPS"] = df[fips_col].astype(str).str.zfill(5)
        df["svi_year"] = sy

        # Keep only columns we want
        keep = ["FIPS", "svi_year"] + [c for c in svi_want if c in df.columns]
        all_svi.append(df[keep])

    if not all_svi:
        print(f"  [SVI] No SVI files found in {svi_dir}")
        return pd.DataFrame()

    svi_all = pd.concat(all_svi, ignore_index=True)
    available_years = sorted(svi_all["svi_year"].unique())
    print(f"  [SVI] Loaded releases: {available_years}")

    # Convert to numeric
    for col in svi_want:
        if col in svi_all.columns:
            svi_all[col] = pd.to_numeric(svi_all[col], errors="coerce")

    # Carry-forward interpolation
    panel_rows = []
    for target_year in years:
        valid = [y for y in available_years if y <= target_year]
        if not valid:
            valid = [min(available_years)]
        source_year = max(valid)
        chunk = svi_all[svi_all["svi_year"] == source_year].copy()
        chunk["Year"] = target_year
        panel_rows.append(chunk)

    result = pd.concat(panel_rows, ignore_index=True)
    result = result.drop(columns=["svi_year"])
    print(f"  [SVI] Built panel: {len(result)} county-year records "
          f"(carry-forward from {len(available_years)} releases)")
    return result


def load_places(data_dir, years=range(2015, 2024)):
    """
    Load CDC PLACES county-level health data (pivoted wide).

    Expected file: data_dir/places/places_nc.csv
    Already pivoted by download_public_data.py.
    """
    path = Path(data_dir) / "places" / "places_nc.csv"
    if not path.exists():
        print(f"  [PLACES] File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df["Year"] = df["Year"].astype(int)
    df = df[df["Year"].isin(years)]

    # Convert measure columns to numeric
    for col in df.columns:
        if col not in ("FIPS", "Year", "County"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  [PLACES] Loaded {len(df)} county-year records "
          f"(years: {sorted(df['Year'].unique())})")
    return df


def load_chr(data_dir, years=range(2015, 2024)):
    """
    Load County Health Rankings data files and extract key measures.

    CHR release year != data year. Approximate mapping:
        CHR 2019 release ~ data year 2017
        CHR 2020 release ~ data year 2018
        CHR 2021 release ~ data year 2019
        CHR 2022 release ~ data year 2020
        CHR 2023 release ~ data year 2021
        CHR 2024 release ~ data year 2022
        CHR 2025 release ~ data year 2023

    Key measures extracted (by looking for known CHR variable names):
        - Premature death (YPLL rate)
        - Poor or fair health %
        - Physically unhealthy days
        - Mentally unhealthy days
        - Primary care physicians rate
        - Uninsured %
        - Preventable hospital stays rate
    """
    chr_dir = Path(data_dir) / "chr"
    if not chr_dir.exists():
        print(f"  [CHR] Directory not found: {chr_dir}")
        return pd.DataFrame()

    # CHR release -> approximate data year mapping
    release_to_data_year = {
        2017: 2015, 2018: 2016,
        2019: 2017, 2020: 2018, 2021: 2019, 2022: 2020,
        2023: 2021, 2024: 2022, 2025: 2023,
    }

    all_chr = []
    for chr_file in sorted(chr_dir.glob("chr_nc_*.csv")):
        try:
            release_year = int(chr_file.stem.split("_")[-1])
        except ValueError:
            continue

        data_year = release_to_data_year.get(release_year, release_year - 2)
        if data_year not in years:
            continue

        try:
            df = pd.read_csv(chr_file, low_memory=False)
        except Exception as e:
            print(f"    [CHR] Error reading {chr_file.name}: {e}")
            continue

        # Find 5-digit FIPS column (search specific names first to avoid
        # matching "State FIPS Code" which only has 2-digit state codes)
        fips_col = None
        for candidate in ["5-digit FIPS Code", "fipscode",
                          "County FIPS Code"]:
            matches = [c for c in df.columns
                       if candidate.lower() == c.lower().strip()]
            if matches:
                fips_col = matches[0]
                break
        if fips_col is None:
            # Fallback: look for any column with 5-digit FIPS-like values
            for c in df.columns:
                sample = df[c].dropna().astype(str).str.strip()
                if len(sample) > 0 and sample.str.match(r"^37\d{3}$").mean() > 0.3:
                    fips_col = c
                    break
        if fips_col is None:
            print(f"    [CHR] No FIPS column in {chr_file.name}")
            continue

        df["FIPS"] = df[fips_col].astype(str).str.zfill(5)
        df = df[df["FIPS"].str.startswith("37")]
        # Exclude state-level rows (county FIPS = 37000)
        df = df[df["FIPS"] != "37000"]
        df["Year"] = data_year

        # Extract known measures by searching column names
        # CHR uses numeric variable codes (v001, v002, etc.)
        # and descriptive names vary by year
        measures = {}
        chr_vars = {
            "chr_ypll_rate": ["v001_rawvalue", "Years of Potential Life Lost Rate",
                              "Premature death"],
            "chr_poor_health_pct": ["v002_rawvalue", "Poor or fair health",
                                    "% Fair or Poor Health"],
            "chr_phys_unhealthy_days": ["v036_rawvalue",
                                        "Physically unhealthy days",
                                        "Average Number of Physically Unhealthy Days"],
            "chr_ment_unhealthy_days": ["v042_rawvalue",
                                        "Mentally unhealthy days",
                                        "Average Number of Mentally Unhealthy Days"],
            "chr_pcp_rate": ["v004_rawvalue", "Primary care physicians",
                             "PCP Rate"],
            "chr_uninsured_pct": ["v085_rawvalue", "Uninsured",
                                  "% Uninsured"],
            "chr_preventable_hosp": ["v005_rawvalue",
                                     "Preventable hospital stays",
                                     "Preventable Hospitalization Rate"],
        }

        for out_col, search_terms in chr_vars.items():
            for term in search_terms:
                matches = [c for c in df.columns
                           if term.lower() in c.lower()]
                if matches:
                    measures[out_col] = matches[0]
                    break

        # Build output for this year
        keep = {"FIPS": df["FIPS"], "Year": df["Year"]}
        for out_col, src_col in measures.items():
            keep[out_col] = pd.to_numeric(df[src_col], errors="coerce")

        year_df = pd.DataFrame(keep)
        all_chr.append(year_df)
        print(f"    [CHR] {chr_file.name} -> data year {data_year}: "
              f"{len(year_df)} counties, {len(measures)} measures")

    if not all_chr:
        print(f"  [CHR] No CHR data loaded")
        return pd.DataFrame()

    result = pd.concat(all_chr, ignore_index=True)
    print(f"  [CHR] Total: {len(result)} county-year records")
    return result


def load_ma_enrollment(data_dir, state_fips="37", years=range(2015, 2024)):
    """
    Load CMS MA enrollment files, filter to NC, compute annual averages.

    Expected: data_dir/ma_enrollment/SCC_Enrollment_MA_YYYY_MM.csv
    File format: County, State, Contract ID, ..., FIPS Code, Enrolled
    """
    ma_dir = Path(data_dir) / "ma_enrollment"
    if not ma_dir.exists():
        print(f"  [MA] Directory not found: {ma_dir}")
        return pd.DataFrame()

    all_monthly = []
    for year in years:
        year_files = sorted(ma_dir.glob(f"*{year}*.csv"))
        for f in year_files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception as e:
                print(f"    [MA] Warning: Could not read {f.name}: {e}")
                continue

            # Standardize FIPS column
            fips_col = None
            for candidate in ["FIPS Code", "FIPS_Code", "fips", "FIPS"]:
                if candidate in df.columns:
                    fips_col = candidate
                    break
            if fips_col is None:
                continue

            df["FIPS"] = df[fips_col].astype(str).str.zfill(5)
            df = df[df["FIPS"].str.startswith(state_fips)]

            # Find enrollment column
            enroll_col = None
            for candidate in ["Enrolled", "Enrollment", "enrolled"]:
                if candidate in df.columns:
                    enroll_col = candidate
                    break
            if enroll_col is None:
                continue

            df["Enrolled"] = pd.to_numeric(
                df[enroll_col].astype(str).str.replace(",", "").str.strip(),
                errors="coerce"
            )

            # Aggregate by county (sum across contracts/plans)
            county_totals = (df.groupby("FIPS")["Enrolled"]
                              .sum().reset_index())
            county_totals["Year"] = year
            county_totals["Month"] = f.stem  # for tracking
            all_monthly.append(county_totals)

    if not all_monthly:
        print(f"  [MA] No enrollment files found")
        return pd.DataFrame()

    monthly = pd.concat(all_monthly, ignore_index=True)

    # Compute annual averages per county
    monthly["FIPS"] = monthly["FIPS"].astype(str).str.replace(r"\.0$", "",
                                                               regex=True).str.zfill(5)
    annual = (monthly.groupby(["FIPS", "Year"])["Enrolled"]
              .mean().reset_index()
              .rename(columns={"Enrolled": "MA_Enrollment_Annual"}))

    annual["MA_Enrollment_Annual"] = annual["MA_Enrollment_Annual"].round(0)
    print(f"  [MA] Loaded {len(annual)} county-year records "
          f"(from {len(all_monthly)} monthly files)")
    return annual


def load_medicaid(data_dir, years=range(2015, 2024)):
    """
    Load NC Medicaid enrollment files.

    Expected: data_dir/medicaid_nc/medicaid_nc_sfyYYYY.xlsx
    Or any .xlsx files in the directory.
    """
    med_dir = Path(data_dir) / "medicaid_nc"
    if not med_dir.exists():
        print(f"  [Medicaid] Directory not found: {med_dir}")
        return pd.DataFrame()

    xlsx_files = list(med_dir.glob("*.xlsx"))
    if not xlsx_files:
        print(f"  [Medicaid] No .xlsx files found in {med_dir}")
        return pd.DataFrame()

    all_med = []
    for f in xlsx_files:
        try:
            # Try to detect the year from filename
            year = None
            fname = f.stem.lower()
            for y in range(2010, 2030):
                if str(y) in fname:
                    year = y
                    break

            df = pd.read_excel(f, engine="openpyxl")

            # Look for county and enrollment count columns
            # Medicaid files vary in format
            county_col = None
            for candidate in df.columns:
                if "county" in str(candidate).lower():
                    county_col = candidate
                    break

            if county_col is None:
                continue

            # Find numeric enrollment column (largest values)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue

            # Use the first large numeric column as enrollment
            enroll_col = numeric_cols[0]
            for col in numeric_cols:
                if "total" in str(col).lower() or "unduplicated" in str(col).lower():
                    enroll_col = col
                    break

            result = df[[county_col, enroll_col]].copy()
            result.columns = ["County", "Medicaid_Enrollment"]
            result["Medicaid_Enrollment"] = pd.to_numeric(
                result["Medicaid_Enrollment"], errors="coerce"
            )
            if year:
                result["Year"] = year

            all_med.append(result)
            print(f"    [Medicaid] {f.name}: {len(result)} rows"
                  f"{f' (SFY {year})' if year else ''}")

        except Exception as e:
            print(f"    [Medicaid] Error reading {f.name}: {e}")
            continue

    if not all_med:
        print(f"  [Medicaid] No data loaded")
        return pd.DataFrame()

    result = pd.concat(all_med, ignore_index=True)
    # Medicaid files use county names, not FIPS -- we'll merge by name
    print(f"  [Medicaid] Total: {len(result)} records")
    return result


def load_sedd_panel(data_dir, state_fips="37", years=range(2015, 2024)):
    """
    Load HCUP SEDD county totals for multiple years.

    Expected: data_dir/sedd/sedd_{state_fips}_{year}.csv
    Or a single combined file: data_dir/sedd/sedd_{state_fips}_all.csv
    """
    sedd_dir = Path(data_dir) / "sedd"
    if not sedd_dir.exists():
        print(f"  [SEDD] Directory not found (DUA required)")
        return pd.DataFrame()

    # Check for combined file first
    combined = sedd_dir / f"sedd_{state_fips}_all.csv"
    if combined.exists():
        df = pd.read_csv(combined)
        df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
        df = df[df["Year"].isin(years)]
        print(f"  [SEDD] Loaded combined file: {len(df)} county-year records")
        return df

    # Try individual year files
    all_years = []
    for year in years:
        year_files = list(sedd_dir.glob(f"*{state_fips}*{year}*"))
        if year_files:
            try:
                from src.database_read.load_sedd import load_sedd_county_totals
                counts = load_sedd_county_totals(
                    year_files[0], state_fips=state_fips, year=year
                )
                counts["Year"] = year
                all_years.append(counts)
            except ImportError:
                df = pd.read_csv(year_files[0])
                df["Year"] = year
                all_years.append(df)

    if not all_years:
        print(f"  [SEDD] No SEDD files found (DUA required)")
        return pd.DataFrame()

    result = pd.concat(all_years, ignore_index=True)
    print(f"  [SEDD] Built panel: {len(result)} county-year records")
    return result


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------


def build_county_year_panel(
    data_dir="data/raw",
    state="NC",
    years=range(2015, 2024),
    output_path=None,
):
    """
    Assemble the master county x year panel from all data sources.

    Parameters
    ----------
    data_dir : str
        Root directory containing raw data subdirectories.
    state : str
        State abbreviation (NC or SC).
    years : range or list
        Years to include in the panel.
    output_path : str, optional
        Path to save the assembled panel CSV.

    Returns
    -------
    pd.DataFrame
        Master panel with columns: FIPS, County, Year, + all features.
    """
    state_fips = STATE_FIPS.get(state, "37")
    fips_map = NC_COUNTY_FIPS if state == "NC" else {}
    years = list(years)

    print(f"Building county-year panel for {state} ({min(years)}-{max(years)})")
    print(f"{'=' * 65}")

    # Start with county x year scaffold
    scaffold = pd.DataFrame([
        {"FIPS": fips, "County": name, "Year": year}
        for fips, name in fips_map.items()
        for year in years
    ])
    print(f"\nScaffold: {len(scaffold)} county-year observations "
          f"({len(fips_map)} counties x {len(years)} years)")

    panel = scaffold.copy()

    # --- Load each source ---
    print(f"\nLoading data sources:")

    # 1. SAIPE (poverty, income)
    saipe = load_saipe(data_dir, state_fips=state_fips, years=years)
    if not saipe.empty:
        panel = panel.merge(saipe, on=["FIPS", "Year"], how="left")

    # 2. BLS Unemployment
    bls = load_bls_unemployment(data_dir, state=state, years=years)
    if not bls.empty:
        panel = panel.merge(bls, on=["FIPS", "Year"], how="left")

    # 3. SVI (with carry-forward)
    svi = load_svi_panel(data_dir, state_fips=state_fips, years=years)
    if not svi.empty:
        svi_cols = [c for c in svi.columns if c not in ("FIPS", "Year", "County")]
        panel = panel.merge(svi[["FIPS", "Year"] + svi_cols],
                            on=["FIPS", "Year"], how="left")

    # 4. CDC PLACES
    places = load_places(data_dir, years=years)
    if not places.empty:
        places_cols = [c for c in places.columns
                       if c not in ("FIPS", "Year", "County")]
        panel = panel.merge(places[["FIPS", "Year"] + places_cols],
                            on=["FIPS", "Year"], how="left")

    # 5. County Health Rankings
    chr_df = load_chr(data_dir, years=years)
    if not chr_df.empty:
        chr_cols = [c for c in chr_df.columns
                    if c not in ("FIPS", "Year", "County")]

        # Cross-walk: for years where both PLACES and CHR are available,
        # prefer PLACES for overlapping health measures (more granular).
        # CHR fills in for years PLACES doesn't cover (pre-2022).
        panel = panel.merge(chr_df[["FIPS", "Year"] + chr_cols],
                            on=["FIPS", "Year"], how="left")

    # 6. MA Enrollment
    ma = load_ma_enrollment(data_dir, state_fips=state_fips, years=years)
    if not ma.empty:
        panel = panel.merge(ma, on=["FIPS", "Year"], how="left")

    # 7. NC Medicaid (merge by county name since Medicaid uses names)
    medicaid = load_medicaid(data_dir, years=years)
    if not medicaid.empty and "Year" in medicaid.columns:
        # Normalize county names for merge -- use upper-case lookup to
        # avoid title-case bugs (e.g., "MCDOWELL".title() -> "Mcdowell"
        # but panel has "McDowell").
        fips_to_name = {v: k for k, v in (fips_map or {}).items()}
        name_upper_to_panel = {name.upper(): name
                               for name in panel["County"].unique()}
        medicaid["County"] = (
            medicaid["County"].astype(str).str.strip().str.upper()
            .map(name_upper_to_panel)
        )
        medicaid = medicaid.dropna(subset=["County"])
        panel_med = panel.merge(
            medicaid[["County", "Year", "Medicaid_Enrollment"]],
            on=["County", "Year"], how="left"
        )
        if "Medicaid_Enrollment" in panel_med.columns:
            panel = panel_med

    # 8. SEDD (target variable)
    sedd = load_sedd_panel(data_dir, state_fips=state_fips, years=years)
    if not sedd.empty:
        sedd_cols = [c for c in sedd.columns
                     if c not in ("FIPS", "Year", "County")]
        panel = panel.merge(sedd[["FIPS", "Year"] + sedd_cols],
                            on=["FIPS", "Year"], how="left")

    # --- Coverage report ---
    print(f"\n{'=' * 65}")
    print(f"PANEL ASSEMBLED: {panel.shape[0]} rows x {panel.shape[1]} columns")
    print(f"{'=' * 65}")

    non_scaffold = [c for c in panel.columns
                    if c not in ["FIPS", "County", "Year"]]

    print(f"\nCoverage by variable:")
    print(f"{'Variable':<45} {'% Available':>10}  {'Years':>20}")
    print("-" * 80)

    for col in non_scaffold:
        pct_avail = panel[col].notna().mean() * 100
        # Which years have data for this column?
        years_with_data = sorted(
            panel.loc[panel[col].notna(), "Year"].unique()
        )
        if len(years_with_data) <= 5:
            year_str = ", ".join(str(y) for y in years_with_data)
        else:
            year_str = f"{min(years_with_data)}-{max(years_with_data)}"
        print(f"  {col:<43} {pct_avail:>8.0f}%  {year_str:>20}")

    # Summary by year
    print(f"\nCoverage by year:")
    print(f"{'Year':<8} {'Variables with data':>20}")
    print("-" * 35)
    for year in sorted(years):
        year_slice = panel[panel["Year"] == year][non_scaffold]
        n_vars = sum(year_slice[col].notna().any() for col in non_scaffold)
        print(f"  {year:<6} {n_vars:>18} / {len(non_scaffold)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return panel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    state = "NC"
    year_range = range(2015, 2024)
    data_dir = "data/raw"
    output = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--state" and i + 1 < len(sys.argv):
            state = sys.argv[i + 1].upper()
            i += 2
        elif sys.argv[i] == "--years" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split("-")
            year_range = range(int(parts[0]), int(parts[1]) + 1)
            i += 2
        elif sys.argv[i] == "--data-dir" and i + 1 < len(sys.argv):
            data_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--help":
            print("Multi-Year County Panel Builder")
            print("=" * 50)
            print()
            print("Usage:")
            print("  python -m src.data_acquisition.build_panel "
                  "--state NC --years 2015-2023")
            print()
            print("Options:")
            print("  --state STATE    State abbreviation (default: NC)")
            print("  --years RANGE    Year range (default: 2015-2023)")
            print("  --data-dir DIR   Raw data directory (default: data/raw)")
            print("  --output PATH    Output CSV path "
                  "(default: data/processed/panel_{state}.csv)")
            sys.exit(0)
        else:
            i += 1

    if output is None:
        output = f"data/processed/panel_{state.lower()}.csv"

    build_county_year_panel(
        data_dir=data_dir,
        state=state,
        years=year_range,
        output_path=output,
    )
