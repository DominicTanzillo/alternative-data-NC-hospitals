"""
HCUP State Emergency Department Databases (SEDD) — County Aggregation
=====================================================================

Aggregates visit-level HCUP SEDD records into county-level ED visit counts
for use in the excess burden pipeline.

SEDD Data Access
----------------
HCUP SEDD files are available through the Healthcare Cost and Utilization
Project (AHRQ) via data use agreements. At Duke University, access is through
the Duke University Health System HCUP Data Use Agreement.

Key SEDD Variables Used
-----------------------
- PSTCO     : Patient state/county FIPS code (based on patient residence)
- PSTCO_GEO : Alternative assignment based on ZIP centroid (when PSTCO
              unavailable from source data)
- YEAR      : Discharge year
- NEDS_STRATUM / HOESSION_ED : ED visit identifiers

State Availability (as of 2024)
-------------------------------
- North Carolina: 2007–2023
- South Carolina: 2006–2023
- Virginia: NOT available in SEDD

Critical Note
-------------
SEDD uses PSTCO (patient RESIDENCE county), unlike NC DETECT/SHEPS which count
visits at hospitals IN a county. This resolves the catchment-area problem where
counties with regional medical centers show artificially high rates.

Usage
-----
    from src.database_read.load_sedd import load_sedd_county_totals

    # From a single CSV (pre-exported from SAS/Stata)
    ed_counts = load_sedd_county_totals("data/SEDD/nc_sedd_2021.csv")

    # From multiple years
    ed_counts = load_sedd_county_totals(
        "data/SEDD/nc_sedd_2021.csv",
        year=2021,
        state_fips="37",
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path


# NC county FIPS → county name mapping (FIPS prefix 37 = North Carolina)
NC_FIPS_TO_COUNTY = {
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

# SC county FIPS → county name mapping (FIPS prefix 45 = South Carolina)
SC_FIPS_TO_COUNTY = {
    "45001": "Abbeville", "45003": "Aiken", "45005": "Allendale",
    "45007": "Anderson", "45009": "Bamberg", "45011": "Barnwell",
    "45013": "Beaufort", "45015": "Berkeley", "45017": "Calhoun",
    "45019": "Charleston", "45021": "Cherokee", "45023": "Chester",
    "45025": "Chesterfield", "45027": "Clarendon", "45029": "Colleton",
    "45031": "Darlington", "45033": "Dillon", "45035": "Dorchester",
    "45037": "Edgefield", "45039": "Fairfield", "45041": "Florence",
    "45043": "Georgetown", "45045": "Greenville", "45047": "Greenwood",
    "45049": "Hampton", "45051": "Horry", "45053": "Jasper",
    "45055": "Kershaw", "45057": "Lancaster", "45059": "Laurens",
    "45061": "Lee", "45063": "Lexington", "45065": "McCormick",
    "45067": "Marion", "45069": "Marlboro", "45071": "Newberry",
    "45073": "Oconee", "45075": "Orangeburg", "45077": "Pickens",
    "45079": "Richland", "45081": "Saluda", "45083": "Spartanburg",
    "45085": "Sumter", "45087": "Union", "45089": "Williamsburg",
    "45091": "York",
}

STATE_FIPS_MAPS = {
    "37": NC_FIPS_TO_COUNTY,
    "45": SC_FIPS_TO_COUNTY,
}


def load_sedd_county_totals(
    filepath,
    state_fips="37",
    year=None,
    pstco_col="PSTCO",
    fallback_col="PSTCO_GEO",
):
    """
    Aggregate SEDD visit-level records to county-level ED visit counts.

    Parameters
    ----------
    filepath : str or Path
        Path to a CSV (or similar tabular file) exported from the SEDD
        SAS/Stata/SPSS dataset. The file should contain at minimum the
        patient county FIPS column.
    state_fips : str
        Two-digit state FIPS code to filter to (default "37" for NC).
        Records with PSTCO outside this state are counted separately
        as out-of-state visits.
    year : int, optional
        If provided, filter to this discharge year (using YEAR column).
    pstco_col : str
        Column name for patient state/county FIPS code.
    fallback_col : str
        Column name for alternative FIPS assignment (ZIP-centroid based).
        Used when pstco_col is missing for a record.

    Returns
    -------
    pd.DataFrame
        Columns: County, FIPS, Total_ED, Total_ED_Residence
        - Total_ED_Residence: ED visits by county of patient residence
        - One row per county (including counties with 0 visits)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"SEDD file not found: {filepath}\n"
            f"HCUP SEDD data must be obtained through a Data Use Agreement.\n"
            f"See: https://hcup-us.ahrq.gov/tech_assist/dua.jsp"
        )

    # Read the data
    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath, low_memory=False)
    elif filepath.suffix in (".dta", ".DTA"):
        df = pd.read_stata(filepath)
    elif filepath.suffix in (".sas7bdat",):
        df = pd.read_sas(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False)

    print(f"Loaded {len(df):,} SEDD records from {filepath.name}")

    # Resolve FIPS: prefer PSTCO, fall back to PSTCO_GEO
    if pstco_col in df.columns:
        fips = df[pstco_col].astype(str).str.zfill(5)
    elif fallback_col in df.columns:
        print(f"  Warning: {pstco_col} not found, using {fallback_col}")
        fips = df[fallback_col].astype(str).str.zfill(5)
    else:
        raise ValueError(
            f"Neither {pstco_col} nor {fallback_col} found in data. "
            f"Available columns: {list(df.columns[:20])}..."
        )

    df["_fips"] = fips

    # Filter by year if specified
    if year is not None and "YEAR" in df.columns:
        df = df[df["YEAR"] == year].copy()
        print(f"  Filtered to year {year}: {len(df):,} records")

    # Filter to in-state records
    in_state = df[df["_fips"].str[:2] == state_fips].copy()
    out_of_state = len(df) - len(in_state)
    print(f"  In-state records: {len(in_state):,}")
    if out_of_state > 0:
        print(f"  Out-of-state records excluded: {out_of_state:,}")

    # Aggregate by county FIPS
    county_counts = (
        in_state.groupby("_fips")
        .size()
        .reset_index(name="Total_ED_Residence")
    )

    # Map FIPS to county names
    fips_map = STATE_FIPS_MAPS.get(state_fips, {})
    county_counts["County"] = county_counts["_fips"].map(fips_map)
    county_counts.rename(columns={"_fips": "FIPS"}, inplace=True)

    # Ensure all counties in the state are represented (even with 0 visits)
    all_fips = pd.DataFrame(
        [(fips, name) for fips, name in fips_map.items()],
        columns=["FIPS", "County"],
    )
    result = all_fips.merge(
        county_counts[["FIPS", "Total_ED_Residence"]],
        on="FIPS",
        how="left",
    )
    result["Total_ED_Residence"] = result["Total_ED_Residence"].fillna(0).astype(int)

    # Also provide Total_ED alias for pipeline compatibility
    result["Total_ED"] = result["Total_ED_Residence"]

    result = result.sort_values("County").reset_index(drop=True)

    print(f"\n  County-level summary:")
    print(f"    Counties with ED visits: {(result['Total_ED'] > 0).sum()}")
    print(f"    Counties with 0 visits:  {(result['Total_ED'] == 0).sum()}")
    print(f"    Total ED visits:         {result['Total_ED'].sum():,}")
    print(f"    Mean per county:         {result['Total_ED'].mean():,.0f}")
    print(f"    Median per county:       {result['Total_ED'].median():,.0f}")

    return result


def merge_sedd_with_predictors(sedd_counts, predictor_path, county_col="County"):
    """
    Merge SEDD county-level ED counts with the predictor dataset.

    This replaces the facility-based Total_ED in merged_county_data.csv
    with residence-based Total_ED_Residence from SEDD.

    Parameters
    ----------
    sedd_counts : pd.DataFrame
        Output of load_sedd_county_totals().
    predictor_path : str or Path
        Path to the merged county predictor CSV (e.g., merged_county_data.csv).
    county_col : str
        Column to merge on.

    Returns
    -------
    pd.DataFrame
        Full merged dataset with residence-based ED counts.
    """
    predictors = pd.read_csv(predictor_path)

    # Drop existing facility-based ED columns
    facility_ed_cols = [
        c for c in predictors.columns
        if c.startswith("Payer_") or c.startswith("Point_of_Origin_")
        or c.startswith("Race_") or c.startswith("Age_Group_")
        or c == "Total_ED"
    ]
    predictors = predictors.drop(columns=facility_ed_cols, errors="ignore")

    # Merge residence-based ED counts
    merged = predictors.merge(
        sedd_counts[[county_col, "Total_ED_Residence", "Total_ED", "FIPS"]],
        on=county_col,
        how="left",
    )

    n_matched = merged["Total_ED"].notna().sum()
    print(f"Merged {n_matched}/{len(merged)} counties with SEDD ED counts")

    return merged


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("HCUP SEDD County Aggregation Tool")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python -m src.database_read.load_sedd <sedd_file> [options]")
        print()
        print("Arguments:")
        print("  sedd_file          Path to exported SEDD CSV/DTA/SAS file")
        print()
        print("Options:")
        print("  --state FIPS       State FIPS code (default: 37 for NC)")
        print("  --year YEAR        Filter to specific year")
        print("  --merge PATH       Merge with predictor CSV and save")
        print("  --output PATH      Output path for county totals CSV")
        print()
        print("Examples:")
        print("  # NC, all years:")
        print("  python -m src.database_read.load_sedd data/SEDD/nc_sedd.csv")
        print()
        print("  # NC, 2021 only, merge with predictors:")
        print("  python -m src.database_read.load_sedd data/SEDD/nc_sedd.csv \\")
        print("      --year 2021 --merge data/final/merged_county_data.csv")
        print()
        print("  # SC:")
        print("  python -m src.database_read.load_sedd data/SEDD/sc_sedd.csv \\")
        print("      --state 45")
        print()
        print("Note: HCUP SEDD data requires a Data Use Agreement.")
        print("See: https://hcup-us.ahrq.gov/tech_assist/dua.jsp")
        sys.exit(0)

    sedd_file = sys.argv[1]
    state = "37"
    year = None
    merge_path = None
    output_path = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--state" and i + 1 < len(sys.argv):
            state = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--year" and i + 1 < len(sys.argv):
            year = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--merge" and i + 1 < len(sys.argv):
            merge_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Load and aggregate
    counts = load_sedd_county_totals(sedd_file, state_fips=state, year=year)

    # Save county totals
    if output_path is None:
        output_path = f"data/final/sedd_county_totals_{state}.csv"
    counts.to_csv(output_path, index=False)
    print(f"\nSaved county totals to {output_path}")

    # Optionally merge with predictors
    if merge_path:
        merged = merge_sedd_with_predictors(counts, merge_path)
        merged_out = output_path.replace(".csv", "_merged.csv")
        merged.to_csv(merged_out, index=False)
        print(f"Saved merged dataset to {merged_out}")
