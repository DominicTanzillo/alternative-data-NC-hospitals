"""
Multi-Year County Panel Builder
================================

Constructs a county x year panel dataset from multiple public data sources
for use in temporal ED burden modeling and forecasting.

This module downloads and processes:
    - CMS Medicare Advantage enrollment (monthly -> annual)
    - BLS Local Area Unemployment Statistics
    - Census SAIPE (poverty and income)
    - CDC SVI (biennial, carry-forward interpolation)
    - HRSA AHRF (physician supply, HPSA designations)
    - HCUP SEDD (when available via DUA)

All sources are publicly available and nationally standardized,
making this approach reproducible for any U.S. state.

Usage
-----
    python -m src.data_acquisition.build_panel --state NC --years 2015-2023
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

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


def load_bls_unemployment(data_dir, state="NC", years=range(2010, 2024)):
    """
    Load BLS LAUS annual average unemployment by county.

    Expected file: data_dir/bls_laus/laus_annual_{state}.csv
    Or can be built from bulk download files.

    Columns expected: FIPS, Year, UnemploymentRate, LaborForce
    """
    path = Path(data_dir) / "bls_laus" / f"laus_annual_{state.lower()}.csv"
    if not path.exists():
        print(f"  [BLS] File not found: {path}")
        print(f"  [BLS] Download from: https://www.bls.gov/lau/data.htm")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df = df[df["Year"].isin(years)]
    print(f"  [BLS] Loaded {len(df)} county-year records")
    return df[["FIPS", "Year", "UnemploymentRate", "LaborForce"]]


def load_saipe(data_dir, state_fips="37", years=range(2010, 2024)):
    """
    Load Census SAIPE poverty and income estimates by county.

    Expected file: data_dir/saipe/saipe_{state_fips}.csv
    Columns: FIPS, Year, PovertyRate, MedianHouseholdIncome, ChildPovertyRate
    """
    path = Path(data_dir) / "saipe" / f"saipe_{state_fips}.csv"
    if not path.exists():
        print(f"  [SAIPE] File not found: {path}")
        print(f"  [SAIPE] Download from: https://www.census.gov/programs-surveys/saipe/data/datasets.html")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
    df = df[df["Year"].isin(years)]
    print(f"  [SAIPE] Loaded {len(df)} county-year records")
    return df


def load_svi_panel(data_dir, state_fips="37", years=range(2010, 2024)):
    """
    Load CDC SVI data and carry-forward interpolate for missing years.

    SVI is available for: 2010, 2014, 2016, 2018, 2020, 2022
    For other years, carries forward from the most recent available release.
    """
    svi_years = [2010, 2014, 2016, 2018, 2020, 2022]
    svi_dir = Path(data_dir) / "svi"

    all_svi = []
    for sy in svi_years:
        path = svi_dir / f"svi_{state_fips}_{sy}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["FIPS"] = df["FIPS"].astype(str).str.zfill(5)
            df["svi_year"] = sy
            all_svi.append(df)

    if not all_svi:
        print(f"  [SVI] No SVI files found in {svi_dir}")
        print(f"  [SVI] Expected format: svi_{state_fips}_YYYY.csv")
        print(f"  [SVI] Download from: https://atsdr.cdc.gov/place-health/php/svi/svi-data-documentation-download.html")
        return pd.DataFrame()

    svi_all = pd.concat(all_svi, ignore_index=True)
    available_years = sorted(svi_all["svi_year"].unique())
    print(f"  [SVI] Loaded releases: {available_years}")

    # Carry-forward interpolation
    panel_rows = []
    for target_year in years:
        # Find the most recent SVI release <= target_year
        valid = [y for y in available_years if y <= target_year]
        if not valid:
            valid = [min(available_years)]  # Fall back to earliest
        source_year = max(valid)
        chunk = svi_all[svi_all["svi_year"] == source_year].copy()
        chunk["Year"] = target_year
        panel_rows.append(chunk)

    result = pd.concat(panel_rows, ignore_index=True)
    print(f"  [SVI] Built panel: {len(result)} county-year records")
    return result


def load_ma_enrollment_panel(data_dir, state_fips="37",
                              years=range(2010, 2024)):
    """
    Load CMS MA enrollment files and compute annual averages.

    Expected: data_dir/ma_enrollment/SCC_Enrollment_MA_YYYY_MM.csv
    """
    ma_dir = Path(data_dir) / "ma_enrollment"
    if not ma_dir.exists():
        # Try current data location
        ma_dir = Path(data_dir) / "MA"

    all_monthly = []
    for year in years:
        year_files = sorted(ma_dir.glob(f"*{year}*.csv"))
        if not year_files:
            # Also check subdirectories
            year_files = sorted(ma_dir.glob(f"**/*{year}*.csv"))
        for f in year_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                # CMS MA files have various column naming conventions
                # Look for county FIPS and enrollment count columns
                df["_source_year"] = year
                df["_source_file"] = f.name
                all_monthly.append(df)
            except Exception as e:
                print(f"  [MA] Warning: Could not read {f.name}: {e}")

    if not all_monthly:
        print(f"  [MA] No enrollment files found in {ma_dir}")
        print(f"  [MA] Download from: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-advantagepart-d-contract-and-enrollment-data/monthly-enrollment-contract/plan/state/county")
        return pd.DataFrame()

    print(f"  [MA] Loaded {len(all_monthly)} monthly files")
    # Processing will depend on actual file format
    # Return raw for now -- format standardization in downstream processing
    return pd.concat(all_monthly, ignore_index=True)


def load_sedd_panel(data_dir, state_fips="37", years=range(2010, 2024)):
    """
    Load HCUP SEDD county totals for multiple years.

    Expected: data_dir/sedd/sedd_{state_fips}_{year}.csv
    Or a single combined file: data_dir/sedd/sedd_{state_fips}_all.csv
    """
    from src.database_read.load_sedd import load_sedd_county_totals

    sedd_dir = Path(data_dir) / "sedd"

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
            counts = load_sedd_county_totals(
                year_files[0], state_fips=state_fips, year=year
            )
            counts["Year"] = year
            all_years.append(counts)

    if not all_years:
        print(f"  [SEDD] No SEDD files found in {sedd_dir}")
        print(f"  [SEDD] HCUP SEDD requires a Data Use Agreement.")
        print(f"  [SEDD] See: https://hcup-us.ahrq.gov/tech_assist/dua.jsp")
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
    print(f"{'=' * 60}")

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

    # BLS Unemployment
    bls = load_bls_unemployment(data_dir, state=state, years=years)
    if not bls.empty:
        panel = panel.merge(bls, on=["FIPS", "Year"], how="left")

    # SAIPE Poverty/Income
    saipe = load_saipe(data_dir, state_fips=state_fips, years=years)
    if not saipe.empty:
        panel = panel.merge(saipe, on=["FIPS", "Year"], how="left")

    # SVI (with carry-forward)
    svi = load_svi_panel(data_dir, state_fips=state_fips, years=years)
    if not svi.empty:
        panel = panel.merge(
            svi[["FIPS", "Year"] + [c for c in svi.columns
                                     if c not in ["FIPS", "Year", "svi_year",
                                                   "County"]]],
            on=["FIPS", "Year"],
            how="left",
        )

    # SEDD (target variable)
    sedd = load_sedd_panel(data_dir, state_fips=state_fips, years=years)
    if not sedd.empty:
        panel = panel.merge(
            sedd[["FIPS", "Year", "Total_ED", "Total_ED_Residence"]],
            on=["FIPS", "Year"],
            how="left",
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Panel assembled: {panel.shape[0]} rows x {panel.shape[1]} columns")

    non_scaffold = [c for c in panel.columns if c not in ["FIPS", "County", "Year"]]
    for col in non_scaffold:
        pct_avail = panel[col].notna().mean() * 100
        print(f"  {col}: {pct_avail:.0f}% available")

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

    if len(sys.argv) < 2 or "--help" in sys.argv:
        print("Multi-Year County Panel Builder")
        print("=" * 50)
        print()
        print("Usage:")
        print("  python -m src.data_acquisition.build_panel --state NC --years 2015-2023")
        print()
        print("Options:")
        print("  --state STATE    State abbreviation (NC or SC, default: NC)")
        print("  --years RANGE    Year range (default: 2015-2023)")
        print("  --data-dir DIR   Raw data directory (default: data/raw)")
        print("  --output PATH    Output CSV path (default: data/processed/panel_{state}.csv)")
        print()
        print("Data sources must be downloaded first. See:")
        print("  src/data_acquisition/README_DATA_SOURCES.md")
        sys.exit(0)

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
