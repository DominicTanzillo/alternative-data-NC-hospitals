"""Download and aggregate NY SPARCS inpatient discharge data (2015-2023).

Uses SoQL to aggregate ED-related inpatient admissions by hospital county
directly on the Socrata API, avoiding downloading millions of individual records.
Also downloads PPV (Potentially Preventable Visit) rates by patient county.
"""
import urllib.request
import json
import ssl
import csv
import time
from pathlib import Path

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

OUT = Path("data/new_york")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    2015: "82xm-y6g8",
    2016: "gnzp-ekau",
    2017: "22g3-z7e7",
    2018: "yjgt-tq93",
    2019: "4ny4-j5zv",
    2020: "nxi5-zj9x",
    2021: "tg3i-cinn",
    2022: "5dtw-tffi",
    2023: "46xm-urtu",
}


def socrata_query(dataset_id, select, group_by=None, where=None, limit=5000):
    """Run SoQL query against Socrata API."""
    from urllib.parse import quote
    base = "https://health.data.ny.gov/resource/{}.json".format(dataset_id)
    params = ["$limit={}".format(limit), "$select={}".format(quote(select))]
    if group_by:
        params.append("$group={}".format(quote(group_by)))
    if where:
        params.append("$where={}".format(quote(where)))
    url = base + "?" + "&".join(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, context=ctx, timeout=120)
    return json.loads(resp.read())


# =========================================================================
# 1. Aggregate inpatient discharges by hospital county + ED indicator
# =========================================================================
print("=" * 65)
print("DOWNLOADING NY SPARCS INPATIENT AGGREGATES (2015-2023)")
print("=" * 65)

all_rows = []
for year, did in sorted(DATASETS.items()):
    print("  {} ({}): ".format(year, did), end="", flush=True)
    try:
        data = socrata_query(
            did,
            select="hospital_county,emergency_department_indicator,count(*) as n",
            group_by="hospital_county,emergency_department_indicator",
        )
        for row in data:
            row["year"] = year
        all_rows.extend(data)

        # Quick summary
        ed_total = sum(int(r.get("n", 0)) for r in data
                       if r.get("emergency_department_indicator") == "Y")
        non_ed = sum(int(r.get("n", 0)) for r in data
                     if r.get("emergency_department_indicator") != "Y")
        counties = len(set(r.get("hospital_county", "") for r in data))
        print("{} counties, {:,} ED admissions, {:,} non-ED".format(
            counties, ed_total, non_ed))
    except Exception as e:
        print("FAILED: {}".format(e))

    time.sleep(0.5)  # Be polite to the API

# Save
outpath = OUT / "ny_sparcs_inpatient_by_county.csv"
with open(outpath, "w", newline="") as f:
    writer = csv.DictWriter(f,
                            fieldnames=["year", "hospital_county",
                                        "emergency_department_indicator", "n"])
    writer.writeheader()
    for row in all_rows:
        writer.writerow({
            "year": row["year"],
            "hospital_county": row.get("hospital_county", ""),
            "emergency_department_indicator": row.get("emergency_department_indicator", ""),
            "n": row.get("n", 0),
        })
print()
print("Saved -> {}".format(outpath))

# =========================================================================
# 2. Aggregate by type of admission (Emergency vs Elective etc.)
# =========================================================================
print()
print("=" * 65)
print("DOWNLOADING ADMISSION TYPE AGGREGATES (2015-2023)")
print("=" * 65)

type_rows = []
for year, did in sorted(DATASETS.items()):
    print("  {} ({}): ".format(year, did), end="", flush=True)
    try:
        data = socrata_query(
            did,
            select="hospital_county,type_of_admission,count(*) as n",
            group_by="hospital_county,type_of_admission",
        )
        for row in data:
            row["year"] = year
        type_rows.extend(data)

        emerg = sum(int(r["n"]) for r in data
                    if r.get("type_of_admission") == "Emergency")
        print("{:,} emergency admissions".format(emerg))
    except Exception as e:
        print("FAILED: {}".format(e))
    time.sleep(0.5)

outpath2 = OUT / "ny_sparcs_admission_type_by_county.csv"
with open(outpath2, "w", newline="") as f:
    writer = csv.DictWriter(f,
                            fieldnames=["year", "hospital_county",
                                        "type_of_admission", "n"])
    writer.writeheader()
    for row in type_rows:
        writer.writerow({
            "year": row["year"],
            "hospital_county": row.get("hospital_county", ""),
            "type_of_admission": row.get("type_of_admission", ""),
            "n": row.get("n", 0),
        })
print()
print("Saved -> {}".format(outpath2))

# =========================================================================
# 3. Payment type breakdown (Medicaid/Medicare/uninsured = need signal)
# =========================================================================
print()
print("=" * 65)
print("DOWNLOADING PAYER MIX BY COUNTY (2015-2023)")
print("=" * 65)

payer_rows = []
for year, did in sorted(DATASETS.items()):
    print("  {} ({}): ".format(year, did), end="", flush=True)
    try:
        data = socrata_query(
            did,
            select="hospital_county,payment_typology_1,count(*) as n",
            group_by="hospital_county,payment_typology_1",
        )
        for row in data:
            row["year"] = year
        payer_rows.extend(data)

        medicaid = sum(int(r["n"]) for r in data
                       if "medicaid" in r.get("payment_typology_1", "").lower())
        print("{:,} Medicaid discharges".format(medicaid))
    except Exception as e:
        print("FAILED: {}".format(e))
    time.sleep(0.5)

outpath3 = OUT / "ny_sparcs_payer_by_county.csv"
with open(outpath3, "w", newline="") as f:
    writer = csv.DictWriter(f,
                            fieldnames=["year", "hospital_county",
                                        "payment_typology_1", "n"])
    writer.writeheader()
    for row in payer_rows:
        writer.writerow({
            "year": row["year"],
            "hospital_county": row.get("hospital_county", ""),
            "payment_typology_1": row.get("payment_typology_1", ""),
            "n": row.get("n", 0),
        })
print()
print("Saved -> {}".format(outpath3))

print()
print("=" * 65)
print("NY SPARCS DOWNLOAD COMPLETE")
print("=" * 65)
print("Files in {}:".format(OUT))
for f in sorted(OUT.iterdir()):
    print("  {} ({:,.0f} KB)".format(f.name, f.stat().st_size / 1024))
