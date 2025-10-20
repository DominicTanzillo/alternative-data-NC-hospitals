import pdfplumber
import pandas as pd
import re

def extract_sheps_2021_wide_debug(pdf_path):
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        county, hospital = None, None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            # --- Detect county ---
            county_match = re.search(r"([A-Za-z]+)\s+County", text)
            if county_match:
                county = county_match.group(1).strip()

            # --- Detect hospital name ---
            hosp_match = re.search(r"Emergency Department Visits.*?\n([A-Za-z0-9 ,&\-\(\)']+)", text)
            if hosp_match:
                hospital = hosp_match.group(1).strip()
                print(f"[Page {page_num}] Hospital found:", hospital, "| County:", county)

            # --- Split blocks more loosely ---
            blocks = re.split(r"(?:\n|^)([A-Z][A-Za-z /,&]+)\nFiscal Year", text)
            for i in range(1, len(blocks), 2):
                section = blocks[i].strip()
                section_block = blocks[i + 1] if i + 1 < len(blocks) else ""
                lines = [l.strip() for l in section_block.split("\n") if l.strip()]

                # Skip if we don't have expected year
                if not any("2021" in l for l in lines):
                    continue

                for line in lines:
                    # Skip headings
                    if "2021" in line or re.match(r"^[A-Za-z]", line):
                        metric = re.split(r"\s{2,}", line)[0]
                        nums = re.findall(r"[\d,\.]+", line)
                        if len(nums) >= 2:
                            n_2021, pct_2021 = nums[0], nums[1]
                            col = f"{section}_{metric}".replace(" ", "_")
                            records.append({
                                "County": county,
                                "Hospital": hospital,
                                f"{col}_N": n_2021.replace(",", ""),
                                f"{col}_Pct": pct_2021
                            })
                print(f"  └─ Section parsed:", section)

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ No records captured — check regex patterns or PDF text layout.")
        return df

    wide = df.groupby(["County", "Hospital"], as_index=False).agg(lambda x: x.iloc[0])
    wide = wide.loc[:, ~wide.columns.duplicated()]
    return wide

import pdfplumber
import pandas as pd
import re

def extract_sheps_tables(pdf_path):
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        county, hospital = None, None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            county_match = re.search(r"([A-Za-z]+)\s+County", text)
            if county_match:
                county = county_match.group(1).strip()

            hosp_match = re.search(r"Emergency Department Visits.*?\n([A-Za-z0-9 ,&\-\(\)']+)", text)
            if hosp_match:
                hospital = hosp_match.group(1).strip()
                print(f"[Page {page_num}] Hospital found:", hospital, "| County:", county)

            # Extract all detected tables
            tables = page.extract_tables()
            for t in tables:
                df = pd.DataFrame(t)
                if df.shape[1] < 3:
                    continue

                # Standardize headers
                header_idx = df.index[df.iloc[:,0].str.contains("Fiscal Year", na=False)].tolist()
                if not header_idx:
                    continue
                start = header_idx[0] + 1
                df = df.iloc[start:, :3]
                df.columns = ["Metric", "N_2021", "Pct_2021"]

                # Clean numeric columns
                df["N_2021"] = df["N_2021"].str.replace(",", "", regex=False)
                df["Pct_2021"] = df["Pct_2021"].str.replace("%", "", regex=False)

                for _, r in df.iterrows():
                    metric = str(r["Metric"]).strip()
                    n = r["N_2021"]
                    pct = r["Pct_2021"]
                    if not metric or metric.lower().startswith("total"):
                        continue
                    records.append({
                        "County": county,
                        "Hospital": hospital,
                        "Metric": metric,
                        "N_2021": n,
                        "Pct_2021": pct
                    })

    df_long = pd.DataFrame(records)
    df_wide = df_long.pivot_table(index=["County", "Hospital"],
                                  columns="Metric",
                                  values=["N_2021", "Pct_2021"],
                                  aggfunc="first").reset_index()
    df_wide.columns = ["_".join(col).strip("_") for col in df_wide.columns.values]
    return df_wide

import pdfplumber
import pandas as pd
import re

def extract_sheps_2021_structured(pdf_path):
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        county, hospital = None, None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # Detect hospital and county once per page
            for i, line in enumerate(lines[:15]):
                if re.search(r"County", line):
                    parts = line.split()
                    county = parts[0].strip()
                if "Emergency Department Visits" in line:
                    # hospital name usually next line
                    hospital = lines[i + 1].strip() if i + 1 < len(lines) else None

            # join wrapped lines
            combined = []
            buffer = ""
            for line in lines:
                # if line doesn't contain numbers, it's probably a continuation
                if not re.search(r"\d", line):
                    buffer += " " + line
                else:
                    combined.append(buffer.strip() + " " + line)
                    buffer = ""
            if buffer:
                combined.append(buffer.strip())

            # extract metrics and values
            for c in combined:
                # Match pattern: "Metric text ... N % N % N %"
                m = re.match(r"^(.*?)\s+([\d,]+)\s+([\d.]+)", c)
                if not m:
                    continue
                metric = m.group(1).strip()
                n_2021 = m.group(2).replace(",", "")
                pct_2021 = m.group(3)

                # Skip total line
                if metric.lower().startswith("all"):
                    continue

                records.append({
                    "County": county,
                    "Hospital": hospital,
                    "Metric": metric,
                    "N_2021": n_2021,
                    "Pct_2021": pct_2021
                })

    df_long = pd.DataFrame(records)
    df_wide = df_long.pivot_table(
        index=["County", "Hospital"],
        columns="Metric",
        values=["N_2021", "Pct_2021"],
        aggfunc="first"
    ).reset_index()
    df_wide.columns = ["_".join(col).strip("_") for col in df_wide.columns.values]
    return df_wide

import pdfplumber
import pandas as pd
import re

def extract_sheps_clean(pdf_path):
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        county, hospital = None, None

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # detect header
            for i, line in enumerate(lines[:15]):
                if re.search(r"County", line):
                    parts = line.split()
                    county = parts[0].strip()
                if "Emergency Department Visits" in line:
                    hospital = lines[i + 1].strip() if i + 1 < len(lines) else None

            # join wrapped lines
            combined = []
            buffer = ""
            for line in lines:
                if not re.search(r"\d", line):
                    buffer += " " + line
                else:
                    combined.append(buffer.strip() + " " + line)
                    buffer = ""
            if buffer:
                combined.append(buffer.strip())

            # extract metrics and numbers
            for c in combined:
                m = re.match(r"^(.*?)\s+([\d,]+)\s+([\d.]+)", c)
                if not m:
                    continue
                metric = m.group(1).strip()
                n_2021 = m.group(2).replace(",", "")
                pct_2021 = m.group(3)
                if metric.lower().startswith("all"):
                    continue
                records.append({
                    "County": county,
                    "Hospital": hospital,
                    "Metric": metric,
                    "N_2021": n_2021,
                    "Pct_2021": pct_2021
                })

    df_long = pd.DataFrame(records)
    print("✅ Records extracted:", len(df_long))
    print(df_long.head(10))  # check metric names here

    # Only pivot if metric names look valid
    if not df_long.empty and df_long["Metric"].nunique() > 3:
        df_wide = df_long.pivot_table(
            index=["County", "Hospital"],
            columns="Metric",
            values=["N_2021", "Pct_2021"],
            aggfunc="first"
        ).reset_index()
        df_wide.columns = ["_".join(col).strip("_") for col in df_wide.columns.values]
        return df_wide
    else:
        return df_long


import pdfplumber
import pandas as pd
import re

def extract_sheps_ed_2021(pdf_path):
    records = []
    current_section = None
    county, hospital = None, None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # Detect hospital and county
            for i, line in enumerate(lines[:10]):
                if "County" in line:
                    county = line.split("County")[0].strip()
                if "Emergency Department Visits" in line:
                    hospital = lines[i + 1].strip() if i + 1 < len(lines) else None

            # Join wrapped lines until digits appear
            joined = []
            buffer = ""
            for line in lines:
                if not re.search(r"\d", line):
                    buffer += " " + line
                else:
                    joined.append(buffer.strip() + " " + line)
                    buffer = ""
            if buffer:
                joined.append(buffer.strip())

            for line in joined:
                # Detect section headers
                if any(kw in line for kw in [
                    "Patient Residence", "Age Group", "Race",
                    "Ethnicity", "Point of Origin", "Payer", "Patient Disposition"
                ]):
                    current_section = re.sub(r"\s+\d.*", "", line).strip()
                    continue

                # Match metric + numeric columns
                m = re.match(r"^(.*?)\s+([\d,]+)\s+([\d.]+)\s+([\d,\.]*)\s*([\d\.]*)?$", line)
                if not m:
                    continue

                metric = m.group(1).strip()
                n_2021 = m.group(2).replace(",", "")
                pct_2021 = m.group(3)

                # Skip totals
                if metric.lower().startswith("all"):
                    continue

                records.append({
                    "County": county,
                    "Hospital": hospital,
                    "Section": current_section,
                    "Metric": metric,
                    "N_2021": n_2021,
                    "Pct_2021": pct_2021
                })

    df = pd.DataFrame(records)
    return df

import pdfplumber
import pandas as pd
import re

def extract_sheps_ed_2021_clean(pdf_path):
    records = []
    county, hospital, section = None, None, None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # detect hospital & county
            for i, line in enumerate(lines[:10]):
                if "County" in line:
                    county = line.split("County")[0].strip()
                if "Emergency Department Visits" in line:
                    hospital = lines[i + 1].strip() if i + 1 < len(lines) else None

            # join wrapped lines
            joined = []
            buffer = ""
            for line in lines:
                if not re.search(r"\d", line):
                    buffer += " " + line
                else:
                    joined.append(buffer.strip() + " " + line)
                    buffer = ""
            if buffer:
                joined.append(buffer.strip())

            for line in joined:
                # detect and update section when header appears by itself
                if any(kw in line for kw in [
                    "Patient Residence", "Age Group", "Race",
                    "Ethnicity", "Point of Origin", "Payer", "Patient Disposition"
                ]) and not re.search(r"\d", line):
                    section = re.sub(r"\s+\d.*", "", line).strip()
                    continue

                # detect lines where section + metric appear together (inline)
                inline_section_match = re.search(
                    r"(Patient Residence|Age Group|Race|Ethnicity|Point of Origin|Payer|Patient Disposition)[^0-9]+",
                    line
                )
                if inline_section_match:
                    section = inline_section_match.group(0).strip()
                    # remove section text from the line so metric regex can isolate properly
                    line = line.replace(section, "").strip()

                # match metrics with 2021 data (first two numbers)
                match = re.search(
                    r"^(?:N\s*%.*?)*([A-Za-z0-9, \-/()]+?)\s+([\d,]+)\s+([\d.]+)\s+[\d,]+\s+[\d.]+",
                    line
                )
                if not match:
                    continue

                metric = match.group(1).strip()
                n_2021 = match.group(2).replace(",", "")
                pct_2021 = match.group(3)

                if metric.lower().startswith("all") or "Fiscal Year" in metric:
                    continue

                records.append({
                    "County": county,
                    "Hospital": hospital,
                    "Section": section,
                    "Metric": metric,
                    "N_2021": n_2021,
                    "Pct_2021": pct_2021
                })

    df = pd.DataFrame(records)
    return df


# Example run
pdf_path = "data/ED_visits/ed_ptchar_all_and_by_hosp_2021_and.pdf"
df = extract_sheps_ed_2021_clean(pdf_path)
print(df.shape)
print(df.head(15))

import pdfplumber
import pandas as pd
import re

def extract_sheps_ed_2021_final(pdf_path):
    records = []
    county, hospital, section = None, None, None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # detect hospital & county
            for i, line in enumerate(lines[:10]):
                if "County" in line:
                    county = line.split("County")[0].strip()
                if "Emergency Department Visits" in line:
                    hospital = lines[i + 1].strip() if i + 1 < len(lines) else None

            # join wrapped lines
            joined, buffer = [], ""
            for line in lines:
                if not re.search(r"\d", line):
                    buffer += " " + line
                else:
                    joined.append(buffer.strip() + " " + line)
                    buffer = ""
            if buffer:
                joined.append(buffer.strip())

            for line in joined:
                # Clean hospital name bleeding into section
                line = re.sub(r"^Health\s+", "", line)

                # Section detection (header alone)
                if any(kw in line for kw in [
                    "Patient Residence", "Age Group", "Race",
                    "Ethnicity", "Point of Origin", "Payer", "Patient Disposition"
                ]) and not re.search(r"\d", line):
                    section = re.sub(r"\s+\d.*", "", line).strip()
                    continue

                # Inline section + metric
                inline_section_match = re.search(
                    r"(Patient Residence|Age Group|Race|Ethnicity|Point of Origin|Payer|Patient Disposition)[^0-9]+",
                    line
                )
                if inline_section_match:
                    section = inline_section_match.group(0).strip()
                    line = line.replace(section, "").strip()

                # Match 2021 N/% values
                match = re.search(
                    r"^(?:N\s*%.*?)*([A-Za-z0-9, \-/()]+?)?\s+([\d,]+)\s+([\d.]+)\s+[\d,\.]+\s+[\d\.]+",
                    line
                )
                if not match:
                    continue

                metric = (match.group(1) or "").strip()
                n_2021 = match.group(2).replace(",", "")
                pct_2021 = match.group(3)

                if metric.lower().startswith("all") or "Fiscal Year" in metric:
                    continue

                # ---- Fix specific cases ----
                if section and section.startswith("Patient Residence") and metric == "":
                    metric = "NC"
                if section and section.startswith("Patient Disposition") and metric == "":
                    metric = "Home, self, or outpatient care"
                if section and section.startswith("Ethnicity") and metric == "":
                    metric = "Hispanic"
                if section and section.startswith("Race Caucasian"):
                    section = "Race"
                    if metric == "":
                        metric = "Caucasian"
                if section and section.startswith("Health Patient Residence"):
                    section = "Patient Residence State"
                    if metric == "":
                        metric = "NC"

                records.append({
                    "County": county,
                    "Hospital": hospital,
                    "Section": section,
                    "Metric": metric,
                    "N_2021": n_2021,
                    "Pct_2021": pct_2021
                })

    df = pd.DataFrame(records)
    return df


import pdfplumber
import pandas as pd
import re

def extract_sheps_ed_2021_final1(pdf_path):
    records = []
    county, hospital, section = None, None, None

    # Regex for detecting sections and multi-metric lines
    SECTION_PREFIXES = [
        r"Patient Residence(?: State)?",
        r"Age Group",
        r"Race",
        r"Ethnicity",
        r"Point of Origin(?: Non-health care facility)?",
        r"Payer(?: .+)?",
        r"Patient Disposition(?: Home, self, or outpatient care)?",
    ]
    SECTION_RE = re.compile(r"^(" + "|".join(SECTION_PREFIXES) + r")\b", flags=re.I)
    MULTIMETRIC_RE = re.compile(
        r"([A-Za-z][A-Za-z0-9/\-,() ]*?)\s+([\d,]+)\s+([\d.]+)\s+([\d,]+)\s+([\d.]+)"
    )

    def _clean_section_and_content(line):
        line = re.sub(r"^Health\s+", "", line).strip()
        m = SECTION_RE.search(line)
        if not m:
            return None, line
        section = m.group(1).strip()
        content = line[m.end():].strip()
        # normalize section variants
        if section.lower().startswith("health patient residence"):
            section = "Patient Residence State"
        elif section.lower().startswith("patient residence") and "state" not in section.lower():
            section = "Patient Residence State"
        elif section.lower().startswith("patient disposition"):
            section = "Patient Disposition"
        elif section.lower().startswith("point of origin"):
            section = "Point of Origin"
        elif section.lower().startswith("payer"):
            section = "Payer"
        elif section.lower().startswith("race"):
            section = "Race"
        elif section.lower().startswith("ethnicity"):
            section = "Ethnicity"
        return section, content

    def _default_metric_for_section(section):
        s = (section or "").lower()
        if s.startswith("patient residence"):
            return "NC"
        if s.startswith("patient disposition"):
            return "Home, self, or outpatient care"
        if s.startswith("ethnicity"):
            return "Hispanic"
        return None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # detect hospital & county
            for i, line in enumerate(lines[:10]):
                if "County" in line:
                    county = line.split("County")[0].strip()
                if "Emergency Department Visits" in line:
                    hospital = lines[i + 1].strip() if i + 1 < len(lines) else hospital

            # join wrapped lines
            joined, buffer = [], ""
            for line in lines:
                if not re.search(r"\d", line):
                    buffer += " " + line
                else:
                    joined.append((buffer + " " + line).strip())
                    buffer = ""
            if buffer:
                joined.append(buffer.strip())

            for line in joined:
                sec, content = _clean_section_and_content(line)
                if sec:
                    section = sec

                found_any = False
                for m in MULTIMETRIC_RE.finditer(content):
                    metric = m.group(1).strip()
                    n_2021 = m.group(2).replace(",", "")
                    pct_2021 = m.group(3)
                    if metric.lower().startswith("all") or "fiscal year" in metric.lower():
                        continue
                    if not metric and section == "Race":
                        metric = "Caucasian"
                    records.append({
                        "County": county,
                        "Hospital": hospital,
                        "Section": section,
                        "Metric": metric,
                        "N_2021": n_2021,
                        "Pct_2021": pct_2021
                    })
                    found_any = True

                if found_any:
                    continue

                single = re.search(r"^\s*([\d,]+)\s+([\d.]+)\s+[\d,\.]+\s+[\d\.]+", content)
                if single and section:
                    metric = _default_metric_for_section(section) or ""
                    n_2021 = single.group(1).replace(",", "")
                    pct_2021 = single.group(2)
                    if metric:
                        records.append({
                            "County": county,
                            "Hospital": hospital,
                            "Section": section,
                            "Metric": metric,
                            "N_2021": n_2021,
                            "Pct_2021": pct_2021
                        })
                    continue

                m1 = re.search(
                    r"([A-Za-z][A-Za-z0-9/\-,() ]*?)\s+([\d,]+)\s+([\d.]+)\s+[\d,\.]+\s+[\d\.]+",
                    content
                )
                if m1 and section:
                    metric = (m1.group(1) or "").strip()
                    if metric.lower().startswith("all") or "fiscal year" in metric.lower():
                        continue
                    if not metric:
                        metric = _default_metric_for_section(section) or metric
                    n_2021 = m1.group(2).replace(",", "")
                    pct_2021 = m1.group(3)
                    records.append({
                        "County": county,
                        "Hospital": hospital,
                        "Section": section,
                        "Metric": metric,
                        "N_2021": n_2021,
                        "Pct_2021": pct_2021
                    })

    df = pd.DataFrame(records)
    return df
