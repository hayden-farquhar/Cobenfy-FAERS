"""
Script 01: Download and extract FAERS quarterly ASCII files.

Downloads FDA FAERS quarterly ASCII zip files from Q4 2024 to Q1 2026
(covering ~18 months post-approval of Cobenfy, September 2024).
Extracts all 7 data tables: DEMO, DRUG, REAC, OUTC, INDI, THER, RPSR.

Adapted from Project 41 (07_faers_download.py) but expanded to extract
all tables (not just DRUG/REAC) and targeted to the post-approval window.

Usage:
    cd "48 Cobenfy FAERS"
    python scripts/01_download_faers.py

Downloads to: data/raw/     (zip files)
Extracts to: data/raw/parsed/  (CSV per quarter per table)
"""

import csv
import sys
import urllib.request
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARSED_DIR = RAW_DIR / "parsed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Quarters to download (post-approval window) ────────────────────────────
# Cobenfy FDA approval: September 26, 2024
# First reports expected in Q4 2024 data file
QUARTERS = [
    (2024, 4),
    (2025, 1),
    (2025, 2),
    (2025, 3),
    (2025, 4),
    (2026, 1),
]

# ── Tables to extract ──────────────────────────────────────────────────────
# Each entry: (keyword in filename, output prefix, columns to extract)
TABLES = {
    "DEMO": {
        "columns": [
            "primaryid", "caseid", "caseversion", "i_f_code", "event_dt",
            "mfr_dt", "init_fda_dt", "fda_dt", "rept_cod", "mfr_num",
            "mfr_sndr", "age", "age_cod", "age_grp", "sex", "wt", "wt_cod",
            "rept_dt", "occp_cod", "reporter_country", "occr_country",
        ],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
            "sex": ["gndr_cod"],
        },
    },
    "DRUG": {
        "columns": [
            "primaryid", "caseid", "drug_seq", "role_cod", "drugname",
            "prod_ai", "val_vbm", "route", "dose_vbm", "cum_dose_chr",
            "cum_dose_unit", "dechal", "rechal", "lot_num", "nda_num",
            "dose_amt", "dose_unit", "dose_form", "dose_freq",
        ],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
            "drugname": ["drug_name"],
        },
    },
    "REAC": {
        "columns": ["primaryid", "caseid", "pt", "drug_rec_act"],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
            "pt": ["preferred_term"],
        },
    },
    "OUTC": {
        "columns": ["primaryid", "caseid", "outc_cod"],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
        },
    },
    "INDI": {
        "columns": ["primaryid", "caseid", "indi_drug_seq", "indi_pt"],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
            "indi_drug_seq": ["drug_seq"],
        },
    },
    "THER": {
        "columns": ["primaryid", "caseid", "dsg_drug_seq", "start_dt", "end_dt", "dur", "dur_cod"],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
            "dsg_drug_seq": ["drug_seq"],
        },
    },
    "RPSR": {
        "columns": ["primaryid", "caseid", "rpsr_cod"],
        "fallbacks": {
            "primaryid": ["isr"],
            "caseid": ["case"],
        },
    },
}


def get_download_url(year: int, quarter: int) -> str:
    """Get the FAERS download URL for a given year and quarter."""
    return f"https://fis.fda.gov/content/Exports/faers_ascii_{year}Q{quarter}.zip"


def download_quarter(year: int, quarter: int) -> Path | None:
    """Download a single quarter's zip file. Returns path or None if failed."""
    url = get_download_url(year, quarter)
    filename = f"faers_{year}Q{quarter}.zip"
    filepath = RAW_DIR / filename

    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  {year}Q{quarter}: already downloaded ({size_mb:.1f} MB)")
        return filepath

    print(f"  Downloading {year}Q{quarter}...", end=" ", flush=True)

    # Try primary URL, then alternates
    urls = [
        url,
        f"https://fis.fda.gov/content/Exports/faers_ascii_{year}q{quarter}.zip",
    ]
    for attempt_url in urls:
        try:
            urllib.request.urlretrieve(attempt_url, filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"{size_mb:.1f} MB")
            return filepath
        except Exception:
            continue

    print("FAILED (all URL patterns)")
    if filepath.exists():
        filepath.unlink()
    return None


def _find_col(header: list[str], target: str, fallbacks: list[str] | None = None) -> int | None:
    """
    Find column index by name with fallback support.

    Tries exact match on target first, then fallbacks, then partial matches.
    """
    candidates = [target] + (fallbacks or [])
    # Exact match
    for c in candidates:
        if c in header:
            return header.index(c)
    # Partial match
    for i, h in enumerate(header):
        for c in candidates:
            if c in h:
                return i
    return None


def extract_table(
    zf: zipfile.ZipFile,
    table_name: str,
    table_config: dict,
    year: int,
    quarter: int,
) -> bool:
    """Extract a single table from the FAERS zip file."""
    out_path = PARSED_DIR / f"{table_name.lower()}_{year}Q{quarter}.csv"
    if out_path.exists():
        return True

    # Find the file in the zip
    target_file = None
    for name in zf.namelist():
        if table_name in name.upper() and name.upper().endswith(".TXT"):
            target_file = name
            break

    if not target_file:
        print(f"    WARNING: {table_name} not found in zip")
        return False

    try:
        with zf.open(target_file) as f:
            content = f.read().decode("latin-1", errors="replace")

        lines = content.strip().split("\n")
        if not lines:
            return False

        # Parse header
        header = [h.strip().strip('"').lower() for h in lines[0].split("$")]
        fallbacks = table_config.get("fallbacks", {})

        # Map desired columns to their indices
        col_map = {}
        for col in table_config["columns"]:
            idx = _find_col(header, col, fallbacks.get(col))
            if idx is not None:
                col_map[col] = idx

        if "primaryid" not in col_map and "caseid" not in col_map:
            print(f"    WARNING: No primary key column found for {table_name}")
            return False

        # Write CSV
        with open(out_path, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(list(col_map.keys()))

            for line in lines[1:]:
                if not line.strip():
                    continue
                fields = [f.strip().strip('"') for f in line.split("$")]
                row = []
                for col, idx in col_map.items():
                    if idx < len(fields):
                        row.append(fields[idx])
                    else:
                        row.append("")
                writer.writerow(row)

        return True

    except Exception as e:
        print(f"    ERROR extracting {table_name}: {e}")
        if out_path.exists():
            out_path.unlink()
        return False


def extract_all_tables(zip_path: Path, year: int, quarter: int) -> dict[str, bool]:
    """Extract all 7 FAERS tables from a quarterly zip file."""
    results = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for table_name, config in TABLES.items():
                ok = extract_table(zf, table_name, config, year, quarter)
                results[table_name] = ok
    except Exception as e:
        print(f"  ERROR opening {zip_path.name}: {e}")
        return {t: False for t in TABLES}
    return results


def main():
    print("=" * 70)
    print("  FAERS Quarterly Data Download & Extraction")
    print("  Project 48: Cobenfy Pharmacovigilance")
    print("  Quarters: Q4 2024 – Q1 2026")
    print("=" * 70)

    success_count = 0

    for year, quarter in QUARTERS:
        print(f"\n{'─' * 50}")
        print(f"  {year} Q{quarter}")
        print(f"{'─' * 50}")

        zip_path = download_quarter(year, quarter)
        if not zip_path:
            print(f"  SKIPPED (download failed)")
            continue

        results = extract_all_tables(zip_path, year, quarter)
        extracted = sum(results.values())
        total = len(results)
        print(f"  Extracted: {extracted}/{total} tables")

        for table, ok in results.items():
            status = "OK" if ok else "MISSING"
            print(f"    {table:6s}: {status}")

        if extracted >= 3:  # At minimum need DEMO, DRUG, REAC
            success_count += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Quarters processed: {success_count}/{len(QUARTERS)}")
    print(f"  Raw zips:   {RAW_DIR}")
    print(f"  Parsed CSVs: {PARSED_DIR}")
    print(f"{'=' * 70}")

    # List parsed files
    parsed_files = sorted(PARSED_DIR.glob("*.csv"))
    if parsed_files:
        print(f"\n  Parsed files ({len(parsed_files)}):")
        for f in parsed_files:
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:35s} {size_kb:>8.0f} KB")

    print(f"\n  Next step: python scripts/02_load_deduplicate.py")


if __name__ == "__main__":
    main()
