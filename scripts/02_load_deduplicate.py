"""
Script 02: Load FAERS CSVs into DuckDB and deduplicate.

Loads all parsed CSV files from script 01 into a single DuckDB database.
Deduplicates by caseid, keeping only the most recent caseversion per case.
Builds indexes for efficient downstream queries.

Deduplication strategy:
    FAERS contains multiple versions of the same case (same caseid,
    different caseversion or primaryid). We keep only the row with the
    highest caseversion (most recent update). All child tables (DRUG,
    REAC, etc.) are filtered to retain only primaryids from the
    deduplicated DEMO table.

Usage:
    python scripts/02_load_deduplicate.py

Output: data/processed/faers.duckdb
"""

import duckdb
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "raw" / "parsed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROCESSED_DIR / "faers.duckdb"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Table definitions: table_name -> (file_prefix, column_types)
# Column types help DuckDB handle messy FAERS data correctly
TABLE_DEFS = {
    "demo_raw": {
        "prefix": "demo_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "caseversion": "VARCHAR",
            "i_f_code": "VARCHAR",
            "event_dt": "VARCHAR",
            "mfr_dt": "VARCHAR",
            "init_fda_dt": "VARCHAR",
            "fda_dt": "VARCHAR",
            "rept_cod": "VARCHAR",
            "mfr_num": "VARCHAR",
            "mfr_sndr": "VARCHAR",
            "age": "VARCHAR",
            "age_cod": "VARCHAR",
            "age_grp": "VARCHAR",
            "sex": "VARCHAR",
            "wt": "VARCHAR",
            "wt_cod": "VARCHAR",
            "rept_dt": "VARCHAR",
            "occp_cod": "VARCHAR",
            "reporter_country": "VARCHAR",
            "occr_country": "VARCHAR",
        },
    },
    "drug_raw": {
        "prefix": "drug_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "drug_seq": "VARCHAR",
            "role_cod": "VARCHAR",
            "drugname": "VARCHAR",
            "prod_ai": "VARCHAR",
            "val_vbm": "VARCHAR",
            "route": "VARCHAR",
            "dose_vbm": "VARCHAR",
            "cum_dose_chr": "VARCHAR",
            "cum_dose_unit": "VARCHAR",
            "dechal": "VARCHAR",
            "rechal": "VARCHAR",
            "lot_num": "VARCHAR",
            "nda_num": "VARCHAR",
            "dose_amt": "VARCHAR",
            "dose_unit": "VARCHAR",
            "dose_form": "VARCHAR",
            "dose_freq": "VARCHAR",
        },
    },
    "reac_raw": {
        "prefix": "reac_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "pt": "VARCHAR",
            "drug_rec_act": "VARCHAR",
        },
    },
    "outc_raw": {
        "prefix": "outc_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "outc_cod": "VARCHAR",
        },
    },
    "indi_raw": {
        "prefix": "indi_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "indi_drug_seq": "VARCHAR",
            "indi_pt": "VARCHAR",
        },
    },
    "ther_raw": {
        "prefix": "ther_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "dsg_drug_seq": "VARCHAR",
            "start_dt": "VARCHAR",
            "end_dt": "VARCHAR",
            "dur": "VARCHAR",
            "dur_cod": "VARCHAR",
        },
    },
    "rpsr_raw": {
        "prefix": "rpsr_",
        "types": {
            "primaryid": "VARCHAR",
            "caseid": "VARCHAR",
            "rpsr_cod": "VARCHAR",
        },
    },
}


def load_table(con: duckdb.DuckDBPyConnection, table_name: str, config: dict) -> int:
    """Load all quarterly CSVs for a table into DuckDB."""
    prefix = config["prefix"]
    csv_files = sorted(PARSED_DIR.glob(f"{prefix}*.csv"))

    if not csv_files:
        print(f"  WARNING: No files found for {table_name} (pattern: {prefix}*.csv)")
        return 0

    # Drop existing table
    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Build column type string for read_csv
    type_map = config["types"]

    total_rows = 0
    for i, csv_file in enumerate(csv_files):
        try:
            if i == 0:
                # Create table from first file
                con.execute(f"""
                    CREATE TABLE {table_name} AS
                    SELECT * FROM read_csv('{csv_file}',
                        header=true,
                        all_varchar=true,
                        ignore_errors=true,
                        null_padding=true)
                """)
            else:
                # Append subsequent files
                con.execute(f"""
                    INSERT INTO {table_name}
                    SELECT * FROM read_csv('{csv_file}',
                        header=true,
                        all_varchar=true,
                        ignore_errors=true,
                        null_padding=true)
                """)

            count = con.execute(f"SELECT count(*) FROM read_csv('{csv_file}', header=true, all_varchar=true, ignore_errors=true, null_padding=true)").fetchone()[0]
            total_rows += count
            print(f"    {csv_file.name}: {count:>10,} rows")
        except Exception as e:
            print(f"    ERROR loading {csv_file.name}: {e}")

    actual = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    print(f"    Total in {table_name}: {actual:>10,} rows")
    return actual


def deduplicate(con: duckdb.DuckDBPyConnection):
    """
    Deduplicate FAERS by caseid, keeping the most recent caseversion.

    Creates clean tables (demo, drug, reac, etc.) containing only
    primaryids from the deduplicated demo table.
    """
    print("\n  Deduplicating by caseid (keeping latest caseversion)...")

    # Step 1: Identify the latest primaryid per caseid
    con.execute("""
        CREATE OR REPLACE TABLE demo AS
        WITH ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY caseid
                    ORDER BY TRY_CAST(caseversion AS INTEGER) DESC NULLS LAST,
                             primaryid DESC
                ) AS rn
            FROM demo_raw
            WHERE caseid IS NOT NULL
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
    """)

    raw_count = con.execute("SELECT count(*) FROM demo_raw").fetchone()[0]
    dedup_count = con.execute("SELECT count(*) FROM demo").fetchone()[0]
    removed = raw_count - dedup_count
    print(f"    DEMO: {raw_count:,} raw → {dedup_count:,} deduplicated ({removed:,} duplicates removed)")

    # Step 2: Get set of valid primaryids
    con.execute("""
        CREATE OR REPLACE TEMP TABLE valid_pids AS
        SELECT DISTINCT primaryid FROM demo
    """)

    # Step 3: Filter all child tables to valid primaryids
    child_tables = {
        "drug": "drug_raw",
        "reac": "reac_raw",
        "outc": "outc_raw",
        "indi": "indi_raw",
        "ther": "ther_raw",
        "rpsr": "rpsr_raw",
    }

    for clean_name, raw_name in child_tables.items():
        try:
            con.execute(f"""
                CREATE OR REPLACE TABLE {clean_name} AS
                SELECT r.*
                FROM {raw_name} r
                INNER JOIN valid_pids v ON r.primaryid = v.primaryid
            """)
            raw = con.execute(f"SELECT count(*) FROM {raw_name}").fetchone()[0]
            clean = con.execute(f"SELECT count(*) FROM {clean_name}").fetchone()[0]
            print(f"    {clean_name.upper():6s}: {raw:>10,} raw → {clean:>10,} deduplicated")
        except Exception as e:
            print(f"    WARNING: Could not filter {clean_name}: {e}")

    # Drop raw tables to save space
    for raw_name in ["demo_raw"] + list(child_tables.values()):
        con.execute(f"DROP TABLE IF EXISTS {raw_name}")

    # Drop temp table
    con.execute("DROP TABLE IF EXISTS valid_pids")


def build_indexes(con: duckdb.DuckDBPyConnection):
    """Create indexes for efficient querying."""
    print("\n  Building indexes...")
    indexes = [
        ("idx_demo_primaryid", "demo", "primaryid"),
        ("idx_demo_caseid", "demo", "caseid"),
        ("idx_drug_primaryid", "drug", "primaryid"),
        ("idx_drug_drugname", "drug", "drugname"),
        ("idx_drug_role_cod", "drug", "role_cod"),
        ("idx_reac_primaryid", "reac", "primaryid"),
        ("idx_reac_pt", "reac", "pt"),
        ("idx_ther_primaryid", "ther", "primaryid"),
    ]
    for idx_name, table, col in indexes:
        try:
            con.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({col})")
            print(f"    {idx_name}")
        except Exception:
            pass  # DuckDB may not support all index types


def print_summary(con: duckdb.DuckDBPyConnection):
    """Print database summary statistics."""
    print(f"\n{'=' * 70}")
    print(f"  DATABASE SUMMARY")
    print(f"{'=' * 70}")

    tables = ["demo", "drug", "reac", "outc", "indi", "ther", "rpsr"]
    for table in tables:
        try:
            count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            cols = con.execute(f"SELECT * FROM {table} LIMIT 0").description
            n_cols = len(cols) if cols else 0
            print(f"  {table:6s}: {count:>12,} rows × {n_cols} cols")
        except Exception:
            print(f"  {table:6s}: NOT LOADED")

    # Quick data quality checks
    print(f"\n  DATA QUALITY CHECKS")
    print(f"  {'─' * 50}")

    # Unique cases
    n_cases = con.execute("SELECT count(DISTINCT caseid) FROM demo").fetchone()[0]
    print(f"  Unique caseids:          {n_cases:>10,}")

    # Date range
    try:
        dates = con.execute("""
            SELECT MIN(fda_dt), MAX(fda_dt)
            FROM demo
            WHERE fda_dt IS NOT NULL AND LENGTH(fda_dt) >= 8
        """).fetchone()
        print(f"  FDA date range:          {dates[0]} to {dates[1]}")
    except Exception:
        pass

    # Reporter countries
    try:
        countries = con.execute("""
            SELECT reporter_country, count(*) as n
            FROM demo
            WHERE reporter_country IS NOT NULL
            GROUP BY reporter_country
            ORDER BY n DESC
            LIMIT 5
        """).fetchall()
        print(f"  Top reporter countries:  {', '.join(f'{c[0]}({c[1]:,})' for c in countries)}")
    except Exception:
        pass

    # Drug name preview (check for Cobenfy)
    try:
        cobenfy_check = con.execute("""
            SELECT drugname, count(*) as n
            FROM drug
            WHERE UPPER(drugname) LIKE '%XANOMELINE%'
               OR UPPER(drugname) LIKE '%COBENFY%'
               OR UPPER(drugname) LIKE '%KARXT%'
               OR UPPER(drugname) LIKE '%TROSPIUM%'
            GROUP BY drugname
            ORDER BY n DESC
            LIMIT 10
        """).fetchall()
        if cobenfy_check:
            print(f"\n  COBENFY-RELATED DRUG NAMES FOUND:")
            for name, n in cobenfy_check:
                print(f"    {name:50s} {n:>6,} records")
        else:
            print(f"\n  WARNING: No Cobenfy-related drug names found in database!")
    except Exception:
        pass


def main():
    t0 = time.time()

    print("=" * 70)
    print("  FAERS DuckDB Loader and Deduplicator")
    print("  Project 48: Cobenfy Pharmacovigilance")
    print("=" * 70)

    # Check for parsed CSVs
    csv_files = list(PARSED_DIR.glob("*.csv"))
    if not csv_files:
        print(f"\n  ERROR: No CSV files found in {PARSED_DIR}")
        print(f"  Run script 01 first: python scripts/01_download_faers.py")
        sys.exit(1)

    print(f"\n  Found {len(csv_files)} CSV files in {PARSED_DIR}")

    # Remove existing database for clean build
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  Removed existing database: {DB_PATH}")

    # Connect to DuckDB
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET memory_limit='8GB'")

    # Load each table
    print(f"\n{'─' * 70}")
    print("  LOADING TABLES")
    print(f"{'─' * 70}")

    for table_name, config in TABLE_DEFS.items():
        print(f"\n  Loading {table_name}...")
        load_table(con, table_name, config)

    # Deduplicate
    print(f"\n{'─' * 70}")
    print("  DEDUPLICATION")
    print(f"{'─' * 70}")
    deduplicate(con)

    # Build indexes
    build_indexes(con)

    # Summary
    print_summary(con)

    # Database file size
    con.close()
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  LOAD COMPLETE ({elapsed:.0f}s)")
    print(f"  Database: {DB_PATH} ({db_size_mb:.1f} MB)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/03_drug_standardisation.py")


if __name__ == "__main__":
    main()
