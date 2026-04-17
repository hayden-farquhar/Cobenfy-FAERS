"""
Script 04: Case identification and demographics.

Extracts Cobenfy and comparator cases from the standardised FAERS
database. Builds the case/non-case structure needed for
disproportionality analysis.

Case definitions:
    - Cobenfy cases: reports where xanomeline-trospium is PS or SS
    - Comparator cases: reports where comparator drug is PS or SS
    - Reference set: all reports in the database (for standard disproportionality)

Outputs:
    - outputs/tables/demographics.csv: Table 1 demographics
    - outputs/tables/cobenfy_cases.csv: All Cobenfy case details
    - Prints to console: case counts, demographics summary

Usage:
    python scripts/04_case_identification.py

Requires: data/processed/faers.duckdb (with drug_std table from script 03)
"""

import duckdb
import pandas as pd
import numpy as np
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Suspect drug roles to include in case definition
SUSPECT_ROLES = ("PS", "SS")


def standardise_age(age_val: str, age_cod: str) -> float | None:
    """Convert age to years."""
    try:
        age = float(age_val)
    except (ValueError, TypeError):
        return None

    if age_cod is None:
        return None

    cod = str(age_cod).upper().strip()
    if cod in ("YR", "YEAR", "Y"):
        return age
    elif cod in ("MON", "MONTH", "MO"):
        return age / 12.0
    elif cod in ("WK", "WEEK"):
        return age / 52.0
    elif cod in ("DY", "DAY", "D"):
        return age / 365.25
    elif cod in ("DEC", "DECADE"):
        return age * 10.0
    elif cod in ("HR", "HOUR"):
        return age / (365.25 * 24)
    else:
        return age  # Assume years if unrecognised


def extract_cases(con: duckdb.DuckDBPyConnection):
    """Extract Cobenfy and comparator case sets."""

    print("\n  CASE IDENTIFICATION")
    print(f"  {'─' * 60}")

    # Total reports in database
    total_reports = con.execute("SELECT count(DISTINCT primaryid) FROM demo").fetchone()[0]
    print(f"  Total reports in database: {total_reports:,}")

    # Cobenfy cases (PS or SS)
    cobenfy_cases = con.execute(f"""
        SELECT DISTINCT d.primaryid
        FROM drug_std d
        WHERE d.is_cobenfy = TRUE
          AND UPPER(d.role_cod) IN ('PS', 'SS')
    """).fetchdf()
    n_cobenfy = len(cobenfy_cases)
    print(f"\n  Cobenfy cases (PS+SS):     {n_cobenfy:,}")

    # Cobenfy PS only
    cobenfy_ps = con.execute("""
        SELECT count(DISTINCT primaryid)
        FROM drug_std
        WHERE is_cobenfy = TRUE AND UPPER(role_cod) = 'PS'
    """).fetchone()[0]
    print(f"  Cobenfy cases (PS only):   {cobenfy_ps:,}")

    # Comparator cases
    comparators = ["olanzapine", "risperidone", "aripiprazole",
                    "quetiapine", "lurasidone", "brexpiprazole"]

    print(f"\n  COMPARATOR CASE COUNTS (PS+SS)")
    print(f"  {'─' * 40}")
    for comp in comparators:
        n = con.execute(f"""
            SELECT count(DISTINCT primaryid)
            FROM drug_std
            WHERE std_drug = '{comp}'
              AND UPPER(role_cod) IN ('PS', 'SS')
        """).fetchone()[0]
        print(f"  {comp:<20s}: {n:>8,}")

    return cobenfy_cases


def build_demographics_table(con: duckdb.DuckDBPyConnection):
    """Build Table 1: Demographics for Cobenfy vs comparators vs all FAERS."""

    drugs_of_interest = [
        "xanomeline-trospium", "olanzapine", "risperidone",
        "aripiprazole", "quetiapine", "lurasidone", "brexpiprazole",
    ]

    results = {}

    for drug in drugs_of_interest:
        df = con.execute(f"""
            SELECT
                dem.primaryid,
                dem.age,
                dem.age_cod,
                dem.sex,
                dem.reporter_country,
                dem.occr_country,
                dem.occp_cod,
                dem.event_dt
            FROM demo dem
            INNER JOIN (
                SELECT DISTINCT primaryid
                FROM drug_std
                WHERE std_drug = '{drug}'
                  AND UPPER(role_cod) IN ('PS', 'SS')
            ) d ON dem.primaryid = d.primaryid
        """).fetchdf()

        n = len(df)
        if n == 0:
            results[drug] = {"n": 0}
            continue

        # Age
        df["age_years"] = df.apply(
            lambda row: standardise_age(row["age"], row["age_cod"]), axis=1
        )
        valid_ages = df["age_years"].dropna()
        valid_ages = valid_ages[(valid_ages >= 0) & (valid_ages <= 120)]

        # Sex
        sex_counts = df["sex"].str.upper().value_counts()

        # Reporter country
        us_reports = (df["reporter_country"].str.upper() == "US").sum()

        # Occupation (reporter type)
        occp_counts = df["occp_cod"].str.upper().value_counts()

        # Outcomes (serious)
        outcomes = con.execute(f"""
            SELECT outc_cod, count(DISTINCT primaryid) as n
            FROM outc
            WHERE primaryid IN (
                SELECT DISTINCT primaryid
                FROM drug_std
                WHERE std_drug = '{drug}'
                  AND UPPER(role_cod) IN ('PS', 'SS')
            )
            GROUP BY outc_cod
            ORDER BY n DESC
        """).fetchdf()

        results[drug] = {
            "n": n,
            "age_median": valid_ages.median() if len(valid_ages) > 0 else None,
            "age_iqr": (valid_ages.quantile(0.25), valid_ages.quantile(0.75)) if len(valid_ages) > 0 else (None, None),
            "age_missing": n - len(valid_ages),
            "female_n": sex_counts.get("F", 0),
            "male_n": sex_counts.get("M", 0),
            "sex_missing": n - sex_counts.get("F", 0) - sex_counts.get("M", 0),
            "us_reports": us_reports,
            "hcp_reports": occp_counts.get("MD", 0) + occp_counts.get("HP", 0) + occp_counts.get("OT", 0),
            "consumer_reports": occp_counts.get("CN", 0),
            "outcomes": outcomes,
        }

    # Print demographics table
    print(f"\n{'=' * 90}")
    print(f"  TABLE 1: DEMOGRAPHICS")
    print(f"{'=' * 90}")

    header_drugs = ["xanomeline-trospium", "olanzapine", "risperidone", "quetiapine"]
    print(f"\n  {'Variable':<30s}", end="")
    for drug in header_drugs:
        label = "Cobenfy" if drug == "xanomeline-trospium" else drug.capitalize()
        print(f"  {label:>14s}", end="")
    print()
    print(f"  {'─' * 90}")

    # N
    print(f"  {'N':<30s}", end="")
    for drug in header_drugs:
        r = results[drug]
        print(f"  {r['n']:>14,}", end="")
    print()

    # Age
    print(f"  {'Age, median (IQR)':<30s}", end="")
    for drug in header_drugs:
        r = results[drug]
        if r["n"] > 0 and r["age_median"] is not None:
            print(f"  {r['age_median']:>4.0f} ({r['age_iqr'][0]:.0f}-{r['age_iqr'][1]:.0f})", end="")
        else:
            print(f"  {'—':>14s}", end="")
    print()

    # Female
    print(f"  {'Female, n (%)':<30s}", end="")
    for drug in header_drugs:
        r = results[drug]
        if r["n"] > 0:
            pct = 100 * r["female_n"] / r["n"]
            print(f"  {r['female_n']:>5,} ({pct:>4.1f}%)", end="")
        else:
            print(f"  {'—':>14s}", end="")
    print()

    # US reports
    print(f"  {'US reports, n (%)':<30s}", end="")
    for drug in header_drugs:
        r = results[drug]
        if r["n"] > 0:
            pct = 100 * r["us_reports"] / r["n"]
            print(f"  {r['us_reports']:>5,} ({pct:>4.1f}%)", end="")
        else:
            print(f"  {'—':>14s}", end="")
    print()

    # HCP reports
    print(f"  {'HCP reports, n (%)':<30s}", end="")
    for drug in header_drugs:
        r = results[drug]
        if r["n"] > 0:
            pct = 100 * r["hcp_reports"] / r["n"]
            print(f"  {r['hcp_reports']:>5,} ({pct:>4.1f}%)", end="")
        else:
            print(f"  {'—':>14s}", end="")
    print()

    # Save demographics data
    rows = []
    for drug, r in results.items():
        if r["n"] > 0:
            rows.append({
                "drug": drug,
                "n_cases": r["n"],
                "age_median": r.get("age_median"),
                "age_q25": r["age_iqr"][0] if r.get("age_iqr") else None,
                "age_q75": r["age_iqr"][1] if r.get("age_iqr") else None,
                "age_missing": r.get("age_missing"),
                "female_n": r.get("female_n"),
                "male_n": r.get("male_n"),
                "sex_missing": r.get("sex_missing"),
                "us_reports": r.get("us_reports"),
                "hcp_reports": r.get("hcp_reports"),
                "consumer_reports": r.get("consumer_reports"),
            })
    demo_df = pd.DataFrame(rows)
    demo_path = OUTPUT_DIR / "demographics.csv"
    demo_df.to_csv(demo_path, index=False)
    print(f"\n  Saved: {demo_path}")

    return results


def export_cobenfy_cases(con: duckdb.DuckDBPyConnection):
    """Export detailed Cobenfy case listing for review."""

    df = con.execute("""
        SELECT
            dem.primaryid,
            dem.caseid,
            dem.age,
            dem.age_cod,
            dem.sex,
            dem.reporter_country,
            dem.occp_cod,
            dem.event_dt,
            dem.fda_dt,
            d.drugname,
            d.role_cod,
            d.dose_vbm,
            d.route
        FROM demo dem
        INNER JOIN drug_std d ON dem.primaryid = d.primaryid
        WHERE d.is_cobenfy = TRUE
          AND UPPER(d.role_cod) IN ('PS', 'SS')
        ORDER BY dem.primaryid
    """).fetchdf()

    out_path = OUTPUT_DIR / "cobenfy_cases.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Exported {len(df)} Cobenfy case-drug records to {out_path}")

    # Top adverse events for Cobenfy
    print(f"\n  TOP 30 ADVERSE EVENTS FOR COBENFY")
    print(f"  {'─' * 50}")

    top_aes = con.execute("""
        SELECT
            r.pt,
            count(DISTINCT r.primaryid) as n_cases
        FROM reac r
        INNER JOIN (
            SELECT DISTINCT primaryid
            FROM drug_std
            WHERE is_cobenfy = TRUE
              AND UPPER(role_cod) IN ('PS', 'SS')
        ) cob ON r.primaryid = cob.primaryid
        WHERE r.pt IS NOT NULL
        GROUP BY r.pt
        ORDER BY n_cases DESC
        LIMIT 30
    """).fetchall()

    for pt, n in top_aes:
        print(f"  {str(pt):45s} {n:>6,}")

    return df


def main():
    t0 = time.time()

    print("=" * 70)
    print("  FAERS Case Identification")
    print("  Project 48: Cobenfy Pharmacovigilance")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        print(f"  Run scripts 01-03 first.")
        return

    con = duckdb.connect(str(DB_PATH))

    # Check that drug_std exists
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "drug_std" not in tables:
        print(f"\n  ERROR: drug_std table not found. Run script 03 first.")
        con.close()
        return

    extract_cases(con)
    build_demographics_table(con)
    export_cobenfy_cases(con)

    con.close()
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  CASE IDENTIFICATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/05_disproportionality.py")


if __name__ == "__main__":
    main()
