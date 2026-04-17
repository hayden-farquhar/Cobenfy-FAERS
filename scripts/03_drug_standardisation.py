"""
Script 03: Drug name standardisation.

Maps FAERS free-text drug names to standardised names for Cobenfy
and all active comparator antipsychotics. Uses pattern matching on
both the drugname and prod_ai (active ingredient) fields.

Drug mapping strategy:
    1. Uppercase the drugname and prod_ai fields
    2. Match against curated pattern lists (generic names, brand names,
       common misspellings, combination products)
    3. Assign a standardised drug name (std_drug) to each match
    4. Write a drug_standardised table back to DuckDB

Comparator drugs:
    - Olanzapine (Zyprexa)
    - Risperidone (Risperdal)
    - Aripiprazole (Abilify)
    - Quetiapine (Seroquel)
    - Lurasidone (Latuda)
    - Brexpiprazole (Rexulti)

Usage:
    python scripts/03_drug_standardisation.py

Requires: data/processed/faers.duckdb (from script 02)
"""

import duckdb
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"

# ── Drug pattern definitions ────────────────────────────────────────────────
# Each entry: standardised name -> list of UPPERCASE patterns to match
# Patterns are matched as substrings in drugname or prod_ai

DRUG_PATTERNS: dict[str, list[str]] = {
    "xanomeline-trospium": [
        "XANOMELINE",
        "COBENFY",
        "KARXT",
        "KAR-XT",
        # Trospium alone is an overactive bladder drug — only match when
        # paired with xanomeline. We handle this in the matching logic.
    ],
    "olanzapine": [
        "OLANZAPINE",
        "ZYPREXA",
        "ZALASTA",
        "ZYDIS",       # Zyprexa Zydis (orally disintegrating)
        "LYBALVI",     # olanzapine + samidorphan combination
    ],
    "risperidone": [
        "RISPERIDONE",
        "RISPERDAL",
        "PERSERIS",    # risperidone extended-release injectable
        "UZEDY",       # risperidone subcutaneous
    ],
    "aripiprazole": [
        "ARIPIPRAZOLE",
        "ABILIFY",
        "ARISTADA",    # aripiprazole lauroxil (long-acting injectable)
    ],
    "quetiapine": [
        "QUETIAPINE",
        "SEROQUEL",
    ],
    "lurasidone": [
        "LURASIDONE",
        "LATUDA",
    ],
    "brexpiprazole": [
        "BREXPIPRAZOLE",
        "REXULTI",
    ],
}

# CYP2D6 inhibitors for exploratory interaction analysis
CYP2D6_INHIBITORS: dict[str, list[str]] = {
    "bupropion": ["BUPROPION", "WELLBUTRIN", "ZYBAN", "APLENZIN"],
    "fluoxetine": ["FLUOXETINE", "PROZAC", "SARAFEM"],
    "paroxetine": ["PAROXETINE", "PAXIL", "SEROXAT", "BRISDELLE"],
}


def match_drug(drugname: str, prod_ai: str, patterns: list[str]) -> bool:
    """Check if a drug record matches any pattern in the list."""
    text = f"{drugname} {prod_ai}".upper()
    return any(pat in text for pat in patterns)


def standardise_drugs(con: duckdb.DuckDBPyConnection):
    """
    Add standardised drug columns to the drug table.

    Creates a new table 'drug_std' with:
        - All original drug columns
        - std_drug: standardised drug name (NULL if not a drug of interest)
        - is_cobenfy: boolean flag
        - is_comparator: boolean flag
        - cyp2d6_inhibitor: name of CYP2D6 inhibitor if co-prescribed (NULL otherwise)
    """
    print("  Building SQL CASE expression for drug matching...")

    # Build CASE expression for main drugs
    case_parts = []
    for std_name, patterns in DRUG_PATTERNS.items():
        conditions = []
        for pat in patterns:
            conditions.append(f"UPPER(COALESCE(drugname,'')) LIKE '%{pat}%'")
            conditions.append(f"UPPER(COALESCE(prod_ai,'')) LIKE '%{pat}%'")
        condition_str = " OR ".join(conditions)
        case_parts.append(f"WHEN ({condition_str}) THEN '{std_name}'")

    std_drug_case = "CASE " + " ".join(case_parts) + " ELSE NULL END"

    # Build CASE expression for CYP2D6 inhibitors
    cyp_parts = []
    for std_name, patterns in CYP2D6_INHIBITORS.items():
        conditions = []
        for pat in patterns:
            conditions.append(f"UPPER(COALESCE(drugname,'')) LIKE '%{pat}%'")
            conditions.append(f"UPPER(COALESCE(prod_ai,'')) LIKE '%{pat}%'")
        condition_str = " OR ".join(conditions)
        cyp_parts.append(f"WHEN ({condition_str}) THEN '{std_name}'")

    cyp_case = "CASE " + " ".join(cyp_parts) + " ELSE NULL END"

    # Create standardised drug table
    con.execute(f"""
        CREATE OR REPLACE TABLE drug_std AS
        SELECT
            *,
            {std_drug_case} AS std_drug,
            CASE WHEN ({std_drug_case}) = 'xanomeline-trospium' THEN TRUE ELSE FALSE END AS is_cobenfy,
            CASE WHEN ({std_drug_case}) IN (
                'olanzapine', 'risperidone', 'aripiprazole',
                'quetiapine', 'lurasidone', 'brexpiprazole'
            ) THEN TRUE ELSE FALSE END AS is_comparator,
            {cyp_case} AS cyp2d6_inhibitor
        FROM drug
    """)

    # Summary statistics
    print("\n  DRUG STANDARDISATION RESULTS")
    print(f"  {'─' * 60}")

    # Count by standardised drug
    results = con.execute("""
        SELECT
            COALESCE(std_drug, '(other)') as drug,
            count(*) as n_records,
            count(DISTINCT primaryid) as n_cases
        FROM drug_std
        WHERE std_drug IS NOT NULL
        GROUP BY std_drug
        ORDER BY n_cases DESC
    """).fetchall()

    print(f"  {'Drug':<25s} {'Records':>10s} {'Cases':>10s}")
    print(f"  {'─' * 50}")
    for drug, n_records, n_cases in results:
        print(f"  {drug:<25s} {n_records:>10,} {n_cases:>10,}")

    # Total database size
    total_records = con.execute("SELECT count(*) FROM drug_std").fetchone()[0]
    total_matched = con.execute("SELECT count(*) FROM drug_std WHERE std_drug IS NOT NULL").fetchone()[0]
    print(f"\n  Total drug records:      {total_records:>10,}")
    print(f"  Matched to study drugs:  {total_matched:>10,} ({100*total_matched/total_records:.1f}%)")

    # Role code breakdown for Cobenfy
    print(f"\n  COBENFY BY ROLE CODE")
    print(f"  {'─' * 40}")
    roles = con.execute("""
        SELECT
            UPPER(role_cod) as role,
            count(*) as n,
            count(DISTINCT primaryid) as n_cases
        FROM drug_std
        WHERE is_cobenfy = TRUE
        GROUP BY UPPER(role_cod)
        ORDER BY n DESC
    """).fetchall()

    for role, n, n_cases in roles:
        role_desc = {"PS": "Primary suspect", "SS": "Secondary suspect",
                     "C": "Concomitant", "I": "Interacting"}.get(role, role)
        print(f"  {role:3s} ({role_desc:<20s}): {n:>6,} records ({n_cases:>5,} cases)")

    # CYP2D6 co-medication in Cobenfy cases
    print(f"\n  CYP2D6 INHIBITOR CO-MEDICATION IN COBENFY CASES")
    print(f"  {'─' * 50}")
    cyp_results = con.execute("""
        SELECT
            cyp.cyp2d6_inhibitor,
            count(DISTINCT cyp.primaryid) as n_cases
        FROM drug_std cyp
        INNER JOIN (
            SELECT DISTINCT primaryid
            FROM drug_std
            WHERE is_cobenfy = TRUE
        ) cob ON cyp.primaryid = cob.primaryid
        WHERE cyp.cyp2d6_inhibitor IS NOT NULL
        GROUP BY cyp.cyp2d6_inhibitor
        ORDER BY n_cases DESC
    """).fetchall()

    if cyp_results:
        for inh, n_cases in cyp_results:
            print(f"  {inh:<20s}: {n_cases:>5,} Cobenfy cases")
    else:
        print(f"  No CYP2D6 inhibitor co-prescriptions found")

    # Show raw drugname variants that matched Cobenfy
    print(f"\n  RAW DRUGNAME VARIANTS MATCHED TO COBENFY")
    print(f"  {'─' * 60}")
    variants = con.execute("""
        SELECT drugname, count(*) as n
        FROM drug_std
        WHERE is_cobenfy = TRUE
        GROUP BY drugname
        ORDER BY n DESC
        LIMIT 20
    """).fetchall()
    for name, n in variants:
        print(f"  {str(name):50s} {n:>6,}")


def main():
    t0 = time.time()

    print("=" * 70)
    print("  FAERS Drug Name Standardisation")
    print("  Project 48: Cobenfy Pharmacovigilance")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        print(f"  Run script 02 first.")
        return

    con = duckdb.connect(str(DB_PATH))

    standardise_drugs(con)

    con.close()
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  STANDARDISATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/04_case_identification.py")


if __name__ == "__main__":
    main()
