"""
Script 06: Active-comparator disproportionality analysis.

Head-to-head comparison of Cobenfy vs each D2-antagonist antipsychotic
for prespecified adverse event preferred terms.

Method:
    For each comparator and each prespecified PT, compute a 2×2 table
    using Cobenfy reports as cases and comparator reports as the reference
    (NOT the full FAERS database). This partially controls for indication
    bias (channelling), as both drugs are used in schizophrenia.

    ROR_active = (a_cob / b_cob) / (a_comp / b_comp)

    where:
        a_cob  = Cobenfy cases with PT
        b_cob  = Cobenfy cases without PT
        a_comp = Comparator cases with PT
        b_comp = Comparator cases without PT

    Multiple testing: Bonferroni correction across all PT×comparator tests.

Comparators: olanzapine, risperidone, aripiprazole, quetiapine,
             lurasidone, brexpiprazole

Outputs:
    - outputs/tables/active_comparator_results.csv
    - outputs/tables/active_comparator_forest.csv (for forest plot)

Usage:
    python scripts/06_active_comparator.py
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Prespecified PTs for active-comparator analysis ────────────────────────
# These are hypothesis-driven based on known mechanistic differences
PRESPECIFIED_PTS = [
    # GI (expect higher with Cobenfy — muscarinic agonism)
    "NAUSEA",
    "VOMITING",
    "CONSTIPATION",
    "DYSPEPSIA",
    "DIARRHOEA",
    # Cardiovascular (expect signal with Cobenfy)
    "TACHYCARDIA",
    "HYPERTENSION",
    "BLOOD PRESSURE INCREASED",
    # Metabolic (expect LOWER with Cobenfy vs D2 antagonists)
    "WEIGHT INCREASED",
    "HYPERGLYCAEMIA",
    "DIABETES MELLITUS",
    "METABOLIC SYNDROME",
    "BLOOD GLUCOSE INCREASED",
    "DYSLIPIDAEMIA",
    # EPS (expect LOWER with Cobenfy vs D2 antagonists)
    "DYSTONIA",
    "AKATHISIA",
    "PARKINSONISM",
    "TARDIVE DYSKINESIA",
    "EXTRAPYRAMIDAL DISORDER",
    "TREMOR",
    # Hormonal (expect LOWER with Cobenfy)
    "HYPERPROLACTINAEMIA",
    "GALACTORRHOEA",
    "AMENORRHOEA",
    # Sedation
    "SOMNOLENCE",
    "SEDATION",
    # Cardiac conduction
    "ELECTROCARDIOGRAM QT PROLONGED",
    # Anticholinergic (trospium-related)
    "URINARY RETENTION",
    "DRY MOUTH",
    "VISION BLURRED",
    # Other
    "INSOMNIA",
    "HEADACHE",
    "DIZZINESS",
]

COMPARATORS = [
    "olanzapine",
    "risperidone",
    "aripiprazole",
    "quetiapine",
    "lurasidone",
    "brexpiprazole",
]


def compute_active_comparator_ror(
    con: duckdb.DuckDBPyConnection,
    drug_a: str,
    drug_b: str,
    pt: str,
) -> dict | None:
    """
    Compute active-comparator ROR for drug_a vs drug_b for a given PT.

    Returns dict with ROR, CI, counts, or None if insufficient data.
    """
    # Get case counts
    result = con.execute(f"""
        WITH drug_a_cases AS (
            SELECT DISTINCT primaryid
            FROM drug_std
            WHERE std_drug = '{drug_a}' AND UPPER(role_cod) IN ('PS', 'SS')
        ),
        drug_b_cases AS (
            SELECT DISTINCT primaryid
            FROM drug_std
            WHERE std_drug = '{drug_b}' AND UPPER(role_cod) IN ('PS', 'SS')
        ),
        -- Cobenfy cases with this PT
        a_with_pt AS (
            SELECT count(DISTINCT r.primaryid) as n
            FROM reac r
            INNER JOIN drug_a_cases d ON r.primaryid = d.primaryid
            WHERE UPPER(r.pt) = '{pt}'
        ),
        -- Cobenfy cases total
        a_total AS (
            SELECT count(*) as n FROM drug_a_cases
        ),
        -- Comparator cases with this PT
        b_with_pt AS (
            SELECT count(DISTINCT r.primaryid) as n
            FROM reac r
            INNER JOIN drug_b_cases d ON r.primaryid = d.primaryid
            WHERE UPPER(r.pt) = '{pt}'
        ),
        -- Comparator cases total
        b_total AS (
            SELECT count(*) as n FROM drug_b_cases
        )
        SELECT
            (SELECT n FROM a_with_pt) as a,
            (SELECT n FROM a_total) as n_a,
            (SELECT n FROM b_with_pt) as c,
            (SELECT n FROM b_total) as n_b
    """).fetchone()

    a = result[0]   # Cobenfy + PT
    n_a = result[1]  # Cobenfy total
    c = result[2]   # Comparator + PT
    n_b = result[3]  # Comparator total

    b = n_a - a      # Cobenfy without PT
    d = n_b - c      # Comparator without PT

    if n_a == 0 or n_b == 0:
        return None

    # ROR with 0.5 continuity correction
    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ror = (ac * dc) / (bc * cc)
    ln_ror = np.log(ror)
    se = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)

    # p-value (two-sided)
    z = ln_ror / se
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "pt": pt,
        "a_cobenfy": a,
        "n_cobenfy": n_a,
        "pct_cobenfy": 100 * a / n_a if n_a > 0 else 0,
        "a_comparator": c,
        "n_comparator": n_b,
        "pct_comparator": 100 * c / n_b if n_b > 0 else 0,
        "ror": ror,
        "ror_lower95": np.exp(ln_ror - 1.96 * se),
        "ror_upper95": np.exp(ln_ror + 1.96 * se),
        "p_value": p_value,
    }


def main():
    t0 = time.time()

    print("=" * 70)
    print("  Active-Comparator Disproportionality Analysis")
    print("  Cobenfy vs D2-Antagonist Antipsychotics")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Compute all comparisons
    all_results = []
    n_tests = len(PRESPECIFIED_PTS) * len(COMPARATORS)
    bonferroni_alpha = 0.05 / n_tests

    print(f"\n  Prespecified PTs: {len(PRESPECIFIED_PTS)}")
    print(f"  Comparators: {len(COMPARATORS)}")
    print(f"  Total tests: {n_tests}")
    print(f"  Bonferroni threshold: p < {bonferroni_alpha:.6f}")

    for comp in COMPARATORS:
        print(f"\n  {'─' * 50}")
        print(f"  Cobenfy vs {comp}")
        print(f"  {'─' * 50}")

        for pt in PRESPECIFIED_PTS:
            result = compute_active_comparator_ror(con, "xanomeline-trospium", comp, pt)
            if result:
                result["bonferroni_sig"] = result["p_value"] < bonferroni_alpha
                all_results.append(result)

                # Print significant results
                if result["a_cobenfy"] > 0 or result["a_comparator"] > 0:
                    sig = "***" if result["bonferroni_sig"] else ("*" if result["p_value"] < 0.05 else "")
                    direction = "↑" if result["ror"] > 1 else "↓"
                    print(
                        f"    {pt:<35s} "
                        f"Cob:{result['a_cobenfy']:>4d}/{result['n_cobenfy']:>5d} "
                        f"vs {result['a_comparator']:>5d}/{result['n_comparator']:>6d}  "
                        f"ROR={result['ror']:>6.2f} "
                        f"({result['ror_lower95']:.2f}-{result['ror_upper95']:.2f}) "
                        f"{direction} {sig}"
                    )

    con.close()

    if not all_results:
        print("\n  No results computed. Check that drug_std table has data.")
        return

    # Build results DataFrame
    results_df = pd.DataFrame(all_results)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  ACTIVE-COMPARATOR SUMMARY")
    print(f"{'=' * 70}")

    sig_results = results_df[results_df["bonferroni_sig"]]
    print(f"\n  Total comparisons: {len(results_df)}")
    print(f"  Bonferroni-significant: {len(sig_results)}")

    if len(sig_results) > 0:
        print(f"\n  SIGNIFICANT RESULTS (Bonferroni-corrected)")
        print(f"  {'─' * 85}")
        for _, row in sig_results.sort_values("ror", ascending=False).iterrows():
            direction = "HIGHER in Cobenfy" if row["ror"] > 1 else "LOWER in Cobenfy"
            print(
                f"    {row['pt']:<30s} vs {row['drug_b']:<15s} "
                f"ROR={row['ror']:.2f} ({row['ror_lower95']:.2f}-{row['ror_upper95']:.2f})  "
                f"{direction}"
            )

    # Key mechanistic comparisons
    print(f"\n  KEY MECHANISTIC COMPARISONS (vs olanzapine)")
    print(f"  {'─' * 60}")
    olz = results_df[results_df["drug_b"] == "olanzapine"]
    key_pts = ["WEIGHT INCREASED", "NAUSEA", "AKATHISIA", "TACHYCARDIA",
               "CONSTIPATION", "SOMNOLENCE", "TARDIVE DYSKINESIA"]
    for pt in key_pts:
        row = olz[olz["pt"] == pt]
        if len(row) > 0:
            r = row.iloc[0]
            direction = "↑ Cobenfy" if r["ror"] > 1 else "↓ Cobenfy"
            sig = "**" if r["bonferroni_sig"] else ("*" if r["p_value"] < 0.05 else "ns")
            print(
                f"    {pt:<30s} ROR={r['ror']:>6.2f} "
                f"({r['ror_lower95']:.2f}-{r['ror_upper95']:.2f})  "
                f"{direction:>12s}  {sig}"
            )

    # Save
    print(f"\n{'─' * 70}")
    print("  Saving results...")

    out_path = OUTPUT_DIR / "active_comparator_results.csv"
    results_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  {out_path.name}: {len(results_df)} rows")

    # Forest plot data (for Figure 2)
    forest_path = OUTPUT_DIR / "active_comparator_forest.csv"
    forest_cols = ["drug_b", "pt", "ror", "ror_lower95", "ror_upper95",
                   "a_cobenfy", "n_cobenfy", "a_comparator", "n_comparator",
                   "p_value", "bonferroni_sig"]
    results_df[forest_cols].to_csv(forest_path, index=False, float_format="%.4f")
    print(f"  {forest_path.name}: {len(results_df)} rows")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ACTIVE-COMPARATOR ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/07_time_to_onset.py")


if __name__ == "__main__":
    main()
