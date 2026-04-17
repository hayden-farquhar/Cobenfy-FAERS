"""
Script 10: Supplementary analyses for robustness and completeness.

Implements eight additional analyses:
    1. Outcome severity analysis (OUTC table)
    2. Indication-restricted analysis (schizophrenia only, INDI table)
    3. Time-stratified disproportionality (early vs late quarters)
    4. Positive/negative control validation
    5. Reporting completeness metrics
    6. MedDRA SOC-level aggregation
    7. COVID-19 vaccine co-report exclusion
    8. Concomitant medication profiling

Usage:
    python scripts/10_supplementary.py

Requires: data/processed/faers.duckdb (with drug_std from script 03)
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
SUPP_DIR = PROJECT_ROOT / "outputs" / "supplementary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

MIN_REPORTS = 3
DRUG = "xanomeline-trospium"
COMPARATORS = ["olanzapine", "risperidone", "aripiprazole",
               "quetiapine", "lurasidone", "brexpiprazole"]


# ── Shared helper: compute ROR for a given set of primaryids ───────────────

def compute_ror_for_subset(con, case_pids_sql, ref_role_filter="IN ('PS','SS')",
                           demo_where="1=1", min_reports=3):
    """Compute ROR for a drug defined by a set of primaryids against
    a reference set defined by filters. Returns DataFrame."""
    N = con.execute(f"""
        SELECT count(DISTINCT d.primaryid)
        FROM drug_std d
        INNER JOIN demo dem ON d.primaryid = dem.primaryid
        WHERE UPPER(d.role_cod) {ref_role_filter} AND {demo_where}
    """).fetchone()[0]

    n_drug = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM ({case_pids_sql})
    """).fetchone()[0]

    if N == 0 or n_drug == 0:
        return pd.DataFrame()

    df = con.execute(f"""
        WITH drug_cases AS ({case_pids_sql}),
        all_reactions AS (
            SELECT DISTINCT r.primaryid, UPPER(r.pt) as pt
            FROM reac r
            INNER JOIN drug_std d ON r.primaryid = d.primaryid
            INNER JOIN demo dem ON r.primaryid = dem.primaryid
            WHERE UPPER(d.role_cod) {ref_role_filter}
              AND r.pt IS NOT NULL AND TRIM(r.pt) != ''
              AND {demo_where}
        ),
        pair_counts AS (
            SELECT ar.pt, count(DISTINCT ar.primaryid) as a
            FROM all_reactions ar
            INNER JOIN drug_cases dc ON ar.primaryid = dc.primaryid
            GROUP BY ar.pt
            HAVING count(DISTINCT ar.primaryid) >= {min_reports}
        ),
        reaction_marginals AS (
            SELECT pt, count(DISTINCT primaryid) as n_reaction
            FROM all_reactions GROUP BY pt
        )
        SELECT p.pt, p.a, rm.n_reaction, {n_drug} as n_drug, {N} as N
        FROM pair_counts p
        INNER JOIN reaction_marginals rm ON p.pt = rm.pt
    """).fetchdf()

    if len(df) == 0:
        return pd.DataFrame()

    df["b"] = df["n_drug"] - df["a"]
    df["c"] = df["n_reaction"] - df["a"]
    df["d"] = df["N"] - df["a"] - df["b"] - df["c"]
    df["expected"] = df["n_drug"].astype(float) * df["n_reaction"].astype(float) / df["N"]

    a, b, c, d = [df[x].values.astype(float) + 0.5 for x in ["a","b","c","d"]]
    ror = (a*d)/(b*c)
    ln_ror = np.log(ror)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    df["ror"] = ror
    df["ror_lower95"] = np.exp(ln_ror - 1.96*se)
    df["ror_upper95"] = np.exp(ln_ror + 1.96*se)
    df["signal_ror"] = df["ror_lower95"] > 1.0

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  1. OUTCOME SEVERITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analysis_1_outcome_severity(con):
    """Compare outcome severity for Cobenfy vs comparators."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Outcome Severity")
    print("=" * 70)

    drugs = [DRUG] + COMPARATORS
    rows = []

    for drug in drugs:
        # Total cases
        n_total = con.execute(f"""
            SELECT count(DISTINCT primaryid)
            FROM drug_std
            WHERE std_drug = '{drug}' AND UPPER(role_cod) IN ('PS','SS')
        """).fetchone()[0]

        # Outcome breakdown
        outcomes = con.execute(f"""
            SELECT
                UPPER(TRIM(o.outc_cod)) as outcome,
                count(DISTINCT o.primaryid) as n
            FROM outc o
            INNER JOIN (
                SELECT DISTINCT primaryid FROM drug_std
                WHERE std_drug = '{drug}' AND UPPER(role_cod) IN ('PS','SS')
            ) d ON o.primaryid = d.primaryid
            GROUP BY UPPER(TRIM(o.outc_cod))
        """).fetchdf()

        outcome_map = dict(zip(outcomes["outcome"], outcomes["n"])) if len(outcomes) > 0 else {}

        label = "Cobenfy" if drug == DRUG else drug.capitalize()
        row = {
            "drug": label,
            "n_total": n_total,
            "death": outcome_map.get("DE", 0),
            "life_threatening": outcome_map.get("LT", 0),
            "hospitalisation": outcome_map.get("HO", 0),
            "disability": outcome_map.get("DS", 0),
            "congenital_anomaly": outcome_map.get("CA", 0),
            "other_serious": outcome_map.get("OT", 0),
            "required_intervention": outcome_map.get("RI", 0),
        }
        row["pct_death"] = 100 * row["death"] / n_total if n_total > 0 else 0
        row["pct_hosp"] = 100 * row["hospitalisation"] / n_total if n_total > 0 else 0
        rows.append(row)

    result_df = pd.DataFrame(rows)

    print(f"\n  {'Drug':<18s} {'N':>6s} {'Death':>8s} {'%':>6s} {'Hosp':>8s} "
          f"{'%':>6s} {'LT':>6s} {'Disab':>6s}")
    print(f"  {'─' * 75}")
    for _, r in result_df.iterrows():
        print(f"  {r['drug']:<18s} {r['n_total']:>6,} {r['death']:>8,} "
              f"{r['pct_death']:>5.1f}% {r['hospitalisation']:>8,} "
              f"{r['pct_hosp']:>5.1f}% {r['life_threatening']:>6,} "
              f"{r['disability']:>6,}")

    # Chi-squared test: Cobenfy death rate vs each comparator
    cob_row = result_df[result_df["drug"] == "Cobenfy"].iloc[0]
    print(f"\n  Death rate comparison (chi-squared):")
    for _, r in result_df.iterrows():
        if r["drug"] == "Cobenfy":
            continue
        table = np.array([
            [cob_row["death"], cob_row["n_total"] - cob_row["death"]],
            [r["death"], r["n_total"] - r["death"]]
        ])
        if table.min() >= 0:
            chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
            print(f"    vs {r['drug']:<15s}: Cobenfy {cob_row['pct_death']:.1f}% "
                  f"vs {r['pct_death']:.1f}%, chi2={chi2:.1f}, p={p:.4f}")

    result_df.to_csv(SUPP_DIR / "outcome_severity.csv", index=False)
    print(f"\n  Saved: supplementary/outcome_severity.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  2. INDICATION-RESTRICTED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analysis_2_indication_restricted(con):
    """Restrict to schizophrenia indication only."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Indication-Restricted (Schizophrenia Only)")
    print("=" * 70)

    # Check what indications Cobenfy has
    indications = con.execute(f"""
        SELECT UPPER(i.indi_pt) as indication, count(DISTINCT d.primaryid) as n
        FROM indi i
        INNER JOIN drug_std d ON i.primaryid = d.primaryid
            AND i.indi_drug_seq = d.drug_seq
        WHERE d.std_drug = '{DRUG}' AND UPPER(d.role_cod) IN ('PS','SS')
        GROUP BY UPPER(i.indi_pt)
        ORDER BY n DESC
        LIMIT 15
    """).fetchall()

    print(f"\n  Top indications for Cobenfy:")
    for ind, n in indications:
        print(f"    {str(ind):45s} {n:>5,}")

    # Restrict to schizophrenia
    schiz_patterns = "('SCHIZOPHRENIA', 'SCHIZOAFFECTIVE DISORDER', 'PSYCHOTIC DISORDER', 'PSYCHOSIS')"
    case_sql = f"""
        SELECT DISTINCT d.primaryid
        FROM drug_std d
        INNER JOIN indi i ON d.primaryid = i.primaryid AND d.drug_seq = i.indi_drug_seq
        WHERE d.std_drug = '{DRUG}'
          AND UPPER(d.role_cod) IN ('PS','SS')
          AND UPPER(i.indi_pt) IN {schiz_patterns}
    """

    n_schiz = con.execute(f"SELECT count(DISTINCT primaryid) FROM ({case_sql})").fetchone()[0]
    n_total = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS','SS')
    """).fetchone()[0]
    print(f"\n  Cobenfy cases with schizophrenia indication: {n_schiz:,} / {n_total:,} ({100*n_schiz/n_total:.0f}%)")

    if n_schiz < 10:
        print("  Insufficient cases for indication-restricted analysis.")
        return

    df = compute_ror_for_subset(con, case_sql)
    if len(df) == 0:
        print("  No drug-PT pairs met threshold.")
        return

    n_sig = df["signal_ror"].sum()
    print(f"  Drug-PT pairs: {len(df):,}  |  ROR signals: {n_sig}")

    # Compare key PTs with primary analysis
    primary = pd.read_csv(OUTPUT_DIR / "disproportionality_cobenfy_full.csv")
    key_pts = ["NAUSEA", "VOMITING", "CONSTIPATION", "URINARY RETENTION",
               "TACHYCARDIA", "AKATHISIA", "WEIGHT INCREASED"]

    print(f"\n  {'PT':<30s} {'ROR (all)':>12s} {'ROR (schiz)':>12s}")
    print(f"  {'─' * 60}")
    for pt in key_pts:
        p_match = primary[primary["pt"].str.upper() == pt]
        s_match = df[df["pt"].str.upper() == pt]
        p_ror = f"{p_match.iloc[0]['ror']:.1f}" if len(p_match) > 0 else "—"
        s_ror = f"{s_match.iloc[0]['ror']:.1f}" if len(s_match) > 0 else "—"
        print(f"  {pt:<30s} {p_ror:>12s} {s_ror:>12s}")

    df.to_csv(SUPP_DIR / "indication_restricted_schizophrenia.csv",
              index=False, float_format="%.4f")
    print(f"\n  Saved: supplementary/indication_restricted_schizophrenia.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  3. TIME-STRATIFIED DISPROPORTIONALITY
# ═══════════════════════════════════════════════════════════════════════════

def analysis_3_time_stratified(con):
    """Compute ROR in early vs late quarters."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Time-Stratified Disproportionality")
    print("=" * 70)

    # Define periods: early (Q4 2024 + Q1 2025) vs late (Q2-Q4 2025)
    periods = {
        "early (Q4'24–Q1'25)": "SUBSTRING(dem.fda_dt,1,6) <= '202503'",
        "late (Q2'25–Q4'25)": "SUBSTRING(dem.fda_dt,1,6) > '202503'",
    }

    key_pts = ["NAUSEA", "VOMITING", "CONSTIPATION", "URINARY RETENTION",
               "TACHYCARDIA", "DROOLING", "DRY MOUTH", "DYSPEPSIA",
               "VISION BLURRED", "HYPERHIDROSIS", "AKATHISIA", "TREMOR"]

    results = []
    for period_label, date_filter in periods.items():
        case_sql = f"""
            SELECT DISTINCT d.primaryid
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) IN ('PS','SS')
              AND dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6
              AND {date_filter}
        """

        n_cases = con.execute(f"SELECT count(*) FROM ({case_sql})").fetchone()[0]
        print(f"\n  {period_label}: {n_cases:,} Cobenfy cases")

        df = compute_ror_for_subset(con, case_sql, demo_where=f"dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6 AND {date_filter}")

        for pt in key_pts:
            match = df[df["pt"].str.upper() == pt]
            if len(match) > 0:
                r = match.iloc[0]
                results.append({
                    "period": period_label,
                    "pt": pt,
                    "n": r["a"],
                    "ror": r["ror"],
                    "ror_lower95": r["ror_lower95"],
                    "ror_upper95": r["ror_upper95"],
                    "signal": r["signal_ror"],
                })
            else:
                results.append({
                    "period": period_label, "pt": pt, "n": 0,
                    "ror": None, "ror_lower95": None, "ror_upper95": None,
                    "signal": False,
                })

    results_df = pd.DataFrame(results)

    # Display comparison
    print(f"\n  {'PT':<30s} {'Early ROR':>12s} {'Late ROR':>12s} {'Stable?':>8s}")
    print(f"  {'─' * 65}")
    for pt in key_pts:
        early = results_df[(results_df["pt"]==pt) & (results_df["period"].str.contains("early"))]
        late = results_df[(results_df["pt"]==pt) & (results_df["period"].str.contains("late"))]
        e_ror = f"{early.iloc[0]['ror']:.1f}" if len(early)>0 and pd.notna(early.iloc[0]['ror']) else "—"
        l_ror = f"{late.iloc[0]['ror']:.1f}" if len(late)>0 and pd.notna(late.iloc[0]['ror']) else "—"

        # Assess stability
        if e_ror != "—" and l_ror != "—":
            ratio = float(l_ror) / float(e_ror) if float(e_ror) > 0 else 0
            stable = "Yes" if 0.5 < ratio < 2.0 else "No"
        else:
            stable = "—"

        print(f"  {pt:<30s} {e_ror:>12s} {l_ror:>12s} {stable:>8s}")

    results_df.to_csv(SUPP_DIR / "time_stratified_disproportionality.csv",
                      index=False, float_format="%.4f")
    print(f"\n  Saved: supplementary/time_stratified_disproportionality.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  4. POSITIVE/NEGATIVE CONTROL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def analysis_4_controls(con):
    """Validate pipeline with known drug-event associations."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Positive/Negative Control Validation")
    print("=" * 70)

    # Positive controls: well-established drug-AE associations
    positive_controls = [
        ("olanzapine", "WEIGHT INCREASED", "Expected strong signal"),
        ("olanzapine", "HYPERGLYCAEMIA", "Expected signal"),
        ("olanzapine", "DIABETES MELLITUS", "Expected signal"),
        ("risperidone", "HYPERPROLACTINAEMIA", "Expected strong signal"),
        ("risperidone", "TARDIVE DYSKINESIA", "Expected signal"),
        ("quetiapine", "SOMNOLENCE", "Expected strong signal"),
        ("quetiapine", "WEIGHT INCREASED", "Expected signal"),
        ("aripiprazole", "AKATHISIA", "Expected signal"),
    ]

    # Negative controls: associations that should NOT be signals
    negative_controls = [
        ("olanzapine", "URINARY RETENTION", "Not expected"),
        ("aripiprazole", "WEIGHT INCREASED", "Weak/no signal expected"),
        ("quetiapine", "HYPERPROLACTINAEMIA", "Weak signal expected"),
    ]

    print(f"\n  POSITIVE CONTROLS (should detect signal)")
    print(f"  {'─' * 80}")
    print(f"  {'Drug':<18s} {'PT':<30s} {'n':>5s} {'ROR':>7s} {'95% CI':>16s} {'Signal':>7s}")
    print(f"  {'─' * 80}")

    control_results = []
    for drug, pt, expectation in positive_controls:
        result = _compute_single_ror(con, drug, pt)
        if result:
            sig = "YES" if result["signal"] else "no"
            print(f"  {drug:<18s} {pt:<30s} {result['a']:>5.0f} {result['ror']:>7.1f} "
                  f"({result['ror_lower']:.1f}-{result['ror_upper']:.1f}) {sig:>7s}")
            control_results.append({**result, "type": "positive", "expectation": expectation})
        else:
            print(f"  {drug:<18s} {pt:<30s}     — (insufficient data)")

    print(f"\n  NEGATIVE CONTROLS (should NOT detect strong signal)")
    print(f"  {'─' * 80}")
    print(f"  {'Drug':<18s} {'PT':<30s} {'n':>5s} {'ROR':>7s} {'95% CI':>16s} {'Signal':>7s}")
    print(f"  {'─' * 80}")

    for drug, pt, expectation in negative_controls:
        result = _compute_single_ror(con, drug, pt)
        if result:
            sig = "YES" if result["signal"] else "no"
            print(f"  {drug:<18s} {pt:<30s} {result['a']:>5.0f} {result['ror']:>7.1f} "
                  f"({result['ror_lower']:.1f}-{result['ror_upper']:.1f}) {sig:>7s}")
            control_results.append({**result, "type": "negative", "expectation": expectation})
        else:
            print(f"  {drug:<18s} {pt:<30s}     — (insufficient data)")

    # Summary
    pos_results = [r for r in control_results if r["type"] == "positive"]
    neg_results = [r for r in control_results if r["type"] == "negative"]
    pos_detected = sum(1 for r in pos_results if r["signal"])
    neg_not_detected = sum(1 for r in neg_results if not r["signal"])
    print(f"\n  Positive controls detected: {pos_detected}/{len(pos_results)}")
    print(f"  Negative controls correctly negative: {neg_not_detected}/{len(neg_results)}")

    if control_results:
        pd.DataFrame(control_results).to_csv(
            SUPP_DIR / "control_validation.csv", index=False, float_format="%.4f")
        print(f"\n  Saved: supplementary/control_validation.csv")


def _compute_single_ror(con, drug, pt):
    """Compute ROR for a single drug-PT pair against full database."""
    result = con.execute(f"""
        WITH drug_cases AS (
            SELECT DISTINCT primaryid FROM drug_std
            WHERE std_drug = '{drug}' AND UPPER(role_cod) IN ('PS','SS')
        ),
        all_suspected AS (
            SELECT DISTINCT primaryid FROM drug_std
            WHERE UPPER(role_cod) IN ('PS','SS')
        ),
        a_val AS (
            SELECT count(DISTINCT r.primaryid) as n FROM reac r
            INNER JOIN drug_cases dc ON r.primaryid = dc.primaryid
            WHERE UPPER(r.pt) = '{pt}'
        ),
        n_drug AS (SELECT count(*) as n FROM drug_cases),
        n_reaction AS (
            SELECT count(DISTINCT r.primaryid) as n FROM reac r
            INNER JOIN all_suspected s ON r.primaryid = s.primaryid
            WHERE UPPER(r.pt) = '{pt}'
        ),
        n_total AS (SELECT count(*) as n FROM all_suspected)
        SELECT
            (SELECT n FROM a_val) as a,
            (SELECT n FROM n_drug) as n_drug,
            (SELECT n FROM n_reaction) as n_reaction,
            (SELECT n FROM n_total) as N
    """).fetchone()

    a, n_drug, n_reaction, N = result
    if a < 3 or n_drug == 0:
        return None

    b = n_drug - a
    c = n_reaction - a
    d = N - a - b - c

    ac, bc, cc, dc = a+0.5, b+0.5, c+0.5, d+0.5
    ror = (ac*dc)/(bc*cc)
    ln_ror = np.log(ror)
    se = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)

    return {
        "drug": drug, "pt": pt, "a": a, "n_drug": n_drug,
        "ror": ror, "ror_lower": np.exp(ln_ror - 1.96*se),
        "ror_upper": np.exp(ln_ror + 1.96*se),
        "signal": np.exp(ln_ror - 1.96*se) > 1.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. REPORTING COMPLETENESS METRICS
# ═══════════════════════════════════════════════════════════════════════════

def analysis_5_completeness(con):
    """Report data completeness for Cobenfy vs comparators."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: Reporting Completeness")
    print("=" * 70)

    drugs = [DRUG] + COMPARATORS
    rows = []

    for drug in drugs:
        result = con.execute(f"""
            SELECT
                count(DISTINCT dem.primaryid) as n_total,
                count(DISTINCT CASE WHEN dem.age IS NOT NULL AND TRIM(dem.age) != ''
                    THEN dem.primaryid END) as has_age,
                count(DISTINCT CASE WHEN dem.sex IS NOT NULL AND UPPER(dem.sex) IN ('M','F')
                    THEN dem.primaryid END) as has_sex,
                count(DISTINCT CASE WHEN dem.event_dt IS NOT NULL AND LENGTH(dem.event_dt) >= 6
                    THEN dem.primaryid END) as has_event_dt,
                count(DISTINCT CASE WHEN dem.reporter_country IS NOT NULL AND TRIM(dem.reporter_country) != ''
                    THEN dem.primaryid END) as has_country,
                count(DISTINCT CASE WHEN dem.occp_cod IS NOT NULL AND TRIM(dem.occp_cod) != ''
                    THEN dem.primaryid END) as has_reporter_type
            FROM demo dem
            INNER JOIN (
                SELECT DISTINCT primaryid FROM drug_std
                WHERE std_drug = '{drug}' AND UPPER(role_cod) IN ('PS','SS')
            ) d ON dem.primaryid = d.primaryid
        """).fetchone()

        n = result[0]
        label = "Cobenfy" if drug == DRUG else drug.capitalize()
        rows.append({
            "drug": label,
            "n_cases": n,
            "age_pct": 100 * result[1] / n if n > 0 else 0,
            "sex_pct": 100 * result[2] / n if n > 0 else 0,
            "event_dt_pct": 100 * result[3] / n if n > 0 else 0,
            "country_pct": 100 * result[4] / n if n > 0 else 0,
            "reporter_pct": 100 * result[5] / n if n > 0 else 0,
        })

    # Check indication completeness separately (INDI table)
    for i, drug in enumerate(drugs):
        n_with_indi = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN indi i ON d.primaryid = i.primaryid AND d.drug_seq = i.indi_drug_seq
            WHERE d.std_drug = '{drug}' AND UPPER(d.role_cod) IN ('PS','SS')
              AND i.indi_pt IS NOT NULL AND TRIM(i.indi_pt) != ''
        """).fetchone()[0]
        rows[i]["indication_pct"] = 100 * n_with_indi / rows[i]["n_cases"] if rows[i]["n_cases"] > 0 else 0

    # Therapy dates
    for i, drug in enumerate(drugs):
        n_with_ther = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN ther t ON d.primaryid = t.primaryid AND d.drug_seq = t.dsg_drug_seq
            WHERE d.std_drug = '{drug}' AND UPPER(d.role_cod) IN ('PS','SS')
              AND t.start_dt IS NOT NULL AND TRIM(t.start_dt) != ''
        """).fetchone()[0]
        rows[i]["therapy_dt_pct"] = 100 * n_with_ther / rows[i]["n_cases"] if rows[i]["n_cases"] > 0 else 0

    comp_df = pd.DataFrame(rows)

    print(f"\n  {'Drug':<18s} {'N':>6s} {'Age':>6s} {'Sex':>6s} {'EvDt':>6s} "
          f"{'Ctry':>6s} {'Rptr':>6s} {'Indi':>6s} {'Ther':>6s}")
    print(f"  {'─' * 75}")
    for _, r in comp_df.iterrows():
        print(f"  {r['drug']:<18s} {r['n_cases']:>6,} {r['age_pct']:>5.0f}% "
              f"{r['sex_pct']:>5.0f}% {r['event_dt_pct']:>5.0f}% "
              f"{r['country_pct']:>5.0f}% {r['reporter_pct']:>5.0f}% "
              f"{r['indication_pct']:>5.0f}% {r['therapy_dt_pct']:>5.0f}%")

    comp_df.to_csv(SUPP_DIR / "reporting_completeness.csv",
                   index=False, float_format="%.1f")
    print(f"\n  Saved: supplementary/reporting_completeness.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  6. MedDRA SOC-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════

# Simplified PT-to-SOC mapping for the most common PTs
# Full MedDRA requires a licence; this covers our key signals
PT_TO_SOC = {
    # Gastrointestinal disorders
    "NAUSEA": "Gastrointestinal disorders",
    "VOMITING": "Gastrointestinal disorders",
    "CONSTIPATION": "Gastrointestinal disorders",
    "DYSPEPSIA": "Gastrointestinal disorders",
    "DIARRHOEA": "Gastrointestinal disorders",
    "ABDOMINAL PAIN UPPER": "Gastrointestinal disorders",
    "ABDOMINAL DISCOMFORT": "Gastrointestinal disorders",
    "GASTROOESOPHAGEAL REFLUX DISEASE": "Gastrointestinal disorders",
    "VOMITING PROJECTILE": "Gastrointestinal disorders",
    "GASTROINTESTINAL DISORDER": "Gastrointestinal disorders",
    "DRY MOUTH": "Gastrointestinal disorders",
    "DROOLING": "Gastrointestinal disorders",
    "SALIVARY HYPERSECRETION": "Gastrointestinal disorders",
    "HICCUPS": "Gastrointestinal disorders",
    # Nervous system disorders
    "DIZZINESS": "Nervous system disorders",
    "HEADACHE": "Nervous system disorders",
    "SOMNOLENCE": "Nervous system disorders",
    "TREMOR": "Nervous system disorders",
    "AKATHISIA": "Nervous system disorders",
    "DYSTONIA": "Nervous system disorders",
    "PARKINSONISM": "Nervous system disorders",
    "TARDIVE DYSKINESIA": "Nervous system disorders",
    "EXTRAPYRAMIDAL DISORDER": "Nervous system disorders",
    "SEDATION": "Nervous system disorders",
    "TACHYPHRENIA": "Nervous system disorders",
    # Psychiatric disorders
    "INSOMNIA": "Psychiatric disorders",
    "HALLUCINATION, AUDITORY": "Psychiatric disorders",
    "HALLUCINATION, VISUAL": "Psychiatric disorders",
    "PSYCHOTIC DISORDER": "Psychiatric disorders",
    "PARANOIA": "Psychiatric disorders",
    "MANIA": "Psychiatric disorders",
    "ANGER": "Psychiatric disorders",
    "DELUSION": "Psychiatric disorders",
    "SCHIZOPHRENIA": "Psychiatric disorders",
    "INTRUSIVE THOUGHTS": "Psychiatric disorders",
    "NEGATIVE THOUGHTS": "Psychiatric disorders",
    "THINKING ABNORMAL": "Psychiatric disorders",
    "VIOLENCE-RELATED SYMPTOM": "Psychiatric disorders",
    # Cardiac disorders
    "TACHYCARDIA": "Cardiac disorders",
    # Vascular disorders
    "HYPERTENSION": "Vascular disorders",
    "BLOOD PRESSURE INCREASED": "Investigations",
    # Renal and urinary disorders
    "URINARY RETENTION": "Renal and urinary disorders",
    "DYSURIA": "Renal and urinary disorders",
    # Eye disorders
    "VISION BLURRED": "Eye disorders",
    # Skin disorders
    "HYPERHIDROSIS": "Skin and subcutaneous tissue disorders",
    # Metabolism
    "WEIGHT INCREASED": "Investigations",
    "HYPERGLYCAEMIA": "Metabolism and nutrition disorders",
    "DIABETES MELLITUS": "Metabolism and nutrition disorders",
    # General
    "FATIGUE": "General disorders",
    "DRUG INEFFECTIVE": "General disorders",
    "TREATMENT NONCOMPLIANCE": "General disorders",
    "OFF LABEL USE": "Product issues",
}


def analysis_6_soc_aggregation(con):
    """Aggregate Cobenfy signals by MedDRA System Organ Class."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: MedDRA SOC-Level Aggregation")
    print("=" * 70)

    # Load primary results
    primary = pd.read_csv(OUTPUT_DIR / "disproportionality_cobenfy_full.csv")
    primary["soc"] = primary["pt"].str.upper().map(PT_TO_SOC)

    # Count signals by SOC
    signals = primary[primary["n_methods_signal"] >= 3].copy()
    signals["soc"] = signals["pt"].str.upper().map(PT_TO_SOC)

    # All PTs by SOC
    all_by_soc = primary.groupby("soc").agg(
        n_pts=("pt", "count"),
        n_signals=("n_methods_signal", lambda x: (x >= 3).sum()),
        total_reports=("a", "sum"),
        max_ror=("ror", "max"),
        median_ror=("ror", "median"),
    ).reset_index().sort_values("n_signals", ascending=False)

    print(f"\n  {'SOC':<45s} {'PTs':>5s} {'Sigs':>5s} {'Reports':>8s} {'Max ROR':>8s}")
    print(f"  {'─' * 75}")
    for _, r in all_by_soc.iterrows():
        if pd.notna(r["soc"]):
            print(f"  {str(r['soc'])[:44]:<45s} {r['n_pts']:>5.0f} "
                  f"{r['n_signals']:>5.0f} {r['total_reports']:>8.0f} {r['max_ror']:>8.1f}")

    # Unmapped PTs
    unmapped = primary[primary["soc"].isna()]
    if len(unmapped) > 0:
        print(f"\n  Unmapped PTs: {len(unmapped)} (not in simplified SOC dictionary)")

    all_by_soc.to_csv(SUPP_DIR / "soc_aggregation.csv",
                      index=False, float_format="%.2f")
    print(f"\n  Saved: supplementary/soc_aggregation.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  7. COVID-19 VACCINE CO-REPORT EXCLUSION
# ═══════════════════════════════════════════════════════════════════════════

def analysis_7_covid_exclusion(con):
    """Exclude reports with COVID-19 vaccine co-reporting."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: COVID-19 Vaccine Co-Report Exclusion")
    print("=" * 70)

    # Find Cobenfy cases with COVID vaccine co-reported
    covid_cobenfy = con.execute(f"""
        SELECT count(DISTINCT cob.primaryid) as n
        FROM (
            SELECT DISTINCT primaryid FROM drug_std
            WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
        ) cob
        INNER JOIN drug_std vax ON cob.primaryid = vax.primaryid
        WHERE (UPPER(vax.drugname) LIKE '%COVID%'
            OR UPPER(vax.drugname) LIKE '%PFIZER%BIONTECH%'
            OR UPPER(vax.drugname) LIKE '%MODERNA%'
            OR UPPER(vax.drugname) LIKE '%COMIRNATY%'
            OR UPPER(vax.drugname) LIKE '%SPIKEVAX%'
            OR UPPER(vax.drugname) LIKE '%JANSSEN%'
            OR UPPER(vax.drugname) LIKE '%NOVAVAX%')
    """).fetchone()[0]

    n_total = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
    """).fetchone()[0]

    print(f"\n  Cobenfy cases with COVID vaccine co-report: {covid_cobenfy:,} / {n_total:,} "
          f"({100*covid_cobenfy/n_total:.1f}%)")

    if covid_cobenfy == 0:
        print("  No COVID vaccine co-reports found. No exclusion needed.")
        pd.DataFrame([{"covid_coreports": 0, "total": n_total,
                       "pct": 0}]).to_csv(SUPP_DIR / "covid_exclusion.csv", index=False)
        return

    # Rerun with exclusion
    case_sql = f"""
        SELECT DISTINCT d.primaryid
        FROM drug_std d
        WHERE d.is_cobenfy = TRUE AND UPPER(d.role_cod) IN ('PS','SS')
          AND d.primaryid NOT IN (
              SELECT DISTINCT primaryid FROM drug_std
              WHERE UPPER(drugname) LIKE '%COVID%'
                 OR UPPER(drugname) LIKE '%COMIRNATY%'
                 OR UPPER(drugname) LIKE '%SPIKEVAX%'
          )
    """

    df = compute_ror_for_subset(con, case_sql)
    n_sig = df["signal_ror"].sum() if len(df) > 0 else 0
    print(f"  After exclusion: {len(df)} pairs, {n_sig} ROR signals")

    if len(df) > 0:
        df.to_csv(SUPP_DIR / "covid_exclusion_results.csv",
                  index=False, float_format="%.4f")
    print(f"\n  Saved: supplementary/covid_exclusion*.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  8. CONCOMITANT MEDICATION PROFILING
# ═══════════════════════════════════════════════════════════════════════════

def analysis_8_concomitant(con):
    """Profile concomitant medications in Cobenfy cases."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 8: Concomitant Medication Profiling")
    print("=" * 70)

    # Top concomitant drugs (role_cod = 'C')
    concom = con.execute(f"""
        SELECT
            UPPER(COALESCE(c.prod_ai, c.drugname)) as concom_drug,
            count(DISTINCT c.primaryid) as n_cases
        FROM drug_std c
        INNER JOIN (
            SELECT DISTINCT primaryid FROM drug_std
            WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
        ) cob ON c.primaryid = cob.primaryid
        WHERE UPPER(c.role_cod) = 'C'
          AND c.drugname IS NOT NULL AND TRIM(c.drugname) != ''
          AND c.is_cobenfy = FALSE
        GROUP BY UPPER(COALESCE(c.prod_ai, c.drugname))
        ORDER BY n_cases DESC
        LIMIT 30
    """).fetchall()

    n_total = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
    """).fetchone()[0]

    print(f"\n  TOP 30 CONCOMITANT MEDICATIONS IN COBENFY CASES")
    print(f"  (n = {n_total:,} total Cobenfy cases)")
    print(f"  {'─' * 55}")
    print(f"  {'Drug':<40s} {'n':>6s} {'%':>6s}")
    print(f"  {'─' * 55}")

    concom_rows = []
    for drug, n in concom:
        pct = 100 * n / n_total
        print(f"  {str(drug)[:39]:<40s} {n:>6,} {pct:>5.1f}%")
        concom_rows.append({"drug": drug, "n_cases": n, "pct": pct})

    # Also check for other antipsychotics as concomitant
    print(f"\n  OTHER ANTIPSYCHOTICS AS CONCOMITANT IN COBENFY CASES")
    print(f"  {'─' * 55}")
    other_ap = con.execute(f"""
        SELECT
            c.std_drug,
            UPPER(c.role_cod) as role,
            count(DISTINCT c.primaryid) as n
        FROM drug_std c
        INNER JOIN (
            SELECT DISTINCT primaryid FROM drug_std
            WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
        ) cob ON c.primaryid = cob.primaryid
        WHERE c.is_comparator = TRUE
        GROUP BY c.std_drug, UPPER(c.role_cod)
        ORDER BY c.std_drug, role
    """).fetchall()

    for drug, role, n in other_ap:
        role_label = {"PS": "Suspect", "SS": "Suspect", "C": "Concomitant",
                      "I": "Interacting"}.get(role, role)
        print(f"  {drug:<25s} {role_label:<15s} {n:>5,}")

    # Drug class grouping
    print(f"\n  CONCOMITANT DRUG CLASSES")
    print(f"  {'─' * 40}")
    class_patterns = {
        "Antidepressants": ["SERTRALINE", "FLUOXETINE", "ESCITALOPRAM", "CITALOPRAM",
                           "VENLAFAXINE", "DULOXETINE", "TRAZODONE", "BUPROPION",
                           "MIRTAZAPINE", "PAROXETINE"],
        "Benzodiazepines": ["LORAZEPAM", "CLONAZEPAM", "DIAZEPAM", "ALPRAZOLAM"],
        "Mood stabilisers": ["LITHIUM", "VALPROATE", "VALPROIC", "LAMOTRIGINE",
                             "CARBAMAZEPINE", "DIVALPROEX"],
        "Antihypertensives": ["LISINOPRIL", "AMLODIPINE", "LOSARTAN", "METOPROLOL",
                              "ATENOLOL", "HYDROCHLOROTHIAZIDE"],
        "Anticholinergics": ["BENZTROPINE", "TRIHEXYPHENIDYL", "DIPHENHYDRAMINE"],
    }

    for class_name, patterns in class_patterns.items():
        n = con.execute(f"""
            SELECT count(DISTINCT c.primaryid)
            FROM drug_std c
            INNER JOIN (
                SELECT DISTINCT primaryid FROM drug_std
                WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
            ) cob ON c.primaryid = cob.primaryid
            WHERE ({' OR '.join(f"UPPER(c.drugname) LIKE '%{p}%'" for p in patterns)})
              AND c.is_cobenfy = FALSE
        """).fetchone()[0]
        pct = 100 * n / n_total if n_total > 0 else 0
        print(f"  {class_name:<25s}: {n:>5,} ({pct:.1f}%)")

    if concom_rows:
        pd.DataFrame(concom_rows).to_csv(SUPP_DIR / "concomitant_medications.csv",
                                         index=False, float_format="%.1f")
    print(f"\n  Saved: supplementary/concomitant_medications.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 70)
    print("  Supplementary Analyses")
    print("  Project 48: Cobenfy Pharmacovigilance")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)

    analysis_1_outcome_severity(con)
    analysis_2_indication_restricted(con)
    analysis_3_time_stratified(con)
    analysis_4_controls(con)
    analysis_5_completeness(con)
    analysis_6_soc_aggregation(con)
    analysis_7_covid_exclusion(con)
    analysis_8_concomitant(con)

    con.close()
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  ALL SUPPLEMENTARY ANALYSES COMPLETE ({elapsed:.0f}s)")
    print(f"  Results in: outputs/supplementary/")
    print(f"{'=' * 70}")

    # List output files
    supp_files = sorted(SUPP_DIR.glob("*.csv"))
    print(f"\n  Output files ({len(supp_files)}):")
    for f in supp_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:50s} {size_kb:>6.1f} KB")


if __name__ == "__main__":
    main()
