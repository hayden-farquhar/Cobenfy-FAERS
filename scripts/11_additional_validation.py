"""
Script 11: Additional validation and strengthening analyses.

Implements eight analyses to strengthen the manuscript for high-impact review:
    1. Sex-stratified disproportionality (ROR by sex for consensus signals)
    2. Age-stratified disproportionality (≥65 vs <65 for anticholinergic concern)
    3. E-value for unmeasured confounding (point estimate + lower CI)
    4. CYP2D6 inhibitor interaction analysis (event profile ± CYP2D6 inhibitors)
    5. Cumulative sequential signal detection (quarter-by-quarter ROR evolution)
    6. Label concordance analysis (detected signals vs FDA prescribing information)
    7. Signal masking analysis (interaction-based masking detection)
    8. Dose-response analysis (high vs low dose if data permits)

Usage:
    python scripts/11_additional_validation.py

Requires: data/processed/faers.duckdb (with drug_std from script 03)
          outputs/tables/disproportionality_cobenfy_full.csv (from script 09)
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
SUPP_DIR = PROJECT_ROOT / "outputs" / "supplementary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

MIN_REPORTS = 3
DRUG = "xanomeline-trospium"

# CYP2D6 inhibitors (strong) — xanomeline is a CYP2D6 substrate
CYP2D6_INHIBITORS = [
    "BUPROPION", "FLUOXETINE", "PAROXETINE", "QUINIDINE",
    "DULOXETINE", "SERTRALINE",
]

# FDA label-listed adverse reactions (from Cobenfy prescribing information)
# Source: packageinserts.bms.com/pi/pi_cobenfy.pdf — Table 3 and Warnings
LABEL_ADRS = {
    # Table 3 — adverse reactions ≥2% and greater than placebo
    "NAUSEA": "Table 3 (24%)",
    "VOMITING": "Table 3 (12%)",
    "DYSPEPSIA": "Table 3 (9%)",
    "CONSTIPATION": "Table 3 (7%)",
    "DIARRHOEA": "Table 3 (4%)",
    "ABDOMINAL PAIN": "Table 3 (4%)",
    "GASTROESOPHAGEAL REFLUX DISEASE": "Table 3 (3%)",
    "GASTROOESOPHAGEAL REFLUX DISEASE": "Table 3 (3%)",
    "HYPERTENSION": "Table 3 (5%)",
    "TACHYCARDIA": "Table 3 (4%)",
    "HEART RATE INCREASED": "Table 3 (3%)",
    "DIZZINESS": "Table 3 (3%)",
    "SOMNOLENCE": "Table 3 (3%)",
    "HEADACHE": "Table 3 (2%)",
    "WEIGHT INCREASED": "Table 3 (2%)",
    "ABDOMINAL DISTENSION": "Table 3 (2%)",
    # Warnings and precautions
    "URINARY RETENTION": "Warning 5.3",
    "ANGIOEDEMA": "Warning 5.4",
    "QT PROLONGATION": "Warning 5.2",
    "SYNCOPE": "Warning 5.5",
    "SEIZURE": "Warning — epilepsy caution",
}


# ── Shared helpers ───────────────────────────────────────────────────────────

def compute_ror_from_2x2(a, b, c, d):
    """Compute ROR with 95% CI from 2x2 counts (with 0.5 continuity correction)."""
    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    ror = (ac * dc) / (bc * cc)
    ln_ror = np.log(ror)
    se = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)
    lower = np.exp(ln_ror - 1.96 * se)
    upper = np.exp(ln_ror + 1.96 * se)
    p = 2 * (1 - stats.norm.cdf(abs(ln_ror / se)))
    return ror, lower, upper, p


def e_value(ror, ci_lower=None):
    """
    Compute E-value (VanderWeele & Ding, 2017).
    E = ROR + sqrt(ROR * (ROR - 1)) for ROR > 1.
    For CI lower bound, same formula applied to the lower CI.
    """
    def _ev(r):
        if r < 1:
            return 1.0
        return r + np.sqrt(r * (r - 1))

    ev_point = _ev(ror)
    ev_ci = _ev(ci_lower) if ci_lower is not None and ci_lower > 1 else 1.0
    return ev_point, ev_ci


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: SEX-STRATIFIED DISPROPORTIONALITY
# ═════════════════════════════════════════════════════════════════════════════

def sex_stratified_analysis(con, consensus_pts):
    """Compute ROR for consensus signals stratified by sex."""
    print("\n  ANALYSIS 1: Sex-Stratified Disproportionality")
    print(f"  {'─' * 60}")

    results = []
    for sex_code, sex_label in [("F", "Female"), ("M", "Male")]:
        N = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE UPPER(d.role_cod) IN ('PS', 'SS')
              AND UPPER(dem.sex) = '{sex_code}'
        """).fetchone()[0]

        n_drug = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) IN ('PS', 'SS')
              AND UPPER(dem.sex) = '{sex_code}'
        """).fetchone()[0]

        print(f"    {sex_label}: N={N:,}, n_drug={n_drug:,}")

        for pt in consensus_pts:
            pt_esc = pt.replace("'", "''")
            row = con.execute(f"""
                WITH drug_cases AS (
                    SELECT DISTINCT d.primaryid
                    FROM drug_std d
                    INNER JOIN demo dem ON d.primaryid = dem.primaryid
                    WHERE d.std_drug = '{DRUG}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND UPPER(dem.sex) = '{sex_code}'
                ),
                all_reactions AS (
                    SELECT DISTINCT r.primaryid
                    FROM reac r
                    INNER JOIN drug_std d ON r.primaryid = d.primaryid
                    INNER JOIN demo dem ON r.primaryid = dem.primaryid
                    WHERE UPPER(r.pt) = '{pt_esc}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND UPPER(dem.sex) = '{sex_code}'
                )
                SELECT
                    (SELECT count(*) FROM drug_cases dc
                     INNER JOIN all_reactions ar ON dc.primaryid = ar.primaryid) as a,
                    (SELECT count(*) FROM all_reactions) as n_reaction
            """).fetchone()

            a = row[0]
            n_reaction = row[1]
            b = n_drug - a
            c = n_reaction - a
            d = N - a - b - c

            if a >= MIN_REPORTS:
                ror, lower, upper, p = compute_ror_from_2x2(a, b, c, d)
                results.append({
                    "pt": pt, "sex": sex_label, "a": a,
                    "n_drug": n_drug, "n_reaction": n_reaction, "N": N,
                    "ror": ror, "ror_lower95": lower, "ror_upper95": upper,
                    "ror_pvalue": p, "signal": lower > 1.0,
                })
            else:
                results.append({
                    "pt": pt, "sex": sex_label, "a": a,
                    "n_drug": n_drug, "n_reaction": n_reaction, "N": N,
                    "ror": np.nan, "ror_lower95": np.nan, "ror_upper95": np.nan,
                    "ror_pvalue": np.nan, "signal": False,
                })

    df = pd.DataFrame(results)
    out = SUPP_DIR / "sex_stratified_disproportionality.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    n_f = df[(df["sex"] == "Female") & (df["signal"])].shape[0]
    n_m = df[(df["sex"] == "Male") & (df["signal"])].shape[0]
    print(f"    Signals: Female={n_f}, Male={n_m}")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: AGE-STRATIFIED DISPROPORTIONALITY
# ═════════════════════════════════════════════════════════════════════════════

def age_stratified_analysis(con, consensus_pts):
    """Compute ROR stratified by age (<65 vs ≥65) for anticholinergic signals."""
    print("\n  ANALYSIS 2: Age-Stratified Disproportionality")
    print(f"  {'─' * 60}")

    results = []
    for age_filter, age_label in [
        ("CAST(dem.age AS FLOAT) < 65", "<65"),
        ("CAST(dem.age AS FLOAT) >= 65", "≥65"),
    ]:
        # Filter to reports with valid numeric age
        age_where = f"""
            dem.age IS NOT NULL
            AND dem.age != ''
            AND TRIM(dem.age) != ''
            AND dem.age_cod = 'YR'
            AND TRY_CAST(dem.age AS FLOAT) IS NOT NULL
            AND {age_filter}
        """

        N = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE UPPER(d.role_cod) IN ('PS', 'SS')
              AND {age_where}
        """).fetchone()[0]

        n_drug = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) IN ('PS', 'SS')
              AND {age_where}
        """).fetchone()[0]

        print(f"    {age_label}: N={N:,}, n_drug={n_drug:,}")

        for pt in consensus_pts:
            pt_esc = pt.replace("'", "''")
            row = con.execute(f"""
                WITH drug_cases AS (
                    SELECT DISTINCT d.primaryid
                    FROM drug_std d
                    INNER JOIN demo dem ON d.primaryid = dem.primaryid
                    WHERE d.std_drug = '{DRUG}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND {age_where}
                ),
                all_reactions AS (
                    SELECT DISTINCT r.primaryid
                    FROM reac r
                    INNER JOIN drug_std d ON r.primaryid = d.primaryid
                    INNER JOIN demo dem ON r.primaryid = dem.primaryid
                    WHERE UPPER(r.pt) = '{pt_esc}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND {age_where}
                )
                SELECT
                    (SELECT count(*) FROM drug_cases dc
                     INNER JOIN all_reactions ar ON dc.primaryid = ar.primaryid) as a,
                    (SELECT count(*) FROM all_reactions) as n_reaction
            """).fetchone()

            a = row[0]
            n_reaction = row[1]
            b = n_drug - a
            c = n_reaction - a
            d_val = N - a - b - c

            if a >= MIN_REPORTS:
                ror, lower, upper, p = compute_ror_from_2x2(a, b, c, d_val)
                results.append({
                    "pt": pt, "age_group": age_label, "a": a,
                    "n_drug": n_drug, "n_reaction": n_reaction, "N": N,
                    "ror": ror, "ror_lower95": lower, "ror_upper95": upper,
                    "ror_pvalue": p, "signal": lower > 1.0,
                })
            else:
                results.append({
                    "pt": pt, "age_group": age_label, "a": a,
                    "n_drug": n_drug, "n_reaction": n_reaction, "N": N,
                    "ror": np.nan, "ror_lower95": np.nan, "ror_upper95": np.nan,
                    "ror_pvalue": np.nan, "signal": False,
                })

    df = pd.DataFrame(results)
    out = SUPP_DIR / "age_stratified_disproportionality.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    n_young = df[(df["age_group"] == "<65") & (df["signal"])].shape[0]
    n_old = df[(df["age_group"] == "≥65") & (df["signal"])].shape[0]
    print(f"    Signals: <65={n_young}, ≥65={n_old}")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: E-VALUE FOR UNMEASURED CONFOUNDING
# ═════════════════════════════════════════════════════════════════════════════

def evalue_analysis(primary_df):
    """Compute E-values for all consensus signals from primary analysis."""
    print("\n  ANALYSIS 3: E-Value for Unmeasured Confounding")
    print(f"  {'─' * 60}")

    consensus = primary_df[primary_df["n_methods_signal"] >= 3].copy()

    evalues = []
    for _, row in consensus.iterrows():
        ror = row["ror"]
        ror_lower = row["ror_lower95"]
        ev_point, ev_ci = e_value(ror, ror_lower)
        evalues.append({
            "pt": row["pt"],
            "a": row["a"],
            "ror": ror,
            "ror_lower95": ror_lower,
            "e_value_point": ev_point,
            "e_value_ci": ev_ci,
        })

    df = pd.DataFrame(evalues).sort_values("e_value_point", ascending=False)

    # Summary
    strong = df[df["e_value_ci"] >= 3.0]
    moderate = df[(df["e_value_ci"] >= 2.0) & (df["e_value_ci"] < 3.0)]
    weak = df[df["e_value_ci"] < 2.0]
    print(f"    Strong (E-value CI ≥ 3.0): {len(strong)} signals")
    print(f"    Moderate (2.0–3.0):         {len(moderate)} signals")
    print(f"    Weak (<2.0):                {len(weak)} signals")

    print(f"\n    {'PT':<35s} {'ROR':>7s} {'E-val':>7s} {'E-CI':>7s}")
    print(f"    {'─' * 60}")
    for _, r in df.head(15).iterrows():
        print(f"    {str(r['pt'])[:34]:<35s} {r['ror']:>7.1f} "
              f"{r['e_value_point']:>7.1f} {r['e_value_ci']:>7.1f}")

    out = SUPP_DIR / "e_values.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: CYP2D6 INHIBITOR INTERACTION
# ═════════════════════════════════════════════════════════════════════════════

def cyp2d6_analysis(con, consensus_pts):
    """Compare event profiles with and without CYP2D6 inhibitor co-medication."""
    print("\n  ANALYSIS 4: CYP2D6 Inhibitor Interaction Analysis")
    print(f"  {'─' * 60}")

    # Identify Cobenfy cases with CYP2D6 inhibitors as concomitant
    inhibitor_patterns = " OR ".join(
        [f"UPPER(d2.drugname) LIKE '%{inh}%'" for inh in CYP2D6_INHIBITORS]
    )

    cyp_cases = con.execute(f"""
        SELECT DISTINCT d1.primaryid
        FROM drug_std d1
        INNER JOIN drug d2 ON d1.primaryid = d2.primaryid
        WHERE d1.std_drug = '{DRUG}'
          AND UPPER(d1.role_cod) IN ('PS', 'SS')
          AND ({inhibitor_patterns})
          AND (d2.primaryid != d1.primaryid OR d2.drug_seq != d1.drug_seq)
    """).fetchdf()

    all_cobenfy = con.execute(f"""
        SELECT DISTINCT primaryid FROM drug_std
        WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS', 'SS')
    """).fetchdf()

    cyp_ids = set(cyp_cases["primaryid"].tolist()) if len(cyp_cases) > 0 else set()
    all_ids = set(all_cobenfy["primaryid"].tolist())
    non_cyp_ids = all_ids - cyp_ids

    print(f"    Cobenfy cases with CYP2D6 inhibitor: {len(cyp_ids)}")
    print(f"    Cobenfy cases without: {len(non_cyp_ids)}")

    if len(cyp_ids) < 5:
        print("    ⚠ Too few CYP2D6 co-prescribed cases for meaningful analysis")
        # Still produce a descriptive output
        if len(cyp_ids) > 0:
            cyp_id_list = ",".join([f"'{x}'" for x in cyp_ids])
            cyp_events = con.execute(f"""
                SELECT UPPER(r.pt) as pt, count(DISTINCT r.primaryid) as n
                FROM reac r
                WHERE r.primaryid IN ({cyp_id_list})
                GROUP BY UPPER(r.pt)
                ORDER BY n DESC
                LIMIT 30
            """).fetchdf()
            out = SUPP_DIR / "cyp2d6_descriptive.csv"
            cyp_events.to_csv(out, index=False)
            print(f"    → {out.name} (descriptive only)")
            return cyp_events

        return pd.DataFrame()

    # Compare event rates between groups
    results = []
    cyp_id_list = ",".join([f"'{x}'" for x in cyp_ids])
    non_cyp_id_list = ",".join([f"'{x}'" for x in non_cyp_ids])

    for pt in consensus_pts:
        pt_esc = pt.replace("'", "''")

        a_cyp = con.execute(f"""
            SELECT count(DISTINCT r.primaryid) FROM reac r
            WHERE r.primaryid IN ({cyp_id_list})
              AND UPPER(r.pt) = '{pt_esc}'
        """).fetchone()[0]

        a_non = con.execute(f"""
            SELECT count(DISTINCT r.primaryid) FROM reac r
            WHERE r.primaryid IN ({non_cyp_id_list})
              AND UPPER(r.pt) = '{pt_esc}'
        """).fetchone()[0]

        pct_cyp = 100 * a_cyp / len(cyp_ids) if len(cyp_ids) > 0 else 0
        pct_non = 100 * a_non / len(non_cyp_ids) if len(non_cyp_ids) > 0 else 0

        # Fisher's exact test
        table = np.array([
            [a_cyp, len(cyp_ids) - a_cyp],
            [a_non, len(non_cyp_ids) - a_non]
        ])
        _, p_fisher = stats.fisher_exact(table) if min(table.shape) > 0 else (np.nan, np.nan)

        results.append({
            "pt": pt,
            "n_cyp2d6": a_cyp, "pct_cyp2d6": pct_cyp,
            "n_no_cyp2d6": a_non, "pct_no_cyp2d6": pct_non,
            "n_cyp2d6_total": len(cyp_ids),
            "n_no_cyp2d6_total": len(non_cyp_ids),
            "fisher_p": p_fisher,
        })

    df = pd.DataFrame(results)
    out = SUPP_DIR / "cyp2d6_interaction.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    sig = df[df["fisher_p"] < 0.05]
    print(f"    Significant differences (p<0.05): {len(sig)}")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 5: CUMULATIVE SEQUENTIAL SIGNAL DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def sequential_signal_detection(con, key_pts):
    """Track ROR evolution quarter by quarter (cumulative)."""
    print("\n  ANALYSIS 5: Cumulative Sequential Signal Detection")
    print(f"  {'─' * 60}")

    # Get available quarters
    quarters = con.execute("""
        SELECT DISTINCT
            SUBSTRING(dem.fda_dt, 1, 4) || 'Q' ||
            CASE
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 3 THEN '1'
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 6 THEN '2'
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 9 THEN '3'
                ELSE '4'
            END as quarter
        FROM demo dem
        INNER JOIN drug_std d ON dem.primaryid = d.primaryid
        WHERE d.std_drug = 'xanomeline-trospium'
          AND UPPER(d.role_cod) IN ('PS', 'SS')
          AND dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6
        ORDER BY quarter
    """).fetchdf()["quarter"].tolist()

    print(f"    Quarters: {', '.join(quarters)}")

    results = []
    for i, cutoff_q in enumerate(quarters):
        # Cumulative: include all quarters up to and including cutoff_q
        included = quarters[:i + 1]
        q_filter = " OR ".join([
            f"""(SUBSTRING(dem.fda_dt, 1, 4) || 'Q' ||
            CASE
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 3 THEN '1'
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 6 THEN '2'
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 9 THEN '3'
                ELSE '4'
            END = '{q}')""" for q in included
        ])

        N = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE UPPER(d.role_cod) IN ('PS', 'SS')
              AND dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6
              AND ({q_filter})
        """).fetchone()[0]

        n_drug = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) IN ('PS', 'SS')
              AND dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6
              AND ({q_filter})
        """).fetchone()[0]

        for pt in key_pts:
            pt_esc = pt.replace("'", "''")
            row = con.execute(f"""
                WITH drug_cases AS (
                    SELECT DISTINCT d.primaryid
                    FROM drug_std d
                    INNER JOIN demo dem ON d.primaryid = dem.primaryid
                    WHERE d.std_drug = '{DRUG}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6
                      AND ({q_filter})
                ),
                all_reactions AS (
                    SELECT DISTINCT r.primaryid
                    FROM reac r
                    INNER JOIN drug_std d ON r.primaryid = d.primaryid
                    INNER JOIN demo dem ON r.primaryid = dem.primaryid
                    WHERE UPPER(r.pt) = '{pt_esc}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND dem.fda_dt IS NOT NULL AND LENGTH(dem.fda_dt) >= 6
                      AND ({q_filter})
                )
                SELECT
                    (SELECT count(*) FROM drug_cases dc
                     INNER JOIN all_reactions ar ON dc.primaryid = ar.primaryid) as a,
                    (SELECT count(*) FROM all_reactions) as n_reaction
            """).fetchone()

            a = row[0]
            n_reaction = row[1]
            b = n_drug - a
            c = n_reaction - a
            d_val = N - a - b - c

            if a >= MIN_REPORTS and d_val > 0:
                ror, lower, upper, p = compute_ror_from_2x2(a, b, c, d_val)
            else:
                ror, lower, upper, p = np.nan, np.nan, np.nan, np.nan

            results.append({
                "pt": pt, "cumulative_through": cutoff_q,
                "n_quarters": i + 1, "a": a, "n_drug": n_drug, "N": N,
                "ror": ror, "ror_lower95": lower, "ror_upper95": upper,
                "signal": lower > 1.0 if not np.isnan(lower) else False,
            })

    df = pd.DataFrame(results)
    out = SUPP_DIR / "sequential_signal_detection.csv"
    df.to_csv(out, index=False, float_format="%.4f")

    # Print summary for top signals
    print(f"\n    {'PT':<30s}", end="")
    for q in quarters:
        print(f" {q:>10s}", end="")
    print()
    print(f"    {'─' * (30 + 11 * len(quarters))}")
    for pt in key_pts[:12]:
        pt_data = df[df["pt"] == pt]
        print(f"    {pt[:29]:<30s}", end="")
        for q in quarters:
            qrow = pt_data[pt_data["cumulative_through"] == q]
            if len(qrow) > 0 and not np.isnan(qrow.iloc[0]["ror"]):
                sig = "*" if qrow.iloc[0]["signal"] else ""
                print(f" {qrow.iloc[0]['ror']:>8.1f}{sig:<1s}", end="")
            else:
                print(f" {'—':>10s}", end="")
        print()

    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 6: LABEL CONCORDANCE
# ═════════════════════════════════════════════════════════════════════════════

def label_concordance_analysis(primary_df):
    """Compare detected FAERS signals against FDA prescribing information."""
    print("\n  ANALYSIS 6: Label Concordance Analysis")
    print(f"  {'─' * 60}")

    consensus = set(
        primary_df[primary_df["n_methods_signal"] >= 3]["pt"].str.upper().tolist()
    )
    all_tested = set(primary_df["pt"].str.upper().tolist())
    label_pts = set(LABEL_ADRS.keys())

    # Categories
    concordant = label_pts & consensus  # on label AND detected
    label_not_detected = label_pts - consensus  # on label but NOT detected
    # Split label_not_detected into: tested but not signal vs not tested
    label_tested_no_signal = label_not_detected & all_tested
    label_not_tested = label_not_detected - all_tested
    novel = consensus - label_pts  # detected but NOT on label

    rows = []
    for pt in sorted(concordant):
        rows.append({"pt": pt, "category": "concordant",
                      "label_section": LABEL_ADRS.get(pt, ""),
                      "signal": True})
    for pt in sorted(label_tested_no_signal):
        rows.append({"pt": pt, "category": "label_only (no signal)",
                      "label_section": LABEL_ADRS.get(pt, ""),
                      "signal": False})
    for pt in sorted(label_not_tested):
        rows.append({"pt": pt, "category": "label_only (not tested, <3 reports)",
                      "label_section": LABEL_ADRS.get(pt, ""),
                      "signal": False})
    for pt in sorted(novel):
        # Get ROR for display
        match = primary_df[primary_df["pt"].str.upper() == pt]
        ror_val = match.iloc[0]["ror"] if len(match) > 0 else np.nan
        rows.append({"pt": pt, "category": "novel (not on label)",
                      "label_section": "",
                      "signal": True, "ror": ror_val})

    df = pd.DataFrame(rows)

    print(f"    Concordant (label + signal):     {len(concordant)}")
    print(f"    Label only (tested, no signal):  {len(label_tested_no_signal)}")
    print(f"    Label only (not tested):         {len(label_not_tested)}")
    print(f"    Novel (signal, not on label):    {len(novel)}")
    sensitivity = len(concordant) / len(label_pts & all_tested) if len(label_pts & all_tested) > 0 else 0
    print(f"    Label sensitivity (tested PTs):  {sensitivity:.1%}")

    print(f"\n    NOVEL SIGNALS (not in FDA label):")
    for pt in sorted(novel):
        match = primary_df[primary_df["pt"].str.upper() == pt]
        if len(match) > 0:
            r = match.iloc[0]
            print(f"      {pt:<40s} ROR={r['ror']:.1f}  n={r['a']:.0f}")

    out = SUPP_DIR / "label_concordance.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 7: SIGNAL MASKING
# ═════════════════════════════════════════════════════════════════════════════

def signal_masking_analysis(con, primary_df):
    """
    Detect signal masking using the interaction-based method.

    Masking occurs when a strongly reported event inflates the denominator
    of disproportionality measures for other events of the same drug.
    We identify this by removing the top event and recomputing RORs.
    """
    print("\n  ANALYSIS 7: Signal Masking Analysis")
    print(f"  {'─' * 60}")

    # Top events by case count that could mask others
    top_events = primary_df.nlargest(5, "a")[["pt", "a"]].values.tolist()
    print(f"    Potential masking events (by volume):")
    for pt, n in top_events:
        print(f"      {pt}: n={n:.0f}")

    # For each top event, remove its reports and recompute ROR for borderline PTs
    borderline = primary_df[
        (primary_df["n_methods_signal"] < 3) &
        (primary_df["ror_lower95"] > 0.5) &
        (primary_df["a"] >= MIN_REPORTS)
    ]["pt"].tolist()

    if not borderline:
        borderline = primary_df[
            primary_df["n_methods_signal"] < 3
        ]["pt"].head(20).tolist()

    results = []
    for mask_pt, mask_n in top_events:
        mask_pt_esc = mask_pt.replace("'", "''")

        # Get primaryids that reported the masking event for Cobenfy
        mask_ids = con.execute(f"""
            SELECT DISTINCT r.primaryid
            FROM reac r
            INNER JOIN drug_std d ON r.primaryid = d.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) IN ('PS', 'SS')
              AND UPPER(r.pt) = '{mask_pt_esc}'
        """).fetchdf()["primaryid"].tolist()

        if not mask_ids:
            continue

        excl = ",".join([f"'{x}'" for x in mask_ids])

        # Recompute for borderline PTs excluding masking-event reports
        N_adj = con.execute(f"""
            SELECT count(DISTINCT primaryid) FROM drug_std
            WHERE UPPER(role_cod) IN ('PS', 'SS')
              AND primaryid NOT IN ({excl})
        """).fetchone()[0]

        n_drug_adj = con.execute(f"""
            SELECT count(DISTINCT primaryid) FROM drug_std
            WHERE std_drug = '{DRUG}'
              AND UPPER(role_cod) IN ('PS', 'SS')
              AND primaryid NOT IN ({excl})
        """).fetchone()[0]

        for pt in borderline:
            pt_esc = pt.replace("'", "''")
            row = con.execute(f"""
                WITH drug_cases AS (
                    SELECT DISTINCT d.primaryid FROM drug_std d
                    WHERE d.std_drug = '{DRUG}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND d.primaryid NOT IN ({excl})
                ),
                all_reactions AS (
                    SELECT DISTINCT r.primaryid FROM reac r
                    INNER JOIN drug_std d ON r.primaryid = d.primaryid
                    WHERE UPPER(r.pt) = '{pt_esc}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND r.primaryid NOT IN ({excl})
                )
                SELECT
                    (SELECT count(*) FROM drug_cases dc
                     INNER JOIN all_reactions ar ON dc.primaryid = ar.primaryid) as a,
                    (SELECT count(*) FROM all_reactions) as n_reaction
            """).fetchone()

            a = row[0]
            n_reaction = row[1]
            b = n_drug_adj - a
            c = n_reaction - a
            d_val = N_adj - a - b - c

            if a >= MIN_REPORTS and d_val > 0:
                ror_adj, lower_adj, upper_adj, _ = compute_ror_from_2x2(a, b, c, d_val)
            else:
                ror_adj, lower_adj, upper_adj = np.nan, np.nan, np.nan

            # Get original ROR
            orig = primary_df[primary_df["pt"].str.upper() == pt.upper()]
            ror_orig = orig.iloc[0]["ror"] if len(orig) > 0 else np.nan

            results.append({
                "masked_by": mask_pt,
                "pt": pt,
                "a_original": orig.iloc[0]["a"] if len(orig) > 0 else np.nan,
                "a_adjusted": a,
                "ror_original": ror_orig,
                "ror_adjusted": ror_adj,
                "ror_lower95_adjusted": lower_adj,
                "ror_change_pct": 100 * (ror_adj - ror_orig) / ror_orig if (
                    not np.isnan(ror_adj) and not np.isnan(ror_orig) and ror_orig > 0
                ) else np.nan,
                "unmasked": (lower_adj > 1.0) if not np.isnan(lower_adj) else False,
            })

    df = pd.DataFrame(results)

    # Report unmasked signals
    unmasked = df[df["unmasked"] & (df["ror_change_pct"] > 10)]
    if len(unmasked) > 0:
        print(f"\n    POTENTIALLY MASKED SIGNALS (ROR increased >10% after removal):")
        for _, r in unmasked.drop_duplicates("pt").iterrows():
            print(f"      {r['pt']:<35s} ROR: {r['ror_original']:.2f} → "
                  f"{r['ror_adjusted']:.2f} (excl. {r['masked_by']})")
    else:
        print(f"\n    No meaningful masking detected.")

    out = SUPP_DIR / "signal_masking.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 8: DOSE-RESPONSE
# ═════════════════════════════════════════════════════════════════════════════

def dose_response_analysis(con, consensus_pts):
    """Analyse dose information if available in FAERS drug table."""
    print("\n  ANALYSIS 8: Dose-Response Analysis")
    print(f"  {'─' * 60}")

    # Check dose data availability for Cobenfy
    dose_info = con.execute(f"""
        SELECT
            dose_amt, dose_unit, dose_form, dose_freq,
            count(DISTINCT primaryid) as n
        FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND UPPER(role_cod) IN ('PS', 'SS')
          AND dose_amt IS NOT NULL
          AND TRIM(CAST(dose_amt AS VARCHAR)) != ''
          AND TRIM(CAST(dose_amt AS VARCHAR)) != '0'
        GROUP BY dose_amt, dose_unit, dose_form, dose_freq
        ORDER BY n DESC
        LIMIT 20
    """).fetchdf()

    total_cobenfy = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS', 'SS')
    """).fetchone()[0]

    dose_available = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS', 'SS')
          AND dose_amt IS NOT NULL
          AND TRIM(CAST(dose_amt AS VARCHAR)) != ''
          AND TRIM(CAST(dose_amt AS VARCHAR)) != '0'
    """).fetchone()[0]

    print(f"    Total Cobenfy cases: {total_cobenfy}")
    print(f"    With dose info: {dose_available} ({100*dose_available/total_cobenfy:.1f}%)")

    if len(dose_info) > 0:
        print(f"\n    Dose distribution:")
        for _, r in dose_info.iterrows():
            print(f"      {r['dose_amt']} {r['dose_unit'] or ''} "
                  f"{r['dose_form'] or ''} {r['dose_freq'] or ''}: n={r['n']}")

    if dose_available < 20:
        print("    ⚠ Insufficient dose data for stratified analysis")
        out = SUPP_DIR / "dose_response.csv"
        dose_info.to_csv(out, index=False)
        print(f"    → {out.name} (descriptive only)")
        return dose_info

    # Try to classify into dose groups (Cobenfy is 50/20mg BID or 125/30mg BID)
    # Look for numeric dose values
    dose_cases = con.execute(f"""
        SELECT DISTINCT
            primaryid,
            TRY_CAST(dose_amt AS FLOAT) as dose_num,
            dose_unit
        FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND UPPER(role_cod) IN ('PS', 'SS')
          AND dose_amt IS NOT NULL
          AND TRY_CAST(dose_amt AS FLOAT) IS NOT NULL
          AND TRY_CAST(dose_amt AS FLOAT) > 0
    """).fetchdf()

    if len(dose_cases) < 20:
        print("    ⚠ Insufficient parseable dose data")
        out = SUPP_DIR / "dose_response.csv"
        dose_info.to_csv(out, index=False)
        return dose_info

    # Classify: low dose (≤100mg total, i.e. 50/20 BID) vs high dose (>100mg)
    # Cobenfy doses: 50mg xanomeline / 20mg trospium OR 125mg / 30mg
    median_dose = dose_cases["dose_num"].median()
    dose_cases["dose_group"] = np.where(
        dose_cases["dose_num"] <= median_dose, "low", "high"
    )

    low_ids = set(dose_cases[dose_cases["dose_group"] == "low"]["primaryid"].tolist())
    high_ids = set(dose_cases[dose_cases["dose_group"] == "high"]["primaryid"].tolist())

    print(f"    Dose groups (split at median {median_dose}): "
          f"low={len(low_ids)}, high={len(high_ids)}")

    results = []
    for pt in consensus_pts[:20]:
        pt_esc = pt.replace("'", "''")
        for group, ids in [("low", low_ids), ("high", high_ids)]:
            if not ids:
                continue
            id_list = ",".join([f"'{x}'" for x in ids])
            a = con.execute(f"""
                SELECT count(DISTINCT r.primaryid)
                FROM reac r
                WHERE r.primaryid IN ({id_list})
                  AND UPPER(r.pt) = '{pt_esc}'
            """).fetchone()[0]

            results.append({
                "pt": pt, "dose_group": group,
                "n_events": a, "n_total": len(ids),
                "pct": 100 * a / len(ids) if len(ids) > 0 else 0,
            })

    df = pd.DataFrame(results)

    # Pivot for comparison
    if len(df) > 0:
        pivot = df.pivot_table(
            index="pt", columns="dose_group",
            values=["n_events", "pct"], aggfunc="first"
        )
        # Fisher exact for each PT
        for pt in df["pt"].unique():
            pt_data = df[df["pt"] == pt]
            low_row = pt_data[pt_data["dose_group"] == "low"]
            high_row = pt_data[pt_data["dose_group"] == "high"]
            if len(low_row) > 0 and len(high_row) > 0:
                table = np.array([
                    [low_row.iloc[0]["n_events"],
                     low_row.iloc[0]["n_total"] - low_row.iloc[0]["n_events"]],
                    [high_row.iloc[0]["n_events"],
                     high_row.iloc[0]["n_total"] - high_row.iloc[0]["n_events"]],
                ])
                _, p = stats.fisher_exact(table)
                df.loc[df["pt"] == pt, "fisher_p"] = p

    out = SUPP_DIR / "dose_response.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 70)
    print("  Additional Validation & Strengthening Analyses")
    print("  Sex · Age · E-value · CYP2D6 · Sequential · Label · Masking · Dose")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    # Load primary results
    primary_path = OUTPUT_DIR / "disproportionality_cobenfy_full.csv"
    if not primary_path.exists():
        print(f"\n  ERROR: Primary results not found: {primary_path}")
        print("  Run scripts 05 and 09 first.")
        return

    primary_df = pd.read_csv(primary_path)
    consensus_pts = primary_df[
        primary_df["n_methods_signal"] >= 3
    ].sort_values("ror", ascending=False)["pt"].tolist()

    # Key PTs for sequential analysis (top 12 by clinical relevance)
    key_pts = [
        "NAUSEA", "VOMITING", "CONSTIPATION", "URINARY RETENTION",
        "DYSPEPSIA", "DRY MOUTH", "TACHYCARDIA", "DROOLING",
        "HYPERHIDROSIS", "VISION BLURRED", "AKATHISIA", "TREMOR",
    ]

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ── Run all 8 analyses ─────────────────────────────────────────────────
    sex_df = sex_stratified_analysis(con, consensus_pts)
    age_df = age_stratified_analysis(con, consensus_pts)
    ev_df = evalue_analysis(primary_df)
    cyp_df = cyp2d6_analysis(con, consensus_pts)
    seq_df = sequential_signal_detection(con, key_pts)
    label_df = label_concordance_analysis(primary_df)
    mask_df = signal_masking_analysis(con, primary_df)
    dose_df = dose_response_analysis(con, consensus_pts)

    con.close()
    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  ALL 8 VALIDATION ANALYSES COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Outputs in: {SUPP_DIR}")
    print(f"    1. sex_stratified_disproportionality.csv")
    print(f"    2. age_stratified_disproportionality.csv")
    print(f"    3. e_values.csv")
    print(f"    4. cyp2d6_interaction.csv or cyp2d6_descriptive.csv")
    print(f"    5. sequential_signal_detection.csv")
    print(f"    6. label_concordance.csv")
    print(f"    7. signal_masking.csv")
    print(f"    8. dose_response.csv")
    print(f"\n  Ready for manuscript drafting.")


if __name__ == "__main__":
    main()
