"""
Script 12: Strengthening analyses for peer-review robustness.

Implements five analyses addressing anticipated reviewer concerns:
    1. Serious-outcomes-only sensitivity (death, hospitalisation, life-threatening, disability)
    2. Polypharmacy sensitivity (exclude concomitant antipsychotic cases)
    3. Disease-manifestation flagging in label concordance
    4. Reporter-stratified ROR comparison (consumer vs HCP)
    5. Age-sex adjusted active-comparator analysis

Usage:
    python scripts/12_strengthening.py

Requires: data/processed/faers.duckdb (with drug_std from script 03)
          outputs/tables/disproportionality_cobenfy_full.csv (from script 09)
          outputs/tables/active_comparator_results.csv (from script 06)
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

# Concomitant antipsychotics to exclude in polypharmacy sensitivity
CONCOMITANT_ANTIPSYCHOTICS = [
    "CLOZAPINE", "OLANZAPINE", "ARIPIPRAZOLE", "RISPERIDONE",
    "QUETIAPINE", "HALOPERIDOL", "PALIPERIDONE", "LURASIDONE",
    "BREXPIPRAZOLE", "CARIPRAZINE", "LUMATEPERONE", "PIMOZIDE",
    "ZIPRASIDONE", "CHLORPROMAZINE", "FLUPHENAZINE", "PERPHENAZINE",
    "THIORIDAZINE", "THIOTHIXENE", "TRIFLUOPERAZINE", "LOXAPINE",
    "MOLINDONE", "ASENAPINE", "ILOPERIDONE",
    # Include palmitate/decanoate formulations
    "PALIPERIDONE PALMITATE", "ARIPIPRAZOLE LAUROXIL",
    "RISPERIDONE.*LONG", "HALOPERIDOL DECANOATE",
]

# Psychiatric PTs that are disease manifestations of schizophrenia, not drug effects
DISEASE_MANIFESTATION_PTS = {
    "HALLUCINATION": "Core positive symptom of schizophrenia",
    "HALLUCINATION, AUDITORY": "Core positive symptom of schizophrenia",
    "HALLUCINATION, VISUAL": "Positive symptom of schizophrenia",
    "PSYCHOTIC DISORDER": "Primary diagnosis / disease worsening",
    "PARANOIA": "Core positive symptom of schizophrenia",
    "DELUSION": "Core positive symptom of schizophrenia",
    "AGGRESSION": "Behavioural symptom of schizophrenia",
    "AGITATION": "Behavioural symptom of schizophrenia",
    "ABNORMAL BEHAVIOUR": "Behavioural symptom of schizophrenia",
    "ANGER": "Behavioural symptom of schizophrenia",
    "VIOLENCE-RELATED SYMPTOM": "Behavioural symptom of schizophrenia",
    "SCHIZOPHRENIA": "Primary diagnosis itself",
    "MANIA": "Mood episode / schizoaffective overlap",
    "NEGATIVE THOUGHTS": "Negative/cognitive symptom",
    "INTRUSIVE THOUGHTS": "Cognitive symptom overlap",
    "TACHYPHRENIA": "Thought acceleration (psychotic feature)",
    "THINKING ABNORMAL": "Cognitive symptom of schizophrenia",
    "DISORGANISED SPEECH": "Core disorganisation symptom",
    "FLAT AFFECT": "Negative symptom of schizophrenia",
    "SOCIAL AVOIDANT BEHAVIOUR": "Negative symptom of schizophrenia",
    "SUICIDAL IDEATION": "Common comorbid symptom",
    "COMPLETED SUICIDE": "Common comorbid outcome",
    "INTENTIONAL SELF-INJURY": "Common comorbid behaviour",
    "SELF-INJURIOUS BEHAVIOUR": "Common comorbid behaviour",
    "AMPHETAMINES POSITIVE": "Substance use comorbidity marker",
}

# Active comparator PTs (from script 06)
COMPARATORS = ["olanzapine", "risperidone", "aripiprazole",
               "quetiapine", "lurasidone", "brexpiprazole"]

AC_PTS = [
    "NAUSEA", "VOMITING", "CONSTIPATION", "DYSPEPSIA", "DIARRHOEA",
    "TACHYCARDIA", "HYPERTENSION", "BLOOD PRESSURE INCREASED",
    "WEIGHT INCREASED", "HYPERGLYCAEMIA", "DIABETES MELLITUS",
    "METABOLIC SYNDROME", "BLOOD GLUCOSE INCREASED", "DYSLIPIDAEMIA",
    "SOMNOLENCE", "SEDATION", "AKATHISIA", "DYSTONIA", "TREMOR",
    "EXTRAPYRAMIDAL DISORDER", "TARDIVE DYSKINESIA", "PARKINSONISM",
    "HYPERPROLACTINAEMIA", "GALACTORRHOEA", "AMENORRHOEA",
    "DRY MOUTH", "URINARY RETENTION", "VISION BLURRED",
    "DROOLING", "SALIVARY HYPERSECRETION", "HYPERHIDROSIS",
    "QT PROLONGATION",
]


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


def run_disproportionality_on_subset(con, drug_pids_sql, ref_pids_sql, pts,
                                     min_reports=MIN_REPORTS):
    """Run ROR for a drug subset vs a reference subset."""
    N = con.execute(f"SELECT count(DISTINCT primaryid) FROM ({ref_pids_sql})").fetchone()[0]
    n_drug = con.execute(f"SELECT count(DISTINCT primaryid) FROM ({drug_pids_sql})").fetchone()[0]

    if N == 0 or n_drug == 0:
        return pd.DataFrame()

    results = []
    for pt in pts:
        pt_esc = pt.replace("'", "''")
        row = con.execute(f"""
            WITH drug_cases AS ({drug_pids_sql}),
            all_reactions AS (
                SELECT DISTINCT r.primaryid
                FROM reac r
                WHERE r.primaryid IN (SELECT primaryid FROM ({ref_pids_sql}))
                  AND UPPER(r.pt) = '{pt_esc}'
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

        if a >= min_reports and d_val > 0:
            ror, lower, upper, p = compute_ror_from_2x2(a, b, c, d_val)
            results.append({
                "pt": pt, "a": a, "n_drug": n_drug,
                "n_reaction": n_reaction, "N": N,
                "ror": ror, "ror_lower95": lower, "ror_upper95": upper,
                "ror_pvalue": p, "signal": lower > 1.0,
            })
        else:
            results.append({
                "pt": pt, "a": a, "n_drug": n_drug,
                "n_reaction": n_reaction, "N": N,
                "ror": np.nan, "ror_lower95": np.nan, "ror_upper95": np.nan,
                "ror_pvalue": np.nan, "signal": False,
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: SERIOUS-OUTCOMES-ONLY SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════

def serious_outcomes_sensitivity(con, consensus_pts):
    """Disproportionality restricted to cases with serious outcomes."""
    print("\n  ANALYSIS 1: Serious-Outcomes-Only Sensitivity")
    print(f"  {'─' * 60}")

    # FAERS outc_cod values for serious outcomes
    # 'DE' = death, 'LT' = life-threatening, 'HO' = hospitalisation,
    # 'DS' = disability, 'CA' = congenital anomaly, 'RI' = required intervention
    serious_codes = "('DE', 'LT', 'HO', 'DS', 'CA', 'RI')"

    # Check if outc table exists
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "outc" not in tables:
        print("    ⚠ OUTC table not found — checking for outcome data in demo")
        # Try occp_cod in demo table or skip
        print("    Skipping serious-outcomes analysis (no outcome table)")
        return pd.DataFrame()

    # Reference: all cases with serious outcomes
    ref_sql = f"""
        SELECT DISTINCT d.primaryid
        FROM drug_std d
        INNER JOIN outc o ON d.primaryid = o.primaryid
        WHERE UPPER(d.role_cod) IN ('PS', 'SS')
          AND UPPER(o.outc_cod) IN {serious_codes}
    """

    # Drug: Cobenfy cases with serious outcomes
    drug_sql = f"""
        SELECT DISTINCT d.primaryid
        FROM drug_std d
        INNER JOIN outc o ON d.primaryid = o.primaryid
        WHERE d.std_drug = '{DRUG}'
          AND UPPER(d.role_cod) IN ('PS', 'SS')
          AND UPPER(o.outc_cod) IN {serious_codes}
    """

    n_serious_drug = con.execute(
        f"SELECT count(DISTINCT primaryid) FROM ({drug_sql})"
    ).fetchone()[0]
    n_serious_total = con.execute(
        f"SELECT count(DISTINCT primaryid) FROM ({ref_sql})"
    ).fetchone()[0]

    print(f"    Serious Cobenfy cases: {n_serious_drug:,}")
    print(f"    Serious total cases: {n_serious_total:,}")

    if n_serious_drug < 10:
        print("    ⚠ Too few serious Cobenfy cases for reliable analysis")

    df = run_disproportionality_on_subset(con, drug_sql, ref_sql, consensus_pts)

    if len(df) > 0:
        n_sig = df["signal"].sum()
        print(f"    Signals retained: {n_sig}/{len(consensus_pts)} "
              f"({100*n_sig/len(consensus_pts):.0f}%)")

        # Compare with primary
        print(f"\n    {'PT':<35s} {'ROR (serious)':>15s} {'Signal':>8s}")
        print(f"    {'─' * 60}")
        for _, r in df[df["a"] >= MIN_REPORTS].head(20).iterrows():
            sig = "✓" if r["signal"] else "✗"
            print(f"    {str(r['pt'])[:34]:<35s} "
                  f"{r['ror']:>7.1f} ({r['ror_lower95']:.1f}–{r['ror_upper95']:.1f})"
                  f" {sig:>4s}")

    out = SUPP_DIR / "sensitivity_serious_only.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: POLYPHARMACY SENSITIVITY (EXCLUDE CONCOMITANT ANTIPSYCHOTICS)
# ═════════════════════════════════════════════════════════════════════════════

def polypharmacy_sensitivity(con, consensus_pts):
    """Exclude Cobenfy cases co-prescribed with other antipsychotics."""
    print("\n  ANALYSIS 2: Polypharmacy Sensitivity (Exclude Concomitant Antipsychotics)")
    print(f"  {'─' * 60}")

    # Build LIKE patterns for antipsychotics
    ap_patterns = " OR ".join(
        [f"UPPER(d2.drugname) LIKE '%{ap}%'" for ap in CONCOMITANT_ANTIPSYCHOTICS]
    )

    # Identify Cobenfy cases WITH concomitant antipsychotics
    cobenfy_with_ap = con.execute(f"""
        SELECT DISTINCT d1.primaryid
        FROM drug_std d1
        INNER JOIN drug d2 ON d1.primaryid = d2.primaryid
        WHERE d1.std_drug = '{DRUG}'
          AND UPPER(d1.role_cod) IN ('PS', 'SS')
          AND ({ap_patterns})
          AND d2.drug_seq != d1.drug_seq
    """).fetchdf()

    all_cobenfy = con.execute(f"""
        SELECT DISTINCT primaryid FROM drug_std
        WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS', 'SS')
    """).fetchdf()

    ap_ids = set(cobenfy_with_ap["primaryid"].tolist()) if len(cobenfy_with_ap) > 0 else set()
    all_ids = set(all_cobenfy["primaryid"].tolist())
    mono_ids = all_ids - ap_ids

    print(f"    Total Cobenfy cases: {len(all_ids):,}")
    print(f"    With concomitant antipsychotic: {len(ap_ids):,} ({100*len(ap_ids)/len(all_ids):.1f}%)")
    print(f"    Monotherapy (no concomitant AP): {len(mono_ids):,} ({100*len(mono_ids)/len(all_ids):.1f}%)")

    if len(mono_ids) < 50:
        print("    ⚠ Too few monotherapy cases for reliable analysis")

    # Build SQL for monotherapy subset
    mono_id_list = ",".join([f"'{x}'" for x in mono_ids])
    drug_sql = f"SELECT DISTINCT primaryid FROM drug_std WHERE primaryid IN ({mono_id_list})"
    ref_sql = """
        SELECT DISTINCT primaryid FROM drug_std
        WHERE UPPER(role_cod) IN ('PS', 'SS')
    """

    df = run_disproportionality_on_subset(con, drug_sql, ref_sql, consensus_pts)

    if len(df) > 0:
        # Load primary results for comparison
        primary_path = OUTPUT_DIR / "disproportionality_cobenfy_full.csv"
        primary_df = pd.read_csv(primary_path) if primary_path.exists() else pd.DataFrame()

        n_sig = df["signal"].sum()
        print(f"    Signals retained: {n_sig}/{len(consensus_pts)}")

        # Key comparisons for drooling and sialorrhea (clozapine-confounded)
        key_check = ["DROOLING", "SALIVARY HYPERSECRETION", "URINARY RETENTION",
                     "NAUSEA", "VOMITING"]
        print(f"\n    {'PT':<30s} {'Primary ROR':>13s} {'Mono ROR':>13s} {'Change':>8s}")
        print(f"    {'─' * 68}")
        for pt in key_check:
            mono_row = df[df["pt"] == pt]
            prim_row = primary_df[primary_df["pt"] == pt] if len(primary_df) > 0 else pd.DataFrame()
            if len(mono_row) > 0 and not np.isnan(mono_row.iloc[0]["ror"]):
                ror_mono = mono_row.iloc[0]["ror"]
                ror_prim = prim_row.iloc[0]["ror"] if len(prim_row) > 0 else np.nan
                change = f"{100*(ror_mono-ror_prim)/ror_prim:+.1f}%" if not np.isnan(ror_prim) else "—"
                print(f"    {pt:<30s} {ror_prim:>13.1f} {ror_mono:>13.1f} {change:>8s}")

    out = SUPP_DIR / "sensitivity_polypharmacy_monotherapy.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")

    # Also output the list of excluded cases with their concomitant APs
    if len(ap_ids) > 0:
        ap_id_list = ",".join([f"'{x}'" for x in ap_ids])
        ap_detail = con.execute(f"""
            SELECT DISTINCT d1.primaryid, UPPER(d2.drugname) as concomitant_ap
            FROM drug_std d1
            INNER JOIN drug d2 ON d1.primaryid = d2.primaryid
            WHERE d1.std_drug = '{DRUG}'
              AND d1.primaryid IN ({ap_id_list})
              AND ({ap_patterns})
              AND d2.drug_seq != d1.drug_seq
        """).fetchdf()
        ap_summary = ap_detail.groupby("concomitant_ap").size().reset_index(name="n_cases")
        ap_summary = ap_summary.sort_values("n_cases", ascending=False)
        ap_out = SUPP_DIR / "polypharmacy_concomitant_ap_detail.csv"
        ap_summary.to_csv(ap_out, index=False)
        print(f"\n    Top concomitant antipsychotics:")
        for _, r in ap_summary.head(10).iterrows():
            print(f"      {r['concomitant_ap']}: n={r['n_cases']}")
        print(f"    → {ap_out.name}")

    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: DISEASE-MANIFESTATION FLAGGING
# ═════════════════════════════════════════════════════════════════════════════

def disease_manifestation_flagging(primary_df):
    """Re-annotate label concordance with disease-manifestation flags."""
    print("\n  ANALYSIS 3: Disease-Manifestation Flagging of Signals")
    print(f"  {'─' * 60}")

    consensus = primary_df[primary_df["n_methods_signal"] >= 3].copy()

    # Classify each signal
    rows = []
    for _, r in consensus.iterrows():
        pt_upper = r["pt"].upper() if isinstance(r["pt"], str) else str(r["pt"]).upper()
        is_disease = pt_upper in DISEASE_MANIFESTATION_PTS
        reason = DISEASE_MANIFESTATION_PTS.get(pt_upper, "")

        rows.append({
            "pt": r["pt"],
            "a": r["a"],
            "ror": r["ror"],
            "ror_lower95": r["ror_lower95"],
            "ror_upper95": r["ror_upper95"],
            "n_methods_signal": r["n_methods_signal"],
            "disease_manifestation": is_disease,
            "classification": "Disease manifestation" if is_disease else "Pharmacological signal",
            "disease_reason": reason,
        })

    df = pd.DataFrame(rows)

    n_disease = df["disease_manifestation"].sum()
    n_pharma = len(df) - n_disease
    print(f"    Total consensus signals: {len(df)}")
    print(f"    Pharmacological signals: {n_pharma}")
    print(f"    Disease manifestations:  {n_disease}")

    print(f"\n    PHARMACOLOGICAL SIGNALS (true drug effects):")
    pharma = df[~df["disease_manifestation"]].sort_values("ror", ascending=False)
    for _, r in pharma.iterrows():
        print(f"      {str(r['pt']):<35s} ROR={r['ror']:.1f} (n={r['a']:.0f})")

    print(f"\n    DISEASE MANIFESTATIONS (confounding by indication):")
    disease = df[df["disease_manifestation"]].sort_values("ror", ascending=False)
    for _, r in disease.iterrows():
        print(f"      {str(r['pt']):<35s} ROR={r['ror']:.1f} — {r['disease_reason']}")

    out = SUPP_DIR / "signal_classification_disease_vs_drug.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: REPORTER-STRATIFIED ROR COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def reporter_stratified_analysis(con, consensus_pts):
    """Compare ROR between consumer and HCP reporters."""
    print("\n  ANALYSIS 4: Reporter-Stratified ROR Comparison")
    print(f"  {'─' * 60}")

    # Reporter types in FAERS: MD, PH (pharmacist), OT (other HCP), HP (health professional),
    # CN (consumer), LW (lawyer)
    reporter_groups = {
        "HCP": "UPPER(dem.occp_cod) IN ('MD', 'PH', 'OT', 'HP', 'DN', 'RN')",
        "Consumer": "UPPER(dem.occp_cod) IN ('CN', 'LW') OR dem.occp_cod IS NULL OR TRIM(dem.occp_cod) = ''",
    }

    results = []
    for group_name, group_filter in reporter_groups.items():
        N = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE UPPER(d.role_cod) IN ('PS', 'SS')
              AND ({group_filter})
        """).fetchone()[0]

        n_drug = con.execute(f"""
            SELECT count(DISTINCT d.primaryid)
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) IN ('PS', 'SS')
              AND ({group_filter})
        """).fetchone()[0]

        print(f"    {group_name}: N={N:,}, n_drug={n_drug:,}")

        for pt in consensus_pts:
            pt_esc = pt.replace("'", "''")
            row = con.execute(f"""
                WITH drug_cases AS (
                    SELECT DISTINCT d.primaryid
                    FROM drug_std d
                    INNER JOIN demo dem ON d.primaryid = dem.primaryid
                    WHERE d.std_drug = '{DRUG}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND ({group_filter})
                ),
                all_reactions AS (
                    SELECT DISTINCT r.primaryid
                    FROM reac r
                    INNER JOIN drug_std d ON r.primaryid = d.primaryid
                    INNER JOIN demo dem ON r.primaryid = dem.primaryid
                    WHERE UPPER(r.pt) = '{pt_esc}'
                      AND UPPER(d.role_cod) IN ('PS', 'SS')
                      AND ({group_filter})
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
                results.append({
                    "pt": pt, "reporter_group": group_name, "a": a,
                    "n_drug": n_drug, "n_reaction": n_reaction, "N": N,
                    "ror": ror, "ror_lower95": lower, "ror_upper95": upper,
                    "ror_pvalue": p, "signal": lower > 1.0,
                })
            else:
                results.append({
                    "pt": pt, "reporter_group": group_name, "a": a,
                    "n_drug": n_drug, "n_reaction": n_reaction, "N": N,
                    "ror": np.nan, "ror_lower95": np.nan, "ror_upper95": np.nan,
                    "ror_pvalue": np.nan, "signal": False,
                })

    df = pd.DataFrame(results)

    # Compare HCP vs consumer RORs
    hcp = df[df["reporter_group"] == "HCP"].set_index("pt")
    con_df = df[df["reporter_group"] == "Consumer"].set_index("pt")
    common_pts = set(hcp.index) & set(con_df.index)

    print(f"\n    {'PT':<30s} {'HCP ROR':>10s} {'Consumer ROR':>14s} {'Ratio':>8s}")
    print(f"    {'─' * 65}")
    for pt in sorted(common_pts):
        h_ror = hcp.loc[pt, "ror"] if pt in hcp.index and not np.isnan(hcp.loc[pt, "ror"]) else np.nan
        c_ror = con_df.loc[pt, "ror"] if pt in con_df.index and not np.isnan(con_df.loc[pt, "ror"]) else np.nan
        if not np.isnan(h_ror) and not np.isnan(c_ror) and c_ror > 0:
            ratio = h_ror / c_ror
            print(f"    {pt[:29]:<30s} {h_ror:>10.1f} {c_ror:>14.1f} {ratio:>8.2f}")

    out = SUPP_DIR / "reporter_stratified_ror.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 5: AGE-SEX ADJUSTED ACTIVE COMPARATOR
# ═════════════════════════════════════════════════════════════════════════════

def age_sex_adjusted_comparator(con):
    """Age-sex stratified (Mantel-Haenszel) active-comparator ROR."""
    print("\n  ANALYSIS 5: Age-Sex Adjusted Active-Comparator Analysis")
    print(f"  {'─' * 60}")

    # Define strata: age (<40, 40-64, ≥65) × sex (M, F)
    strata = []
    for age_label, age_filter in [
        ("<40", "TRY_CAST(dem.age AS FLOAT) < 40"),
        ("40–64", "TRY_CAST(dem.age AS FLOAT) >= 40 AND TRY_CAST(dem.age AS FLOAT) < 65"),
        ("≥65", "TRY_CAST(dem.age AS FLOAT) >= 65"),
    ]:
        for sex_label, sex_filter in [("M", "UPPER(dem.sex) = 'M'"), ("F", "UPPER(dem.sex) = 'F'")]:
            combined = (f"dem.age IS NOT NULL AND dem.age != '' AND dem.age_cod = 'YR' "
                        f"AND TRY_CAST(dem.age AS FLOAT) IS NOT NULL "
                        f"AND {age_filter} AND {sex_filter}")
            strata.append((f"{age_label}_{sex_label}", combined))

    results = []
    for comparator in COMPARATORS:
        print(f"\n    Comparator: {comparator}")

        for pt in AC_PTS:
            pt_esc = pt.replace("'", "''")

            # Mantel-Haenszel pooled ROR across strata
            mh_num = 0.0  # Σ (a_i * d_i / T_i)
            mh_den = 0.0  # Σ (b_i * c_i / T_i)
            var_ln_mh = 0.0
            total_a = 0
            n_strata_used = 0

            for stratum_name, stratum_filter in strata:
                # 2x2 within stratum: drug (Cobenfy) vs comparator
                row = con.execute(f"""
                    WITH cobenfy_cases AS (
                        SELECT DISTINCT d.primaryid
                        FROM drug_std d
                        INNER JOIN demo dem ON d.primaryid = dem.primaryid
                        WHERE d.std_drug = '{DRUG}'
                          AND UPPER(d.role_cod) IN ('PS', 'SS')
                          AND {stratum_filter}
                    ),
                    comp_cases AS (
                        SELECT DISTINCT d.primaryid
                        FROM drug_std d
                        INNER JOIN demo dem ON d.primaryid = dem.primaryid
                        WHERE d.std_drug = '{comparator}'
                          AND UPPER(d.role_cod) IN ('PS', 'SS')
                          AND {stratum_filter}
                    ),
                    cobenfy_event AS (
                        SELECT count(DISTINCT cc.primaryid) as n
                        FROM cobenfy_cases cc
                        INNER JOIN reac r ON cc.primaryid = r.primaryid
                        WHERE UPPER(r.pt) = '{pt_esc}'
                    ),
                    comp_event AS (
                        SELECT count(DISTINCT cc.primaryid) as n
                        FROM comp_cases cc
                        INNER JOIN reac r ON cc.primaryid = r.primaryid
                        WHERE UPPER(r.pt) = '{pt_esc}'
                    ),
                    cobenfy_total AS (SELECT count(*) as n FROM cobenfy_cases),
                    comp_total AS (SELECT count(*) as n FROM comp_cases)
                    SELECT
                        (SELECT n FROM cobenfy_event) as a,
                        (SELECT n FROM cobenfy_total) - (SELECT n FROM cobenfy_event) as b,
                        (SELECT n FROM comp_event) as c,
                        (SELECT n FROM comp_total) - (SELECT n FROM comp_event) as d_val,
                        (SELECT n FROM cobenfy_total) + (SELECT n FROM comp_total) as T
                """).fetchone()

                a_i, b_i, c_i, d_i, T_i = row
                if T_i == 0 or T_i is None:
                    continue

                total_a += a_i

                # MH components
                mh_num += (a_i * d_i) / T_i
                mh_den += (b_i * c_i) / T_i
                n_strata_used += 1

                # Robins-Breslow-Greenland variance (simplified)
                if T_i > 0:
                    P_i = (a_i + d_i) / T_i
                    Q_i = (b_i + c_i) / T_i
                    R_i = (a_i * d_i) / T_i
                    S_i = (b_i * c_i) / T_i
                    var_ln_mh += (P_i * R_i) / (2 * R_i**2 + 1e-10) + \
                                 (P_i * S_i + Q_i * R_i) / (2 * R_i * S_i + 1e-10) + \
                                 (Q_i * S_i) / (2 * S_i**2 + 1e-10)

            if mh_den > 0 and mh_num > 0:
                mh_ror = mh_num / mh_den
                se_ln_mh = np.sqrt(max(var_ln_mh, 1e-10))
                mh_lower = np.exp(np.log(mh_ror) - 1.96 * se_ln_mh)
                mh_upper = np.exp(np.log(mh_ror) + 1.96 * se_ln_mh)
                mh_p = 2 * (1 - stats.norm.cdf(abs(np.log(mh_ror) / se_ln_mh)))
            else:
                mh_ror, mh_lower, mh_upper, mh_p = np.nan, np.nan, np.nan, np.nan

            results.append({
                "drug_a": DRUG, "drug_b": comparator, "pt": pt,
                "a_cobenfy": total_a,
                "n_strata_used": n_strata_used,
                "mh_ror": mh_ror, "mh_ror_lower95": mh_lower,
                "mh_ror_upper95": mh_upper, "mh_pvalue": mh_p,
                "mh_significant": mh_lower > 1.0 or mh_upper < 1.0 if not np.isnan(mh_lower) else False,
            })

    df = pd.DataFrame(results)

    # Bonferroni correction
    n_tests = len(df.dropna(subset=["mh_pvalue"]))
    alpha_bonf = 0.05 / n_tests if n_tests > 0 else 0.05
    df["bonferroni_sig"] = df["mh_pvalue"] < alpha_bonf

    # Compare with crude results
    crude_path = OUTPUT_DIR / "active_comparator_results.csv"
    if crude_path.exists():
        crude_df = pd.read_csv(crude_path)

        # Merge for comparison
        crude_sub = crude_df[["drug_b", "pt", "ror", "bonferroni_sig"]].rename(
            columns={"ror": "ror_crude", "bonferroni_sig": "bonferroni_sig_crude"}
        )
        merged = df.merge(crude_sub, on=["drug_b", "pt"], how="left")
        merged["ror_change_pct"] = np.where(
            merged["ror_crude"].notna() & merged["mh_ror"].notna() & (merged["ror_crude"] > 0),
            100 * (merged["mh_ror"] - merged["ror_crude"]) / merged["ror_crude"],
            np.nan
        )

        # Summary: how many change significance direction after adjustment
        both_valid = merged.dropna(subset=["mh_ror", "ror_crude"])
        concordant = (both_valid["bonferroni_sig"] == both_valid["bonferroni_sig_crude"]).sum()
        discordant = len(both_valid) - concordant
        print(f"\n    Crude vs MH-adjusted concordance: {concordant}/{len(both_valid)} "
              f"({100*concordant/len(both_valid):.1f}%)")
        print(f"    Discordant: {discordant}")

        # Show discordant results
        disc = both_valid[both_valid["bonferroni_sig"] != both_valid["bonferroni_sig_crude"]]
        if len(disc) > 0:
            print(f"\n    DISCORDANT RESULTS (significance changed after adjustment):")
            for _, r in disc.iterrows():
                direction = "gained" if r["bonferroni_sig"] else "lost"
                print(f"      {r['pt']:<25s} vs {r['drug_b']:<15s} "
                      f"crude={r['ror_crude']:.2f} → MH={r['mh_ror']:.2f} ({direction} sig)")

    out = SUPP_DIR / "active_comparator_mh_adjusted.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"    → {out.name}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 70)
    print("  Strengthening Analyses for Peer-Review Robustness")
    print("  Serious · Polypharmacy · Disease-flag · Reporter · MH-adjusted")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    # Load primary results
    primary_path = OUTPUT_DIR / "disproportionality_cobenfy_full.csv"
    if not primary_path.exists():
        print(f"\n  ERROR: Primary results not found: {primary_path}")
        return

    primary_df = pd.read_csv(primary_path)
    consensus_pts = primary_df[
        primary_df["n_methods_signal"] >= 3
    ].sort_values("ror", ascending=False)["pt"].tolist()

    print(f"\n  Consensus signals from primary analysis: {len(consensus_pts)}")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ── Run all 5 analyses ─────────────────────────────────────────────────
    serious_df = serious_outcomes_sensitivity(con, consensus_pts)
    poly_df = polypharmacy_sensitivity(con, consensus_pts)
    disease_df = disease_manifestation_flagging(primary_df)
    reporter_df = reporter_stratified_analysis(con, consensus_pts)
    mh_df = age_sex_adjusted_comparator(con)

    con.close()
    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  ALL 5 STRENGTHENING ANALYSES COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Outputs in: {SUPP_DIR}")
    print(f"    1. sensitivity_serious_only.csv")
    print(f"    2. sensitivity_polypharmacy_monotherapy.csv")
    print(f"    3. signal_classification_disease_vs_drug.csv")
    print(f"    4. reporter_stratified_ror.csv")
    print(f"    5. active_comparator_mh_adjusted.csv")


if __name__ == "__main__":
    main()
