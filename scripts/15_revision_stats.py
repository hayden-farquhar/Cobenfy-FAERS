"""
Script 15: Revision statistics for Schizophrenia Bulletin major revision.

Addresses reviewer comments that require new computation:
    - R1.6: bivariate tests of age / sex / HCP-reporter proportion ACROSS the
            7 agents (Kruskal-Wallis for age; chi-square for sex and reporter).
    - R2.2: psychiatric comorbidity / indication description for the Cobenfy cohort.

Read-only against data/processed/faers.duckdb. Reuses the exact case definitions
from script 04 (PS or SS role; std_drug / is_cobenfy on the drug_std table).

Outputs:
    outputs/supplementary/revision_bivariate_demographics.csv
    outputs/supplementary/revision_cobenfy_indications.csv
    Console: test statistics, degrees of freedom, p-values.

Usage:
    python scripts/15_revision_stats.py
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
OUT_DIR = PROJECT_ROOT / "outputs" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRUGS = [
    "xanomeline-trospium", "olanzapine", "risperidone",
    "aripiprazole", "quetiapine", "lurasidone", "brexpiprazole",
]


def standardise_age(age_val, age_cod):
    try:
        age = float(age_val)
    except (ValueError, TypeError):
        return None
    if age_cod is None:
        return None
    cod = str(age_cod).upper().strip()
    factor = {"YR": 1, "YEAR": 1, "Y": 1, "MON": 1/12, "MONTH": 1/12, "MO": 1/12,
              "WK": 1/52, "WEEK": 1/52, "DY": 1/365.25, "DAY": 1/365.25, "D": 1/365.25,
              "DEC": 10, "DECADE": 10, "HR": 1/(365.25*24), "HOUR": 1/(365.25*24)}
    return age * factor.get(cod, 1)


def case_query(drug):
    """Per-case demo rows for a drug, matching script 04 (PS/SS)."""
    if drug == "xanomeline-trospium":
        pred = "d.is_cobenfy = TRUE"
    else:
        pred = f"d.std_drug = '{drug}'"
    return f"""
        SELECT dem.primaryid, dem.age, dem.age_cod, dem.sex, dem.occp_cod
        FROM demo dem
        INNER JOIN (
            SELECT DISTINCT primaryid FROM drug_std d
            WHERE {pred} AND UPPER(d.role_cod) IN ('PS','SS')
        ) c ON dem.primaryid = c.primaryid
    """


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)

    per_drug = {}
    for drug in DRUGS:
        df = con.execute(case_query(drug)).fetchdf()
        df["age_years"] = [standardise_age(a, c) for a, c in zip(df["age"], df["age_cod"])]
        df.loc[(df["age_years"] < 0) | (df["age_years"] > 120), "age_years"] = np.nan
        df["sex_u"] = df["sex"].str.upper()
        occ = df["occp_cod"].str.upper()
        df["is_hcp"] = occ.isin(["MD", "HP", "OT", "PH"])  # healthcare professional roles
        per_drug[drug] = df

    print("=" * 68)
    print("  R1.6 BIVARIATE DEMOGRAPHIC TESTS ACROSS 7 AGENTS")
    print("=" * 68)

    # --- Age: Kruskal-Wallis ---
    age_groups = [per_drug[d]["age_years"].dropna().values for d in DRUGS]
    kw_h, kw_p = stats.kruskal(*age_groups)
    print(f"\n  Age (Kruskal-Wallis): H = {kw_h:.1f}, df = {len(DRUGS)-1}, "
          f"p = {kw_p:.3e}")
    for d in DRUGS:
        a = per_drug[d]["age_years"].dropna()
        print(f"    {d:<22s} median {a.median():>4.0f}  (IQR {a.quantile(.25):.0f}-{a.quantile(.75):.0f}), n={len(a)}")

    # --- Sex: chi-square (F vs M, excluding missing) ---
    sex_tab = np.array([[ (per_drug[d]["sex_u"] == "F").sum(),
                          (per_drug[d]["sex_u"] == "M").sum() ] for d in DRUGS])
    chi2_s, p_s, dof_s, _ = stats.chi2_contingency(sex_tab)
    print(f"\n  Sex F/M (chi-square): chi2 = {chi2_s:.1f}, df = {dof_s}, p = {p_s:.3e}")

    # --- Reporter: chi-square (HCP vs non-HCP) ---
    rep_tab = np.array([[ per_drug[d]["is_hcp"].sum(),
                          (~per_drug[d]["is_hcp"]).sum() ] for d in DRUGS])
    chi2_r, p_r, dof_r, _ = stats.chi2_contingency(rep_tab)
    print(f"\n  Reporter HCP/non-HCP (chi-square): chi2 = {chi2_r:.1f}, "
          f"df = {dof_r}, p = {p_r:.3e}")
    for i, d in enumerate(DRUGS):
        hcp, non = rep_tab[i]
        print(f"    {d:<22s} HCP {hcp:>5,} / {hcp+non:>5,}  ({100*hcp/(hcp+non):.1f}%)")

    # Save bivariate summary
    rows = []
    for i, d in enumerate(DRUGS):
        a = per_drug[d]["age_years"].dropna()
        rows.append({
            "drug": d, "n": len(per_drug[d]),
            "age_median": a.median(), "age_q25": a.quantile(.25), "age_q75": a.quantile(.75),
            "age_n": len(a),
            "female_n": int((per_drug[d]["sex_u"] == "F").sum()),
            "male_n": int((per_drug[d]["sex_u"] == "M").sum()),
            "hcp_n": int(rep_tab[i][0]), "non_hcp_n": int(rep_tab[i][1]),
        })
    bdf = pd.DataFrame(rows)
    bdf.attrs  # noop
    out1 = OUT_DIR / "revision_bivariate_demographics.csv"
    bdf.to_csv(out1, index=False)
    # append test results as a trailing metadata file
    with open(OUT_DIR / "revision_bivariate_tests.txt", "w") as f:
        f.write(f"Age Kruskal-Wallis: H={kw_h:.4f}, df={len(DRUGS)-1}, p={kw_p:.6e}\n")
        f.write(f"Sex chi-square: chi2={chi2_s:.4f}, df={dof_s}, p={p_s:.6e}\n")
        f.write(f"Reporter HCP chi-square: chi2={chi2_r:.4f}, df={dof_r}, p={p_r:.6e}\n")
    print(f"\n  Saved: {out1}")

    # --- R2.2 Cobenfy indications / comorbidity ---
    print("\n" + "=" * 68)
    print("  R2.2 COBENFY COHORT INDICATIONS (psychiatric comorbidity)")
    print("=" * 68)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "indi" in tables:
        cols = [c[1] for c in con.execute("PRAGMA table_info('indi')").fetchall()]
        pt_col = "indi_pt" if "indi_pt" in cols else ("indi_drug_rec_act" if "indi_drug_rec_act" in cols else cols[-1])
        indi = con.execute(f"""
            SELECT UPPER(i.{pt_col}) AS indication, count(DISTINCT i.primaryid) AS n
            FROM indi i
            INNER JOIN (
                SELECT DISTINCT primaryid FROM drug_std
                WHERE is_cobenfy = TRUE AND UPPER(role_cod) IN ('PS','SS')
            ) c ON i.primaryid = c.primaryid
            WHERE i.{pt_col} IS NOT NULL
            GROUP BY 1 ORDER BY n DESC LIMIT 30
        """).fetchdf()
        print(indi.to_string(index=False))
        out2 = OUT_DIR / "revision_cobenfy_indications.csv"
        indi.to_csv(out2, index=False)
        print(f"\n  Saved: {out2}")
    else:
        print("  (indi table not present; indication analysis skipped)")

    con.close()


if __name__ == "__main__":
    main()
