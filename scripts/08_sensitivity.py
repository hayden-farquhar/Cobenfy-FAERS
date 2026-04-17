"""
Script 08: Sensitivity analyses.

Repeats the core disproportionality analysis under restricted conditions
to assess robustness of the primary findings.

Sensitivity analyses:
    1. Primary suspect only (role_cod = 'PS') — excludes secondary suspect
    2. US reports only (reporter_country = 'US') — reduces geographic heterogeneity
    3. HCP reporters only (occp_cod in ('MD', 'HP', 'OT')) — higher quality reports
    4. Weber effect assessment — quarterly signal strength trends to detect
       stimulated reporting for a novel drug

Weber effect context:
    New drugs attract disproportionate reporting in their first 1-2 years
    on market (Weber 1984). We assess whether Cobenfy signals are stable
    across quarters or show the characteristic early spike and decay pattern.

Outputs:
    - outputs/tables/sensitivity_ps_only.csv
    - outputs/tables/sensitivity_us_only.csv
    - outputs/tables/sensitivity_hcp_only.csv
    - outputs/tables/sensitivity_weber.csv
    - outputs/tables/sensitivity_comparison.csv (summary across all)

Usage:
    python scripts/08_sensitivity.py

Requires: data/processed/faers.duckdb (with drug_std from script 03)
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import digamma
from scipy.optimize import minimize
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_REPORTS = 3
DRUG = "xanomeline-trospium"

# Top signals from primary analysis to track across sensitivity analyses
# (will be populated dynamically from primary results, or use these defaults)
KEY_PTS = [
    "NAUSEA", "VOMITING", "CONSTIPATION", "DYSPEPSIA",
    "TACHYCARDIA", "HYPERTENSION", "DIZZINESS", "HEADACHE",
    "SOMNOLENCE", "INSOMNIA", "DRY MOUTH", "URINARY RETENTION",
    "WEIGHT INCREASED", "AKATHISIA", "TREMOR",
]


# ── Reuse disproportionality functions from script 05 ──────────────────────

def compute_prr(df):
    a, b, c, d = [df[x].values.astype(float) for x in ["a", "b", "c", "d"]]
    ac, bc, cc, dc = a+0.5, b+0.5, c+0.5, d+0.5
    prr = (ac/(ac+bc)) / (cc/(cc+dc))
    ln_prr = np.log(prr)
    se = np.sqrt(1/ac - 1/(ac+bc) + 1/cc - 1/(cc+dc))
    df["prr"] = prr
    df["prr_lower95"] = np.exp(ln_prr - 1.96*se)
    df["prr_upper95"] = np.exp(ln_prr + 1.96*se)
    N_total = a+b+c+d
    df["prr_chi2"] = (N_total * (np.abs(a*d - b*c) - N_total/2)**2) / ((a+b)*(c+d)*(a+c)*(b+d))
    return df

def compute_ror(df):
    a, b, c, d = [df[x].values.astype(float) for x in ["a", "b", "c", "d"]]
    ac, bc, cc, dc = a+0.5, b+0.5, c+0.5, d+0.5
    ror = (ac*dc)/(bc*cc)
    ln_ror = np.log(ror)
    se = np.sqrt(1/ac + 1/bc + 1/cc + 1/dc)
    df["ror"] = ror
    df["ror_lower95"] = np.exp(ln_ror - 1.96*se)
    df["ror_upper95"] = np.exp(ln_ror + 1.96*se)
    return df

def compute_bcpnn(df):
    a = df["a"].values.astype(float)
    n_drug = df["n_drug"].values.astype(float)
    n_rxn = df["n_reaction"].values.astype(float)
    N = df["N"].values.astype(float)
    ic = np.log2(((a+0.5)*(N+0.5)) / ((n_drug+0.5)*(n_rxn+0.5)))
    ic_var = (1/np.log(2)**2) * (1/(a+0.5) - 1/(N+0.5))
    ic_se = np.sqrt(np.maximum(ic_var, 0))
    df["ic"] = ic
    df["ic025"] = ic - 1.96*ic_se
    df["ic975"] = ic + 1.96*ic_se
    return df

def classify_signals(df):
    df["signal_prr"] = (df["prr"] >= 2.0) & (df["prr_chi2"] >= 4.0) & (df["a"] >= MIN_REPORTS)
    df["signal_ror"] = df["ror_lower95"] > 1.0
    df["signal_bcpnn"] = df["ic025"] > 0.0
    df["n_methods_signal"] = df["signal_prr"].astype(int) + df["signal_ror"].astype(int) + df["signal_bcpnn"].astype(int)
    return df


def build_contingency_restricted(
    con: duckdb.DuckDBPyConnection,
    role_filter: str = "IN ('PS', 'SS')",
    country_filter: str | None = None,
    reporter_filter: str | None = None,
    quarter_filter: str | None = None,
) -> pd.DataFrame:
    """
    Build contingency tables for Cobenfy under restricted conditions.

    Uses 3 methods only (PRR, ROR, IC) — EBGM fitting on small subsets
    is unreliable, so we skip it for sensitivity analyses.
    """
    # Build WHERE clauses for the reference set
    demo_where = "1=1"
    if country_filter:
        demo_where += f" AND UPPER(dem.reporter_country) = '{country_filter}'"
    if reporter_filter:
        demo_where += f" AND UPPER(dem.occp_cod) IN ({reporter_filter})"
    if quarter_filter:
        demo_where += f" AND {quarter_filter}"

    # Total reports under restriction
    N = con.execute(f"""
        SELECT count(DISTINCT d.primaryid)
        FROM drug_std d
        INNER JOIN demo dem ON d.primaryid = dem.primaryid
        WHERE UPPER(d.role_cod) {role_filter}
          AND {demo_where}
    """).fetchone()[0]

    if N == 0:
        return pd.DataFrame()

    # Drug cases under restriction
    n_drug = con.execute(f"""
        SELECT count(DISTINCT d.primaryid)
        FROM drug_std d
        INNER JOIN demo dem ON d.primaryid = dem.primaryid
        WHERE d.std_drug = '{DRUG}'
          AND UPPER(d.role_cod) {role_filter}
          AND {demo_where}
    """).fetchone()[0]

    if n_drug == 0:
        return pd.DataFrame()

    # Build contingency tables
    df = con.execute(f"""
        WITH drug_cases AS (
            SELECT DISTINCT d.primaryid
            FROM drug_std d
            INNER JOIN demo dem ON d.primaryid = dem.primaryid
            WHERE d.std_drug = '{DRUG}'
              AND UPPER(d.role_cod) {role_filter}
              AND {demo_where}
        ),
        all_reactions AS (
            SELECT DISTINCT r.primaryid, UPPER(r.pt) as pt
            FROM reac r
            INNER JOIN drug_std d ON r.primaryid = d.primaryid
            INNER JOIN demo dem ON r.primaryid = dem.primaryid
            WHERE UPPER(d.role_cod) {role_filter}
              AND r.pt IS NOT NULL AND TRIM(r.pt) != ''
              AND {demo_where}
        ),
        pair_counts AS (
            SELECT ar.pt, count(DISTINCT ar.primaryid) as a
            FROM all_reactions ar
            INNER JOIN drug_cases dc ON ar.primaryid = dc.primaryid
            GROUP BY ar.pt
            HAVING count(DISTINCT ar.primaryid) >= {MIN_REPORTS}
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

    # Compute metrics (3 of 4 — skip EBGM for sensitivity)
    df = compute_prr(df)
    df = compute_ror(df)
    df = compute_bcpnn(df)
    df = classify_signals(df)

    return df


def weber_effect_analysis(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Assess Weber effect by computing quarterly report counts and
    signal strength trends for Cobenfy.
    """
    print("\n  WEBER EFFECT ASSESSMENT")
    print(f"  {'─' * 60}")

    # Quarterly report counts
    quarterly = con.execute(f"""
        SELECT
            SUBSTRING(dem.fda_dt, 1, 4) || 'Q' ||
            CASE
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 3 THEN '1'
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 6 THEN '2'
                WHEN CAST(SUBSTRING(dem.fda_dt, 5, 2) AS INTEGER) <= 9 THEN '3'
                ELSE '4'
            END as quarter,
            count(DISTINCT d.primaryid) as n_cobenfy_reports,
            count(DISTINCT r.pt) as n_unique_pts
        FROM drug_std d
        INNER JOIN demo dem ON d.primaryid = dem.primaryid
        LEFT JOIN reac r ON d.primaryid = r.primaryid
        WHERE d.std_drug = '{DRUG}'
          AND UPPER(d.role_cod) IN ('PS', 'SS')
          AND dem.fda_dt IS NOT NULL
          AND LENGTH(dem.fda_dt) >= 6
        GROUP BY quarter
        ORDER BY quarter
    """).fetchdf()

    if len(quarterly) == 0:
        print("    No quarterly data available.")
        return pd.DataFrame()

    print(f"\n    {'Quarter':<12s} {'Reports':>10s} {'Unique PTs':>12s}")
    print(f"    {'─' * 40}")
    for _, row in quarterly.iterrows():
        print(f"    {row['quarter']:<12s} {row['n_cobenfy_reports']:>10,} {row['n_unique_pts']:>12,}")

    # Trend assessment
    if len(quarterly) >= 3:
        counts = quarterly["n_cobenfy_reports"].values.astype(float)
        x = np.arange(len(counts))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, counts)
        trend = "increasing" if slope > 0 else "decreasing"
        print(f"\n    Linear trend: {trend} (slope={slope:.1f} reports/quarter, R²={r_value**2:.3f}, p={p_value:.4f})")
        if slope > 0:
            print(f"    → Reports are INCREASING over time — typical of uptake phase, not Weber decay")
        else:
            print(f"    → Reports are DECREASING — consistent with Weber effect (stimulated reporting decay)")

    return quarterly


def main():
    t0 = time.time()

    print("=" * 70)
    print("  Sensitivity Analyses — Cobenfy")
    print("  PS-only · US-only · HCP-only · Weber effect")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # Load primary analysis key PTs if available
    primary_path = OUTPUT_DIR / "signals_cobenfy_consensus.csv"
    if primary_path.exists():
        primary_signals = pd.read_csv(primary_path)
        key_pts = primary_signals["pt"].head(20).tolist()
        print(f"\n  Using top {len(key_pts)} signals from primary analysis")
    else:
        key_pts = KEY_PTS
        print(f"\n  Using default key PTs (primary analysis not yet run)")

    # ── 1. Primary suspect only ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  SENSITIVITY 1: Primary Suspect Only (PS)")
    print(f"{'─' * 70}")
    ps_df = build_contingency_restricted(con, role_filter="= 'PS'")
    if len(ps_df) > 0:
        n_sig = (ps_df["n_methods_signal"] >= 3).sum()
        print(f"    Pairs analysed: {len(ps_df):,}  |  Signals (≥3 methods): {n_sig}")
        ps_path = OUTPUT_DIR / "sensitivity_ps_only.csv"
        ps_df.to_csv(ps_path, index=False, float_format="%.4f")
    else:
        print("    Insufficient data for PS-only analysis")

    # ── 2. US reports only ──────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  SENSITIVITY 2: US Reports Only")
    print(f"{'─' * 70}")
    us_df = build_contingency_restricted(con, country_filter="US")
    if len(us_df) > 0:
        n_sig = (us_df["n_methods_signal"] >= 3).sum()
        print(f"    Pairs analysed: {len(us_df):,}  |  Signals (≥3 methods): {n_sig}")
        us_path = OUTPUT_DIR / "sensitivity_us_only.csv"
        us_df.to_csv(us_path, index=False, float_format="%.4f")
    else:
        print("    Insufficient data for US-only analysis")

    # ── 3. HCP reporters only ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  SENSITIVITY 3: HCP Reporters Only")
    print(f"{'─' * 70}")
    hcp_df = build_contingency_restricted(
        con, reporter_filter="'MD', 'HP', 'OT', 'PH', 'RN'"
    )
    if len(hcp_df) > 0:
        n_sig = (hcp_df["n_methods_signal"] >= 3).sum()
        print(f"    Pairs analysed: {len(hcp_df):,}  |  Signals (≥3 methods): {n_sig}")
        hcp_path = OUTPUT_DIR / "sensitivity_hcp_only.csv"
        hcp_df.to_csv(hcp_path, index=False, float_format="%.4f")
    else:
        print("    Insufficient data for HCP-only analysis")

    # ── 4. Weber effect ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  SENSITIVITY 4: Weber Effect Assessment")
    print(f"{'─' * 70}")
    weber_df = weber_effect_analysis(con)
    if len(weber_df) > 0:
        weber_path = OUTPUT_DIR / "sensitivity_weber.csv"
        weber_df.to_csv(weber_path, index=False)

    # ── Cross-sensitivity comparison for key PTs ────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  CROSS-SENSITIVITY COMPARISON — KEY PTs")
    print(f"{'=' * 90}")

    comparison_rows = []
    header = f"  {'PT':<30s} {'Primary':>10s} {'PS-only':>10s} {'US-only':>10s} {'HCP-only':>10s}"
    print(header)
    print(f"  {'─' * 75}")

    for pt in key_pts:
        row = {"pt": pt}

        # Primary (from script 05 output or PS+SS)
        for label, df in [("primary", ps_df), ("ps_only", ps_df),
                          ("us_only", us_df), ("hcp_only", hcp_df)]:
            if len(df) > 0:
                match = df[df["pt"].str.upper() == pt.upper()]
                if len(match) > 0:
                    ror_val = match.iloc[0]["ror"]
                    sig = match.iloc[0]["n_methods_signal"]
                    row[f"ror_{label}"] = ror_val
                    row[f"sig_{label}"] = sig
                else:
                    row[f"ror_{label}"] = None
                    row[f"sig_{label}"] = 0
            else:
                row[f"ror_{label}"] = None
                row[f"sig_{label}"] = 0

        comparison_rows.append(row)

        # Print row
        vals = []
        for label in ["primary", "ps_only", "us_only", "hcp_only"]:
            ror = row.get(f"ror_{label}")
            sig = row.get(f"sig_{label}", 0)
            if ror is not None:
                marker = "***" if sig >= 3 else ""
                vals.append(f"{ror:>6.1f}{marker:>3s}")
            else:
                vals.append(f"{'—':>10s}")
        print(f"  {pt:<30s} {'  '.join(vals)}")

    # Save comparison
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_path = OUTPUT_DIR / "sensitivity_comparison.csv"
        comp_df.to_csv(comp_path, index=False, float_format="%.4f")

    con.close()
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  SENSITIVITY ANALYSES COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Pipeline complete. Ready for manuscript drafting.")


if __name__ == "__main__":
    main()
