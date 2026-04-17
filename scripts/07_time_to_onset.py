"""
Script 07: Time-to-onset (TTO) analysis.

Computes time-to-onset for Cobenfy adverse events using FAERS therapy
dates (THER table) and event dates (DEMO table).

Method:
    TTO = event_dt - start_dt (from THER table for the Cobenfy drug entry)
    Fit Weibull distribution to characterise onset patterns:
        - β < 1: decreasing hazard (early-onset, suggests direct toxicity)
        - β = 1: constant hazard (exponential; time-independent)
        - β > 1: increasing hazard (late-onset, suggests cumulative effect)

    Categorisation:
        - Early:        <30 days
        - Intermediate: 30–180 days
        - Late:         >180 days

    Stratification: age (<65 vs ≥65), sex, CYP2D6 inhibitor co-medication

Outputs:
    - outputs/tables/time_to_onset_summary.csv
    - outputs/tables/time_to_onset_weibull.csv (Weibull parameters per PT)

Usage:
    python scripts/07_time_to_onset.py

Requires: data/processed/faers.duckdb
"""

import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import time
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TTO_REPORTS = 20  # minimum reports for Weibull fit


def parse_faers_date(date_str: str) -> pd.Timestamp | None:
    """
    Parse FAERS date strings which come in various formats:
    YYYYMMDD, YYYYMM, YYYY, or sometimes with slashes.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    if len(date_str) < 4:
        return None
    try:
        if len(date_str) == 8:
            return pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:8])
        elif len(date_str) == 6:
            return pd.Timestamp(date_str[:4] + "-" + date_str[4:6] + "-15")  # mid-month
        elif len(date_str) == 4:
            return pd.Timestamp(date_str + "-07-01")  # mid-year
        else:
            return pd.to_datetime(date_str, errors="coerce")
    except Exception:
        return None


def compute_tto(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute time-to-onset for all Cobenfy cases with available dates.

    Strategy:
        1. Get event_dt from DEMO table
        2. Get start_dt from THER table (matching on drug_seq for Cobenfy)
        3. TTO = event_dt - start_dt
    """
    print("  Extracting dates for Cobenfy cases...")

    # Join DEMO (event_dt) with THER (start_dt) for Cobenfy drug entries
    df = con.execute("""
        SELECT
            dem.primaryid,
            dem.event_dt,
            dem.age,
            dem.age_cod,
            dem.sex,
            t.start_dt,
            r.pt,
            d.drug_seq
        FROM demo dem
        INNER JOIN drug_std d ON dem.primaryid = d.primaryid
        LEFT JOIN ther t ON dem.primaryid = t.primaryid
                         AND d.drug_seq = t.dsg_drug_seq
        INNER JOIN reac r ON dem.primaryid = r.primaryid
        WHERE d.is_cobenfy = TRUE
          AND UPPER(d.role_cod) IN ('PS', 'SS')
          AND r.pt IS NOT NULL
    """).fetchdf()

    print(f"    Raw case-reaction-therapy records: {len(df):,}")

    # Parse dates
    df["event_date"] = df["event_dt"].apply(parse_faers_date)
    df["start_date"] = df["start_dt"].apply(parse_faers_date)

    # Compute TTO in days
    mask = df["event_date"].notna() & df["start_date"].notna()
    df.loc[mask, "tto_days"] = (df.loc[mask, "event_date"] - df.loc[mask, "start_date"]).dt.days

    # Filter to valid TTOs (>= 0 and <= 730 days / 2 years)
    valid = df[(df["tto_days"] >= 0) & (df["tto_days"] <= 730)].copy()
    print(f"    Records with valid TTO (0–730 days): {len(valid):,}")
    print(f"    Unique cases with TTO: {valid['primaryid'].nunique():,}")
    print(f"    Unique PTs with TTO: {valid['pt'].nunique():,}")

    # Check for CYP2D6 co-medication
    cyp_cases = con.execute("""
        SELECT DISTINCT primaryid, cyp2d6_inhibitor
        FROM drug_std
        WHERE cyp2d6_inhibitor IS NOT NULL
          AND primaryid IN (
              SELECT DISTINCT primaryid
              FROM drug_std
              WHERE is_cobenfy = TRUE
          )
    """).fetchdf()

    if len(cyp_cases) > 0:
        valid = valid.merge(cyp_cases[["primaryid", "cyp2d6_inhibitor"]],
                           on="primaryid", how="left")
    else:
        valid["cyp2d6_inhibitor"] = None

    return valid


def fit_weibull(tto_days: np.ndarray) -> dict:
    """
    Fit Weibull distribution to time-to-onset data.

    Returns shape (β), scale (η), and onset category percentages.
    """
    tto = tto_days[tto_days > 0].astype(float)
    if len(tto) < 5:
        return {"shape": None, "scale": None, "n_fitted": len(tto)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            shape, loc, scale = stats.weibull_min.fit(tto, floc=0)
        except Exception:
            return {"shape": None, "scale": None, "n_fitted": len(tto)}

    # Onset categories
    n_total = len(tto_days)
    early = np.sum(tto_days < 30) / n_total * 100
    intermediate = np.sum((tto_days >= 30) & (tto_days <= 180)) / n_total * 100
    late = np.sum(tto_days > 180) / n_total * 100

    return {
        "shape": shape,
        "scale": scale,
        "median_tto": np.median(tto_days),
        "mean_tto": np.mean(tto_days),
        "n_fitted": len(tto),
        "pct_early": early,
        "pct_intermediate": intermediate,
        "pct_late": late,
        "hazard_pattern": "decreasing" if shape < 1 else ("constant" if abs(shape - 1) < 0.1 else "increasing"),
    }


def main():
    t0 = time.time()

    print("=" * 70)
    print("  Time-to-Onset Analysis — Cobenfy")
    print("  Weibull Parameterisation")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)
    valid = compute_tto(con)
    con.close()

    if len(valid) == 0:
        print("\n  No valid TTO data available.")
        return

    # ── Overall TTO distribution ────────────────────────────────────────
    all_tto = valid.drop_duplicates(subset=["primaryid", "pt"])["tto_days"].dropna()
    print(f"\n  OVERALL TTO DISTRIBUTION (n={len(all_tto):,})")
    print(f"  {'─' * 40}")
    print(f"    Median: {all_tto.median():.0f} days")
    print(f"    Mean:   {all_tto.mean():.0f} days")
    print(f"    IQR:    {all_tto.quantile(0.25):.0f}–{all_tto.quantile(0.75):.0f} days")
    print(f"    Early (<30d):        {(all_tto < 30).mean()*100:.1f}%")
    print(f"    Intermediate (30-180d): {((all_tto >= 30) & (all_tto <= 180)).mean()*100:.1f}%")
    print(f"    Late (>180d):        {(all_tto > 180).mean()*100:.1f}%")

    # ── Weibull fit per PT ──────────────────────────────────────────────
    print(f"\n  WEIBULL FIT PER PREFERRED TERM (≥{MIN_TTO_REPORTS} reports)")
    print(f"  {'─' * 85}")

    pt_groups = valid.drop_duplicates(subset=["primaryid", "pt"]).groupby("pt")
    weibull_results = []

    header = f"  {'PT':<35s} {'n':>5s} {'Med':>5s} {'β':>6s} {'η':>7s} {'Early%':>7s} {'Inter%':>7s} {'Late%':>6s} {'Pattern':<12s}"
    print(header)
    print(f"  {'─' * 85}")

    for pt, group in pt_groups:
        tto = group["tto_days"].dropna().values
        if len(tto) < MIN_TTO_REPORTS:
            continue

        result = fit_weibull(tto)
        result["pt"] = pt
        result["n_reports"] = len(tto)
        weibull_results.append(result)

        if result["shape"] is not None:
            print(
                f"  {str(pt)[:34]:<35s} {result['n_reports']:>5d} "
                f"{result['median_tto']:>5.0f} "
                f"{result['shape']:>6.2f} {result['scale']:>7.1f} "
                f"{result['pct_early']:>6.1f}% {result['pct_intermediate']:>6.1f}% "
                f"{result['pct_late']:>5.1f}% {result['hazard_pattern']:<12s}"
            )

    # ── Stratified analysis ─────────────────────────────────────────────
    print(f"\n  STRATIFIED TTO ANALYSIS")
    print(f"  {'─' * 60}")

    # By sex
    for sex_label, sex_code in [("Female", "F"), ("Male", "M")]:
        subset = valid[valid["sex"].str.upper() == sex_code]["tto_days"].dropna()
        if len(subset) >= 10:
            print(f"    {sex_label:15s}: n={len(subset):>5,}  median={subset.median():>5.0f}d  mean={subset.mean():>5.0f}d")

    # By age group
    def parse_age(row):
        try:
            age = float(row["age"])
            cod = str(row.get("age_cod", "YR")).upper()
            if cod in ("YR", "YEAR", "Y"):
                return age
            elif cod in ("MON", "MONTH"):
                return age / 12
            return age
        except (ValueError, TypeError):
            return None

    valid["age_years"] = valid.apply(parse_age, axis=1)
    for label, mask in [("<65 years", valid["age_years"] < 65), ("≥65 years", valid["age_years"] >= 65)]:
        subset = valid[mask]["tto_days"].dropna()
        if len(subset) >= 10:
            print(f"    {label:15s}: n={len(subset):>5,}  median={subset.median():>5.0f}d  mean={subset.mean():>5.0f}d")

    # By CYP2D6 inhibitor co-medication
    cyp_yes = valid[valid["cyp2d6_inhibitor"].notna()]["tto_days"].dropna()
    cyp_no = valid[valid["cyp2d6_inhibitor"].isna()]["tto_days"].dropna()
    if len(cyp_yes) >= 5:
        print(f"    {'CYP2D6 inh +':15s}: n={len(cyp_yes):>5,}  median={cyp_yes.median():>5.0f}d  mean={cyp_yes.mean():>5.0f}d")
        print(f"    {'CYP2D6 inh -':15s}: n={len(cyp_no):>5,}  median={cyp_no.median():>5.0f}d  mean={cyp_no.mean():>5.0f}d")

        # Wilcoxon rank-sum test
        if len(cyp_yes) >= 10 and len(cyp_no) >= 10:
            stat, p = stats.mannwhitneyu(cyp_yes, cyp_no, alternative="two-sided")
            print(f"    Mann-Whitney U test: p={p:.4f}")

    # ── Save results ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Saving results...")

    # Summary per PT
    summary_df = valid.drop_duplicates(subset=["primaryid", "pt"]).groupby("pt").agg(
        n_reports=("tto_days", "count"),
        median_tto=("tto_days", "median"),
        mean_tto=("tto_days", "mean"),
        q25_tto=("tto_days", lambda x: x.quantile(0.25)),
        q75_tto=("tto_days", lambda x: x.quantile(0.75)),
    ).reset_index()
    summary_df = summary_df.sort_values("n_reports", ascending=False)
    summary_path = OUTPUT_DIR / "time_to_onset_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.1f")
    print(f"  {summary_path.name}: {len(summary_df)} rows")

    # Weibull parameters
    if weibull_results:
        weibull_df = pd.DataFrame(weibull_results)
        weibull_df = weibull_df.sort_values("n_reports", ascending=False)
        weibull_path = OUTPUT_DIR / "time_to_onset_weibull.csv"
        weibull_df.to_csv(weibull_path, index=False, float_format="%.4f")
        print(f"  {weibull_path.name}: {len(weibull_df)} rows")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  TIME-TO-ONSET ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/08_sensitivity.py")


if __name__ == "__main__":
    main()
