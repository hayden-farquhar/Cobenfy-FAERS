"""
Script 09: Database-wide MGPS prior fitting and corrected EBGM.

The original script 05 fitted the MGPS two-gamma mixture prior on only
the ~148 Cobenfy drug-event pairs. The correct DuMouchel (1999) approach
requires fitting the prior on ALL drug-event pairs in the entire database,
then applying that prior to compute EBGM/EB05 for the drug of interest.

This script:
    1. Builds contingency tables for ALL drug-PT pairs in the database
       (suspected drugs only, pairs with ≥3 reports)
    2. Fits the MGPS prior on the full set
    3. Recomputes EBGM/EB05 for Cobenfy using the database-wide prior
    4. Reclassifies signals with the corrected EBGM values
    5. Compares old vs new signal classifications

Usage:
    python scripts/09_mgps_refit.py

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
EBGM_EB05_THRESHOLD = 2.0


# ── MGPS functions (from P14, validated against openEBGM) ──────────────────

def fit_mgps_prior(n_obs: np.ndarray, E: np.ndarray, max_iter: int = 5000,
                   n_starts: int = 5):
    """
    Fit the two-component gamma mixture prior via MLE with multiple restarts.

    Using multiple random starts to avoid local optima, which is important
    when fitting on hundreds of thousands of pairs.

    Returns (alpha, a1, b1, a2, b2, OptimizeResult).
    """
    def neg_log_lik(params):
        alpha = 1.0 / (1.0 + np.exp(-params[0]))
        a1 = np.exp(params[1])
        b1 = np.exp(params[2])
        a2 = np.exp(params[3])
        b2 = np.exp(params[4])

        p1 = b1 / (b1 + E)
        p2 = b2 / (b2 + E)

        log_nb1 = stats.nbinom.logpmf(n_obs, a1, p1)
        log_nb2 = stats.nbinom.logpmf(n_obs, a2, p2)

        log_mix = np.logaddexp(np.log(alpha) + log_nb1,
                               np.log(1 - alpha) + log_nb2)

        nll = -np.sum(log_mix)
        if not np.isfinite(nll):
            return 1e15
        return nll

    # Multiple starting points
    start_points = [
        # Default (from P14)
        np.array([np.log(0.2/0.8), np.log(0.2), np.log(0.1),
                  np.log(2.0), np.log(2.0)]),
        # Alternative starts
        np.array([np.log(0.5/0.5), np.log(0.5), np.log(0.5),
                  np.log(1.0), np.log(1.0)]),
        np.array([np.log(0.1/0.9), np.log(0.1), np.log(0.05),
                  np.log(3.0), np.log(3.0)]),
        np.array([np.log(0.3/0.7), np.log(1.0), np.log(0.2),
                  np.log(0.5), np.log(4.0)]),
        np.array([np.log(0.8/0.2), np.log(0.3), np.log(0.3),
                  np.log(1.5), np.log(1.5)]),
    ]

    best_result = None
    best_nll = np.inf

    for i, x0 in enumerate(start_points[:n_starts]):
        result = minimize(
            neg_log_lik, x0, method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8},
        )
        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result
        print(f"    Start {i+1}: -LL = {result.fun:,.2f} "
              f"{'(best)' if result.fun == best_nll else ''}")

    r = best_result
    alpha = 1.0 / (1.0 + np.exp(-r.x[0]))
    a1 = np.exp(r.x[1])
    b1 = np.exp(r.x[2])
    a2 = np.exp(r.x[3])
    b2 = np.exp(r.x[4])

    return alpha, a1, b1, a2, b2, r


def _eb05_bisection(n_obs, E, a1, b1, a2, b2, Q,
                    target=0.05, tol=1e-4, max_iter=60):
    """Compute 5th percentile of posterior mixture via vectorised bisection."""
    m = len(n_obs)
    shape1 = a1 + n_obs
    scale1 = 1.0 / (b1 + E)
    shape2 = a2 + n_obs
    scale2 = 1.0 / (b2 + E)

    lower = np.zeros(m)
    upper = np.maximum(n_obs / np.maximum(E, 1e-10), 10.0) * 10.0

    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        cdf = (Q * stats.gamma.cdf(mid, shape1, scale=scale1) +
               (1 - Q) * stats.gamma.cdf(mid, shape2, scale=scale2))
        lower = np.where(cdf < target, mid, lower)
        upper = np.where(cdf >= target, mid, upper)
        if np.max(upper - lower) < tol:
            break

    return (lower + upper) / 2.0


def compute_ebgm(n_obs, E, alpha, a1, b1, a2, b2):
    """Compute EBGM and EB05 using database-wide prior."""
    n = n_obs.astype(float)

    # Posterior mixing weight Q
    p1 = b1 / (b1 + E)
    p2 = b2 / (b2 + E)

    log_nb1 = stats.nbinom.logpmf(n.astype(int), a1, p1)
    log_nb2 = stats.nbinom.logpmf(n.astype(int), a2, p2)

    log_num = np.log(alpha) + log_nb1
    log_den = np.logaddexp(log_num, np.log(1 - alpha) + log_nb2)
    Q = np.exp(log_num - log_den)

    # EBGM = exp(E[log λ])
    ebgm = np.exp(
        Q * (digamma(a1 + n) - np.log(b1 + E))
        + (1 - Q) * (digamma(a2 + n) - np.log(b2 + E))
    )

    # EB05
    eb05 = _eb05_bisection(n, E, a1, b1, a2, b2, Q)

    return ebgm, eb05, Q


def build_database_wide_contingency(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Build contingency tables for ALL drug-PT pairs in the database.

    This uses std_drug where available, and raw drugname (uppercased)
    otherwise. Only suspected drugs (PS/SS) are included.
    """
    print("  Counting total reports...")
    N = con.execute("""
        SELECT count(DISTINCT primaryid)
        FROM drug_std
        WHERE UPPER(role_cod) IN ('PS', 'SS')
    """).fetchone()[0]
    print(f"    N = {N:,}")

    print("  Building ALL drug-PT pair counts (this may take a few minutes)...")
    df = con.execute(f"""
        WITH drug_cases AS (
            SELECT DISTINCT
                primaryid,
                COALESCE(std_drug, UPPER(drugname)) as drug_key
            FROM drug_std
            WHERE UPPER(role_cod) IN ('PS', 'SS')
              AND drugname IS NOT NULL
              AND TRIM(drugname) != ''
        ),
        drug_marginals AS (
            SELECT drug_key, count(DISTINCT primaryid) as n_drug
            FROM drug_cases
            GROUP BY drug_key
        ),
        all_reactions AS (
            SELECT DISTINCT r.primaryid, UPPER(r.pt) as pt
            FROM reac r
            INNER JOIN drug_std d ON r.primaryid = d.primaryid
            WHERE UPPER(d.role_cod) IN ('PS', 'SS')
              AND r.pt IS NOT NULL AND TRIM(r.pt) != ''
        ),
        reaction_marginals AS (
            SELECT pt, count(DISTINCT primaryid) as n_reaction
            FROM all_reactions
            GROUP BY pt
        ),
        pair_counts AS (
            SELECT
                dc.drug_key,
                ar.pt,
                count(DISTINCT dc.primaryid) as a
            FROM drug_cases dc
            INNER JOIN all_reactions ar ON dc.primaryid = ar.primaryid
            GROUP BY dc.drug_key, ar.pt
            HAVING count(DISTINCT dc.primaryid) >= {MIN_REPORTS}
        )
        SELECT
            p.drug_key,
            p.pt,
            p.a,
            dm.n_drug,
            rm.n_reaction,
            {N} as N
        FROM pair_counts p
        INNER JOIN drug_marginals dm ON p.drug_key = dm.drug_key
        INNER JOIN reaction_marginals rm ON p.pt = rm.pt
    """).fetchdf()

    # Compute expected counts
    df["expected"] = (df["n_drug"].astype(float) * df["n_reaction"].astype(float)
                      / df["N"].astype(float))

    print(f"    Total drug-PT pairs (≥{MIN_REPORTS} reports): {len(df):,}")
    print(f"    Unique drugs: {df['drug_key'].nunique():,}")
    print(f"    Unique PTs: {df['pt'].nunique():,}")

    return df


def main():
    t0 = time.time()

    print("=" * 70)
    print("  MGPS Database-Wide Prior Refit")
    print("  Correcting EBGM/EB05 with proper DuMouchel methodology")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ── Step 1: Build database-wide contingency tables ──────────────────
    print("\n  STEP 1: Database-wide contingency tables")
    print(f"  {'─' * 50}")
    all_pairs = build_database_wide_contingency(con)

    # ── Step 2: Fit MGPS prior on full database ────────────────────────
    print("\n  STEP 2: Fitting MGPS prior on full database")
    print(f"  {'─' * 50}")

    n_obs_all = all_pairs["a"].values.astype(int)
    E_all = all_pairs["expected"].values

    print(f"    Fitting on {len(n_obs_all):,} drug-PT pairs...")
    alpha, a1, b1, a2, b2, opt = fit_mgps_prior(n_obs_all, E_all, n_starts=5)

    print(f"\n    DATABASE-WIDE PRIOR PARAMETERS:")
    print(f"    Mixing weight (alpha): {alpha:.6f}")
    print(f"    Component 1: Gamma(a={a1:.4f}, b={b1:.4f})  mean={a1/b1:.4f}")
    print(f"    Component 2: Gamma(a={a2:.4f}, b={b2:.4f})  mean={a2/b2:.4f}")
    print(f"    Converged: {opt.success}  |  Final -LL: {opt.fun:,.2f}")

    # ── Step 3: Recompute EBGM for Cobenfy ─────────────────────────────
    print("\n  STEP 3: Recomputing EBGM/EB05 for Cobenfy")
    print(f"  {'─' * 50}")

    # Load original Cobenfy results
    orig_path = OUTPUT_DIR / "disproportionality_cobenfy_full.csv"
    if not orig_path.exists():
        print(f"    ERROR: {orig_path} not found. Run script 05 first.")
        con.close()
        return

    cobenfy = pd.read_csv(orig_path)
    print(f"    Cobenfy drug-PT pairs: {len(cobenfy)}")

    # Save old values for comparison
    cobenfy["ebgm_old"] = cobenfy["ebgm"].copy()
    cobenfy["eb05_old"] = cobenfy["eb05"].copy()

    # Recompute with database-wide prior
    n_cob = cobenfy["a"].values.astype(int)
    E_cob = cobenfy["expected"].values

    ebgm_new, eb05_new, Q_new = compute_ebgm(n_cob, E_cob,
                                              alpha, a1, b1, a2, b2)

    cobenfy["ebgm"] = ebgm_new
    cobenfy["eb05"] = eb05_new
    cobenfy["ebgm_Q"] = Q_new

    # ── Step 4: Reclassify signals ─────────────────────────────────────
    print("\n  STEP 4: Reclassifying signals")
    print(f"  {'─' * 50}")

    cobenfy["signal_ebgm"] = cobenfy["eb05"] >= EBGM_EB05_THRESHOLD

    cobenfy["n_methods_signal"] = (
        cobenfy["signal_prr"].astype(int)
        + cobenfy["signal_ror"].astype(int)
        + cobenfy["signal_ebgm"].astype(int)
        + cobenfy["signal_bcpnn"].astype(int)
    )

    # ── Step 5: Compare old vs new ─────────────────────────────────────
    print("\n  STEP 5: Comparing original vs corrected EBGM")
    print(f"  {'─' * 50}")

    n_ebgm_old = (cobenfy["eb05_old"] >= EBGM_EB05_THRESHOLD).sum()
    n_ebgm_new = cobenfy["signal_ebgm"].sum()

    # Old signal counts (reconstruct)
    old_n_methods = (
        cobenfy["signal_prr"].astype(int)
        + cobenfy["signal_ror"].astype(int)
        + (cobenfy["eb05_old"] >= EBGM_EB05_THRESHOLD).astype(int)
        + cobenfy["signal_bcpnn"].astype(int)
    )
    old_consensus = (old_n_methods >= 3).sum()
    new_consensus = (cobenfy["n_methods_signal"] >= 3).sum()
    old_all4 = (old_n_methods == 4).sum()
    new_all4 = (cobenfy["n_methods_signal"] == 4).sum()

    print(f"    EBGM signals (EB05 ≥ 2):  {n_ebgm_old:>4d} (original) → "
          f"{n_ebgm_new:>4d} (corrected)")
    print(f"    Consensus (≥3 methods):   {old_consensus:>4d} (original) → "
          f"{new_consensus:>4d} (corrected)")
    print(f"    All 4 methods:            {old_all4:>4d} (original) → "
          f"{new_all4:>4d} (corrected)")

    # Correlation between old and new EBGM
    corr = np.corrcoef(cobenfy["ebgm_old"], cobenfy["ebgm"])[0, 1]
    print(f"\n    Pearson correlation (old vs new EBGM): {corr:.4f}")

    # Show PTs where signal classification changed
    old_sig = old_n_methods >= 3
    new_sig = cobenfy["n_methods_signal"] >= 3
    gained = cobenfy[~old_sig & new_sig]
    lost = cobenfy[old_sig & ~new_sig]

    if len(gained) > 0:
        print(f"\n    SIGNALS GAINED (not signal before, signal now):")
        for _, r in gained.iterrows():
            print(f"      {r['pt']:<40s} EB05: {r['eb05_old']:.2f} → {r['eb05']:.2f}")

    if len(lost) > 0:
        print(f"\n    SIGNALS LOST (signal before, not signal now):")
        for _, r in lost.iterrows():
            print(f"      {r['pt']:<40s} EB05: {r['eb05_old']:.2f} → {r['eb05']:.2f}")

    if len(gained) == 0 and len(lost) == 0:
        print(f"\n    No changes in signal classification.")

    # Detailed comparison for top signals
    print(f"\n    TOP 20 SIGNALS — OLD vs NEW EBGM")
    print(f"    {'─' * 80}")
    print(f"    {'PT':<35s} {'n':>5s} {'EBGM_old':>9s} {'EBGM_new':>9s} "
          f"{'EB05_old':>9s} {'EB05_new':>9s} {'#M':>3s}")
    print(f"    {'─' * 80}")
    top = cobenfy.sort_values("ebgm", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"    {str(r['pt'])[:34]:<35s} {r['a']:>5.0f} "
              f"{r['ebgm_old']:>9.2f} {r['ebgm']:>9.2f} "
              f"{r['eb05_old']:>9.2f} {r['eb05']:>9.2f} "
              f"{r['n_methods_signal']:>3.0f}")

    # ── Save corrected results ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Saving corrected results...")

    out_cols = [
        "pt", "a", "expected", "n_drug", "n_reaction", "N",
        "prr", "prr_lower95", "prr_upper95", "prr_chi2",
        "ror", "ror_lower95", "ror_upper95",
        "ebgm", "eb05", "ebgm_Q",
        "ic", "ic025", "ic975",
        "signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn",
        "n_methods_signal",
    ]

    # Also keep ror_pvalue and ror_fdr if they exist
    for extra in ["ror_pvalue", "ror_fdr", "signal_ror_fdr"]:
        if extra in cobenfy.columns:
            out_cols.append(extra)

    # Overwrite the primary results with corrected values
    full_path = OUTPUT_DIR / "disproportionality_cobenfy_full.csv"
    cobenfy.sort_values("n_methods_signal", ascending=False)[out_cols].to_csv(
        full_path, index=False, float_format="%.4f"
    )
    print(f"  {full_path.name}: {len(cobenfy)} rows (CORRECTED)")

    # Consensus signals (≥3 methods)
    consensus = cobenfy[cobenfy["n_methods_signal"] >= 3].sort_values(
        "ebgm", ascending=False)
    consensus_path = OUTPUT_DIR / "signals_cobenfy_consensus.csv"
    consensus[out_cols].to_csv(consensus_path, index=False, float_format="%.4f")
    print(f"  {consensus_path.name}: {len(consensus)} rows (CORRECTED)")

    # All-4 signals
    all4 = cobenfy[cobenfy["n_methods_signal"] == 4].sort_values(
        "ebgm", ascending=False)
    all4_path = OUTPUT_DIR / "signals_cobenfy_all4.csv"
    all4[out_cols].to_csv(all4_path, index=False, float_format="%.4f")
    print(f"  {all4_path.name}: {len(all4)} rows (CORRECTED)")

    # Save the database-wide prior parameters
    prior_path = OUTPUT_DIR / "mgps_prior_parameters.csv"
    prior_df = pd.DataFrame([{
        "scope": "database-wide",
        "n_pairs_fitted": len(all_pairs),
        "n_unique_drugs": all_pairs["drug_key"].nunique(),
        "alpha": alpha,
        "a1": a1, "b1": b1, "mean1": a1/b1,
        "a2": a2, "b2": b2, "mean2": a2/b2,
        "neg_log_lik": opt.fun,
        "converged": opt.success,
    }])
    prior_df.to_csv(prior_path, index=False, float_format="%.6f")
    print(f"  {prior_path.name}: prior parameters saved")

    # Save comparison table
    comp_path = OUTPUT_DIR / "ebgm_correction_comparison.csv"
    comp_cols = ["pt", "a", "expected", "ebgm_old", "ebgm", "eb05_old", "eb05",
                 "n_methods_signal"]
    cobenfy.sort_values("ebgm", ascending=False)[comp_cols].to_csv(
        comp_path, index=False, float_format="%.4f"
    )
    print(f"  {comp_path.name}: old vs new comparison")

    con.close()
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  MGPS REFIT COMPLETE ({elapsed:.0f}s)")
    print(f"  Primary results files have been UPDATED with corrected EBGM.")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/10_supplementary.py")


if __name__ == "__main__":
    main()
