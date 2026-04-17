"""
Script 05: Disproportionality Analysis — PRR, ROR, EBGM, BCPNN/IC.

Computes four standard pharmacovigilance signal detection metrics for
xanomeline-trospium (Cobenfy) against the full FAERS reference set.

Methods (adapted from Project 14, cross-validated against openEBGM):
    1. PRR  — Proportional Reporting Ratio (Evans et al. 2001)
    2. ROR  — Reporting Odds Ratio
    3. EBGM — Empirical Bayesian Geometric Mean / MGPS (DuMouchel 1999)
    4. IC   — Information Component / BCPNN (Bate et al. 1998)

Signal definition: ≥3 of 4 methods positive (consistent with P14):
    PRR:  PRR ≥ 2  AND  χ² ≥ 4  AND  N ≥ 3
    ROR:  lower 95% CI > 1
    EBGM: EB05 ≥ 2
    IC:   IC025 > 0

Multiple testing: Benjamini-Hochberg FDR at 0.05

Outputs:
    - outputs/tables/disproportionality_cobenfy_full.csv
    - outputs/tables/signals_cobenfy_consensus.csv (≥3 methods)
    - outputs/tables/signals_cobenfy_all4.csv (all 4 methods)

Usage:
    python scripts/05_disproportionality.py

Requires: data/processed/faers.duckdb (with drug_std table from script 03)
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

# ── Configuration ───────────────────────────────────────────────────────────

MIN_REPORTS = 3          # minimum case count per drug-PT pair
FDR_ALPHA = 0.05         # Benjamini-Hochberg FDR threshold

# Signal thresholds (matching P14)
PRR_THRESHOLD = 2.0
PRR_CHI2_THRESHOLD = 4.0
ROR_LOWER_CI_THRESHOLD = 1.0
EBGM_EB05_THRESHOLD = 2.0
IC025_THRESHOLD = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  CONTINGENCY TABLE CONSTRUCTION (via DuckDB)
# ═══════════════════════════════════════════════════════════════════════════


def build_contingency_tables(con: duckdb.DuckDBPyConnection, drug: str = "xanomeline-trospium") -> pd.DataFrame:
    """
    Build 2×2 contingency tables for all drug-PT pairs in FAERS.

    Uses DuckDB SQL for efficient computation across the full database.

    The 2×2 table for drug D and adverse event E:
                   | E present | E absent |
        D present  |     a     |    b     |  n_drug
        D absent   |     c     |    d     |
                   | n_reaction|          |  N
    """
    print(f"  Building contingency tables for {drug}...")

    # Total unique reports in the database (using suspected drugs only)
    N = con.execute("""
        SELECT count(DISTINCT primaryid)
        FROM drug_std
        WHERE UPPER(role_cod) IN ('PS', 'SS')
    """).fetchone()[0]
    print(f"    Total reports (suspected drugs): {N:,}")

    # Drug marginal: number of reports with this drug
    n_drug = con.execute(f"""
        SELECT count(DISTINCT primaryid)
        FROM drug_std
        WHERE std_drug = '{drug}'
          AND UPPER(role_cod) IN ('PS', 'SS')
    """).fetchone()[0]
    print(f"    Reports with {drug}: {n_drug:,}")

    # Build pair counts (a) and reaction marginals (n_reaction) in one query
    df = con.execute(f"""
        WITH drug_cases AS (
            SELECT DISTINCT primaryid
            FROM drug_std
            WHERE std_drug = '{drug}'
              AND UPPER(role_cod) IN ('PS', 'SS')
        ),
        -- All reactions linked to suspected drugs (for marginals)
        all_reactions AS (
            SELECT DISTINCT r.primaryid, UPPER(r.pt) as pt
            FROM reac r
            INNER JOIN drug_std d ON r.primaryid = d.primaryid
            WHERE UPPER(d.role_cod) IN ('PS', 'SS')
              AND r.pt IS NOT NULL
              AND TRIM(r.pt) != ''
        ),
        -- Pair counts: drug + specific reaction
        pair_counts AS (
            SELECT
                ar.pt,
                count(DISTINCT ar.primaryid) as a
            FROM all_reactions ar
            INNER JOIN drug_cases dc ON ar.primaryid = dc.primaryid
            GROUP BY ar.pt
            HAVING count(DISTINCT ar.primaryid) >= {MIN_REPORTS}
        ),
        -- Reaction marginals: total reports with each reaction
        reaction_marginals AS (
            SELECT
                pt,
                count(DISTINCT primaryid) as n_reaction
            FROM all_reactions
            GROUP BY pt
        )
        SELECT
            p.pt,
            p.a,
            rm.n_reaction,
            {n_drug} as n_drug,
            {N} as N
        FROM pair_counts p
        INNER JOIN reaction_marginals rm ON p.pt = rm.pt
        ORDER BY p.a DESC
    """).fetchdf()

    # Compute remaining 2×2 cells
    df["b"] = df["n_drug"] - df["a"]
    df["c"] = df["n_reaction"] - df["a"]
    df["d"] = df["N"] - df["a"] - df["b"] - df["c"]

    # Expected count under independence
    df["expected"] = df["n_drug"].astype(float) * df["n_reaction"].astype(float) / df["N"]

    print(f"    Drug-PT pairs with ≥{MIN_REPORTS} reports: {len(df):,}")
    print(f"    Unique PTs: {df['pt'].nunique():,}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FREQUENTIST METHODS: PRR AND ROR (from P14)
# ═══════════════════════════════════════════════════════════════════════════


def compute_prr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proportional Reporting Ratio with 95% CI and chi-squared.
    PRR = (a/(a+b)) / (c/(c+d))
    Signal: PRR >= 2, chi² >= 4, a >= 3
    """
    a = df["a"].values.astype(float)
    b = df["b"].values.astype(float)
    c = df["c"].values.astype(float)
    d = df["d"].values.astype(float)

    # Continuity correction for zero cells
    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    prr = (ac / (ac + bc)) / (cc / (cc + dc))
    ln_prr = np.log(prr)
    se = np.sqrt(1 / ac - 1 / (ac + bc) + 1 / cc - 1 / (cc + dc))

    df["prr"] = prr
    df["prr_lower95"] = np.exp(ln_prr - 1.96 * se)
    df["prr_upper95"] = np.exp(ln_prr + 1.96 * se)

    # Yates-corrected chi-squared
    N_total = a + b + c + d
    df["prr_chi2"] = (N_total * (np.abs(a * d - b * c) - N_total / 2) ** 2) / (
        (a + b) * (c + d) * (a + c) * (b + d)
    )

    return df


def compute_ror(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reporting Odds Ratio with 95% CI.
    ROR = (a*d) / (b*c)
    Signal: lower 95% CI > 1
    """
    a = df["a"].values.astype(float)
    b = df["b"].values.astype(float)
    c = df["c"].values.astype(float)
    d = df["d"].values.astype(float)

    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    ror = (ac * dc) / (bc * cc)
    ln_ror = np.log(ror)
    se = np.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)

    df["ror"] = ror
    df["ror_lower95"] = np.exp(ln_ror - 1.96 * se)
    df["ror_upper95"] = np.exp(ln_ror + 1.96 * se)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  EMPIRICAL BAYESIAN GEOMETRIC MEAN (MGPS / DuMouchel 1999 — from P14)
# ═══════════════════════════════════════════════════════════════════════════


def fit_mgps_prior(n_obs: np.ndarray, E: np.ndarray, max_iter: int = 5000):
    """
    Fit the two-component gamma mixture prior via MLE.

    Returns (alpha, a1, b1, a2, b2, OptimizeResult).
    Identical to P14 implementation (validated against openEBGM R package).
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

        log_mix = np.logaddexp(np.log(alpha) + log_nb1, np.log(1 - alpha) + log_nb2)

        nll = -np.sum(log_mix)
        if not np.isfinite(nll):
            return 1e15
        return nll

    x0 = np.array([
        np.log(0.2 / 0.8),
        np.log(0.2),
        np.log(0.1),
        np.log(2.0),
        np.log(2.0),
    ])

    result = minimize(
        neg_log_lik, x0, method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8},
    )

    alpha = 1.0 / (1.0 + np.exp(-result.x[0]))
    a1 = np.exp(result.x[1])
    b1 = np.exp(result.x[2])
    a2 = np.exp(result.x[3])
    b2 = np.exp(result.x[4])

    return alpha, a1, b1, a2, b2, result


def _eb05_bisection(n_obs, E, a1, b1, a2, b2, Q, target=0.05, tol=1e-4, max_iter=60):
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
        cdf = Q * stats.gamma.cdf(mid, shape1, scale=scale1) + (1 - Q) * stats.gamma.cdf(
            mid, shape2, scale=scale2
        )
        lower = np.where(cdf < target, mid, lower)
        upper = np.where(cdf >= target, mid, upper)
        if np.max(upper - lower) < tol:
            break

    return (lower + upper) / 2.0


def compute_ebgm(df: pd.DataFrame, alpha, a1, b1, a2, b2) -> pd.DataFrame:
    """Compute EBGM and EB05 for all drug-PT pairs."""
    n = df["a"].values.astype(float)
    E = df["expected"].values

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

    # EB05 via vectorised bisection
    print("    Computing EB05 (vectorised bisection)...")
    eb05 = _eb05_bisection(n, E, a1, b1, a2, b2, Q)

    df["ebgm"] = ebgm
    df["eb05"] = eb05
    df["ebgm_Q"] = Q

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  BCPNN INFORMATION COMPONENT (Bate et al. 1998 — from P14)
# ═══════════════════════════════════════════════════════════════════════════


def compute_bcpnn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Information Component (IC) with credibility intervals.
    IC = log2( (a+0.5)(N+0.5) / ((n_drug+0.5)(n_reaction+0.5)) )
    Signal: IC025 > 0
    """
    a = df["a"].values.astype(float)
    n_drug = df["n_drug"].values.astype(float)
    n_rxn = df["n_reaction"].values.astype(float)
    N = df["N"].values.astype(float)

    ic = np.log2(((a + 0.5) * (N + 0.5)) / ((n_drug + 0.5) * (n_rxn + 0.5)))

    ic_var = (1.0 / np.log(2) ** 2) * (1.0 / (a + 0.5) - 1.0 / (N + 0.5))
    ic_se = np.sqrt(np.maximum(ic_var, 0))

    df["ic"] = ic
    df["ic025"] = ic - 1.96 * ic_se
    df["ic975"] = ic + 1.96 * ic_se

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL CLASSIFICATION AND FDR CORRECTION
# ═══════════════════════════════════════════════════════════════════════════


def classify_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply signal thresholds for each method."""
    df["signal_prr"] = (
        (df["prr"] >= PRR_THRESHOLD)
        & (df["prr_chi2"] >= PRR_CHI2_THRESHOLD)
        & (df["a"] >= MIN_REPORTS)
    )
    df["signal_ror"] = df["ror_lower95"] > ROR_LOWER_CI_THRESHOLD
    df["signal_ebgm"] = df["eb05"] >= EBGM_EB05_THRESHOLD
    df["signal_bcpnn"] = df["ic025"] > IC025_THRESHOLD

    df["n_methods_signal"] = (
        df["signal_prr"].astype(int)
        + df["signal_ror"].astype(int)
        + df["signal_ebgm"].astype(int)
        + df["signal_bcpnn"].astype(int)
    )

    return df


def apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to ROR p-values."""
    # Compute p-values from ROR (two-sided Wald test)
    ln_ror = np.log(df["ror"].values)
    a = df["a"].values.astype(float) + 0.5
    b = df["b"].values.astype(float) + 0.5
    c = df["c"].values.astype(float) + 0.5
    d = df["d"].values.astype(float) + 0.5
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)

    z = ln_ror / se
    p_values = 2 * stats.norm.sf(np.abs(z))

    # BH-FDR
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, n_tests + 1)

    fdr_threshold = FDR_ALPHA * ranks / n_tests
    df["ror_pvalue"] = p_values
    df["ror_fdr"] = p_values * n_tests / ranks
    df["ror_fdr"] = np.minimum.accumulate(df["ror_fdr"].values[np.argsort(-ranks)])[np.argsort(np.argsort(-ranks))]
    df["ror_fdr"] = np.minimum(df["ror_fdr"], 1.0)

    df["signal_ror_fdr"] = df["ror_fdr"] < FDR_ALPHA

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("  FAERS Disproportionality Analysis — Cobenfy")
    print("  PRR · ROR · EBGM (MGPS) · BCPNN (IC)")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"\n  ERROR: Database not found: {DB_PATH}")
        return

    con = duckdb.connect(str(DB_PATH), read_only=True)

    # ── Contingency tables ──────────────────────────────────────────────
    print("\n  Building contingency tables...")
    df = build_contingency_tables(con, drug="xanomeline-trospium")
    con.close()

    if len(df) == 0:
        print("\n  ERROR: No drug-PT pairs found. Check drug standardisation.")
        return

    # ── PRR ──────────────────────────────────────────────────────────────
    print("\n  Computing PRR...")
    df = compute_prr(df)

    # ── ROR ──────────────────────────────────────────────────────────────
    print("  Computing ROR...")
    df = compute_ror(df)

    # ── EBGM ─────────────────────────────────────────────────────────────
    print("\n  Fitting MGPS two-gamma mixture prior...")
    n_obs = df["a"].values.astype(int)
    E = df["expected"].values
    alpha, a1, b1, a2, b2, opt = fit_mgps_prior(n_obs, E)

    print(f"    Mixing weight (alpha): {alpha:.4f}")
    print(f"    Component 1: Gamma(a={a1:.4f}, b={b1:.4f})  mean={a1/b1:.3f}")
    print(f"    Component 2: Gamma(a={a2:.4f}, b={b2:.4f})  mean={a2/b2:.3f}")
    print(f"    Converged: {opt.success}  |  Final -LL: {opt.fun:,.0f}")

    print("  Computing EBGM and EB05...")
    df = compute_ebgm(df, alpha, a1, b1, a2, b2)

    # ── BCPNN ────────────────────────────────────────────────────────────
    print("\n  Computing IC (BCPNN)...")
    df = compute_bcpnn(df)

    # ── Classify signals ─────────────────────────────────────────────────
    print("  Applying signal thresholds...")
    df = classify_signals(df)

    # ── FDR correction ───────────────────────────────────────────────────
    print("  Applying Benjamini-Hochberg FDR correction...")
    df = apply_fdr(df)

    # ── Summary ──────────────────────────────────────────────────────────
    n_prr = df["signal_prr"].sum()
    n_ror = df["signal_ror"].sum()
    n_ebgm = df["signal_ebgm"].sum()
    n_bcpnn = df["signal_bcpnn"].sum()
    n_3plus = (df["n_methods_signal"] >= 3).sum()
    n_all4 = (df["n_methods_signal"] == 4).sum()

    print(f"\n{'=' * 70}")
    print(f"  SIGNAL DETECTION SUMMARY — COBENFY")
    print(f"{'=' * 70}")
    print(f"\n  Total drug-PT pairs analysed:          {len(df):>8,}")
    print(f"\n  PRR signals  (PRR≥2, χ²≥4, n≥3):      {n_prr:>8,}")
    print(f"  ROR signals  (lower 95% CI > 1):       {n_ror:>8,}")
    print(f"  EBGM signals (EB05 ≥ 2):               {n_ebgm:>8,}")
    print(f"  BCPNN signals (IC025 > 0):              {n_bcpnn:>8,}")
    print(f"  {'─' * 45}")
    print(f"  Flagged by ≥ 3 methods (SIGNAL):        {n_3plus:>8,}")
    print(f"  Flagged by ALL 4 methods:               {n_all4:>8,}")

    # Top signals (≥3 methods)
    signals = df[df["n_methods_signal"] >= 3].sort_values("ebgm", ascending=False)
    if len(signals) > 0:
        print(f"\n  TOP 30 SIGNALS (≥3 methods, ranked by EBGM)")
        print(f"  {'─' * 90}")
        header = f"  {'PT':<40s} {'n':>5s} {'E':>7s} {'ROR':>7s} {'PRR':>7s} {'EBGM':>6s} {'EB05':>5s} {'IC':>6s} {'#M':>3s}"
        print(header)
        print(f"  {'─' * 90}")
        for _, row in signals.head(30).iterrows():
            pt = str(row["pt"])[:39]
            print(
                f"  {pt:<40s} {row['a']:>5.0f} {row['expected']:>7.1f}"
                f" {row['ror']:>7.1f} {row['prr']:>7.1f}"
                f" {row['ebgm']:>6.1f} {row['eb05']:>5.1f}"
                f" {row['ic']:>6.2f} {row['n_methods_signal']:>3.0f}"
            )

    # Hypothesis-specific checks
    print(f"\n  HYPOTHESIS CHECKS")
    print(f"  {'─' * 60}")

    # H1: GI events
    gi_terms = ["NAUSEA", "VOMITING", "CONSTIPATION", "DYSPEPSIA", "DIARRHOEA", "DIARRHEA"]
    gi_signals = df[df["pt"].str.upper().isin(gi_terms)]
    print(f"\n  H1 — GI events:")
    for _, row in gi_signals.iterrows():
        status = "SIGNAL" if row["n_methods_signal"] >= 3 else "no signal"
        print(f"    {row['pt']:<30s} n={row['a']:.0f}  ROR={row['ror']:.1f}  [{status}]")

    # H2: CV events
    cv_terms = ["TACHYCARDIA", "HYPERTENSION", "BLOOD PRESSURE INCREASED"]
    cv_signals = df[df["pt"].str.upper().isin(cv_terms)]
    print(f"\n  H2 — Cardiovascular events:")
    for _, row in cv_signals.iterrows():
        status = "SIGNAL" if row["n_methods_signal"] >= 3 else "no signal"
        print(f"    {row['pt']:<30s} n={row['a']:.0f}  ROR={row['ror']:.1f}  [{status}]")

    # H3: Metabolic events (expect NO signal or low ROR)
    met_terms = ["WEIGHT INCREASED", "HYPERGLYCAEMIA", "HYPERGLYCEMIA",
                 "DIABETES MELLITUS", "METABOLIC SYNDROME"]
    met_signals = df[df["pt"].str.upper().isin(met_terms)]
    print(f"\n  H3 — Metabolic events (expect low/absent):")
    for _, row in met_signals.iterrows():
        status = "SIGNAL" if row["n_methods_signal"] >= 3 else "no signal"
        print(f"    {row['pt']:<30s} n={row['a']:.0f}  ROR={row['ror']:.1f}  [{status}]")

    # H4: EPS (expect NO signal)
    eps_terms = ["DYSTONIA", "AKATHISIA", "PARKINSONISM", "TARDIVE DYSKINESIA",
                 "EXTRAPYRAMIDAL DISORDER", "TREMOR"]
    eps_signals = df[df["pt"].str.upper().isin(eps_terms)]
    print(f"\n  H4 — Extrapyramidal symptoms (expect absent):")
    for _, row in eps_signals.iterrows():
        status = "SIGNAL" if row["n_methods_signal"] >= 3 else "no signal"
        print(f"    {row['pt']:<30s} n={row['a']:.0f}  ROR={row['ror']:.1f}  [{status}]")

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Saving results...")

    out_cols = [
        "pt", "a", "expected", "n_drug", "n_reaction", "N",
        "prr", "prr_lower95", "prr_upper95", "prr_chi2",
        "ror", "ror_lower95", "ror_upper95", "ror_pvalue", "ror_fdr",
        "ebgm", "eb05", "ebgm_Q",
        "ic", "ic025", "ic975",
        "signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn",
        "signal_ror_fdr", "n_methods_signal",
    ]

    # Full results
    full_path = OUTPUT_DIR / "disproportionality_cobenfy_full.csv"
    df.sort_values("n_methods_signal", ascending=False)[out_cols].to_csv(
        full_path, index=False, float_format="%.4f"
    )
    print(f"  {full_path.name:50s} {len(df):>6,} rows")

    # Consensus signals (≥3 methods)
    consensus = df[df["n_methods_signal"] >= 3].sort_values("ebgm", ascending=False)
    consensus_path = OUTPUT_DIR / "signals_cobenfy_consensus.csv"
    consensus[out_cols].to_csv(consensus_path, index=False, float_format="%.4f")
    print(f"  {consensus_path.name:50s} {len(consensus):>6,} rows")

    # All-4 signals
    all4 = df[df["n_methods_signal"] == 4].sort_values("ebgm", ascending=False)
    all4_path = OUTPUT_DIR / "signals_cobenfy_all4.csv"
    all4[out_cols].to_csv(all4_path, index=False, float_format="%.4f")
    print(f"  {all4_path.name:50s} {len(all4):>6,} rows")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DISPROPORTIONALITY ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/06_active_comparator.py")


if __name__ == "__main__":
    main()
