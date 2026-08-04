"""
Script 17: Q2 2026 post-registration robustness supplement.

Builds an ISOLATED extended-window analysis (Q4 2024 - Q2 2026, 7 quarters) to test
whether the pre-registered primary signals (Q4 2024 - Q1 2026) are stable when the
observation window is pushed to the most recent available quarter. This is an
EXPLICITLY POST-REGISTRATION sensitivity analysis for the supplement; it does NOT
touch the primary database or the primary outputs.

Isolation:
    - Extended parsed dir: data/raw/parsed_ext/  (symlinks to the 6 primary quarters
      + freshly fetched Q2 2026)
    - Extended DB:         data/processed/faers_ext.duckdb
    - Extended outputs:    outputs_ext/tables/
    Achieved via the FAERS_DB / FAERS_PARSED / FAERS_OUT env vars honoured by
    scripts 02, 03, 04, 05, 09 (defaults unchanged when the vars are unset).

Usage:
    python scripts/17_q2_robustness.py
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "raw" / "parsed"
PARSED_EXT = ROOT / "data" / "raw" / "parsed_ext"
EXT_DB = ROOT / "data" / "processed" / "faers_ext.duckdb"
OUT_EXT = ROOT / "outputs_ext"


def prepare_parsed_ext():
    """Symlink the 6 primary quarters + fetch/parse Q2 2026 into parsed_ext/."""
    PARSED_EXT.mkdir(parents=True, exist_ok=True)
    for f in PARSED.glob("*.csv"):
        link = PARSED_EXT / f.name
        if not link.exists():
            link.symlink_to(f)

    # Q2 2026 already parsed?
    if list(PARSED_EXT.glob("demo_2026Q2.csv")):
        print("  Q2 2026 already parsed into parsed_ext/")
        return

    # Load script 01 and redirect its parse output to parsed_ext/
    spec = importlib.util.spec_from_file_location("s01", ROOT / "scripts" / "01_download_faers.py")
    s01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s01)
    s01.PARSED_DIR = PARSED_EXT   # extract_table() reads this module global at call time

    print("  Fetching Q2 2026 ...")
    zip_path = s01.download_quarter(2026, 2)
    if not zip_path:
        print("  ERROR: Q2 2026 download failed.")
        sys.exit(1)
    results = s01.extract_all_tables(zip_path, 2026, 2)
    print(f"  Q2 2026 extracted: {sum(results.values())}/{len(results)} tables")


def run_pipeline():
    env = {**os.environ,
           "FAERS_DB": str(EXT_DB),
           "FAERS_PARSED": str(PARSED_EXT),
           "FAERS_OUT": str(OUT_EXT)}
    for s in ["02_load_deduplicate", "03_drug_standardisation",
              "04_case_identification", "05_disproportionality", "09_mgps_refit"]:
        print(f"\n=== [ext] {s} ===", flush=True)
        r = subprocess.run([sys.executable, f"scripts/{s}.py"], env=env, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  ERROR: {s} failed (exit {r.returncode})")
            sys.exit(1)


def main():
    print("=" * 68)
    print("  Q2 2026 ROBUSTNESS SUPPLEMENT (isolated extended window)")
    print("=" * 68)
    prepare_parsed_ext()
    run_pipeline()
    print("\n  Extended run complete.")
    print(f"  Extended consensus: {OUT_EXT / 'tables' / 'signals_cobenfy_consensus.csv'}")
    print("  Compare with primary via scripts/18_q2_compare.py")


if __name__ == "__main__":
    main()
