"""
Script 18: Compare primary (Q1 2026) vs extended (Q2 2026) for the robustness supplement.

Reads:
    outputs/tables/signals_cobenfy_consensus.csv          (primary, pre-registered)
    outputs_ext/tables/signals_cobenfy_consensus.csv      (extended, post-registration)
    plus disproportionality_cobenfy_full.csv from each for headline RORs.

Writes:
    outputs_ext/q2_robustness_comparison.csv    (per-signal retained/new/lost + ROR deltas)
    Console: concordance summary for the supplement text.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIM = ROOT / "outputs" / "tables"
EXT = ROOT / "outputs_ext" / "tables"
OUT = ROOT / "outputs_ext" / "q2_robustness_comparison.csv"

HEADLINE = ["NAUSEA", "VOMITING", "CONSTIPATION", "URINARY RETENTION", "DROOLING",
            "DRY MOUTH", "SALIVARY HYPERSECRETION", "WEIGHT INCREASED", "AKATHISIA",
            "TACHYCARDIA", "VISION BLURRED", "HYPERHIDROSIS"]


def sigset(base):
    df = pd.read_csv(base / "signals_cobenfy_consensus.csv")
    col = "pt" if "pt" in df.columns else df.columns[0]
    return set(df[col].str.upper())


def ror_map(base):
    df = pd.read_csv(base / "disproportionality_cobenfy_full.csv")
    c = [x for x in df.columns if x.lower() == "pt"][0]
    rc = [x for x in df.columns if x.lower() == "ror"][0]
    nc = [x for x in df.columns if x.lower() in ("a", "n")][0]
    df["PT"] = df[c].str.upper()
    return df.set_index("PT"), rc, nc


def main():
    prim, ext = sigset(PRIM), sigset(EXT)
    retained = sorted(prim & ext)
    lost = sorted(prim - ext)
    new = sorted(ext - prim)

    print("=" * 64)
    print("  Q2 2026 ROBUSTNESS: consensus-signal concordance")
    print("=" * 64)
    print(f"  Primary (Q1 2026) signals:     {len(prim)}")
    print(f"  Extended (Q2 2026) signals:    {len(ext)}")
    print(f"  Retained:                      {len(retained)} / {len(prim)} "
          f"({100*len(retained)/len(prim):.0f}%)")
    print(f"  Lost with extra quarter:       {len(lost)}  {lost}")
    print(f"  New with extra quarter:        {len(new)}  {new}")

    pd_, prc, pnc = ror_map(PRIM)
    ed_, erc, enc = ror_map(EXT)
    print("\n  Headline ROR (n)  primary -> extended:")
    rows = []
    for pt in HEADLINE:
        p = f"{pd_.loc[pt, prc]:.1f} (n={int(pd_.loc[pt, pnc])})" if pt in pd_.index else "—"
        e = f"{ed_.loc[pt, erc]:.1f} (n={int(ed_.loc[pt, enc])})" if pt in ed_.index else "—"
        print(f"    {pt:<24s} {p:>16s}  ->  {e}")
        rows.append({"pt": pt, "primary": p, "extended": e})

    # persist per-signal status
    allpts = sorted(prim | ext)
    status = [{"pt": pt,
               "in_primary": pt in prim,
               "in_extended": pt in ext,
               "status": ("retained" if pt in prim and pt in ext
                          else "new_in_extended" if pt in ext else "lost_in_extended")}
              for pt in allpts]
    pd.DataFrame(status).to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
