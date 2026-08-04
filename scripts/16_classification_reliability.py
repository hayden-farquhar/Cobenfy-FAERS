"""
Script 16: Intra-rater reliability for the signal classification (R3.5, R2.4, R3.3).

Design (per revision decision D1 = intra-rater):
    Occasion 1 = the original classification in
        outputs/supplementary/signal_classification_disease_vs_drug.csv
        (two categories: Pharmacological / Disease manifestation).
    Occasion 2 = the author's blinded re-rating in the completed rating form
        submissions/schizophrenia-bulletin/revision/signal_rating_form.md
        (three categories permitted: P / D / I = indeterminate, per R2.4),
        performed blinded to the original assignments.

    Cohen's kappa is computed over the union of category labels {P, D, I}.
    The re-rating (occasion 2) becomes the FINAL classification going into the
    revised paper; the kappa quantifies intra-rater reproducibility.

This script does NOT invent ratings. It exits with a clear message until the
worksheet's `blinded_rating` column is filled in by the author.

Usage:
    python scripts/16_classification_reliability.py
"""

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "outputs/supplementary/signal_classification_disease_vs_drug.csv"
FORM = ROOT / "submissions/schizophrenia-bulletin/revision/signal_rating_form.md"
OUT = ROOT / "outputs/supplementary/revision_classification_reliability.csv"

CODE = {"P": "Pharmacological", "D": "Disease manifestation", "I": "Indeterminate"}


def parse_form(path):
    """Parse the markdown rating form -> DataFrame[pt, blinded_rating].

    Reads the data rows of the '| # | Preferred term | n | ROR | Rating | Notes |'
    table: takes the preferred term (col 2) and the rating (col 5).
    """
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        idx, pt, rating = cells[0], cells[1], cells[4]
        if not idx.isdigit():   # skip header and separator rows
            continue
        rows.append({"pt": pt, "blinded_rating": rating})
    return pd.DataFrame(rows)


def cohens_kappa(labels_a, labels_b, categories):
    """Unweighted Cohen's kappa over a fixed category set."""
    n = len(labels_a)
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        obs[idx[a]][idx[b]] += 1
    po = sum(obs[i][i] for i in range(k)) / n
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return kappa, po, pe, obs


def main():
    orig = pd.read_csv(ORIG)
    ws = parse_form(FORM)

    # Occasion-1 labels collapsed to P/D codes
    orig["occ1"] = orig["disease_manifestation"].map({True: "D", False: "P"})
    o1 = dict(zip(orig["pt"].str.upper(), orig["occ1"]))

    ws["blinded_rating"] = ws["blinded_rating"].astype(str).str.strip().str.upper()
    unfilled = ws[~ws["blinded_rating"].isin(CODE.keys())]
    if len(unfilled):
        print(f"  {len(unfilled)}/{len(ws)} rows not yet rated (need P/D/I).")
        print("  Fill the `Rating (P/D/I)` column in the form, then re-run:")
        print(f"    {FORM}")
        print("\n  Unrated PTs:")
        for pt in unfilled["pt"].head(60):
            print(f"    - {pt}")
        sys.exit(1)

    merged = ws.copy()
    merged["occ1"] = merged["pt"].str.upper().map(o1)
    merged["occ2"] = merged["blinded_rating"]

    cats = ["P", "D", "I"]
    kappa, po, pe, obs = cohens_kappa(list(merged["occ1"]), list(merged["occ2"]), cats)

    print("=" * 60)
    print("  INTRA-RATER RELIABILITY (signal classification)")
    print("=" * 60)
    print(f"  n PTs               : {len(merged)}")
    print(f"  Observed agreement  : {po:.3f}")
    print(f"  Expected agreement  : {pe:.3f}")
    print(f"  Cohen's kappa       : {kappa:.3f}")
    print("\n  Confusion matrix (rows=occasion1, cols=occasion2) [P D I]:")
    for i, c in enumerate(cats):
        print(f"    {c}: {obs[i]}")

    disagree = merged[merged["occ1"] != merged["occ2"]]
    print(f"\n  Disagreements ({len(disagree)}):")
    for _, r in disagree.iterrows():
        print(f"    {r['pt']:<32s} occ1={r['occ1']} -> occ2={r['occ2']}")

    merged.to_csv(OUT, index=False)
    print(f"\n  Saved: {OUT}")


if __name__ == "__main__":
    main()
