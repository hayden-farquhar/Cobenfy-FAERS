"""
Script 19: Finalise the signal classification from the expert-adjudicated ratings.

The blinded expert rating (submissions/.../revision/signal_rating_form.md) is taken as
the FINAL three-category classification (Pharmacological / Disease manifestation /
Indeterminate). This supersedes the pre-specified rule-based classification, which is
retained only as the comparator for the reliability statistic.

Outputs:
    outputs/supplementary/signal_classification_final.csv     (machine-readable, all metrics)
    outputs/supplementary/signal_classification_final_table.md (formatted eTable for R1.3)
    outputs/supplementary/classification_reliability_summary.md (rule-based vs expert, kappa)
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORM = ROOT / "submissions/schizophrenia-bulletin/revision/signal_rating_form.md"
FULL = ROOT / "outputs/tables/disproportionality_cobenfy_full.csv"
RULE = ROOT / "outputs/supplementary/signal_classification_disease_vs_drug.csv"
SUPP = ROOT / "outputs/supplementary"

LABEL = {"P": "Pharmacological", "D": "Disease manifestation", "I": "Indeterminate"}


def parse_form(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) < 5 or not c[0].isdigit():
            continue
        rows.append({"pt": c[1].upper(), "rating": c[4].strip().upper(),
                     "notes": c[5] if len(c) > 5 else ""})
    return pd.DataFrame(rows)


def cohens_kappa(a, b, cats):
    n = len(a); idx = {c: i for i, c in enumerate(cats)}; k = len(cats)
    obs = [[0]*k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    po = sum(obs[i][i] for i in range(k))/n
    row = [sum(obs[i]) for i in range(k)]; col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i]/n)*(col[i]/n) for i in range(k))
    return (po-pe)/(1-pe), po, pe, obs


def main():
    form = parse_form(FORM)
    full = pd.read_csv(FULL); full["PT"] = full["pt"].str.upper()
    m = form.merge(full, left_on="pt", right_on="PT", how="left", suffixes=("", "_f"))

    m["classification"] = m["rating"].map(LABEL)
    out = m[["pt", "a", "ror", "ror_lower95", "ror_upper95", "prr", "prr_chi2",
             "ebgm", "eb05", "ic", "ic025", "n_methods_signal",
             "classification", "notes"]].rename(columns={"a": "n"})
    out = out.sort_values(["classification", "n"], ascending=[True, False])
    out.to_csv(SUPP / "signal_classification_final.csv", index=False)

    counts = m["classification"].value_counts()
    print("FINAL CLASSIFICATION (expert-adjudicated):")
    for k in ["Pharmacological", "Disease manifestation", "Indeterminate"]:
        print(f"  {k:<22s}: {counts.get(k,0)}")

    # --- Supplementary markdown table (R1.3) ---
    L = ["## eTable. Consensus disproportionality signals with final classification",
         "",
         f"All {len(m)} consensus signals (positivity on ≥3 of 4 methods) for xanomeline-trospium, "
         "with the expert-adjudicated classification. ROR with 95% CI; n = reports.",
         "",
         "| Preferred term | n | ROR (95% CI) | Methods | Classification |",
         "|---|---|---|---|---|"]
    for _, r in out.iterrows():
        L.append(f"| {r['pt'].title()} | {int(r['n'])} | {r['ror']:.1f} "
                 f"({r['ror_lower95']:.1f}–{r['ror_upper95']:.1f}) | "
                 f"{int(r['n_methods_signal'])}/4 | {r['classification']} |")
    (SUPP / "signal_classification_final_table.md").write_text("\n".join(L) + "\n")

    # --- Reliability: rule-based vs expert ---
    rule = pd.read_csv(RULE)
    rule["occ1"] = rule["disease_manifestation"].map({True: "D", False: "P"})
    o1 = dict(zip(rule["pt"].str.upper(), rule["occ1"]))
    rel = form.copy()
    rel["occ1"] = rel["pt"].map(o1)
    rel = rel.dropna(subset=["occ1"])
    kappa, po, pe, obs = cohens_kappa(list(rel["occ1"]), list(rel["rating"]), ["P", "D", "I"])

    R = ["## Classification reliability: pre-specified rule-based vs expert adjudication",
         "",
         f"The pre-specified rule-based classification (keyword lookup) was compared against an "
         f"independent, blinded expert re-adjudication of all {len(rel)} consensus signals. The expert "
         f"classification, which additionally permitted an indeterminate category, is taken as final.",
         "",
         f"- Observed agreement: {po:.1%}",
         f"- Cohen's kappa (three categories P/D/I): {kappa:.2f}",
         "",
         "Confusion matrix (rows = rule-based, columns = expert), categories P / D / I:",
         "",
         "| rule-based \\ expert | P | D | I |",
         "|---|---|---|---|",
         f"| P | {obs[0][0]} | {obs[0][1]} | {obs[0][2]} |",
         f"| D | {obs[1][0]} | {obs[1][1]} | {obs[1][2]} |",
         f"| I | {obs[2][0]} | {obs[2][1]} | {obs[2][2]} |",
         "",
         "Most disagreements reflect two expected sources: newer preferred terms (from the added "
         "reporting quarter) that the pre-specified rule set did not enumerate and therefore defaulted "
         "to pharmacological, and reassignment of mechanistically ambiguous terms to the indeterminate "
         "category. The expert adjudication is used throughout.",
         ""]
    (SUPP / "classification_reliability_summary.md").write_text("\n".join(R))
    print(f"\n  Cohen's kappa (rule-based vs expert): {kappa:.2f}  (obs agr {po:.1%})")
    print(f"  Saved: signal_classification_final.csv, _table.md, classification_reliability_summary.md")


if __name__ == "__main__":
    main()
