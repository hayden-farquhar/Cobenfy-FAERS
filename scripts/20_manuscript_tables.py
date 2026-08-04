"""
Script 20: Regenerate manuscript Tables 1-5 (markdown) from the 6-quarter outputs.
Table 2 drops the E-value column (revision decision D2). Writes to
outputs/tables/manuscript_tables.md for slotting into the SSOT manuscript.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "outputs" / "tables"
S = ROOT / "outputs" / "supplementary"
OUT = T / "manuscript_tables.md"

DRUGS = ["xanomeline-trospium", "olanzapine", "risperidone", "aripiprazole",
         "quetiapine", "lurasidone", "brexpiprazole"]
LABEL = {"xanomeline-trospium": "Xanomeline-trospium", "olanzapine": "Olanzapine",
         "risperidone": "Risperidone", "aripiprazole": "Aripiprazole",
         "quetiapine": "Quetiapine", "lurasidone": "Lurasidone", "brexpiprazole": "Brexpiprazole"}

GROUP = {
 "Gastrointestinal": ["NAUSEA","VOMITING","VOMITING PROJECTILE","CONSTIPATION","DYSPEPSIA",
   "GASTROOESOPHAGEAL REFLUX DISEASE","RETCHING","HICCUPS","GASTROINTESTINAL DISORDER",
   "ABDOMINAL DISCOMFORT","DYSPHAGIA"],
 "Urogenital / anticholinergic": ["URINARY RETENTION","URINARY INCONTINENCE","DYSURIA","ANURIA",
   "DRY MOUTH","SALIVARY HYPERSECRETION","DROOLING"],
 "Cardiovascular": ["TACHYCARDIA","HEART RATE INCREASED"],
 "Neurological / autonomic": ["VISION BLURRED","HYPERHIDROSIS","TREMOR","SOMNOLENCE","SEDATION",
   "SEDATION COMPLICATION","HYPERSOMNIA","COLD SWEAT","NIGHT SWEATS",
   "FEELING OF BODY TEMPERATURE CHANGE"],
 "Other": ["ANGIOEDEMA"],
}


def t1():
    dem = pd.read_csv(T / "demographics.csv").set_index("drug")
    out = pd.read_csv(S / "outcome_severity.csv")
    out["drug"] = out["drug"].str.lower().str.replace("cobenfy", "xanomeline-trospium")
    out = out.set_index("drug")
    L = ["### Table 1. Baseline characteristics of xanomeline-trospium and active comparator cases in FAERS (Q4 2024--Q1 2026)", ""]
    hdr = "| Characteristic | " + " | ".join(f"{LABEL[d]} (n={int(dem.loc[d,'n_cases']):,})" for d in DRUGS) + " |"
    L += [hdr, "|" + "---|" * (len(DRUGS) + 1)]
    def row(name, fn):
        return "| " + name + " | " + " | ".join(fn(d) for d in DRUGS) + " |"
    def age(d):
        r = dem.loc[d]; return f"{r.age_median:.0f} ({r.age_q25:.0f}--{r.age_q75:.0f})"
    def agavail(d):
        r = dem.loc[d]; a = r.n_cases - r.age_missing; return f"{int(a):,} ({100*a/r.n_cases:.1f})"
    def fem(d):
        r = dem.loc[d]; return f"{int(r.female_n):,} ({100*r.female_n/r.n_cases:.1f})"
    def male(d):
        r = dem.loc[d]; return f"{int(r.male_n):,} ({100*r.male_n/r.n_cases:.1f})"
    def us(d):
        r = dem.loc[d]; return f"{int(r.us_reports):,} ({100*r.us_reports/r.n_cases:.1f})"
    def hcp(d):
        r = dem.loc[d]; return f"{int(r.hcp_reports):,} ({100*r.hcp_reports/r.n_cases:.1f})"
    def death(d):
        r = out.loc[d]; return f"{int(r.death):,} ({r.pct_death:.1f})"
    def hosp(d):
        r = out.loc[d]; return f"{int(r.hospitalisation):,} ({r.pct_hosp:.1f})"
    L.append(row("Age, median (IQR), y", age))
    L.append(row("Age available, n (%)", agavail))
    L.append(row("Female, n (%)", fem))
    L.append(row("Male, n (%)", male))
    L.append(row("US reports, n (%)", us))
    L.append(row("Healthcare professional, n (%)", hcp))
    L.append(row("Death, n (%)", death))
    L.append(row("Hospitalisation, n (%)", hosp))
    return "\n".join(L)


def t2():
    full = pd.read_csv(T / "disproportionality_cobenfy_full.csv"); full["PT"] = full["pt"].str.upper()
    cls = pd.read_csv(S / "signal_classification_final.csv"); cls["PT"] = cls["pt"].str.upper()
    pharm = set(cls[cls["classification"] == "Pharmacological"]["PT"])
    f = full.set_index("PT")
    L = ["### Table 2. Pharmacological consensus disproportionality signals for xanomeline-trospium",
         "",
         "| Preferred Term | n | ROR (95% CI) | PRR (chi-sq) | EBGM (EB05) | IC (IC025) | Methods |",
         "|---|---|---|---|---|---|---|"]
    seen = set()
    for grp, pts in GROUP.items():
        rows = [pt for pt in pts if pt in pharm and pt in f.index]
        if not rows:
            continue
        L.append(f"| **{grp}** | | | | | | |")
        for pt in sorted(rows, key=lambda p: -f.loc[p, "ror"]):
            r = f.loc[pt]; seen.add(pt)
            L.append(f"| {pt.title()} | {int(r.a)} | {r.ror:.1f} ({r.ror_lower95:.1f}--{r.ror_upper95:.1f}) "
                     f"| {r.prr:.1f} ({r.prr_chi2:.0f}) | {r.ebgm:.1f} ({r.eb05:.1f}) "
                     f"| {r.ic:.2f} ({r.ic025:.2f}) | {int(r.n_methods_signal)}/4 |")
    missing = pharm - seen
    if missing:
        L.append(f"\n<!-- ungrouped pharmacological PTs (add to a group): {sorted(missing)} -->")
    return "\n".join(L)


def t3():
    ac = pd.read_csv(T / "active_comparator_results.csv"); ac["PT"] = ac["pt"].str.upper()
    comps = ["olanzapine", "risperidone", "aripiprazole", "quetiapine", "lurasidone", "brexpiprazole"]
    higher = ["NAUSEA","VOMITING","CONSTIPATION","DYSPEPSIA","URINARY RETENTION","DRY MOUTH","VISION BLURRED"]
    lower = ["WEIGHT INCREASED","SOMNOLENCE","SEDATION","AKATHISIA","DYSTONIA","PARKINSONISM",
             "EXTRAPYRAMIDAL DISORDER","TARDIVE DYSKINESIA"]
    def cell(pt, comp):
        r = ac[(ac.PT == pt) & (ac.drug_b == comp)]
        if len(r) == 0:
            return "--"
        r = r.iloc[0]
        s = f"{r.ror:.2f}"
        return f"**{s}\\***" if r.bonferroni_sig else s
    L = ["### Table 3. Active-comparator reporting odds ratios: xanomeline-trospium versus D2 antagonists for key adverse events",
         "",
         "| Preferred Term | vs Olanzapine | vs Risperidone | vs Aripiprazole | vs Quetiapine | vs Lurasidone | vs Brexpiprazole |",
         "|---|---|---|---|---|---|---|"]
    L.append("| **Higher for xanomeline-trospium** | | | | | | |")
    for pt in higher:
        L.append(f"| {pt.title()} | " + " | ".join(cell(pt, c) for c in comps) + " |")
    L.append("| **Lower for xanomeline-trospium** | | | | | | |")
    for pt in lower:
        L.append(f"| {pt.title()} | " + " | ".join(cell(pt, c) for c in comps) + " |")
    L.append("")
    L.append("Bold with asterisk (\\*) = Bonferroni-significant (p < 0.00026). Values are reporting odds ratios (xanomeline-trospium / comparator).")
    return "\n".join(L)


def t5():
    w = pd.read_csv(T / "time_to_onset_weibull.csv")
    L = ["### Table 5. Time-to-onset analysis: Weibull parameters for key adverse events",
         "",
         "| Preferred Term | n | Median TTO (days) | Weibull beta | Weibull scale | % onset <30d |",
         "|---|---|---|---|---|---|"]
    for _, r in w.sort_values("n_reports", ascending=False).iterrows():
        L.append(f"| {r['pt']} | {int(r['n_fitted'])} | {r['median_tto']:.1f} | {r['shape']:.2f} "
                 f"| {r['scale']:.1f} | {r['pct_early']:.1f} |")
    L.append("")
    L.append("All shape parameters (beta) < 1 indicate early-onset events with decreasing hazard.")
    return "\n".join(L)


def main():
    parts = [t1(), "", t2(), "", t3(), "", t5()]
    OUT.write_text("\n".join(parts) + "\n")
    print(f"Wrote {OUT}")
    print("\n--- preview ---\n")
    print("\n".join(parts)[:2500])


if __name__ == "__main__":
    main()
