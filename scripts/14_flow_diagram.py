"""
Script 14: Generate the case-selection flow diagram for the manuscript.

Emits a Mermaid flowchart and renders it to PNG and PDF. Mermaid is the
portfolio standard for flowcharts; matplotlib is reserved for data-driven
figures. Hand-placing boxes and connectors in matplotlib had produced
misaligned arrows and a large amount of dead white space.

Counts are read from the database at build time, never hardcoded, so the
diagram cannot drift from the analysis.

Note on the exclusion step. The previous version reported the excluded cases
as `count(role_cod = 'C')`, labelled "concomitant role only". That query
counts every case carrying a concomitant record, including cases that ALSO
carry a suspect record, so it both mislabelled the quantity and failed to
balance: 1,779 - 16 != 1,758. The exclusion is now the complement of the
PS/SS set, broken down by role, and the balance is asserted before rendering.

Usage:
    python scripts/14_flow_diagram.py

Requires: data/processed/faers.duckdb, mermaid-cli (mmdc) on PATH.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
SUPP_DIR = PROJECT_ROOT / "outputs" / "supplementary"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DRUG = "xanomeline-trospium"
COMPARATORS = ["olanzapine", "risperidone", "aripiprazole",
               "quetiapine", "lurasidone", "brexpiprazole"]

# Okabe-Ito, the portfolio's colourblind-safe palette. Light fills with
# saturated strokes so the boxes stay legible in greyscale print.
PROCESS_FILL, PROCESS_LINE = "#E4F0F8", "#0072B2"
EXCLUDE_FILL, EXCLUDE_LINE = "#FBEAE1", "#D55E00"
RESULT_FILL, RESULT_LINE = "#E2F2EC", "#009E73"
ASIDE_FILL, ASIDE_LINE = "#F7EAF1", "#CC79A7"


def get_counts(con):
    """Case counts at each processing stage, read from the database."""
    def q(sql):
        return con.execute(sql).fetchone()[0]

    c = {
        "total_dedup": q("SELECT count(DISTINCT primaryid) FROM demo"),
        "with_drug": q("SELECT count(DISTINCT primaryid) FROM drug_std"),
        "all_roles": q(f"SELECT count(DISTINCT primaryid) FROM drug_std "
                       f"WHERE std_drug = '{DRUG}'"),
        "ps_ss": q(f"SELECT count(DISTINCT primaryid) FROM drug_std "
                   f"WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS','SS')"),
        "ps": q(f"SELECT count(DISTINCT primaryid) FROM drug_std "
                f"WHERE std_drug = '{DRUG}' AND UPPER(role_cod) = 'PS'"),
        "ss": q(f"SELECT count(DISTINCT primaryid) FROM drug_std "
                f"WHERE std_drug = '{DRUG}' AND UPPER(role_cod) = 'SS'"),
    }

    # Excluded = has a record for the index drug but never as primary or
    # secondary suspect. Broken down by role so the arithmetic is auditable.
    by_role = dict(con.execute(f"""
        SELECT UPPER(role_cod) AS rc, count(DISTINCT primaryid) AS n
        FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND primaryid NOT IN (
              SELECT primaryid FROM drug_std
              WHERE std_drug = '{DRUG}' AND UPPER(role_cod) IN ('PS','SS'))
        GROUP BY 1
    """).fetchall())
    c["excl_concomitant"] = by_role.get("C", 0)
    c["excl_interacting"] = by_role.get("I", 0)
    c["excl_total"] = c["all_roles"] - c["ps_ss"]
    c["not_drug"] = c["with_drug"] - c["all_roles"]

    c["comparators"] = {
        d: q(f"SELECT count(DISTINCT primaryid) FROM drug_std "
             f"WHERE std_drug = '{d}' AND UPPER(role_cod) IN ('PS','SS')")
        for d in COMPARATORS
    }

    dis = pd.read_csv(TABLE_DIR / "disproportionality_cobenfy_full.csv")
    c["pairs"] = len(dis)
    c["consensus"] = int((dis["n_methods_signal"] >= 3).sum())
    c["all_four"] = int((dis["n_methods_signal"] == 4).sum())

    cls = pd.read_csv(SUPP_DIR / "signal_classification_final.csv")
    vc = cls["classification"].value_counts()
    c["pharmacological"] = int(vc.get("Pharmacological", 0))
    c["disease"] = int(vc.get("Disease manifestation", 0))
    c["indeterminate"] = int(vc.get("Indeterminate", 0))

    _assert_balances(c)
    return c


def _assert_balances(c):
    """Fail loudly rather than render a flow diagram that does not add up."""
    checks = [
        ("suspect-role exclusion balances",
         c["all_roles"] - c["excl_total"] == c["ps_ss"]),
        ("exclusion breakdown sums to total",
         c["excl_concomitant"] + c["excl_interacting"] == c["excl_total"]),
        ("non-index cases balance",
         c["with_drug"] - c["not_drug"] == c["all_roles"]),
        ("classification sums to consensus",
         c["pharmacological"] + c["disease"] + c["indeterminate"] == c["consensus"]),
        ("all-four is a subset of consensus", c["all_four"] <= c["consensus"]),
    ]
    bad = [name for name, ok in checks if not ok]
    if bad:
        sys.exit("flow diagram counts do not balance: " + "; ".join(bad))


def build_mermaid(c):
    """Vertical spine, exclusions to one side, comparator panel to the other."""
    comp = c["comparators"]
    # Two per line keeps the aside compact; a tall narrow panel widens the
    # whole graph and pushes the main spine off-centre.
    ordered = sorted(COMPARATORS, key=lambda d: -comp[d])
    pairs = [ordered[i:i + 2] for i in range(0, len(ordered), 2)]
    comp_lines = "<br/>".join(
        " &nbsp;&middot;&nbsp; ".join(f"{d.capitalize()} {comp[d]:,}" for d in row)
        for row in pairs)

    return f"""---
config:
  theme: base
  themeVariables:
    fontFamily: Helvetica, Arial, sans-serif
    fontSize: 15px
    lineColor: "#55606B"
    edgeLabelBackground: "#FFFFFF"
  flowchart:
    nodeSpacing: 34
    rankSpacing: 42
    padding: 12
    useMaxWidth: false
    curve: linear
    wrappingWidth: 320
---
flowchart TD
    A["<b>FAERS quarterly ASCII files</b><br/>Q4 2024 to Q1 2026 (6 quarters)"]
    B["<b>Deduplicated by CASEID</b><br/>most recent report version retained<br/><b>n = {c['total_dedup']:,}</b> unique cases"]
    C["<b>Drug names standardised (RxNorm)</b><br/>6 xanomeline-trospium name variants mapped<br/><b>n = {c['with_drug']:,}</b> cases with drug records"]
    D["<b>Xanomeline-trospium cases</b><br/>any role code &nbsp;&middot;&nbsp; <b>n = {c['all_roles']:,}</b>"]
    E["<b>Primary or secondary suspect</b><br/><b>n = {c['ps_ss']:,}</b> analytic cohort<br/>PS {c['ps']:,} &nbsp;&middot;&nbsp; SS {c['ss']:,}"]
    F["<b>Drug-event pairs with n &ge; 3</b><br/><b>{c['pairs']}</b> preferred terms tested<br/>four-method disproportionality battery"]
    G["<b>{c['consensus']} consensus signals</b><br/>positive on &ge; 3 of 4 methods ({c['all_four']} on all four)<br/>{c['pharmacological']} pharmacological &nbsp;&middot;&nbsp; {c['disease']} disease &nbsp;&middot;&nbsp; {c['indeterminate']} indeterminate"]

    X1["Not xanomeline-trospium<br/><b>n = {c['not_drug']:,}</b>"]
    X2["No suspect role &nbsp;&middot;&nbsp; <b>n = {c['excl_total']}</b><br/>concomitant only {c['excl_concomitant']} &nbsp;&middot;&nbsp; interacting only {c['excl_interacting']}"]
    P["<b>Active comparators</b> (PS or SS)<br/>{comp_lines}"]

    A --> B --> C --> D --> E --> F --> G
    P -. head-to-head .-> F
    C -- excluded --> X1
    D -- excluded --> X2

    classDef process fill:{PROCESS_FILL},stroke:{PROCESS_LINE},stroke-width:1.4px,color:#10242E
    classDef exclude fill:{EXCLUDE_FILL},stroke:{EXCLUDE_LINE},stroke-width:1.2px,color:#3A1D0C
    classDef result  fill:{RESULT_FILL},stroke:{RESULT_LINE},stroke-width:1.8px,color:#0C2B22
    classDef aside   fill:{ASIDE_FILL},stroke:{ASIDE_LINE},stroke-width:1.2px,color:#331020

    class A,B,C,D,E,F process
    class X1,X2 exclude
    class G result
    class P aside
"""


def _puppeteer_config():
    """Point mermaid-cli at whatever headless Chrome is actually installed.

    mmdc pins an exact Chrome build and errors if only a different one is
    present, which is the common case after `puppeteer browsers install`.
    Passing an explicit executablePath avoids a version-pin failure that has
    nothing to do with the diagram.
    """
    cache = Path.home() / ".cache" / "puppeteer"
    candidates = sorted(cache.glob("chrome-headless-shell/*/*/chrome-headless-shell"))
    candidates += sorted(cache.glob("chrome/*/*/Google Chrome for Testing.app/"
                                    "Contents/MacOS/Google Chrome for Testing"))
    if not candidates:
        sys.exit("no headless Chrome found for mermaid-cli. Install one with:\n"
                 "  npx puppeteer browsers install chrome-headless-shell")
    cfg = FIG_DIR / ".puppeteer.json"
    cfg.write_text(json.dumps(
        {"executablePath": str(candidates[-1]), "args": ["--no-sandbox"]}))
    return cfg


def render(mmd_path):
    if shutil.which("mmdc") is None:
        sys.exit("mermaid-cli (mmdc) not found on PATH; cannot render the flow diagram")
    cfg = _puppeteer_config()
    png = FIG_DIR / "fig0_flow_diagram.png"
    pdf = FIG_DIR / "fig0_flow_diagram.pdf"
    base = ["mmdc", "-i", str(mmd_path), "-b", "white", "-p", str(cfg)]
    subprocess.run(base + ["-s", "3", "-o", str(png)], check=True, capture_output=True)
    subprocess.run(base + ["-o", str(pdf), "--pdfFit"], check=True, capture_output=True)
    cfg.unlink(missing_ok=True)
    return png, pdf


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    counts = get_counts(con)

    mmd = FIG_DIR / "fig0_flow_diagram.mmd"
    mmd.write_text(build_mermaid(counts))
    png, pdf = render(mmd)

    (FIG_DIR / "fig0_flow_diagram_counts.json").write_text(
        json.dumps(counts, indent=2))

    print("  Case-selection flow diagram (Mermaid)")
    print(f"    balance: {counts['all_roles']:,} - {counts['excl_total']} "
          f"= {counts['ps_ss']:,}  "
          f"({counts['excl_concomitant']} concomitant + "
          f"{counts['excl_interacting']} interacting)")
    for p in (mmd, png, pdf):
        print(f"    -> {p.name}")


if __name__ == "__main__":
    main()
