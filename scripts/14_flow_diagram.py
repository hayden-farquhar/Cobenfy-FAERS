"""
Script 14: Generate case-selection flow diagram for manuscript.

Produces a PRISMA-style flow diagram showing the data processing pipeline
from raw FAERS downloads through to the final analytic cohort.

Usage:
    python scripts/14_flow_diagram.py

Requires: data/processed/faers.duckdb
"""

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "faers.duckdb"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DRUG = "xanomeline-trospium"


def get_counts(con):
    """Extract case counts at each processing stage."""
    counts = {}

    # Total cases in database (after dedup)
    counts["total_dedup"] = con.execute(
        "SELECT count(DISTINCT primaryid) FROM demo"
    ).fetchone()[0]

    # Total with any drug record
    counts["with_drug"] = con.execute(
        "SELECT count(DISTINCT primaryid) FROM drug_std"
    ).fetchone()[0]

    # Cobenfy - all roles
    counts["cobenfy_all"] = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}'
    """).fetchone()[0]

    # Cobenfy PS+SS
    counts["cobenfy_ps_ss"] = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND UPPER(role_cod) IN ('PS', 'SS')
    """).fetchone()[0]

    # Cobenfy PS only
    counts["cobenfy_ps"] = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND UPPER(role_cod) = 'PS'
    """).fetchone()[0]

    # Cobenfy SS only
    counts["cobenfy_ss"] = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND UPPER(role_cod) = 'SS'
    """).fetchone()[0]

    # Cobenfy concomitant only
    counts["cobenfy_concom"] = con.execute(f"""
        SELECT count(DISTINCT primaryid) FROM drug_std
        WHERE std_drug = '{DRUG}'
          AND UPPER(role_cod) = 'C'
    """).fetchone()[0]

    # Drug-PT pairs tested
    counts["drug_pt_pairs"] = con.execute(f"""
        SELECT count(DISTINCT UPPER(r.pt))
        FROM reac r
        INNER JOIN drug_std d ON r.primaryid = d.primaryid
        WHERE d.std_drug = '{DRUG}'
          AND UPPER(d.role_cod) IN ('PS', 'SS')
        GROUP BY UPPER(r.pt)
        HAVING count(DISTINCT r.primaryid) >= 3
    """).fetchdf().shape[0]

    # Comparator counts
    for comp in ["olanzapine", "risperidone", "aripiprazole",
                 "quetiapine", "lurasidone", "brexpiprazole"]:
        counts[f"comp_{comp}"] = con.execute(f"""
            SELECT count(DISTINCT primaryid) FROM drug_std
            WHERE std_drug = '{comp}'
              AND UPPER(role_cod) IN ('PS', 'SS')
        """).fetchone()[0]

    # Quarters
    counts["quarters"] = con.execute("""
        SELECT count(DISTINCT
            SUBSTRING(fda_dt, 1, 4) || 'Q' ||
            CASE
                WHEN CAST(SUBSTRING(fda_dt, 5, 2) AS INTEGER) <= 3 THEN '1'
                WHEN CAST(SUBSTRING(fda_dt, 5, 2) AS INTEGER) <= 6 THEN '2'
                WHEN CAST(SUBSTRING(fda_dt, 5, 2) AS INTEGER) <= 9 THEN '3'
                ELSE '4'
            END)
        FROM demo
        WHERE fda_dt IS NOT NULL AND LENGTH(fda_dt) >= 6
    """).fetchone()[0]

    return counts


def draw_flow_diagram(counts):
    """Draw the case-selection flow diagram."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Box style
    box_main = dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                    edgecolor="#1565C0", linewidth=1.5)
    box_excl = dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                    edgecolor="#E65100", linewidth=1)
    box_final = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                     edgecolor="#2E7D32", linewidth=1.5)
    box_comp = dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5",
                    edgecolor="#6A1B9A", linewidth=1)

    # Title
    ax.text(5, 13.5, "Case Selection Flow Diagram",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Box 1: FAERS download
    ax.text(5, 12.5,
            f"FAERS quarterly ASCII files downloaded\n"
            f"Q4 2024 -- Q4 2025 (5 quarters)",
            ha="center", va="center", fontsize=9, bbox=box_main)

    # Arrow
    ax.annotate("", xy=(5, 11.6), xytext=(5, 12.0),
                arrowprops=dict(arrowstyle="->", color="#333"))

    # Box 2: Deduplication
    ax.text(5, 11.2,
            f"Deduplicated by CASEID\n"
            f"(most recent report version retained)\n"
            f"n = {counts['total_dedup']:,} unique cases",
            ha="center", va="center", fontsize=9, bbox=box_main)

    # Arrow
    ax.annotate("", xy=(5, 10.1), xytext=(5, 10.6),
                arrowprops=dict(arrowstyle="->", color="#333"))

    # Box 3: Drug standardisation
    ax.text(5, 9.7,
            f"Drug names standardised via RxNorm\n"
            f"6 xanomeline-trospium name variants mapped\n"
            f"n = {counts['with_drug']:,} cases with drug records",
            ha="center", va="center", fontsize=9, bbox=box_main)

    # Arrow down + exclusion to right
    ax.annotate("", xy=(5, 8.6), xytext=(5, 9.1),
                arrowprops=dict(arrowstyle="->", color="#333"))

    # Exclusion box: non-Cobenfy
    excluded = counts["with_drug"] - counts["cobenfy_all"]
    ax.text(8.2, 9.0,
            f"Not xanomeline-trospium\n"
            f"n = {excluded:,}",
            ha="center", va="center", fontsize=8, bbox=box_excl)
    ax.annotate("", xy=(7.2, 9.0), xytext=(5.8, 9.3),
                arrowprops=dict(arrowstyle="->", color="#E65100",
                                linestyle="--"))

    # Box 4: Cobenfy cases identified
    ax.text(5, 8.2,
            f"Xanomeline-trospium cases identified\n"
            f"n = {counts['cobenfy_all']:,} (any role code)",
            ha="center", va="center", fontsize=9, bbox=box_main)

    # Arrow + exclusion
    ax.annotate("", xy=(5, 7.1), xytext=(5, 7.6),
                arrowprops=dict(arrowstyle="->", color="#333"))

    # Exclusion box: concomitant only
    ax.text(8.2, 7.5,
            f"Concomitant role only\n"
            f"n = {counts['cobenfy_concom']:,}",
            ha="center", va="center", fontsize=8, bbox=box_excl)
    ax.annotate("", xy=(7.2, 7.5), xytext=(5.8, 7.8),
                arrowprops=dict(arrowstyle="->", color="#E65100",
                                linestyle="--"))

    # Box 5: PS+SS cases
    ax.text(5, 6.7,
            f"Primary or secondary suspect cases\n"
            f"n = {counts['cobenfy_ps_ss']:,}\n"
            f"(PS: {counts['cobenfy_ps']:,}  |  SS: {counts['cobenfy_ss']:,})",
            ha="center", va="center", fontsize=9, bbox=box_main)

    # Arrow
    ax.annotate("", xy=(5, 5.6), xytext=(5, 6.1),
                arrowprops=dict(arrowstyle="->", color="#333"))

    # Box 6: Disproportionality
    ax.text(5, 5.2,
            f"Drug-PT pairs with >= 3 reports\n"
            f"n = {counts['drug_pt_pairs']} preferred terms tested\n"
            f"4-method disproportionality battery applied",
            ha="center", va="center", fontsize=9, bbox=box_main)

    # Arrow
    ax.annotate("", xy=(5, 4.1), xytext=(5, 4.6),
                arrowprops=dict(arrowstyle="->", color="#333"))

    # Box 7: Final signals
    ax.text(5, 3.7,
            "CONSENSUS SIGNALS\n"
            "56 signals (>= 3/4 methods positive)\n"
            "37 pharmacological  |  19 disease manifestations",
            ha="center", va="center", fontsize=9, fontweight="bold",
            bbox=box_final)

    # Comparator box (to the left)
    comp_text = (
        "Active Comparators (PS+SS)\n"
        f"Olanzapine: {counts['comp_olanzapine']:,}\n"
        f"Risperidone: {counts['comp_risperidone']:,}\n"
        f"Aripiprazole: {counts['comp_aripiprazole']:,}\n"
        f"Quetiapine: {counts['comp_quetiapine']:,}\n"
        f"Lurasidone: {counts['comp_lurasidone']:,}\n"
        f"Brexpiprazole: {counts['comp_brexpiprazole']:,}"
    )
    ax.text(1.5, 5.2, comp_text,
            ha="center", va="center", fontsize=7.5, bbox=box_comp)
    ax.annotate("", xy=(3.0, 5.2), xytext=(3.8, 5.2),
                arrowprops=dict(arrowstyle="<-", color="#6A1B9A",
                                linestyle="--"))

    # Sensitivity analyses box (bottom)
    ax.text(5, 2.3,
            "Sensitivity Analyses\n"
            "PS-only (n=1,416)  |  US-only (n=1,408)  |  HCP-only (n=706)\n"
            "Serious-only (n=131)  |  Monotherapy (n=1,298)\n"
            "Reporter-stratified  |  Age-sex stratified  |  Sequential (5 quarters)",
            ha="center", va="center", fontsize=8, bbox=box_main)
    ax.annotate("", xy=(5, 2.8), xytext=(5, 3.2),
                arrowprops=dict(arrowstyle="->", color="#333"))

    plt.tight_layout()
    out = FIG_DIR / "fig0_flow_diagram.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig0_flow_diagram.pdf")
    plt.close(fig)
    print(f"  Flow diagram saved: {out.name}")


def main():
    print("=" * 50)
    print("  Generating Case-Selection Flow Diagram")
    print("=" * 50)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    counts = get_counts(con)
    con.close()

    print(f"\n  Key counts:")
    for k, v in counts.items():
        print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

    draw_flow_diagram(counts)
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
