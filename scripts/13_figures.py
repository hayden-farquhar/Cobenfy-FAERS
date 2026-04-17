"""
Script 13: Generate publication-quality figures for manuscript.

Produces 6 figures:
    1. Forest plot of consensus signals (primary disproportionality)
    2. Active-comparator heatmap (ROR Cobenfy vs each comparator)
    3. Sequential signal detection (cumulative ROR by quarter)
    4. Time-to-onset Weibull survival curves
    5. Label concordance (detected vs FDA label)
    6. Signal classification (pharmacological vs disease manifestation)

Usage:
    python scripts/13_figures.py

Requires: outputs/tables/ and outputs/supplementary/ CSVs from prior scripts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from scipy.stats import weibull_min

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
SUPP_DIR = PROJECT_ROOT / "outputs" / "supplementary"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: FOREST PLOT OF CONSENSUS SIGNALS
# ═════════════════════════════════════════════════════════════════════════════

def fig1_forest_plot():
    """Forest plot showing ROR with 95% CI for all consensus signals."""
    print("  Figure 1: Forest plot of consensus signals")

    df = pd.read_csv(TABLE_DIR / "signals_cobenfy_consensus.csv")
    # Sort by ROR descending; cap display at top 30 for readability
    df = df.sort_values("ror", ascending=True).tail(35)

    # Load disease classification if available
    disease_path = SUPP_DIR / "signal_classification_disease_vs_drug.csv"
    disease_df = pd.read_csv(disease_path) if disease_path.exists() else None

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.28)))

    y_pos = range(len(df))
    colors = []
    for _, r in df.iterrows():
        if disease_df is not None:
            match = disease_df[disease_df["pt"] == r["pt"]]
            if len(match) > 0 and match.iloc[0]["disease_manifestation"]:
                colors.append("#999999")
            else:
                colors.append("#2166AC")
        else:
            colors.append("#2166AC")

    # Plot CIs
    for i, (_, r) in enumerate(df.iterrows()):
        upper_cap = min(r["ror_upper95"], r["ror"] * 3) if r["ror_upper95"] > r["ror"] * 3 else r["ror_upper95"]
        ax.plot([r["ror_lower95"], upper_cap], [i, i],
                color=colors[i], linewidth=1.2, solid_capstyle="round")
        if r["ror_upper95"] > upper_cap:
            ax.annotate(">", xy=(upper_cap, i), fontsize=7, color=colors[i],
                        ha="left", va="center")

    # Plot point estimates
    ax.scatter([r["ror"] for _, r in df.iterrows()], y_pos,
               c=colors, s=30, zorder=5, edgecolors="white", linewidth=0.5)

    # Reference line at ROR = 1
    ax.axvline(x=1, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["pt"].tolist())
    ax.set_xlabel("Reporting Odds Ratio (95% CI)")
    ax.set_xscale("log")
    ax.set_xlim(0.5, None)

    # Add case counts on right
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(ax.get_xlim()[1] * 0.85, i, f"n={int(r['a'])}",
                fontsize=7, va="center", ha="left", color="#555555")

    # Legend
    if disease_df is not None:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC',
                   markersize=7, label='Pharmacological signal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
                   markersize=7, label='Disease manifestation'),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True,
                  framealpha=0.9, edgecolor="#cccccc")

    ax.set_title("Cobenfy (Xanomeline-Trospium): Disproportionality Signals in FAERS",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    out = FIG_DIR / "fig1_forest_plot.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig1_forest_plot.pdf")
    plt.close(fig)
    print(f"    → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: ACTIVE-COMPARATOR HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

def fig2_comparator_heatmap():
    """Heatmap of log2(ROR) for Cobenfy vs each active comparator."""
    print("  Figure 2: Active-comparator heatmap")

    df = pd.read_csv(TABLE_DIR / "active_comparator_results.csv")

    # Pivot to matrix: PT × comparator
    pivot = df.pivot_table(index="pt", columns="drug_b", values="ror")

    # Keep PTs with at least one significant comparison
    sig_pts = df[df["bonferroni_sig"]]["pt"].unique()
    pivot = pivot.loc[pivot.index.isin(sig_pts)]

    if len(pivot) == 0:
        print("    ⚠ No significant comparisons to plot")
        return

    # Log2 transform for colour scale (symmetric around 0 = equal)
    log_pivot = np.log2(pivot.replace(0, np.nan))

    # Clip extreme values for readability
    log_pivot = log_pivot.clip(-5, 5)

    # Sort by mean log2 ROR (most Cobenfy-elevated at top)
    log_pivot = log_pivot.loc[log_pivot.mean(axis=1).sort_values(ascending=True).index]

    # Clean column names
    log_pivot.columns = [c.replace("xanomeline-trospium vs ", "").title()
                         for c in log_pivot.columns]

    fig, ax = plt.subplots(figsize=(8, max(5, len(log_pivot) * 0.32)))

    # Diverging colourmap: blue (Cobenfy lower) → white (equal) → red (Cobenfy higher)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(log_pivot, ax=ax, cmap=cmap, center=0,
                vmin=-5, vmax=5,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "log₂(ROR) — Cobenfy vs Comparator",
                          "shrink": 0.6},
                annot=False, fmt=".1f")

    # Add significance markers
    for i, pt in enumerate(log_pivot.index):
        for j, comp in enumerate(log_pivot.columns):
            pt_df = df[(df["pt"] == pt) & (df["drug_b"].str.contains(comp.lower(), case=False))]
            if len(pt_df) > 0 and pt_df.iloc[0]["bonferroni_sig"]:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                        fontsize=8, color="black", fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Active-Comparator Disproportionality: Cobenfy vs D2 Antagonists\n"
                 "(* = Bonferroni-significant)", fontweight="bold", pad=12)

    plt.tight_layout()
    out = FIG_DIR / "fig2_comparator_heatmap.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig2_comparator_heatmap.pdf")
    plt.close(fig)
    print(f"    → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: SEQUENTIAL SIGNAL DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def fig3_sequential_detection():
    """Line plot showing cumulative ROR evolution by quarter."""
    print("  Figure 3: Sequential signal detection")

    df = pd.read_csv(SUPP_DIR / "sequential_signal_detection.csv")

    # Select key PTs
    key_pts = ["NAUSEA", "VOMITING", "CONSTIPATION", "URINARY RETENTION",
               "DROOLING", "TACHYCARDIA", "DRY MOUTH", "HYPERHIDROSIS"]
    df = df[df["pt"].isin(key_pts)]

    quarters = sorted(df["cumulative_through"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    palette = sns.color_palette("husl", len(key_pts))
    for i, pt in enumerate(key_pts):
        pt_data = df[df["pt"] == pt].sort_values("cumulative_through")
        rors = pt_data["ror"].values
        lowers = pt_data["ror_lower95"].values
        uppers = pt_data["ror_upper95"].values
        x = range(len(pt_data))

        # Cap upper CI for plotting
        uppers_cap = np.minimum(uppers, np.where(np.isnan(rors), np.nan, rors * 3))

        valid = ~np.isnan(rors)
        if valid.sum() > 0:
            ax.plot(np.array(list(x))[valid], rors[valid],
                    marker="o", markersize=4, label=pt.title(),
                    color=palette[i], linewidth=1.5)
            ax.fill_between(np.array(list(x))[valid],
                            lowers[valid], uppers_cap[valid],
                            alpha=0.1, color=palette[i])

    ax.axhline(y=1, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels(quarters, rotation=45)
    ax.set_xlabel("Cumulative through Quarter")
    ax.set_ylabel("Reporting Odds Ratio")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_title("Cumulative Sequential Signal Detection: ROR by Quarter",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    out = FIG_DIR / "fig3_sequential_detection.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig3_sequential_detection.pdf")
    plt.close(fig)
    print(f"    → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: TIME-TO-ONSET DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════════════════

def fig4_time_to_onset():
    """Weibull-fitted time-to-onset curves for key adverse events."""
    print("  Figure 4: Time-to-onset distributions")

    weibull_df = pd.read_csv(TABLE_DIR / "time_to_onset_weibull.csv")

    if len(weibull_df) == 0:
        print("    ⚠ No Weibull data available")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("husl", len(weibull_df))

    t = np.linspace(0, 90, 500)  # 0 to 90 days

    for i, (_, row) in enumerate(weibull_df.iterrows()):
        shape = row["shape"]
        scale = row["scale"]
        pt = row["pt"]
        n = int(row["n_reports"])

        # Weibull survival function: S(t) = exp(-(t/scale)^shape)
        survival = np.exp(-(t / scale) ** shape)

        ax.plot(t, 1 - survival, label=f"{pt} (n={n}, β={shape:.2f})",
                color=palette[i], linewidth=1.5)

    ax.set_xlabel("Days from Drug Start to Event Onset")
    ax.set_ylabel("Cumulative Probability")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.05)

    # Add vertical lines for clinical milestones
    ax.axvline(x=30, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(31, 0.95, "30d", fontsize=7, color="grey")

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    ax.set_title("Time-to-Onset: Weibull Cumulative Distribution\n"
                 "(β < 1 indicates early-onset, decreasing hazard)",
                 fontweight="bold", pad=12)

    plt.tight_layout()
    out = FIG_DIR / "fig4_time_to_onset.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig4_time_to_onset.pdf")
    plt.close(fig)
    print(f"    → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: LABEL CONCORDANCE
# ═════════════════════════════════════════════════════════════════════════════

def fig5_label_concordance():
    """Stacked bar / summary visual of detected vs FDA label."""
    print("  Figure 5: Label concordance")

    df = pd.read_csv(SUPP_DIR / "label_concordance.csv")

    # Count by category
    cats = df["category"].value_counts()

    # Rename for display
    label_map = {
        "concordant": "Concordant\n(label + detected)",
        "label_only (no signal)": "Label only\n(tested, no signal)",
        "label_only (not tested, <3 reports)": "Label only\n(not tested)",
        "novel (not on label)": "Novel signals\n(not on label)",
    }

    categories = []
    counts = []
    colours = []
    colour_map = {
        "concordant": "#2166AC",
        "label_only (no signal)": "#F4A582",
        "label_only (not tested, <3 reports)": "#FDDBC7",
        "novel (not on label)": "#B2182B",
    }

    for cat in ["concordant", "label_only (no signal)",
                "label_only (not tested, <3 reports)", "novel (not on label)"]:
        if cat in cats:
            categories.append(label_map.get(cat, cat))
            counts.append(cats[cat])
            colours.append(colour_map.get(cat, "#999999"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                    gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: bar chart
    bars = ax1.barh(categories, counts, color=colours, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Number of Preferred Terms")
    ax1.set_title("Signal vs Label Classification", fontweight="bold")
    ax1.invert_yaxis()

    # Right: list novel signals with ROR
    novel = df[df["category"] == "novel (not on label)"].copy()
    if "ror" in novel.columns:
        novel = novel.dropna(subset=["ror"]).sort_values("ror", ascending=False)

    # Load disease classification
    disease_path = SUPP_DIR / "signal_classification_disease_vs_drug.csv"
    if disease_path.exists():
        disease_df = pd.read_csv(disease_path)
        novel = novel.merge(
            disease_df[["pt", "disease_manifestation", "classification"]],
            on="pt", how="left"
        )

    ax2.axis("off")
    text_lines = ["NOVEL SIGNALS (not in FDA label)\n"]
    if "disease_manifestation" in novel.columns:
        pharma = novel[novel["disease_manifestation"] == False]
        disease = novel[novel["disease_manifestation"] == True]
        text_lines.append(f"Pharmacological ({len(pharma)}):")
        for _, r in pharma.head(15).iterrows():
            text_lines.append(f"  • {r['pt']:<35s} ROR={r['ror']:.1f}")
        text_lines.append(f"\nDisease manifestation ({len(disease)}):")
        for _, r in disease.head(10).iterrows():
            text_lines.append(f"  • {r['pt']:<35s} ROR={r['ror']:.1f}")
    else:
        for _, r in novel.head(20).iterrows():
            text_lines.append(f"  • {r['pt']:<35s} ROR={r['ror']:.1f}")

    ax2.text(0.02, 0.98, "\n".join(text_lines), transform=ax2.transAxes,
             fontsize=7.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                       edgecolor="#cccccc"))

    plt.suptitle("Label Concordance Analysis: FAERS Signals vs FDA Prescribing Information",
                 fontweight="bold", fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig5_label_concordance.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig5_label_concordance.pdf")
    plt.close(fig)
    print(f"    → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: OUTCOME SEVERITY COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def fig6_outcome_severity():
    """Grouped bar chart comparing outcome severity across drugs."""
    print("  Figure 6: Outcome severity comparison")

    df = pd.read_csv(SUPP_DIR / "outcome_severity.csv")

    if len(df) == 0:
        print("    ⚠ No outcome severity data")
        return

    # Focus on key outcomes
    outcome_cols = ["pct_death", "pct_hosp"]
    outcome_labels = ["Death (%)", "Hospitalisation (%)"]

    drugs = df["drug"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    palette = ["#D32F2F" if d == "Cobenfy" else "#78909C" for d in drugs]

    for idx, (col, label) in enumerate(zip(outcome_cols, outcome_labels)):
        ax = axes[idx]
        bars = ax.barh(drugs, df[col], color=palette, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(label, fontweight="bold")
        ax.invert_yaxis()

    plt.suptitle("Outcome Severity: Cobenfy vs Active Comparators",
                 fontweight="bold", fontsize=11, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "fig6_outcome_severity.png"
    fig.savefig(out)
    fig.savefig(FIG_DIR / "fig6_outcome_severity.pdf")
    plt.close(fig)
    print(f"    → {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════���══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Generating Publication Figures")
    print("=" * 70)

    fig1_forest_plot()
    fig2_comparator_heatmap()
    fig3_sequential_detection()
    fig4_time_to_onset()
    fig5_label_concordance()
    fig6_outcome_severity()

    print(f"\n{'=' * 70}")
    print(f"  ALL FIGURES GENERATED")
    print(f"  Output: {FIG_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
