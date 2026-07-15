"""Compare the independent experimental-PDB cohort to the published main dataset.

Produces:
  outputs/independent_cohort/cohort_vs_main_table.csv   summary comparison table
  outputs/independent_cohort/cohort_vs_main_table.tex   LaTeX version
  outputs/independent_cohort/cohort_vs_main_figure.png  4-panel comparison figure
  outputs/independent_cohort/cohort_vs_main_figure.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COH = HERE / "outputs" / "independent_cohort"
# The updated paper's dataset is the FULL v12 metadata (main + round2 + round3,
# n=10,148) -- not just the source=='main' subset.
MAIN_META = HERE.parent / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"

DOMAIN_ORDER = ["Eukaryota", "Bacteria", "Archaea"]
C_MAIN, C_COH = "#55A868", "#4C72B0"  # Main = green, PDB cohort = blue


def load():
    main = pd.read_csv(MAIN_META, low_memory=False)  # full v12 (all sources)
    coh = pd.read_csv(COH / "cohort_pdb_scoring_inputs.csv")
    return main, coh


def summary_table(main, coh):
    def length_of(df):
        if "Length" in df:
            return pd.to_numeric(df["Length"], errors="coerce")
        return pd.to_numeric(df["resolved_len"], errors="coerce")

    rows = []
    rows.append(("N proteins", f"{len(main):,}", f"{len(coh):,}"))
    rows.append(("Structure source", "AlphaFold2 model", "Experimental X-ray"))
    rows.append(("Sequence scored", "Full UniProt", "Resolved chain"))
    for d in DOMAIN_ORDER:
        mp = (main["domain"] == d).mean() * 100
        cp = (coh["domain"] == d).mean() * 100
        rows.append((f"{d} (\\%)", f"{mp:.1f}", f"{cp:.1f}"))
    rows.append(("Distinct families", f"{main['protein_family'].nunique():,}",
                 f"{coh['protein_family'].nunique():,}"))
    sp_main = main["species"].nunique() if "species" in main else np.nan
    sp_coh = coh["species_collapsed"].nunique()
    rows.append(("Distinct species", f"{sp_main:,}", f"{sp_coh:,}"))
    rows.append(("Ribosomal (\\%)",
                 f"{(main['broad_function']=='ribosomal').mean()*100:.1f}",
                 f"{(coh['broad_function']=='ribosomal').mean()*100:.1f}"))
    lm, lc = length_of(main), length_of(coh)
    rows.append(("Median length (aa)", f"{lm.median():.0f}", f"{lc.median():.0f}"))
    if "resolution_A" in coh:
        rows.append(("Median resolution (\\AA)", "--", f"{coh['resolution_A'].median():.2f}"))
    return pd.DataFrame(rows, columns=["Property", "Main dataset", "Independent PDB cohort"])


def write_tex(tab):
    body = "\n".join(
        f"{r.Property} & {r._1} & {r._2} \\\\"
        for r in tab.itertuples(index=False, name="R")
    )
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Comparison of the published main dataset and the independent "
        "experimental-PDB replication cohort.}\n"
        "\\label{tab:cohort_vs_main}\n"
        "\\begin{tabular}{lrr}\n\\hline\n"
        "Property & Main dataset & Independent PDB cohort \\\\\n\\hline\n"
        + body
        + "\n\\hline\n\\end{tabular}\n\\end{table}\n"
    )
    (COH / "cohort_vs_main_table.tex").write_text(tex)


def fig(main, coh):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    # (a) domain composition
    mp = [ (main["domain"] == d).mean() * 100 for d in DOMAIN_ORDER ]
    cp = [ (coh["domain"] == d).mean() * 100 for d in DOMAIN_ORDER ]
    x = np.arange(len(DOMAIN_ORDER)); w = 0.38
    ax[0,0].bar(x - w/2, mp, w, label="Main", color=C_MAIN)
    ax[0,0].bar(x + w/2, cp, w, label="PDB cohort", color=C_COH)
    ax[0,0].set_xticks(x); ax[0,0].set_xticklabels(DOMAIN_ORDER)
    ax[0,0].set_ylabel("% of proteins"); ax[0,0].set_title("(a) Taxonomic domain")
    ax[0,0].legend(frameon=False)

    # (b) broad_function composition (top categories by main)
    order = main["broad_function"].value_counts().head(10).index.tolist()
    for c in coh["broad_function"].value_counts().head(10).index:
        if c not in order:
            order.append(c)
    order = order[:12]
    mf = [ (main["broad_function"] == c).mean() * 100 for c in order ]
    cf = [ (coh["broad_function"] == c).mean() * 100 for c in order ]
    y = np.arange(len(order))
    ax[0,1].barh(y - w/2, mf, w, label="Main", color=C_MAIN)
    ax[0,1].barh(y + w/2, cf, w, label="PDB cohort", color=C_COH)
    ax[0,1].set_yticks(y); ax[0,1].set_yticklabels(order, fontsize=8)
    ax[0,1].invert_yaxis(); ax[0,1].set_xlabel("% of proteins")
    ax[0,1].set_title("(b) Broad function"); ax[0,1].legend(frameon=False)

    # (c) length distribution
    lm = pd.to_numeric(main["Length"], errors="coerce").dropna()
    lc = pd.to_numeric(coh["resolved_len"], errors="coerce").dropna()
    bins = np.linspace(0, 1000, 41)
    ax[1,0].hist(lm, bins=bins, density=True, alpha=0.6, color=C_MAIN, label="Main (UniProt len)")
    ax[1,0].hist(lc, bins=bins, density=True, alpha=0.6, color=C_COH, label="PDB cohort (resolved len)")
    ax[1,0].set_xlabel("Length (aa)"); ax[1,0].set_ylabel("Density")
    ax[1,0].set_title("(c) Sequence length"); ax[1,0].legend(frameon=False)

    # (d) cohort resolution distribution (experimental only)
    res = pd.to_numeric(coh["resolution_A"], errors="coerce").dropna()
    ax[1,1].hist(res, bins=np.linspace(0.8, 2.6, 28), color=C_COH, alpha=0.85)
    ax[1,1].axvline(res.median(), color="k", ls="--", lw=1,
                    label=f"median {res.median():.2f} Å")
    ax[1,1].set_xlabel("Resolution (Å)"); ax[1,1].set_ylabel("Proteins")
    ax[1,1].set_title("(d) PDB cohort resolution"); ax[1,1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(COH / "cohort_vs_main_figure.png", dpi=200)
    fig.savefig(COH / "cohort_vs_main_figure.pdf")


def main():
    m, c = load()
    tab = summary_table(m, c)
    tab.to_csv(COH / "cohort_vs_main_table.csv", index=False)
    write_tex(tab)
    fig(m, c)
    print(tab.to_string(index=False))
    print(f"\nwrote cohort_vs_main_table.csv/.tex and cohort_vs_main_figure.png/.pdf")


if __name__ == "__main__":
    main()
