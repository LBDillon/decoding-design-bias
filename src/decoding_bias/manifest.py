"""Paper-output manifest: every figure/table -> stage, command, output, status.

This is the machine-readable map the CLI `figures` and `verify` commands drive,
and the single place that records which outputs are reproducible from the shipped
analysis table versus blocked on external inputs (weights, structures, design
CSVs) or manual (illustrator) or pending in the draft.

status:
  ready   - reproducible now from the shipped table via `decoding-bias <cmd>`
  R       - reproducible but needs R/mgcv (GAM landscapes / deviance)
  blocked - needs an external input not shipped (weights/structures/design CSVs)
  manual  - hand-made (no code)
  pending - analysis/figure not yet produced in the current draft
"""
from __future__ import annotations

# fields: id, label, kind, main_or_si, stage, command, outputs (repo-relative,
# under results/<stage>/...), status, note
OUTPUTS = [
    # ---------------- MAIN FIGURES ----------------
    dict(id="fig1", label="Figure 1", kind="figure", section="main", stage="-",
         command=None, outputs=[], status="manual",
         note="Structure- vs sequence-conditioned schematic; illustrator, no code."),
    dict(id="fig2", label="Figure 2", kind="figure", section="main", stage="taxonomy",
         command="decoding-bias taxonomy",
         outputs=["results/elo/elo_full_unweighted/results/all_models_species_ratings_long.csv",
                  "results/elo/elo_full_unweighted/results/archaea_eukaryota_gap.csv"],
         status="ready",
         note="Panels 2A/2C from species Elo; 2B phylum heatmap needs paths.metadata_table."),
    dict(id="fig3", label="Figure 3", kind="figure", section="main", stage="pca",
         command="decoding-bias pca",
         outputs=["results/pca/pca_loadings.csv", "results/pca/pca_coordinates.csv"],
         status="R",
         note="3A/3B PCA (Python, ready); 3C GAM landscapes need the R notebook."),
    dict(id="fig4", label="Figure 4", kind="figure", section="main", stage="design",
         command="decoding-bias design",
         outputs=["results/design/physchem_effect_sizes.csv",
                  "results/design/physchem_shift_heatmap.png"],
         status="blocked", note="Design shifts (dz). Runs andreproduces Table S21 dz exactly "
                                "(verify: 84/84 cells) once paths.design_dir is set; tables not shipped."),
    dict(id="fig5", label="Figure 5", kind="figure", section="main", stage="finetune",
         command="decoding-bias finetune", outputs=[], status="blocked",
         note="Fine-tuned surface acid-base shift + self-consistency; needs weights/designs."),
    # ---------------- MAIN TABLES ----------------
    dict(id="tab1", label="Table 1", kind="table", section="main", stage="variance",
         command="decoding-bias variance",
         outputs=["results/variance_decomposition/score_variance_decomposition.csv"],
         status="ready", note="Nested variance decomposition (species/attenuation/retention)."),
    dict(id="tab2", label="Table 2", kind="table", section="main", stage="variance",
         command="decoding-bias variance",
         outputs=["results/variance_decomposition/score_variance_decomposition.csv"],
         status="ready", note="Biophysical increment Δbio (same CSV, different columns)."),
    dict(id="tab3", label="Table 3", kind="table", section="main", stage="taxonomy",
         command="decoding-bias taxonomy",
         outputs=["results/elo/elo_full_unweighted/results/model_analysis_summary.txt",
                  "results/elo/elo_full_unweighted/results/archaea_eukaryota_gap.csv"],
         status="ready", note="Mean species Elo by domain + Archaea-Eukaryota gap."),
    dict(id="tab4", label="Table 4", kind="table", section="main", stage="design",
         command="decoding-bias design",
         outputs=["results/design/functional_residue_conservation_by_model.csv"],
         status="blocked", note="Functional-residue recovery. Runs andreproduces exactly "
                                "(MIF-ST Δ=-0.158, p=1.8e-4; verify 9/9 models) once design_dir "
                                "has all_designs_and_wt.csv + _uniprot_features_cache.json."),
    dict(id="tab5", label="Table 5", kind="table", section="main", stage="finetune",
         command="decoding-bias finetune", outputs=[], status="blocked",
         note="Base-relative design shift along surface acid-base PCA (R, needs designs)."),
    dict(id="tab6", label="Table 6", kind="table", section="main", stage="finetune",
         command="decoding-bias finetune", outputs=[], status="blocked",
         note="Direct surface-feature shifts; needs designs_ph_features.csv."),
    dict(id="tab7", label="Table 7", kind="table", section="main", stage="dataset",
         command="decoding-bias dataset",
         outputs=["results/dataset/composition.csv"], status="ready",
         note="Main dataset composition (proteins/species/families by domain)."),
    dict(id="tab8", label="Table 8", kind="table", section="main", stage="-",
         command=None, outputs=[], status="manual",
         note="The 14 modelling features; descriptive (catalog.BIOPHYS_14)."),
    # ---------------- SUPPLEMENTARY (selected reproducible) ----------------
    dict(id="figS2", label="Fig S2", kind="figure", section="si", stage="variance",
         command="decoding-bias variance",
         outputs=["results/variance_decomposition/fig_variance_decomposition.pdf"], status="ready",
         note="Consolidated 3-panel variance decomposition (species attenuation, unique "
              "contributions, within-family retention). The four fig_vd_*.pdf single panels "
              "are also emitted."),
    dict(id="figS6", label="Fig S6", kind="figure", section="si", stage="importance",
         command="decoding-bias importance",
         outputs=["results/property_importance/property_importance_expanded.csv"], status="ready",
         note="3-panel importance heatmap; CSV ready, figure via physchem_importance_table_figure."),
    dict(id="tabS9", label="Table S9", kind="table", section="si", stage="variance",
         command="decoding-bias variance",
         outputs=["results/variance_decomposition/score_variance_decomposition.csv"], status="ready"),
    dict(id="tabS10", label="Table S10", kind="table", section="si", stage="variance",
         command="decoding-bias variance",
         outputs=["results/variance_decomposition/score_variance_decomposition.csv"], status="ready"),
    dict(id="tabS14", label="Table S14", kind="table", section="si", stage="pca",
         command="decoding-bias pca",
         outputs=["results/pca/pca_loadings.csv"], status="ready", note="14-feature PCA loadings."),
    dict(id="tabS15", label="Table S15", kind="table", section="si", stage="pca",
         command="decoding-bias pca",
         outputs=["results/pca/gam_deviance.csv"], status="ready",
         note="GAM deviance (R/mgcv); deposited from the notebook to 00_data/pca_gam/ and "
              "emitted by `pca`. The Python PCA that feeds it is cross-checked against the R run."),
    dict(id="compactness", label="Tab S (compactness)", kind="table", section="si", stage="pca",
         command="decoding-bias pca",
         outputs=["results/pca/compactness_pairwise.csv"], status="ready"),
    dict(id="tabS16_18", label="Tables S16-S18", kind="table", section="si", stage="importance",
         command="decoding-bias importance",
         outputs=["results/property_importance/physchem_relweight_wide_expanded.csv",
                  "results/property_importance/physchem_relweight_wide_finetune020.csv"],
         status="ready"),
    dict(id="tabS5_8", label="Tables S5-S8", kind="table", section="si", stage="taxonomy",
         command="decoding-bias taxonomy --all-variants",
         outputs=["results/elo/elo_full_unweighted/results/model_analysis_summary.txt"],
         status="ready", note="Per-domain Elo, Cohen's d, weighting-scheme gaps."),
    dict(id="tabS13", label="Table S13", kind="table", section="si", stage="variance",
         command="decoding-bias variance --plddt", outputs=["results/variance_decomposition/plddt_vd.csv"],
         status="ready", note="pLDDT-as-score decomposition; avg_plddt is in the shipped table."),
    dict(id="tabS1_4", label="Tables S1-S4", kind="table", section="si", stage="-",
         command=None, outputs=[], status="manual",
         note="Model/feature specifications (models, checkpoints, 11 sequence + 5 structure features); descriptive."),
    dict(id="tabS11_12", label="Tables S11-S12", kind="table", section="si", stage="pdb-cohort",
         command="decoding-bias pdb-cohort", outputs=[], status="blocked",
         note="Main dataset vs experimental-PDB cohort; residual species PDB vs matched AFDB."),
    # ---------------- SUPPLEMENTARY (blocked / pending) ----------------
    dict(id="figS1", label="Fig S1", kind="figure", section="si", stage="taxonomy",
         command="decoding-bias taxonomy", outputs=[], status="pending",
         note="Placeholder in the current draft."),
    dict(id="figS3_4", label="Figs S3-S4", kind="figure", section="si", stage="pdb-cohort",
         command="decoding-bias pdb-cohort", outputs=[], status="blocked",
         note="Experimental-PDB cohort vs main; needs RCSB structures."),
    dict(id="figS5", label="Fig S5", kind="figure", section="si", stage="pca",
         command="decoding-bias pca --gam-landscapes", outputs=[], status="R",
         note="Per-model GAM preference landscapes."),
    dict(id="figS7", label="Fig S7", kind="figure", section="si", stage="design",
         command="decoding-bias design", outputs=[], status="blocked",
         note="Functional-residue recovery figure."),
    dict(id="figS8_11", label="Figs S8-S11", kind="figure", section="si", stage="finetune",
         command="decoding-bias finetune", outputs=[], status="blocked",
         note="Fine-tuning surface shift (S8), self-consistency (S9), matched surface shift "
              "(S10), ESM2-35M per-position map (S11)."),
    dict(id="tabS21", label="Table S21", kind="table", section="si", stage="design",
         command="decoding-bias design",
         outputs=["results/design/physchem_effect_sizes.csv"], status="blocked",
         note="WT->design dz shifts. Runs andverified exact (max |Δdz|=1e-16) once design_dir is set."),
    dict(id="tabS20", label="Table S20", kind="table", section="si", stage="design",
         command="decoding-bias design",
         outputs=["results/design/functional_residue_conservation_by_model.csv"], status="blocked",
         note="Functional-residue recovery (9 models incl. FT). Runs andverified exact once "
              "design_dir has the sequences + UniProt cache."),
    dict(id="tabS19", label="Table S19", kind="table", section="si", stage="design",
         command=None, outputs=[], status="manual", note="The 25 design-template list; descriptive."),
    dict(id="tabS22", label="Table S22", kind="table", section="si", stage="finetune",
         command="decoding-bias finetune",
         outputs=["results/finetune/table_s22_surface_steer.csv"], status="ready",
         note="ESM2-35M vs ProteinMPNN matched surface steer; reproduces exactly from the "
              "deposited surface_shift_matched.csv (ProteinMPNN -0.232, ESM2-35M -0.075)."),
]


def by_status():
    d = {}
    for o in OUTPUTS:
        d.setdefault(o["status"], []).append(o)
    return d
