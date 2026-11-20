# Recompute the 7-model x 12-property design tests of Table S20 in R and compare them
# with the stored reference. Usage: Rscript scripts/verify_design_statistics.R [ROOT] [OUTPUT_DIR]
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "."
out_dir <- if (length(args) >= 2) args[2] else file.path(root, "results/design_R")
design <- read.csv(file.path(root, "data/design/designs_features.csv"), check.names = FALSE)
wt <- read.csv(file.path(root, "data/design/wt_features.csv"), check.names = FALSE)
models <- c("ProteinMPNN", "SolubleMPNN", "Caliby", "SolubleCaliby", "ESM-IF", "MIF", "MIF-ST")
features <- c("mw_per_residue", "isoelectric_point", "acidic_residue_fraction", "basic_residue_fraction", "gravy", "aromaticity", "instability_index", "proline_fraction", "ordered_percent", "helix_sheet_contrast", "rco", "avg_cb_distance")
means <- aggregate(design[design$model %in% models, features], by = design[design$model %in% models, c("model", "uniprot_id")], FUN = mean)
rows <- list()
for (model in models) {
  sub <- means[means$model == model, ]
  wild <- wt[match(sub$uniprot_id, wt$uniprot_id), ]
  stopifnot(!anyNA(wild$uniprot_id), nrow(sub) == 25)
  for (feature in features) {
    delta <- sub[[feature]] - wild[[feature]]
    # Fractions that are mathematically tied can differ at floating-point precision.
    # Round test inputs only; retain unrounded means and effect sizes.
    rows[[length(rows) + 1]] <- data.frame(model = model, feature = feature, n_templates = length(delta), mean_delta = mean(delta), dz = mean(delta) / sd(delta), wilcoxon_p = wilcox.test(round(delta, 12), alternative = "two.sided", exact = FALSE, correct = TRUE)$p.value)
  }
}
result <- do.call(rbind, rows)
result$p_fdr <- p.adjust(result$wilcoxon_p, method = "BH")
stopifnot(nrow(result) == 84)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(result, file.path(out_dir, "table_s20_statistics_R.csv"), row.names = FALSE)
reference <- read.csv(file.path(root, "expected/design/table_s20_statistics_R.csv"), check.names = FALSE)
stopifnot(isTRUE(all.equal(result, reference, tolerance = 1e-12, check.attributes = FALSE)))
cat("Matches expected/design/table_s20_statistics_R.csv:", nrow(result), "tests, BH across all 84\n")
