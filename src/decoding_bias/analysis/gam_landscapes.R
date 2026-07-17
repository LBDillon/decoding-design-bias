#!/usr/bin/env Rscript
# Minimal R/mgcv reproduction for Figure 3C and Table S15.

args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1) args[[1]] else "data/main_analysis.csv"
output_dir <- if (length(args) >= 2) args[[2]] else "results/gam"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  stop("R package 'mgcv' is required. Install it with install.packages('mgcv').")
}

features <- c(
  "sequence_length", "mw_per_residue", "isoelectric_point",
  "acidic_residue_fraction", "basic_residue_fraction", "gravy", "aromaticity",
  "instability_index", "proline_fraction", "ordered_percent",
  "helix_sheet_contrast", "rco", "avg_cb_distance", "surface_exposure"
)
models <- c(
  "proteinmpnn_score", "solublempnn_score", "caliby_score", "soluble_caliby_score",
  "esmif_score", "triflow_score", "mif_score", "mifst_score",
  "esm3_struct_cond_score", "esm3_seq_only_score", "ESM2_15B_pppl_score",
  "carp_640M_score", "progen2_XL_score", "protgpt2_score"
)

data <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
stopifnot(all(features %in% names(data)), all(models %in% names(data)))
complete <- data[complete.cases(data[, features]), , drop = FALSE]
scaled <- scale(complete[, features])
pca <- prcomp(scaled, center = FALSE, scale. = FALSE)
if (pca$rotation["sequence_length", "PC1"] < 0) pca$x[, "PC1"] <- -pca$x[, "PC1"]
if (pca$rotation["instability_index", "PC2"] < 0) pca$x[, "PC2"] <- -pca$x[, "PC2"]
complete$PC1 <- pca$x[, "PC1"]
complete$PC2 <- pca$x[, "PC2"]

fit_one <- function(score_column) {
  fit_data <- complete[, c("PC1", "PC2", score_column)]
  fit_data <- fit_data[complete.cases(fit_data), ]
  names(fit_data)[3] <- "score"
  fit_data$score <- as.numeric(scale(fit_data$score))
  basis_k <- max(12, min(50, floor(nrow(fit_data) / 300)))
  model <- mgcv::gam(score ~ s(PC1, PC2, k = basis_k), data = fit_data, method = "REML")
  list(
    model = model,
    data = fit_data,
    row = data.frame(
      model = score_column,
      n = nrow(fit_data),
      gam_dev_explained_pct = 100 * summary(model)$dev.expl,
      stringsAsFactors = FALSE
    )
  )
}

fits <- lapply(models, fit_one)
names(fits) <- models
table <- do.call(rbind, lapply(fits, function(x) x$row))
table <- table[order(-table$gam_dev_explained_pct), ]
write.csv(table, file.path(output_dir, "gam_deviance.csv"), row.names = FALSE)

# A compact six-model landscape figure. White cells lie outside the observed hull.
representative <- c(
  "proteinmpnn_score", "esmif_score", "mif_score",
  "mifst_score", "ESM2_15B_pppl_score", "protgpt2_score"
)
labels <- c(
  proteinmpnn_score = "ProteinMPNN", esmif_score = "ESM-IF", mif_score = "MIF",
  mifst_score = "MIF-ST", ESM2_15B_pppl_score = "ESM2-15B", protgpt2_score = "ProtGPT2"
)
limits <- range(c(complete$PC1, complete$PC2), finite = TRUE)
axis_values <- seq(limits[1], limits[2], length.out = 100)
grid <- expand.grid(PC1 = axis_values, PC2 = axis_values)
hull_index <- grDevices::chull(complete$PC1, complete$PC2)
hull <- as.matrix(complete[c(hull_index, hull_index[1]), c("PC1", "PC2")])
inside <- mgcv::in.out(hull, as.matrix(grid[, c("PC1", "PC2")]))
palette <- grDevices::colorRampPalette(c("#2166ac", "white", "#b2182b"))(100)

grDevices::pdf(file.path(output_dir, "figure3c_gam_landscapes.pdf"), width = 10, height = 7)
par(mfrow = c(2, 3), mar = c(3.2, 3.2, 2.5, 1), oma = c(1, 1, 1, 1))
for (score_column in representative) {
  prediction <- predict(fits[[score_column]]$model, newdata = grid)
  prediction[!inside] <- NA_real_
  matrix_prediction <- matrix(prediction, nrow = length(axis_values), ncol = length(axis_values))
  image(axis_values, axis_values, matrix_prediction, col = palette, zlim = c(-2, 2),
        xlab = "PC1", ylab = "PC2", main = sprintf("%s (%.1f%%)", labels[[score_column]],
        100 * summary(fits[[score_column]]$model)$dev.expl), useRaster = TRUE)
  contour(axis_values, axis_values, matrix_prediction, add = TRUE, drawlabels = FALSE,
          levels = seq(-2, 2, by = .5), col = "grey45", lwd = .45)
}
dev.off()

cat(sprintf("[gam] %d models fitted on the 14-feature PCA -> %s\n", length(models), output_dir))
