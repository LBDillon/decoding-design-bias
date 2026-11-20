#!/usr/bin/env Rscript
# Shared R/mgcv fitting for the reproduction table and exported figure grids.

GAM_FEATURES <- c(
  "sequence_length", "mw_per_residue", "isoelectric_point",
  "acidic_residue_fraction", "basic_residue_fraction", "gravy", "aromaticity",
  "instability_index", "proline_fraction", "ordered_percent",
  "helix_sheet_contrast", "rco", "avg_cb_distance", "surface_exposure"
)
GAM_MODELS <- c(
  "proteinmpnn_score", "solublempnn_score", "caliby_score", "soluble_caliby_score",
  "esmif_score", "triflow_score", "mif_score", "mifst_score",
  "esm3_struct_cond_score", "esm3_seq_only_score", "ESM2_15B_pppl_score",
  "carp_640M_score", "progen2_score", "protgpt2_score", "progen2_XL_score",
  "AlkSecMPNN_020_score", "AcidSecMPNN_020_score"
)

prepare_gam_data <- function(input_csv) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("R package 'mgcv' is required; see the R setup instructions in README.md.")
  }
  data <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
  data <- data[!data$Entry %in% c("P19949", "Q8TFG7"), , drop = FALSE]
  stopifnot(all(GAM_FEATURES %in% names(data)))
  data <- data[complete.cases(data[, GAM_FEATURES]), , drop = FALSE]
  scaled <- scale(data[, GAM_FEATURES])
  pca <- prcomp(scaled, center = FALSE, scale. = FALSE)
  for (axis in 1:2) {
    orientation <- c("sequence_length", "instability_index")[axis]
    if (pca$rotation[orientation, axis] < 0) {
      pca$x[, axis] <- -pca$x[, axis]
      pca$rotation[, axis] <- -pca$rotation[, axis]
    }
  }
  data$PC1 <- pca$x[, 1]
  data$PC2 <- pca$x[, 2]
  list(data = data, scaled = scaled, pca = pca)
}

fit_score_gam <- function(data, column) {
  fit_data <- data[complete.cases(data[, c("PC1", "PC2", column)]), c("PC1", "PC2", column)]
  names(fit_data)[3] <- "score"
  raw_difference <- grepl("minus_base$", column)
  if (!raw_difference) fit_data$score <- as.numeric(scale(fit_data$score))
  basis_k <- max(12, min(50, floor(nrow(fit_data) / 300)))
  fit <- mgcv::gam(score ~ s(PC1, PC2, k = basis_k), data = fit_data, method = "REML")
  list(model = fit, data = fit_data, basis_k = basis_k,
       deviance = 100 * summary(fit)$dev.expl, raw_difference = raw_difference)
}

reproduce_gams <- function(input_csv, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  prepared <- prepare_gam_data(input_csv)
  stopifnot(all(GAM_MODELS %in% names(prepared$data)))
  rows <- lapply(GAM_MODELS, function(column) {
    fit <- fit_score_gam(prepared$data, column)
    data.frame(model = column, n = nrow(fit$data), gam_dev_explained_pct = fit$deviance)
  })
  table <- do.call(rbind, rows)
  table <- table[order(-table$gam_dev_explained_pct), ]
  write.csv(table, file.path(output_dir, "gam_deviance.csv"), row.names = FALSE)
  cat(sprintf("[gam] %d models fitted on the 14-feature PCA -> %s\n", length(GAM_MODELS), output_dir))
}

export_gam_grids <- function(input_csv, output_dir, requested = NULL) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  prepared <- prepare_gam_data(input_csv)
  data <- prepared$data
  stopifnot(all(c("AlkSecMPNN_020_score", "AcidSecMPNN_020_score", "proteinmpnn_score") %in% names(data)))
  # proteinmpnn_score is the v_48_020 checkpoint, the base of both fine-tuned models.
  stopifnot(identical(data$ProteinMPNN_v020_score, data$proteinmpnn_score))
  data$AlkSecMPNN_minus_base <- as.numeric(scale(data$AlkSecMPNN_020_score)) - as.numeric(scale(data$proteinmpnn_score))
  data$AcidSecMPNN_minus_base <- as.numeric(scale(data$AcidSecMPNN_020_score)) - as.numeric(scale(data$proteinmpnn_score))
  stopifnot(all(is.finite(data$AlkSecMPNN_minus_base)), all(is.finite(data$AcidSecMPNN_minus_base)))
  if (is.null(requested)) requested <- c(GAM_MODELS, "AlkSecMPNN_minus_base", "AcidSecMPNN_minus_base")
  stopifnot(length(requested) == length(unique(requested)), all(requested %in% names(data)))
  L <- max(abs(c(data$PC1, data$PC2))) * 1.05
  axis <- seq(-L, L, length.out = 160)
  grid <- expand.grid(PC1 = axis, PC2 = axis)
  write.csv(data.frame(feature = GAM_FEATURES,
                      center = attr(prepared$scaled, "scaled:center"),
                      scale = attr(prepared$scaled, "scaled:scale"),
                      PC1 = prepared$pca$rotation[, 1], PC2 = prepared$pca$rotation[, 2]),
            file.path(output_dir, "global_pca_transform.csv"), row.names = FALSE)
  write.csv(data[, c("Entry", "PC1", "PC2")], file.path(output_dir, "global_pca_coordinates.csv"), row.names = FALSE)
  rows <- list()
  for (column in requested) {
    fit <- fit_score_gam(data, column)
    d <- fit$data
    g <- grid
    g$predicted_score <- as.numeric(predict(fit$model, newdata = g))
    breaks <- seq(-L, L, length.out = 56)
    ix <- pmin(pmax(findInterval(d$PC1, breaks, all.inside = TRUE), 1), 55)
    iy <- pmin(pmax(findInterval(d$PC2, breaks, all.inside = TRUE), 1), 55)
    counts <- table(factor(ix, levels = 1:55), factor(iy, levels = 1:55))
    gx <- pmin(pmax(findInterval(g$PC1, breaks, all.inside = TRUE), 1), 55)
    gy <- pmin(pmax(findInterval(g$PC2, breaks, all.inside = TRUE), 1), 55)
    g$training_n <- as.numeric(counts[cbind(gx, gy)])
    h <- chull(d$PC1, d$PC2)
    hull <- as.matrix(d[c(h, h[1]), c("PC1", "PC2")])
    g$outside_hull <- !mgcv::in.out(hull, as.matrix(g[, c("PC1", "PC2")]))
    g$supported <- g$training_n >= 3 & !g$outside_hull
    write.csv(g, file.path(output_dir, paste0(column, "_grid.csv")), row.names = FALSE)
    saveRDS(fit$model, file.path(output_dir, paste0(column, "_fit.rds")))
    rows[[length(rows) + 1]] <- data.frame(
      model = column, n = nrow(d), basis_k = fit$basis_k,
      deviance_explained_pct = fit$deviance, grid_n = 160, density_bins = 55,
      minimum_support = 3, axis_limit = L,
      score_normalisation = if (fit$raw_difference) "difference of within-model z scores; no further scaling" else "within-model z score",
      R_version = as.character(getRversion()), mgcv_version = as.character(packageVersion("mgcv")))
    cat(column, "exported\n")
  }
  write.csv(do.call(rbind, rows), file.path(output_dir, "fit_metadata.csv"), row.names = FALSE)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  input <- if (length(args) >= 1) args[1] else "data/main_analysis.csv"
  output <- if (length(args) >= 2) args[2] else "results/gam"
  reproduce_gams(input, output)
}
