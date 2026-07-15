# Three corrections to the PCA analysis, in response to reviewer comments
# (a) GAM contour interpretation: compute per-grid-cell training density and
#     mark low-density cells so contour interpretation is bounded
# (b/c) Distinguish biophysical-feature variance (PCA cumvar) from
#     model-preference variance (GAM deviance explained)
# (d) Quantify per-domain cluster compactness instead of eye-balling
#
# Runs on the base cohort (7,843 proteins) so the corrected numbers can be
# compared against the published "51%" framing.
#
# Requires: mgcv (bundled with R), base. Avoids tidyverse/patchwork etc. for
# minimum-dependency reproducibility.

suppressPackageStartupMessages({
  library(mgcv)
  library(MASS)
})

# --- Config (match the published notebook) -------------------------------
WT_CSV <- "/Users/lauradillon/decoding-design-bias/dataset_update/Decoding_Bias_Dataset_updated.csv"
OUT_DIR <- "/Users/lauradillon/decoding-design-bias/dataset_update/family_analysis/pca_corrections"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

SCORE_COLUMNS <- c(
  "proteinmpnn_score", "esmif_score", "mif_score", "mifst_score",
  "ESM2_15B_pppl_score", "carp_640M_score", "AlkSecMPNN_score",
  "caliby_score", "triflow_score",
  "esm3_struct_cond_score", "esm3_seq_only_score"
)

# Mixed feature set from the published notebook
sequence_features <- c("mw_per_residue", "isoelectric_point", "instability_index",
                       "gravy", "sequence_length", "aromaticity")
# In the published notebook these come from inverted structural fields; we
# construct them here from the columns present in the base dataset.
make_structure_features <- function(df) {
  if (!"structural_compactness" %in% names(df) && "compactness" %in% names(df)) {
    df$structural_compactness <- 1 / df$compactness
  }
  if (!"centralization" %in% names(df) && "avg_cb_distance" %in% names(df)) {
    df$centralization <- 1 / df$avg_cb_distance
  }
  if (!"helix_sheet_contrast" %in% names(df) &&
      all(c("helix_percent", "sheet_percent") %in% names(df))) {
    df$helix_sheet_contrast <- df$helix_percent - df$sheet_percent
  }
  if (!"ordered_percent" %in% names(df) &&
      all(c("helix_percent", "sheet_percent") %in% names(df))) {
    df$ordered_percent <- df$helix_percent + df$sheet_percent
  }
  df
}
structure_features <- c("helix_sheet_contrast", "ordered_percent", "rco",
                        "structural_compactness", "centralization")
mixed_features <- c(sequence_features, structure_features)

# --- Load ----------------------------------------------------------------
df <- read.csv(WT_CSV, check.names = FALSE)
df <- make_structure_features(df)
cat(sprintf("Loaded %d rows from %s\n", nrow(df), basename(WT_CSV)))

# Keep only rows with complete biophysical-feature data
ok <- complete.cases(df[, mixed_features, drop = FALSE])
df <- df[ok, , drop = FALSE]
cat(sprintf("After complete-case filter on biophysical features: %d rows\n",
            nrow(df)))

# --- PCA on biophysical features (this is the part that gives the 51%) ---
fm <- as.matrix(df[, mixed_features, drop = FALSE])
cn <- colMeans(fm); sc <- apply(fm, 2, sd)
fm_s <- scale(fm, center = cn, scale = sc)
pca <- prcomp(fm_s, center = FALSE, scale. = FALSE)
var_explained <- pca$sdev^2 / sum(pca$sdev^2)
cat("\n=== PCA on biophysical-feature matrix ===\n")
cat(sprintf("PC1 captures %.1f%% of BIOPHYSICAL-FEATURE variance\n",
            100 * var_explained[1]))
cat(sprintf("PC2 captures %.1f%% of BIOPHYSICAL-FEATURE variance\n",
            100 * var_explained[2]))
cat(sprintf("PC1+PC2 cumulative: %.1f%%   <-- this is the 'X%%' published\n",
            100 * sum(var_explained[1:2])))

pcs <- as.data.frame(pca$x[, 1:2])
names(pcs) <- c("PC1", "PC2")
df_pc <- cbind(df, pcs)

# --- (b/c) GAM deviance explained per model = model-preference variance --
cat("\n=== GAM deviance explained per model ",
    "(= model-PREFERENCE variance captured by the PC1-PC2 surface) ===\n")
cat(sprintf("%-25s %12s %15s %10s\n",
            "Model", "N_with_score", "dev_explained_%", "adj_R^2"))
cat(strrep("-", 70), "\n", sep="")

gam_results <- list()
for (sc in SCORE_COLUMNS) {
  if (!(sc %in% names(df_pc))) next
  d <- df_pc[!is.na(df_pc[[sc]]), c("PC1", "PC2", sc)]
  d <- d[complete.cases(d), ]
  if (nrow(d) < 100) next
  names(d)[3] <- "y"
  m <- gam(y ~ s(PC1, PC2, k = 12), data = d, method = "REML")
  s <- summary(m)
  gam_results[[sc]] <- list(model = m, n = nrow(d),
                            dev_explained = s$dev.expl, r2 = s$r.sq)
  cat(sprintf("%-25s %12d %14.2f%% %10.4f\n",
              sc, nrow(d), 100 * s$dev.expl, s$r.sq))
}

# --- (a) Per-cell training density on a regular grid ---------------------
cat("\n=== Density-mask diagnostic for GAM contour interpretation ===\n")
grid_size <- 60
gx <- seq(min(df_pc$PC1), max(df_pc$PC1), length.out = grid_size)
gy <- seq(min(df_pc$PC2), max(df_pc$PC2), length.out = grid_size)
dx <- diff(gx)[1]; dy <- diff(gy)[1]
# 2D histogram count per grid cell
ix <- pmin(pmax(findInterval(df_pc$PC1, gx), 1), grid_size)
iy <- pmin(pmax(findInterval(df_pc$PC2, gy), 1), grid_size)
counts <- table(factor(ix, levels = 1:grid_size),
                factor(iy, levels = 1:grid_size))
# Cells with >= MIN_DENSITY training points are considered "interpretable"
MIN_DENSITY <- 3
n_interp <- sum(counts >= MIN_DENSITY)
n_total <- grid_size * grid_size
cat(sprintf("Grid: %d x %d = %d cells\n", grid_size, grid_size, n_total))
cat(sprintf("Cells with >= %d training points (interpretable): %d (%.1f%%)\n",
            MIN_DENSITY, n_interp, 100 * n_interp / n_total))
cat(sprintf("Cells with 0 training points (must be greyed out): %d (%.1f%%)\n",
            sum(counts == 0), 100 * sum(counts == 0) / n_total))

# --- (d) Per-domain cluster compactness ----------------------------------
cat("\n=== Per-domain cluster compactness in (PC1, PC2) space ===\n")
cat(sprintf("%-12s %8s %12s %14s %15s\n",
            "Domain", "N", "mean PC1/2", "median dist", "spread (1-sd ell area)"))
cat(strrep("-", 75), "\n", sep="")
for (d in c("Eukaryota", "Bacteria", "Archaea")) {
  sub <- df_pc[df_pc$domain == d, c("PC1", "PC2")]
  sub <- sub[complete.cases(sub), ]
  if (nrow(sub) < 5) next
  centroid <- colMeans(sub)
  # Mean Euclidean distance from each protein to the domain centroid
  d_to_centroid <- sqrt(rowSums((as.matrix(sub) - matrix(centroid, nrow(sub), 2,
                                                          byrow = TRUE))^2))
  # Per-domain covariance + 1-SD ellipse area = pi * sqrt(det(2x2 covariance))
  cv <- cov(sub)
  ellipse_area <- pi * sqrt(det(cv))
  cat(sprintf("%-12s %8d  (%5.2f,%5.2f)  %10.3f   %15.3f\n",
              d, nrow(sub), centroid[1], centroid[2],
              median(d_to_centroid), ellipse_area))
}

# Pairwise between-domain centroid distance and Bhattacharyya overlap proxy
cat("\nBetween-domain centroid distances and 1-SD ellipse overlap:\n")
domains <- c("Eukaryota", "Bacteria", "Archaea")
ctr <- list(); cv <- list()
for (d in domains) {
  sub <- df_pc[df_pc$domain == d, c("PC1", "PC2")]
  sub <- sub[complete.cases(sub), ]
  ctr[[d]] <- colMeans(sub)
  cv[[d]] <- cov(sub)
}
for (i in seq_along(domains)) {
  for (j in seq_along(domains)) {
    if (j <= i) next
    cd <- sqrt(sum((ctr[[domains[i]]] - ctr[[domains[j]]])^2))
    # Bhattacharyya distance for 2 multivariate Gaussians
    Sigma <- (cv[[domains[i]]] + cv[[domains[j]]]) / 2
    delta <- ctr[[domains[i]]] - ctr[[domains[j]]]
    bdist <- 0.125 * t(delta) %*% solve(Sigma) %*% delta +
             0.5 * log(det(Sigma) / sqrt(det(cv[[domains[i]]]) * det(cv[[domains[j]]])))
    overlap <- exp(-as.numeric(bdist))
    cat(sprintf("  %s vs %s: centroid distance = %.3f, Bhattacharyya overlap = %.3f\n",
                domains[i], domains[j], cd, overlap))
  }
}

# --- Save outputs ---------------------------------------------------------
res_df <- do.call(rbind, lapply(names(gam_results), function(sc) {
  data.frame(
    model = sc, n = gam_results[[sc]]$n,
    gam_deviance_explained_pct = 100 * gam_results[[sc]]$dev_explained,
    gam_adj_r2 = gam_results[[sc]]$r2,
    pca_PC1_pct = 100 * var_explained[1],
    pca_PC2_pct = 100 * var_explained[2],
    pca_PC1_PC2_pct = 100 * sum(var_explained[1:2])
  )
}))
write.csv(res_df, file.path(OUT_DIR, "pca_corrections_summary.csv"),
          row.names = FALSE)
cat(sprintf("\nWrote %s\n", file.path(OUT_DIR, "pca_corrections_summary.csv")))
