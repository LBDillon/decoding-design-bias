# pH-specific WT -> design projection for ProteinMPNN_v002, AlkSecMPNN,
# and AcidSecMPNN designs.
#
# Use in the R PCA notebook after the main dataset `df` has been loaded, or run
# standalone with Rscript from the repo root. The input feature CSVs are made by:
#   python design/build_design_ph_axis_inputs.py

required_packages <- c("dplyr", "ggplot2", "tidyr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ", paste(missing_packages, collapse = ", "), "\n",
    "Run the PCA notebook setup cell first, or install them with:\n",
    "install.packages(c(", paste(sprintf('\"%s\"', missing_packages), collapse = ", "), "))",
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

path_first_existing <- function(paths) {
  hits <- paths[file.exists(paths)]
  if (length(hits) == 0) {
    stop("None of these paths exists:\n", paste(paths, collapse = "\n"))
  }
  hits[[1]]
}

if (!exists("df")) {
  big_csv <- path_first_existing(c(
    Sys.getenv("BIG_CSV", unset = ""),
    "dataset_update/main_plus_r2_r3_analysis_v12_cli.csv",
    "/content/main_plus_r2_r3_analysis_v12_cli.csv"
  ))
  message("Loading main dataset from: ", big_csv)
  df <- read.csv(big_csv, stringsAsFactors = FALSE)
}

design_ph_dir <- Sys.getenv(
  "DESIGN_PH_DIR",
  unset = "/Users/lauradillon/Downloads/Designs/ph_axis_features"
)
out_dir <- Sys.getenv("DESIGN_PH_OUT_DIR", unset = design_ph_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

designs_ph <- read.csv(file.path(design_ph_dir, "designs_ph_features.csv"), stringsAsFactors = FALSE)
wt_ph <- read.csv(file.path(design_ph_dir, "wt_ph_features.csv"), stringsAsFactors = FALSE)

pH_features <- intersect(
  c(
    "buffer_capacity",
    "charge_per_residue",
    "acidic_residue_fraction",
    "basic_residue_fraction",
    "ionizable_residue_fraction",
    "isoelectric_point",
    "charge_at_ph7"
  ),
  colnames(df)
)

if (length(pH_features) < 3) {
  stop("Too few pH features in df: ", paste(pH_features, collapse = ", "))
}
if (!all(pH_features %in% colnames(designs_ph)) || !all(pH_features %in% colnames(wt_ph))) {
  stop("Design/WT pH feature tables are missing at least one of: ", paste(pH_features, collapse = ", "))
}

model_levels <- c("ProteinMPNN_v002", "AlkSecMPNN", "AcidSecMPNN")
model_colors <- c(
  "ProteinMPNN_v002" = "#7f8c8d",
  "AlkSecMPNN" = "#c0392b",
  "AcidSecMPNN" = "#2980b9"
)

# Fit the pH PCA in the natural-protein feature frame.
cc <- df[stats::complete.cases(df[, pH_features, drop = FALSE]), ]
ph_pca <- prcomp(cc[, pH_features, drop = FALSE], center = TRUE, scale. = TRUE)

# Orient +PC1 toward acidification: higher D/E fraction, lower pI / more negative charge.
if ("acidic_residue_fraction" %in% rownames(ph_pca$rotation) &&
    ph_pca$rotation["acidic_residue_fraction", "PC1"] < 0) {
  ph_pca$rotation[, "PC1"] <- -ph_pca$rotation[, "PC1"]
  ph_pca$x[, "PC1"] <- -ph_pca$x[, "PC1"]
}

score_center <- colMeans(ph_pca$x[, 1:2, drop = FALSE])
score_scale <- apply(ph_pca$x[, 1:2, drop = FALSE], 2, sd)
var_explained <- (ph_pca$sdev^2 / sum(ph_pca$sdev^2))[1:2]

project_ph <- function(tab) {
  mat <- as.matrix(tab[, pH_features, drop = FALSE])
  scaled <- scale(mat, center = ph_pca$center[pH_features], scale = ph_pca$scale[pH_features])
  scaled[!is.finite(scaled)] <- 0
  z <- scaled %*% ph_pca$rotation[pH_features, 1:2, drop = FALSE]
  z <- sweep(z, 2, score_center, "-")
  z <- sweep(z, 2, score_scale, "/")
  data.frame(PC1 = z[, 1], PC2 = z[, 2])
}

natural_xy <- cbind(
  cc[, intersect(c("Entry", "domain"), colnames(cc)), drop = FALSE],
  project_ph(cc)
)
design_xy <- cbind(
  designs_ph[, intersect(
    c("uniprot_id", "species", "domain", "rank_class", "model", "sample_idx", "seed",
      "isoelectric_point", "charge_at_ph7", "acidic_residue_fraction", "basic_residue_fraction"),
    colnames(designs_ph)
  ), drop = FALSE],
  project_ph(designs_ph)
) %>%
  filter(model %in% model_levels) %>%
  mutate(model = factor(model, levels = model_levels))

wt_xy <- cbind(
  wt_ph[, intersect(
    c("uniprot_id", "isoelectric_point", "charge_at_ph7", "acidic_residue_fraction", "basic_residue_fraction"),
    colnames(wt_ph)
  ), drop = FALSE],
  project_ph(wt_ph)
) %>%
  rename(
    PC1_wt = PC1,
    PC2_wt = PC2,
    pI_wt = isoelectric_point,
    charge_at_ph7_wt = charge_at_ph7,
    acidic_fraction_wt = acidic_residue_fraction,
    basic_fraction_wt = basic_residue_fraction
  )

design_shift <- design_xy %>%
  left_join(wt_xy, by = "uniprot_id") %>%
  mutate(
    shift_PC1 = PC1 - PC1_wt,
    shift_PC2 = PC2 - PC2_wt,
    shift_mag = sqrt(shift_PC1^2 + shift_PC2^2),
    delta_pI = isoelectric_point - pI_wt,
    delta_charge_at_ph7 = charge_at_ph7 - charge_at_ph7_wt,
    delta_acidic_fraction = acidic_residue_fraction - acidic_fraction_wt,
    delta_basic_fraction = basic_residue_fraction - basic_fraction_wt
  )

protein_model_shift <- design_shift %>%
  group_by(uniprot_id, domain, rank_class, model) %>%
  summarise(
    PC1_design = mean(PC1, na.rm = TRUE),
    PC2_design = mean(PC2, na.rm = TRUE),
    PC1_wt = first(PC1_wt),
    PC2_wt = first(PC2_wt),
    shift_PC1 = mean(shift_PC1, na.rm = TRUE),
    shift_PC2 = mean(shift_PC2, na.rm = TRUE),
    shift_mag = sqrt(shift_PC1^2 + shift_PC2^2),
    delta_pI = mean(delta_pI, na.rm = TRUE),
    delta_charge_at_ph7 = mean(delta_charge_at_ph7, na.rm = TRUE),
    delta_acidic_fraction = mean(delta_acidic_fraction, na.rm = TRUE),
    delta_basic_fraction = mean(delta_basic_fraction, na.rm = TRUE),
    n_designs = dplyr::n(),
    .groups = "drop"
  )

summary_by_model <- protein_model_shift %>%
  group_by(model) %>%
  summarise(
    n_wt = dplyr::n(),
    mean_shift_PC1 = mean(shift_PC1, na.rm = TRUE),
    se_shift_PC1 = sd(shift_PC1, na.rm = TRUE) / sqrt(n_wt),
    ci95_shift_PC1 = qt(0.975, df = n_wt - 1) * se_shift_PC1,
    mean_shift_PC2 = mean(shift_PC2, na.rm = TRUE),
    mean_shift_mag = mean(shift_mag, na.rm = TRUE),
    mean_delta_pI = mean(delta_pI, na.rm = TRUE),
    mean_delta_charge_at_ph7 = mean(delta_charge_at_ph7, na.rm = TRUE),
    mean_delta_acidic_fraction = mean(delta_acidic_fraction, na.rm = TRUE),
    mean_delta_basic_fraction = mean(delta_basic_fraction, na.rm = TRUE),
    .groups = "drop"
  )

paired_contrast <- function(wide, a, b) {
  if (!all(c(a, b) %in% names(wide))) {
    return(data.frame(contrast = paste(a, "-", b), mean_delta = NA_real_, p_value = NA_real_))
  }
  d <- wide[[a]] - wide[[b]]
  data.frame(
    contrast = paste(a, "-", b),
    mean_delta = mean(d, na.rm = TRUE),
    p_value = if (sum(!is.na(d)) >= 3) stats::t.test(d)$p.value else NA_real_
  )
}

shift_wide <- protein_model_shift %>%
  select(uniprot_id, model, shift_PC1) %>%
  tidyr::pivot_wider(names_from = model, values_from = shift_PC1)
contrasts <- bind_rows(
  paired_contrast(shift_wide, "AlkSecMPNN", "ProteinMPNN_v002"),
  paired_contrast(shift_wide, "AcidSecMPNN", "ProteinMPNN_v002"),
  paired_contrast(shift_wide, "AlkSecMPNN", "AcidSecMPNN")
)

write.csv(design_shift, file.path(out_dir, "design_ph_pca_coords.csv"), row.names = FALSE)
write.csv(protein_model_shift, file.path(out_dir, "design_ph_shift_vectors.csv"), row.names = FALSE)
write.csv(summary_by_model, file.path(out_dir, "design_ph_axis_summary.csv"), row.names = FALSE)
write.csv(contrasts, file.path(out_dir, "design_ph_axis_model_contrasts.csv"), row.names = FALSE)

p_shift <- ggplot() +
  geom_point(data = natural_xy, aes(PC1, PC2), color = "grey88", alpha = 0.35, size = 0.7) +
  geom_segment(
    data = protein_model_shift,
    aes(x = PC1_wt, y = PC2_wt, xend = PC1_design, yend = PC2_design, color = model),
    arrow = grid::arrow(length = grid::unit(0.12, "cm")),
    linewidth = 0.35,
    alpha = 0.8
  ) +
  geom_point(data = protein_model_shift, aes(PC1_wt, PC2_wt), shape = 4, size = 1.8, stroke = 0.8) +
  scale_color_manual(values = model_colors, drop = FALSE) +
  facet_wrap(~ model, nrow = 1) +
  coord_equal() +
  labs(
    x = sprintf("pH PC1 z-score (%.0f%% var; + = acidic / low pI)", var_explained[1] * 100),
    y = sprintf("pH PC2 z-score (%.0f%% var)", var_explained[2] * 100),
    title = "WT -> design shifts in pH-feature PCA space",
    subtitle = "Grey = natural proteins; x = WT; arrows end at each model's per-protein design centroid"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none")

p_bar <- ggplot(summary_by_model, aes(model, mean_shift_PC1, fill = model)) +
  geom_col(width = 0.62, alpha = 0.9) +
  geom_errorbar(
    aes(ymin = mean_shift_PC1 - ci95_shift_PC1, ymax = mean_shift_PC1 + ci95_shift_PC1),
    width = 0.18,
    linewidth = 0.7
  ) +
  geom_hline(yintercept = 0, linewidth = 0.3) +
  scale_fill_manual(values = model_colors, drop = FALSE) +
  labs(
    x = NULL,
    y = "mean WT -> design shift on pH PC1\n(+ acidifying; - basifying)",
    title = "pH-axis design shift by model",
    subtitle = "Error bars are 95% CIs across the 25 WT proteins"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

ggsave(file.path(out_dir, "design_ph_wt_to_design_shift.png"), p_shift, width = 12, height = 4.8, dpi = 300, bg = "white")
ggsave(file.path(out_dir, "design_ph_axis_shift_summary.png"), p_bar, width = 7.5, height = 5, dpi = 300, bg = "white")

message("pH PCA features: ", paste(pH_features, collapse = ", "))
message(sprintf("pH PCA variance: PC1 %.1f%%, PC2 %.1f%%", var_explained[1] * 100, var_explained[2] * 100))
print(summary_by_model)
print(contrasts)
message("Wrote pH-axis design outputs to: ", out_dir)
