# Export exact figure grids using the same fitting functions as the reproduction.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_path <- normalizePath(gsub("~+~", " ", sub("^--file=", "", script_arg[1]), fixed = TRUE), mustWork = TRUE)
source(file.path(dirname(script_path), "../src/decoding_bias/analysis/gam_landscapes.R"))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript scripts/export_gam_grids.R INPUT.csv OUTPUT_DIR [MODEL_COLUMNS]")
requested <- if (length(args) >= 3) strsplit(args[3], ",", fixed = TRUE)[[1]] else NULL
export_gam_grids(args[1], args[2], requested)
