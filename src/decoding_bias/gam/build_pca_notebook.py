#!/usr/bin/env python3
"""Builds PCA_paper_figures.ipynb (R kernel). Run: python build_pca_notebook.py"""
import json, pathlib

CELLS = []
def md(s):   CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))

md(r"""# Paper PCA / GAM figures - clean rebuild

Self-contained R notebook that produces **only** the PCA-space figures and tables the paper needs,
with improved presentation (paletteer palettes, `coord_equal`, density contours, hard-masked GAM
surfaces) and three new centroid views (per-species, per-family, per-broad-function).

**Inputs** (auto-found in `/content`, `./data`, `./inputs`, or `.`):
`main_plus_r2_r3_analysis_v12_cli.csv` (required); `designs_features.csv`, `wt_features.csv`,
`designs_ph_features.csv` (only for Fig 4 / Fig 5).

**Feature set:** 14 mixed features (9 sequence + 5 structure); `charge_at_ph7` and
`small_residue_fraction` dropped for collinearity.

**Outputs** → `pca_paper_outputs/{figures,tables}/`:

| file | paper destination |
|---|---|
| `fig3_scatter_density.png`, `fig3_pca_guide.png`, `fig3A_combined.png` | **Fig 3A** |
| `fig3_gam_landscapes_*.png` | **Fig 3B** + SI `fig:s-gam` |
| `fig3_variance_pareto.png` | SI scree |
| `fig3_centroids_{species,family,function}.png` | new SI / exploratory panels |
| `fig4_design_centroids.png` | **Fig 4** centroid panel |
| `fig5_ft_surface_consolidated.png` | **Fig 5** + SI `fig:s-ft-surface` |
| `pca_loadings.csv` | SI `tab:s-pca-loadings` |
| `mixed_features_pca_coordinates.csv` | feeds the variance decomposition |
| `pca_variance.csv`, `gam_deviance.csv`, `compactness_{perdomain,pairwise}.csv` | SI tables |
| `ft_surface_centroid_base_relative.csv` | main Table `tab:ft_surface_pca` |
""")

code(r"""# === 0. Packages =====================================================
pkgs <- c("tidyverse","mgcv","patchwork","ggrepel","paletteer","scico","scales","ggridges","hexbin")
new  <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(new)) install.packages(new, quiet = TRUE)
suppressMessages(invisible(lapply(pkgs, library, character.only = TRUE)))
cat("Packages loaded.\n")""")

code(r"""# === 1. Inputs (Colab /content or local) =============================
find_file <- function(name) {
  cand <- c(file.path("/content", name), file.path("data", name),
            file.path("inputs", name), name)
  hit  <- cand[file.exists(cand)]
  if (length(hit)) hit[1] else NA_character_
}
BIG_CSV <- find_file("main_plus_r2_r3_analysis_v12_cli.csv")
stopifnot(!is.na(BIG_CSV))
DES_CSV <- find_file("designs_features.csv")      # Fig 4 (optional)
WT_CSV  <- find_file("wt_features.csv")           # Fig 4 (optional)
PH_CSV  <- find_file("designs_ph_features.csv")   # Fig 5 (optional)

OUT <- "pca_paper_outputs"; FIG <- file.path(OUT,"figures"); TAB <- file.path(OUT,"tables")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(BIG_CSV, stringsAsFactors = FALSE, check.names = FALSE)
cat(sprintf("Loaded %d proteins x %d cols from %s\n", nrow(df), ncol(df), BIG_CSV))""")

code(r"""# === 2. Feature set, theme, palettes =================================
sequence_features <- c("sequence_length","mw_per_residue","isoelectric_point",
  "acidic_residue_fraction","basic_residue_fraction","gravy","aromaticity",
  "instability_index","proline_fraction")                                  # 9
structure_features <- c("ordered_percent","helix_sheet_contrast","rco",
  "avg_cb_distance","surface_exposure")                                    # 5
mixed_features <- c(sequence_features, structure_features)                 # 14
stopifnot(all(mixed_features %in% names(df)))

pretty_feat <- c(sequence_length="sequence length", mw_per_residue="MW / residue",
  isoelectric_point="isoelectric point", acidic_residue_fraction="acidic fraction",
  basic_residue_fraction="basic fraction", gravy="GRAVY (hydrophobic)",
  aromaticity="aromaticity", instability_index="instability",
  proline_fraction="proline", ordered_percent="ordered %",
  helix_sheet_contrast="helix-sheet", rco="contact order (RCO)",
  avg_cb_distance="C-beta distance", surface_exposure="surface exposure")

theme_paper <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 12),
        axis.title = element_text(face = "bold"),
        legend.position = "bottom")
theme_set(theme_paper)

domain_levels <- c("Archaea","Bacteria","Eukaryota")
df$domain <- factor(df$domain, levels = domain_levels)
cb <- as.character(paletteer::paletteer_d("ggthemes::colorblind"))   # Okabe-Ito (colourblind-safe)
domain_pal <- c(Archaea = cb[7], Bacteria = cb[6], Eukaryota = cb[4])  # vermillion / blue / green

# diverging fill for z-scored GAM surfaces, centred at 0
zfill <- function(name = "Preference z\n(SD from mean)", lim = 2)
  scale_fill_gradientn(name = name,
    colours = as.character(paletteer::paletteer_c("scico::vik", 64)),
    limits = c(-lim, lim), oob = scales::squish, na.value = "white")
cat(sprintf("Feature set: %d (%d seq + %d struct)\n",
            length(mixed_features), length(sequence_features), length(structure_features)))""")

code(r"""# === 3. Mixed 14-feature PCA (oriented to paper convention) ==========
fit_mixed_pca <- function(df, feats) {
  cc <- df[complete.cases(df[, feats]), , drop = FALSE]
  X  <- scale(cc[, feats]); pr <- prcomp(X, center = FALSE, scale. = FALSE)
  ve <- pr$sdev^2 / sum(pr$sdev^2); rot <- pr$rotation
  # PC1+ = large/hydrophobic (sequence length +);  PC2+ = heavy/less-stable (instability +)
  if (rot["sequence_length","PC1"]  < 0) { rot[,"PC1"] <- -rot[,"PC1"]; pr$x[,"PC1"] <- -pr$x[,"PC1"] }
  if (rot["instability_index","PC2"] < 0) { rot[,"PC2"] <- -rot[,"PC2"]; pr$x[,"PC2"] <- -pr$x[,"PC2"] }
  score_cols <- grep("_score$", names(cc), value = TRUE)
  sc <- data.frame(Entry = cc$Entry, domain = cc$domain, species = cc$species,
                   protein_family = cc$protein_family, broad_function = cc$broad_function,
                   PC1 = pr$x[,1], PC2 = pr$x[,2], PC3 = pr$x[,3], check.names = FALSE)
  sc <- cbind(sc, cc[, score_cols, drop = FALSE])
  list(scores = sc,
       loadings = data.frame(feature = feats, PC1 = rot[,1], PC2 = rot[,2], row.names = NULL),
       ve = ve, center = attr(X,"scaled:center"), scale = attr(X,"scaled:scale"), rotation = rot)
}
P <- fit_mixed_pca(df, mixed_features)
cat(sprintf("PCA: PC1 %.1f%%  PC2 %.1f%%  (n=%d)\n", 100*P$ve[1], 100*P$ve[2], nrow(P$scores)))""")

code(r"""# === 4. Tables: loadings, coordinates, variance, compactness =========
lo <- P$loadings; lo$pretty <- pretty_feat[lo$feature]
write.csv(lo, file.path(TAB,"pca_loadings.csv"), row.names = FALSE)
write.csv(P$scores[, c("Entry","PC1","PC2","domain")],
          file.path(TAB,"mixed_features_pca_coordinates.csv"), row.names = FALSE)
vartab <- data.frame(PC = paste0("PC", seq_along(P$ve)),
                     variance_pct = round(100*P$ve, 2),
                     cumulative_pct = round(100*cumsum(P$ve), 2))
write.csv(head(vartab, 8), file.path(TAB,"pca_variance.csv"), row.names = FALSE)

bhatt <- function(a, b) {                   # Bhattacharyya overlap of two bivariate Gaussians
  ma <- colMeans(a); mb <- colMeans(b); Sa <- cov(a); Sb <- cov(b); S <- (Sa+Sb)/2
  d  <- as.numeric(0.125*t(ma-mb) %*% solve(S) %*% (ma-mb) + 0.5*log(det(S)/sqrt(det(Sa)*det(Sb))))
  exp(-d)
}
dl <- split(P$scores[, c("PC1","PC2")], P$scores$domain)
perdom <- do.call(rbind, lapply(names(dl), function(k) data.frame(domain = k, n = nrow(dl[[k]]),
            centroid_PC1 = round(mean(dl[[k]]$PC1),3), centroid_PC2 = round(mean(dl[[k]]$PC2),3),
            ellipse_area = round(pi*prod(sqrt(eigen(cov(dl[[k]]))$values)),2))))
pairtab <- do.call(rbind, lapply(combn(names(dl), 2, simplify = FALSE), function(p)
  data.frame(pair = paste(p, collapse = "-vs-"),
             bhattacharyya_overlap = round(bhatt(as.matrix(dl[[p[1]]]), as.matrix(dl[[p[2]]])),3))))
write.csv(perdom,  file.path(TAB,"compactness_perdomain.csv"), row.names = FALSE)
write.csv(pairtab, file.path(TAB,"compactness_pairwise.csv"),  row.names = FALSE)
print(perdom); print(pairtab)""")

code(r"""# === 5. Shared helpers ===============================================
pc_lab  <- function(P, i) sprintf("PC%d (%.1f%%)", i, 100*P$ve[i])
sym_lim <- function(P, ex = 0.05) { L <- max(abs(c(P$scores$PC1, P$scores$PC2)))*(1+ex); c(-L, L) }
axis_terms <- function(lo, pc, sign, k = 3) {           # plain-language axis caption from loadings
  v <- sort(setNames(lo[[pc]], lo$feature), decreasing = sign > 0)[1:k]
  paste(pretty_feat[names(v)], collapse = ", ")
}
project_into_pca <- function(newdata, P) {              # place new proteins in the fitted PCA
  X  <- as.matrix(newdata[, names(P$center), drop = FALSE])
  Xs <- scale(X, center = P$center, scale = P$scale); Xs[!is.finite(Xs)] <- 0
  Z  <- Xs %*% P$rotation[, 1:2]; data.frame(PC1 = Z[,1], PC2 = Z[,2])
}
dom_mode <- function(x) names(which.max(table(x)))""")

code(r"""# === 6. Fig 3A (primary) - faceted per-domain density contours =======
sc <- P$scores; lim <- sym_lim(P)
bg <- transform(sc, domain = NULL)                       # shared grey background in every facet
p_facet <- ggplot(sc, aes(PC1, PC2)) +
  geom_point(data = bg, colour = "grey88", size = .35, alpha = .25) +
  geom_density_2d(aes(colour = domain), bins = 7, linewidth = .5) +
  geom_point(data = aggregate(cbind(PC1,PC2) ~ domain, sc, mean),
             aes(colour = domain), shape = 4, size = 4, stroke = 1.6, show.legend = FALSE) +
  geom_hline(yintercept=0,linetype=2,colour="grey80") + geom_vline(xintercept=0,linetype=2,colour="grey80") +
  scale_colour_manual(values = domain_pal, guide = "none") +
  facet_wrap(~ domain) + coord_equal(xlim = lim, ylim = lim) +
  labs(x = pc_lab(P,1), y = pc_lab(P,2),
       title = "Each domain's density over the shared biophysical plane")
ggsave(file.path(FIG,"fig3A_faceted_contours.png"), p_facet, width=11, height=4.4, dpi=300, bg="white")
print(p_facet)""")

code(r"""# === 6b. Fig 3A guide - hexbin density + loading arrows ==============
lo <- P$loadings; lo$mag <- sqrt(lo$PC1^2 + lo$PC2^2)
top <- lo[order(-lo$mag), ][1:8, ]; s <- max(abs(c(sc$PC1,sc$PC2)))*0.45/max(top$mag)
top$x <- top$PC1*s; top$y <- top$PC2*s; top$lab <- pretty_feat[top$feature]
p_guide <- ggplot(sc, aes(PC1, PC2)) +
  geom_hex(bins = 45) + scale_fill_gradient(low="grey92", high="grey55", guide="none") +
  geom_hline(yintercept=0,linetype=2,colour="grey70") + geom_vline(xintercept=0,linetype=2,colour="grey70") +
  geom_segment(data=top, aes(0,0,xend=x,yend=y), arrow=arrow(length=unit(.14,"in"),type="closed"),
               linewidth=.7, colour="grey15") +
  ggrepel::geom_text_repel(data=top, aes(x,y,label=lab), size=3.4, fontface="bold", seed=1, max.overlaps=Inf) +
  annotate("text", x=lim[2], y=0, hjust=1, vjust=-.4, size=3.1, colour="#1f6f8b", label=axis_terms(lo,"PC1",+1)) +
  annotate("text", x=lim[1], y=0, hjust=0, vjust=-.4, size=3.1, colour="#b5651d", label=axis_terms(lo,"PC1",-1)) +
  annotate("text", x=0, y=lim[2], hjust=.5, vjust=1,  size=3.1, colour="#6a3d9a", label=axis_terms(lo,"PC2",+1)) +
  annotate("text", x=0, y=lim[1], hjust=.5, vjust=0,  size=3.1, colour="#2e7d32", label=axis_terms(lo,"PC2",-1)) +
  coord_equal(xlim=lim, ylim=lim, clip="off") +
  labs(x=pc_lab(P,1), y=pc_lab(P,2), title="Biophysical axes (hexbin density + loadings)")
ggsave(file.path(FIG,"fig3A_hexbin_guide.png"), p_guide, width=6.5, height=6, dpi=300, bg="white")
print(p_guide)""")

code(r"""# === 6c. Ridgeline pair - PC1 and PC2 distributions by domain ========
scr <- P$scores; scr$domain <- factor(scr$domain, rev(domain_levels))   # Archaea at bottom
ridge <- function(col, ttl) ggplot(scr, aes(.data[[col]], domain, fill = domain)) +
  ggridges::geom_density_ridges(scale = 1.5, alpha = .85, colour = "white", linewidth = .4) +
  scale_fill_manual(values = domain_pal, guide = "none") +
  labs(x = pc_lab(P, as.integer(sub("PC","",col))), y = NULL, title = ttl) +
  theme(panel.grid.major.y = element_blank())
p_ridge <- ridge("PC1","PC1 by domain  (basic / compact <- -> large / hydrophobic)") +
           ridge("PC2","PC2 by domain  (ordered <- -> heavy / less-stable)")
ggsave(file.path(FIG,"fig3_ridgeline.png"), p_ridge, width=11, height=4, dpi=300, bg="white")
print(p_ridge)""")

code(r"""# === 7. Scree as a single Pareto panel ===============================
k <- min(6, length(P$ve))
vd <- data.frame(PC=factor(paste0("PC",1:k), paste0("PC",1:k)),
                 var=100*P$ve[1:k], cum=100*cumsum(P$ve[1:k]))
p_scree <- ggplot(vd, aes(PC)) +
  geom_col(aes(y=var), fill="#2c3e50", width=.7, alpha=.85) +
  geom_text(aes(y=var, label=sprintf("%.1f%%",var)), vjust=-.5, size=3.2, fontface="bold") +
  geom_line(aes(y=cum, group=1), colour="#e74c3c", linewidth=1.1) +
  geom_point(aes(y=cum), colour="#e74c3c", size=2.6) +
  geom_text(aes(y=cum, label=sprintf("%.0f%%",cum)), vjust=-.9, size=2.9, colour="#c0392b", fontface="bold") +
  scale_y_continuous("Variance explained (%)", limits=c(0,100), breaks=seq(0,100,25)) +
  labs(x="Principal component", title="PCA variance (bars) and cumulative (red line)",
       subtitle=sprintf("PC1 + PC2 = %.0f%% of the %d-feature variance", vd$cum[2], length(mixed_features)))
ggsave(file.path(FIG,"fig3_variance_pareto.png"), p_scree, width=7, height=4.6, dpi=300, bg="white")
print(p_scree)""")

code(r"""# === 8. Fig 3B - GAM preference landscapes (HARD-masked) =============
# Low-density cells AND everything outside the data's convex hull are set to NA
# (white), so no extrapolated corners are ever drawn.
preference_landscape <- function(P, score_col, grid_n=160, min_density=3, density_bins=55, lim=2) {
  d <- P$scores[, c("PC1","PC2", score_col)]; d <- d[complete.cases(d), ]
  if (nrow(d) < 60) return(NULL)
  d[[score_col]] <- (d[[score_col]] - mean(d[[score_col]])) / sd(d[[score_col]])
  k <- max(12, min(50, floor(nrow(d)/300)))
  m <- tryCatch(mgcv::gam(as.formula(sprintf("%s ~ s(PC1,PC2,k=%d)", score_col, k)),
                          data=d, method="REML"), error=function(e) NULL)
  if (is.null(m)) return(NULL)
  L <- sym_lim(P)
  g <- expand.grid(PC1=seq(L[1],L[2],length.out=grid_n), PC2=seq(L[1],L[2],length.out=grid_n))
  g$pred <- as.numeric(predict(m, g))
  bx <- seq(L[1],L[2],length.out=density_bins+1); by <- bx
  cnt <- table(factor(pmin(pmax(findInterval(d$PC1,bx,all.inside=TRUE),1),density_bins),1:density_bins),
               factor(pmin(pmax(findInterval(d$PC2,by,all.inside=TRUE),1),density_bins),1:density_bins))
  gi <- pmin(pmax(findInterval(g$PC1,bx,all.inside=TRUE),1),density_bins)
  gj <- pmin(pmax(findInterval(g$PC2,by,all.inside=TRUE),1),density_bins)
  low <- cnt[cbind(gi,gj)] < min_density
  h <- grDevices::chull(d$PC1,d$PC2); bnd <- as.matrix(d[c(h,h[1]), c("PC1","PC2")])
  out <- !mgcv::in.out(bnd, as.matrix(g[,c("PC1","PC2")]))
  g$pred[low | out] <- NA; g$ok <- !(low | out)
  gk <- subset(g, ok); brk <- seq(-lim, lim, by = 0.5)
  gk$predc <- pmax(pmin(gk$pred, lim - 1e-6), -lim + 1e-6)            # clamp into band range
  bandcols <- as.character(paletteer::paletteer_c("scico::vik", length(brk) - 1))
  p <- ggplot(gk, aes(PC1, PC2, z = predc)) +
    geom_contour_filled(breaks = brk) +
    scale_fill_manual(values = bandcols, name = "Preference z\n(SD from mean)", drop = FALSE,
                      guide = guide_legend(reverse = TRUE)) +
    coord_equal(xlim = L, ylim = L, expand = FALSE) +
    labs(title=gsub("_score","",score_col), subtitle=sprintf("dev.expl %.1f%%", 100*summary(m)$dev.expl),
         x=NULL, y=NULL) +
    theme(panel.grid=element_blank(), panel.background=element_rect(fill="white",colour=NA),
          plot.subtitle=element_text(size=8.5,hjust=.5,colour="grey35"), plot.title=element_text(size=10,hjust=.5))
  attr(p,"dev") <- 100*summary(m)$dev.expl; p
}

groups <- list(
  structure = c("ProteinMPNN_v020_score","solublempnn_score","caliby_score","soluble_caliby_score",
                "esmif_score","triflow_score","esm3_struct_cond_score","mif_score"),
  sequence  = c("ESM2_15B_pppl_score","esm3_seq_only_score","carp_640M_score",
                "progen2_score","progen2_XL_score","protgpt2_score","mifst_score"),
  finetune  = c("ProteinMPNN_v020_score","AlkalineMPNN_020_score","AcidophileMPNN_020_score"))
devrows <- list()
for (gn in names(groups)) {
  ms <- intersect(groups[[gn]], names(P$scores))
  ps <- list()
  for (s in ms) { pl <- preference_landscape(P, s); if (!is.null(pl)) { ps[[s]] <- pl
                  devrows[[s]] <- data.frame(model=gsub("_score","",s), dev_expl_pct=round(attr(pl,"dev"),1)) } }
  if (!length(ps)) next
  g <- patchwork::wrap_plots(ps) + patchwork::plot_layout(guides="collect") &
       theme(legend.position="right")
  ggsave(file.path(FIG, sprintf("fig3_gam_landscapes_%s.png", gn)), g, width=11, height=8, dpi=300, bg="white")
}
gam_tbl <- do.call(rbind, devrows); gam_tbl <- gam_tbl[order(-gam_tbl$dev_expl_pct), ]
write.csv(gam_tbl, file.path(TAB,"gam_deviance.csv"), row.names=FALSE)
print(gam_tbl, row.names=FALSE)""")

code(r"""# === 9. NEW - per-species centroids (coloured by domain) =============
sp <- aggregate(cbind(PC1,PC2) ~ species + domain, P$scores, mean)
sp_n <- aggregate(PC1 ~ species, P$scores, length); names(sp_n)[2] <- "n"
sp <- merge(sp, sp_n, by="species")
p_sp <- ggplot(sp, aes(PC1, PC2, colour=domain)) +
  geom_hline(yintercept=0,linetype=2,colour="grey85") + geom_vline(xintercept=0,linetype=2,colour="grey85") +
  geom_point(aes(size=n), alpha=.75) +
  stat_ellipse(type="norm", level=0.68, linewidth=1) +
  scale_colour_manual(values=domain_pal, name="Domain") +
  scale_size_continuous(range=c(1,5), name="proteins / species") +
  coord_equal(xlim=lim, ylim=lim) +
  labs(title=sprintf("Per-species centroids (%d species)", nrow(sp)), x=pc_lab(P,1), y=pc_lab(P,2))
ggsave(file.path(FIG,"fig3_centroids_species.png"), p_sp, width=6.8, height=6.4, dpi=300, bg="white")
print(p_sp)""")

code(r"""# === 10. Broad functions on the biophysical axes (lollipop) ==========
fn  <- aggregate(cbind(PC1,PC2) ~ broad_function, P$scores, mean)
fnn <- aggregate(PC1 ~ broad_function, P$scores, length); names(fnn)[2] <- "n"
fnd <- do.call(rbind, by(P$scores, P$scores$broad_function,
        function(d) data.frame(broad_function=d$broad_function[1], dom=dom_mode(d$domain))))
fn  <- merge(merge(fn, fnn), fnd); fn$dom <- factor(fn$dom, domain_levels)
lolli <- function(col, xlab) { fn$ord <- reorder(fn$broad_function, fn[[col]])
  ggplot(fn, aes(.data[[col]], ord, colour = dom)) +
    geom_vline(xintercept = 0, linetype = 2, colour = "grey70") +
    geom_segment(aes(x = 0, xend = .data[[col]], yend = ord), colour = "grey75", linewidth = .8) +
    geom_point(aes(size = n)) +
    scale_colour_manual(values = domain_pal, name = "Dominant domain") +
    scale_size_continuous(range = c(2,7), name = "n proteins") +
    labs(x = xlab, y = NULL) }
p_lf1 <- lolli("PC1", paste0(pc_lab(P,1), "  (basic/compact <- -> large/hydrophobic)")) +
         ggtitle("Broad functions ranked on PC1")
p_lf2 <- lolli("PC2", paste0(pc_lab(P,2), "  (ordered <- -> heavy/unstable)")) +
         ggtitle("Broad functions ranked on PC2")
ggsave(file.path(FIG,"fig_lollipop_function_PC1.png"), p_lf1, width=7, height=8, dpi=300, bg="white")
ggsave(file.path(FIG,"fig_lollipop_function_PC2.png"), p_lf2, width=7, height=8, dpi=300, bg="white")
print(p_lf1)""")

code(r"""# === 11. Most extreme protein families on PC1 (lollipop) =============
fam  <- aggregate(cbind(PC1,PC2) ~ protein_family, P$scores, mean)
famn <- aggregate(PC1 ~ protein_family, P$scores, length); names(famn)[2] <- "n"
famd <- do.call(rbind, by(P$scores, P$scores$protein_family,
         function(d) data.frame(protein_family=d$protein_family[1], dom=dom_mode(d$domain))))
fam  <- merge(merge(fam, famn), famd); fam$dom <- factor(fam$dom, domain_levels)
ord  <- fam[order(fam$PC1), ]; ext <- rbind(head(ord, 20), tail(ord, 20))
ext$ord <- reorder(ext$protein_family, ext$PC1)
p_famlol <- ggplot(ext, aes(PC1, ord, colour = dom)) +
  geom_vline(xintercept = 0, linetype = 2, colour = "grey70") +
  geom_segment(aes(x = 0, xend = PC1, yend = ord), colour = "grey75", linewidth = .7) +
  geom_point(aes(size = n)) +
  scale_colour_manual(values = domain_pal, name = "Dominant domain") +
  scale_size_continuous(range = c(2,6), name = "n proteins") +
  labs(x = paste0(pc_lab(P,1), "  (basic/compact <- -> large/hydrophobic)"), y = NULL,
       title = "Most extreme protein families on PC1 (top/bottom 20)")
ggsave(file.path(FIG,"fig_lollipop_family_PC1.png"), p_famlol, width=7.5, height=8.5, dpi=300, bg="white")
print(p_famlol)""")

code(r"""# === 12. Fig 4 - design centroid shifts (6 structure models) =========
if (!is.na(DES_CSV) && !is.na(WT_CSV)) {
  des <- read.csv(DES_CSV, stringsAsFactors=FALSE); wt <- read.csv(WT_CSV, stringsAsFactors=FALSE)
  id_wt <- if ("uniprot_id" %in% names(wt)) "uniprot_id" else "Entry"
  D6 <- c("ProteinMPNN","SolubleMPNN","Caliby","SolubleCaliby","ESM-IF","MIF")
  des <- des[des$model %in% D6 & complete.cases(des[, mixed_features]), ]
  if (nrow(des) > 0) {
    des_xy <- cbind(des[, c("uniprot_id","model")], project_into_pca(des, P))
    wt_xy  <- data.frame(uniprot_id=wt[[id_wt]], project_into_pca(wt, P))
    names(wt_xy)[2:3] <- c("PC1w","PC2w")
    dd  <- merge(des_xy, wt_xy, by="uniprot_id")
    per <- aggregate(cbind(PC1,PC2,PC1w,PC2w) ~ uniprot_id + model, dd, mean)
    per$dPC1 <- per$PC1-per$PC1w; per$dPC2 <- per$PC2-per$PC2w
    cen <- aggregate(cbind(dPC1,dPC2) ~ model, per, mean); cen$model <- factor(cen$model, D6)
    mpal <- setNames(as.character(paletteer::paletteer_d("ggsci::default_jco"))[1:length(D6)], D6)
    Lk <- max(abs(c(cen$dPC1,cen$dPC2)))*1.3
    p4 <- ggplot(cen) +
      geom_hline(yintercept=0,linetype=2,colour="grey80") + geom_vline(xintercept=0,linetype=2,colour="grey80") +
      geom_segment(aes(0,0,xend=dPC1,yend=dPC2,colour=model),
                   arrow=arrow(length=unit(.18,"in"),type="closed"), linewidth=1.2) +
      geom_point(aes(0,0), shape=4, size=4, stroke=1.5) +
      scale_colour_manual(values=mpal, name=NULL) + coord_equal(xlim=c(-Lk,Lk), ylim=c(-Lk,Lk)) +
      labs(title="Design centroid shifts (WT -> design mean)", x=pc_lab(P,1), y=pc_lab(P,2))
    ggsave(file.path(FIG,"fig4_design_centroids.png"), p4, width=6.5, height=6.5, dpi=300, bg="white")
    print(p4)
  } else cat("Fig 4 skipped: no matching design rows.\n")
} else cat("Fig 4 skipped: designs_features.csv / wt_features.csv not found.\n")""")

code(r"""# === 13. Fig 5 - consolidated fine-tuning surface acid-base shift =====
if (!is.na(PH_CSV)) {
  ph <- read.csv(PH_CSV, stringsAsFactors=FALSE)
  sf <- intersect(c("surface_acidic_fraction","surface_basic_fraction",
                    "surface_net_charge","surface_ionizable_fraction"), names(ph))
  prim <- c("ProteinMPNN","ProteinMPNN_v020","AlkalineMPNN_020","AcidophileMPNN_020")
  if (length(sf) >= 3 && any(ph$model %in% prim)) {
    cc2 <- ph[complete.cases(ph[, sf]), ]
    prs <- prcomp(cc2[, sf], center=TRUE, scale.=TRUE); ves <- prs$sdev^2/sum(prs$sdev^2)
    if (prs$rotation["surface_acidic_fraction","PC1"] < 0) {
      prs$rotation[,"PC1"] <- -prs$rotation[,"PC1"]; prs$x[,"PC1"] <- -prs$x[,"PC1"] }
    projS <- function(nd) { Xs <- scale(as.matrix(nd[, sf]), center=prs$center, scale=prs$scale)
      Xs[!is.finite(Xs)] <- 0; Z <- Xs %*% prs$rotation[, 1:2]; data.frame(PC1=Z[,1], PC2=Z[,2]) }
    dz <- ph[ph$model %in% prim & complete.cases(ph[, sf]), ]
    dz_xy <- cbind(dz[, c("uniprot_id","model")], projS(dz))
    keep <- intersect(prim, unique(dz_xy$model)); dz_xy$model <- factor(dz_xy$model, keep)
    cenS <- aggregate(cbind(PC1,PC2) ~ model, dz_xy, mean)
    pal5 <- c(ProteinMPNN="#7f8c8d", ProteinMPNN_v020="#7f8c8d",
              AlkalineMPNN_020="#c0392b", AcidophileMPNN_020="#2980b9")
    lbl5 <- c(ProteinMPNN="ProteinMPNN (base)", ProteinMPNN_v020="ProteinMPNN (base)",
              AlkalineMPNN_020="AlkSecMPNN", AcidophileMPNN_020="AcidSecMPNN")
    p5 <- ggplot() +
      geom_hline(yintercept=0,linetype=2,colour="grey80") + geom_vline(xintercept=0,linetype=2,colour="grey80") +
      geom_point(data=dz_xy, aes(PC1,PC2,colour=model), size=.8, alpha=.18) +
      geom_point(data=cenS, aes(PC1,PC2,colour=model), shape=18, size=6) +
      ggrepel::geom_text_repel(data=cenS, aes(PC1,PC2,label=lbl5[as.character(model)],colour=model),
                               size=3.4, fontface="bold", show.legend=FALSE, seed=1) +
      scale_colour_manual(values=pal5[keep], labels=lbl5[keep], name=NULL) + coord_equal() +
      labs(title="Fine-tuning shifts designs along the surface acid-base axis",
           x=sprintf("PC1 (%.0f%%): basic / high-charge -> acidic / low-charge", 100*ves[1]),
           y=sprintf("PC2 (%.0f%%)", 100*ves[2]))
    ggsave(file.path(FIG,"fig5_ft_surface_consolidated.png"), p5, width=7, height=7, dpi=300, bg="white")
    print(p5)
    base_nm <- intersect(c("ProteinMPNN_v020","ProteinMPNN"), keep)[1]
    if (!is.na(base_nm)) {
      b <- cenS[cenS$model==base_nm, ]
      rel <- transform(cenS, dPC1=round(PC1-b$PC1,3), dPC2=round(PC2-b$PC2,3))
      write.csv(rel, file.path(TAB,"ft_surface_centroid_base_relative.csv"), row.names=FALSE)
      print(rel)
    }
  } else cat("Fig 5 skipped: surface features or primary models absent in designs_ph_features.csv.\n")
} else cat("Fig 5 skipped: designs_ph_features.csv not found.\n")""")

code(r"""# === 13b. FT surface-feature dumbbell (base -> AlkSec / AcidSec) ======
if (!is.na(PH_CSV)) {
  ph <- read.csv(PH_CSV, stringsAsFactors=FALSE)
  sf <- intersect(c("surface_acidic_fraction","surface_basic_fraction","surface_net_charge",
                    "surface_ionizable_fraction","charge_per_residue","isoelectric_point"), names(ph))
  base_lv <- intersect(c("ProteinMPNN_v020","ProteinMPNN"), unique(ph$model))[1]
  arms <- c(base_lv, "AlkalineMPNN_020","AcidophileMPNN_020")
  sub  <- ph[ph$model %in% arms & complete.cases(ph[, sf]), ]
  if (length(sf) >= 2 && !is.na(base_lv) && all(c("AlkalineMPNN_020","AcidophileMPNN_020") %in% sub$model)) {
    Z   <- as.data.frame(scale(sub[, sf]))                       # z-score features for comparability
    agg <- aggregate(Z, by = list(model = sub$model), mean)
    base_row <- agg[agg$model == base_lv, sf]; labs0 <- gsub("_", " ", sf)
    long <- do.call(rbind, lapply(c("AlkalineMPNN_020","AcidophileMPNN_020"), function(m)
      data.frame(feature = labs0, arm = m, shift = as.numeric(agg[agg$model == m, sf] - base_row))))
    long$arm <- factor(long$arm, c("AlkalineMPNN_020","AcidophileMPNN_020"), c("AlkSecMPNN","AcidSecMPNN"))
    o <- long$shift[long$arm == "AlkSecMPNN"]; long$feature <- factor(long$feature, labs0[order(o)])
    p_db <- ggplot(long, aes(shift, feature)) +
      geom_vline(xintercept = 0, linetype = 2, colour = "grey70") +
      geom_line(aes(group = feature), colour = "grey75", linewidth = 1) +
      geom_point(aes(colour = arm), size = 3.6) +
      scale_colour_manual(values = c(AlkSecMPNN = "#c0392b", AcidSecMPNN = "#2980b9"), name = NULL) +
      labs(x = "Mean surface-feature shift vs base (SD units)", y = NULL,
           title = "Fine-tuning moves surface chemistry in opposite directions")
    ggsave(file.path(FIG,"fig5_ft_dumbbell.png"), p_db, width=7, height=4.6, dpi=300, bg="white")
    print(p_db)
  } else cat("FT dumbbell skipped: needed surface features/models absent.\n")
}""")

code(r"""# === 14. Manifest ====================================================
cat("=================== OUTPUTS in", OUT, "===================\n")
cat("FIGURES (", FIG, "):\n"); print(list.files(FIG))
cat("\nTABLES (", TAB, "):\n");  print(list.files(TAB))
cat("\nMap to paper:\n",
    " Fig 3A  fig3A_faceted_contours.png (primary) + fig3A_hexbin_guide.png (axes) + fig3_ridgeline.png\n",
    " Fig 3B  fig3_gam_landscapes_{structure,sequence,finetune}.png (contour bands) + SI fig:s-gam\n",
    " SI      fig3_variance_pareto.png ; fig3_centroids_species.png ;\n",
    "         fig_lollipop_function_{PC1,PC2}.png ; fig_lollipop_family_PC1.png\n",
    " Fig 4   fig4_design_centroids.png\n",
    " Fig 5   fig5_ft_surface_consolidated.png + fig5_ft_dumbbell.png  + SI fig:s-ft-surface\n",
    " tables  pca_loadings.csv (tab:s-pca-loadings) ; mixed_features_pca_coordinates.csv (-> variance decomp)\n",
    "         gam_deviance.csv (tab:s-gamdev) ; compactness_*.csv (SI S5) ; ft_surface_centroid_base_relative.csv (tab:ft_surface_pca)\n")""")

nb = {
  "cells": [
    {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True)}
      | ({"outputs": [], "execution_count": None} if t == "code" else {})
    for t, s in CELLS
  ],
  "metadata": {
    "kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
    "language_info": {"name": "R", "codemirror_mode": "r", "file_extension": ".r",
                      "mimetype": "text/x-r-source", "pygments_lexer": "r", "version": "4.3"}
  },
  "nbformat": 4, "nbformat_minor": 5
}
# write the assembled notebook into the canonical notebooks/ tree (not next to the builder)
out = pathlib.Path(__file__).resolve().parents[3] / "notebooks" / "04_pca_gam" / "PCA_paper_figures.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(CELLS), "cells")
