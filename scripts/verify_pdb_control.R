# Refit every PDB-control regression with R lm.fit and compare ranks and estimates.
# Usage: Rscript scripts/verify_pdb_control.R RESULTS_DIR [ORIGINAL_MAIN_INPUT.csv]
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
features <- c('sequence_length','mw_per_residue','isoelectric_point','acidic_residue_fraction','basic_residue_fraction','gravy','aromaticity','instability_index','proline_fraction','ordered_percent','helix_sheet_contrast','rco','avg_cb_distance','surface_exposure')
pdb <- read.csv(file.path(root,'PDB_analysis_input.csv'),check.names=FALSE)
af <- read.csv(file.path(root,'AF_analysis_input_excluded.csv'),check.names=FALSE)
# The replay of the original sampling rule starts from the input before the two exclusions.
legacy <- if(length(args)>=2) read.csv(args[2],check.names=FALSE) else af
jobs <- list(
 list('pdb_rank_corrected',pdb),list('AF_full_rank_corrected',af),
 list('primary_matched_draw',af),list('common_PDB',pdb),list('common_matched_draw',af),
 list('historical_rule_corrected_draw',legacy))
results <- list(); idx <- 0
weighted_sd <- function(x,w) sqrt(weighted.mean((x-weighted.mean(x,w))^2,w))
for(job in jobs) {
 prefix <- job[[1]]; source <- job[[2]]
 # The replay membership file lists every sampled row with its complete-case flag.
 if(prefix=='historical_rule_corrected_draw') {
   members <- read.csv(file.path(root,'historical_rule_all_draw_members.csv'))
   members <- members[as.logical(members$complete_case),]
   expected <- read.csv(file.path(root,'historical_rule_corrected_draw_estimates.csv'),check.names=FALSE)
 } else {
   members <- read.csv(file.path(root,paste0(prefix,'_members.csv')))
   expected <- read.csv(file.path(root,paste0(prefix,'_estimates.csv')),check.names=FALSE)
 }
 for(j in seq_len(nrow(expected))) {
   rec <- expected[j,,drop=FALSE]
   selected <- members
   if('model' %in% names(selected)) selected <- selected[selected$model==rec$model,]
   if('draw' %in% names(selected)) selected <- selected[selected$draw==rec$draw,]
   d <- source[match(selected$Entry,source$Entry),,drop=FALSE]
   stopifnot(nrow(d)==rec$n,!anyNA(d$Entry))
   z <- scale(d[,features,drop=FALSE])
   # Every actual input has variable features; no hidden constant-column handling.
   stopifnot(all(is.finite(z)))
   fam <- model.matrix(~factor(protein_family),data=d)[,-1,drop=FALSE]
   sp <- model.matrix(~factor(species),data=d)[,-1,drop=FALSE]
   reduced <- cbind(1,z,fam); full <- cbind(1,z,fam,sp); family <- cbind(1,fam)
   y <- d[[rec$score_column]]; y <- (y-mean(y))/sqrt(mean((y-mean(y))^2))
   fit_r <- lm.fit(reduced,y,tol=1e-9)
   fit_f <- lm.fit(full,y,tol=1e-9)
   fit_family <- lm.fit(family,y,tol=1e-9)
   sst <- sum((y-mean(y))^2)
   r2r <- 1-sum(fit_r$residuals^2)/sst
   r2f <- 1-sum(fit_f$residuals^2)/sst
   raw <- r2f-r2r
   adjusted <- (1-(1-r2f)*(nrow(d)-1)/(nrow(d)-fit_f$rank)) - (1-(1-r2r)*(nrow(d)-1)/(nrow(d)-fit_r$rank))
   species_factor <- factor(d$species)
   weights <- as.numeric(table(species_factor))
   marginal <- as.numeric(tapply(y,species_factor,mean))
   adjusted_effect <- as.numeric(tapply(fit_r$residuals,species_factor,mean))
   retention <- weighted_sd(adjusted_effect,weights)/weighted_sd(marginal,weights)
   results[[length(results)+1]] <- data.frame(cohort=prefix,model=rec$model,draw=if('draw'%in%names(rec))rec$draw else -1,
    n=nrow(d),rank_reduced=fit_r$rank,rank_full=fit_f$rank,
    rank_match=fit_r$rank==rec$`rank_Biophys+Family` && fit_f$rank==rec$rank_Full,
    raw_R=raw,adjusted_R=adjusted,retention_R=retention,
    raw_abs_difference=abs(raw-rec$residual_species_raw),
    adjusted_abs_difference=abs(adjusted-rec$residual_species_adj),
    retention_abs_difference=abs(retention-rec$species_effect_retention_given_family_biophys))
   idx <- idx+1
   if(idx%%100==0) cat(idx,'fits verified\n')
 }
}
out <- do.call(rbind,results)
write.csv(out,file.path(root,'independent_R_verification.csv'),row.names=FALSE)
stopifnot(all(out$rank_match),max(out$raw_abs_difference)<1e-9,max(out$adjusted_abs_difference)<1e-9,max(out$retention_abs_difference)<1e-9)
cat('PASS',nrow(out),'fits; maximum absolute difference',max(out[,c('raw_abs_difference','adjusted_abs_difference','retention_abs_difference')]),'\n')
