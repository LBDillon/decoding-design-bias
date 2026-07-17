# R requirement

Only Figure 3C and Table S15 require R. The compact implementation is
`src/decoding_bias/analysis/gam_landscapes.R` and depends on `mgcv`:

```r
k <- max(12, min(50, floor(nrow(model_data) / 300)))
gam(score ~ s(PC1, PC2, k = k), data = model_data, method = "REML")
```

`environment.yaml` installs both R and `r-mgcv`. For a system R, run:

```r
install.packages("mgcv")
```

The quick Python reproduction does not require R.
