# R environment (PCA / GAM preference landscapes)

The GAM preference-landscape figures (main **Fig 3**, SI **S6 "PCA and GAM
preference landscapes"**) are fit in **R with `mgcv`**, not in Python. The canonical
GAM fit lives in [`04_pca_gam/pca_corrections.R`](04_pca_gam/pca_corrections.R):

```r
library(mgcv)
m <- gam(y ~ s(PC1, PC2, k = 12), data = d, method = "REML")
```

`mgcv`'s penalised-spline `gam()` has **no faithful Python equivalent** (`pygam` /
`statsmodels` GAMs differ in penalty construction and REML), so the R toolchain is a
hard requirement for these panels and cannot be captured in `environment.yaml`
alone - the conda packages are listed there for convenience, but the notebook was
developed against a system R.

## Required R packages
| package | used for |
|---|---|
| `mgcv` | penalised-spline GAM fit (`gam`, `s()`, REML) - the load-bearing dependency |
| `MASS` | supporting stats in `pca_corrections.R` |
| `tidyverse` (`dplyr`, `ggplot2`, `tidyr`, `readr`) | data wrangling + plotting in the design pH-axis R scripts |
| `ggrepel`, `patchwork`, `viridis` | figure composition |

## Install
Either via the conda `r-*` packages in `environment.yaml`, or in a system R:

```r
install.packages(c("mgcv", "MASS", "tidyverse", "ggrepel", "patchwork", "viridis"))
```

Developed against **R ≥ 4.2**. `mgcv` ships with base R; the version used was the one
bundled with R 4.2.
