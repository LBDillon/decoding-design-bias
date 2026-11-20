"""Score-variance decomposition (main Table 1; Tables S6, S9 and S24; Figures S1-S2).

Method:
  - 14-feature biophysical set (catalog.BIOPHYS_14) and its PC1/PC2 from `pca.fit`.
  - Per model, standardise the score, then QR-based R^2 / adjusted R^2 of the score
    on each predictor block (LowDim PC1+PC2, Biophys, Family, Species, and unions),
    nested partial-R^2 F-tests, species attenuation, and the Simpson's-paradox
    per-species effect retention check.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import f as fdist

from .. import catalog
from ..data import load_analysis_table
from .pca import fit as fit_pca

warnings.filterwarnings("ignore")

BIOPHYS = catalog.BIOPHYS_14
MODELS = catalog.FULL_COHORT

PREDICTOR_SETS = [
    "LowDim", "Biophys", "Family", "Species",
    "Fam+Spec", "Biophys+Family", "Biophys+Species", "Full",
]


def make_Q(X):
    """Basis for the numerical column space, including an intercept.

    Pivoting identifies redundant predictors. Retaining the extra columns from
    an unpivoted economic QR would fit arbitrary directions outside that space.
    The parameter count excludes the intercept and uses the actual rank.
    """
    Xi = np.column_stack([np.ones(len(X)), X])
    Q, R, _ = qr(Xi, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(R))
    tolerance = np.finfo(Xi.dtype).eps * max(Xi.shape) * diagonal.max()
    rank = int(np.count_nonzero(diagonal > tolerance))
    return Q[:, :rank], rank - 1


def ssr_from_Q(y, Q):
    fitted = Q @ (Q.T @ y)
    return ((y - fitted) ** 2).sum()


def r2_from_Q(y, Q, p):
    sst = ((y - y.mean()) ** 2).sum()
    ssr = ssr_from_Q(y, Q)
    R2 = 1 - ssr / sst
    n = len(y)
    adj = 1 - (1 - R2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    return R2, adj


def nested_test(y, reduced_key, full_key, bases):
    ssr_r = ssr_from_Q(y, bases[reduced_key][0])
    ssr_f = ssr_from_Q(y, bases[full_key][0])
    p_r = bases[reduced_key][1]
    p_f = bases[full_key][1]
    dfdiff = p_f - p_r
    dfden = len(y) - p_f - 1
    Fstat = ((ssr_r - ssr_f) / dfdiff) / (ssr_f / dfden)
    partial_R2 = (ssr_r - ssr_f) / ssr_r
    pval = fdist.sf(Fstat, dfdiff, dfden)
    return partial_R2, Fstat, pval


def group_means(values, codes, n_groups):
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means, counts


def weighted_corr(a, b, w):
    ma = np.average(a, weights=w)
    mb = np.average(b, weights=w)
    cov = np.average((a - ma) * (b - mb), weights=w)
    va = np.average((a - ma) ** 2, weights=w)
    vb = np.average((b - mb) ** 2, weights=w)
    denom = np.sqrt(va * vb)
    return cov / denom if denom > 0 else np.nan


def weighted_sd(a, w):
    m = np.average(a, weights=w)
    return np.sqrt(np.average((a - m) ** 2, weights=w))


def species_effect_collapse(y, sp_codes, n_sp, bases):
    """Reference-free per-species effects: marginal vs family/-biophysics-adjusted."""
    marg, counts = group_means(y, sp_codes, n_sp)
    resid_fam = y - bases["Family"][0] @ (bases["Family"][0].T @ y)
    adj_fam, _ = group_means(resid_fam, sp_codes, n_sp)
    resid_fb = y - bases["Biophys+Family"][0] @ (bases["Biophys+Family"][0].T @ y)
    adj_fb, _ = group_means(resid_fb, sp_codes, n_sp)
    out = {}
    for tag, adj in (("given_family", adj_fam), ("given_family_biophys", adj_fb)):
        out[f"species_effect_corr_{tag}"] = weighted_corr(marg, adj, counts)
        sd_marg = weighted_sd(marg, counts)
        out[f"species_effect_retention_{tag}"] = (
            weighted_sd(adj, counts) / sd_marg if sd_marg > 0 else np.nan
        )
        flip = (marg * adj < 0) & (np.abs(marg) > 1e-12) & (np.abs(adj) > 1e-12)
        out[f"species_effect_signflip_{tag}"] = np.average(flip, weights=counts)
    return out


def decompose_response(y, bases, sp_codes, n_sp):
    """R^2 decomposition record for one standardised response (a model score, or
    AlphaFold's avg_plddt) on prebuilt QR bases. Shared by `compute` and `run_plddt`."""
    y = (y - y.mean()) / y.std()
    R = {k: r2_from_Q(y, Q, p) for k, (Q, p) in bases.items()}
    rec = {}
    for k in PREDICTOR_SETS:
        rec[f"R2_{k}"] = R[k][0]
    for k in PREDICTOR_SETS:
        rec[f"R2adj_{k}"] = R[k][1]
    rec["dBiophys_given_family_species"] = R["Full"][0] - R["Fam+Spec"][0]
    rec["dSpecies_given_family_biophys"] = R["Full"][0] - R["Biophys+Family"][0]
    rec["dFamily_given_species_biophys"] = R["Full"][0] - R["Biophys+Species"][0]
    rec["dFamilySpecies_given_biophys"] = R["Full"][0] - R["Biophys"][0]
    # The SI's names for five of the values above, kept so the output matches Source Data.
    rec["species_only_R2"] = rec["R2_Species"]
    rec["family_only_R2"] = rec["R2_Family"]
    rec["biophysics_only_R2"] = rec["R2_Biophys"]
    rec["residual_species_R2"] = rec["dSpecies_given_family_biophys"]
    rec["residual_biophysics_R2"] = rec["dBiophys_given_family_species"]
    rec["species_attenuation"] = (
        (rec["species_only_R2"] - rec["residual_species_R2"]) / rec["species_only_R2"]
        if rec["species_only_R2"] > 0 else np.nan)
    species_only_adj = R["Species"][1]
    residual_species_adj = R["Full"][1] - R["Biophys+Family"][1]
    rec["species_attenuation_adj"] = (
        (species_only_adj - residual_species_adj) / species_only_adj
        if species_only_adj > 0 else np.nan)
    rec.update(species_effect_collapse(y, sp_codes, n_sp, bases))
    sp_partial, sp_F, sp_p = nested_test(y, "Biophys+Family", "Full", bases)
    bio_partial, bio_F, bio_p = nested_test(y, "Fam+Spec", "Full", bases)
    rec["species_partial_R2_given_family_biophys"] = sp_partial
    rec["species_F_given_family_biophys"] = sp_F
    rec["species_p_given_family_biophys"] = sp_p
    rec["biophys_partial_R2_given_family_species"] = bio_partial
    rec["biophys_F_given_family_species"] = bio_F
    rec["biophys_p_given_family_species"] = bio_p
    return rec


def _build_bases(df):
    """QR design bases for the LowDim/Biophys/Family/Species blocks + species codes."""
    Z = (df[BIOPHYS] - df[BIOPHYS].mean()) / df[BIOPHYS].std()
    fam = pd.get_dummies(df["protein_family"], drop_first=True).values.astype(float)
    spc = pd.get_dummies(df["species"], drop_first=True).values.astype(float)
    sp_codes, sp_levels = pd.factorize(df["species"])
    PCm = df[["PC1", "PC2"]].values
    BIO = Z.values
    bases = {
        "LowDim": make_Q(PCm),
        "Biophys": make_Q(BIO),
        "Family": make_Q(fam),
        "Species": make_Q(spc),
        "Fam+Spec": make_Q(np.hstack([fam, spc])),
        "Biophys+Family": make_Q(np.hstack([BIO, fam])),
        "Biophys+Species": make_Q(np.hstack([BIO, spc])),
        "Full": make_Q(np.hstack([BIO, fam, spc])),
    }
    return bases, sp_codes, len(sp_levels)


OUTPUT_COLS = (
    ["model", "type"]
    + [f"R2_{k}" for k in PREDICTOR_SETS]
    + [f"R2adj_{k}" for k in PREDICTOR_SETS]
    + [
        "dBiophys_given_family_species", "dSpecies_given_family_biophys",
        "dFamily_given_species_biophys", "dFamilySpecies_given_biophys",
        "species_only_R2", "family_only_R2", "biophysics_only_R2",
        "residual_species_R2", "residual_biophysics_R2",
        "species_attenuation", "species_attenuation_adj",
        "biophys_partial_R2_given_family_species", "biophys_F_given_family_species",
        "biophys_p_given_family_species",
        "species_partial_R2_given_family_biophys", "species_F_given_family_biophys",
        "species_p_given_family_biophys",
        "species_effect_corr_given_family", "species_effect_retention_given_family",
        "species_effect_signflip_given_family",
        "species_effect_corr_given_family_biophys",
        "species_effect_retention_given_family_biophys",
        "species_effect_signflip_given_family_biophys",
    ]
)


def pca_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """PC1/PC2 of the 14-feature PCA for every protein with complete features."""
    return fit_pca(df)[0][["Entry", "PC1", "PC2"]]


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose every model's score on a table that already carries PC1/PC2."""
    df = df[df.domain.isin(catalog.DOMAINS)].dropna(subset=BIOPHYS + ["PC1", "PC2"])
    score_cols = [c for c, _ in MODELS.values()]
    df = df.dropna(subset=score_cols).reset_index(drop=True)

    bases, sp_codes, n_sp = _build_bases(df)

    rows = []
    for model, (col, kind) in MODELS.items():
        rec = {"model": model, "type": kind}
        rec.update(decompose_response(df[col].values, bases, sp_codes, n_sp))
        rows.append(rec)

    return pd.DataFrame(rows)[OUTPUT_COLS]


def run(cfg, out_dir: Path | None = None, make_figures: bool = True) -> pd.DataFrame:
    """Load the table, decompose every model's score and write the CSV and figure."""
    out_dir = Path(out_dir) if out_dir else cfg.stage_output("variance_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_analysis_table(cfg.analysis_table, domains_only=False)
    coords = pca_coordinates(df)
    coords.to_csv(out_dir / "pca_coords_14feat.csv", index=False)
    merged = df.merge(coords, on="Entry", how="inner")
    res = compute(merged)
    res.to_csv(out_dir / "score_variance_decomposition.csv", index=False)
    print(f"[variance] {len(res)} models -> {out_dir/'score_variance_decomposition.csv'}")
    if make_figures:
        from .. import plotting
        plotting.variance_decomposition_figure(res, out_dir)
    return res


def run_plddt(cfg, out_dir: Path | None = None) -> pd.DataFrame:
    """Decompose AlphaFold's confidence (avg_plddt) as if it were a model score (Table S9).

    The response is avg_plddt and the cohort is every protein with complete features,
    without requiring model scores."""
    out_dir = Path(out_dir) if out_dir else cfg.stage_output("variance_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_analysis_table(cfg.analysis_table, domains_only=True)
    if "avg_plddt" not in df.columns:
        raise KeyError("avg_plddt column missing from the analysis table")
    coords = pca_coordinates(df)
    df = df.merge(coords, on="Entry", how="inner")
    df = df.dropna(subset=BIOPHYS + ["PC1", "PC2", "avg_plddt"]).reset_index(drop=True)
    bases, sp_codes, n_sp = _build_bases(df)
    rec = {"model": "AlphaFold pLDDT", "type": "AF-confidence"}
    rec.update(decompose_response(df["avg_plddt"].values, bases, sp_codes, n_sp))
    res = pd.DataFrame([rec])[OUTPUT_COLS]
    res.to_csv(out_dir / "plddt_vd.csv", index=False)
    print(f"[variance/plddt] n={len(df)}  Biophys R²={rec['R2_Biophys']:.3f}  "
          f"resid species|fam+bio={rec['dSpecies_given_family_biophys']:.3f}  "
          f"attenuation={rec['species_attenuation']*100:.0f}%  -> {out_dir/'plddt_vd.csv'}")
    return res
