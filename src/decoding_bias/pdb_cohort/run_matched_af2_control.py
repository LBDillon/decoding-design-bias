"""
run_matched_af2_control.py - apples-to-apples control for the PDB-cohort VD/ELO.

The 876-protein experimental-PDB cohort is small and ribosomal-free, so its nested
variance decomposition is degrees-of-freedom-limited (the raw residual-species R²
inflates and the structure-vs-sequence separation muddies). To tell whether that is
the STRUCTURE SOURCE (PDB vs AF2) or just the small N + composition, we run the
IDENTICAL pipeline on AF2 (the main 10k) restricted to:
    * the SAME models present in the cohort,
    * a subsample matched to the cohort's domain marginal (Euk 396 / Bac 364 / Arch 116),
    * ribosomal proteins excluded (the cohort has ~none),
repeated B times. We then compare the PDB cohort point estimate to the AF2-matched
distribution (mean ± sd). If AF2-matched looks like the cohort, the muddiness is N/
composition, not PDB; if the cohort departs from AF2-matched, that is a structure-source
effect.

Outputs (design/outputs/independent_cohort/):
    matched_af2_vd_control.csv          per-model AF2-matched VD (mean±sd) vs PDB cohort
    matched_af2_elo_control.csv         per-model AF2-matched Elo gap (mean±sd) vs PDB cohort
"""
import importlib.util, shutil, sys, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
SVD_DIR = REPO / "paper_code" / "03_variance_decomposition"
ELO_DIR = REPO / "paper_code" / "02_elo"
sys.path.insert(0, str(SVD_DIR))
import score_variance_decomposition as svd  # noqa: E402
_spec = importlib.util.spec_from_file_location("elor", ELO_DIR / "elo_rating.py")
E = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(E)

MAIN = REPO / "dataset_update" / "main_plus_r2_r3_analysis_v12_cli.csv"
COH_VD = REPO / "design" / "outputs" / "independent_cohort" / "cohort_score_variance_decomposition.csv"
COH_ELO = REPO / "design" / "outputs" / "independent_cohort" / "elo_cohort_unweighted" / "cohort_elo_domain_summary.csv"
OUT = REPO / "design" / "outputs" / "independent_cohort"

SEQ = [f for f in svd.SEQ if f not in ("charge_at_ph7", "small_residue_fraction")]
BIOPHYS = SEQ + svd.STRUCT

# canonical name -> (main column, type); ProGen2 = the XL column the cohort uses
PANEL = {
    "ProteinMPNN": ("proteinmpnn_score", "structure"),
    "SolubleMPNN": ("solublempnn_score", "structure"),
    "Caliby": ("caliby_score", "structure"),
    "SolubleCaliby": ("soluble_caliby_score", "structure"),
    "ESM-IF": ("esmif_score", "structure"),
    "ESM3-struct": ("esm3_struct_cond_score", "structure"),
    "MIF": ("mif_score", "hybrid"),
    "MIF-ST": ("mifst_score", "hybrid-ST"),
    "ESM3-seq": ("esm3_seq_only_score", "sequence"),
    "ESM2-15B": ("ESM2_15B_pppl_score", "sequence"),
    "CARP-640M": ("carp_640M_score", "sequence"),
    "ProGen2-XL": ("progen2_XL_score", "sequence"),
    "ProtGPT2": ("protgpt2_score", "sequence"),
    "ESMC-6B": ("esmc_6b_score", "sequence"),
}
TARGET = {"Eukaryota": 396, "Bacteria": 364, "Archaea": 116}   # cohort domain marginal
B_VD = 30
B_ELO = 8


def cohort_models():
    """Models the cohort VD produced AND that exist in the AFDB table (so the matched
    control can be computed). A cohort-only model - e.g. ESMC scored only on the PDB
    chains - is skipped here until it is also scored on the main AFDB dataset."""
    cv = pd.read_csv(COH_VD)
    main_cols = set(pd.read_csv(MAIN, nrows=1).columns)
    return [m for m in cv["model"] if m in PANEL and PANEL[m][0] in main_cols]


def subsample(pool, seed):
    parts = [pool[pool.domain == d].sample(n=min(n, int((pool.domain == d).sum())),
                                           random_state=seed)
             for d, n in TARGET.items()]
    return pd.concat(parts).reset_index(drop=True)


def make_bases(sub, score_cols):
    cc = sub.dropna(subset=BIOPHYS + ["protein_family", "species"] + score_cols).copy()
    Z = ((cc[BIOPHYS] - cc[BIOPHYS].mean()) / cc[BIOPHYS].std()).values
    U, S, _ = np.linalg.svd(Z, full_matrices=False)            # inline 14-feat PCA
    PC = U[:, :2] * S[:2]
    fam = pd.get_dummies(cc["protein_family"], drop_first=True).values.astype(float)
    spc = pd.get_dummies(cc["species"], drop_first=True).values.astype(float)
    sp_codes, sp_levels = pd.factorize(cc["species"])
    bases = {
        "Family": svd.make_Q(fam),
        "Biophys+Family": svd.make_Q(np.hstack([Z, fam])),
        "Full": svd.make_Q(np.hstack([Z, fam, spc])),
    }
    return cc, bases, sp_codes, len(sp_levels)


def resid_metrics(y, bases, sp_codes, n_sp):
    R = {k: svd.r2_from_Q(y, Q, p) for k, (Q, p) in bases.items()}
    d_raw = R["Full"][0] - R["Biophys+Family"][0]
    d_adj = R["Full"][1] - R["Biophys+Family"][1]
    coll = svd.species_effect_collapse(y, sp_codes, n_sp, bases)
    return d_raw, d_adj, coll["species_effect_retention_given_family_biophys"]


def vd_control(models):
    pool = pd.read_csv(MAIN, low_memory=False)
    pool = pool[pool.domain.isin(TARGET)]
    pool = pool[~pool["broad_function"].astype(str).str.contains("ribosom", case=False)]
    cols = [PANEL[m][0] for m in models]
    print(f"AF2 pool (no ribosomal): {len(pool)}  | matched subsamples N={sum(TARGET.values())} x {B_VD}")
    acc = {m: {"raw": [], "adj": [], "ret": []} for m in models}
    for b in range(B_VD):
        sub = subsample(pool, seed=1000 + b)
        cc, bases, sp_codes, n_sp = make_bases(sub, cols)
        for m in models:
            col = PANEL[m][0]
            y = cc[col].values.astype(float); y = (y - y.mean()) / y.std()
            d_raw, d_adj, ret = resid_metrics(y, bases, sp_codes, n_sp)
            acc[m]["raw"].append(d_raw); acc[m]["adj"].append(d_adj); acc[m]["ret"].append(ret)
    cv = pd.read_csv(COH_VD).set_index("model")
    rows = []
    for m in models:
        a = acc[m]
        rows.append(dict(
            model=m, type=PANEL[m][1],
            PDB_resid_raw=cv.loc[m, "dSpecies_given_family_biophys"],
            AF2_resid_raw_mean=np.mean(a["raw"]), AF2_resid_raw_sd=np.std(a["raw"]),
            PDB_resid_adj=cv.loc[m, "residual_species_R2_adj"],
            AF2_resid_adj_mean=np.mean(a["adj"]), AF2_resid_adj_sd=np.std(a["adj"]),
            PDB_retention=cv.loc[m, "species_effect_retention_given_family_biophys"],
            AF2_retention_mean=np.mean(a["ret"]), AF2_retention_sd=np.std(a["ret"]),
        ))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "matched_af2_vd_control.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== VD: PDB cohort vs AF2-matched (N & composition matched) ===")
    print(res.round(3).to_string(index=False))
    print(f"Wrote {OUT/'matched_af2_vd_control.csv'}")
    return res


def elo_control(models):
    pool = pd.read_csv(MAIN, low_memory=False)
    pool = pool[pool.domain.isin(TARGET)]
    pool = pool[~pool["broad_function"].astype(str).str.contains("ribosom", case=False)]
    if "avg_plddt" not in pool.columns:
        pool["avg_plddt"] = np.nan
    cols = [PANEL[m][0] for m in models]
    gaps = {m: [] for m in models}
    tmp = Path(tempfile.mkdtemp())
    for b in range(B_ELO):
        sub = subsample(pool, seed=2000 + b)
        outdir = tmp / f"rep{b}"
        E.run_full_elo_analysis(sub, str(outdir), score_columns=cols, n_permutations=30,
                                protein_column="protein_family", species_column="species",
                                use_plddt_weighting=False)
        long = pd.read_csv(outdir / "results" / "all_models_species_ratings_long.csv")
        g = long.groupby("model").apply(
            lambda x: x[x.domain == "Archaea"].rating.mean() - x[x.domain == "Eukaryota"].rating.mean())
        col2name = {v[0]: k for k, v in PANEL.items()}
        for col, val in g.items():
            name = col2name.get(col, col)
            if name in gaps:
                gaps[name].append(val)
        print(f"  elo rep {b+1}/{B_ELO} done")
    shutil.rmtree(tmp, ignore_errors=True)
    coh = pd.read_csv(COH_ELO)
    col2name = {v[0]: k for k, v in PANEL.items()}
    coh["name"] = coh["model"].map(lambda c: col2name.get(c, c))
    coh_gap = coh.set_index("name")["Arch_minus_Euk"]
    rows = []
    for m in models:
        if not gaps[m]:
            continue
        rows.append(dict(model=m, type=PANEL[m][1],
                         PDB_arch_minus_euk=coh_gap.get(m, np.nan),
                         AF2_arch_minus_euk_mean=np.mean(gaps[m]),
                         AF2_arch_minus_euk_sd=np.std(gaps[m])))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "matched_af2_elo_control.csv", index=False)
    print("\n=== Elo Arch−Euk gap: PDB cohort vs AF2-matched ===")
    print(res.round(0).to_string(index=False))
    print(f"Wrote {OUT/'matched_af2_elo_control.csv'}")
    return res


if __name__ == "__main__":
    models = cohort_models()
    print(f"Matching {len(models)} cohort-present models: {models}\n")
    vd_control(models)
    elo_control(models)
