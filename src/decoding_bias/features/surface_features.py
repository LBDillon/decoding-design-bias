"""
DIAGNOSTIC: does the alkaline signal live on the protein SURFACE (even though bulk
sequence composition barely separates cases from controls)?

For every extracted single-chain structure, compute per-residue solvent accessibility
(Shrake-Rupley, biotite), classify residues as surface-exposed (relative SASA >= 0.25,
Tien 2013 maxASA), then compute SURFACE-residue charge/composition. Compare alkaline
cases vs matched neutral controls (paired), with the BULK equivalents alongside.

Hypothesis (alkaliphile literature): alkaline-adapted surfaces carry MORE acidic
(D/E), FEWER basic residues (esp. Lys), giving a more negative, lower-pI surface.

Output: design/outputs/alkaline_surface_features_high_confidence.csv  + printed report.
"""
import sys, warnings, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio
from build_pdb_scoring_inputs import three_to_one

OUT = HERE / "outputs"
RSA_CUT = 0.25
# Tien et al. 2013 theoretical maxASA (A^2)
MAXASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,
          'I':197,'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,
          'Y':263,'V':174}
ACIDIC, BASIC = set("DE"), set("KRH")


def comp(letters):
    n = len(letters)
    if n == 0: return {}
    c = {a: letters.count(a) for a in set(letters)}
    g = lambda S: sum(c.get(a, 0) for a in S) / n
    return {"n": n, "acidic": g(ACIDIC), "basic": g(BASIC),
            "lys": c.get("K", 0)/n, "arg": c.get("R", 0)/n,
            "asp_glu": g(ACIDIC), "net_KR_DE": (c.get("K",0)+c.get("R",0)-c.get("D",0)-c.get("E",0))/n,
            "KtoKR": (c.get("K",0)/(c.get("K",0)+c.get("R",0))) if (c.get("K",0)+c.get("R",0)) else np.nan}


def one_structure(path):
    arr = pdbio.get_structure(pdbio.PDBFile.read(str(path)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    aa = aa[aa.chain_id == "A"]
    sasa = struc.sasa(aa, vdw_radii="ProtOr")
    res_sasa = struc.apply_residue_wise(aa, sasa, np.nansum)
    starts = struc.get_residue_starts(aa)
    letters = [three_to_one(aa.res_name[s]) for s in starts]
    rsa = []
    for L, s in zip(letters, res_sasa):
        m = MAXASA.get(L)
        rsa.append(s/m if m else np.nan)
    rsa = np.array(rsa)
    surf = [L for L, r in zip(letters, rsa) if (not np.isnan(r) and r >= RSA_CUT)]
    bulk = [L for L in letters if L in MAXASA]
    out = {}
    for tag, comps in [("surf", comp(surf)), ("bulk", comp(bulk))]:
        for k, v in comps.items():
            out[f"{tag}_{k}"] = v
    return out


def worker(rec):
    try:
        return rec["acc"], rec["role"], rec["pair_case"], one_structure(rec["path"])
    except Exception as ex:
        return rec["acc"], rec["role"], rec["pair_case"], {"err": str(ex)[:40]}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "high_confidence"
    m = pd.read_csv(OUT / "alkaline_structures_manifest.csv")
    pth = m.chain_pdb_path.astype(str)
    ok = m[m.fail_reason.isna() & (pth.str.len() > 0) & (pth != "nan") & (m.set == tag)].copy()
    # restrict to chains that made it into the trainable jsonl (40-700, paired)
    import json
    keep = set()
    pair_of = {}
    for sp in ["train", "val", "test"]:
        f = OUT / f"alkaline_parsed_{tag}_{sp}.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line); keep.add((r["name"], r["role"])); pair_of[r["name"]] = r["pair_case"]
    recs = [{"acc": r.acc, "role": r.role, "path": r.chain_pdb_path,
             "pair_case": pair_of.get(r.acc, "")}
            for _, r in ok.iterrows() if (r.acc, r.role) in keep]
    print(f"computing SASA surface composition for {len(recs)} chains ...")

    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(worker, r) for r in recs]
        for i, fut in enumerate(as_completed(futs)):
            acc, role, pc, d = fut.result()
            if "err" not in d:
                rows.append({"acc": acc, "role": role, "pair_case": pc, **d})
            if i % 200 == 0: print(f"  {i}/{len(recs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"alkaline_surface_features_{tag}.csv", index=False)

    ca, co = df[df.role == "case"], df[df.role == "control"]
    feats = [("surf_acidic", "SURFACE acidic (D+E) frac"), ("surf_basic", "SURFACE basic (K+R+H) frac"),
             ("surf_lys", "SURFACE Lys frac"), ("surf_arg", "SURFACE Arg frac"),
             ("surf_net_KR_DE", "SURFACE net (K+R-D-E)/n"), ("surf_KtoKR", "SURFACE K/(K+R)"),
             ("bulk_acidic", "bulk acidic frac"), ("bulk_basic", "bulk basic frac"),
             ("bulk_lys", "bulk Lys frac"), ("bulk_net_KR_DE", "bulk net (K+R-D-E)/n")]

    # paired deltas (case - matched control) + simple separation (Mann-Whitney AUC)
    from scipy.stats import mannwhitneyu, wilcoxon
    merged = ca.merge(co, left_on="pair_case", right_on="pair_case", suffixes=("_case", "_ctrl"))
    print(f"\n  paired comparison on {len(merged)} pairs ({len(ca)} cases, {len(co)} controls)\n")
    print(f"  {'feature':<28}{'case':>9}{'ctrl':>9}{'Δ(mean)':>9}{'Δ_pair':>9}{'AUC':>7}{'p':>9}")
    for key, lbl in feats:
        c1, c0 = ca[key].dropna(), co[key].dropna()
        u, p = mannwhitneyu(c1, c0, alternative="two-sided")
        auc = u / (len(c1) * len(c0))
        dpair = (merged[f"{key}_case"] - merged[f"{key}_ctrl"]).mean()
        print(f"  {lbl:<28}{c1.mean():9.4f}{c0.mean():9.4f}{c1.mean()-c0.mean():9.4f}"
              f"{dpair:9.4f}{auc:7.3f}{p:9.1e}")
    print("\n  AUC 0.5 = no separation; >0.6 (or <0.4) = a usable axis. "
          "Δ_pair = mean(case-control) within matched pairs.")


if __name__ == "__main__":
    main()
