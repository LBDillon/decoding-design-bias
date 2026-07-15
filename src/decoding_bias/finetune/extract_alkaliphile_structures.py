"""
Structure extraction for the environmental-alkaliphile matched set. Reuses the core
extractors from extract_alkaline_structures (PDB mmCIF chain extraction with gapped-identity
verification; AF model download), writes single-chain backbones, a manifest, and ProteinMPNN
jsonl per split (pair-level survival, 40-700 enforced). Carries group/clade/label/split.

Run with dangerouslyDisableSandbox.
"""
import sys, json, warnings, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import extract_alkaline_structures as E
from extract_alkaline_structures import extract_pdb, extract_af, backbone_record, LEN_MIN, LEN_MAX
from _cohort import cfg

OUT = HERE / "outputs"
STRUCT = MANIFEST = PARTIAL = None   # set per-cohort in main()


def process(rec):
    acc, role = rec["acc"], rec["role"]
    dest = STRUCT / ("cases" if role == "case" else "controls") / f"{acc}.pdb"
    out = dict(rec); out["chain_pdb_path"] = ""; out["extracted_seq"] = ""
    out["extracted_identity"] = np.nan; out["fail_reason"] = None
    try:
        if dest.exists() and dest.stat().st_size > 0:
            out["chain_pdb_path"] = str(dest); return out
        if rec["structure_source"] == "PDB":
            seq, idn, fail = extract_pdb(acc, rec["pdb_id"], rec["pdb_chain"], rec["sequence"], dest)
        elif rec["structure_source"] == "AF":
            seq, idn, fail = extract_af(acc, rec["sequence"], dest)
        else:
            seq, idn, fail = None, np.nan, "no_structure_source"
        out["extracted_identity"] = idn; out["fail_reason"] = fail
        if not fail:
            out["chain_pdb_path"] = str(dest); out["extracted_seq"] = seq
    except Exception as ex:
        out["fail_reason"] = f"error:{str(ex)[:50]}"
    return out


def collect(C):
    ca = pd.read_csv(OUT / C["cases_D"])
    co = pd.read_csv(OUT / C["ctrls_D"])
    rows = []
    for df, role in [(ca, "case"), (co, "control")]:
        d = df[(df.qc_pass == True) & df.split.notna()].copy()
        for _, r in d.iterrows():
            rows.append({"acc": r.acc, "role": role, "split": r.split,
                         "structure_source": r.structure_source,
                         "pdb_id": r.get("pdb_id", ""), "pdb_chain": r.get("pdb_chain", ""),
                         "cluster_id": r.get("cluster_id", ""), "group": r.group,
                         "clade": r.clade, "sequence": r.sequence,
                         "tight_pair": r.get("tight_pair", r.get("bacillaceae_pair", False)),
                         "label": 1 if role == "case" else 0})
    return rows, ca, co


def main():
    global STRUCT, MANIFEST, PARTIAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"])
    a = ap.parse_args(); C = cfg(a.cohort)
    STRUCT = OUT / C["struct_dir"]
    (STRUCT / "cases").mkdir(parents=True, exist_ok=True)
    (STRUCT / "controls").mkdir(parents=True, exist_ok=True)
    MANIFEST = OUT / C["manifest"]
    PARTIAL = OUT / (C["manifest"].replace(".csv", ".partial.csv"))
    rows, ca, co = collect(C)
    seen = {}
    for r in rows: seen[(r["acc"], r["role"])] = r
    rows = list(seen.values())
    print(f"structures to extract: {len(rows)}  "
          f"(PDB {sum(r['structure_source']=='PDB' for r in rows)}, "
          f"AF {sum(r['structure_source']=='AF' for r in rows)})")

    done = {}
    if PARTIAL.exists():
        for r in pd.read_csv(PARTIAL).to_dict("records"): done[(r["acc"], r["role"])] = r
    todo = [r for r in rows if (r["acc"], r["role"]) not in done]
    results = list(done.values())
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(process, r) for r in todo]
        for i, f in enumerate(tqdm(as_completed(futs), total=len(futs), desc="extract")):
            results.append(f.result())
            if i % 40 == 0: pd.DataFrame(results).to_csv(PARTIAL, index=False)
    man = pd.DataFrame(results); man.to_csv(MANIFEST, index=False)
    if PARTIAL.exists(): PARTIAL.unlink()

    pth = man.chain_pdb_path.astype(str)
    ok = man[man.fail_reason.isna() & (pth.str.len() > 0) & (pth != "nan")].copy()
    print(f"\nextracted OK: {len(ok)}/{len(man)} | by source {dict(ok.structure_source.value_counts())} "
          f"| by role {dict(ok.role.value_counts())}")
    if man.fail_reason.notna().any(): print("failures:", dict(man.fail_reason.value_counts()))

    # ProteinMPNN jsonl per split (pair-level survival, 40-700)
    L = {**dict(zip(ca.acc, ca.length)), **dict(zip(co.acc, co.length))}
    okset = ok.set_index(["acc", "role"])
    pairs = ca[(ca.qc_pass == True) & ca.split.notna()]
    n = {"train": 0, "val": 0, "test": 0}; skip_struct = skip_len = 0
    writers = {sp: open(OUT / f"{C['jsonl_prefix']}_parsed_{sp}.jsonl", "w") for sp in n}
    for _, p in pairs.iterrows():
        kc, kk = (p.acc, "case"), (p.matched_control_uniprot, "control")
        if kc not in okset.index or kk not in okset.index: skip_struct += 1; continue
        lc, lk = L.get(p.acc), L.get(p.matched_control_uniprot)
        if not (lc and lk and LEN_MIN <= lc <= LEN_MAX and LEN_MIN <= lk <= LEN_MAX):
            skip_len += 1; continue
        sp = p.split
        for acc, role, key in [(p.acc, "case", kc), (p.matched_control_uniprot, "control", kk)]:
            row = okset.loc[key]
            extra = {"role": role, "label": 1 if role == "case" else 0, "split": sp,
                     "cluster_id": p.cluster_id, "pair_case": p.acc,
                     "group": row.group, "clade": row.clade,
                     "tight_pair": bool(p.get("tight_pair", False))}
            try:
                writers[sp].write(json.dumps(backbone_record(row.chain_pdb_path, acc, extra)) + "\n")
                n[sp] += 1
            except Exception as ex:
                print(f"   jsonl skip {acc}: {str(ex)[:40]}")
    for w in writers.values(): w.close()
    print(f"jsonl chains: {n} ({n['train']//2}+{n['val']//2}+{n['test']//2} pairs); "
          f"skipped: no_struct {skip_struct}, length {skip_len}")


if __name__ == "__main__":
    main()
