"""
Structure extraction for the alkaline-optimum fine-tuning set.

For every QC-passing, split-assigned case and matched control:
  - structure_source == PDB : fetch RCSB mmCIF (cached), extract the UniProt-mapped
        chain (auto-correct chain id by best alignment if needed), relabel to 'A',
        write a single-chain PDB.
  - structure_source == AF  : download the AlphaFold model PDB from AFDB (already a
        single chain 'A' covering the full UniProt sequence).
Each extracted chain's observed sequence is aligned to the stored UniProt sequence;
chains below IDENTITY_MIN are flagged (kept out of the training manifest).

Outputs (design/outputs/):
  structures/{cases,controls}/<acc>.pdb            single-chain backbone-bearing PDBs
  alkaline_structures_manifest.csv                 acc, role, set, split, source, path, QC
  alkaline_parsed_<set>.jsonl                      ProteinMPNN-format backbone records
                                                   (name, seq, coords_chain_A{N,CA,C,O},
                                                    plus role/label/split/cluster_id)

Resumable (partial manifest) and threaded. Needs network; run with dangerouslyDisableSandbox.
"""
import sys, json, warnings, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc
from build_pdb_scoring_inputs import three_to_one
from Bio import Align

# Gapped global alignment: robust to unmodeled loops (a gapless window under-counts
# identity for cryo-EM/partial chains). identity = identical aligned residues / chain length.
_aligner = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=-1,
                                 open_gap_score=-5, extend_gap_score=-0.5)


def gapped_identity(chain, uni):
    if not chain or not uni: return 0.0
    aln = _aligner.align(chain, uni)[0]
    ident = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        ident += sum(1 for x, y in zip(chain[s1:e1], uni[s2:e2]) if x == y)
    return ident / len(chain)

OUT = HERE / "outputs"
CIF_CACHE = OUT / "_cif_cache"; CIF_CACHE.mkdir(parents=True, exist_ok=True)
STRUCT = OUT / "structures"
(STRUCT / "cases").mkdir(parents=True, exist_ok=True)
(STRUCT / "controls").mkdir(parents=True, exist_ok=True)
AF_CACHE = OUT / "_af_cache"; AF_CACHE.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT / "alkaline_structures_manifest.csv"
PARTIAL = OUT / "alkaline_structures_manifest.partial.csv"
IDENTITY_MIN = 0.90
LEN_MIN, LEN_MAX = 40, 700    # designable-length gate, applied consistently to PDB and AF
MAX_CIF_MB = 60               # keep QC-passed EM structures; only skip pathological assemblies
AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"
BB = ["N", "CA", "C", "O"]


def _write_chain(sel, dest):
    sel = sel.copy(); sel.chain_id[:] = "A"
    pf = pdbio.PDBFile(); pf.set_structure(sel); pf.write(str(dest))


def extract_pdb(acc, pdb_id, chain, uni, dest):
    pdb_id = str(pdb_id).strip().upper(); chain = str(chain).strip()
    cif = CIF_CACHE / f"{pdb_id}.cif"
    if not cif.exists():
        rcsb.fetch(pdb_id, "cif", str(CIF_CACHE))
    if cif.stat().st_size > MAX_CIF_MB * 1e6:
        return None, None, "large_assembly_skipped"
    arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    sel = aa[aa.chain_id == chain]
    seq = "".join(three_to_one(r) for r in sel[sel.atom_name == "CA"].res_name) if sel.array_length() else ""
    idn = gapped_identity(seq, uni)
    if idn < IDENTITY_MIN:                              # wrong/absent chain -> auto-select best
        best = (None, 0.0, "")
        for ch in sorted(set(aa.chain_id)):
            ca = aa[(aa.chain_id == ch) & (aa.atom_name == "CA")]
            if ca.array_length() == 0: continue
            s = "".join(three_to_one(r) for r in ca.res_name)
            i = gapped_identity(s, uni)
            if i > best[1]: best = (ch, i, s)
        chain, idn, seq = best
        if chain is None: return None, None, "no_protein_chain"
        sel = aa[aa.chain_id == chain]
    if idn < IDENTITY_MIN:
        return None, round(idn, 4), "identity_low"
    _write_chain(sel, dest)
    return seq, round(idn, 4), None


def af_pdb_url(acc):
    api = json.load(urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=60))
    return api[0]["pdbUrl"]


def extract_af(acc, uni, dest):
    cached = AF_CACHE / f"AF-{acc}.pdb"
    if not cached.exists():
        urllib.request.urlretrieve(af_pdb_url(acc), str(cached))
    arr = pdbio.get_structure(pdbio.PDBFile.read(str(cached)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    ca = aa[aa.atom_name == "CA"]
    seq = "".join(three_to_one(r) for r in ca.res_name)
    idn = gapped_identity(seq, uni)
    if idn < IDENTITY_MIN:
        return None, round(idn, 4), "identity_low"
    _write_chain(aa, dest)
    return seq, round(idn, 4), None


def process(rec):
    acc, role = rec["acc"], rec["role"]
    dest = STRUCT / ("cases" if role == "case" else "controls") / f"{acc}.pdb"
    out = dict(rec); out["chain_pdb_path"] = ""; out["extracted_seq"] = ""
    out["extracted_identity"] = np.nan; out["fail_reason"] = None
    try:
        if dest.exists() and dest.stat().st_size > 0:
            out["chain_pdb_path"] = str(dest); out["fail_reason"] = None
            return out  # already extracted
        if rec["structure_source"] == "PDB":
            seq, idn, fail = extract_pdb(acc, rec["pdb_id"], rec["pdb_chain"], rec["sequence"], dest)
        else:
            seq, idn, fail = extract_af(acc, rec["sequence"], dest)
        out["extracted_identity"] = idn; out["fail_reason"] = fail
        if not fail:
            out["chain_pdb_path"] = str(dest); out["extracted_seq"] = seq
    except Exception as ex:
        out["fail_reason"] = f"error:{str(ex)[:50]}"
    return out


def backbone_record(path, name, extra):
    arr = pdbio.get_structure(pdbio.PDBFile.read(str(path)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]
    aa = aa[aa.chain_id == "A"]
    res_ids = []
    for rid in aa.res_id:
        if not res_ids or res_ids[-1] != rid: res_ids.append(rid)
    coords = {b: [] for b in BB}; seq = []
    for rid in res_ids:
        r = aa[aa.res_id == rid]
        seq.append(three_to_one(r.res_name[0]))
        for b in BB:
            at = r[r.atom_name == b]
            coords[b].append([float(x) for x in at.coord[0]] if at.array_length() else [float("nan")]*3)
    s = "".join(seq)
    rec = {"name": name, "num_of_chains": 1, "seq": s, "seq_chain_A": s,
           "coords_chain_A": {f"{b}_chain_A": coords[b] for b in BB}}
    rec.update(extra)
    return rec


def collect(tag):
    ca = pd.read_csv(OUT / f"alkaline_optimum_cases_{tag}_stageD.csv")
    co = pd.read_csv(OUT / f"matched_neutral_controls_for_{tag}_cases_stageD.csv")
    rows = []
    for df, role in [(ca, "case"), (co, "control")]:
        d = df[df.qc_pass & df.split.notna()].copy()
        for _, r in d.iterrows():
            rows.append({"acc": r.acc, "role": role, "set": tag, "split": r.split,
                         "structure_source": r.structure_source,
                         "pdb_id": r.get("pdb_id", ""), "pdb_chain": r.get("pdb_chain", ""),
                         "cluster_id": r.get("cluster_id", ""), "sequence": r.sequence,
                         "label": 1 if role == "case" else 0})
    return rows


def main():
    tags = sys.argv[1:] or ["high_confidence"]
    rows = [r for t in tags for r in collect(t)]
    # dedupe by (acc, role) - an acc can be a case in one set and control-pool in another
    seen = {};
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
    man = pd.DataFrame(results)
    man.to_csv(MANIFEST, index=False)
    if PARTIAL.exists(): PARTIAL.unlink()

    pth = man.chain_pdb_path.astype(str)
    ok = man[man.fail_reason.isna() & (pth.str.len() > 0) & (pth != "nan")].copy()
    print(f"\nextracted OK: {len(ok)}/{len(man)}")
    print("by source:", dict(ok.structure_source.value_counts()))
    print("by role  :", dict(ok.role.value_counts()))
    if man.fail_reason.notna().any():
        print("failures :", dict(man.fail_reason.value_counts()))

    # build ProteinMPNN-format jsonl per set/split. Pair-level survival: a pair is
    # written only if BOTH case and control extracted OK (keeps splits balanced).
    for tag in tags:
        ca = pd.read_csv(OUT / f"alkaline_optimum_cases_{tag}_stageD.csv")
        co = pd.read_csv(OUT / f"matched_neutral_controls_for_{tag}_cases_stageD.csv")
        L = {**dict(zip(ca.acc, ca.length)), **dict(zip(co.acc, co.length))}
        okset = ok[ok.set == tag].set_index(["acc", "role"])
        pairs = ca[ca.qc_pass & ca.split.notna()]
        n = {"train": 0, "val": 0, "test": 0}; skipped = 0; skip_len = 0
        writers = {sp: open(OUT / f"alkaline_parsed_{tag}_{sp}.jsonl", "w") for sp in n}
        for _, p in pairs.iterrows():
            kc, kk = (p.acc, "case"), (p.matched_control_uniprot, "control")
            if kc not in okset.index or kk not in okset.index:
                skipped += 1; continue            # pair-level skip, not member-level
            lc, lk = L.get(p.acc), L.get(p.matched_control_uniprot)   # enforce 40-700 on both
            if not (lc and lk and LEN_MIN <= lc <= LEN_MAX and LEN_MIN <= lk <= LEN_MAX):
                skip_len += 1; continue
            sp = p.split
            for acc, role, key in [(p.acc, "case", kc), (p.matched_control_uniprot, "control", kk)]:
                extra = {"role": role, "label": 1 if role == "case" else 0,
                         "split": sp, "set": tag, "cluster_id": p.cluster_id, "pair_case": p.acc}
                try:
                    rec = backbone_record(okset.loc[key].chain_pdb_path, acc, extra)
                    writers[sp].write(json.dumps(rec) + "\n"); n[sp] += 1
                except Exception as ex:
                    print(f"   jsonl skip {acc}: {str(ex)[:40]}")
        for w in writers.values(): w.close()
        print(f"[{tag}] jsonl chains written: {n} ({n['train']//2}+{n['val']//2}+{n['test']//2} pairs); "
              f"skipped: missing structure {skipped}, length 40-700 {skip_len}")


if __name__ == "__main__":
    main()
