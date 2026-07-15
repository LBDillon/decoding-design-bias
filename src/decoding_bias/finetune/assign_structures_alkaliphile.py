"""
Assign structure_source (PDB-first, AF fallback) + per-source QC to the alkaliphile
matched set, reusing build_alkaline_dataset.assign_structure (same QC as the pH-optimum
set). Fetches each protein's PDB cross-references (with Method/Resolution/Chains) + length
from UniProt JSON, runs assignment, and merges the structure columns into the Stage D
case + control tables in place.

Needs network. Run with dangerouslyDisableSandbox.
"""
import sys, time, argparse
from pathlib import Path
import pandas as pd, requests
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; sys.path.insert(0, str(HERE))
from build_alkaline_dataset import assign_structure, STRUCT_COLS
from _cohort import cfg


def fetch_pdb_len(accs):
    url = "https://rest.uniprot.org/uniprotkb/search"; out = {}
    for i in range(0, len(accs), 80):
        chunk = accs[i:i+80]
        params = {"query": "accession:(" + " OR ".join(chunk) + ")", "format": "json",
                  "size": 100, "fields": "accession,length,xref_pdb"}
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=90); r.raise_for_status(); break
            except Exception:
                if attempt == 3: raise
                time.sleep(3*(attempt+1))
        for e in r.json().get("results", []):
            pdb = [(x["id"], {p["key"]: p["value"] for p in x.get("properties", [])})
                   for x in e.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"]
            out[e["primaryAccession"]] = dict(length=e.get("sequence", {}).get("length"), pdb=pdb)
        print(f"  fetched {min(i+80, len(accs))}/{len(accs)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"])
    a = ap.parse_args(); C = cfg(a.cohort)
    CASES = OUT / C["cases_D"]; CTRLS = OUT / C["ctrls_D"]
    ca = pd.read_csv(CASES); co = pd.read_csv(CTRLS)
    accs = sorted(set(ca.acc) | set(co.acc))
    print(f"fetching PDB xrefs + length for {len(accs)} accessions ...")
    info = fetch_pdb_len(accs)
    rows = [dict(acc=a, length=info.get(a, {}).get("length"), pdb=info.get(a, {}).get("pdb", []))
            for a in accs]
    print("assigning structure_source + per-source QC (PDB then AF) ...", flush=True)
    assign_structure(rows)
    bymap = {r["acc"]: r for r in rows}

    def merge(df):
        for col in STRUCT_COLS:
            df[col] = df.acc.map(lambda a: bymap.get(a, {}).get(col, ""))
        return df
    merge(ca).to_csv(CASES, index=False); merge(co).to_csv(CTRLS, index=False)

    import collections
    src = collections.Counter(r.get("structure_source", "none") for r in rows)
    print(f"\nstructure_source over {len(rows)}: {dict(src)}")
    for name, df in [("cases", ca), ("controls", co)]:
        sp = df[df.split.notna()]
        npass = int(sp.qc_pass.sum()) if "qc_pass" in sp else 0
        print(f"  {name}: {npass}/{len(sp)} QC-pass (split-assigned)")
    # both-pass matched pairs
    cok = set(ca[ca.qc_pass == True].acc); ook = set(co[co.qc_pass == True].acc)
    pairs = ca[(ca.qc_pass == True) & ca.matched_control_uniprot.isin(ook) & ca.split.notna()]
    print(f"  both-pass matched pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
