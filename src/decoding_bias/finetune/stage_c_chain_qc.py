"""
Stage C: PDB chain mapping + QC for the alkaline-optimum cases and matched controls.

Uses UniProt PDB cross-reference metadata (Method, Resolution, Chains=range) to pick
the best chain per protein and compute coverage - no bulk structure download needed
for the gate. Fills: pdb_id, pdb_chain, uniprot_start/end, chain_coverage_fraction,
resolution, method; sets qc_pass + qc_flags; downgrades label_confidence when the
chain covers only part of the protein.

QC pass rule (configurable): best chain is X-ray <=3.0 A (or EM <=4.0 A, or NMR),
coverage >= 0.80, designable length 40-700. Coverage 0.70-0.80 = borderline (kept,
confidence downgraded). <0.70 / mismatch method / fragment = fail.

Run with network (UniProt REST). Writes *_stageC.csv next to each input.
"""
import requests, re, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"
FILES = ["alkaline_optimum_cases_high_confidence","matched_neutral_controls_for_high_confidence_cases",
         "alkaline_optimum_cases_inclusive","matched_neutral_controls_for_inclusive_cases"]
COV_PASS, COV_BORDER = 0.80, 0.70
RES_XRAY, RES_EM = 3.0, 4.0
LEN_MIN, LEN_MAX = 40, 700


def parse_xref(props):
    d = {p["key"]: p["value"] for p in props}
    method = d.get("Method", "")
    res = None
    mr = re.search(r'([\d.]+)\s*A', d.get("Resolution", "") or "")
    if mr: res = float(mr.group(1))
    spans = re.findall(r'=(\d+)-(\d+)', d.get("Chains", "") or "")
    chain = (d.get("Chains", "").split("=")[0].split("/")[0]) if "=" in d.get("Chains","") else ""
    if spans:
        st = min(int(a) for a, _ in spans); en = max(int(b) for _, b in spans)
    else:
        st = en = None
    return method, res, chain, st, en


def best_chain(xrefs, ulen):
    best = None
    for x in xrefs:
        if x["database"] != "PDB": continue
        method, res, chain, st, en = parse_xref(x.get("properties", []))
        if st is None or not ulen: continue
        cov = (en - st + 1) / ulen
        rank_method = {"X-ray":0, "EM":1, "NMR":2}.get(method, 3)
        score = (rank_method, -cov, res if res is not None else 9.9)
        rec = dict(pdb_id=x["id"], chain=chain, method=method, res=res, start=st, end=en, cov=round(cov,3))
        if best is None or score < best[0]: best = (score, rec)
    return best[1] if best else None


def qc(rec):
    if rec is None: return False, "no_mapped_chain"
    flags = []
    if rec["method"] == "X-ray":
        if rec["res"] is None or rec["res"] > RES_XRAY: flags.append(f"res>{RES_XRAY}")
    elif rec["method"] == "EM":
        if rec["res"] is None or rec["res"] > RES_EM: flags.append(f"EM_res>{RES_EM}")
    elif rec["method"] != "NMR":
        flags.append(f"method={rec['method'] or '?'}")
    if rec["cov"] < COV_BORDER: flags.append("coverage<0.70")
    elif rec["cov"] < COV_PASS: flags.append("coverage_borderline")
    span = rec["end"] - rec["start"] + 1
    if not (LEN_MIN <= span <= LEN_MAX): flags.append("length_out_of_range")
    hard = [f for f in flags if f != "coverage_borderline"]
    return (len(hard) == 0), ";".join(flags)


def fetch_xrefs(accs):
    import time
    out = {}; url = "https://rest.uniprot.org/uniprotkb/search"
    for i in range(0, len(accs), 50):
        chunk = accs[i:i+50]
        params = {"query": "accession:(" + " OR ".join(chunk) + ")",
                  "format": "json", "size": 50, "fields": "accession,length,xref_pdb"}
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=60); r.raise_for_status(); break
            except Exception as e:
                if attempt == 3: raise
                time.sleep(3*(attempt+1))
        for e in r.json().get("results", []):
            out[e["primaryAccession"]] = (e.get("uniProtKBCrossReferences", []),
                                          e.get("sequence", {}).get("length"))
        print(f"  fetched {min(i+50,len(accs))}/{len(accs)}", flush=True)
    return out


def main():
    import pandas as pd
    dfs = {f: pd.read_csv(OUT/f"{f}.csv") for f in FILES}
    accs = sorted({a for df in dfs.values() for a in df.acc})
    print(f"fetching PDB xrefs for {len(accs)} unique accessions ...")
    xr = fetch_xrefs(accs)

    chainmap = {}
    for acc,(xrefs, ulen) in xr.items():
        rec = best_chain(xrefs, ulen)
        ok, flags = qc(rec)
        chainmap[acc] = (rec, ok, flags, ulen)

    for f, df in dfs.items():
        rows = []
        for _, r in df.iterrows():
            rec, ok, flags, ulen = chainmap.get(r.acc, (None, False, "not_found", None))
            d = r.to_dict()
            if rec:
                d.update(pdb_id=rec["pdb_id"], pdb_chain=rec["chain"], method=rec["method"],
                         resolution=rec["res"], uniprot_start=rec["start"], uniprot_end=rec["end"],
                         chain_coverage_fraction=rec["cov"])
            d["qc_pass"] = ok; d["qc_flags"] = flags
            # downgrade confidence on borderline coverage
            if "label_confidence" in d and rec and "coverage_borderline" in flags and d["label_confidence"]=="high":
                d["label_confidence"]="medium"
            rows.append(d)
        out = pd.DataFrame(rows); out.to_csv(OUT/f"{f}_stageC.csv", index=False)
        npass = out.qc_pass.sum()
        print(f"  {f}: {npass}/{len(out)} pass QC")
        if "matched_case_uniprot" not in out.columns and len(out):
            pass

    # matched-pair survival (both case and control pass) for each set
    for tag in ["high_confidence","inclusive"]:
        ca = pd.read_csv(OUT/f"alkaline_optimum_cases_{tag}_stageC.csv")
        co = pd.read_csv(OUT/f"matched_neutral_controls_for_{tag}_cases_stageC.csv")
        cok = set(ca[ca.qc_pass].acc); ctrl_ok = set(co[co.qc_pass].acc)
        pairs = ca[(ca.qc_pass) & (ca.matched_control_uniprot.isin(ctrl_ok)) & (ca.matched_control_uniprot!="")]
        print(f"[{tag}] cases pass {len(cok)} | controls pass {len(ctrl_ok)} | BOTH-pass matched pairs {len(pairs)}")
        if tag=="high_confidence":
            import collections
            print("   case qc_flags:", dict(collections.Counter(";".join(ca[~ca.qc_pass].qc_flags.fillna("")).split(";")).most_common(6)))


if __name__ == "__main__":
    main()
