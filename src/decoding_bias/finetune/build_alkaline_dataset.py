"""
Build the alkaline-optimum enzyme set (cases) + matched neutral controls, with
AlphaFold inclusion. Label = MEASURED pH optimum (functional); NOT charge/pI.

Structure per protein (structure_source):
  PDB  : best experimental chain passes QC  (X-ray<=3.0/EM<=4.0/NMR, coverage>=0.80, len 40-700)
  AF   : else AlphaFold model passes the original paper's pLDDT gate
         (mean pLDDT>=70, fraction pLDDT>70 >=0.70, fraction pLDDT<50 <=0.30)
  (fail: neither)  -> dropped
PDB QC from UniProt xref metadata; AF QC from the AFDB API (globalMetricValue +
fractionPlddt*). charge/pI/acidic-basic are downstream readouts only.

Case sets: high_confidence (opt_lo>=8.5 & width<=1.0), inclusive (opt_hi>=8.5).
Controls: range fully within 6.0-7.5; 1:1 hierarchical match (Pfam->EC3->EC2->EC class,
prefer same structure_source) + length +/-25% + same domain.
Writes *_stageC.csv (QC + structure_source filled) for Stage D clustering/splits.
Run with network (UniProt REST + AFDB API).
"""
import requests, re, csv, collections, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"
B = "https://rest.uniprot.org/uniprotkb/search"
AFDB = "https://alphafold.ebi.ac.uk/api/prediction/"
DOMAINS = {"Bacteria","Archaea","Eukaryota"}
COV_PASS, COV_BORDER, RES_XRAY, RES_EM, LEN_MIN, LEN_MAX = 0.80, 0.70, 3.0, 4.0, 40, 700
AF_PLDDT, AF_CONF, AF_VLOW = 70.0, 0.70, 0.30


def ph_opt(text):
    m=re.search(r'optimum ph(?:\s*is|:)?\s*(?:around|about|approximately|~|>|>=|of)?\s*'
                r'([\d]+(?:\.\d+)?)(?:\s*(?:-|-|to|and)\s*([\d]+(?:\.\d+)?))?', text.lower())
    if not m: return None
    a=float(m.group(1)); b=float(m.group(2)) if m.group(2) else a; return (min(a,b),max(a,b))


def harvest(query):
    rows=[]; url=B; params={"query":query,"format":"json","size":500}
    while url:
        r=requests.get(url,params=params,timeout=120); params=None
        if r.status_code!=200: raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        for e in r.json().get("results",[]):
            ph=None; ev=[]; scl=[]; subunit=""
            for c in e.get("comments",[]):
                if c.get("phDependence"):
                    txts=c["phDependence"].get("texts",[]); ph=" ".join(t["value"] for t in txts)
                    ev=[f"PubMed:{x['id']}" for t in txts for x in t.get("evidences",[]) if x.get("source")=="PubMed"]
                if c.get("commentType")=="SUBCELLULAR LOCATION":
                    scl+=[l.get("location",{}).get("value","") for l in c.get("subcellularLocations",[])]
                if c.get("commentType")=="SUBUNIT": subunit=" ".join(t["value"] for t in c.get("texts",[]))[:60]
            opt=ph_opt(ph) if ph else None
            pd_=e.get("proteinDescription",{}).get("recommendedName",{})
            ec=";".join(x["value"] for x in pd_.get("ecNumbers",[])) if pd_.get("ecNumbers") else ""
            kws=[k.get("name") for k in e.get("keywords",[])]
            pfam=";".join(x["id"] for x in e.get("uniProtKBCrossReferences",[]) if x["database"]=="Pfam")
            pdb=[(x["id"], {p["key"]:p["value"] for p in x.get("properties",[])})
                 for x in e.get("uniProtKBCrossReferences",[]) if x["database"]=="PDB"]
            lineage=e.get("organism",{}).get("lineage",[])
            dom=next((d for d in DOMAINS if d in lineage), lineage[0] if lineage else "?")
            rows.append(dict(acc=e["primaryAccession"], length=e.get("sequence",{}).get("length"),
                sequence=e.get("sequence",{}).get("value",""),
                opt_lo=opt[0] if opt else None, opt_hi=opt[1] if opt else None,
                ph_text=(ph or "").replace("\n"," ")[:220], evidence_ids=";".join(ev),
                ec_full=ec, secreted=("Secreted" in kws or any("Secreted" in s for s in scl)),
                localization=";".join(sorted(set(scl)))[:80], oligomeric_note=subunit, pfam=pfam,
                domain=dom, organism=e.get("organism",{}).get("scientificName",""), pdb=pdb))
        m=re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("link","")); url=m.group(1) if m else None
    return rows


def ec_parts(ec):
    e=(ec or "").split(";")[0].strip(); p=e.split(".") if e and e[0].isdigit() else []
    return (".".join(p[:3]) if len(p)>=3 and "-" not in p[2] else "",
            ".".join(p[:2]) if len(p)>=2 and "-" not in p[1] else "", p[0] if p else "none")


def best_pdb(pdb, ulen):
    best=None
    for pid, props in pdb:
        method=props.get("Method",""); res=None
        mr=re.search(r'([\d.]+)\s*A', props.get("Resolution","") or "")
        if mr: res=float(mr.group(1))
        spans=re.findall(r'=(\d+)-(\d+)', props.get("Chains","") or "")
        if not spans or not ulen: continue
        st=min(int(a) for a,_ in spans); en=max(int(b) for _,b in spans)
        chain=props.get("Chains","").split("=")[0].split("/")[0]
        cov=(en-st+1)/ulen
        score=({"X-ray":0,"EM":1,"NMR":2}.get(method,3), -cov, res if res is not None else 9.9)
        rec=dict(pdb_id=pid, chain=chain, method=method, res=res, start=st, end=en, cov=round(cov,3))
        if best is None or score<best[0]: best=(score, rec)
    return best[1] if best else None


def pdb_qc(rec):
    if rec is None: return False, "no_chain", False
    f=[]
    if rec["method"]=="X-ray" and (rec["res"] is None or rec["res"]>RES_XRAY): f.append("res>3.0")
    elif rec["method"]=="EM" and (rec["res"] is None or rec["res"]>RES_EM): f.append("EM_res>4.0")
    elif rec["method"] not in ("X-ray","EM","NMR"): f.append(f"method={rec['method'] or '?'}")
    if rec["cov"]<COV_BORDER: f.append("coverage<0.70")
    elif rec["cov"]<COV_PASS: f.append("coverage_borderline")
    if not (LEN_MIN<=rec["end"]-rec["start"]+1<=LEN_MAX): f.append("length_out_of_range")
    hard=[x for x in f if x!="coverage_borderline"]
    return len(hard)==0, ";".join(f), ("coverage_borderline" in f)


def af_qc(acc):
    try:
        r=requests.get(AFDB+acc, timeout=30)
        if r.status_code!=200: return None
        d=r.json()[0]
        mean=d.get("globalMetricValue"); conf=(d.get("fractionPlddtConfident",0)+d.get("fractionPlddtVeryHigh",0))
        vlow=d.get("fractionPlddtVeryLow",0)
        ok=(mean is not None and mean>=AF_PLDDT and conf>=AF_CONF and vlow<=AF_VLOW)
        return dict(ok=ok, mean=round(mean,1) if mean else None, conf=round(conf,3), vlow=round(vlow,3),
                    url=d.get("pdbUrl",""))
    except Exception: return None


def assign_structure(rows):
    """Set structure_source + QC for each row in-place. PDB-first, AF fallback (threaded)."""
    need_af=[]
    for r in rows:
        rec=best_pdb(r["pdb"], r["length"]); ok,flags,border=pdb_qc(rec)
        r["_pdb_rec"]=rec
        if ok:
            r.update(structure_source="PDB", pdb_id=rec["pdb_id"], pdb_chain=rec["chain"],
                     method=rec["method"], resolution=rec["res"], uniprot_start=rec["start"],
                     uniprot_end=rec["end"], chain_coverage_fraction=rec["cov"],
                     af_mean_plddt="", af_conf_fraction="", af_verylow_fraction="",
                     qc_pass=True, qc_flags=("coverage_borderline" if border else ""), _border=border)
        else:
            need_af.append(r)
    # AF fallback, threaded
    def work(r): return r, af_qc(r["acc"])
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r, af in ((f.result()) for f in as_completed([ex.submit(work,r) for r in need_af])):
            len_ok = bool(r["length"]) and LEN_MIN <= r["length"] <= LEN_MAX
            if af and af["ok"] and len_ok:
                r.update(structure_source="AF", pdb_id="", pdb_chain="", method="AlphaFold", resolution="",
                         uniprot_start="", uniprot_end="", chain_coverage_fraction="",
                         af_mean_plddt=af["mean"], af_conf_fraction=af["conf"], af_verylow_fraction=af["vlow"],
                         qc_pass=True, qc_flags="", _border=False)
            else:
                why = ("af_length_out_of_range" if (af and af["ok"] and not len_ok)
                       else "pdb_fail+af_fail" if af else "pdb_fail+no_af")
                r.update(structure_source="none", qc_pass=False,
                         qc_flags=why, _border=False,
                         pdb_id="", pdb_chain="", method="", resolution="", uniprot_start="", uniprot_end="",
                         chain_coverage_fraction="", af_mean_plddt=af["mean"] if af else "",
                         af_conf_fraction=af["conf"] if af else "", af_verylow_fraction=af["vlow"] if af else "")


def pfam_set(s): return set(x for x in (s or "").split(";") if x)


def match(cases, pool):
    used=set(); res={}
    pool_ok=[c for c in pool if c["qc_pass"]]
    for case in sorted(cases, key=lambda c:(c["length"] or 0, c["acc"])):
        chosen=None; tier="unmatched"; cpf=pfam_set(case["pfam"])
        base=[c for c in pool_ok if c["acc"] not in used and c["domain"]==case["domain"]
              and case["length"] and c["length"] and abs(c["length"]-case["length"])<=0.25*case["length"]]
        tiers=[("Pfam",lambda c:bool(cpf & pfam_set(c["pfam"]))),
               ("EC3",lambda c:case["ec3"] and c["ec3"]==case["ec3"]),
               ("EC2",lambda c:case["ec2"] and c["ec2"]==case["ec2"]),
               ("EC_class",lambda c:c["ec_class"]==case["ec_class"])]
        for tname,pred in tiers:
            cand=[c for c in base if pred(c)]
            if cand:
                # prefer same structure_source, then closest length
                chosen=min(cand, key=lambda c:(c["structure_source"]!=case["structure_source"],
                                               abs(c["length"]-case["length"]), c["acc"]))
                tier=tname; used.add(chosen["acc"]); break
        res[case["acc"]]=(tier, chosen)
    return res


STRUCT_COLS=["structure_source","pdb_id","pdb_chain","method","resolution","uniprot_start","uniprot_end",
             "chain_coverage_fraction","af_mean_plddt","af_conf_fraction","af_verylow_fraction","qc_pass","qc_flags"]


def write_set(tag, cases, matches, conf):
    base=["acc","label_source","case_set","pH_optimum_value","pH_optimum_range","range_width","label_confidence",
          "annotation_text","evidence_ids","ec_full","ec3","ec_class","pfam","domain","organism","localization",
          "secreted","oligomeric_note","length","sequence","surface_charge_features_available"]
    extra=["match_tier","matched_control_uniprot","cluster_id","sequence_identity_to_control","split"]
    with open(OUT/f"alkaline_optimum_cases_{tag}_stageC.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=base+STRUCT_COLS+extra, extrasaction="ignore"); w.writeheader()
        for r in cases:
            t,c=matches[r["acc"]]
            row={k:r.get(k,"") for k in base+STRUCT_COLS}
            row.update(label_source="UniProt pH optimum", case_set=tag,
                pH_optimum_value=round((r["opt_lo"]+r["opt_hi"])/2,2), pH_optimum_range=f"{r['opt_lo']}-{r['opt_hi']}",
                range_width=round(r["opt_hi"]-r["opt_lo"],2), label_confidence=conf(r),
                annotation_text=r["ph_text"], surface_charge_features_available=True,
                match_tier=t, matched_control_uniprot=c["acc"] if c else "",
                cluster_id="TBD", sequence_identity_to_control="TBD", split="TBD")
            w.writerow(row)
    cbase=["acc","label_source","for_case_set","pH_optimum_value","pH_optimum_range","ec_full","ec3","ec_class",
           "pfam","domain","organism","length","sequence"]
    with open(OUT/f"matched_neutral_controls_for_{tag}_cases_stageC.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=cbase+STRUCT_COLS+["matched_case_uniprot","match_tier","cluster_id","split"], extrasaction="ignore"); w.writeheader()
        for r in cases:
            t,c=matches[r["acc"]]
            if not c: continue
            row={k:c.get(k,"") for k in cbase+STRUCT_COLS}
            row.update(label_source="UniProt pH optimum (neutral)", for_case_set=tag,
                pH_optimum_value=round((c["opt_lo"]+c["opt_hi"])/2,2), pH_optimum_range=f"{c['opt_lo']}-{c['opt_hi']}",
                matched_case_uniprot=r["acc"], match_tier=t, cluster_id="TBD", split="TBD")
            w.writerow(row)


def main():
    print("harvesting ALL reviewed pH-annotated entries (no PDB filter) ...")
    rows=harvest('(cc_bpcp_ph_dependence:*) AND reviewed:true')
    for r in rows: r["ec3"],r["ec2"],r["ec_class"]=ec_parts(r["ec_full"])
    parsed=[r for r in rows if r["opt_hi"] is not None]
    def conf(r): return "high" if (r["opt_lo"]>=8.5 and (r["opt_hi"]-r["opt_lo"])<=1.0) else "medium"
    cases_incl=[r for r in parsed if r["opt_hi"]>=8.5]
    cases_high=[r for r in cases_incl if r["opt_lo"]>=8.5 and (r["opt_hi"]-r["opt_lo"])<=1.0]
    pool=[r for r in parsed if r["opt_lo"]>=6.0 and r["opt_hi"]<=7.5]
    print(f"parsed {len(parsed)} | high-conf {len(cases_high)} | inclusive {len(cases_incl)} | neutral pool {len(pool)}")

    universe={r["acc"]:r for r in cases_incl+pool}.values()
    print(f"assigning structure_source + per-source QC to {len(list(universe))} proteins (PDB then AF) ...", flush=True)
    universe=list({r["acc"]:r for r in cases_incl+pool}.values())
    assign_structure(universe)
    src=collections.Counter((r["structure_source"]) for r in universe)
    print(f"  structure_source over union: {dict(src)}")

    for tag, cs in [("high_confidence",cases_high),("inclusive",cases_incl)]:
        cs_ok=[r for r in cs if r["qc_pass"]]
        m=match(cs_ok, pool)
        write_set(tag, cs_ok, m, conf)
        npair=sum(1 for r in cs_ok if m[r["acc"]][1])
        bs=collections.Counter(r["structure_source"] for r in cs_ok)
        print(f"  [{tag}] QC-pass cases {len(cs_ok)}/{len(cs)} {dict(bs)} | matched pairs {npair}")

    # ecological supplement (separate, unchanged)
    genera=['Alkalihalobacillus','Bacillus halodurans','Bacillus pseudofirmus','Halalkalibacterium',
            'Natronomonas','Halorhodospira','Alkaliphilus','Thioalkalivibrio','Natranaerobius','Natrialba']
    orgq=" OR ".join(f'organism_name:"{g}"' for g in genera)
    eco=harvest(f'reviewed:true AND (keyword:KW-0964) AND ({orgq})')
    with open(OUT/"alkaline_ecological_supplement.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["acc","organism","domain","localization","ec_full","pfam","length","ph_text","sequence"], extrasaction="ignore")
        w.writeheader(); [w.writerow(r) for r in eco]
    print(f"ecological supplement: {len(eco)} (separate). Next: Stage D clustering/splits on the _stageC.csv.")


if __name__ == "__main__":
    main()
