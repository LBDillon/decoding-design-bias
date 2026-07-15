"""
Environmental extremophile SECRETED-protein dataset builder - UNIFIED acid/alkaline (--cohort).

Cases  = secreted/signal-peptide proteins from extremophile organisms (alkaliphiles pH ~9-11, OR
         acidophiles pH ~1-4). Controls = secreted proteins from NEUTRALOPHILES (pH ~7), matched 1:1
         by family (Pfam->EC) + length. Label = organism ecology; composition is the readout -> non-circular.

  python build_dataset.py --cohort alkaline   # -> alkaliphile_cases_stageC.csv (designs go ACIDIC)
  python build_dataset.py --cohort acid        # -> acidophile_cases_stageC.csv  (designs go BASIC, the reverse)

Switching the cohort swaps only the organism panels (below) + the output prefix; everything else is
identical -- this is the modularity demonstration (docs/MODULARITY.md). Needs network.
"""
import sys, io, time, re, argparse, collections, random, warnings
from pathlib import Path
import pandas as pd, requests
warnings.filterwarnings("ignore")
from Bio.SeqUtils.ProtParam import ProteinAnalysis
random.seed(0)
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"
PER_ORG_CAP = 150
LEN_MIN, LEN_MAX = 40, 700
_STD = set("ACDEFGHIKLMNPQRSTVWY")


def gravy_of(seq):
    """Kyte-Doolittle GRAVY of a sequence (standard residues only); used to match controls on
    hydrophobicity so the case-control contrast isolates surface CHARGE, not bulk hydrophobicity."""
    c = "".join(x for x in str(seq) if x in _STD)
    try: return ProteinAnalysis(c).gravy() if c else 0.0
    except Exception: return 0.0

# ---- cohort organism panels: taxonomy_id -> (short name, clade) ----------------------------------
PANELS = {
    "alkaline": {
        "label": "alkaliphile", "tight_family": "Bacillaceae",
        "cases": {79880:("A. clausii","Bacilli"), 79681:("A. pseudofirmus","Bacilli"),
                  86665:("B. halodurans","Bacilli"), 1445:("A. alcalophilus","Bacilli"),
                  1218:("Alkaliphilus metalliredigens","Clostridia"), 461876:("Alkaliphilus oremlandii","Clostridia"),
                  375929:("Natranaerobius thermophilus","Clostridia"), 490314:("Dethiobacter alkaliphilus","Clostridia"),
                  106633:("Thioalkalivibrio","Gammaproteobacteria"), 2257:("Natronomonas pharaonis","Haloarchaea"),
                  29288:("Natronococcus occultus","Haloarchaea"), 13769:("Natrialba magadii","Haloarchaea"),
                  44930:("Natronobacterium gregoryi","Haloarchaea")},
        "neutral": {1423:("B. subtilis","Bacilli"), 1402:("B. licheniformis","Bacilli"),
                    1390:("B. amyloliquefaciens","Bacilli"), 1488:("Clostridium acetobutylicum","Clostridia"),
                    1502:("Clostridium perfringens","Clostridia"), 562:("E. coli","Gammaproteobacteria"),
                    287:("Pseudomonas aeruginosa","Gammaproteobacteria"), 2746:("Halomonas elongata","Gammaproteobacteria"),
                    2242:("Halobacterium salinarum","Haloarchaea"), 2246:("Haloferax volcanii","Haloarchaea")},
    },
    "acid": {
        "label": "acidophile", "tight_family": "Acidithiobacillus",
        "cases": {920:("A. ferrooxidans","Gammaproteobacteria"), 930:("A. thiooxidans","Gammaproteobacteria"),
                  33059:("A. caldus","Gammaproteobacteria"), 524:("Acidiphilium cryptum","Alphaproteobacteria"),
                  179:("Leptospirillum","Nitrospira")},
        "neutral": {562:("E. coli","Gammaproteobacteria"), 287:("Pseudomonas aeruginosa","Gammaproteobacteria"),
                    294:("Pseudomonas fluorescens","Gammaproteobacteria"), 90371:("Salmonella Typhimurium","Gammaproteobacteria"),
                    155892:("Caulobacter crescentus","Alphaproteobacteria")},
        # NOTE: a gravy_band was trialled here to fix the case>control hydrophobicity gap (+0.138) but did
        # NOT close it - within Pfam+length matches there is no gravy freedom (the hydrophobic neutralophile
        # proteins aren't in the cases' families). Left OFF; the confound is documented in docs/RESULTS.md.
        # To retry, add e.g. "gravy_band": 0.25 here AND broaden the neutral panel so in-family hydrophobic
        # homologs exist. match() still honours gravy_band when present.
    },
}


def harvest_secreted(taxid):
    url = "https://rest.uniprot.org/uniprotkb/search"
    q = (f"(taxonomy_id:{taxid}) AND (cc_scl_term:SL-0243 OR ft_signal:*) AND (length:[{LEN_MIN} TO {LEN_MAX}])")
    params = {"query": q, "format": "tsv", "size": 500,
              "fields": "accession,length,sequence,organism_name,lineage,xref_pfam,ec,protein_name,ft_signal,cc_subcellular_location"}
    frames = []
    while True:
        for attempt in range(4):
            try:
                r = requests.get(url, params=params, timeout=120); r.raise_for_status(); break
            except Exception:
                if attempt == 3: raise
                time.sleep(3*(attempt+1))
        frames.append(pd.read_csv(io.StringIO(r.text), sep="\t"))
        nxt = r.links.get("next", {}).get("url")
        if not nxt: break
        url, params = nxt, None
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pfam_set(s): return set(re.findall(r"(PF\d{5})", s)) if isinstance(s, str) else set()


def ec_parts(s):
    if not isinstance(s, str) or not s.strip(): return "", "", ""
    e = s.split(";")[0].strip(); p = e.split(".")
    ec3 = ".".join(p[:3]) if len(p) >= 3 and "-" not in p[:3] else ""
    ec2 = ".".join(p[:2]) if len(p) >= 2 and "-" not in p[:2] else ""
    return e, ec3, ec2


def collect(panel, group):
    rows = []
    for tid, (name, clade) in panel.items():
        d = harvest_secreted(tid)
        if len(d):
            d = d.drop_duplicates("Entry")
            if len(d) > PER_ORG_CAP: d = d.sample(PER_ORG_CAP, random_state=0)
            d["taxid"] = tid; d["org_short"] = name; d["clade"] = clade; d["group"] = group
            rows.append(d)
        print(f"  {group:<13} {tid:>7} {name:<28} secreted(40-700): {0 if not len(d) else len(d)}", flush=True)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return df.drop_duplicates("Sequence") if "Sequence" in df.columns else df


def prep(df, tight_family):
    df = df.rename(columns={"Entry":"acc","Length":"length","Sequence":"sequence","Organism":"organism",
                            "Taxonomic lineage":"lineage","Pfam":"pfam_raw","EC number":"ec_raw","Protein names":"protein_name"})
    df["pfam"] = df.pfam_raw.apply(lambda s: ";".join(sorted(pfam_set(s))))
    ecp = df.ec_raw.apply(ec_parts)
    df["ec_full"] = [a for a,_,_ in ecp]; df["ec3"] = [b for _,b,_ in ecp]; df["ec2"] = [c for _,_,c in ecp]
    df["ec_class"] = df.ec_full.apply(lambda e: e.split(".")[0] if isinstance(e,str) and e else "")
    df["gravy"] = df.sequence.apply(gravy_of)
    df["in_tight_subset"] = df.lineage.apply(lambda s: isinstance(s, str) and tight_family in s)
    df["secreted_evidence"] = df.apply(
        lambda r: "SCL_secreted" if isinstance(r.get("Subcellular location [CC]"),str) and "Secreted" in r["Subcellular location [CC]"]
        else "signal_peptide", axis=1)
    return df


def match(cases, pool, gravy_band=None):
    used = set(); res = {}
    for case in sorted(cases, key=lambda c:(c["length"] or 0, c["acc"])):
        cpf = pfam_set(case["pfam"])
        base = [c for c in pool if c["acc"] not in used and case["length"] and c["length"]
                and abs(c["length"]-case["length"]) <= 0.25*case["length"]]
        tiers = [("Pfam", lambda c: bool(cpf & pfam_set(c["pfam"]))),
                 ("EC3", lambda c: case["ec3"] and c["ec3"]==case["ec3"]),
                 ("EC2", lambda c: case["ec2"] and c["ec2"]==case["ec2"]),
                 ("EC_class", lambda c: case["ec_class"] and c["ec_class"]==case["ec_class"])]
        chosen, tier = None, "unmatched"
        for tname, pred in tiers:
            cand = [c for c in base if pred(c)]
            if not cand: continue
            if gravy_band is not None:   # prefer hydrophobicity-matched controls; fall back if none in band
                gd = lambda c: abs(c.get("gravy", 0.0) - case.get("gravy", 0.0))
                near = [c for c in cand if gd(c) <= gravy_band]
                if near: cand = near
                key = lambda c: (gd(c), c["clade"] == case["clade"], abs(c["length"]-case["length"]), c["acc"])
            else:
                key = lambda c: (c["clade"] == case["clade"], abs(c["length"]-case["length"]), c["acc"])
            chosen = min(cand, key=key); tier = tname; used.add(chosen["acc"]); break
        res[case["acc"]] = (tier, chosen)
    return res


def main(cohort):
    P = PANELS[cohort]; lab = P["label"]
    print(f"[cohort={cohort}] harvesting secreted proteins ({lab}) ...")
    cas = prep(collect(P["cases"], lab), P["tight_family"])
    print(f"[cohort={cohort}] harvesting secreted proteins (neutralophiles) ...")
    neu = prep(collect(P["neutral"], "neutralophile"), P["tight_family"])
    print(f"\n{lab} cases {len(cas)} | neutralophile pool {len(neu)}")
    print(f"  {lab} by clade:", dict(cas.clade.value_counts()))

    cases = cas.to_dict("records"); pool = neu.to_dict("records")
    gravy_band = P.get("gravy_band")
    m = match(cases, pool, gravy_band)
    npair = sum(1 for c in cases if m[c["acc"]][1])
    cross = sum(1 for c in cases if m[c["acc"]][1] and m[c["acc"]][1]["clade"]!=c["clade"])
    tiers = collections.Counter(m[c["acc"]][0] for c in cases if m[c["acc"]][1])
    print(f"\nmatched pairs {npair}/{len(cases)} | cross-clade {cross} | tiers {dict(tiers)}")
    if gravy_band is not None:   # report whether the hydrophobicity confound closed
        dg = [m[c["acc"]][1]["gravy"] - c["gravy"] for c in cases if m[c["acc"]][1]]
        mc = sum(c["gravy"] for c in cases if m[c["acc"]][1]) / max(1, npair)
        mk = mc + sum(dg)/max(1, npair)
        print(f"  GRAVY match (band ±{gravy_band}): case {mc:+.3f} vs control {mk:+.3f} "
              f"(mean case-control gap {-sum(dg)/max(1,npair):+.3f}; was +0.138 unmatched)")

    keep = ["acc","group","clade","org_short","organism","in_tight_subset","secreted_evidence",
            "pfam","ec_full","ec3","ec_class","length","gravy","sequence"]
    crows, ctrows = [], []
    for c in cases:
        t, ctrl = m[c["acc"]]
        row = {k:c.get(k,"") for k in keep}
        row.update(match_tier=t, matched_control_uniprot=ctrl["acc"] if ctrl else "",
                   cross_clade=(ctrl["clade"]!=c["clade"]) if ctrl else "")
        crows.append(row)
        if ctrl:
            cr = {k:ctrl.get(k,"") for k in keep}
            cr.update(matched_case_uniprot=c["acc"], match_tier=t, cross_clade=ctrl["clade"]!=c["clade"])
            ctrows.append(cr)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(crows).to_csv(OUT/f"{lab}_cases_stageC.csv", index=False)
    pd.DataFrame(ctrows).to_csv(OUT/f"{lab}_neutral_controls_stageC.csv", index=False)
    print(f"\nwrote {lab}_cases_stageC.csv ({len(crows)}) + {lab}_neutral_controls_stageC.csv ({len(ctrows)})")
    arrow = "ACIDIC (net negative)" if cohort == "alkaline" else "BASIC (net positive, the reverse)"
    print(f"EXPECTED on fine-tuning: designs go {arrow}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=["alkaline", "acid"])
    main(ap.parse_args().cohort)
