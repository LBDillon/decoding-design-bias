"""
Stage D for the environmental matched datasets (alkaliphile OR acidophile): cluster at 40%
identity and assign CLUSTERS to train(70)/val(15)/test(15) so no pair across splits shares >40%
identity. Matched case+control are forced into the SAME split (within-split contrast). Flags the
tight-family-only pairs (both members) as the phylogeny-controlled sensitivity subset (`tight_pair`).

Reuses the 40%-identity clustering machinery from stage_d_cluster_split.py. `--cohort {alkaline,acid}`
selects the stage-C inputs / stage-D outputs (see _cohort.py). Reads {cohort} cases + controls
stage-C; writes *_stageD.csv with cluster_id, split, sequence_identity_to_control, tight_pair.
"""
import sys, random, collections, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"; sys.path.insert(0, str(HERE))
import pandas as pd
from stage_d_cluster_split import UF, pid, kmers, ID_THRESH
from _cohort import cfg
random.seed(0)


def cluster_and_split(C, target=(0.70, 0.15, 0.15)):
    ca = pd.read_csv(OUT / C["cases_C"])
    co = pd.read_csv(OUT / C["ctrls_C"])
    ca["matched_control_uniprot"] = ca.matched_control_uniprot.fillna("").astype(str)
    pairs = ca[ca.matched_control_uniprot.str.len() > 0].copy()
    cseq = dict(zip(ca.acc, ca.sequence)); oseq = dict(zip(co.acc, co.sequence))
    cbac = dict(zip(ca.acc, ca[C["tight_in"]])); obac = dict(zip(co.acc, co[C["tight_in"]]))
    seqs = {a: cseq[a] for a in pairs.acc} | {r.matched_control_uniprot: oseq[r.matched_control_uniprot]
                                              for r in pairs.itertuples() if r.matched_control_uniprot in oseq}
    accs = list(seqs); km = {a: kmers(seqs[a]) for a in accs}
    print(f"usable pairs {len(pairs)} | sequences {len(accs)}")

    # 40%-identity clustering: 5-mer prefilter then alignment
    uf = UF(accs); n_align = 0
    for i in range(len(accs)):
        ai = accs[i]; ki = km[ai]
        for j in range(i+1, len(accs)):
            aj = accs[j]
            if len(ki & km[aj]) / max(1, min(len(ki), len(km[aj]))) < 0.10:
                continue
            n_align += 1
            if pid(seqs[ai], seqs[aj]) >= ID_THRESH: uf.union(ai, aj)
    for r in pairs.itertuples():
        if r.matched_control_uniprot in seqs: uf.union(r.acc, r.matched_control_uniprot)
    clusters = collections.defaultdict(list)
    for a in accs: clusters[uf.find(a)].append(a)
    cl_ids = {a: f"{C['cl_prefix']}_cl{idx}" for idx, members in
              enumerate(sorted(clusters.values(), key=len, reverse=True)) for a in members}
    print(f"{len(clusters)} clusters at {int(ID_THRESH*100)}% identity ({n_align} alignments); "
          f"largest cluster {max(len(v) for v in clusters.values())} seqs")

    # assign cluster-groups to splits 70/15/15 by pair count
    groups = collections.defaultdict(list)
    for r in pairs.itertuples(): groups[uf.find(r.acc)].append(r.acc)
    npair = len(pairs); tgt = {s: t*npair for s, t in zip(["train","val","test"], target)}
    cur = {"train":0,"val":0,"test":0}; assign = {}
    for g in sorted(groups, key=lambda g: -len(groups[g])):
        s = min(["train","val","test"], key=lambda s: cur[s]-tgt[s])
        assign[g] = s; cur[s] += len(groups[g])
    pair_split = {a: assign[uf.find(a)] for a in pairs.acc}

    ident = {r.acc: round(pid(cseq[r.acc], oseq[r.matched_control_uniprot]), 3)
             for r in pairs.itertuples() if r.matched_control_uniprot in oseq}
    bpair = {r.acc: bool(cbac.get(r.acc)) and bool(obac.get(r.matched_control_uniprot))
             for r in pairs.itertuples()}

    ca["cluster_id"] = ca.acc.map(cl_ids); ca["split"] = ca.acc.map(pair_split)
    ca["sequence_identity_to_control"] = ca.acc.map(ident)
    ca["tight_pair"] = ca.acc.map(bpair)
    co["cluster_id"] = co.acc.map(cl_ids)
    co["split"] = co.matched_case_uniprot.map(pair_split)
    co["tight_pair"] = co.matched_case_uniprot.map(bpair)
    ca.to_csv(OUT / C["cases_D"], index=False)
    co.to_csv(OUT / C["ctrls_D"], index=False)

    sp = ca.dropna(subset=["split"])
    print("split (pairs):", dict(sp.split.value_counts()))
    print(f"case-control identity: median {sp.sequence_identity_to_control.median():.2f} "
          f"(>40% pairs: {(sp.sequence_identity_to_control>0.4).sum()})")
    print(f"tight-family-only pairs (sensitivity subset): {int(sp.tight_pair.sum())}")
    print("  by split:", dict(sp[sp.tight_pair].split.value_counts()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="alkaline", choices=["alkaline", "acid"])
    a = ap.parse_args()
    cluster_and_split(cfg(a.cohort))
    print(f"\nWrote {a.cohort} *_stageD.csv. Train on matched cases (Option 3), hold out val/test "
          "clusters; tight_pair subset = phylogeny-controlled sensitivity.")
