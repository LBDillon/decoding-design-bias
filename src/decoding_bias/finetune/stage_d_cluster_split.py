"""
Stage D: sequence clustering + train/val/test splits for the alkaline-optimum set.

Follows the original paper's method (FINAL_21_oct): cluster at 40% sequence identity,
randomly assign CLUSTERS to train(70)/val(15)/test(15) so no pair across splits shares
>40% identity. CD-HIT is unavailable, so we use a Biopython 40%-identity equivalent
(5-mer prefilter -> global alignment -> identity = identical/min(len)).

Improvements over the original (case-control design + clean external eval):
  - matched case & control are forced into the SAME split (union-find edge), so the
    contrast is always within-split;
  - cases overlapping the v12 cohort are flagged (in_v12) and routed to a 'train'
    exclusion note, so the v12 cohort stays a clean EXTERNAL evaluation set.

Writes *_stageD.csv with cluster_id, split, sequence_identity_to_control, in_v12.
"""
import sys, random, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "outputs"
import pandas as pd
from Bio import Align
random.seed(0)
ID_THRESH = 0.40
aligner = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=-1,
                                open_gap_score=-5, extend_gap_score=-0.5)


def kmers(s, k=5): return {s[i:i+k] for i in range(len(s)-k+1)} if len(s) >= k else {s}


def pid(a, b):
    if not a or not b: return 0.0
    aln = aligner.align(a, b)[0]
    ident = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        ident += sum(1 for x, y in zip(a[s1:e1], b[s2:e2]) if x == y)
    return ident / min(len(a), len(b))


class UF:
    def __init__(s, items): s.p = {i: i for i in items}
    def find(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b): s.p[s.find(a)] = s.find(b)


def cluster_and_split(tag, target=(0.70, 0.15, 0.15)):
    ca = pd.read_csv(OUT/f"alkaline_optimum_cases_{tag}_stageC.csv")
    co = pd.read_csv(OUT/f"matched_neutral_controls_for_{tag}_cases_stageC.csv")
    ctrl_ok = set(co[co.qc_pass].acc)
    pairs = ca[(ca.qc_pass) & (ca.matched_control_uniprot.isin(ctrl_ok)) & (ca.matched_control_uniprot != "")].copy()
    cseq = dict(zip(ca.acc, ca.sequence)); oseq = dict(zip(co.acc, co.sequence))
    seqs = {a: cseq[a] for a in pairs.acc} | {r.matched_control_uniprot: oseq[r.matched_control_uniprot] for r in pairs.itertuples()}
    accs = list(seqs); km = {a: kmers(seqs[a]) for a in accs}
    print(f"[{tag}] usable pairs {len(pairs)} | sequences {len(accs)}")

    # 40%-identity clustering: 5-mer prefilter then alignment
    uf = UF(accs); n_align = 0
    for i in range(len(accs)):
        ai = accs[i]; ki = km[ai]
        for j in range(i+1, len(accs)):
            aj = accs[j]
            inter = len(ki & km[aj])
            if inter / max(1, min(len(ki), len(km[aj]))) < 0.10:   # loose prefilter
                continue
            n_align += 1
            if pid(seqs[ai], seqs[aj]) >= ID_THRESH: uf.union(ai, aj)
    # force matched pairs together
    for r in pairs.itertuples(): uf.union(r.acc, r.matched_control_uniprot)
    clusters = collections.defaultdict(list)
    for a in accs: clusters[uf.find(a)].append(a)
    cl_ids = {a: f"{tag[:2]}_cl{idx}" for idx, members in enumerate(sorted(clusters.values(), key=len, reverse=True)) for a in members}
    print(f"   {len(clusters)} clusters at {int(ID_THRESH*100)}% identity ({n_align} alignments); "
          f"largest cluster {max(len(v) for v in clusters.values())} seqs")

    # assign clusters (as pair-groups) to splits 70/15/15, balancing pair count
    pair_cluster = {r.acc: uf.find(r.acc) for r in pairs.itertuples()}  # case-cluster = group
    groups = collections.defaultdict(list)
    for r in pairs.itertuples(): groups[uf.find(r.acc)].append(r.acc)
    gids = list(groups); random.shuffle(gids)
    npair = len(pairs); tgt = {s: t*npair for s, t in zip(["train","val","test"], target)}
    assign = {}; cur = {"train":0,"val":0,"test":0}
    for g in sorted(gids, key=lambda g: -len(groups[g])):  # big groups first
        s = min(["train","val","test"], key=lambda s: cur[s]-tgt[s])
        assign[g] = s; cur[s] += len(groups[g])
    pair_split = {ca_acc: assign[uf.find(ca_acc)] for ca_acc in pairs.acc}

    # per-pair identity + v12 flag
    v12 = set(pd.read_csv(HERE.parent/"dataset_update"/"main_plus_r2_r3_analysis_v12_corrected.csv",
                          usecols=["Entry"], low_memory=False).Entry)
    ident = {r.acc: round(pid(cseq[r.acc], oseq[r.matched_control_uniprot]), 3) for r in pairs.itertuples()}

    ca["cluster_id"] = ca.acc.map(cl_ids); ca["split"] = ca.acc.map(pair_split)
    ca["sequence_identity_to_control"] = ca.acc.map(ident); ca["in_v12"] = ca.acc.isin(v12)
    co["cluster_id"] = co.acc.map(cl_ids)
    co["split"] = co.matched_case_uniprot.map(pair_split)
    ca.to_csv(OUT/f"alkaline_optimum_cases_{tag}_stageD.csv", index=False)
    co.to_csv(OUT/f"matched_neutral_controls_for_{tag}_cases_stageD.csv", index=False)

    sp = ca.dropna(subset=["split"])
    print("   split (pairs):", dict(sp.split.value_counts()))
    print(f"   case-control identity: median {sp.sequence_identity_to_control.median():.2f} "
          f"(>40% pairs: {(sp.sequence_identity_to_control>0.4).sum()})")
    print(f"   in_v12 cases (exclude from training or from external eval): {int(sp.in_v12.sum())}")
    return pairs


def main():
    for tag in ["high_confidence", "inclusive"]:
        cluster_and_split(tag)
    print("\nWrote *_stageD.csv. Recommendation: train on the matched cases (Option 3),")
    print("hold out val/test clusters, use v12 (minus in_v12 cases) as the external eval.")


if __name__ == "__main__":
    main()
