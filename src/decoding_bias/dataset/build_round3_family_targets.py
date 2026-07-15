"""
Build the round-3 family-centric expansion target table.

Reads the main dataset family composition and produces
  dataset_update/expansion_round3_family_targets.csv

with one row per (protein_family, target_domain, move) and a UniProt query
string ready for the family-aware orchestrator.

Three "moves":

  A - Multi-domain promotion. For each single-domain family with ≥5 members,
      add a small number of members in each of the two missing domains.
      Goal: turn single-domain families into multi-domain families so they
      become usable for cross-domain variance decomposition.

  B - Non-ribosomal big-family boost. For each family with 30-120 current
      members that is NOT ribosomal, top up to ~80 members, distributed
      across domains where the family is already present. Goal: break the
      ribosomal dominance among large families.

  C - Within-cell density. For each "universal" family (n_domains==3,
      n_members ≥ 50) that is NOT ribosomal, target adding paralogs in the
      species that currently have only 1 member of that family in main.
      Goal: increase the family×species cells that have ≥2 members
      (the within-cell variance enables variance decomposition).
"""

from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).parent
MAIN_PATH = HERE / "Decoding_Bias_Dataset_updated.csv"
OUT = HERE / "expansion_round3_family_targets.csv"

DOMAIN_TAXID = {"Archaea": 2157, "Bacteria": 2, "Eukaryota": 2759}
ALL_DOMAINS = list(DOMAIN_TAXID.keys())

# Tunables
MOVE_A_PER_MISSING_DOMAIN = 5        # 5 per missing domain per family
MOVE_A_MIN_CURRENT_MEMBERS = 5       # only promote families that already have ≥5 members
MOVE_B_TARGET_TOTAL = 80             # boost non-ribo families to ~80 members
MOVE_B_SIZE_RANGE = (30, 120)        # only apply to families currently in this range
MOVE_C_UNIVERSAL_MIN_MEMBERS = 50    # families must be ≥50 members to be a Move-C target
MOVE_C_MAX_PARALOGS_PER_SPECIES = 1  # target 1 additional paralog per single-member species
MOVE_C_MAX_PER_FAMILY = 40           # cap per family

# UniProt query fragments
BASE_FILTER = (
    'reviewed:true AND '
    'length:[50 TO 1000] AND '
    'NOT keyword:KW-0689'  # exclude ribosomal entries even when the family is non-ribosomal
)


def build_query(family_name, domain=None, species_id=None):
    parts = [f'({BASE_FILTER})', f'family:"{family_name}"']
    if domain:
        parts.append(f'taxonomy_id:{DOMAIN_TAXID[domain]}')
    if species_id:
        parts.append(f'organism_id:{int(species_id)}')
    return ' AND '.join(parts)


def main():
    df = pd.read_csv(MAIN_PATH)
    print(f"Loaded {len(df)} proteins, {df['protein_family'].nunique()} families from main.")

    # Per-family stats
    fam = df.groupby('protein_family').agg(
        n_members=('Entry', 'size'),
        n_species=('species', 'nunique'),
        n_domains=('domain', 'nunique'),
        dominant_function=('broad_function', lambda x: x.mode().iloc[0]),
    )
    # Which domains is each family already in?
    fam_domains = df.groupby('protein_family')['domain'].agg(lambda x: set(x.dropna().unique()))
    fam['domains_present'] = fam_domains
    fam = fam.reset_index()
    # Skip 'Unclassified' - it's a catch-all label, not a real family
    fam = fam[fam['protein_family'] != 'Unclassified'].copy()

    rows = []

    # ---- MOVE A: single-domain → multi-domain ----
    move_a = fam[(fam['n_domains'] == 1) & (fam['n_members'] >= MOVE_A_MIN_CURRENT_MEMBERS)]
    print(f"\nMove A - single-domain promotion: {len(move_a)} families")
    for _, r in move_a.iterrows():
        missing = [d for d in ALL_DOMAINS if d not in r['domains_present']]
        for d in missing:
            rows.append({
                'move': 'A_multidomain',
                'protein_family': r['protein_family'],
                'current_n_members': int(r['n_members']),
                'current_n_domains': int(r['n_domains']),
                'current_n_species': int(r['n_species']),
                'target_domain': d,
                'target_n': MOVE_A_PER_MISSING_DOMAIN,
                'query': build_query(r['protein_family'], domain=d),
                'rationale': f"promote {r['n_members']}-member single-domain family to {d}",
            })

    # ---- MOVE B: non-ribo big family boost ----
    move_b = fam[
        (fam['dominant_function'] != 'ribosomal')
        & (fam['n_members'] >= MOVE_B_SIZE_RANGE[0])
        & (fam['n_members'] <= MOVE_B_SIZE_RANGE[1])
    ]
    print(f"Move B - non-ribosomal big-family boost: {len(move_b)} families")
    for _, r in move_b.iterrows():
        deficit = max(0, MOVE_B_TARGET_TOTAL - int(r['n_members']))
        if deficit == 0:
            continue
        # Distribute the deficit across domains the family already inhabits
        present = sorted(r['domains_present'])
        per_dom = max(1, deficit // len(present))
        for d in present:
            rows.append({
                'move': 'B_bignonribo',
                'protein_family': r['protein_family'],
                'current_n_members': int(r['n_members']),
                'current_n_domains': int(r['n_domains']),
                'current_n_species': int(r['n_species']),
                'target_domain': d,
                'target_n': per_dom,
                'query': build_query(r['protein_family'], domain=d),
                'rationale': f"boost {r['dominant_function']} family from "
                             f"{r['n_members']} → ~{MOVE_B_TARGET_TOTAL}",
            })

    # ---- MOVE C: density boost in universal non-ribo families ----
    move_c = fam[
        (fam['n_domains'] == 3)
        & (fam['n_members'] >= MOVE_C_UNIVERSAL_MIN_MEMBERS)
        & (fam['dominant_function'] != 'ribosomal')
    ]
    print(f"Move C - universal non-ribo density: {len(move_c)} families")
    for _, r in move_c.iterrows():
        # For each species in this family with exactly 1 member, target 1 paralog
        sub = df[df['protein_family'] == r['protein_family']]
        sp_counts = sub['species'].value_counts()
        single_member_species = sp_counts[sp_counts == 1].head(MOVE_C_MAX_PER_FAMILY)
        # Add one row per family, query covers all those species (we'll cap at fetch time)
        # Keep this lighter than per-species rows; query is family-wide and we accept any species
        rows.append({
            'move': 'C_density',
            'protein_family': r['protein_family'],
            'current_n_members': int(r['n_members']),
            'current_n_domains': int(r['n_domains']),
            'current_n_species': int(r['n_species']),
            'target_domain': 'any',
            'target_n': min(len(single_member_species), MOVE_C_MAX_PER_FAMILY),
            'query': build_query(r['protein_family']),  # no domain filter
            'rationale': f"add paralogs in {len(single_member_species)} "
                         f"single-member species (universal {r['dominant_function']} family)",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    # Summary
    print(f"\nWrote {OUT}: {len(out)} target rows")
    print(f"  Total target_n across all moves: {out['target_n'].sum()}")
    print(f"  Breakdown by move:")
    print(out.groupby('move').agg(rows=('target_n','size'), total_target=('target_n','sum')).to_string())
    print(f"\n  Breakdown by target_domain (Move A + B only):")
    ab = out[out['move'].isin(['A_multidomain', 'B_bignonribo'])]
    print(ab.groupby('target_domain')['target_n'].sum().to_string())
    print(f"\n  Sample queries:")
    for _, r in out.head(5).iterrows():
        print(f"    [{r['move']}] {r['protein_family'][:60]} → {r['target_domain']} (n={r['target_n']})")
        print(f"      query: {r['query']}")


if __name__ == "__main__":
    main()
