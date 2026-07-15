# External services & compute environments

Several analyses depend on services or model weights that are **not conda-pinnable**
and are fetched from public sources at run time. They are deliberately kept out of
git (see `.gitignore`); this file records versions/endpoints so a reviewer can
reproduce them.

## Model scoring (per-model likelihoods)
Scoring was run across three runtimes; each notebook/script header records where it
ran. Model weights are downloaded from the sources below and are **not committed**.

| model(s) | runtime | source of weights |
|---|---|---|
| ProteinMPNN, SolubleMPNN, fine-tuned MPNN (v0.2.0) | local / Colab | dauparas/ProteinMPNN (GitHub) |
| ESM2 (650M) , ESM2-35M continual-pretraining | ARC / SLURM (`07_finetuning/slurm/`) | facebookresearch/esm; HuggingFace `facebook/esm2_t*` |
| ESM-IF | local / ARC | facebookresearch/esm |
| ESM3 | Colab | EvolutionaryScale (`esm` SDK; requires HF token - **do not commit**) |
| ProtGPT2, ProGen2-xl | Colab | HuggingFace `nferruz/ProtGPT2`, `hugohrban/progen2-xl` |
| CARP-640M, MIF, MIF-ST | Colab | microsoft/protein-sequence-models |
| Caliby, Soluble-Caliby | Colab | (per notebook header) |
| TriFlow | Colab | (per notebook header) |

> The ESM3 notebook uses the EvolutionaryScale `esm` SDK, which requires an HF token.
> No live credential is committed; supply your own token at run time.

## ColabFold (self-consistency / scTM - SI S8.4, S9.5, Fig 5)
Single-sequence refolding for design self-consistency (scTM) and the AFDB↔ColabFold
mean-offset calibration was run in **ColabFold** (Colab). Refold outputs (`*.pdb`,
`colabfold_output/`) are gitignored. Driven from
`notebooks/06_design/5_evaluate_self_consistency.ipynb`,
`notebooks/06_design/ft_self_consistency_colab.ipynb`, and
`07_finetuning/ft020_self_consistency_vs_afdb.py`.
- ColabFold: `sokrypton/ColabFold` (record the exact version tag used when re-running).

## DeepStabP (predicted thermostability / ΔTm - SI S8.3)
Melting-temperature predictions use the **DeepStabP** web service, called from
[`06_design/predict_tm_full.py`](06_design/predict_tm_full.py) (results feed
`06_design/tm_shift_analysis.py`).
- DeepStabP: https://csb-deepstabp.bio.rptu.de/ (record API version/date at run time).

## RCSB PDB (experimental-PDB cohort - SI S5)
The 876-chain experimental-PDB cohort is built by querying **RCSB** for
UniProt→PDB mappings and structures:
[`08_experimental_pdb/scan_uniprot_pdb.py`](08_experimental_pdb/scan_uniprot_pdb.py),
[`08_experimental_pdb/implement_uniprot_pdb_retrieval.py`](08_experimental_pdb/implement_uniprot_pdb_retrieval.py),
[`08_experimental_pdb/build_independent_pdb_cohort.py`](08_experimental_pdb/build_independent_pdb_cohort.py).
- RCSB Search API + data endpoints: https://data.rcsb.org / https://search.rcsb.org
  (record API version/date at run time). Structures are gitignored.

## UniProt
Dataset construction pulls UniProt annotations/ID mappings. Note that
`00_data/analyze_oligomerization_by_taxa.py` reads two dated UniProt exports from
`~/Downloads/` (`uniprotkb_..._2026_05_08.tsv`, `idmapping_2026_05_08.tsv`) - these
are **not portable** and must be re-downloaded.
