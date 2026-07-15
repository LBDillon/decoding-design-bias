"""Build the MIF/MIF-ST-safe subset of the independent-cohort scoring inputs.

The Microsoft `sequence_models.pdb_utils.parse_PDB` parser builds a backbone
coordinate array over the full residue-number span of the chain and requires
complete N/CA/C atoms; residue-numbering gaps or missing backbone atoms produce
NaN coordinates that break `process_coords`. We therefore emit a conservative
subset whose single-chain PDBs satisfy, for the resolved chain:
  - contiguous residue numbering (max-min+1 == n_residues; no internal gaps),
  - no insertion codes,
  - complete N, CA, C backbone atoms for every residue.

Inputs : outputs/independent_cohort/cohort_pdb_scoring_inputs.csv (chain 'A' PDBs)
Outputs: outputs/independent_cohort/cohort_pdb_scoring_inputs_mif_safe.csv
         outputs/independent_cohort/cohort_pdb_scoring_inputs_mif_qc.csv  (per-row reason)
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import biotite.structure.io.pdb as pdbio
import biotite.structure as struc

HERE = Path(__file__).resolve().parent
COH = HERE / "outputs" / "independent_cohort"
IN_CSV = COH / "cohort_pdb_scoring_inputs.csv"
BACKBONE = {"N", "CA", "C"}


def check(pdb_path: str, seq_len: int) -> str | None:
    """Return None if MIF-safe, else a short reason string."""
    try:
        arr = pdbio.PDBFile.read(pdb_path).get_structure(model=1)
    except Exception as ex:  # noqa: BLE001
        return f"parse_error:{str(ex)[:30]}"
    aa = arr[struc.filter_amino_acids(arr)]
    if aa.array_length() == 0:
        return "no_amino_acids"
    if np.any(aa.ins_code != ""):
        return "insertion_codes"

    # residue order along the chain
    res_ids = aa.res_id[struc.get_residue_starts(aa)]
    n_res = len(res_ids)
    if n_res != seq_len:
        return f"residue_count_{n_res}!=seq_{seq_len}"
    span = int(res_ids.max() - res_ids.min() + 1)
    if span != n_res:
        return f"numbering_gap_span_{span}!=n_{n_res}"

    # complete N/CA/C per residue
    starts = struc.get_residue_starts(aa, add_exclusive_stop=True)
    for i in range(len(starts) - 1):
        names = set(aa.atom_name[starts[i]:starts[i + 1]])
        if not BACKBONE.issubset(names):
            return "incomplete_backbone"
    return None


def main():
    df = pd.read_csv(IN_CSV)
    reasons = []
    for r in df.itertuples(index=False):
        reasons.append(check(str(r.chain_pdb_path), len(str(r.sequence))))
    df = df.copy()
    df["mif_fail_reason"] = reasons
    df.to_csv(COH / "cohort_pdb_scoring_inputs_mif_qc.csv", index=False)

    safe = df[df["mif_fail_reason"].isna()].drop(columns=["mif_fail_reason"])
    safe.to_csv(COH / "cohort_pdb_scoring_inputs_mif_safe.csv", index=False)

    print(f"total: {len(df)}  |  MIF-safe: {len(safe)} ({len(safe)/len(df)*100:.0f}%)")
    print("exclusion reasons (categorised):")
    cat = df["mif_fail_reason"].dropna().str.replace(r"_.*", "", regex=True)
    print(cat.value_counts().to_string())
    print("\nMIF-safe by domain:")
    print(safe["domain"].value_counts().to_string())
    print(f"\nwrote cohort_pdb_scoring_inputs_mif_safe.csv + _mif_qc.csv")


if __name__ == "__main__":
    main()
