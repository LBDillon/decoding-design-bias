"""
design_common.py - shared configuration, I/O schema, and comparability guards
for the inverse-folding design-bias experiment.

EVERY design notebook imports this module so that the knobs that must stay
constant across models live in exactly one place. If a notebook needs to
deviate (e.g. a model that cannot honour a setting), it must do so loudly via
`CONFIG.note_deviation(...)` so the deviation is recorded in the output.

Design philosophy
-----------------
We separate DESIGN (each model produces sequences) from EVALUATION (one shared
notebook refolds everything and computes properties). This module defines the
contract between the two stages: the standardized per-sequence output schema.

Usage in a Colab notebook
-------------------------
    import design_common as dc
    proteins = dc.load_inputs()                 # DataFrame of the 25 templates
    rows = []
    for p in proteins.itertuples():
        for i in range(dc.CONFIG.num_seqs_per_protein):
            seq, score = my_model_design(p.structure_path, temperature=dc.CONFIG.temperature)
            rows.append(dc.make_record(p, model="MyModel", sample_idx=i,
                                       designed_sequence=seq, model_score=score))
    df = dc.finalize(rows, model="MyModel")      # validates + returns DataFrame
    dc.write_designs(df, model="MyModel")        # writes designs_MyModel.csv
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1. SHARED CONFIG - the constants that MUST be identical across all models
# ──────────────────────────────────────────────────────────────────────────────

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids


@dataclass
class DesignConfig:
    # --- locked sampling knobs (identical for every model) ---
    temperature: float = 0.1            # sampling temperature for all models
    num_seqs_per_protein: int = 8       # sequences generated per template
    seeds: tuple = (0, 1, 2, 3, 4, 5, 6, 7)  # one seed per sample; len must == num_seqs

    # --- locked sampling scheme ---
    # We use TEMPERATURE-ONLY sampling everywhere. Extra nucleus/top-k filters are
    # disabled so models that expose them (e.g. MIF) match those that do not
    # (Caliby, ProteinMPNN, ESM-IF).
    top_k: int = 0                      # 0 = disabled
    top_p: float = 1.0                  # 1.0 = disabled
    omit_aas: tuple = ()                # do not omit any AA (e.g. keep Cys)

    # --- locked design scope ---
    design_chain: str = "A"             # monomer, chain A
    full_redesign: bool = True          # design every position; no fixed positions
    structure_model_version: str = "v6" # all structures are AF v6

    # --- locked evaluation protocol (Stage 2, shared refold) ---
    sc_num_models: int = 5              # AF2 models for self-consistency
    sc_num_recycles: int = 3            # AF2 recycles

    # --- deviations recorded at runtime (model couldn't honour a setting) ---
    deviations: list = field(default_factory=list)

    def note_deviation(self, model: str, setting: str, used, reason: str):
        """Record (loudly) that a model could not honour a locked setting."""
        msg = f"[DEVIATION] {model}: {setting} -> {used!r} ({reason})"
        print(msg)
        self.deviations.append(
            {"model": model, "setting": setting, "used": str(used), "reason": reason}
        )

    def __post_init__(self):
        assert len(self.seeds) == self.num_seqs_per_protein, (
            f"seeds ({len(self.seeds)}) must match num_seqs_per_protein "
            f"({self.num_seqs_per_protein})"
        )


CONFIG = DesignConfig()


# ──────────────────────────────────────────────────────────────────────────────
# 2. PATHS - resolve the bundle layout (works both locally and on Colab)
# ──────────────────────────────────────────────────────────────────────────────

def _bundle_root() -> Path:
    """Directory containing design_input_proteins.csv and structures/.
    On Colab after unzipping the bundle this is usually the CWD; locally it is
    the directory of this file. We try a few sensible locations."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for cand in (here, Path.cwd(), Path.cwd() / "design"):
        if (cand / "design_input_proteins.csv").exists():
            return cand
    return here


ROOT = _bundle_root()
INPUT_CSV = ROOT / "design_input_proteins.csv"
STRUCTURE_DIR = ROOT / "structures"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. INPUT LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_inputs(require_structures: bool = True) -> pd.DataFrame:
    """Load the 25-protein design template table and resolve structure paths.

    Adds a `structure_path` column pointing at the local v6 PDB in
    `structures/`. If `require_structures`, asserts every file is present."""
    df = pd.read_csv(INPUT_CSV)

    def resolve(row):
        # Prefer the colab-relative path; fall back to basename in structures/
        cand = ROOT / str(row.get("colab_structure_path", ""))
        if cand.exists():
            return str(cand)
        fn = os.path.basename(str(row["structure_pdb_v6"]))
        return str(STRUCTURE_DIR / fn)

    df["structure_path"] = df.apply(resolve, axis=1)

    if require_structures:
        missing = df.loc[~df["structure_path"].apply(os.path.exists), "uniprot_id"].tolist()
        assert not missing, f"Missing structure files for: {missing}"
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. STANDARDIZED OUTPUT SCHEMA  (the contract for Stage-2 evaluation)
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "uniprot_id",        # template id
    "species", "domain", "rank_class", "target_cell",  # carried metadata
    "model",             # e.g. "Caliby", "ProteinMPNN", "MIF", "ESM-IF"
    "soluble_variant",   # bool: soluble-trained variant?
    "sample_idx",        # 0..num_seqs-1
    "seed",              # seed used for this sample
    "temperature",       # sampling temperature actually used
    "wt_sequence",       # the native sequence (from CSV)
    "designed_sequence", # the model's design
    "seq_length",        # len(designed_sequence)
    "model_score",       # model's own score (Caliby U / mean log-prob / etc.)
    "score_type",        # what model_score means, e.g. "neglogp_sum", "mean_logp"
    "structure_path",    # the v6 PDB used as input
]


def make_record(protein, *, model: str, sample_idx: int,
                designed_sequence: str, model_score: Optional[float] = None,
                score_type: str = "", seed: Optional[int] = None,
                soluble_variant: bool = False,
                temperature: Optional[float] = None) -> dict:
    """Build one standardized output row from an input-protein namedtuple/row."""
    g = (lambda k: getattr(protein, k) if hasattr(protein, k) else protein[k])
    seq = designed_sequence.strip().upper()
    if seed is None:
        seed = CONFIG.seeds[sample_idx % len(CONFIG.seeds)]
    if temperature is None:
        temperature = CONFIG.temperature
    return {
        "uniprot_id": g("uniprot_id"),
        "species": g("species"),
        "domain": g("domain"),
        "rank_class": g("rank_class"),
        "target_cell": g("target_cell"),
        "model": model,
        "soluble_variant": soluble_variant,
        "sample_idx": sample_idx,
        "seed": seed,
        "temperature": temperature,
        "wt_sequence": g("wt_sequence"),
        "designed_sequence": seq,
        "seq_length": len(seq),
        "model_score": model_score,
        "score_type": score_type,
        "structure_path": g("structure_path"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. VALIDATION / COMPARABILITY GUARDS
# ──────────────────────────────────────────────────────────────────────────────

class ComparabilityError(AssertionError):
    pass


def validate_designs(df: pd.DataFrame, model: str, *, strict: bool = True) -> pd.DataFrame:
    """Run faithfulness/comparability checks on a model's design table.

    Checks:
      1. all required columns present
      2. designed sequences contain only canonical AAs
      3. each design length == its WT length (full redesign, no indels)
      4. correct number of sequences per protein
      5. temperature matches CONFIG (unless a deviation was recorded)
      6. designs are not trivially identical to WT (model actually designed)
    Returns the df unchanged; raises ComparabilityError on hard failures."""
    problems = []

    missing_cols = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")

    if df.empty:
        problems.append("no design rows produced")
        _raise_or_warn(problems, strict); return df

    # 2. canonical AAs only
    bad_aa = df[~df["designed_sequence"].apply(_is_canonical)]
    if len(bad_aa):
        ex = bad_aa["uniprot_id"].head(3).tolist()
        problems.append(f"{len(bad_aa)} designs contain non-canonical residues (e.g. {ex})")

    # 3. length preserved vs WT
    lmism = df[df["designed_sequence"].str.len() != df["wt_sequence"].str.len()]
    if len(lmism):
        ex = lmism["uniprot_id"].head(3).tolist()
        problems.append(f"{len(lmism)} designs differ in length from WT (e.g. {ex})")

    # 4. correct count per protein
    counts = df.groupby("uniprot_id").size()
    wrong = counts[counts != CONFIG.num_seqs_per_protein]
    if len(wrong):
        problems.append(
            f"{len(wrong)} proteins have != {CONFIG.num_seqs_per_protein} designs "
            f"(e.g. {wrong.head(3).to_dict()})"
        )

    # 5. temperature consistency
    temps = df["temperature"].unique()
    deviated_temp = any(d["setting"] == "temperature" for d in CONFIG.deviations)
    if not deviated_temp and not (len(temps) == 1 and abs(temps[0] - CONFIG.temperature) < 1e-9):
        problems.append(f"temperature(s) {temps} != CONFIG.temperature {CONFIG.temperature}")

    # 6. not identical to WT (recovery < 100%); warn-only
    rec = df.apply(lambda r: _seq_recovery(r["designed_sequence"], r["wt_sequence"]), axis=1)
    if (rec >= 0.999).all():
        problems.append("EVERY design is identical to WT - model likely not sampling")

    _raise_or_warn(problems, strict)

    # Informational summary
    print(f"[validate] {model}: {len(df)} designs across "
          f"{df['uniprot_id'].nunique()} proteins | "
          f"median seq-recovery vs WT = {rec.median():.1%} | "
          f"mean length = {df['seq_length'].mean():.0f} aa")
    return df


def _is_canonical(seq: str) -> bool:
    return bool(seq) and set(seq.upper()) <= set(CANONICAL_AA)


def _seq_recovery(a: str, b: str) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _raise_or_warn(problems, strict):
    if problems:
        msg = "Comparability/faithfulness issues:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ComparabilityError(msg)
        print("[WARN] " + msg)


def finalize(rows, model: str, *, strict: bool = True) -> pd.DataFrame:
    """Turn a list of make_record() dicts into a validated DataFrame."""
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return validate_designs(df, model, strict=strict)


# ──────────────────────────────────────────────────────────────────────────────
# 6. OUTPUT WRITING
# ──────────────────────────────────────────────────────────────────────────────

def write_designs(df: pd.DataFrame, model: str) -> Path:
    """Write designs_<model>.csv plus a run-manifest capturing the config used."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out = OUTPUT_DIR / f"designs_{safe}.csv"
    df.to_csv(out, index=False)

    manifest = {
        "model": model,
        "n_designs": int(len(df)),
        "n_proteins": int(df["uniprot_id"].nunique()),
        "config": {k: v for k, v in asdict(CONFIG).items() if k != "deviations"},
        "deviations": CONFIG.deviations,
    }
    (OUTPUT_DIR / f"manifest_{safe}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[write] {out}  ({len(df)} rows)")
    print(f"[write] {OUTPUT_DIR / f'manifest_{safe}.json'}")
    return out


def write_fasta(df: pd.DataFrame, model: str) -> Path:
    """Also emit a FASTA of all designs (handy for batch refolding tools)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
    out = OUTPUT_DIR / f"designs_{safe}.fasta"
    with open(out, "w") as fh:
        for r in df.itertuples():
            fh.write(f">{r.uniprot_id}|{r.model}|sample{r.sample_idx}|seed{r.seed}\n")
            fh.write(r.designed_sequence + "\n")
    print(f"[write] {out}")
    return out
