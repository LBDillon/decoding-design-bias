"""
ARC/GPU ESM2 masked pseudo-likelihood scoring.

This is the cluster-oriented version of score_esm2_local.py. It computes the
mean masked log-probability for each protein sequence:

    mean_i log p(sequence[i] | sequence with position i masked)

The legacy paper column, ESM2_15B_pppl_score, is kept as an alias for the mean
log-probability score, so higher is better. Pseudo-perplexity is also written as
exp(-mean_logp).

Examples
--------
Smoke test with the 650M model:

    python dataset_update/score_esm2_arc.py \
        --input dataset_update/round2/expansion_round2_for_scoring.csv \
        --output-dir dataset_update/scoring_results/esm2_650M_arc_smoke \
        --model-size 650M \
        --limit 5 \
        --rows-per-chunk 5 \
        --mask-batch-size 8

Array job task, zero-based shard index:

    python dataset_update/score_esm2_arc.py \
        --input dataset_update/main_plus_r2_r3_scored.csv \
        --output-dir dataset_update/scoring_results/esm2_15B_main_plus_r2_r3 \
        --model-size 15B \
        --only-missing-score-column ESM2_15B_pppl_score \
        --num-shards 32 \
        --shard-index 0 \
        --rows-per-chunk 10 \
        --mask-batch-size 1 \
        --no-merge

After all array tasks finish:

    python dataset_update/score_esm2_arc.py \
        --input dataset_update/main_plus_r2_r3_scored.csv \
        --output-dir dataset_update/scoring_results/esm2_15B_main_plus_r2_r3 \
        --model-size 15B \
        --merge-only
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import importlib.metadata as package_metadata
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


VALID_STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 1022
SESSION_ID = time.strftime("%Y%m%d_%H%M%S")

SCORE_DEFINITION = "mean log p(true residue | sequence with that residue masked)"
SEQUENCE_POLICY = "20 standard amino acids only; terminal stop removed; no X/B/Z/U/O replacement"

MODEL_MAP = {
    # Small checkpoints - useful for local CPU smoke tests:
    "8M":   "esm2_t6_8M_UR50D",
    "35M":  "esm2_t12_35M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    # Production sizes:
    "650M": "esm2_t33_650M_UR50D",
    "3B":   "esm2_t36_3B_UR50D",
    "15B":  "esm2_t48_15B_UR50D",
}

METADATA_COL_CANDIDATES = [
    "Entry",
    "species",
    "domain",
    "protein_family",
    "broad_function",
    "protein_name",
    "source",
    "structure_source",
    "pdb_id",
    "pdb_chain",
]

_MODEL_CACHE = {}


def score_columns(model_size: str) -> tuple[str, str, str, str, str]:
    prefix = f"ESM2_{model_size}"
    return (
        f"{prefix}_masked_mean_logp",
        f"{prefix}_masked_nll",
        f"{prefix}_pseudo_perplexity",
        f"{prefix}_positions_scored",
        f"{prefix}_pppl_score",
    )


def clean_sequence(seq) -> tuple[str, str]:
    if pd.isna(seq):
        return "", "missing"
    seq = str(seq).strip().upper().replace(" ", "").replace("\n", "")
    if seq.endswith("*"):
        seq = seq[:-1]
    bad = "".join(sorted(set(seq) - VALID_STANDARD_AA))
    return seq, bad


def sequence_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def classify_for_scoring(seq: str, bad: str, max_len: int) -> str:
    if len(seq) == 0:
        return "empty_sequence"
    if bad:
        return "nonstandard_amino_acid"
    if len(seq) > max_len:
        return "too_long"
    return "included"


def safe_package_version(package_name: str) -> str:
    try:
        return package_metadata.version(package_name)
    except Exception:
        return "unknown"


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_arg not in dtype_map:
        raise ValueError(f"Unknown dtype {dtype_arg!r}; choose auto, fp32, fp16, or bf16")
    return dtype_map[dtype_arg]


def cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def load_model(model_size: str, device_arg: str = "auto", dtype_arg: str = "auto"):
    cache_key = (model_size, device_arg, dtype_arg)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if model_size not in MODEL_MAP:
        raise ValueError(f"Unknown ESM2 model size {model_size!r}; choose from {sorted(MODEL_MAP)}")

    import esm

    device = choose_device(device_arg)
    dtype = choose_dtype(dtype_arg, device)
    model_name = MODEL_MAP[model_size]
    print(f"Loading {model_name} on {device} with dtype={dtype}...", flush=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    model = model.eval()
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device)

    _MODEL_CACHE[cache_key] = (model, alphabet, device, dtype, model_name)
    print("Model loaded.", flush=True)
    return _MODEL_CACHE[cache_key]


def score_sequence_batched(seq: str, model, alphabet, device: torch.device, mask_batch_size: int):
    data = [("seq", seq)]
    _, _, tokens = alphabet.get_batch_converter()(data)
    tokens = tokens.to(device)
    positions = torch.arange(1, tokens.shape[1] - 1, device=device)
    true_tokens = tokens[0, positions]
    logp_chunks = []

    for start in range(0, int(positions.numel()), mask_batch_size):
        pos = positions[start:start + mask_batch_size]
        true = true_tokens[start:start + pos.numel()]
        batch = tokens.repeat(pos.numel(), 1)
        batch[torch.arange(pos.numel(), device=device), pos] = alphabet.mask_idx

        with torch.inference_mode():
            out = model(batch, repr_layers=[], return_contacts=False)
            log_probs = F.log_softmax(out["logits"].float(), dim=-1)
            picked = log_probs[torch.arange(pos.numel(), device=device), pos, true]

        logp_chunks.append(picked.detach().cpu())
        del out, log_probs, picked, batch
        cleanup()

    logp = torch.cat(logp_chunks)
    mean_logp = float(logp.mean().item())
    nll = -mean_logp
    ppl = math.exp(nll) if nll < 80 else float("inf")
    return mean_logp, nll, ppl, int(logp.numel())


def score_with_retries(
    seq: str,
    model_size: str,
    mask_batch_size: int,
    device_arg: str,
    dtype_arg: str,
):
    model, alphabet, device, _dtype, model_name = load_model(model_size, device_arg, dtype_arg)
    current_batch = max(1, int(mask_batch_size))
    while True:
        try:
            mean_logp, nll, ppl, n_pos = score_sequence_batched(
                seq, model, alphabet, device, current_batch
            )
            mean_col, nll_col, ppl_col, pos_col, legacy_col = score_columns(model_size)
            return {
                mean_col: mean_logp,
                nll_col: nll,
                ppl_col: ppl,
                pos_col: n_pos,
                legacy_col: mean_logp,
                "ESM2_mask_batch_size_used": current_batch,
                "ESM2_error": "",
                "model_name": model_name,
            }
        except RuntimeError as err:
            msg = str(err).lower()
            oom = "out of memory" in msg or "cuda" in msg or "memory" in msg or "alloc" in msg
            if oom and current_batch > 1:
                next_batch = max(1, current_batch // 2)
                print(
                    f"  GPU/memory issue at mask_batch_size={current_batch}; retrying at {next_batch}",
                    flush=True,
                )
                current_batch = next_batch
                cleanup()
                continue
            cleanup()
            raise


def read_input(input_csv: str, max_len: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)
    if "sequence" not in df.columns:
        raise ValueError(f"Input CSV must have a 'sequence' column. Got: {list(df.columns)}")
    cleaned = df["sequence"].map(clean_sequence)
    df = df.copy()
    df["_sequence_clean"] = [item[0] for item in cleaned]
    df["_bad_chars"] = [item[1] for item in cleaned]
    df["_seq_sha1"] = df["_sequence_clean"].map(sequence_sha1)
    df["sequence_length_original"] = df["_sequence_clean"].str.len().astype(int)
    df["sequence_filter_status"] = [
        classify_for_scoring(seq, bad, max_len)
        for seq, bad in zip(df["_sequence_clean"], df["_bad_chars"])
    ]
    return df


def truthy_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def filtered_rows_for_scoring(
    df: pd.DataFrame,
    only_missing_score_column: str | None = None,
    where_column_is_true: str | None = None,
    offset: int | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if only_missing_score_column:
        if only_missing_score_column not in df.columns:
            print(
                f"Warning: --only-missing-score-column {only_missing_score_column!r} "
                "is not in the input; scoring all rows.",
                flush=True,
            )
        else:
            mask &= df[only_missing_score_column].isna()

    if where_column_is_true:
        if where_column_is_true not in df.columns:
            raise ValueError(f"--where-column-is-true column not found: {where_column_is_true}")
        mask &= truthy_series(df[where_column_is_true])

    filtered = df.loc[mask].copy()
    if offset:
        filtered = filtered.iloc[offset:]
    if limit:
        filtered = filtered.iloc[:limit]
    return filtered


def completed_hashes_from_chunks(output_dir: Path, model_size: str) -> set[str]:
    done = set()
    pattern = output_dir / f"esm2_{model_size}_masked_scores_*.csv"
    for prev in glob.glob(str(pattern)):
        try:
            prev_df = pd.read_csv(prev, usecols=lambda c: c in {"_seq_sha1", "sequence_filter_status"})
            if "_seq_sha1" not in prev_df.columns:
                continue
            if "sequence_filter_status" in prev_df.columns:
                ok = prev_df["sequence_filter_status"].ne("scoring_error")
                prev_df = prev_df.loc[ok]
            done |= set(prev_df["_seq_sha1"].dropna().astype(str))
        except Exception as err:
            print(f"Warning: could not read existing chunk {prev}: {err}", flush=True)
    return done


def make_unique_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    meta_cols = [c for c in METADATA_COL_CANDIDATES if c in df.columns]
    keep_cols = meta_cols + [
        "_seq_sha1",
        "_sequence_clean",
        "_bad_chars",
        "sequence_filter_status",
        "sequence_length_original",
    ]
    unique = df[keep_cols].drop_duplicates("_seq_sha1").reset_index(drop=True)
    return unique, meta_cols


def write_chunk(rows: list[dict], out_path: Path) -> None:
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_scoring(
    input_csv: str,
    output_dir: str,
    model_size: str,
    rows_per_chunk: int,
    mask_batch_size: int,
    max_len: int,
    score_truncated: bool,
    only_missing_score_column: str | None,
    where_column_is_true: str | None,
    num_shards: int,
    shard_index: int,
    limit: int | None,
    offset: int | None,
    partial_save_every: int,
    device_arg: str,
    dtype_arg: str,
    do_merge: bool,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full_df = read_input(input_csv, max_len=max_len)
    score_df = filtered_rows_for_scoring(
        full_df,
        only_missing_score_column=only_missing_score_column,
        where_column_is_true=where_column_is_true,
        offset=offset,
        limit=limit,
    )
    unique, meta_cols = make_unique_frame(score_df)

    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be zero-based and less than --num-shards")
    if num_shards > 1:
        unique = unique.iloc[shard_index::num_shards].reset_index(drop=True)

    done_sha1 = completed_hashes_from_chunks(output_path, model_size)
    todo = unique[~unique["_seq_sha1"].isin(done_sha1)].reset_index(drop=True)

    print(
        f"Input rows: {len(full_df):,} | rows selected for scoring: {len(score_df):,} | "
        f"unique selected: {len(unique):,} | done in chunks: {len(done_sha1):,} | "
        f"remaining in this shard: {len(todo):,}",
        flush=True,
    )
    if num_shards > 1:
        print(f"Shard: {shard_index}/{num_shards}", flush=True)

    if todo.empty:
        if do_merge:
            merge_chunks(input_csv, output_dir, model_size, max_len=max_len)
        return

    mean_col, nll_col, ppl_col, pos_col, legacy_col = score_columns(model_size)
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", str(shard_index))

    for chunk_start in range(0, len(todo), rows_per_chunk):
        part = todo.iloc[chunk_start:chunk_start + rows_per_chunk]
        chunk_end = chunk_start + len(part)
        out_file = (
            output_path
            / f"esm2_{model_size}_masked_scores_shard{shard_index:04d}_"
              f"{chunk_start:06d}_{chunk_end:06d}_{SESSION_ID}_{job_id}_{task_id}.csv"
        )
        rows = []

        for _, row in tqdm(part.iterrows(), total=len(part), desc=f"shard {shard_index} rows {chunk_start}-{chunk_end}", file=sys.stdout):
            rec = {c: row[c] for c in meta_cols}
            rec.update({
                "_seq_sha1": row["_seq_sha1"],
                "sequence_length_original": int(row["sequence_length_original"]),
                "sequence_filter_status": row["sequence_filter_status"],
                "model_family": "ESM2",
                "model_size": model_size,
                "score_definition": SCORE_DEFINITION,
                "sequence_policy": SEQUENCE_POLICY,
                "max_length": max_len,
                "session_id": SESSION_ID,
                "slurm_job_id": job_id,
                "slurm_array_task_id": task_id,
                "torch_version": torch.__version__,
                "fair_esm_version": safe_package_version("fair-esm"),
            })

            status = row["sequence_filter_status"]
            should_score = status == "included" or (status == "too_long" and score_truncated)
            if not should_score:
                rec.update({
                    mean_col: np.nan,
                    nll_col: np.nan,
                    ppl_col: np.nan,
                    pos_col: 0,
                    legacy_col: np.nan,
                    "sequence_length_scored": 0,
                    "sequence_was_truncated": False,
                    "ESM2_error": "",
                })
            else:
                seq_to_score = row["_sequence_clean"][:max_len]
                rec["sequence_length_scored"] = len(seq_to_score)
                rec["sequence_was_truncated"] = len(seq_to_score) < int(row["sequence_length_original"])
                if rec["sequence_was_truncated"]:
                    rec["sequence_filter_status"] = "too_long_truncated"
                try:
                    rec.update(
                        score_with_retries(
                            seq_to_score,
                            model_size=model_size,
                            mask_batch_size=mask_batch_size,
                            device_arg=device_arg,
                            dtype_arg=dtype_arg,
                        )
                    )
                except Exception as err:
                    cleanup()
                    rec["sequence_filter_status"] = "scoring_error"
                    rec.update({
                        mean_col: np.nan,
                        nll_col: np.nan,
                        ppl_col: np.nan,
                        pos_col: 0,
                        legacy_col: np.nan,
                        "ESM2_error": repr(err),
                    })
                    print(f"  ERROR on {rec.get('Entry', rec['_seq_sha1'])}: {err}", flush=True)

            rows.append(rec)
            if partial_save_every and len(rows) % partial_save_every == 0:
                write_chunk(rows, out_file)

        write_chunk(rows, out_file)
        print(f"Saved chunk: {out_file}", flush=True)

    if do_merge:
        merge_chunks(input_csv, output_dir, model_size, max_len=max_len)


def read_chunk_scores(output_dir: str, model_size: str) -> pd.DataFrame:
    output_path = Path(output_dir)
    parts = []
    for path in sorted(output_path.glob(f"esm2_{model_size}_masked_scores_*.csv")):
        try:
            part = pd.read_csv(path, low_memory=False)
            part["_source_file"] = path.name
            part["_source_mtime"] = path.stat().st_mtime
            parts.append(part)
        except Exception as err:
            print(f"Warning: could not read {path}: {err}", flush=True)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def derive_missing_score_companions(df: pd.DataFrame, model_size: str) -> pd.DataFrame:
    mean_col, nll_col, ppl_col, pos_col, legacy_col = score_columns(model_size)
    out = df.copy()
    if model_size == "15B" and legacy_col in out.columns:
        legacy = pd.to_numeric(out[legacy_col], errors="coerce")
        if mean_col not in out.columns:
            out[mean_col] = legacy
        else:
            out[mean_col] = pd.to_numeric(out[mean_col], errors="coerce").combine_first(legacy)
        if nll_col not in out.columns:
            out[nll_col] = -out[mean_col]
        if ppl_col not in out.columns:
            out[ppl_col] = np.exp(-out[mean_col])
    return out


def merge_chunks(input_csv: str, output_dir: str, model_size: str, max_len: int = DEFAULT_MAX_LEN) -> Path | None:
    full_df = read_input(input_csv, max_len=max_len)
    full_df = derive_missing_score_companions(full_df, model_size)
    scores = read_chunk_scores(output_dir, model_size)
    output_path = Path(output_dir)
    if scores.empty:
        print(f"No chunk files found in {output_path}", flush=True)
        return None

    scores = scores.sort_values(["_source_mtime", "_source_file"]).drop_duplicates("_seq_sha1", keep="last")
    mean_col, nll_col, ppl_col, pos_col, legacy_col = score_columns(model_size)
    score_cols = [
        mean_col,
        nll_col,
        ppl_col,
        pos_col,
        legacy_col,
        "sequence_length_scored",
        "sequence_was_truncated",
        "sequence_filter_status",
        "ESM2_error",
        "model_name",
        "model_size",
        "session_id",
        "_source_file",
    ]
    score_cols = [c for c in score_cols if c in scores.columns]
    right = scores[["_seq_sha1"] + score_cols].rename(columns={c: f"{c}__new" for c in score_cols})
    out = full_df.merge(right, on="_seq_sha1", how="left")

    for col in [mean_col, nll_col, ppl_col, pos_col, legacy_col]:
        new_col = f"{col}__new"
        if new_col not in out.columns:
            continue
        if col in out.columns:
            out[col] = out[col].combine_first(out[new_col])
        else:
            out[col] = out[new_col]

    for col in [
        "sequence_length_scored",
        "sequence_was_truncated",
        "ESM2_error",
        "model_name",
        "model_size",
        "session_id",
        "_source_file",
    ]:
        new_col = f"{col}__new"
        if new_col in out.columns and col not in out.columns:
            out[col] = out[new_col]

    new_status = "sequence_filter_status__new"
    if new_status in out.columns:
        out["ESM2_sequence_filter_status"] = out[new_status].combine_first(out["sequence_filter_status"])

    drop_cols = [c for c in out.columns if c.endswith("__new")]
    out = out.drop(columns=drop_cols)

    final_path = output_path / f"esm2_{model_size}_results_merged.csv"
    out.to_csv(final_path, index=False)
    print(f"Wrote merged results: {final_path}", flush=True)
    print(f"  Rows with {legacy_col}: {out[legacy_col].notna().sum():,} / {len(out):,}", flush=True)
    print(f"  Rows with new chunk score files: {out['_source_file'].notna().sum():,} / {len(out):,}" if "_source_file" in out.columns else "", flush=True)
    return final_path


def validate(model_size: str, mask_batch_size: int, device_arg: str, dtype_arg: str) -> None:
    seqs = [
        "MENDKGQLVELYVPRKCSATNRIIKAKDHASVQISIAKVDEDGRAIAGENITYALSGYVRGRGEADDSLNRLAQQDGLLKNVWSYSR",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "GASGDLGKKVTAQELDQLLREVDAGLVEKALAEGATEVLGDLAK",
    ]
    mean_col, _nll_col, ppl_col, _pos_col, _legacy_col = score_columns(model_size)
    for seq in seqs:
        t0 = time.time()
        result = score_with_retries(
            seq,
            model_size=model_size,
            mask_batch_size=mask_batch_size,
            device_arg=device_arg,
            dtype_arg=dtype_arg,
        )
        print(
            f"len={len(seq):4d} mean_logp={result[mean_col]: .4f} "
            f"ppl={result[ppl_col]: .2f} time={time.time() - t0:.1f}s",
            flush=True,
        )


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="dataset_update/main_plus_r2_r3_scored.csv")
    ap.add_argument("--output-dir", default="dataset_update/scoring_results/esm2_15B_main_plus_r2_r3")
    ap.add_argument("--model-size", choices=sorted(MODEL_MAP), default="15B")
    ap.add_argument("--rows-per-chunk", type=int, default=10)
    ap.add_argument("--mask-batch-size", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--score-truncated", action="store_true",
                    help="Score the first --max-len residues of sequences longer than --max-len")
    ap.add_argument("--only-missing-score-column",
                    help="Only score rows where this existing score column is NaN")
    ap.add_argument("--where-column-is-true",
                    help="Only score rows where this boolean/true-like column is true")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int)
    ap.add_argument("--partial-save-every", type=int, default=5)
    ap.add_argument("--device", default="auto", help="auto, cuda, cpu, mps, cuda:0, etc.")
    ap.add_argument("--dtype", default="auto", help="auto, fp32, fp16, or bf16")
    ap.add_argument("--validate", action="store_true", help="Score three short sequences and exit")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args(argv)

    if args.validate:
        validate(args.model_size, args.mask_batch_size, args.device, args.dtype)
        return

    if args.merge_only:
        merge_chunks(args.input, args.output_dir, args.model_size, max_len=args.max_len)
        return

    run_scoring(
        input_csv=args.input,
        output_dir=args.output_dir,
        model_size=args.model_size,
        rows_per_chunk=args.rows_per_chunk,
        mask_batch_size=args.mask_batch_size,
        max_len=args.max_len,
        score_truncated=args.score_truncated,
        only_missing_score_column=args.only_missing_score_column,
        where_column_is_true=args.where_column_is_true,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        limit=args.limit,
        offset=args.offset,
        partial_save_every=args.partial_save_every,
        device_arg=args.device,
        dtype_arg=args.dtype,
        do_merge=not args.no_merge,
    )


if __name__ == "__main__":
    main()
