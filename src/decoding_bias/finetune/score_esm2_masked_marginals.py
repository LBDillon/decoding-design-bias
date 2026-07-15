#!/usr/bin/env python3
"""Score sequences with ESM2 masked-marginal pseudo-log-likelihood."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL, help="HF model name or local continued-pretraining run.")
    parser.add_argument("--input_csv", help="CSV with sequences to score.")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--seq_col", default="sequence")
    parser.add_argument("--max_len", type=int, default=1022)
    parser.add_argument("--batch_masked_positions", type=int, default=64)
    parser.add_argument("--model_name", default="", help="Name to store in output; defaults to model_dir basename.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--limit", type=int, default=0, help="Score at most this many rows.")
    parser.add_argument("--self_test", action="store_true", help="Score three short built-in sequences.")
    return parser.parse_args()


def choose_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_input_rows(args: argparse.Namespace) -> List[dict]:
    if args.self_test:
        return [
            {args.id_col: "selftest_1", args.seq_col: "MKTAYIAKQRQISFVKSHFSRQ"},
            {args.id_col: "selftest_2", args.seq_col: "GASPVTCILNDQKEMHFRYW"},
            {args.id_col: "selftest_3", args.seq_col: "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAEL"},
        ]
    if not args.input_csv:
        raise SystemExit("--input_csv is required unless --self_test is set.")
    with Path(args.input_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    return rows


def validate_sequence(seq: str, max_len: int) -> Optional[str]:
    if not seq:
        return "empty_sequence"
    bad = sorted(set(seq) - CANONICAL_AA)
    if bad:
        return "noncanonical:" + "".join(bad)
    if len(seq) > max_len:
        return f"length_gt_{max_len}"
    return None


def masked_marginal_score(sequence: str, tokenizer, model, device, batch_masked_positions: int) -> float:
    import torch

    encoded = tokenizer(sequence, return_tensors="pt", return_special_tokens_mask=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_mask = encoded["special_tokens_mask"].to(device)
    residue_positions = (special_mask[0] == 0).nonzero(as_tuple=False).flatten()
    if residue_positions.numel() != len(sequence):
        raise ValueError(
            f"Residue-token mismatch: {residue_positions.numel()} non-special tokens for {len(sequence)} residues."
        )
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token.")

    true_token_ids = input_ids[0, residue_positions]
    log_probs: List[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequence), batch_masked_positions):
            end = min(start + batch_masked_positions, len(sequence))
            chunk_positions = residue_positions[start:end]
            batch = input_ids.repeat(chunk_positions.numel(), 1)
            batch_attention = attention_mask.repeat(chunk_positions.numel(), 1)
            row_idx = torch.arange(chunk_positions.numel(), device=device)
            batch[row_idx, chunk_positions] = tokenizer.mask_token_id
            outputs = model(input_ids=batch, attention_mask=batch_attention)
            logits = outputs.logits[row_idx, chunk_positions, :]
            chunk_log_probs = torch.log_softmax(logits, dim=-1)
            true_ids = true_token_ids[start:end]
            gathered = chunk_log_probs.gather(1, true_ids.unsqueeze(1)).squeeze(1)
            log_probs.extend(gathered.detach().cpu().tolist())
    if len(log_probs) != len(sequence):
        raise RuntimeError(f"Expected {len(sequence)} log probabilities, got {len(log_probs)}.")
    score = float(sum(log_probs) / len(log_probs))
    if not math.isfinite(score):
        raise RuntimeError("Non-finite masked-marginal score.")
    return score


def main() -> None:
    args = parse_args()
    rows = read_input_rows(args)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir).to(device)
    model_name = args.model_name or (args.model_dir if "/" in args.model_dir else Path(args.model_dir).name)

    output_rows = []
    for idx, row in enumerate(rows, start=1):
        seq = (row.get(args.seq_col) or "").strip().upper()
        seq_id = row.get(args.id_col) or f"row_{idx}"
        status = validate_sequence(seq, args.max_len)
        score = ""
        if status is None:
            try:
                score = masked_marginal_score(seq, tokenizer, model, device, args.batch_masked_positions)
                status = "ok"
            except Exception as exc:  # keep long ARC array jobs moving
                status = f"error:{type(exc).__name__}:{exc}"
                score = ""
        output_rows.append(
            {
                "id": seq_id,
                "sequence_length": len(seq),
                "esm_mlm_score": score,
                "status": status,
                "model_name": model_name,
            }
        )
        print(f"[{idx}/{len(rows)}] {seq_id} length={len(seq)} status={status}", flush=True)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "sequence_length", "esm_mlm_score", "status", "model_name"],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    ok_scores = [float(row["esm_mlm_score"]) for row in output_rows if row["status"] == "ok"]
    if ok_scores:
        print(
            f"Wrote {out_path}; ok={len(ok_scores)} "
            f"min={min(ok_scores):.4f} median={sorted(ok_scores)[len(ok_scores)//2]:.4f} max={max(ok_scores):.4f}"
        )
    else:
        print(f"Wrote {out_path}; no successful scores.")


if __name__ == "__main__":
    main()
