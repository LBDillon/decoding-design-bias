#!/usr/bin/env python3
"""Export sequence-only ESM2 continued-pretraining CSVs.

The source records are the exact mixed case/control ProteinMPNN JSONLs used by
finetune/alkmpnn/train.py. This script filters those records by filename prefix
and role, removes sequences that ESM2 should not see, and writes split-specific
CSV files for continued masked-language-model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

SOURCES = {
    "alkaline": {
        "prefix": "alkaliphile",
        "case_out": "alkaline_case",
        "control_out": "alkaline_neu",
        "case_group": "alkaliphile",
    },
    "acid": {
        "prefix": "acidophile",
        "case_out": "acid_case",
        "control_out": "acid_neu",
        "case_group": "acidophile",
    },
}

EXPECTED_COUNTS = {
    ("alkaline_case", "train"): 252,
    ("alkaline_case", "val"): 53,
    ("alkaline_case", "test"): 50,
    ("alkaline_neu", "train"): 252,
    ("alkaline_neu", "val"): 53,
    ("alkaline_neu", "test"): 50,
    ("acid_case", "train"): 74,
    ("acid_case", "val"): 20,
    ("acid_case", "test"): 13,
    ("acid_neu", "train"): 74,
    ("acid_neu", "val"): 20,
    ("acid_neu", "test"): 13,
}

OUT_FIELDS = [
    "id",
    "sequence",
    "cohort",
    "role",
    "split",
    "pair_case",
    "cluster_id",
    "source_jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="finetune/data", help="Directory with parsed ProteinMPNN JSONLs.")
    parser.add_argument(
        "--out_dir",
        default="outputs/esm35m_continual_pretraining/data",
        help="Directory for sequence-only CSV exports.",
    )
    parser.add_argument("--max_len", type=int, default=1022, help="Maximum residue length retained for ESM2.")
    parser.add_argument(
        "--skip_expected_count_check",
        action="store_true",
        help="Do not assert the confirmed reviewer-experiment counts.",
    )
    return parser.parse_args()


def clean_sequence(record: dict) -> str:
    return (record.get("seq") or record.get("seq_chain_A") or "").strip().upper()


def exclusion_reasons(seq: str, max_len: int) -> List[str]:
    reasons: List[str] = []
    if not seq:
        reasons.append("empty_sequence")
    noncanonical = sorted(set(seq) - CANONICAL_AA)
    if noncanonical:
        reasons.append("noncanonical:" + "".join(noncanonical))
    if len(seq) > max_len:
        reasons.append(f"length_gt_{max_len}")
    return reasons


def iter_records(input_dir: Path, prefix: str, split: str) -> Iterable[Tuple[Path, dict]]:
    path = input_dir / f"{prefix}_parsed_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing expected parsed JSONL: {path}")
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                yield path, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path}:{line_no}: {exc}") from exc


def validate_no_cross_split_duplicates(rows_by_dataset: Dict[str, Dict[str, List[dict]]]) -> None:
    for dataset, split_rows in rows_by_dataset.items():
        sequence_splits: Dict[str, set] = defaultdict(set)
        for split, rows in split_rows.items():
            for row in rows:
                sequence_splits[row["sequence"]].add(split)
        duplicates = {seq: splits for seq, splits in sequence_splits.items() if len(splits) > 1}
        if duplicates:
            examples = [(len(seq), sorted(splits)) for seq, splits in list(duplicates.items())[:5]]
            raise AssertionError(f"{dataset} has identical sequences across train/val/test: {examples}")


def validate_pair_split_consistency(rows_by_dataset: Dict[str, Dict[str, List[dict]]]) -> None:
    for experiment in ("alkaline", "acid"):
        case_key = f"{experiment}_case"
        control_key = f"{experiment}_neu"
        case_split = {
            row["id"]: split
            for split, rows in rows_by_dataset[case_key].items()
            for row in rows
        }
        mismatches = []
        missing = []
        for split, rows in rows_by_dataset[control_key].items():
            for row in rows:
                pair_case = row.get("pair_case", "")
                if not pair_case:
                    continue
                expected_split = case_split.get(pair_case)
                if expected_split is None:
                    missing.append((row["id"], pair_case, split))
                elif expected_split != split:
                    mismatches.append((row["id"], pair_case, split, expected_split))
        if missing or mismatches:
            raise AssertionError(
                f"{experiment} pair split consistency failed; "
                f"missing={missing[:5]}, mismatches={mismatches[:5]}"
            )


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_counts(path: Path, rows_by_dataset: Dict[str, Dict[str, List[dict]]]) -> List[dict]:
    count_rows: List[dict] = []
    for dataset in ("alkaline_case", "alkaline_neu", "acid_case", "acid_neu"):
        for split in ("train", "val", "test"):
            rows = rows_by_dataset[dataset][split]
            lengths = [len(row["sequence"]) for row in rows]
            count_rows.append(
                {
                    "cohort": dataset,
                    "split": split,
                    "n_sequences": len(rows),
                    "median_length": statistics.median(lengths) if lengths else "",
                    "max_length": max(lengths) if lengths else "",
                }
            )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cohort", "split", "n_sequences", "median_length", "max_length"],
        )
        writer.writeheader()
        writer.writerows(count_rows)
    return count_rows


def write_exclusions(path: Path, exclusions: List[dict]) -> None:
    fields = ["id", "cohort", "role", "split", "sequence_length", "reasons", "source_jsonl"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exclusions)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_dataset: Dict[str, Dict[str, List[dict]]] = {
        dataset: {split: [] for split in ("train", "val", "test")}
        for dataset in ("alkaline_case", "alkaline_neu", "acid_case", "acid_neu")
    }
    exclusions: List[dict] = []

    for experiment, spec in SOURCES.items():
        for split in ("train", "val", "test"):
            for source_path, record in iter_records(input_dir, spec["prefix"], split):
                role = record.get("role")
                if role == "case":
                    dataset = spec["case_out"]
                elif role == "control":
                    dataset = spec["control_out"]
                else:
                    raise ValueError(f"Unexpected role in {source_path}: {role!r}")

                seq = clean_sequence(record)
                reasons = exclusion_reasons(seq, args.max_len)
                if reasons:
                    exclusions.append(
                        {
                            "id": record.get("name", ""),
                            "cohort": record.get("group", ""),
                            "role": role,
                            "split": record.get("split", split),
                            "sequence_length": len(seq),
                            "reasons": ";".join(reasons),
                            "source_jsonl": str(source_path),
                        }
                    )
                    continue

                record_split = record.get("split")
                if record_split != split:
                    raise AssertionError(
                        f"Filename split and record split disagree for {record.get('name')}: "
                        f"{split} vs {record_split}"
                    )

                rows_by_dataset[dataset][split].append(
                    {
                        "id": record.get("name", ""),
                        "sequence": seq,
                        "cohort": record.get("group", ""),
                        "role": role,
                        "split": split,
                        "pair_case": record.get("pair_case", ""),
                        "cluster_id": record.get("cluster_id", ""),
                        "source_jsonl": str(source_path),
                    }
                )

    validate_no_cross_split_duplicates(rows_by_dataset)
    validate_pair_split_consistency(rows_by_dataset)

    for dataset, split_rows in rows_by_dataset.items():
        for split, rows in split_rows.items():
            out_path = out_dir / f"{dataset}_{split}.csv"
            write_csv(out_path, rows)

    heldout_rows: List[dict] = []
    all_rows: List[dict] = []
    for dataset in ("alkaline_case", "alkaline_neu", "acid_case", "acid_neu"):
        for split in ("train", "val", "test"):
            rows = rows_by_dataset[dataset][split]
            all_rows.extend(rows)
            if split == "test":
                heldout_rows.extend(rows)
    write_csv(out_dir / "heldout_secretome_test.csv", heldout_rows)
    write_csv(out_dir / "all_secretome_sequences.csv", all_rows)
    write_exclusions(out_dir / "exclusion_log.csv", exclusions)
    count_rows = write_counts(out_dir / "dataset_counts.csv", rows_by_dataset)

    if not args.skip_expected_count_check:
        for row in count_rows:
            expected = EXPECTED_COUNTS[(row["cohort"], row["split"])]
            if int(row["n_sequences"]) != expected:
                raise AssertionError(f"Unexpected count for {row['cohort']} {row['split']}: {row['n_sequences']} != {expected}")

    print("cohort | split | n_sequences | median_length | max_length")
    for row in count_rows:
        print(
            f"{row['cohort']} | {row['split']} | {row['n_sequences']} | "
            f"{row['median_length']} | {row['max_length']}"
        )
    print(f"Wrote CSVs to {out_dir}")
    print(f"Exclusions: {len(exclusions)}")


if __name__ == "__main__":
    main()
