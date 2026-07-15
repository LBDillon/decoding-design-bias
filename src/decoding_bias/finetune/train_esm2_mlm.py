#!/usr/bin/env python3
"""Continue ESM2-35M masked-language-model training on secretome cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--seq_col", default="sequence")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--max_length", type=int, default=1022, help="Maximum residue length before special tokens.")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--warmup_steps", type=int, default=-1,
                        help="If >=0, use fixed warmup steps instead of warmup_ratio "
                             "(set 0 for the aggressive no-warmup schedule).")
    parser.add_argument("--lr_scheduler_type", default="linear",
                        help="HF LR schedule; 'constant' = the aggressive AlkSecMPNN recipe.")
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", choices=["auto", "no", "fp16", "bf16"], default="auto")
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument(
        "--freeze_encoder_layers",
        type=int,
        default=0,
        help="Freeze this many leading encoder layers. Default 0 trains all parameters.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default="auto",
        help="'auto', 'none', or a checkpoint path.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Use 8 train/4 val sequences and at most 2 optimizer steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "numpy", "pandas"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "version_unknown")
        except Exception as exc:  # pragma: no cover - run-time manifest detail
            versions[name] = f"NOT_AVAILABLE: {type(exc).__name__}: {exc}"
    return versions


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def training_arguments_kwargs(args: argparse.Namespace, use_bf16: bool, use_fp16: bool) -> Dict[str, object]:
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    strategy = "steps" if args.dry_run else "epoch"

    kwargs: Dict[str, object] = {
        "output_dir": args.out_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "num_train_epochs": 1 if args.dry_run else args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        strategy_key: strategy,
        "save_strategy": strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": args.seed,
        "data_seed": args.seed,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "remove_unused_columns": False,
    }
    # Warmup: fixed steps if requested (0 = the aggressive no-warmup recipe),
    # else the ratio. Prefer warmup_steps to dodge the v5 warmup_ratio deprecation.
    if args.warmup_steps >= 0:
        kwargs["warmup_steps"] = args.warmup_steps
    else:
        kwargs["warmup_ratio"] = args.warmup_ratio
    if args.dry_run:
        kwargs.update({"max_steps": 2, "eval_steps": 1, "save_steps": 1, "logging_steps": 1})
    # Drop kwargs the installed transformers no longer accepts (e.g. newer
    # versions removed `overwrite_output_dir`), so the recipe stays portable
    # across HF releases. Warn rather than fail silently.
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
    if not accepts_var_kw:
        dropped = [k for k in kwargs if k not in sig]
        for k in dropped:
            kwargs.pop(k)
        if dropped:
            print(f"[train] transformers {__import__('transformers').__version__} "
                  f"does not accept {dropped}; dropped.", flush=True)
    return kwargs


def choose_precision(mode: str) -> tuple[bool, bool]:
    import torch

    if mode == "no" or not torch.cuda.is_available():
        return False, False
    if mode == "fp16":
        return False, True
    if mode == "bf16":
        return True, False
    if torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def apply_freezing(model, freeze_embeddings: bool, freeze_encoder_layers: int) -> List[str]:
    frozen: List[str] = []
    if freeze_embeddings:
        for name, param in model.named_parameters():
            if "embed" in name.lower():
                param.requires_grad = False
                frozen.append(name)
    if freeze_encoder_layers > 0:
        for name, param in model.named_parameters():
            marker = ".layers."
            if marker in name:
                try:
                    layer_index = int(name.split(marker, 1)[1].split(".", 1)[0])
                except ValueError:
                    continue
                if layer_index < freeze_encoder_layers:
                    param.requires_grad = False
                    frozen.append(name)
    return frozen


def resolve_resume_checkpoint(out_dir: Path, resume_arg: str) -> Optional[str]:
    if resume_arg == "none":
        return None
    if resume_arg != "auto":
        return resume_arg
    try:
        from transformers.trainer_utils import get_last_checkpoint

        if out_dir.exists():
            return get_last_checkpoint(str(out_dir))
    except Exception:
        return None
    return None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    train_rows = read_csv_rows(Path(args.train_csv))
    val_rows = read_csv_rows(Path(args.val_csv))
    if args.dry_run:
        train_rows = train_rows[:8]
        val_rows = val_rows[:4]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    frozen_parameters = apply_freezing(model, args.freeze_embeddings, args.freeze_encoder_layers)

    num_special = tokenizer.num_special_tokens_to_add(pair=False)
    token_max_length = args.max_length + num_special

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch[args.seq_col],
            truncation=True,
            max_length=token_max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)
    remove_train_columns = list(train_dataset.column_names)
    remove_val_columns = list(val_dataset.column_names)
    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=remove_train_columns)
    val_dataset = val_dataset.map(tokenize_batch, batched=True, remove_columns=remove_val_columns)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    use_bf16, use_fp16 = choose_precision(args.mixed_precision)
    train_args = TrainingArguments(**training_arguments_kwargs(args, use_bf16, use_fp16))
    # transformers v5 renamed Trainer's `tokenizer` kwarg to `processing_class`.
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    tok_key = "processing_class" if "processing_class" in trainer_sig else "tokenizer"
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        **{tok_key: tokenizer},
    )

    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_params = int(sum(p.numel() for p in model.parameters()))
    manifest = {
        "base_model_name": args.model_name,
        "training_mode": "continued_masked_language_model_pretraining",
        "no_supervised_head_added": True,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "train_csv_sha256": sha256_file(Path(args.train_csv)),
        "val_csv_sha256": sha256_file(Path(args.val_csv)),
        "train_sequences": len(train_rows),
        "val_sequences": len(val_rows),
        "epochs": 1 if args.dry_run else args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "mlm_probability": args.mlm_probability,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "max_residue_length": args.max_length,
        "token_max_length": token_max_length,
        "dry_run": args.dry_run,
        "mixed_precision": {"bf16": use_bf16, "fp16": use_fp16},
        "frozen_parameter_count": len(frozen_parameters),
        "frozen_parameter_names": frozen_parameters[:50],
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "package_versions": package_versions(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }

    resume_checkpoint = resolve_resume_checkpoint(out_dir, args.resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    trainer.save_state()

    manifest.update(
        {
            "resume_from_checkpoint": resume_checkpoint,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
            "final_eval_loss": eval_metrics.get("eval_loss"),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
        }
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
