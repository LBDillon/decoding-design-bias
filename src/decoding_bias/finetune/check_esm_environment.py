#!/usr/bin/env python3
"""Check whether an ARC environment can load and run ESM2-35M for MLM."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import sys
import time
from pathlib import Path


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as exc:  # pragma: no cover - used for environment diagnostics
        return f"NOT_AVAILABLE: {type(exc).__name__}: {exc}"
    return getattr(module, "__version__", "version_unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--out_txt", default="outputs/esm35m_continual_pretraining/logs/environment.txt")
    parser.add_argument("--require_cuda", action="store_true", help="Exit non-zero if CUDA is not available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "model_name": args.model_name,
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "accelerate": package_version("accelerate"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
    }

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments

    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    report["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    report["cuda_capability"] = (
        ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else ""
    )
    report["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    params = inspect.signature(TrainingArguments.__init__).parameters
    report["training_arguments_has_eval_strategy"] = "eval_strategy" in params
    report["training_arguments_has_evaluation_strategy"] = "evaluation_strategy" in params

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this ARC check but is not available.")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    report["tokenizer_class"] = tokenizer.__class__.__name__
    report["model_class"] = model.__class__.__name__
    report["mask_token"] = tokenizer.mask_token
    report["mask_token_id"] = tokenizer.mask_token_id
    report["model_max_length"] = tokenizer.model_max_length
    report["num_parameters"] = int(sum(p.numel() for p in model.parameters()))

    if tokenizer.mask_token_id is None:
        raise RuntimeError("Tokenizer has no mask token.")

    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    encoded = tokenizer(sequence, return_tensors="pt", return_special_tokens_mask=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_mask = encoded["special_tokens_mask"].to(device)
    residue_positions = (special_mask[0] == 0).nonzero(as_tuple=False).flatten()
    if residue_positions.numel() != len(sequence):
        raise RuntimeError(
            f"Residue-token count mismatch: {residue_positions.numel()} tokens for {len(sequence)} residues."
        )

    labels = torch.full_like(input_ids, -100)
    mask_position = residue_positions[len(sequence) // 2]
    labels[0, mask_position] = input_ids[0, mask_position]
    input_ids[0, mask_position] = tokenizer.mask_token_id
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    report["forward_logits_shape"] = list(outputs.logits.shape)
    report["tiny_mlm_loss"] = float(outputs.loss.detach().cpu())
    report["elapsed_seconds"] = round(time.time() - t0, 2)
    report["status"] = "ok"

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    out_path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
