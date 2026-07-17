"""decoding_bias.finetune.esm2_train

Merged provenance module. Sections (see ARCHIVE_MAP.md):
  - prepare_esm_secretome_data 
  - check_esm_environment 
  - train_esm2_mlm 
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

# ---------- from prepare_esm_secretome_data.py ----------
CANONICAL_AA = set('ACDEFGHIKLMNPQRSTVWY')
SOURCES = {'alkaline': {'prefix': 'alkaliphile', 'case_out': 'alkaline_case', 'control_out': 'alkaline_neu', 'case_group': 'alkaliphile'}, 'acid': {'prefix': 'acidophile', 'case_out': 'acid_case', 'control_out': 'acid_neu', 'case_group': 'acidophile'}}
EXPECTED_COUNTS = {('alkaline_case', 'train'): 252, ('alkaline_case', 'val'): 53, ('alkaline_case', 'test'): 50, ('alkaline_neu', 'train'): 252, ('alkaline_neu', 'val'): 53, ('alkaline_neu', 'test'): 50, ('acid_case', 'train'): 74, ('acid_case', 'val'): 20, ('acid_case', 'test'): 13, ('acid_neu', 'train'): 74, ('acid_neu', 'val'): 20, ('acid_neu', 'test'): 13}
OUT_FIELDS = ['id', 'sequence', 'cohort', 'role', 'split', 'pair_case', 'cluster_id', 'source_jsonl']
def prepare_esm_secretome_data_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', default='finetune/data', help='Directory with parsed ProteinMPNN JSONLs.')
    parser.add_argument('--out_dir', default='outputs/esm35m_continual_pretraining/data', help='Directory for sequence-only CSV exports.')
    parser.add_argument('--max_len', type=int, default=1022, help='Maximum residue length retained for ESM2.')
    parser.add_argument('--skip_expected_count_check', action='store_true', help='Do not assert the confirmed reviewer-experiment counts.')
    return parser.parse_args()
def clean_sequence(record: dict) -> str:
    return (record.get('seq') or record.get('seq_chain_A') or '').strip().upper()
def exclusion_reasons(seq: str, max_len: int) -> List[str]:
    reasons: List[str] = []
    if not seq:
        reasons.append('empty_sequence')
    noncanonical = sorted(set(seq) - CANONICAL_AA)
    if noncanonical:
        reasons.append('noncanonical:' + ''.join(noncanonical))
    if len(seq) > max_len:
        reasons.append(f'length_gt_{max_len}')
    return reasons
def iter_records(input_dir: Path, prefix: str, split: str) -> Iterable[Tuple[Path, dict]]:
    path = input_dir / f'{prefix}_parsed_{split}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f'Missing expected parsed JSONL: {path}')
    with path.open() as handle:
        for (line_no, line) in enumerate(handle, start=1):
            try:
                yield (path, json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Could not parse {path}:{line_no}: {exc}') from exc
def validate_no_cross_split_duplicates(rows_by_dataset: Dict[str, Dict[str, List[dict]]]) -> None:
    for (dataset, split_rows) in rows_by_dataset.items():
        sequence_splits: Dict[str, set] = defaultdict(set)
        for (split, rows) in split_rows.items():
            for row in rows:
                sequence_splits[row['sequence']].add(split)
        duplicates = {seq: splits for (seq, splits) in sequence_splits.items() if len(splits) > 1}
        if duplicates:
            examples = [(len(seq), sorted(splits)) for (seq, splits) in list(duplicates.items())[:5]]
            raise AssertionError(f'{dataset} has identical sequences across train/val/test: {examples}')
def validate_pair_split_consistency(rows_by_dataset: Dict[str, Dict[str, List[dict]]]) -> None:
    for experiment in ('alkaline', 'acid'):
        case_key = f'{experiment}_case'
        control_key = f'{experiment}_neu'
        case_split = {row['id']: split for (split, rows) in rows_by_dataset[case_key].items() for row in rows}
        mismatches = []
        missing = []
        for (split, rows) in rows_by_dataset[control_key].items():
            for row in rows:
                pair_case = row.get('pair_case', '')
                if not pair_case:
                    continue
                expected_split = case_split.get(pair_case)
                if expected_split is None:
                    missing.append((row['id'], pair_case, split))
                elif expected_split != split:
                    mismatches.append((row['id'], pair_case, split, expected_split))
        if missing or mismatches:
            raise AssertionError(f'{experiment} pair split consistency failed; missing={missing[:5]}, mismatches={mismatches[:5]}')
def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
def write_counts(path: Path, rows_by_dataset: Dict[str, Dict[str, List[dict]]]) -> List[dict]:
    count_rows: List[dict] = []
    for dataset in ('alkaline_case', 'alkaline_neu', 'acid_case', 'acid_neu'):
        for split in ('train', 'val', 'test'):
            rows = rows_by_dataset[dataset][split]
            lengths = [len(row['sequence']) for row in rows]
            count_rows.append({'cohort': dataset, 'split': split, 'n_sequences': len(rows), 'median_length': statistics.median(lengths) if lengths else '', 'max_length': max(lengths) if lengths else ''})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['cohort', 'split', 'n_sequences', 'median_length', 'max_length'])
        writer.writeheader()
        writer.writerows(count_rows)
    return count_rows
def write_exclusions(path: Path, exclusions: List[dict]) -> None:
    fields = ['id', 'cohort', 'role', 'split', 'sequence_length', 'reasons', 'source_jsonl']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(exclusions)
def prepare_esm_secretome_data_main() -> None:
    args = prepare_esm_secretome_data_parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_dataset: Dict[str, Dict[str, List[dict]]] = {dataset: {split: [] for split in ('train', 'val', 'test')} for dataset in ('alkaline_case', 'alkaline_neu', 'acid_case', 'acid_neu')}
    exclusions: List[dict] = []
    for (experiment, spec) in SOURCES.items():
        for split in ('train', 'val', 'test'):
            for (source_path, record) in iter_records(input_dir, spec['prefix'], split):
                role = record.get('role')
                if role == 'case':
                    dataset = spec['case_out']
                elif role == 'control':
                    dataset = spec['control_out']
                else:
                    raise ValueError(f'Unexpected role in {source_path}: {role!r}')
                seq = clean_sequence(record)
                reasons = exclusion_reasons(seq, args.max_len)
                if reasons:
                    exclusions.append({'id': record.get('name', ''), 'cohort': record.get('group', ''), 'role': role, 'split': record.get('split', split), 'sequence_length': len(seq), 'reasons': ';'.join(reasons), 'source_jsonl': str(source_path)})
                    continue
                record_split = record.get('split')
                if record_split != split:
                    raise AssertionError(f"Filename split and record split disagree for {record.get('name')}: {split} vs {record_split}")
                rows_by_dataset[dataset][split].append({'id': record.get('name', ''), 'sequence': seq, 'cohort': record.get('group', ''), 'role': role, 'split': split, 'pair_case': record.get('pair_case', ''), 'cluster_id': record.get('cluster_id', ''), 'source_jsonl': str(source_path)})
    validate_no_cross_split_duplicates(rows_by_dataset)
    validate_pair_split_consistency(rows_by_dataset)
    for (dataset, split_rows) in rows_by_dataset.items():
        for (split, rows) in split_rows.items():
            out_path = out_dir / f'{dataset}_{split}.csv'
            write_csv(out_path, rows)
    heldout_rows: List[dict] = []
    all_rows: List[dict] = []
    for dataset in ('alkaline_case', 'alkaline_neu', 'acid_case', 'acid_neu'):
        for split in ('train', 'val', 'test'):
            rows = rows_by_dataset[dataset][split]
            all_rows.extend(rows)
            if split == 'test':
                heldout_rows.extend(rows)
    write_csv(out_dir / 'heldout_secretome_test.csv', heldout_rows)
    write_csv(out_dir / 'all_secretome_sequences.csv', all_rows)
    write_exclusions(out_dir / 'exclusion_log.csv', exclusions)
    count_rows = write_counts(out_dir / 'dataset_counts.csv', rows_by_dataset)
    if not args.skip_expected_count_check:
        for row in count_rows:
            expected = EXPECTED_COUNTS[row['cohort'], row['split']]
            if int(row['n_sequences']) != expected:
                raise AssertionError(f"Unexpected count for {row['cohort']} {row['split']}: {row['n_sequences']} != {expected}")
    print('cohort | split | n_sequences | median_length | max_length')
    for row in count_rows:
        print(f"{row['cohort']} | {row['split']} | {row['n_sequences']} | {row['median_length']} | {row['max_length']}")
    print(f'Wrote CSVs to {out_dir}')
    print(f'Exclusions: {len(exclusions)}')
def prepare_esm_secretome_data__entry():
    prepare_esm_secretome_data_main()

# ---------- from check_esm_environment.py ----------
def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception as exc:
        return f'NOT_AVAILABLE: {type(exc).__name__}: {exc}'
    return getattr(module, '__version__', 'version_unknown')
def check_esm_environment_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name', default='facebook/esm2_t12_35M_UR50D')
    parser.add_argument('--out_txt', default='outputs/esm35m_continual_pretraining/logs/environment.txt')
    parser.add_argument('--require_cuda', action='store_true', help='Exit non-zero if CUDA is not available.')
    return parser.parse_args()
def check_esm_environment_main() -> None:
    args = check_esm_environment_parse_args()
    report = {'python': sys.version.replace('\n', ' '), 'platform': platform.platform(), 'model_name': args.model_name, 'packages': {'torch': package_version('torch'), 'transformers': package_version('transformers'), 'datasets': package_version('datasets'), 'accelerate': package_version('accelerate'), 'numpy': package_version('numpy'), 'pandas': package_version('pandas')}}
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments
    report['cuda_available'] = bool(torch.cuda.is_available())
    report['cuda_device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    report['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
    report['cuda_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else ''
    report['mps_available'] = bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    params = inspect.signature(TrainingArguments.__init__).parameters
    report['training_arguments_has_eval_strategy'] = 'eval_strategy' in params
    report['training_arguments_has_evaluation_strategy'] = 'evaluation_strategy' in params
    if args.require_cuda and (not torch.cuda.is_available()):
        raise SystemExit('CUDA is required for this ARC check but is not available.')
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    report['tokenizer_class'] = tokenizer.__class__.__name__
    report['model_class'] = model.__class__.__name__
    report['mask_token'] = tokenizer.mask_token
    report['mask_token_id'] = tokenizer.mask_token_id
    report['model_max_length'] = tokenizer.model_max_length
    report['num_parameters'] = int(sum((p.numel() for p in model.parameters())))
    if tokenizer.mask_token_id is None:
        raise RuntimeError('Tokenizer has no mask token.')
    sequence = 'MKTAYIAKQRQISFVKSHFSRQ'
    encoded = tokenizer(sequence, return_tensors='pt', return_special_tokens_mask=True)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    special_mask = encoded['special_tokens_mask'].to(device)
    residue_positions = (special_mask[0] == 0).nonzero(as_tuple=False).flatten()
    if residue_positions.numel() != len(sequence):
        raise RuntimeError(f'Residue-token count mismatch: {residue_positions.numel()} tokens for {len(sequence)} residues.')
    labels = torch.full_like(input_ids, -100)
    mask_position = residue_positions[len(sequence) // 2]
    labels[0, mask_position] = input_ids[0, mask_position]
    input_ids[0, mask_position] = tokenizer.mask_token_id
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    report['forward_logits_shape'] = list(outputs.logits.shape)
    report['tiny_mlm_loss'] = float(outputs.loss.detach().cpu())
    report['elapsed_seconds'] = round(time.time() - t0, 2)
    report['status'] = 'ok'
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    out_path.write_text(text + '\n')
    print(text)
def check_esm_environment__entry():
    check_esm_environment_main()

# ---------- from train_esm2_mlm.py ----------
DEFAULT_MODEL = 'facebook/esm2_t12_35M_UR50D'
def train_esm2_mlm_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model_name', default=DEFAULT_MODEL)
    parser.add_argument('--seq_col', default='sequence')
    parser.add_argument('--id_col', default='id')
    parser.add_argument('--max_length', type=int, default=1022, help='Maximum residue length before special tokens.')
    parser.add_argument('--epochs', type=float, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--warmup_steps', type=int, default=-1, help='If >=0, use fixed warmup steps instead of warmup_ratio (set 0 for the aggressive no-warmup schedule).')
    parser.add_argument('--lr_scheduler_type', default='linear', help="HF LR schedule; 'constant' = the aggressive AlkSecMPNN recipe.")
    parser.add_argument('--mlm_probability', type=float, default=0.15)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--mixed_precision', choices=['auto', 'no', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--freeze_embeddings', action='store_true')
    parser.add_argument('--freeze_encoder_layers', type=int, default=0, help='Freeze this many leading encoder layers. Default 0 trains all parameters.')
    parser.add_argument('--resume_from_checkpoint', default='auto', help="'auto', 'none', or a checkpoint path.")
    parser.add_argument('--dry_run', action='store_true', help='Use 8 train/4 val sequences and at most 2 optimizer steps.')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    return parser.parse_args()
def read_csv_rows(path: Path) -> List[dict]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in ('torch', 'transformers', 'datasets', 'accelerate', 'numpy', 'pandas'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', 'version_unknown')
        except Exception as exc:
            versions[name] = f'NOT_AVAILABLE: {type(exc).__name__}: {exc}'
    return versions
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    strategy_key = 'eval_strategy' if 'eval_strategy' in sig else 'evaluation_strategy'
    strategy = 'steps' if args.dry_run else 'epoch'
    kwargs: Dict[str, object] = {'output_dir': args.out_dir, 'overwrite_output_dir': args.overwrite_output_dir, 'num_train_epochs': 1 if args.dry_run else args.epochs, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'lr_scheduler_type': args.lr_scheduler_type, 'per_device_train_batch_size': args.per_device_train_batch_size, 'per_device_eval_batch_size': args.per_device_eval_batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, strategy_key: strategy, 'save_strategy': strategy, 'load_best_model_at_end': True, 'metric_for_best_model': 'eval_loss', 'greater_is_better': False, 'seed': args.seed, 'data_seed': args.seed, 'logging_steps': args.logging_steps, 'save_total_limit': args.save_total_limit, 'report_to': 'none', 'fp16': use_fp16, 'bf16': use_bf16, 'remove_unused_columns': False}
    if args.warmup_steps >= 0:
        kwargs['warmup_steps'] = args.warmup_steps
    else:
        kwargs['warmup_ratio'] = args.warmup_ratio
    if args.dry_run:
        kwargs.update({'max_steps': 2, 'eval_steps': 1, 'save_steps': 1, 'logging_steps': 1})
    accepts_var_kw = any((p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values()))
    if not accepts_var_kw:
        dropped = [k for k in kwargs if k not in sig]
        for k in dropped:
            kwargs.pop(k)
        if dropped:
            print(f"[train] transformers {__import__('transformers').__version__} does not accept {dropped}; dropped.", flush=True)
    return kwargs
def choose_precision(mode: str) -> tuple[bool, bool]:
    import torch
    if mode == 'no' or not torch.cuda.is_available():
        return (False, False)
    if mode == 'fp16':
        return (False, True)
    if mode == 'bf16':
        return (True, False)
    if torch.cuda.is_bf16_supported():
        return (True, False)
    return (False, True)
def apply_freezing(model, freeze_embeddings: bool, freeze_encoder_layers: int) -> List[str]:
    frozen: List[str] = []
    if freeze_embeddings:
        for (name, param) in model.named_parameters():
            if 'embed' in name.lower():
                param.requires_grad = False
                frozen.append(name)
    if freeze_encoder_layers > 0:
        for (name, param) in model.named_parameters():
            marker = '.layers.'
            if marker in name:
                try:
                    layer_index = int(name.split(marker, 1)[1].split('.', 1)[0])
                except ValueError:
                    continue
                if layer_index < freeze_encoder_layers:
                    param.requires_grad = False
                    frozen.append(name)
    return frozen
def resolve_resume_checkpoint(out_dir: Path, resume_arg: str) -> Optional[str]:
    if resume_arg == 'none':
        return None
    if resume_arg != 'auto':
        return resume_arg
    try:
        from transformers.trainer_utils import get_last_checkpoint
        if out_dir.exists():
            return get_last_checkpoint(str(out_dir))
    except Exception:
        return None
    return None
def train_esm2_mlm_main() -> None:
    args = train_esm2_mlm_parse_args()
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
        return tokenizer(batch[args.seq_col], truncation=True, max_length=token_max_length, padding=False, return_special_tokens_mask=True)
    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)
    remove_train_columns = list(train_dataset.column_names)
    remove_val_columns = list(val_dataset.column_names)
    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=remove_train_columns)
    val_dataset = val_dataset.map(tokenize_batch, batched=True, remove_columns=remove_val_columns)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    (use_bf16, use_fp16) = choose_precision(args.mixed_precision)
    train_args = TrainingArguments(**training_arguments_kwargs(args, use_bf16, use_fp16))
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    tok_key = 'processing_class' if 'processing_class' in trainer_sig else 'tokenizer'
    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator, **{tok_key: tokenizer})
    trainable_params = int(sum((p.numel() for p in model.parameters() if p.requires_grad)))
    total_params = int(sum((p.numel() for p in model.parameters())))
    manifest = {'base_model_name': args.model_name, 'training_mode': 'continued_masked_language_model_pretraining', 'no_supervised_head_added': True, 'train_csv': args.train_csv, 'val_csv': args.val_csv, 'train_csv_sha256': sha256_file(Path(args.train_csv)), 'val_csv_sha256': sha256_file(Path(args.val_csv)), 'train_sequences': len(train_rows), 'val_sequences': len(val_rows), 'epochs': 1 if args.dry_run else args.epochs, 'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay, 'warmup_ratio': args.warmup_ratio, 'warmup_steps': args.warmup_steps, 'lr_scheduler_type': args.lr_scheduler_type, 'mlm_probability': args.mlm_probability, 'per_device_train_batch_size': args.per_device_train_batch_size, 'per_device_eval_batch_size': args.per_device_eval_batch_size, 'gradient_accumulation_steps': args.gradient_accumulation_steps, 'seed': args.seed, 'max_residue_length': args.max_length, 'token_max_length': token_max_length, 'dry_run': args.dry_run, 'mixed_precision': {'bf16': use_bf16, 'fp16': use_fp16}, 'frozen_parameter_count': len(frozen_parameters), 'frozen_parameter_names': frozen_parameters[:50], 'trainable_parameters': trainable_params, 'total_parameters': total_params, 'package_versions': package_versions(), 'cuda_available': bool(torch.cuda.is_available()), 'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}
    resume_checkpoint = resolve_resume_checkpoint(out_dir, args.resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    trainer.save_state()
    manifest.update({'resume_from_checkpoint': resume_checkpoint, 'best_checkpoint': trainer.state.best_model_checkpoint, 'best_eval_loss': trainer.state.best_metric, 'final_eval_loss': eval_metrics.get('eval_loss'), 'train_metrics': train_result.metrics, 'eval_metrics': eval_metrics})
    (out_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))
def train_esm2_mlm__entry():
    train_esm2_mlm_main()

_STEPS = {
    'prepare-esm-secretome-data': prepare_esm_secretome_data__entry,
    'check-esm-environment': check_esm_environment__entry,
    'train-esm2-mlm': train_esm2_mlm__entry,
}

def main(argv=None):
    import sys
    argv = sys.argv if argv is None else argv
    if len(argv) < 2 or argv[1] not in _STEPS:
        print('steps:', ', '.join(_STEPS)); return 1
    sys.argv = [argv[0]] + argv[2:]
    _STEPS[argv[1]](); return 0

if __name__ == '__main__':
    raise SystemExit(main())

