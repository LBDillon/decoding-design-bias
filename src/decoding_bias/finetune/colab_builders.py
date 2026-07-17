"""decoding_bias.finetune.colab_builders

Merged provenance module. Sections (see ARCHIVE_MAP.md):
  - build_colab 
  - build_generation_colab 
"""

import json
import pandas as pd
import zipfile
from pathlib import Path

# ---------- from build_colab.py ----------
build_colab_HERE = Path(__file__).resolve().parent
build_colab_ROOT = build_colab_HERE.parents[2]
build_colab_PIPE = build_colab_HERE.parent
build_colab_DATA = build_colab_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'data'
V12_CLI = build_colab_ROOT / 'dataset_update' / 'main_plus_r2_r3_analysis_v12_cli.csv'
METADATA = build_colab_ROOT / 'dataset_update' / 'main_plus_r2_r3_metadata_v12.csv'
ARMS = [('AlkSecESM35M', 'alkaline_case_train.csv', 'alkaline_case_val.csv'), ('AcidSecESM35M', 'acid_case_train.csv', 'acid_case_val.csv'), ('NeuSecESM35M_AlkMatched', 'alkaline_neu_train.csv', 'alkaline_neu_val.csv'), ('NeuSecESM35M_AcidMatched', 'acid_neu_train.csv', 'acid_neu_val.csv')]
TRAIN_FILES = sorted({f for (_, tr, va) in ARMS for f in (tr, va)})
def build_score_input() -> Path:
    """[Entry, sequence, isoelectric_point] for every v12_cli protein."""
    cli = pd.read_csv(V12_CLI, low_memory=False)[['Entry', 'isoelectric_point']]
    meta = pd.read_csv(METADATA, low_memory=False)[['Entry', 'sequence']].dropna(subset=['sequence'])
    df = cli.merge(meta, on='Entry', how='left')
    missing = int(df['sequence'].isna().sum())
    df = df.dropna(subset=['sequence']).reset_index(drop=True)
    out = build_colab_HERE / 'esm_score_input.csv'
    df.to_csv(out, index=False)
    print(f'esm_score_input.csv: {len(df)} proteins with sequence ({missing} v12_cli entries had no sequence and were dropped)')
    return out
def build_colab_md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}
def build_colab_code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n').splitlines(keepends=True)}
def build_colab_build_notebook() -> Path:
    arms_py = 'ARMS = [\n' + ''.join((f'    ("{n}", "data/{tr}", "data/{va}"),\n' for (n, tr, va) in ARMS)) + ']\n'
    cells = [build_colab_md('# ESM2-35M secretome malleability (Reviewer R1.4)\n\nSequence-model analogue of the AlkSecMPNN perturbation: continue masked-LM training of **ESM2-35M** on the *same* secreted-extremophile cohort and splits, then score WT sequences with masked-marginal pseudo-log-likelihood **before vs after** fine-tuning. The pH-feature **cosine test** (score-preference direction vs the acid--base axis) is run afterwards, locally, on the same axis as the structure models.\n\n**You upload one file:** `esm35m_colab_inputs.zip`.\n\n**Runtime:** set Runtime → Change runtime type → **T4 GPU**.\n\nThis notebook only orchestrates the committed CLI scripts (`train_esm2_mlm.py`, `score_esm2_masked_marginals.py`) - no logic lives here.'), build_colab_md('## 1. GPU + dependencies'), build_colab_code('!nvidia-smi -L\n# torch ships with Colab; add the HF stack the scripts need.\n!pip -q install "transformers>=4.45" "datasets>=2.20" "accelerate>=0.33" safetensors scikit-learn scipy 2>/dev/null\nimport transformers, torch\nprint("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())'), build_colab_md('## 2. Upload & unzip the input bundle\nChoose `esm35m_colab_inputs.zip` when prompted.'), build_colab_code("import os, io, zipfile\nfrom google.colab import files\nup = files.upload()\nname = [n for n in up if n.endswith('.zip')][0]\nwith zipfile.ZipFile(io.BytesIO(up[name])) as z:\n    z.extractall('/content/esm_ft')\nos.chdir('/content/esm_ft')\nos.makedirs('scores', exist_ok=True); os.makedirs('runs', exist_ok=True)\nprint('working dir:', os.getcwd())\nprint('contents:', sorted(os.listdir()))\nprint('data:', sorted(os.listdir('data')))"), build_colab_md('## 3. Config\n`SCORE_N` controls the shared scoring set. `0` = all ~10k proteins (faithful to the structure-model set, but ~6-8 h across 5 models). A pI-stratified subset (default 3000) spans the acid--base axis and finishes in ~1.5-2 h with tight bootstrap CIs. The SAME sampled proteins are scored by every model, so the before/after comparison is paired.'), build_colab_code('BASE_MODEL = "facebook/esm2_t12_35M_UR50D"\nEPOCHS = 10\nLR = 5e-5\nSCORE_N = 3000   # 0 = score all proteins\nSEED = 0\n\n' + arms_py), build_colab_md('## 4. Build the shared scoring set (same proteins for every model)'), build_colab_code('import pandas as pd, numpy as np\ninp = pd.read_csv(\'esm_score_input.csv\')\nif SCORE_N and SCORE_N < len(inp):\n    inp[\'_b\'] = pd.qcut(inp[\'isoelectric_point\'], 10, labels=False, duplicates=\'drop\')\n    per = max(1, SCORE_N // (inp[\'_b\'].nunique()))\n    inp = (inp.groupby(\'_b\', group_keys=False)\n              .apply(lambda d: d.sample(min(len(d), per), random_state=SEED))\n              .drop(columns=\'_b\').reset_index(drop=True))\ninp[[\'Entry\', \'sequence\']].to_csv(\'score_set.csv\', index=False)\nprint(\'scoring set:\', len(inp), \'proteins  (pI range \'\n      f"{inp.isoelectric_point.min():.1f}-{inp.isoelectric_point.max():.1f})")'), build_colab_md("## 5. Score the BASE model (the 'before')"), build_colab_code('!python scripts/score_esm2_masked_marginals.py \\\n  --model_dir {BASE_MODEL} --input_csv score_set.csv \\\n  --id_col Entry --seq_col sequence \\\n  --out_csv scores/BaseESM35M_masked_marginals.csv --model_name BaseESM35M'), build_colab_md('## 6. Continue-pretrain the 4 arms\nAlkaliphile / acidophile **cases** = the steered models; the matched **neutralophile controls** are the symmetric specificity check.'), build_colab_code("for name, tr, va in ARMS:\n    print(f'\\n===== training {name} =====')\n    !python scripts/train_esm2_mlm.py \\\n      --train_csv {tr} --val_csv {va} --out_dir runs/{name} \\\n      --epochs {EPOCHS} --learning_rate {LR} --overwrite_output_dir"), build_colab_md("## 7. Score each fine-tuned model (the 'after') on the same set"), build_colab_code("for name, tr, va in ARMS:\n    print(f'\\n===== scoring {name} =====')\n    !python scripts/score_esm2_masked_marginals.py \\\n      --model_dir runs/{name} --input_csv score_set.csv \\\n      --id_col Entry --seq_col sequence \\\n      --out_csv scores/{name}_masked_marginals.csv --model_name {name}"), build_colab_md('## 8. Package & download\nDownload `esm35m_scores.zip`, unzip it into `outputs/esm35m_continual_pretraining/scores/` in the repo, then run `python paper_code/08_pca_figures/charge_pca_cosine_shift.py` - the ESM rows will appear beside the ProteinMPNN rows on the same acid--base axis, each with a bootstrap 95% CI.'), build_colab_code("import shutil, glob\nfor f in glob.glob('runs/*/run_manifest.json'):\n    shutil.copy(f, 'scores/' + f.split('/')[1] + '_run_manifest.json')\nshutil.make_archive('/content/esm35m_scores', 'zip', 'scores')\nfrom google.colab import files\nfiles.download('/content/esm35m_scores.zip')"), build_colab_md("---\n## 9. (Optional) Malleability dose curve\nThe 10-epoch run above barely moved ESM's charge preference - but that could be under-training. This section re-trains the alkaliphile **case** and its matched **neutralophile control** at 30/100 epochs and with the aggressive **constant-lr** recipe AlkSecMPNN used (lr 1e-4, no warmup, constant schedule), scoring each on the *same* set. If the case cosine still doesn't rotate even here, ESM's bias is genuinely entrenched; if it does, malleability exists but is slower/costlier than the structure model.\n\n**Cost:** 8 train+score runs; scoring dominates (~15-20 min each). Trim the `SWEEP` list to shorten. Reuses `score_set.csv` from §4."), build_colab_code("SWEEP_ARMS = {\n    'AlkCase': ('data/alkaline_case_train.csv', 'data/alkaline_case_val.csv'),\n    'AlkNeu':  ('data/alkaline_neu_train.csv',  'data/alkaline_neu_val.csv'),\n    # add 'AcidCase'/'AcidNeu' here to sweep the polar arm too\n}\nRECIPES = {\n    'default': dict(lr=5e-5, extra=''),                                  # warmup+linear\n    'const':   dict(lr=1e-4, extra='--lr_scheduler_type constant --warmup_steps 0'),\n}\nSWEEP = [\n    ('AlkCase', 'default', 30), ('AlkCase', 'default', 100),\n    ('AlkCase', 'const', 30), ('AlkCase', 'const', 100),\n    ('AlkNeu', 'default', 30), ('AlkNeu', 'default', 100),\n    ('AlkNeu', 'const', 30), ('AlkNeu', 'const', 100),\n]"), build_colab_code("import os, pandas as pd\nos.makedirs('sweep_scores', exist_ok=True)\nmanifest = []\nfor arm, recipe, ep in SWEEP:\n    tr, va = SWEEP_ARMS[arm]; rc = RECIPES[recipe]\n    tag = f'{arm}__{recipe}__e{ep}'\n    outdir = f'runs_sweep/{tag}'; scoref = f'{tag}_masked_marginals.csv'\n    if os.path.exists(f'sweep_scores/{scoref}'):\n        print(f'skip {tag} (already scored)'); manifest.append(dict(tag=tag, arm=arm, recipe=recipe, epochs=ep, score_file=scoref)); continue\n    print(f'\\n##### train {tag} #####')\n    !python scripts/train_esm2_mlm.py --train_csv {tr} --val_csv {va} --out_dir {outdir} --epochs {ep} --learning_rate {rc['lr']} {rc['extra']}\n    print(f'##### score {tag} #####')\n    !python scripts/score_esm2_masked_marginals.py --model_dir {outdir} --input_csv score_set.csv --id_col Entry --seq_col sequence --out_csv sweep_scores/{scoref} --model_name {tag}\n    manifest.append(dict(tag=tag, arm=arm, recipe=recipe, epochs=ep, score_file=scoref))\npd.DataFrame(manifest).to_csv('sweep_scores/sweep_manifest.csv', index=False)\nprint('\\nwrote sweep_scores/sweep_manifest.csv')"), build_colab_md('## 10. Package & download the sweep\nUnzip into `outputs/esm35m_continual_pretraining/sweep_scores/`, then run `python paper_code/08_pca_figures/esm_epoch_sweep.py` for the cosine-vs-epoch table + figure.'), build_colab_code("import shutil\nfor f in glob.glob('runs_sweep/*/run_manifest.json'):\n    shutil.copy(f, 'sweep_scores/' + f.split('/')[1] + '_run_manifest.json')\nshutil.make_archive('/content/esm35m_sweep_scores', 'zip', 'sweep_scores')\nfrom google.colab import files\nfiles.download('/content/esm35m_sweep_scores.zip')")]
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
    out = build_colab_HERE.parents[2] / 'notebooks' / '07_finetuning' / 'esm35m_finetune_colab.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + '\n')
    print(f'wrote {out.name} ({len(cells)} cells)')
    return out
def build_colab_build_bundle(score_input: Path) -> Path:
    zip_path = build_colab_HERE / 'esm35m_colab_inputs.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(build_colab_PIPE / 'train_esm2_mlm.py', 'scripts/train_esm2_mlm.py')
        z.write(build_colab_PIPE / 'score_esm2_masked_marginals.py', 'scripts/score_esm2_masked_marginals.py')
        for f in TRAIN_FILES:
            z.write(build_colab_DATA / f, f'data/{f}')
        z.write(score_input, 'esm_score_input.csv')
    mb = zip_path.stat().st_size / 1000000.0
    print(f'wrote {zip_path.name} ({mb:.1f} MB): 2 scripts + {len(TRAIN_FILES)} data CSVs + score input')
    return zip_path
def build_colab__entry():
    si = build_score_input()
    build_colab_build_notebook()
    build_colab_build_bundle(si)
    print("\nDone. Upload esm35m_colab_inputs.zip in the notebook's step 2.")

# ---------- from build_generation_colab.py ----------
build_generation_colab_HERE = Path(__file__).resolve().parent
build_generation_colab_ROOT = build_generation_colab_HERE.parents[2]
build_generation_colab_PIPE = build_generation_colab_HERE.parent
build_generation_colab_DATA = build_generation_colab_ROOT / 'outputs' / 'esm35m_continual_pretraining' / 'data'
TEMPLATE_IDS = ['Q4R312', 'P0A7R6', 'A9A498', 'A0A1D8PCG7', 'Q9Z9J6']
def build_templates() -> Path:
    d = pd.read_csv(build_generation_colab_ROOT / 'design' / 'design_input_proteins.csv')
    sub = d[d.uniprot_id.isin(TEMPLATE_IDS)][['uniprot_id', 'domain', 'sequence_length', 'wt_sequence']]
    sub = sub.set_index('uniprot_id').loc[TEMPLATE_IDS].reset_index()
    out = build_generation_colab_HERE / 'gen_templates.csv'
    sub.to_csv(out, index=False)
    print(f"gen_templates.csv: {len(sub)} templates ({', '.join(sub.uniprot_id)})")
    return out
def build_generation_colab_code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t.strip('\n').splitlines(keepends=True)}
def build_generation_colab_md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.splitlines(keepends=True)}
def build_generation_colab_build_notebook() -> Path:
    cells = [build_generation_colab_md('# ESM2-35M generation malleability (Reviewer R1.4, Phase 1)\n\nApples-to-apples with the ProteinMPNN designs: fine-tune AlkSecESM35M, then compute the **per-position amino-acid probability map** (base vs fine-tuned) for the 5 templates with the largest ProteinMPNN surface shift. Phase 1 answers *does the fine-tune change what ESM would place at each position?*; Phase 2 (optional cell at the end) actually generates sequences by iterative in-filling.\n\nUpload **`esm35m_generation_inputs.zip`**; set Runtime -> **T4 GPU**.'), build_generation_colab_md('## 1. GPU + dependencies'), build_generation_colab_code('!nvidia-smi -L\n!pip -q install "transformers>=4.45" "datasets>=2.20" "accelerate>=0.33" safetensors 2>/dev/null\nimport transformers, torch\nprint("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())'), build_generation_colab_md('## 2. Upload & unzip'), build_generation_colab_code("import os, io, zipfile\nfrom google.colab import files\nup = files.upload()\nname = [n for n in up if n.endswith('.zip')][0]\nwith zipfile.ZipFile(io.BytesIO(up[name])) as z: z.extractall('/content/esm_gen')\nos.chdir('/content/esm_gen')\nimport sys; sys.path.insert(0, 'scripts')\nos.makedirs('gen_out', exist_ok=True)\nprint(sorted(os.listdir()))"), build_generation_colab_md('## 3. Config'), build_generation_colab_code('BASE_MODEL = "facebook/esm2_t12_35M_UR50D"\nEPOCHS = 30   # dose curve saturates by ~30; bump to preempt under-training doubts\nLR = 5e-5'), build_generation_colab_md('## 4. Fine-tune AlkSecESM35M + the neutralophile control\nSame recipe/data as the scoring arm. The matched neutralophile-control fine-tune is the specificity check: a cohort-specific surface shift should appear in AlkSecESM35M but not (or far less) in NeuSecESM35M.'), build_generation_colab_code("ARMS = {'AlkSecESM35M': 'alkaline_case', 'NeuSecESM35M': 'alkaline_neu'}\nfor name, stem in ARMS.items():\n    print(f'\\n===== training {name} =====')\n    !python scripts/train_esm2_mlm.py \\\n      --train_csv data/{stem}_train.csv --val_csv data/{stem}_val.csv \\\n      --out_dir runs/{name} --epochs {EPOCHS} --learning_rate {LR}"), build_generation_colab_md("## 5. Phase 1 - per-position probability maps (base vs fine-tuned)\nFor each template, mask each position and read the model's distribution over the 20 amino acids, for both models. Saved as one `.npz` per template."), build_generation_colab_code('import numpy as np, pandas as pd, torch\nfrom transformers import AutoTokenizer, AutoModelForMaskedLM\nfrom esm_generation import esm_position_distributions, CANONICAL_AA\n\ndevice = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')\ntmpl = pd.read_csv(\'gen_templates.csv\')\nMODELS = {\'base\': BASE_MODEL, \'AlkSecESM35M\': \'runs/AlkSecESM35M\',\n          \'NeuSecESM35M\': \'runs/NeuSecESM35M\'}\nloaded = {}\nfor tag, path in MODELS.items():\n    tok = AutoTokenizer.from_pretrained(path)\n    mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()\n    loaded[tag] = (tok, mdl)\n\nfor _, r in tmpl.iterrows():\n    seq = r[\'wt_sequence\']\n    mats = {}\n    for tag, (tok, mdl) in loaded.items():\n        mats[tag] = esm_position_distributions(seq, tok, mdl, device)\n        print(f"{r[\'uniprot_id\']} {tag}: {mats[tag].shape}")\n    np.savez(f"gen_out/{r[\'uniprot_id\']}_probs.npz",\n             seq=seq, aa_order=np.array(list(CANONICAL_AA)),\n             base=mats[\'base\'], AlkSecESM35M=mats[\'AlkSecESM35M\'],\n             NeuSecESM35M=mats[\'NeuSecESM35M\'])'), build_generation_colab_md('## 6. Download Phase-1 maps\nUnzip into `outputs/esm35m_continual_pretraining/generation/` and run `python paper_code/08_pca_figures/esm_design_heatmaps.py` locally.'), build_generation_colab_code("import shutil\nshutil.make_archive('/content/esm35m_generation_phase1', 'zip', 'gen_out')\nfrom google.colab import files\nfiles.download('/content/esm35m_generation_phase1.zip')"), build_generation_colab_md('---\n## 7. (Optional) Phase 2 - generate sequences by iterative in-filling\nRun only after inspecting the Phase-1 heatmaps. Gibbs-style masked in-filling, T=0.1, 8 designs/template, base vs fine-tuned. Saves designed sequences.'), build_generation_colab_code("from esm_generation import make_esm_predict_fn, iterative_infill\nN_DESIGNS, N_PASSES, TEMP = 8, 1, 0.1\nrows = []\nfor _, r in tmpl.iterrows():\n    seq = r['wt_sequence']\n    for tag, (tok, mdl) in loaded.items():\n        pf = make_esm_predict_fn(tok, mdl, device)\n        for k in range(N_DESIGNS):\n            des = iterative_infill(seq, pf, n_passes=N_PASSES, temperature=TEMP, seed=k)\n            rows.append(dict(uniprot_id=r['uniprot_id'], model=tag, sample_idx=k, sequence=des))\n    print(r['uniprot_id'], 'done')\nimport pandas as pd\npd.DataFrame(rows).to_csv('gen_out/esm_designs.csv', index=False)\nshutil.make_archive('/content/esm35m_generation_designs', 'zip', 'gen_out')\nfiles.download('/content/esm35m_generation_designs.zip')")]
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
    out = build_generation_colab_HERE.parents[2] / 'notebooks' / '07_finetuning' / 'esm35m_generation_colab.ipynb'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + '\n')
    print(f'wrote {out.name} ({len(cells)} cells)')
    return out
def build_generation_colab_build_bundle(templates: Path) -> Path:
    zp = build_generation_colab_HERE / 'esm35m_generation_inputs.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(build_generation_colab_PIPE / 'train_esm2_mlm.py', 'scripts/train_esm2_mlm.py')
        z.write(build_generation_colab_PIPE / 'esm_generation.py', 'scripts/esm_generation.py')
        z.write(build_generation_colab_DATA / 'alkaline_case_train.csv', 'data/alkaline_case_train.csv')
        z.write(build_generation_colab_DATA / 'alkaline_case_val.csv', 'data/alkaline_case_val.csv')
        z.write(build_generation_colab_DATA / 'alkaline_neu_train.csv', 'data/alkaline_neu_train.csv')
        z.write(build_generation_colab_DATA / 'alkaline_neu_val.csv', 'data/alkaline_neu_val.csv')
        z.write(templates, 'gen_templates.csv')
    print(f'wrote {zp.name} ({zp.stat().st_size / 1000.0:.0f} KB)')
    return zp
def build_generation_colab__entry():
    t = build_templates()
    build_generation_colab_build_notebook()
    build_generation_colab_build_bundle(t)
    print("\nDone. Upload esm35m_generation_inputs.zip in the notebook's step 2.")

_STEPS = {
    'build-colab': build_colab__entry,
    'build-generation-colab': build_generation_colab__entry,
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

