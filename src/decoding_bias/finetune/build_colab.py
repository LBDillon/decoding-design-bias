#!/usr/bin/env python
"""Build the ESM2-35M malleability Colab bundle + notebook (Reviewer R1.4).

Produces, under this colab/ directory:
  * esm_score_input.csv        -- [Entry, sequence, isoelectric_point] for all
                                  v12_cli proteins (sequences from metadata_v12),
                                  the shared scoring set the cosine test merges on.
  * esm35m_colab_inputs.zip    -- upload this one file to Colab. Contains the two
                                  CLI scripts, the 4 training arms (train+val), and
                                  esm_score_input.csv.
  * esm35m_finetune_colab.ipynb -- the notebook that trains 4 arms + scores base/FT.

Run locally from the repo root or anywhere:  python build_colab.py
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPE = HERE.parent  # paper_code/10_esm_continual_pretraining
DATA = ROOT / "outputs" / "esm35m_continual_pretraining" / "data"

V12_CLI = ROOT / "dataset_update" / "main_plus_r2_r3_analysis_v12_cli.csv"
METADATA = ROOT / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"

# The 4 continued-pretraining arms (must match ESM_SCORES stems in
# paper_code/08_pca_figures/charge_pca_cosine_shift.py).
ARMS = [
    # model_name / train csv / val csv
    ("AlkSecESM35M", "alkaline_case_train.csv", "alkaline_case_val.csv"),
    ("AcidSecESM35M", "acid_case_train.csv", "acid_case_val.csv"),
    ("NeuSecESM35M_AlkMatched", "alkaline_neu_train.csv", "alkaline_neu_val.csv"),
    ("NeuSecESM35M_AcidMatched", "acid_neu_train.csv", "acid_neu_val.csv"),
]
TRAIN_FILES = sorted({f for _, tr, va in ARMS for f in (tr, va)})


def build_score_input() -> Path:
    """[Entry, sequence, isoelectric_point] for every v12_cli protein."""
    cli = pd.read_csv(V12_CLI, low_memory=False)[["Entry", "isoelectric_point"]]
    meta = pd.read_csv(METADATA, low_memory=False)[["Entry", "sequence"]].dropna(subset=["sequence"])
    df = cli.merge(meta, on="Entry", how="left")
    missing = int(df["sequence"].isna().sum())
    df = df.dropna(subset=["sequence"]).reset_index(drop=True)
    out = HERE / "esm_score_input.csv"
    df.to_csv(out, index=False)
    print(f"esm_score_input.csv: {len(df)} proteins with sequence "
          f"({missing} v12_cli entries had no sequence and were dropped)")
    return out


# --- notebook construction (dependency-free ipynb JSON) ---------------------
def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def build_notebook() -> Path:
    arms_py = "ARMS = [\n" + "".join(
        f'    ("{n}", "data/{tr}", "data/{va}"),\n' for n, tr, va in ARMS) + "]\n"

    cells = [
        md(
            "# ESM2-35M secretome malleability (Reviewer R1.4)\n\n"
            "Sequence-model analogue of the AlkSecMPNN perturbation: continue "
            "masked-LM training of **ESM2-35M** on the *same* secreted-extremophile "
            "cohort and splits, then score WT sequences with masked-marginal "
            "pseudo-log-likelihood **before vs after** fine-tuning. The pH-feature "
            "**cosine test** (score-preference direction vs the acid--base axis) is "
            "run afterwards, locally, on the same axis as the structure models.\n\n"
            "**You upload one file:** `esm35m_colab_inputs.zip`.\n\n"
            "**Runtime:** set Runtime → Change runtime type → **T4 GPU**.\n\n"
            "This notebook only orchestrates the committed CLI scripts "
            "(`train_esm2_mlm.py`, `score_esm2_masked_marginals.py`) - no logic lives here."
        ),
        md("## 1. GPU + dependencies"),
        code(
            "!nvidia-smi -L\n"
            "# torch ships with Colab; add the HF stack the scripts need.\n"
            '!pip -q install "transformers>=4.45" "datasets>=2.20" "accelerate>=0.33" '
            "safetensors scikit-learn scipy 2>/dev/null\n"
            "import transformers, torch\n"
            'print("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())'
        ),
        md("## 2. Upload & unzip the input bundle\nChoose `esm35m_colab_inputs.zip` when prompted."),
        code(
            "import os, io, zipfile\n"
            "from google.colab import files\n"
            "up = files.upload()\n"
            "name = [n for n in up if n.endswith('.zip')][0]\n"
            "with zipfile.ZipFile(io.BytesIO(up[name])) as z:\n"
            "    z.extractall('/content/esm_ft')\n"
            "os.chdir('/content/esm_ft')\n"
            "os.makedirs('scores', exist_ok=True); os.makedirs('runs', exist_ok=True)\n"
            "print('working dir:', os.getcwd())\n"
            "print('contents:', sorted(os.listdir()))\n"
            "print('data:', sorted(os.listdir('data')))"
        ),
        md("## 3. Config\n"
           "`SCORE_N` controls the shared scoring set. `0` = all ~10k proteins "
           "(faithful to the structure-model set, but ~6-8 h across 5 models). A "
           "pI-stratified subset (default 3000) spans the acid--base axis and finishes "
           "in ~1.5-2 h with tight bootstrap CIs. The SAME sampled proteins are scored "
           "by every model, so the before/after comparison is paired."),
        code(
            'BASE_MODEL = "facebook/esm2_t12_35M_UR50D"\n'
            "EPOCHS = 10\n"
            "LR = 5e-5\n"
            "SCORE_N = 3000   # 0 = score all proteins\n"
            "SEED = 0\n\n"
            + arms_py
        ),
        md("## 4. Build the shared scoring set (same proteins for every model)"),
        code(
            "import pandas as pd, numpy as np\n"
            "inp = pd.read_csv('esm_score_input.csv')\n"
            "if SCORE_N and SCORE_N < len(inp):\n"
            "    inp['_b'] = pd.qcut(inp['isoelectric_point'], 10, labels=False, duplicates='drop')\n"
            "    per = max(1, SCORE_N // (inp['_b'].nunique()))\n"
            "    inp = (inp.groupby('_b', group_keys=False)\n"
            "              .apply(lambda d: d.sample(min(len(d), per), random_state=SEED))\n"
            "              .drop(columns='_b').reset_index(drop=True))\n"
            "inp[['Entry', 'sequence']].to_csv('score_set.csv', index=False)\n"
            "print('scoring set:', len(inp), 'proteins  (pI range '\n"
            "      f\"{inp.isoelectric_point.min():.1f}-{inp.isoelectric_point.max():.1f})\")"
        ),
        md("## 5. Score the BASE model (the 'before')"),
        code(
            "!python scripts/score_esm2_masked_marginals.py \\\n"
            "  --model_dir {BASE_MODEL} --input_csv score_set.csv \\\n"
            "  --id_col Entry --seq_col sequence \\\n"
            "  --out_csv scores/BaseESM35M_masked_marginals.csv --model_name BaseESM35M"
        ),
        md("## 6. Continue-pretrain the 4 arms\n"
           "Alkaliphile / acidophile **cases** = the steered models; the matched "
           "**neutralophile controls** are the symmetric specificity check."),
        code(
            "for name, tr, va in ARMS:\n"
            "    print(f'\\n===== training {name} =====')\n"
            "    !python scripts/train_esm2_mlm.py \\\n"
            "      --train_csv {tr} --val_csv {va} --out_dir runs/{name} \\\n"
            "      --epochs {EPOCHS} --learning_rate {LR} --overwrite_output_dir"
        ),
        md("## 7. Score each fine-tuned model (the 'after') on the same set"),
        code(
            "for name, tr, va in ARMS:\n"
            "    print(f'\\n===== scoring {name} =====')\n"
            "    !python scripts/score_esm2_masked_marginals.py \\\n"
            "      --model_dir runs/{name} --input_csv score_set.csv \\\n"
            "      --id_col Entry --seq_col sequence \\\n"
            "      --out_csv scores/{name}_masked_marginals.csv --model_name {name}"
        ),
        md("## 8. Package & download\n"
           "Download `esm35m_scores.zip`, unzip it into "
           "`outputs/esm35m_continual_pretraining/scores/` in the repo, then run "
           "`python paper_code/08_pca_figures/charge_pca_cosine_shift.py` - the ESM "
           "rows will appear beside the ProteinMPNN rows on the same acid--base axis, "
           "each with a bootstrap 95% CI."),
        code(
            "import shutil, glob\n"
            "for f in glob.glob('runs/*/run_manifest.json'):\n"
            "    shutil.copy(f, 'scores/' + f.split('/')[1] + '_run_manifest.json')\n"
            "shutil.make_archive('/content/esm35m_scores', 'zip', 'scores')\n"
            "from google.colab import files\n"
            "files.download('/content/esm35m_scores.zip')"
        ),
        md("---\n## 9. (Optional) Malleability dose curve\n"
           "The 10-epoch run above barely moved ESM's charge preference - but that "
           "could be under-training. This section re-trains the alkaliphile **case** "
           "and its matched **neutralophile control** at 30/100 epochs and with the "
           "aggressive **constant-lr** recipe AlkSecMPNN used (lr 1e-4, no warmup, "
           "constant schedule), scoring each on the *same* set. If the case cosine "
           "still doesn't rotate even here, ESM's bias is genuinely entrenched; if it "
           "does, malleability exists but is slower/costlier than the structure model.\n\n"
           "**Cost:** 8 train+score runs; scoring dominates (~15-20 min each). Trim "
           "the `SWEEP` list to shorten. Reuses `score_set.csv` from §4."),
        code(
            "SWEEP_ARMS = {\n"
            "    'AlkCase': ('data/alkaline_case_train.csv', 'data/alkaline_case_val.csv'),\n"
            "    'AlkNeu':  ('data/alkaline_neu_train.csv',  'data/alkaline_neu_val.csv'),\n"
            "    # add 'AcidCase'/'AcidNeu' here to sweep the polar arm too\n"
            "}\n"
            "RECIPES = {\n"
            "    'default': dict(lr=5e-5, extra=''),                                  # warmup+linear\n"
            "    'const':   dict(lr=1e-4, extra='--lr_scheduler_type constant --warmup_steps 0'),\n"
            "}\n"
            "SWEEP = [\n"
            "    ('AlkCase', 'default', 30), ('AlkCase', 'default', 100),\n"
            "    ('AlkCase', 'const', 30), ('AlkCase', 'const', 100),\n"
            "    ('AlkNeu', 'default', 30), ('AlkNeu', 'default', 100),\n"
            "    ('AlkNeu', 'const', 30), ('AlkNeu', 'const', 100),\n"
            "]"
        ),
        code(
            "import os, pandas as pd\n"
            "os.makedirs('sweep_scores', exist_ok=True)\n"
            "manifest = []\n"
            "for arm, recipe, ep in SWEEP:\n"
            "    tr, va = SWEEP_ARMS[arm]; rc = RECIPES[recipe]\n"
            "    tag = f'{arm}__{recipe}__e{ep}'\n"
            "    outdir = f'runs_sweep/{tag}'; scoref = f'{tag}_masked_marginals.csv'\n"
            "    if os.path.exists(f'sweep_scores/{scoref}'):\n"
            "        print(f'skip {tag} (already scored)'); manifest.append(dict(tag=tag, arm=arm, recipe=recipe, epochs=ep, score_file=scoref)); continue\n"
            "    print(f'\\n##### train {tag} #####')\n"
            "    !python scripts/train_esm2_mlm.py --train_csv {tr} --val_csv {va} --out_dir {outdir} --epochs {ep} --learning_rate {rc['lr']} {rc['extra']}\n"
            "    print(f'##### score {tag} #####')\n"
            "    !python scripts/score_esm2_masked_marginals.py --model_dir {outdir} --input_csv score_set.csv --id_col Entry --seq_col sequence --out_csv sweep_scores/{scoref} --model_name {tag}\n"
            "    manifest.append(dict(tag=tag, arm=arm, recipe=recipe, epochs=ep, score_file=scoref))\n"
            "pd.DataFrame(manifest).to_csv('sweep_scores/sweep_manifest.csv', index=False)\n"
            "print('\\nwrote sweep_scores/sweep_manifest.csv')"
        ),
        md("## 10. Package & download the sweep\n"
           "Unzip into `outputs/esm35m_continual_pretraining/sweep_scores/`, then run "
           "`python paper_code/08_pca_figures/esm_epoch_sweep.py` for the cosine-vs-epoch "
           "table + figure."),
        code(
            "import shutil\n"
            "for f in glob.glob('runs_sweep/*/run_manifest.json'):\n"
            "    shutil.copy(f, 'sweep_scores/' + f.split('/')[1] + '_run_manifest.json')\n"
            "shutil.make_archive('/content/esm35m_sweep_scores', 'zip', 'sweep_scores')\n"
            "from google.colab import files\n"
            "files.download('/content/esm35m_sweep_scores.zip')"
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = HERE.parents[2] / "notebooks" / "07_finetuning" / "esm35m_finetune_colab.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out.name} ({len(cells)} cells)")
    return out


def build_bundle(score_input: Path) -> Path:
    zip_path = HERE / "esm35m_colab_inputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(PIPE / "train_esm2_mlm.py", "scripts/train_esm2_mlm.py")
        z.write(PIPE / "score_esm2_masked_marginals.py", "scripts/score_esm2_masked_marginals.py")
        for f in TRAIN_FILES:
            z.write(DATA / f, f"data/{f}")
        z.write(score_input, "esm_score_input.csv")
    mb = zip_path.stat().st_size / 1e6
    print(f"wrote {zip_path.name} ({mb:.1f} MB): 2 scripts + {len(TRAIN_FILES)} data CSVs + score input")
    return zip_path


if __name__ == "__main__":
    si = build_score_input()
    build_notebook()
    build_bundle(si)
    print("\nDone. Upload esm35m_colab_inputs.zip in the notebook's step 2.")
