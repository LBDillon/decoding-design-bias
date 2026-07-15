#!/usr/bin/env python
"""Build the ESM2-35M *generation* Colab bundle + notebook (Reviewer R1.4, Phase 1).

Apples-to-apples with ProteinMPNN designs: fine-tune AlkSecESM35M, then compute the
per-position amino-acid probability map (base vs fine-tuned) for the 5 templates with
the largest ProteinMPNN surface shift. Phase 2 (iterative in-filling generation) is
included as an optional cell to run after inspecting the Phase-1 heatmaps.

Produces, under this colab/ directory:
  * gen_templates.csv               -- 5 templates (uniprot_id, wt_sequence)
  * esm35m_generation_inputs.zip    -- upload this: train_esm2_mlm.py, esm_generation.py,
                                       alkaline_case train/val CSVs, gen_templates.csv
  * esm35m_generation_colab.ipynb   -- train FT -> Phase-1 prob maps -> download

Run: python build_generation_colab.py
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPE = HERE.parent
DATA = ROOT / "outputs" / "esm35m_continual_pretraining" / "data"

# top-5 templates by ProteinMPNN AlkSec-vs-AcidSec surface net-charge divergence
TEMPLATE_IDS = ["Q4R312", "P0A7R6", "A9A498", "A0A1D8PCG7", "Q9Z9J6"]


def build_templates() -> Path:
    d = pd.read_csv(ROOT / "design" / "design_input_proteins.csv")
    sub = d[d.uniprot_id.isin(TEMPLATE_IDS)][["uniprot_id", "domain", "sequence_length", "wt_sequence"]]
    sub = sub.set_index("uniprot_id").loc[TEMPLATE_IDS].reset_index()  # keep ranked order
    out = HERE / "gen_templates.csv"
    sub.to_csv(out, index=False)
    print(f"gen_templates.csv: {len(sub)} templates ({', '.join(sub.uniprot_id)})")
    return out


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def build_notebook() -> Path:
    cells = [
        md("# ESM2-35M generation malleability (Reviewer R1.4, Phase 1)\n\n"
           "Apples-to-apples with the ProteinMPNN designs: fine-tune AlkSecESM35M, then "
           "compute the **per-position amino-acid probability map** (base vs fine-tuned) for "
           "the 5 templates with the largest ProteinMPNN surface shift. Phase 1 answers "
           "*does the fine-tune change what ESM would place at each position?*; Phase 2 "
           "(optional cell at the end) actually generates sequences by iterative in-filling.\n\n"
           "Upload **`esm35m_generation_inputs.zip`**; set Runtime -> **T4 GPU**."),
        md("## 1. GPU + dependencies"),
        code("!nvidia-smi -L\n"
             '!pip -q install "transformers>=4.45" "datasets>=2.20" "accelerate>=0.33" '
             "safetensors 2>/dev/null\n"
             "import transformers, torch\n"
             'print("transformers", transformers.__version__, "| cuda", torch.cuda.is_available())'),
        md("## 2. Upload & unzip"),
        code("import os, io, zipfile\n"
             "from google.colab import files\n"
             "up = files.upload()\n"
             "name = [n for n in up if n.endswith('.zip')][0]\n"
             "with zipfile.ZipFile(io.BytesIO(up[name])) as z: z.extractall('/content/esm_gen')\n"
             "os.chdir('/content/esm_gen')\n"
             "import sys; sys.path.insert(0, 'scripts')\n"
             "os.makedirs('gen_out', exist_ok=True)\n"
             "print(sorted(os.listdir()))"),
        md("## 3. Config"),
        code('BASE_MODEL = "facebook/esm2_t12_35M_UR50D"\n'
             "EPOCHS = 30   # dose curve saturates by ~30; bump to preempt under-training doubts\n"
             "LR = 5e-5"),
        md("## 4. Fine-tune AlkSecESM35M + the neutralophile control\n"
           "Same recipe/data as the scoring arm. The matched neutralophile-control fine-tune "
           "is the specificity check: a cohort-specific surface shift should appear in "
           "AlkSecESM35M but not (or far less) in NeuSecESM35M."),
        code("ARMS = {'AlkSecESM35M': 'alkaline_case', 'NeuSecESM35M': 'alkaline_neu'}\n"
             "for name, stem in ARMS.items():\n"
             "    print(f'\\n===== training {name} =====')\n"
             "    !python scripts/train_esm2_mlm.py \\\n"
             "      --train_csv data/{stem}_train.csv --val_csv data/{stem}_val.csv \\\n"
             "      --out_dir runs/{name} --epochs {EPOCHS} --learning_rate {LR}"),
        md("## 5. Phase 1 - per-position probability maps (base vs fine-tuned)\n"
           "For each template, mask each position and read the model's distribution over the "
           "20 amino acids, for both models. Saved as one `.npz` per template."),
        code("import numpy as np, pandas as pd, torch\n"
             "from transformers import AutoTokenizer, AutoModelForMaskedLM\n"
             "from esm_generation import esm_position_distributions, CANONICAL_AA\n"
             "\n"
             "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
             "tmpl = pd.read_csv('gen_templates.csv')\n"
             "MODELS = {'base': BASE_MODEL, 'AlkSecESM35M': 'runs/AlkSecESM35M',\n"
             "          'NeuSecESM35M': 'runs/NeuSecESM35M'}\n"
             "loaded = {}\n"
             "for tag, path in MODELS.items():\n"
             "    tok = AutoTokenizer.from_pretrained(path)\n"
             "    mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()\n"
             "    loaded[tag] = (tok, mdl)\n"
             "\n"
             "for _, r in tmpl.iterrows():\n"
             "    seq = r['wt_sequence']\n"
             "    mats = {}\n"
             "    for tag, (tok, mdl) in loaded.items():\n"
             "        mats[tag] = esm_position_distributions(seq, tok, mdl, device)\n"
             "        print(f\"{r['uniprot_id']} {tag}: {mats[tag].shape}\")\n"
            "    np.savez(f\"gen_out/{r['uniprot_id']}_probs.npz\",\n"
             "             seq=seq, aa_order=np.array(list(CANONICAL_AA)),\n"
             "             base=mats['base'], AlkSecESM35M=mats['AlkSecESM35M'],\n"
             "             NeuSecESM35M=mats['NeuSecESM35M'])"),
        md("## 6. Download Phase-1 maps\n"
           "Unzip into `outputs/esm35m_continual_pretraining/generation/` and run "
           "`python paper_code/08_pca_figures/esm_design_heatmaps.py` locally."),
        code("import shutil\n"
             "shutil.make_archive('/content/esm35m_generation_phase1', 'zip', 'gen_out')\n"
             "from google.colab import files\n"
             "files.download('/content/esm35m_generation_phase1.zip')"),
        md("---\n## 7. (Optional) Phase 2 - generate sequences by iterative in-filling\n"
           "Run only after inspecting the Phase-1 heatmaps. Gibbs-style masked in-filling, "
           "T=0.1, 8 designs/template, base vs fine-tuned. Saves designed sequences."),
        code("from esm_generation import make_esm_predict_fn, iterative_infill\n"
             "N_DESIGNS, N_PASSES, TEMP = 8, 1, 0.1\n"
             "rows = []\n"
             "for _, r in tmpl.iterrows():\n"
             "    seq = r['wt_sequence']\n"
             "    for tag, (tok, mdl) in loaded.items():\n"
             "        pf = make_esm_predict_fn(tok, mdl, device)\n"
             "        for k in range(N_DESIGNS):\n"
             "            des = iterative_infill(seq, pf, n_passes=N_PASSES, temperature=TEMP, seed=k)\n"
             "            rows.append(dict(uniprot_id=r['uniprot_id'], model=tag, sample_idx=k, sequence=des))\n"
             "    print(r['uniprot_id'], 'done')\n"
             "import pandas as pd\n"
             "pd.DataFrame(rows).to_csv('gen_out/esm_designs.csv', index=False)\n"
             "shutil.make_archive('/content/esm35m_generation_designs', 'zip', 'gen_out')\n"
             "files.download('/content/esm35m_generation_designs.zip')"),
    ]
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = HERE.parents[2] / "notebooks" / "07_finetuning" / "esm35m_generation_colab.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out.name} ({len(cells)} cells)")
    return out


def build_bundle(templates: Path) -> Path:
    zp = HERE / "esm35m_generation_inputs.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(PIPE / "train_esm2_mlm.py", "scripts/train_esm2_mlm.py")
        z.write(PIPE / "esm_generation.py", "scripts/esm_generation.py")
        z.write(DATA / "alkaline_case_train.csv", "data/alkaline_case_train.csv")
        z.write(DATA / "alkaline_case_val.csv", "data/alkaline_case_val.csv")
        z.write(DATA / "alkaline_neu_train.csv", "data/alkaline_neu_train.csv")
        z.write(DATA / "alkaline_neu_val.csv", "data/alkaline_neu_val.csv")
        z.write(templates, "gen_templates.csv")
    print(f"wrote {zp.name} ({zp.stat().st_size/1e3:.0f} KB)")
    return zp


if __name__ == "__main__":
    t = build_templates()
    build_notebook()
    build_bundle(t)
    print("\nDone. Upload esm35m_generation_inputs.zip in the notebook's step 2.")
