#!/usr/bin/env python3
"""Fingerprint notebooks by their CODE CELLS only (ignoring outputs + metadata).

The repo notebooks have their outputs cleared, so a byte comparison against a
Colab-executed copy is meaningless. This hashes only the concatenated code-cell
source, so two notebooks with the same code but different outputs match.

Usage:
    python notebooks/fingerprint.py                 # fingerprint every repo notebook
    python notebooks/fingerprint.py a.ipynb b.ipynb # fingerprint specific files

To check a repo notebook against your Colab Drive copy:
    1. Download the Drive notebook (File -> Download -> .ipynb).
    2. python notebooks/fingerprint.py notebooks/07_finetuning/esm35m_finetune_colab.ipynb ~/Downloads/esm35m_finetune_colab.ipynb
    3. Same hash  -> code is identical (only outputs differ; repo is up to date).
       Different  -> the code diverged; reconcile before you rely on the repo copy.
"""
import hashlib
import json
import sys
from pathlib import Path


def code_fingerprint(path: Path) -> tuple[str, int]:
    nb = json.loads(path.read_text())
    chunks = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if src.strip():
                chunks.append(src.rstrip())
    blob = "\n# --CELL-- #\n".join(chunks).encode()
    return hashlib.sha256(blob).hexdigest()[:16], len(chunks)


def main(argv):
    paths = [Path(p) for p in argv] if argv else sorted(Path(__file__).parent.rglob("*.ipynb"))
    for p in paths:
        try:
            h, n = code_fingerprint(p)
            print(f"{h}  {n:3d} code cells  {p}")
        except Exception as e:  # pragma: no cover
            print(f"{'ERROR':16}  {p}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
