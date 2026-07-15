"""
Predict DeepStabP melting temperature (Tm) for the whole v12 cohort (10,148 proteins).

DeepStabP (CSBiology/deepStabP) API:  POST {BASE}/api/v1/predict
  body  {"growth_temp": int, "mt_mode": "Lysate"|"Cell",
         "fasta": [{"header": <Entry>, "sequence": <seq, rare AA -> X>}]}
  reply {"Prediction": <table with columns Protein, Tm>}

This script must run on a NETWORKED machine (the analysis sandbox has no network):
  Option A (recommended for 10k) - run DeepStabP locally:
      git clone https://github.com/CSBiology/deepStabP && cd deepStabP
      ./build.cmd dockertest          # API on http://localhost:8000  (UI on :5000)
      python design/predict_tm_full.py   # BASE defaults to localhost:8000
  Option B - use the public server (loads their server; be gentle):
      BASE = "https://csb-deepstabp.bio.rptu.de"

IMPORTANT - comparability: set GROWTH_TEMP and MT_MODE to the SAME values you used
for the design-set Tm (Design_WT_TM.csv). A FIXED growth temperature for all proteins
(default 37) makes Tm reflect intrinsic sequence stability and avoids leaking each
organism's real growth temperature (which would be circular for ranking thermophiles).

Output: design/outputs/dataset_tm_predictions.csv  (Entry, Tm)  - resumable.
"""
import os, sys, time, io, json
from pathlib import Path
import pandas as pd, requests

HERE = Path(__file__).resolve().parent
META = HERE.parent / "dataset_update" / "main_plus_r2_r3_metadata_v12.csv"
OUT = HERE / "outputs" / "dataset_tm_predictions.csv"

BASE = os.environ.get("DEEPSTABP_URL", "http://localhost:8000").rstrip("/")
GROWTH_TEMP = int(os.environ.get("GROWTH_TEMP", "37"))     # match the design-Tm setting
MT_MODE = os.environ.get("MT_MODE", "Lysate")              # "Lysate" or "Cell"
BATCH = int(os.environ.get("BATCH", "100"))                # sequences per request
STD = set("ACDEFGHIKLMNPQRSTVWY")


def clean(seq):
    return "".join(c if c in STD else "X" for c in str(seq).upper())


def parse_prediction(pred):
    """Coerce the {'Prediction': ...} payload to a DataFrame with Protein, Tm."""
    if isinstance(pred, str):
        try: df = pd.read_json(io.StringIO(pred))
        except ValueError: df = pd.DataFrame(json.loads(pred))
    else:
        df = pd.DataFrame(pred)
    cols = {c.lower(): c for c in df.columns}
    pcol = cols.get("protein") or cols.get("header") or list(df.columns)[0]
    tcol = cols.get("tm") or [c for c in df.columns if c.lower().startswith("tm")][0]
    return df[[pcol, tcol]].rename(columns={pcol: "Protein", tcol: "Tm"})


def main():
    m = pd.read_csv(META, low_memory=False).dropna(subset=["sequence"])
    m = m[m.sequence.str.len() > 0][["Entry", "sequence"]]
    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT).Entry.astype(str))
        print(f"resuming: {len(done)} already predicted")
    todo = m[~m.Entry.astype(str).isin(done)].reset_index(drop=True)
    print(f"{len(todo)} to predict via {BASE}/api/v1/predict "
          f"(growth_temp={GROWTH_TEMP}, mt_mode={MT_MODE}, batch={BATCH})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not OUT.exists()
    for i in range(0, len(todo), BATCH):
        sub = todo.iloc[i:i + BATCH]
        payload = {"growth_temp": GROWTH_TEMP, "mt_mode": MT_MODE,
                   "fasta": [{"header": r.Entry, "sequence": clean(r.sequence)} for r in sub.itertuples()]}
        for attempt in range(4):
            try:
                r = requests.post(f"{BASE}/api/v1/predict", json=payload, timeout=600)
                r.raise_for_status()
                df = parse_prediction(r.json()["Prediction"])
                # map the returned Protein header back to Entry (header == Entry here)
                df["Entry"] = df["Protein"].astype(str)
                df[["Entry", "Tm"]].to_csv(OUT, mode="a", header=header_needed, index=False)
                header_needed = False
                print(f"  batch {i//BATCH+1}/{-(-len(todo)//BATCH)}: +{len(df)}  "
                      f"(Tm {df.Tm.min():.1f}-{df.Tm.max():.1f})", flush=True)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"  batch {i//BATCH+1} attempt {attempt+1} failed: {str(e)[:80]} - retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            print("  giving up on this batch; re-run to resume."); sys.exit(1)

    res = pd.read_csv(OUT)
    print(f"\nDone: {len(res)} Tm predictions written to {OUT}")
    print(res.Tm.describe().round(1).to_string())


if __name__ == "__main__":
    main()
