"""
Shared utilities for alkaliphile-secretome fine-tuning.

Holds: self-contained data paths, the surface/axis metric definitions (SASA surface mask +
the locked axis features), and the ProteinMPNN design/evaluation helpers. Imported by
train.py, evaluate.py and select.py so there is one definition of each thing.

NOTE: functions that touch ProteinMPNN import it lazily - the caller must put the ProteinMPNN
clone on sys.path first (e.g. `sys.path.insert(0, args.mpnn)`).
"""
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio
from biotite.sequence import ProteinSequence
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------------------------------------------- paths (self-contained)
HERE = Path(__file__).resolve().parent          # finetune/alkmpnn
FT = HERE.parent                                 # finetune
ROOT = FT.parent
DATA = FT / "data" if (FT / "data").exists() else ROOT / "design" / "outputs"
STRUCT = DATA / "structures_alkaliphile"
OUTPUTS = FT / "outputs"

# ---------------------------------------------------------------- residue helpers
RSA_CUT = 0.25
MAXASA = {'A':129,'R':274,'N':195,'D':193,'C':167,'E':223,'Q':225,'G':104,'H':224,'I':197,
          'L':201,'K':236,'M':224,'F':240,'P':159,'S':155,'T':172,'W':285,'Y':263,'V':174}
ACIDIC, BASIC = set("DE"), set("KRH")
STD = set("ACDEFGHIKLMNPQRSTVWY")
_MODIFIED = {'MSE':'M','SEP':'S','TPO':'T','PTR':'Y','HYP':'P','PCA':'E','CME':'C','CSO':'C',
             'KCX':'K','MLY':'K','LLP':'K','CSD':'C','OCS':'C','CAS':'C'}


def three_to_one(rn):
    try: return ProteinSequence.convert_letter_3to1(rn)
    except Exception: return _MODIFIED.get(rn, 'X')


# ---------------------------------------------------------------- surface + axis metrics
def load_backbone(pdb):
    """Return (native_seq, surface_idx set, core_idx set, res_letters). Surface = relative SASA >= RSA_CUT,
    core = relative SASA < RSA_CUT (Shrake-Rupley, ProtOr, Tien maxASA), on the isolated chain A."""
    arr = pdbio.get_structure(pdbio.PDBFile.read(str(pdb)), model=1)
    aa = arr[struc.filter_amino_acids(arr)]; aa = aa[aa.chain_id == "A"]
    sasa = struc.sasa(aa, vdw_radii="ProtOr")
    rs = struc.apply_residue_wise(aa, sasa, np.nansum)
    starts = struc.get_residue_starts(aa)
    letters = [three_to_one(aa.res_name[s]) for s in starts]
    rsa = [(s / MAXASA[L]) if (L in MAXASA and not np.isnan(s)) else np.nan for L, s in zip(letters, rs)]
    surf = set(i for i, v in enumerate(rsa) if not np.isnan(v) and v >= RSA_CUT)
    core = set(i for i, v in enumerate(rsa) if not np.isnan(v) and v < RSA_CUT)
    return "".join(letters), surf, core, letters


def axis_features(seq, surface_idx, core_idx=None):
    """Locked axis metrics (unchanged) + optional surface-vs-core and multi-pH extras."""
    n = len(seq)
    clean = "".join(c for c in seq if c in STD)
    pa = ProteinAnalysis(clean) if clean else None
    surf = [seq[i] for i in surface_idx if i < n and seq[i] in STD]
    ns = len(surf); sc = {a: surf.count(a) for a in set(surf)}
    sg = lambda Sset: sum(sc.get(a, 0) for a in Sset)
    bc = {a: seq.count(a) for a in set(seq)}
    KR = bc.get("K", 0) + bc.get("R", 0); sKR = sc.get("K", 0) + sc.get("R", 0)
    out = {
        "pI": pa.isoelectric_point() if pa else np.nan,
        "net_charge_per_residue": (pa.charge_at_pH(7.0) / n) if (pa and n) else np.nan,
        "KtoKR": bc.get("K", 0) / KR if KR else np.nan,
        "surface_net": (sg(BASIC) - sg(ACIDIC)) / ns if ns else np.nan,
        "surface_lys": sc.get("K", 0) / ns if ns else np.nan,
        "surface_KtoKR": sc.get("K", 0) / sKR if sKR else np.nan,
        "surface_acidic": sg(ACIDIC) / ns if ns else np.nan,
        "n_surface": ns,
        "gravy": pa.gravy() if pa else np.nan,
        "aromaticity": pa.aromaticity() if pa else np.nan,
        "charge_pH5": (pa.charge_at_pH(5.0) / n) if (pa and n) else np.nan,
        "charge_pH9": (pa.charge_at_pH(9.0) / n) if (pa and n) else np.nan,
    }
    if core_idx is not None:
        cores = [seq[i] for i in core_idx if i < n and seq[i] in STD]
        nc = len(cores); cc = {a: cores.count(a) for a in set(cores)}
        cg = lambda Sset: sum(cc.get(a, 0) for a in Sset)
        out["core_net"] = (cg(BASIC) - cg(ACIDIC)) / nc if nc else np.nan
        out["core_acidic"] = cg(ACIDIC) / nc if nc else np.nan
        out["core_lys"] = cc.get("K", 0) / nc if nc else np.nan
    return out


AXIS = ["pI", "net_charge_per_residue", "surface_net", "surface_lys", "KtoKR"]
OFF_AXIS = ["gravy", "aromaticity"]
EXTRA = ["surface_acidic", "core_net", "core_acidic", "core_lys", "charge_pH5", "charge_pH9"]
DIRECTION = {k: -1 for k in AXIS}   # alkaliphile = lower on every axis metric (set per cohort below)

# ---- cohort switch (alkaline vs acid): selects data files + flips the axis direction --------------
_COHORTS = {
    "alkaline": dict(label="alkaliphile", cases="alkaliphile_cases_stageD.csv",
                     controls="matched_neutralophile_controls_stageD.csv",
                     gap="alkaliphile_natural_gap_test.csv", struct="structures_alkaliphile", direction=-1),
    "acid":     dict(label="acidophile", cases="acidophile_cases_stageD.csv",
                     controls="acidophile_neutral_controls_stageD.csv",
                     gap="acidophile_natural_gap_test.csv", struct="structures_acidophile", direction=+1),
}
_COHORT = "alkaline"


def set_cohort(key):
    """Switch the cohort the eval reads (alkaline | acid). Flips DIRECTION (acidophile = higher = +1)."""
    global _COHORT, DIRECTION
    _COHORT = key
    DIRECTION = {m: _COHORTS[key]["direction"] for m in AXIS}


def _cf(k): return _COHORTS[_COHORT][k]


def _mean_pairwise_diff(seqs):
    """Mean fraction of positions differing across the n designs (collapse audit: low => collapsed)."""
    import itertools
    if len(seqs) < 2: return np.nan
    ds = [np.mean([a[i] != b[i] for i in range(min(len(a), len(b)))]) for a, b in itertools.combinations(seqs, 2)]
    return float(np.nanmean(ds)) if ds else np.nan


def _aa_entropy(seqs):
    """Shannon entropy (bits) of the pooled 20-aa composition across designs (collapse audit)."""
    from math import log2
    pool = "".join(seqs); cnt = {a: pool.count(a) for a in set(pool) if a in STD}; tot = sum(cnt.values())
    return float(-sum((c / tot) * log2(c / tot) for c in cnt.values())) if tot else np.nan


def recovery(designed, native):
    m = min(len(designed), len(native))
    return (sum(a == b for a, b in zip(designed[:m], native[:m])) / m) if m else np.nan


def natural_gap():
    """neu_native - case_native per metric (the denominator for the magnitude test)."""
    return pd.read_csv(DATA / _cf("gap"), index_col=0).iloc[:, 0].to_dict()


# ---------------------------------------------------------------- ProteinMPNN helpers
def load_model(path, device):
    from protein_mpnn_utils import ProteinMPNN
    ckpt = torch.load(path, map_location=device)
    m = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                    num_encoder_layers=3, num_decoder_layers=3,
                    k_neighbors=ckpt["num_edges"], augment_eps=0.0)
    m.to(device); m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m


@torch.no_grad()
def design_backbone(model, pdb, n, temp, device, bias_aa=None):
    """n designed sequences for a backbone + (native_seq, argmax recovery, native NLL).
    @no_grad so it is safe from any caller. bias_aa = optional [21] logit bias (hand-coded-bias baseline)."""
    from protein_mpnn_utils import parse_PDB, StructureDatasetPDB, tied_featurize, _scores, _S_to_seq
    ds = StructureDatasetPDB(parse_PDB(str(pdb)), max_length=20000)
    batch = [ds[0]]
    (X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, omit_AA_mask,
     residue_idx, dihedral_mask, _, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all,
     tied_beta) = tied_featurize(batch, device, None, None, None, None, None, None)
    omit_AAs_np = np.zeros(21, np.float32)
    bias_AAs_np = np.zeros(21, np.float32) if bias_aa is None else np.asarray(bias_aa, np.float32)
    pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()
    seqs = []
    for _ in range(n):
        randn = torch.randn(chain_M.shape, device=device)
        out = model.sample(X, randn, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
                           temperature=temp, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np,
                           chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                           pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=0,
                           pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=0,
                           bias_by_res=bias_by_res_all)
        seqs.append(_S_to_seq(out["S"][0], chain_M[0]))
    log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all,
                      torch.randn(chain_M.shape, device=device))
    mfl = mask * chain_M * chain_M_pos
    nll = _scores(S, log_probs, mfl).cpu().numpy()[0]
    native = _S_to_seq(S[0], chain_M[0])
    am = _S_to_seq(log_probs.argmax(-1)[0], chain_M[0])
    return seqs, native, recovery(am, native), float(nll)


def collect_backbones(split="test"):
    ca = pd.read_csv(DATA / _cf("cases"))
    co = pd.read_csv(DATA / _cf("controls"))
    struct = DATA / _cf("struct")
    bb = []
    for df, sub, grp in [(co, "controls", "neutralophile"), (ca, "cases", _cf("label"))]:
        for _, r in df[(df.qc_pass == True) & (df.split == split)].iterrows():
            p = struct / sub / f"{r.acc}.pdb"
            if p.exists(): bb.append((r.acc, grp, p))
    return bb


def per_backbone(model, pdb, n, temp, device, bias_aa=None):
    seqs, native, rec, nll = design_backbone(model, pdb, n, temp, device, bias_aa=bias_aa)
    _, surf, core, _ = load_backbone(pdb)
    feats = [axis_features(s, surf, core) for s in seqs]
    agg = {k: float(np.nanmean([f.get(k, np.nan) for f in feats])) for k in AXIS + OFF_AXIS + EXTRA}
    agg["recovery"] = rec; agg["nll"] = nll
    agg["diversity"] = _mean_pairwise_diff(seqs); agg["aa_entropy"] = _aa_entropy(seqs)
    return agg


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")
