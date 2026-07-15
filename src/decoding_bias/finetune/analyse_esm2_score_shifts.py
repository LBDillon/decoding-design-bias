#!/usr/bin/env python3
"""Analyse ESM2-35M continued-pretraining masked-marginal score shifts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_FEATURE_TABLE = "dataset_update/main_plus_r2_r3_analysis_v12_cli.csv"
FEATURES = [
    "acidic_residue_fraction",
    "basic_residue_fraction",
    "charge_per_residue",
    "isoelectric_point",
]
MODEL_COHORT = {
    "AlkSecESM35M": "alkaliphile secretome",
    "AcidSecESM35M": "acidophile secretome",
    "NeuSecESM35M_AlkMatched": "alkaliphile-matched neutralophile secretome",
    "NeuSecESM35M_AcidMatched": "acidophile-matched neutralophile secretome",
}
PROTEINMPNN_COLUMNS = {
    "AlkSecMPNN_020": "AlkSecMPNN_020_score",
    "AcidSecMPNN_020": "AcidSecMPNN_020_score",
    "AlkSecMPNN": "AlkSecMPNN_v2_score",
    "AcidSecMPNN": "AcidSecMPNN_score",
}


def parse_model_score(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--continued_score must be ModelName=/path/to/scores.csv")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--continued_score must be ModelName=/path/to/scores.csv")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_scores", required=True)
    parser.add_argument(
        "--continued_score",
        action="append",
        type=parse_model_score,
        required=True,
        help="Repeated ModelName=/path/to/score.csv argument.",
    )
    parser.add_argument(
        "--feature_table",
        default=DEFAULT_FEATURE_TABLE,
        help="Feature table with acid-base features, or a sequence table from which they can be computed.",
    )
    parser.add_argument("--out_dir", default="outputs/esm35m_continual_pretraining")
    parser.add_argument("--score_id_col", default="id")
    parser.add_argument("--feature_id_col", default="auto")
    parser.add_argument("--feature_seq_col", default="sequence")
    parser.add_argument("--proteinmpnn_score_table", default="", help="Defaults to feature_table when possible.")
    parser.add_argument("--proteinmpnn_base_col", default="ProteinMPNN_v020_score")
    return parser.parse_args()


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value) -> float:
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def choose_feature_id_col(rows: List[dict], requested: str) -> str:
    if requested != "auto":
        return requested
    if not rows:
        return "id"
    columns = set(rows[0])
    for candidate in ("id", "Entry", "acc", "name"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer feature ID column from columns: {sorted(columns)[:20]}")


def net_charge_at_ph(seq: str, ph: float) -> float:
    # ProtParam-like pKa set, enough for deterministic acid-base axis features.
    pka_pos = {"K": 10.5, "R": 12.4, "H": 6.0}
    pka_neg = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
    n_term = 9.69
    c_term = 2.34
    charge = 1.0 / (1.0 + 10.0 ** (ph - n_term))
    charge -= 1.0 / (1.0 + 10.0 ** (c_term - ph))
    for aa, pka in pka_pos.items():
        charge += seq.count(aa) / (1.0 + 10.0 ** (ph - pka))
    for aa, pka in pka_neg.items():
        charge -= seq.count(aa) / (1.0 + 10.0 ** (pka - ph))
    return charge


def isoelectric_point(seq: str) -> float:
    low, high = 0.0, 14.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if net_charge_at_ph(seq, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def add_features_from_sequence(rows: List[dict], seq_col: str) -> None:
    for row in rows:
        if all(row.get(feature) not in (None, "") for feature in FEATURES):
            continue
        seq = (row.get(seq_col) or "").strip().upper()
        n = len(seq)
        if not n:
            continue
        row["acidic_residue_fraction"] = row.get("acidic_residue_fraction") or (seq.count("D") + seq.count("E")) / n
        row["basic_residue_fraction"] = row.get("basic_residue_fraction") or (seq.count("K") + seq.count("R") + seq.count("H")) / n
        row["charge_per_residue"] = row.get("charge_per_residue") or net_charge_at_ph(seq, 7.0) / n
        row["isoelectric_point"] = row.get("isoelectric_point") or isoelectric_point(seq)


def read_scores(path: Path, id_col: str) -> Dict[str, float]:
    rows = read_csv(path)
    scores: Dict[str, float] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        score = to_float(row.get("esm_mlm_score"))
        if math.isfinite(score):
            scores[row[id_col]] = score
    return scores


def zscore(values):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr, ddof=0)
    if not math.isfinite(std) or std == 0:
        return arr * 0.0
    return (arr - mean) / std


def spearman(x, y) -> Tuple[float, float]:
    import numpy as np
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan, math.nan
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def acid_base_pca(feature_matrix):
    import numpy as np
    from sklearn.decomposition import PCA

    x = np.asarray(feature_matrix, dtype=float)
    xz = np.column_stack([zscore(x[:, i]) for i in range(x.shape[1])])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(xz)
    # Orient +PC1 as more acidic: more acidic fraction, lower basic/charge/pI.
    loading = pca.components_[0]
    acid_orientation = loading[0] - loading[1] - loading[2] - loading[3]
    if acid_orientation < 0:
        pcs[:, 0] *= -1.0
        pca.components_[0] *= -1.0
    return pcs, pca


def ols_pc_model(delta_scores, pcs) -> Dict[str, float]:
    import numpy as np
    from scipy.stats import t as t_dist

    y = zscore(delta_scores)
    x1 = zscore(pcs[:, 0])
    x2 = zscore(pcs[:, 1])
    X = np.column_stack([np.ones_like(y), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else math.nan
    df = len(y) - X.shape[1]
    pvals = [math.nan, math.nan, math.nan]
    if df > 0:
        sigma2 = ss_res / df
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = beta / se
        pvals = [float(2.0 * t_dist.sf(abs(t), df)) if math.isfinite(float(t)) else math.nan for t in tvals]
    return {
        "grad_PC1": float(beta[1]),
        "grad_PC2": float(beta[2]),
        "grad_PC1_pvalue": pvals[1],
        "grad_PC2_pvalue": pvals[2],
        "R2": r2,
    }


def assemble_analysis_rows(base_scores: Dict[str, float], continued_scores: Dict[str, float], feature_rows: Dict[str, dict]):
    import numpy as np

    ids = sorted(set(base_scores) & set(continued_scores) & set(feature_rows))
    rows = []
    for seq_id in ids:
        features = [to_float(feature_rows[seq_id].get(feature)) for feature in FEATURES]
        if not all(math.isfinite(value) for value in features):
            continue
        rows.append(
            {
                "id": seq_id,
                "delta_score": continued_scores[seq_id] - base_scores[seq_id],
                **{feature: value for feature, value in zip(FEATURES, features)},
            }
        )
    return rows


def analyse_delta(model: str, rows: List[dict]) -> Tuple[dict, List[dict]]:
    import numpy as np

    delta = np.asarray([row["delta_score"] for row in rows], dtype=float)
    feat = np.asarray([[row[feature] for feature in FEATURES] for row in rows], dtype=float)
    pcs, _ = acid_base_pca(feat)
    ols = ols_pc_model(delta, pcs)

    corr_map = {}
    direct_rows = []
    for feature in FEATURES:
        rho, p = spearman(delta, np.asarray([row[feature] for row in rows], dtype=float))
        suffix = {
            "acidic_residue_fraction": "acidic",
            "basic_residue_fraction": "basic",
            "charge_per_residue": "charge",
            "isoelectric_point": "pI",
        }[feature]
        corr_map[f"corr_{suffix}"] = rho
        corr_map[f"corr_{suffix}_pvalue"] = p
        direct_rows.append(
            {
                "model": model,
                "training_cohort": MODEL_COHORT.get(model, ""),
                "feature": feature,
                "spearman_rho": rho,
                "pvalue": p,
                "n": len(rows),
            }
        )

    summary = {
        "model": model,
        "training_cohort": MODEL_COHORT.get(model, ""),
        "n": len(rows),
        **ols,
        **corr_map,
    }
    return summary, direct_rows


def plot_summary(summary_rows: List[dict], direct_rows: List[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = [row["model"] for row in summary_rows]
    values = [float(row["grad_PC1"]) for row in summary_rows]
    colors = ["#b44d4d" if "Alk" in m else "#4d76b4" if "Acid" in m else "#7a7a7a" for m in models]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(models)), 4))
    ax.bar(models, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("gradient on acidic PC1")
    ax.set_title("Delta score = continued-pretrained ESM2-35M score - base ESM2-35M score")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig_ESM35M_dscore_acidbase_PC1.png", dpi=220)
    plt.close(fig)

    features = FEATURES
    matrix = []
    for model in models:
        model_rows = {row["feature"]: row for row in direct_rows if row["model"] == model}
        matrix.append([float(model_rows[feature]["spearman_rho"]) for feature in features])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.7 * len(models))))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(features)), labels=["acidic", "basic", "charge", "pI"])
    ax.set_yticks(range(len(models)), labels=models)
    ax.set_title("Spearman correlations with delta score")
    for i in range(len(models)):
        for j in range(len(features)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig_ESM35M_direct_correlations.png", dpi=220)
    plt.close(fig)


def proteinmpnn_comparison_rows(score_table: Path, base_col: str) -> List[dict]:
    rows = read_csv(score_table)
    if not rows or base_col not in rows[0]:
        return []
    feature_id_col = choose_feature_id_col(rows, "auto")
    add_features_from_sequence(rows, "sequence")
    by_id = {row[feature_id_col]: row for row in rows}
    out = []
    for model, ft_col in PROTEINMPNN_COLUMNS.items():
        if ft_col not in rows[0]:
            continue
        analysis_rows = []
        for row in rows:
            base = to_float(row.get(base_col))
            ft = to_float(row.get(ft_col))
            feats = [to_float(row.get(feature)) for feature in FEATURES]
            if math.isfinite(base) and math.isfinite(ft) and all(math.isfinite(v) for v in feats):
                analysis_rows.append(
                    {
                        "id": row[feature_id_col],
                        "delta_score": ft - base,
                        **{feature: value for feature, value in zip(FEATURES, feats)},
                    }
                )
        if len(analysis_rows) < 4:
            continue
        summary, _ = analyse_delta(model, analysis_rows)
        out.append(
            {
                "model_family": "ProteinMPNN",
                "model": model,
                "training_cohort": "alkaliphile/acidophile secretome",
                "readout": "WT sequence score shift",
                "grad_PC1": summary["grad_PC1"],
                "R2": summary["R2"],
                "corr_acidic": summary["corr_acidic"],
                "corr_charge": summary["corr_charge"],
                "corr_pI": summary["corr_pI"],
                "interpretation": "Also has a structure-conditioned design-generation endpoint; WT-score shift is secondary.",
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    feature_table = Path(args.feature_table)
    feature_rows_list = read_csv(feature_table)
    add_features_from_sequence(feature_rows_list, args.feature_seq_col)
    feature_id_col = choose_feature_id_col(feature_rows_list, args.feature_id_col)
    feature_rows = {row[feature_id_col]: row for row in feature_rows_list}
    print(f"Using feature table: {feature_table} (id column: {feature_id_col})")

    base_scores = read_scores(Path(args.base_scores), args.score_id_col)
    summary_rows = []
    direct_rows = []
    comparison_rows = []

    for model, score_path in args.continued_score:
        continued_scores = read_scores(score_path, args.score_id_col)
        rows = assemble_analysis_rows(base_scores, continued_scores, feature_rows)
        if len(rows) < 4:
            raise RuntimeError(f"Too few overlapping scored/feature rows for {model}: n={len(rows)}")
        summary, direct = analyse_delta(model, rows)
        summary_rows.append(summary)
        direct_rows.extend(direct)
        comparison_rows.append(
            {
                "model_family": "ESM2-35M",
                "model": model,
                "training_cohort": summary["training_cohort"],
                "readout": "WT sequence masked-marginal score shift",
                "grad_PC1": summary["grad_PC1"],
                "R2": summary["R2"],
                "corr_acidic": summary["corr_acidic"],
                "corr_charge": summary["corr_charge"],
                "corr_pI": summary["corr_pI"],
                "interpretation": "Sequence-only model; only WT-score reranking is assessed.",
            }
        )

    summary_fields = [
        "model",
        "training_cohort",
        "n",
        "grad_PC1",
        "grad_PC1_pvalue",
        "grad_PC2",
        "grad_PC2_pvalue",
        "R2",
        "corr_acidic",
        "corr_acidic_pvalue",
        "corr_basic",
        "corr_basic_pvalue",
        "corr_charge",
        "corr_charge_pvalue",
        "corr_pI",
        "corr_pI_pvalue",
    ]
    write_csv(table_dir / "Table_ESM35M_score_shift_summary.csv", summary_rows, summary_fields)
    write_csv(
        table_dir / "Table_ESM35M_direct_correlations.csv",
        direct_rows,
        ["model", "training_cohort", "feature", "spearman_rho", "pvalue", "n"],
    )

    proteinmpnn_table = Path(args.proteinmpnn_score_table) if args.proteinmpnn_score_table else feature_table
    comparison_rows.extend(proteinmpnn_comparison_rows(proteinmpnn_table, args.proteinmpnn_base_col))
    write_csv(
        table_dir / "Table_sequence_vs_structure_score_shift_comparison.csv",
        comparison_rows,
        [
            "model_family",
            "model",
            "training_cohort",
            "readout",
            "grad_PC1",
            "R2",
            "corr_acidic",
            "corr_charge",
            "corr_pI",
            "interpretation",
        ],
    )
    plot_summary(summary_rows, direct_rows, out_dir)
    print(f"Wrote analysis tables and figures under {out_dir}")


if __name__ == "__main__":
    main()
