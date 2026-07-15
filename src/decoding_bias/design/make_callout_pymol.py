#!/usr/bin/env python3
"""Generate presentation-ready PyMOL callout figures for the fine-tuned
AlkSecMPNN / AcidSecMPNN secreted-protein redesigns.

Pure stdlib. Does NOT import pymol; it emits .pml scripts to run in PyMOL.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # design/ -> repo root
TARGETS = ["P02819", "P0A7R6", "Q4R312", "A9CIG1"]
MODELS = {"AlkSecMPNN": "AlkSec", "AcidSecMPNN": "AcidSec"}
BASE_MODELS = {"ProteinMPNN_v002": "ProteinMPNN"}
MODELS_020 = {"AlkSecMPNN_020": "AlkSecMPNN",
              "AcidSecMPNN_020": "AcidSecMPNN"}
BASE_MODELS_020 = {"ProteinMPNN_v020(base)": "ProteinMPNN"}
SOURCE_KIND_BY_MODEL = {"ProteinMPNN_v020(base)": "ProteinMPNN"}
SC_CSV = REPO / "paper_code/09_model_diagnostics/outputs/ft_self_consistency.csv"
SC_CSV_020 = (
    REPO / "paper_code/09_model_diagnostics/outputs/"
    "ft020_self_consistency_vs_afdb_with_base.csv"
)
FASTA = REPO / "design/outputs/ft_to_fold.fasta"
FASTA_ALL = REPO / "design/outputs/all_to_fold.fasta"
FASTA_020 = REPO / "design/outputs/ft020_to_fold.fasta"
FOLD_DIR = REPO / "design/outputs/colabfold_out_ft"
FOLD_DIR_020 = REPO / "design/outputs/colabfold_out_ft020"
BASE_FOLD_DIR_020 = REPO / "design/arc_downloads/rank001_flat"
DEFAULT_OUTDIR = REPO / "figures/design_callouts"
DEFAULT_RENDER_WIDTH = 1600
DEFAULT_RENDER_HEIGHT = 1000
DEFAULT_RENDER_DPI = 300
SURFACE_OVER_SPHERES_COLOR = "grey90"
SURFACE_OVER_SPHERES_TRANSPARENCY = 0.70

POS = set("KR")
NEG = set("DE")


def residue_charge(aa: str) -> int:
    if aa in POS:
        return 1
    if aa in NEG:
        return -1
    return 0


def net_charge(seq: str) -> int:
    return sum(residue_charge(a) for a in seq)


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def parse_fasta(path) -> dict:
    seqs, header = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                header = line[1:]
                seqs[header] = []
            elif header is not None:
                seqs[header].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def parse_fastas(paths) -> dict:
    seqs = {}
    for path in paths:
        seqs.update(parse_fasta(path))
    return seqs


def dedupe_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def callout_models(include_base_proteinmpnn=False, checkpoint="v002") -> dict:
    models = {}
    if checkpoint == "v020":
        if include_base_proteinmpnn:
            models.update(BASE_MODELS_020)
        models.update(MODELS_020)
    else:
        if include_base_proteinmpnn:
            models.update(BASE_MODELS)
        models.update(MODELS)
    return models


def data_sources(checkpoint="v002"):
    if checkpoint == "v020":
        return SC_CSV_020, [FASTA_ALL, FASTA_020], [BASE_FOLD_DIR_020, FOLD_DIR_020]
    return SC_CSV, [FASTA], [FOLD_DIR]


def source_kind(model: str) -> str:
    return SOURCE_KIND_BY_MODEL.get(model, model)


def classify_mutations(wt: str, des: str) -> list:
    """Mutated positions (1-based). delta = charge(des) - charge(wt)."""
    muts = []
    for i, (w, d) in enumerate(zip(wt, des), start=1):
        if w != d:
            muts.append({"resi": i, "wt": w, "des": d,
                         "delta": residue_charge(d) - residue_charge(w)})
    return muts


def split_charge_residues(muts):
    neg = [m["resi"] for m in muts if m["delta"] < 0]
    pos = [m["resi"] for m in muts if m["delta"] > 0]
    other = [m["resi"] for m in muts if m["delta"] == 0]
    return neg, pos, other


def select_best_designs(sc_csv, targets, models) -> dict:
    with open(sc_csv) as fh:
        rows = list(csv.DictReader(fh))
    best = {}
    for uid in targets:
        for model in models:
            cand = [r for r in rows
                    if r["uniprot_id"] == uid and r["model"] == model]
            if not cand:
                continue
            b = max(cand, key=lambda r: float(r["scTM"]))
            best[(uid, model)] = (int(b["sample_idx"]), float(b["scTM"]),
                                  float(b["scRMSD"]), float(b["pLDDT"]))
    return best


def discover_targets(sc_csv, models) -> list:
    with open(sc_csv) as fh:
        rows = csv.DictReader(fh)
        return sorted({r["uniprot_id"] for r in rows if r["model"] in models})


def find_fold(uid, kind, sample_idx=None, fold_dirs=None):
    fold_dirs = fold_dirs or [FOLD_DIR]
    for fold_dir in fold_dirs:
        if kind == "WT":
            pat = fold_dir / f"{uid}__WT_*rank_001*.pdb"
        else:
            pat = fold_dir / f"{uid}__{kind}__s{sample_idx}_*rank_001*.pdb"
        hits = sorted(glob.glob(str(pat)))
        if hits:
            return hits[0]
    return None


def gather(uid, best, seqs, models, fold_dirs=None) -> dict:
    wt_seq = seqs[f"{uid}__WT"]
    entry = {"uid": uid, "wt_seq": wt_seq,
             "wt_fold": find_fold(uid, "WT", fold_dirs=fold_dirs), "designs": {}}
    for model, short in models.items():
        if (uid, model) not in best:
            print(f"  [warn] no self-consistency row for {uid} {model}")
            continue
        idx = best[(uid, model)][0]
        kind = source_kind(model)
        key = f"{uid}__{kind}__s{idx}"
        des_seq = seqs.get(key, "")
        if len(des_seq) != len(wt_seq):
            print(f"  [warn] length mismatch {uid} {short} "
                  f"(wt={len(wt_seq)} des={len(des_seq)}); skipping muts")
            muts = []
        else:
            muts = classify_mutations(wt_seq, des_seq)
        entry["designs"][short] = {
            "model": model, "sample_idx": idx, "seq": des_seq, "muts": muts,
            "source_kind": kind,
            "scTM": best[(uid, model)][1],
            "fold": find_fold(uid, kind, idx, fold_dirs=fold_dirs)}
    return entry


def copy_structures(entry, outdir) -> None:
    uid = entry["uid"]
    pdbs = Path(outdir) / "pdbs"
    pdbs.mkdir(parents=True, exist_ok=True)
    if entry["wt_fold"]:
        shutil.copy(entry["wt_fold"], pdbs / f"{uid}_WT.pdb")
    else:
        print(f"  [warn] no WT fold for {uid}")
    for short, d in entry["designs"].items():
        if d["fold"]:
            shutil.copy(d["fold"], pdbs / f"{uid}_{short}.pdb")
        else:
            print(f"  [warn] no fold for {uid} {short}")


def write_mutations_summary(entries, outdir) -> Path:
    path = Path(outdir) / "mutations_summary.csv"
    cols = ["uniprot_id", "variant", "model", "sample_idx", "scTM",
            "n_mut", "n_more_neg", "n_more_pos", "n_other_mut",
            "wt_net_charge", "variant_net_charge", "delta_net_charge"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in entries:
            wt_nc = net_charge(e["wt_seq"])
            for short, d in e["designs"].items():
                neg, pos, other = split_charge_residues(d["muts"])
                vnc = net_charge(d["seq"])
                w.writerow({
                    "uniprot_id": e["uid"], "variant": short, "model": d["model"],
                    "sample_idx": d["sample_idx"], "scTM": round(d["scTM"], 3),
                    "n_mut": len(d["muts"]), "n_more_neg": len(neg),
                    "n_more_pos": len(pos), "n_other_mut": len(other),
                    "wt_net_charge": wt_nc, "variant_net_charge": vnc,
                    "delta_net_charge": vnc - wt_nc})
    return path


def apbs_tools():
    pdb2pqr = shutil.which("pdb2pqr30") or shutil.which("pdb2pqr")
    apbs = shutil.which("apbs")
    return pdb2pqr, apbs


def generate_apbs_maps(entry, outdir) -> bool:
    pdb2pqr, apbs = apbs_tools()
    uid = entry["uid"]
    if not (pdb2pqr and apbs):
        print(f"  [apbs] pdb2pqr/apbs not on PATH; skipping APBS for {uid}")
        return False
    apbs_dir = Path(outdir) / "apbs"
    apbs_dir.mkdir(parents=True, exist_ok=True)
    objs = ["WT"] if entry["wt_fold"] else []
    objs += [s for s, d in entry["designs"].items() if d["fold"]]
    ok = bool(objs)
    for o in objs:
        pdb = Path(outdir) / "pdbs" / f"{uid}_{o}.pdb"
        pqr = apbs_dir / f"{uid}_{o}.pqr"
        inp = apbs_dir / f"{uid}_{o}.in"
        try:
            subprocess.run(
                [pdb2pqr, "--ff=AMBER", f"--apbs-input={inp}",
                 str(pdb), str(pqr)],
                check=True, cwd=apbs_dir, capture_output=True, text=True)
            before = set(apbs_dir.glob("*.dx"))
            subprocess.run([apbs, inp.name], check=True, cwd=apbs_dir,
                           capture_output=True, text=True)
            new = sorted(set(apbs_dir.glob("*.dx")) - before,
                         key=os.path.getmtime)
            if new:
                new[-1].rename(apbs_dir / f"{uid}_{o}.dx")
            else:
                ok = False
        except (subprocess.CalledProcessError, OSError) as exc:
            msg = getattr(exc, "stderr", "") or str(exc)
            print(f"  [apbs] failed for {uid} {o}: {str(msg)[-200:]}")
            ok = False
    return ok


def _resi_sel(resis):
    return "+".join(str(r) for r in resis)


def _sphere_groups(muts, sphere_color, all_muts):
    """Yield (pymol_color, [resi,...]) groups for the spheres scene.

    sphere_color="direction": red=more acidic, blue=more basic, yellow=other.
    sphere_color="charge":    color by the resulting (design) residue charge,
                              red=D/E, blue=K/R, white=neutral.
    When all_muts is False, only charge-changing sites are shown.
    """
    neg, pos, other = split_charge_residues(muts)
    if sphere_color == "charge":
        shown = muts if all_muts else [m for m in muts if m["delta"] != 0]
        red_r = [m["resi"] for m in shown if residue_charge(m["des"]) < 0]
        blue_r = [m["resi"] for m in shown if residue_charge(m["des"]) > 0]
        white_r = [m["resi"] for m in shown if residue_charge(m["des"]) == 0]
        return [("red", red_r), ("marine", blue_r), ("white", white_r)]
    groups = [("red", neg), ("marine", pos)]
    if all_muts:
        groups.append(("yellow", other))
    return groups


def _write_mutation_spheres(a, entry, objs, all_muts, sphere_color):
    for short, d in entry["designs"].items():
        if not (d and short in objs):
            continue
        for color, resis in _sphere_groups(d["muts"], sphere_color, all_muts):
            if resis:
                a(f"show spheres, {short} and resi {_resi_sel(resis)}")
                a(f"color {color}, {short} and resi {_resi_sel(resis)}")


def write_pml(entry, outdir, all_muts=False, has_apbs=False,
              sphere_color="direction", surface_over_spheres=False,
              render_only_surface_over_spheres=False,
              render_width=DEFAULT_RENDER_WIDTH,
              render_height=DEFAULT_RENDER_HEIGHT,
              render_dpi=DEFAULT_RENDER_DPI) -> Path:
    if render_only_surface_over_spheres:
        surface_over_spheres = True
    outdir = Path(outdir)
    uid = entry["uid"]
    pdbs = outdir / "pdbs"
    renders = outdir / "renders"
    apbs = outdir / "apbs"
    L = []
    a = L.append

    a(f"# Auto-generated PyMOL callout for {uid}")
    a("reinitialize")
    a("bg_color white")
    a("set ray_opaque_background, 0")
    a("set ray_shadows, 1")
    a("set ray_shadow_decay_factor, 0.2")
    a("set antialias, 2")
    a("set cartoon_fancy_helices, 1")
    a("set cartoon_side_chain_helper, 1")
    a("set surface_quality, 1")
    a("set spec_reflect, 0.15")
    a("set sphere_scale, 0.7")
    a("")

    objs = []
    if entry["wt_fold"]:
        a(f"load {pdbs / (uid + '_WT.pdb')}, WT")
        objs.append("WT")
    for short, d in entry["designs"].items():
        if d and d["fold"]:
            a(f"load {pdbs / (uid + '_' + short + '.pdb')}, {short}")
            objs.append(short)
    a("")
    for short in entry["designs"]:
        if "WT" in objs and short in objs:
            a(f"cealign WT, {short}")
    a("")
    a("hide everything")
    a("show cartoon")
    a("color grey80")
    a("set grid_mode, 1")  # set after objects load so each gets its own cell
    a("orient")
    a("")

    # ----- scene: muts (mutations as sticks) -----
    a("# ===== scene: muts (charge-changing mutations, sticks) =====")
    a("hide everything")
    a("show cartoon")
    a("color grey80")
    for short, d in entry["designs"].items():
        if not (d and short in objs):
            continue
        neg, pos, other = split_charge_residues(d["muts"])
        if neg:
            a(f"show sticks, {short} and resi {_resi_sel(neg)} and not name N+C+O")
            a(f"color red, {short} and resi {_resi_sel(neg)}")
        if pos:
            a(f"show sticks, {short} and resi {_resi_sel(pos)} and not name N+C+O")
            a(f"color marine, {short} and resi {_resi_sel(pos)}")
        if all_muts and other:
            a(f"show sticks, {short} and resi {_resi_sel(other)} and not name N+C+O")
            a(f"color yellow, {short} and resi {_resi_sel(other)}")
    a("scene muts, store")
    a("")

    # ----- scene: muts_spheres (mutations as spheres) -----
    cmode = "resulting residue charge" if sphere_color == "charge" \
        else "change direction (red=more acidic, blue=more basic)"
    a(f"# ===== scene: muts_spheres (mutations as spheres; colored by {cmode}) =====")
    a("hide everything")
    a("show cartoon")
    a("color grey80")
    _write_mutation_spheres(a, entry, objs, all_muts, sphere_color)
    a("scene muts_spheres, store")
    a("")

    if surface_over_spheres:
        a("# ===== scene: muts_spheres_surface "
          "(cartoon + translucent surface over mutation spheres) =====")
        a("hide everything")
        a("show cartoon")
        a("color grey80")
        for o in objs:
            a(f"show surface, {o}")
            a(f"set transparency, {SURFACE_OVER_SPHERES_TRANSPARENCY:.2f}, {o}")
            a(f"set surface_color, {SURFACE_OVER_SPHERES_COLOR}, {o}")
        _write_mutation_spheres(a, entry, objs, all_muts, sphere_color)
        a("scene muts_spheres_surface, store")
        a("")

    # ----- scene: surf_residue -----
    a("# ===== scene: surf_residue (surface by residue charge) =====")
    a("hide everything")
    a("show surface")
    a("set transparency, 0")
    for o in objs:
        a(f"set transparency, 0, {o}")
        a(f"set surface_color, -1, {o}")
    a("color white")
    a("color red, resn ASP+GLU")
    a("color marine, resn LYS+ARG")
    a("scene surf_residue, store")
    a("")

    # ----- scene: surf_esp (vacuum electrostatics) -----
    # NOTE: util.protein_vacuum_esp manages its own objects and does NOT tile
    # under grid_mode, so this scene is for interactive viewing one object at a
    # time. It is intentionally skipped in headless mode and excluded from the
    # PNG render below.
    a("# ===== scene: surf_esp (PyMOL vacuum ESP; interactive, single object) =====")
    a("python")
    a("from pymol import cmd, invocation, util")
    a("if not invocation.options.no_gui:")
    a("    cmd.hide('everything')")
    for o in objs:
        a(f"    util.protein_vacuum_esp('{o}', mode=2, quiet=1)")
    a("    cmd.scene('surf_esp', 'store')")
    a("python end")
    a("")

    # render_scenes are auto-saved as PNGs in headless mode; they all tile
    # cleanly under grid_mode. surf_esp is deliberately omitted (see note above).
    if render_only_surface_over_spheres:
        render_scenes = ["muts_spheres_surface"]
    else:
        render_scenes = ["muts", "muts_spheres"]
        if surface_over_spheres:
            render_scenes.append("muts_spheres_surface")
        render_scenes.append("surf_residue")

    # ----- scene: surf_apbs (only if maps exist) -----
    if has_apbs:
        a("# ===== scene: surf_apbs (APBS potential) =====")
        a("hide everything")
        for o in objs:
            dx = apbs / f"{uid}_{o}.dx"
            a(f"load {dx}, {o}_pot")
            a(f"ramp_new {o}_ramp, {o}_pot, [-5, 0, 5], [red, white, blue]")
            a(f"show surface, {o}")
            a(f"set surface_color, {o}_ramp, {o}")
        a("scene surf_apbs, store")
        a("")
        render_scenes.append("surf_apbs")

    # ----- headless render helper -----
    surface_overlay_scene = (
        "muts_spheres_surface" if surface_over_spheres else None
    )
    a("python")
    a("from pymol import cmd, invocation")
    a("import os")
    a(f"_renders = r'{renders}'")
    a(f"_scenes = {render_scenes}")
    a(f"_surface_overlay_scene = {surface_overlay_scene!r}")
    a(f"_surface_overlay_objs = {objs}")
    a(f"_surface_overlay_color = '{SURFACE_OVER_SPHERES_COLOR}'")
    a(f"_surface_overlay_transparency = "
      f"{SURFACE_OVER_SPHERES_TRANSPARENCY:.2f}")
    a(f"_render_width = {render_width}")
    a(f"_render_height = {render_height}")
    a(f"_render_dpi = {render_dpi}")
    a(f"_uid = '{uid}'")
    a("def apply_scene_settings(sc):")
    a("    if sc == _surface_overlay_scene:")
    a("        for obj in _surface_overlay_objs:")
    a("            cmd.set('transparency', _surface_overlay_transparency, obj)")
    a("            cmd.set('surface_color', _surface_overlay_color, obj)")
    a("    elif sc == 'surf_residue':")
    a("        for obj in _surface_overlay_objs:")
    a("            cmd.set('transparency', 0, obj)")
    a("            cmd.set('surface_color', -1, obj)")
    a("def save_pngs():")
    a("    os.makedirs(_renders, exist_ok=True)")
    a("    for sc in _scenes:")
    a("        cmd.scene(sc, 'recall')")
    a("        apply_scene_settings(sc)")
    a("        cmd.png(os.path.join(_renders, _uid + '_' + sc + '.png'),"
      " width=_render_width, height=_render_height, dpi=_render_dpi, ray=1)")
    a("if invocation.options.no_gui:")
    a("    save_pngs()")
    a("python end")
    a("")
    a("scene muts, recall")

    pml_dir = outdir / "pml"
    pml_dir.mkdir(parents=True, exist_ok=True)
    path = pml_dir / f"{uid}.pml"
    path.write_text("\n".join(L) + "\n")
    return path


def write_master_pml(entries, outdir) -> Path:
    outdir = Path(outdir)
    pml_dir = outdir / "pml"
    L = ["# Master script: regenerate every callout figure.",
         "# Headless render of all PNGs:  pymol -cq pml/all_callouts.pml",
         "# Interactive (one protein):    @pml/P02819.pml", ""]
    for e in entries:
        L.append(f"@{pml_dir / (e['uid'] + '.pml')}")
    path = pml_dir / "all_callouts.pml"
    path.write_text("\n".join(L) + "\n")
    return path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--all-targets", action="store_true",
                   help="process every UniProt with AlkSecMPNN/AcidSecMPNN "
                        "self-consistency rows, not just the four curated "
                        "callout targets")
    p.add_argument("--targets", nargs="+",
                   help="explicit UniProt IDs to process; duplicates are ignored")
    p.add_argument("--include-base-proteinmpnn", action="store_true",
                   help="include the ProteinMPNN base-model design as a "
                        "fourth structure")
    p.add_argument("--checkpoint", choices=["v002", "v020"], default="v002",
                   help="which fine-tuning/base checkpoint family to use")
    p.add_argument("--all-muts", action="store_true",
                   help="highlight all mutations, not just charge-changing")
    p.add_argument("--sphere-color", choices=["direction", "charge"],
                   default="direction",
                   help="muts_spheres coloring: 'direction' (red=more acidic / "
                        "blue=more basic) or 'charge' (resulting residue charge: "
                        "red=D/E, blue=K/R, white=neutral)")
    p.add_argument("--surface-over-spheres", action="store_true",
                   help="also render a cartoon + translucent neutral surface "
                        "over the mutation spheres")
    p.add_argument("--only-surface-over-spheres", action="store_true",
                   help="render only the translucent surface-over-spheres scene")
    p.add_argument("--render-width", type=positive_int,
                   default=DEFAULT_RENDER_WIDTH,
                   help=f"PNG width in pixels (default: {DEFAULT_RENDER_WIDTH})")
    p.add_argument("--render-height", type=positive_int,
                   default=DEFAULT_RENDER_HEIGHT,
                   help=f"PNG height in pixels (default: {DEFAULT_RENDER_HEIGHT})")
    p.add_argument("--render-dpi", type=positive_int,
                   default=DEFAULT_RENDER_DPI,
                   help=f"PNG DPI metadata (default: {DEFAULT_RENDER_DPI})")
    p.add_argument("--apbs", action="store_true",
                   help="attempt APBS maps (needs pdb2pqr + apbs on PATH)")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    outdir = Path(args.outdir)
    (outdir / "pdbs").mkdir(parents=True, exist_ok=True)
    (outdir / "pml").mkdir(parents=True, exist_ok=True)

    sc_csv, fasta_paths, fold_dirs = data_sources(args.checkpoint)
    seqs = parse_fastas(fasta_paths)
    models = callout_models(args.include_base_proteinmpnn, args.checkpoint)
    if args.targets:
        targets = dedupe_preserve_order(args.targets)
    elif args.all_targets:
        targets = discover_targets(sc_csv, set(models))
    else:
        targets = TARGETS
    best = select_best_designs(sc_csv, targets, list(models))

    entries = []
    for uid in targets:
        entry = gather(uid, best, seqs, models, fold_dirs=fold_dirs)
        copy_structures(entry, outdir)
        entries.append(entry)

    summary = write_mutations_summary(entries, outdir)
    print("Wrote", summary)

    for entry in entries:
        has_apbs = generate_apbs_maps(entry, outdir) if args.apbs else False
        write_pml(entry, outdir, all_muts=args.all_muts, has_apbs=has_apbs,
                  sphere_color=args.sphere_color,
                  surface_over_spheres=args.surface_over_spheres,
                  render_only_surface_over_spheres=args.only_surface_over_spheres,
                  render_width=args.render_width,
                  render_height=args.render_height,
                  render_dpi=args.render_dpi)
    write_master_pml(entries, outdir)
    print("Done. Outputs in", outdir)


if __name__ == "__main__":
    main()
