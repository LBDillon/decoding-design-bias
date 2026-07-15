import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # design/
import make_callout_pymol as m


def test_net_charge():
    assert m.net_charge("KRDE") == 0
    assert m.net_charge("KKK") == 3
    assert m.net_charge("DDD") == -3
    assert m.net_charge("AAAG") == 0


def test_parse_fasta(tmp_path):
    f = tmp_path / "t.fa"
    f.write_text(">a\nMKT\nED\n>b\nGG\n")
    assert m.parse_fasta(f) == {"a": "MKTED", "b": "GG"}


def test_dedupe_preserve_order():
    assert m.dedupe_preserve_order(["O67786", "Q4R312", "Q4R312", "A9CIG1"]) == [
        "O67786",
        "Q4R312",
        "A9CIG1",
    ]


def test_callout_models_with_base_proteinmpnn():
    models = m.callout_models(include_base_proteinmpnn=True)
    assert list(models.items()) == [
        ("ProteinMPNN_v002", "ProteinMPNN"),
        ("AlkSecMPNN", "AlkSec"),
        ("AcidSecMPNN", "AcidSec"),
    ]


def test_callout_models_v020_with_base_proteinmpnn():
    models = m.callout_models(
        include_base_proteinmpnn=True,
        checkpoint="v020",
    )
    assert list(models.items()) == [
        ("ProteinMPNN_v020(base)", "ProteinMPNN"),
        ("AlkSecMPNN_020", "AlkSecMPNN"),
        ("AcidSecMPNN_020", "AcidSecMPNN"),
    ]


def test_source_kind_maps_v020_base_to_plain_proteinmpnn():
    assert m.source_kind("ProteinMPNN_v020(base)") == "ProteinMPNN"
    assert m.source_kind("AlkSecMPNN_020") == "AlkSecMPNN_020"


def test_classify_added_negative():
    muts = m.classify_mutations("AAA", "ADA")
    assert len(muts) == 1
    assert muts[0]["resi"] == 2 and muts[0]["delta"] == -1


def test_classify_charge_swap_K_to_D():
    muts = m.classify_mutations("KAA", "DAA")
    assert muts[0]["delta"] == -2


def test_split_charge_residues():
    muts = m.classify_mutations("AKA", "DDA")  # pos1 A->D(-1), pos2 K->D(-2)
    neg, pos, other = m.split_charge_residues(muts)
    assert neg == [1, 2] and pos == [] and other == []


def test_split_other_mutation():
    muts = m.classify_mutations("AAA", "GAA")  # A->G, delta 0
    neg, pos, other = m.split_charge_residues(muts)
    assert neg == [] and pos == [] and other == [1]


def test_select_best_designs(tmp_path):
    csvf = tmp_path / "sc.csv"
    csvf.write_text(
        "uniprot_id,model,sample_idx,scTM,scRMSD,pLDDT\n"
        "P1,AlkSecMPNN,0,0.5,2.0,80\n"
        "P1,AlkSecMPNN,1,0.9,1.0,85\n"
        "P1,AcidSecMPNN,2,0.7,1.5,82\n"
    )
    best = m.select_best_designs(csvf, ["P1"], ["AlkSecMPNN", "AcidSecMPNN"])
    assert best[("P1", "AlkSecMPNN")][0] == 1
    assert best[("P1", "AcidSecMPNN")][0] == 2


def test_discover_targets_for_models(tmp_path):
    csvf = tmp_path / "sc.csv"
    csvf.write_text(
        "uniprot_id,model,sample_idx,scTM,scRMSD,pLDDT\n"
        "P2,AlkSecMPNN,0,0.5,2.0,80\n"
        "P1,AcidSecMPNN,2,0.7,1.5,82\n"
        "P3,ProteinMPNN_v002,0,0.6,1.8,81\n"
    )

    assert m.discover_targets(csvf, {"AlkSecMPNN", "AcidSecMPNN"}) == ["P1", "P2"]


def test_sphere_groups_direction():
    muts = m.classify_mutations("AKA", "DDA")  # A->D(-1), K->D(-2): both more neg
    groups = dict(m._sphere_groups(muts, "direction", all_muts=False))
    assert groups["red"] == [1, 2]
    assert groups["marine"] == []


def test_sphere_groups_charge_by_resulting_residue():
    # pos2 D->K: charge-changing, resulting residue K is basic -> blue (marine)
    muts = m.classify_mutations("ADC", "AKC")
    groups = dict(m._sphere_groups(muts, "charge", all_muts=False))
    assert groups["marine"] == [2] and groups["red"] == [] and groups["white"] == []


def test_sphere_groups_charge_neutral_result():
    muts = m.classify_mutations("DA", "AA")  # pos1 D->A: charge-changing, result neutral
    groups = dict(m._sphere_groups(muts, "charge", all_muts=False))
    assert groups["white"] == [1] and groups["red"] == [] and groups["marine"] == []


def test_write_mutations_summary(tmp_path):
    import csv as _csv
    entry = {
        "uid": "P1",
        "wt_seq": "AAKA",                      # net charge +1
        "wt_fold": None,
        "designs": {
            "AlkSec": {"model": "AlkSecMPNN", "sample_idx": 3,
                       "seq": "ADKA",          # pos2 A->D: more negative
                       "muts": m.classify_mutations("AAKA", "ADKA"),
                       "scTM": 0.8, "fold": None},
        },
    }
    out = tmp_path / "out"
    out.mkdir()
    path = m.write_mutations_summary([entry], out)
    rows = list(_csv.DictReader(open(path)))
    assert len(rows) == 1
    r = rows[0]
    assert r["uniprot_id"] == "P1" and r["variant"] == "AlkSec"
    assert int(r["n_more_neg"]) == 1 and int(r["n_more_pos"]) == 0
    assert int(r["wt_net_charge"]) == 1
    assert int(r["variant_net_charge"]) == 0
    assert int(r["delta_net_charge"]) == -1


def test_gather_v020_base_uses_plain_proteinmpnn_files(tmp_path):
    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()
    wt = fold_dir / "P1__WT_unrelaxed_rank_001_model.pdb"
    base = fold_dir / "P1__ProteinMPNN__s2_unrelaxed_rank_001_model.pdb"
    wt.write_text("")
    base.write_text("")

    seqs = {
        "P1__WT": "AKA",
        "P1__ProteinMPNN__s2": "ADA",
    }
    best = {("P1", "ProteinMPNN_v020(base)"): (2, 0.91, 1.2, 88.0)}
    entry = m.gather(
        "P1",
        best,
        seqs,
        {"ProteinMPNN_v020(base)": "ProteinMPNN"},
        fold_dirs=[fold_dir],
    )

    d = entry["designs"]["ProteinMPNN"]
    assert d["model"] == "ProteinMPNN_v020(base)"
    assert d["source_kind"] == "ProteinMPNN"
    assert d["seq"] == "ADA"
    assert d["fold"] == str(base)


def test_surface_over_spheres_flag_is_parsed():
    args = m.build_argparser().parse_args(["--surface-over-spheres"])
    assert args.surface_over_spheres is True


def test_all_targets_flag_is_parsed():
    args = m.build_argparser().parse_args(["--all-targets"])
    assert args.all_targets is True


def test_targets_and_base_model_flags_are_parsed():
    args = m.build_argparser().parse_args([
        "--targets",
        "O67786",
        "Q4R312",
        "Q4R312",
        "--include-base-proteinmpnn",
        "--only-surface-over-spheres",
    ])

    assert args.targets == ["O67786", "Q4R312", "Q4R312"]
    assert args.include_base_proteinmpnn is True
    assert args.only_surface_over_spheres is True


def test_render_size_args_are_parsed():
    args = m.build_argparser().parse_args([
        "--render-width", "3200",
        "--render-height", "2000",
        "--render-dpi", "600",
    ])

    assert args.render_width == 3200
    assert args.render_height == 2000
    assert args.render_dpi == 600


def test_write_pml_surface_over_spheres_scene(tmp_path):
    entry = {
        "uid": "P1",
        "wt_fold": "wt.pdb",
        "designs": {
            "AlkSec": {
                "fold": "alk.pdb",
                "muts": m.classify_mutations("AKA", "DDA"),
            },
            "AcidSec": {
                "fold": "acid.pdb",
                "muts": m.classify_mutations("ADA", "AKA"),
            },
        },
    }

    path = m.write_pml(
        entry,
        tmp_path,
        surface_over_spheres=True,
        render_width=3200,
        render_height=2000,
        render_dpi=600,
    )
    text = path.read_text()

    assert "scene: muts_spheres_surface" in text
    assert "show surface, WT" in text
    assert "set transparency, 0.70, AlkSec" in text
    assert "set surface_color, grey90, AcidSec" in text
    assert "set surface_color, -1, AcidSec" in text
    assert "scene muts_spheres_surface, store" in text
    assert "if not invocation.options.no_gui:" in text
    assert "cmd.scene('surf_esp', 'store')" in text
    assert "_surface_overlay_scene = 'muts_spheres_surface'" in text
    assert "apply_scene_settings(sc)" in text
    assert "_render_width = 3200" in text
    assert "_render_height = 2000" in text
    assert "_render_dpi = 600" in text
    assert "width=_render_width, height=_render_height, dpi=_render_dpi" in text
    assert (
        "_scenes = ['muts', 'muts_spheres', "
        "'muts_spheres_surface', 'surf_residue']"
    ) in text


def test_write_pml_surface_over_spheres_is_opt_in(tmp_path):
    entry = {"uid": "P1", "wt_fold": None, "designs": {}}

    path = m.write_pml(entry, tmp_path)
    text = path.read_text()

    assert "muts_spheres_surface" not in text
    assert "_scenes = ['muts', 'muts_spheres', 'surf_residue']" in text


def test_write_pml_only_surface_over_spheres_scene_with_base(tmp_path):
    entry = {
        "uid": "P1",
        "wt_fold": "wt.pdb",
        "designs": {
            "ProteinMPNN": {
                "fold": "base.pdb",
                "muts": m.classify_mutations("AKA", "ADA"),
            },
            "AlkSec": {
                "fold": "alk.pdb",
                "muts": m.classify_mutations("AKA", "DDA"),
            },
            "AcidSec": {
                "fold": "acid.pdb",
                "muts": m.classify_mutations("ADA", "AKA"),
            },
        },
    }

    path = m.write_pml(
        entry,
        tmp_path,
        render_only_surface_over_spheres=True,
    )
    text = path.read_text()

    assert "load " in text and "P1_ProteinMPNN.pdb" in text
    assert "cealign WT, ProteinMPNN" in text
    assert "scene muts_spheres_surface, store" in text
    assert "_scenes = ['muts_spheres_surface']" in text
