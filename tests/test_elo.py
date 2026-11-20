"""Species Elo calculation, aggregation, confidence controls and figure data."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from decoding_bias.analysis.elo import compute_model, residualize_on_plddt
from decoding_bias.analysis.figure_data import elo_panels

ROOT = Path(__file__).resolve().parents[1]


class TestElo(unittest.TestCase):
    def test_species_bootstrap_matches_original_unweighted_plot(self):
        ratings = pd.read_csv(ROOT / 'expected/elo/all_models_species_ratings_long.csv')
        tax = pd.read_csv(ROOT / 'data/taxonomy.csv')
        gaps, species, matrix, meta, z = elo_panels(ratings, tax)
        expected = pd.read_csv(ROOT / 'expected/taxonomy/figure2_gap_bootstrap_reference.csv')
        actual = gaps.merge(expected, on='label', suffixes=('_new', '_original'), validate='one_to_one')
        self.assertEqual(len(actual), 14)
        for col in ['gap', 'ci_lower', 'ci_upper']:
            np.testing.assert_allclose(actual[col+'_new'], actual[col+'_original'], atol=1e-9, rtol=1e-12)
        self.assertEqual(species.groupby('model').size().tolist(), [495, 495])

    def test_species_standardisation_precedes_phylum_averaging(self):
        ratings = pd.DataFrame({'species': list('ABCD'), 'domain': ['Archaea']*2+['Eukaryota']*2,
                                'model': ['proteinmpnn_score']*4, 'rating': [0., 2., 8., 10.]})
        taxonomy = pd.DataFrame({'species': list('ABCD'), 'phylum_division': ['P', 'P', 'Q', 'Q']})
        _, _, matrix, _, z = elo_panels(ratings, taxonomy, n_boot=20)
        self.assertAlmostEqual(matrix.loc['P', 'proteinmpnn_score'], -4/np.sqrt(17))
        self.assertAlmostEqual(matrix.loc['Q', 'proteinmpnn_score'], 4/np.sqrt(17))
        self.assertAlmostEqual(z.groupby('model').species_z.mean().iloc[0], 0.)

    def test_confidence_weighting_changes_updates_without_changing_primary_default(self):
        d = pd.DataFrame({'protein_family': ['F', 'F'], 'species': ['a', 'b'],
                          'score': [2., 0.], 'avg_plddt': [50., 50.]})
        primary = compute_model(d, 'score', n_permutations=2)
        weighted = compute_model(d, 'score', n_permutations=2, plddt_weighting=True)
        self.assertEqual(primary['a']['rating'], 1516.)
        self.assertEqual(primary['b']['rating'], 1484.)
        self.assertEqual(weighted['a']['rating'], 1504.)
        self.assertEqual(weighted['b']['rating'], 1496.)

    def test_residualisation_removes_the_linear_confidence_component(self):
        x = np.arange(12., dtype=float)
        d = pd.DataFrame({'avg_plddt': x, 'score': 2*x+3})
        got = residualize_on_plddt(d, ['score'])
        np.testing.assert_allclose(got.score, 0., atol=1e-12)
        np.testing.assert_array_equal(d.score, 2*x+3)
