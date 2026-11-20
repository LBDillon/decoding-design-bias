"""Complete GAM reference matching and supported prediction display."""
import unittest

import numpy as np
import pandas as pd

from decoding_bias.analysis.gam_display import display_layers
from decoding_bias.reproduce import _compare_gam_reference


class TestGam(unittest.TestCase):
    def test_missing_reference_model_fails_even_if_many_models_match(self):
        expected = pd.DataFrame({'model': [f'm{i}' for i in range(15)], 'n': 100,
                                 'gam_dev_explained_pct': range(15)})
        actual = expected.iloc[:-1].copy()
        ok, detail = _compare_gam_reference(expected, actual)
        self.assertFalse(ok)
        self.assertIn('m14', detail)

    def test_faint_predictions_do_not_become_supported_contours(self):
        grid = pd.DataFrame({'PC1': [0., 1., 0., 1.], 'PC2': [0., 0., 1., 1.],
                             'predicted_score': [-4., 0., 1., 8.], 'supported': [False, True, True, False]})
        x, y, z, alpha, contours = display_layers(grid, background_alpha=.22)
        np.testing.assert_array_equal(z, [[-4., 0.], [1., 8.]])
        np.testing.assert_allclose(alpha, [[.22, 1.], [1., .22]])
        np.testing.assert_array_equal(np.ma.getmaskarray(contours), [[True, False], [False, True]])
