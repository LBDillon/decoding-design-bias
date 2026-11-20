"""Display layers for GAM grids: opaque supported cells and faint unsupported cells."""
import numpy as np


def display_layers(table, background_alpha=.22):
    """Return x/y axes, unmodified predictions, opacity and supported contours.

    All predictions remain available as a faint background. Only cells marked
    supported are opaque and enter contour estimation/display.
    """
    x = np.sort(table.PC1.unique())
    y = np.sort(table.PC2.unique())
    ordered = table.sort_values(['PC2', 'PC1'])
    if len(ordered) != len(x) * len(y) or ordered.duplicated(['PC1', 'PC2']).any():
        raise ValueError('GAM display requires one prediction per rectangular grid cell')
    if not 0 <= background_alpha < 1:
        raise ValueError('Background opacity must be in [0, 1)')
    z = ordered.predicted_score.to_numpy(float).reshape(len(y), len(x))
    support = ordered.supported.to_numpy(bool).reshape(z.shape)
    alpha = np.where(support, 1., background_alpha)
    return x, y, z, alpha, np.ma.masked_where(~support, z)
