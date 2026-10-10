"""Deterministic sign for a 1-component PCA score.

sklearn's PCA fixes the loadings only up to an overall sign, so ``-PC1`` can flip
between fits/bootstraps whenever the two leading loadings are a near tie (the GFP
case: ocean_heat_transfer[0] vs deep_ocean_efficacy). Orient the score explicitly
by a physical reference instead: flip so it correlates POSITIVELY with the
reference (ECS for the GFP; mean per-sample tipping probability for the tipping PCA).
"""
import numpy as np


def orient_by_correlation(scores, reference):
    """Return ``scores`` sign-flipped so ``corr(scores, reference) > 0``.

    A zero or undefined correlation (constant input) leaves the sign unchanged.
    """
    s = np.asarray(scores, dtype=float)
    r = np.asarray(reference, dtype=float)
    cov = float(((s - s.mean()) * (r - r.mean())).sum())
    return -s if cov < 0 else s
