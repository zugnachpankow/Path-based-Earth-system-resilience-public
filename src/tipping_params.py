"""Shared pycascades tipping-parameter bounds + LHS sampler.
"""
import numpy as np
from pyDOE import lhs

param_bounds = {
    "gis_time":   (1000, 15000),
    "thc_time":   (15, 300),
    "wais_time":  (500, 13000),   # Armstrong McKay 2022; Möller et al. 2024 (was 2000)
    "amaz_time":  (50, 200),
    "limits_gis":    (0.8, 3.0),
    "limits_thc":    (1.4, 8.0),
    "limits_wais":   (1.0, 3.0),
    "limits_amaz":   (2.0, 6.0),
}

# probability fractions (Kriegler-style + 2021 and Möller paper)
pf_bounds = {
    "pf_wais_to_gis":  (0.1, 0.2),
    # AMOC→GIS is stabilising. Tables give signed s_ij (−10…−1); the code takes the
    # magnitude |s|/10 and applies the sign via the minus in Earth_System's coupling line
    # (strength=-(1/T_gis)*d*pf_thc_to_gis), as in pycascades earth.py / its LHS preparator
    # (pf_thc_to_gis = [0.1, 1.]). (was (-1.0, -0.1), which double-negated to destabilising.)
    "pf_thc_to_gis":   (0.1, 1.0),
    "pf_gis_to_thc":   (0.1, 1.0),
    "pf_wais_to_thc":  (-0.3, 0.3),
    "pf_gis_to_wais":  (0.1, 1.0),
    "pf_thc_to_wais":  (0.1, 0.15),
    "pf_thc_to_amaz":  (-0.4, 0.4)
}

strength_bounds = {
    "strength": (0.1, 1.0)
}

all_bounds = {}
all_bounds.update(param_bounds)
all_bounds.update(pf_bounds)
all_bounds.update(strength_bounds)

N_SAMPLES = 1000
SEED = 1234

# tipping integration horizon in years (was 15000). Each scenario keeps its own length; the
# forcing is held from that scenario's last valid year (see src/temperature.py) to T_END.
T_END = 50_000


def sample_lhs_params(n_samples=N_SAMPLES, seed=SEED):
    """Latin-hypercube sample of the tipping parameters, scaled to physical ranges."""
    n_dim = len(all_bounds)
    np.random.seed(seed) 
    lhs_unit = lhs(n_dim, samples=n_samples)
    params = {}
    for i, (key, (low, high)) in enumerate(all_bounds.items()):
        params[key] = low + lhs_unit[:, i] * (high - low)
    return params
