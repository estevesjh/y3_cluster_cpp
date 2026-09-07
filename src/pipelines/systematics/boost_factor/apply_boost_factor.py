"""Publish the boost factor B(R) into the datablock for the Costanzi-2026
GPU shear pipeline.

Evaluates the McClintock et al. 2019 (arXiv:1805.00039) boost factor model

    B(R) = 1 + b0 * (1 - f(x)) / (x^2 - 1),   x = R / rs

(see boost_factor_model() in bf_likelihood_improved.py, same directory) on
the radii used by shear1hmissel / shear_prj, for each of the 12
(richness, redshift) bins.

(rs, b0) are fixed, externally-calibrated values -- not sampled here --
read from the "boost_factor" datablock section, which is populated by the
[boost_factor] block of the values file (see values_gpu.ini).  Real
per-bin values should come from running bf_pipeline_improved.ini against
the Y1 boost-factor data; until that has been done, values_gpu.ini uses
the same rs=b0=1 placeholder as bf_values_all_bins.ini's prior "start".

B(R) is a source-catalog dilution correction: it is independent of the
1h/projection decomposition and must be applied once, to the *combined*
theory shear, by DIVIDING (not multiplying) -- see
    Delta_Sigma_obs(R) = Delta_Sigma_true(R) / B(R)
so the model must be diluted the same way to compare against the raw,
uncorrected data vector. That division happens downstream, in the
shear1hmissel/shear_prj combination step -- not in this module, which
only publishes B(R).

Author: Arwa Abdulghafour
"""
import numpy as np
from cosmosis.datablock import option_section

from bf_likelihood_improved import boost_factor_model

N_LAMBDA_BIN = 4
N_Z_BIN = 3


def setup(options):
    radii = np.asarray(options.get_double_array_1d(option_section, "radii"),
                       dtype=float)
    return dict(radii=radii)


def execute(block, config):
    radii = config["radii"]
    section = "boost_factor"

    block[section, "R"] = radii

    for l in range(N_LAMBDA_BIN):
        for z in range(N_Z_BIN):
            suffix = f"l{l}_z{z}"
            rs = block[section, f"rs_{suffix}"]
            b0 = block[section, f"b0_{suffix}"]
            B = boost_factor_model(radii, rs, b0)
            block[section, f"B_{suffix}"] = B

    return 0


def cleanup(config):
    pass
