"""Fit the McClintock et al. 2019 boost-factor model B(R; rs, b0) to the
real DES Y1 boost-factor data (data/y1_profiles/), one chi-square
minimization per (richness, redshift) bin -- no CosmoSIS/emcee run needed,
just scipy.optimize on boost_factor_model() from bf_likelihood_improved.py.

Cross-checked against an existing real MCMC run (test_per_binc00.txt from
~/Documents/GitHub/Boost_factor_cosmosis/outputs/, richness bin 0 at
redshift bin 0): this script's fit (rs=0.2604, b0=0.2227) matches that
chain's posterior mean (rs=0.2600, b0=0.2232) to ~0.2%.

Prints results in the exact `rs_l{l}_z{z}` / `b0_l{l}_z{z}` key format
values_gpu.ini's [boost_factor] section expects (see
apply_boost_factor.py), so the output can be pasted there directly.

Caveat: chi2/dof is typically 2-4 (not ~1) for this simple analytic model
against the real data -- a known limitation of the model, not a fitting
bug (checked against the reference MCMC result above). These are the best
achievable fits, not a perfect description of the data.

Run: python3 fit_y1_bins.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from bf_likelihood_improved import load_y1_data, compute_likelihood_standalone

DATA_PATH = Path(__file__).resolve().parent / "data" / "y1_profiles"
N_LAMBDA_BIN = 4
N_Z_BIN = 3
N_RADIAL_POINTS = 8

# Multiple starting points -- the likelihood surface in (log rs, log b0)
# is not always unimodal/well-behaved for every bin, so a single
# Nelder-Mead start can land on a local optimum.
START_POINTS = [(0.0, 0.0), (-0.5, -0.5), (0.3, 0.3), (-1.0, -1.0), (0.5, -0.5)]


def fit_bin(l: int, z: int):
    data = load_y1_data(str(DATA_PATH), richness_bin=l, redshift_bin=z,
                        n_points=N_RADIAL_POINTS)

    def neg_log_like(params):
        logrs, logb0 = params
        rs, b0 = 10 ** logrs, 10 ** logb0
        logL, _, _ = compute_likelihood_standalone(
            data.R, data.data_vector, data.covariance, rs, b0)
        return -logL

    best = None
    for x0 in START_POINTS:
        res = minimize(neg_log_like, x0=x0, method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-8, maxiter=5000))
        if best is None or res.fun < best.fun:
            best = res

    rs, b0 = 10 ** best.x[0], 10 ** best.x[1]
    chi2 = 2.0 * best.fun
    dof = len(data.R) - 2
    return rs, b0, chi2, dof


def main():
    print(f"{'bin':>8} {'rs':>10} {'b0':>10} {'chi2/dof':>10}")
    lines = []
    for l in range(N_LAMBDA_BIN):
        for z in range(N_Z_BIN):
            rs, b0, chi2, dof = fit_bin(l, z)
            print(f"l{l}_z{z:>6} {rs:10.4f} {b0:10.4f} {chi2 / dof:10.2f}")
            lines.append(f"rs_l{l}_z{z} = {rs:.4f}")
            lines.append(f"b0_l{l}_z{z} = {b0:.4f}")

    print("\n# Paste into values_gpu.ini's [boost_factor] section:")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
