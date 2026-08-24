#!/usr/bin/env python
"""Quantify the NC impact of the HOD normalization defect (issue #2).

The shifted-Poisson P(lambda_tr | M, z) does not integrate to unity
below mu_sat ~ 2 (up to +19% at mu_sat ~ 0.3 --
docs/known_issues/hod_normalization_defect.md). This script answers the
issue's open question: does that bias survive the HMF weighting in the
actual number-counts integral?

Method: rebuild the sel_function richness selection with the SAME
production machinery (shared/sel_function fused kernel, production GL
bracket), once as-is and once with P_HOD renormalized per (M, z) by its
own bracket integral N(M,z) = sum_q W_q P_q; then push both through the
NC integrand  dV/dOmega/dz * Omega_Y1(z) * dn/dlnM * S_ij  on the dump
inputs and report the per-bin ratio.

Usage:
    source ~/cosmosis_y3/cosmosis_init_macos.sh
    python -B test/make_hod_norm_impact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from shared import datablock_models as dm       # noqa: E402
from shared import sel_function as sf           # noqa: E402

DUMP = REPO / "docs" / "figs" / "real_pipeline_extract_prj2h_output"

# production sel_function config (docs/figs/real_pipeline_extract_prj2h.ini)
LAM_MIN = np.array([20., 30., 45., 60.] * 3)
LAM_MAX = np.array([30., 45., 60., 200.] * 3)
ZOB_MIN = np.repeat([0.20, 0.35, 0.50], 4)
ZOB_MAX = np.repeat([0.35, 0.50, 0.65], 4)
SIGMA_Z = np.full(12, 0.03)
N_LNM, N_Z, N_Q, L_LAM = 192, 64, 32, 6.0
LNM_LO, LNM_HI = 29.9336, 36.8414
ZT_LO, ZT_HI = 0.05, 0.80


def omega_z_des_y1(z):
    """y3_cluster::OMEGA_Z_DES (same polynomial as the xtang126 mock)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    coef1 = [0.0, 0.0, 0.0, -0.00262353, 0.01940118, 0.45133063]
    coef2 = [1.33647377e4, 1.35291046e3, -1.26204891e2,
             -2.83454918e1, -2.26465905, 3.84958753e-1]
    coef3 = [0.0, 0.0, -1.88101967, 4.8071839, -4.11424324, 1.18196785]
    out = np.empty_like(z)
    m1, m2 = z < 0.504, (z >= 0.504) & (z < 0.7)
    out[m1] = np.polyval(coef1, z[m1])
    out[m2] = np.polyval(coef2, z[m2] - 0.6)
    out[~m1 & ~m2] = np.polyval(coef3, z[~m1 & ~m2])
    return out


def build_s_stacks(src):
    """(S_raw, S_norm, mu_sat, norm) on the production grid."""
    phod = dm.PHOD.from_source(src)

    lnm_grid = np.linspace(LNM_LO, LNM_HI, N_LNM)
    z_grid = np.linspace(ZT_LO, ZT_HI, N_Z)
    gl_t, gl_w = sf._gl_nodes(N_Q)

    lam_k, W_k, P_Mz, degenerate = sf._compute_lam_nodes_and_P_HOD(
        lnm_grid, z_grid, phod, gl_t, gl_w, L_LAM)

    # Bracket integral of P_HOD per (M, z): the defect is N != 1.
    norm = np.sum(W_k * P_Mz, axis=-1)
    norm = np.where(norm > 0.0, norm, 1.0)

    # Observed-richness kernel at the unique edges (production path;
    # _make_plob_splines falls back to the canonical PrjParams table
    # when the block has no plob_ltr_params section).
    plob = sf._make_plob_splines(_NullBlock(), {})
    mu_p, sig_p, tau_p, fprj_p = sf._plob_params(lam_k, z_grid, plob)
    edges = sf._unique_edges(LAM_MIN, LAM_MAX)
    cdfs = sf._cdf_lob_stacked(edges, mu_p, sig_p, tau_p, fprj_p)

    weighted = W_k * P_Mz
    e_raw = np.stack([np.sum(weighted * c, axis=-1) for c in cdfs])
    e_norm = e_raw / norm[None, :, :]
    for e in (e_raw, e_norm):
        e[:, degenerate] = 0.0

    lo = np.searchsorted(edges, LAM_MIN)
    hi = np.searchsorted(edges, LAM_MAX)
    rich_raw = e_raw[hi] - e_raw[lo]          # (12, n_lnm, n_z)
    rich_norm = e_norm[hi] - e_norm[lo]

    s_j = sf._S_j(z_grid[None, :], ZOB_MIN[:, None], ZOB_MAX[:, None],
                  SIGMA_Z[:, None])
    mu = phod.mu_sat(np.exp(lnm_grid)[:, None], z_grid[None, :])
    return (rich_raw * s_j[:, None, :], rich_norm * s_j[:, None, :],
            mu, norm, lnm_grid, z_grid)


class _NullBlock:
    def has_section(self, *_):
        return False

    def has_value(self, *_):
        return False


def main():
    src = dm.DumpSource(str(DUMP))
    S_raw, S_norm, mu, norm, lnm_grid, z_grid = build_s_stacks(src)

    hmf = dm.HMF(src)
    dv = dm.DVDoDz(src)
    hmf_mz = hmf(lnm_grid[None, :], z_grid[:, None])       # (n_z, n_lnm)
    w_z = dv(z_grid) * omega_z_des_y1(z_grid)              # (n_z,)

    print("bracket-integral N(M,z) over the live grid: "
          f"min {norm.min():.4f} max {norm.max():.4f}")
    if mu is not None:
        low = mu.T < 2.0
        print(f"mu_sat < 2 on {100*low.mean():.1f}% of the (z, lnM) grid; "
              f"N there: max {norm.T[low].max():.4f}")

    print(f"\n{'bin':>4} {'lam':>10} {'zob':>12} {'NC_norm/NC_raw - 1':>19}")
    for b in range(12):
        # NC integrand ~ sum_z w_z sum_M hmf * S ; trapezoid weights are
        # common to both and cancel except through the S difference.
        nc_raw = np.einsum('z,zm->', w_z, hmf_mz * S_raw[b].T)
        nc_norm = np.einsum('z,zm->', w_z, hmf_mz * S_norm[b].T)
        print(f"{b:>4} [{LAM_MIN[b]:3.0f},{LAM_MAX[b]:3.0f})"
              f" [{ZOB_MIN[b]:.2f},{ZOB_MAX[b]:.2f})"
              f" {nc_norm/nc_raw - 1.0:+18.4%}")


if __name__ == "__main__":
    main()
