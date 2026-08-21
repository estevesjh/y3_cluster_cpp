#!/usr/bin/env python
"""The namespace accuracy report: every path vs the full_ltmz fiducial.

Implements the accuracy policy (src/pipelines/des_y3/README.md,
"Validation required"): accuracy is quoted against the fully explicit
`full_ltmz` calculation, whose own precision is first certified by
internal quadrature convergence; production agreement is an
algorithm-identity check, not an accuracy statement.

Usage: python validate_against_fiducial.py [dump_dir]
dump_dir defaults to docs/figs/real_pipeline_extract_output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import datablock_models as dm            # noqa: E402
from shared import full_ltmz_core                    # noqa: E402
from shared import lensing_profiles as lp            # noqa: E402
from shared import sel_kernels                       # noqa: E402

R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03))
ENV = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)


def main():
    dump = (Path(sys.argv[1]) if len(sys.argv) > 1 else
            sel_kernels.repo_root() / "docs" / "figs"
            / "real_pipeline_extract_output")
    src = dm.DumpSource(str(dump))
    mor = sel_kernels.mor_from_source(src)
    plob = sel_kernels.plob_splines_default()
    hmf, dv, sci = dm.HMF(src), dm.DVDoDz(src), dm.SigmaCritInv(src)
    profile = lp.MisMixtureProfile(
        src, lob_centers=dm.DEFAULT_LOB_CENTERS,
        f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
        omega_m=src.scalar("cosmological_parameters", "omega_m"))

    def weights(**kw):
        return full_ltmz_core.full_ltmz_mass_weights(
            BINS, mor, plob, hmf, dv, **ENV, **kw)

    def shear_of(x, w, big_w):
        out = np.empty((12, R_PERP.size))
        for b in range(12):
            out[b] = profile(b, R_PERP[:, None], x[None, :]) @ (w * big_w[b])
        return out

    # 1. fiducial self-convergence certificate
    x0, w0, W0 = weights()
    x2, w2, W2 = weights(n_lnm=192, n_z=128, n_q=64)
    x3, w3, W3 = weights(l_lam=8.0)
    n0, n2, n3 = W0 @ w0, W2 @ w2, W3 @ w3
    print("fiducial self-convergence (counts): doubled nodes "
          f"{np.max(np.abs(n2/n0-1)):.1e}, L_lam 6->8 "
          f"{np.max(np.abs(n3/n0-1)):.1e}")
    xs0, ws0, Ws0 = weights(sci=sci)
    xs2, ws2, Ws2 = weights(sci=sci, n_lnm=192, n_z=128, n_q=64)
    s0 = shear_of(xs0, ws0, Ws0)
    s2 = shear_of(xs2, ws2, Ws2)
    print("fiducial self-convergence (shear):  doubled nodes "
          f"{np.max(np.abs(s2/s0-1)):.1e}")

    # 2. every path vs the fiducial
    prod_n = src.array("numcountssel", "vals")
    prod_s = src.array("shear1hmissel", "vals").reshape(12, -1)
    print("error vs fiducial:")
    print(f"  counts fast path (production / fast_mass): "
          f"{np.max(np.abs(prod_n/n0-1)):.1e}")
    print(f"  shear  fast path (production / fast_mass): "
          f"{np.max(np.abs(prod_s/s0-1)):.1e}")

    # 3. radial_series total error vs a same-profile doubled-node fiducial
    rs_dir = (Path(__file__).resolve().parent / "observables" / "shear_1h2h"
              / "radial_series" / "python")
    sys.path.insert(0, str(rs_dir))
    import nfw_profile_family as pf
    from shear1h_radial_series import RadialSeriesTable, evaluate_series
    tab = RadialSeriesTable()

    def fixed_profile(b, r_perp, lnm):
        y = pf.y_of_lnM(lnm)
        lnx = np.log(r_perp) - y
        rmis = dm.TAU_MIS_DEFAULT * float(
            dm.R_lambda(np.asarray(dm.DEFAULT_LOB_CENTERS)[b % 4]))
        u = ((1 - dm.F_MIS_DEFAULT) * pf.u_cen(np.exp(lnx))
             + dm.F_MIS_DEFAULT * profile.omega_m
             * tab.u_mis(0, lnx, np.log(rmis) - y))
        return pf.A0_of_y(y) * u

    fid = np.empty((12, R_PERP.size))
    for b in range(12):
        fid[b] = fixed_profile(b, R_PERP[:, None], xs2[None, :]) \
            @ (ws2 * Ws2[b])
    mzw = dm.MassZWeights(src, n_lnm=96, n_z=64, zt_lo=ENV["zt_low"],
                          zt_hi=ENV["zt_high"], lnm_lo=ENV["lnm_low"],
                          lnm_hi=ENV["lnm_high"], include_sci=True)
    norm, ybar, mu = mzw.moments_of(pf.y_of_lnM, ell_max=3)
    rs = np.empty((12, R_PERP.size))
    for b in range(12):
        rmis = dm.TAU_MIS_DEFAULT * float(
            dm.R_lambda(np.asarray(dm.DEFAULT_LOB_CENTERS)[b % 4]))
        rs[b] = evaluate_series(tab, R_PERP, rmis, norm[b], ybar[b], mu[b],
                                f_mis=dm.F_MIS_DEFAULT,
                                rho_mult=profile.omega_m, ell_max=2)
    print(f"  radial_series ell<=2, total (tabulation+truncation+interp): "
          f"{np.max(np.abs(rs/fid-1)):.1e}")


if __name__ == "__main__":
    main()
