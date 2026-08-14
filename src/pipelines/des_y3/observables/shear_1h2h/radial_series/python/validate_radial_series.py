#!/usr/bin/env python
"""Validation battery for the radial_series implementation.

Four checks, in the order the approved plan requires them
(docs/module_reorganization_plan.md, "Validation required"):

1. derivative fidelity, centred: table-interpolated U_ell^cen versus
   mpmath Taylor derivatives of e^s u_cen(x e^-s) at off-grid points —
   exercises the generator AND the runtime cubic interpolation;
2. derivative fidelity, miscentred: committed U_1..U_3^mis versus a
   wide-window diagonal Taylor fit of the committed U_0^mis surface —
   an independent numerical route from the generator's (which
   differentiated the pre-average single-offset profile); U_0 itself is
   tied to the production look-up table and to first principles by the
   generator's recorded self-checks;
3. truncation, real population weights: the ell <= 2 and ell <= 3 series
   versus the exact fixed-GL mass integral of the same profile family,
   for all 12 pinned bins built from a real test-sampler dump (real HMF,
   distances, S_stack). Pass: ell <= 2 max fractional residual <= 0.75%
   and ell <= 3 max <= 1.0% — the study's sub-percent target, with its
   (and our) finding that the mu_3 term does not improve nearly
   symmetric weights; ell <= 2 is the evaluator default;
4. production cross-check (reported, not asserted): the series versus
   Shear1hMisSel.so in shape, where the known ~4% offset from the
   haloModel-vs-fixed-convention centred table applies (study
   §"two independently-sourced tables").

Usage:  python validate_radial_series.py [dump_dir]
dump_dir defaults to docs/figs/real_pipeline_extract_output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

for _p in Path(__file__).resolve().parents:
    if (_p / "des_y3" / "shared" / "datablock_models.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from des_y3.shared import datablock_models as dm
from des_y3.shared import sel_kernels

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfw_profile_family as pf                       # noqa: E402
from shear1h_radial_series import (RadialSeriesTable,  # noqa: E402
                                   evaluate_series)

TRUNC_TOL_ELL2 = 7.5e-3
TRUNC_TOL_ELL3 = 1.0e-2
# The pinned extraction grids (docs/figs/real_pipeline_extract.ini).
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
ZT_LO, ZT_HI = 0.05, 0.80
LNM_LO, LNM_HI = 29.9336, 36.7300


def check1_centred(table):
    import mpmath as mp
    mp.mp.dps = 30
    rng = np.random.default_rng(20260811)
    lnx = rng.uniform(table.lnx[0] + 0.2, table.lnx[-1] - 0.2, 60)
    worst = np.zeros(4)
    for t0 in lnx:
        x0 = float(np.exp(t0))
        f = lambda s: mp.exp(s) * pf.u_cen_mp(x0 * mp.exp(-s), mp)
        tay = mp.taylor(f, 0, 3)
        for ell in range(4):
            ref = float(tay[ell])
            got = float(table.u_cen(ell, t0))
            worst[ell] = max(worst[ell],
                             abs(got - ref) / max(abs(ref), 1e-30))
    print("check 1  centred U_ell (table+interp) vs mpmath Taylor, "
          "60 off-grid points:")
    for ell in range(4):
        print(f"         ell={ell}: max rel dev = {worst[ell]:.2e}")
    assert worst.max() < 1e-4, "centred U_ell derivative fidelity"
    return worst


def check2_mis(table):
    """U_1..U_3^mis vs direct diagonal derivatives of the committed U_0.

    The reference derivatives come from a wide-window polynomial fit of
    F(s) = e^s U_0(ln x - s, ln x_mis - s): U_ell is by definition the
    ell-th Taylor coefficient of F at s = 0. Fitting degree 10 over
    |s| <= 0.45 averages the U_0 interpolation wiggle out instead of
    amplifying it by 1/h^3 the way a narrow finite-difference stencil
    would — this is the route-independence check the plan asks for, at
    a reference accuracy of ~1e-5.
    """
    with np.load(table.path) as z:
        lnx, lnxm = z["lnx"], z["lnxm"]
        u0 = z["U0_mis"]
        u_ell = [z[f"U{ell}_mis"] for ell in range(4)]
    spl = RectBivariateSpline(lnx, lnxm, u0, kx=5, ky=5)
    s = np.linspace(-0.45, 0.45, 25)
    ix = np.arange(40, lnx.size - 40, 23)
    im = np.arange(24, lnxm.size - 24, 17)

    print("check 2  miscentred U_ell vs wide-window Taylor fit of the "
          "committed U_0 surface (independent route):")
    dev = {1: [], 2: [], 3: []}
    for i in ix:
        for m in im:
            f_s = np.exp(s) * spl.ev(lnx[i] - s, lnxm[m] - s)
            coef = np.polynomial.polynomial.polyfit(s, f_s, 10)
            scale = max(abs(u_ell[0][i, m]),
                        0.05 * float(pf.u_cen(np.exp(lnx[i]))))
            for ell in (1, 2, 3):
                dev[ell].append(abs(coef[ell] - u_ell[ell][i, m])
                                / max(abs(u_ell[ell][i, m]), scale))
    worst = {}
    for ell in (1, 2, 3):
        d = np.asarray(dev[ell])
        worst[ell] = float(d.max())
        print(f"         ell={ell}: max rel dev = {worst[ell]:.2e} "
              f"(median {np.median(d):.2e}, {d.size} points)")
    assert max(worst.values()) < 5e-3, "miscentred U_ell derivative fidelity"
    return worst


def exact_mass_integral(weights, table, src_tab, r_mis, b, f_mis, omega_m,
                        use_source_table):
    """Exact fixed-GL mass sum of the mixture profile for one bin."""
    lnm, w_gl = weights.lnm_x, weights.lnm_w
    y = pf.y_of_lnM(lnm)
    out = np.empty(R_PERP.size)
    for i, r in enumerate(R_PERP):
        lnx = np.log(r) - y
        lnxm = np.log(r_mis) - y
        if use_source_table:
            u_mis = src_tab.u(lnx, lnxm)          # production-path table
        else:
            u_mis = table.u_mis(0, lnx, lnxm)     # committed U_0
        phi = pf.A0_of_y(y) * ((1.0 - f_mis) * pf.u_cen(np.exp(lnx))
                               + f_mis * omega_m * u_mis)
        out[i] = np.dot(w_gl, weights.W[b] * phi)
    return out


def main():
    if len(sys.argv) > 1:
        dump = Path(sys.argv[1])
    else:
        dump = (sel_kernels.repo_root() / "docs" / "figs"
                / "real_pipeline_extract_output")
    if not dump.is_dir():
        sys.exit(f"dump directory not found: {dump}\n"
                 "run `cosmosis docs/figs/real_pipeline_extract.ini` first")

    table = RadialSeriesTable()
    src_tab = pf.MisTable()
    check1_centred(table)
    check2_mis(table)

    source = dm.DumpSource(str(dump))
    weights = dm.MassZWeights(source, n_lnm=96, n_z=64,
                              zt_lo=ZT_LO, zt_hi=ZT_HI,
                              lnm_lo=LNM_LO, lnm_hi=LNM_HI,
                              include_sci=True)
    norm, ybar, mu = weights.moments_of(pf.y_of_lnM, ell_max=3)

    n_prod = source.array("numcountssel", "vals")
    counts = dm.MassZWeights(source, n_lnm=96, n_z=64,
                             zt_lo=ZT_LO, zt_hi=ZT_HI,
                             lnm_lo=LNM_LO, lnm_hi=LNM_HI,
                             include_sci=False).norm()
    print("check 3a weight-builder consistency (f=1 vs NumCountsSel.so): "
          f"max |ratio-1| = {np.max(np.abs(counts / n_prod - 1)):.2e}")

    f_mis, tau_mis = dm.F_MIS_DEFAULT, dm.TAU_MIS_DEFAULT
    omega_m = source.scalar("cosmological_parameters", "omega_m")
    lob = np.asarray(dm.DEFAULT_LOB_CENTERS)

    shear_prod = source.array("shear1hmissel", "vals").reshape(12, -1)

    print("check 3  truncation vs exact fixed-GL mass integral "
          "(same profile family), 12 real bins:")
    worst2 = worst3 = 0.0
    worst_src = worst_prod = 0.0
    for b in range(12):
        r_mis = tau_mis * float(dm.R_lambda(lob[b % lob.size]))
        exact = exact_mass_integral(weights, table, src_tab, r_mis, b,
                                    f_mis, omega_m, use_source_table=False)
        exact_src = exact_mass_integral(weights, table, src_tab, r_mis, b,
                                        f_mis, omega_m, use_source_table=True)
        s2 = evaluate_series(table, R_PERP, r_mis, norm[b], ybar[b], mu[b],
                             f_mis=f_mis, rho_mult=omega_m, ell_max=2)
        s3 = evaluate_series(table, R_PERP, r_mis, norm[b], ybar[b], mu[b],
                             f_mis=f_mis, rho_mult=omega_m, ell_max=3)
        e2 = np.max(np.abs(s2 / exact - 1.0))
        e3 = np.max(np.abs(s3 / exact - 1.0))
        esrc = np.max(np.abs(s3 / exact_src - 1.0))
        prod_shape = shear_prod[b] / shear_prod[b][0]
        mine_shape = s3 / s3[0]
        eprod = np.max(np.abs(mine_shape - prod_shape))
        worst2, worst3 = max(worst2, e2), max(worst3, e3)
        worst_src, worst_prod = max(worst_src, esrc), max(worst_prod, eprod)
        print(f"         bin {b:2d}: ell<=2 {100*e2:6.3f}%   "
              f"ell<=3 {100*e3:6.3f}%   vs source-table exact "
              f"{100*esrc:6.3f}%   vs Shear1hMisSel.so shape "
              f"{100*eprod:6.3f}%")
    print(f"         max: ell<=2 {100*worst2:.3f}% (tol "
          f"{100*TRUNC_TOL_ELL2:.2f}%)  ell<=3 {100*worst3:.3f}% (tol "
          f"{100*TRUNC_TOL_ELL3:.2f}%)")
    print(f"check 4  vs production (reported only): max series-vs-"
          f"source-table-exact {100*worst_src:.3f}%, max shape deviation "
          f"vs Shear1hMisSel.so {100*worst_prod:.3f}% "
          "(the disclosed centred-profile convention gap: this family "
          "uses the fixed c=4 W&B centred term, production interpolates "
          "the per-sample haloModel/dSigma_nfw table)")
    if worst2 > TRUNC_TOL_ELL2 or worst3 > TRUNC_TOL_ELL3:
        sys.exit("FAIL: truncation outside tolerance")
    print("PASS")


if __name__ == "__main__":
    main()
