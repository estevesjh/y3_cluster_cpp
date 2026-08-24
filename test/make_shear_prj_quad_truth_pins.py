#!/usr/bin/env python
"""Dump-fed adaptive quad-truth for the projection lensing observable
DSigma_prj (issue #11).

Recomputes rnd/cl channels of ``dsigma_prj`` for a subset of wall points
from the SAME fiducial dump the production evaluators consumed, with an
independent numerical method:

  * z integral: scipy.integrate.quad, adaptive (epsrel 1e-9), with the
    exclusion-ring boundary z's passed as breakpoints
    (production ShearPrjCore: fixed ring + log-|Delta chi| wing GL);
  * theta: the same breakpoint segmentation as production but log-GL
    REFINED 3x (n_per_seg 10 -> 30) -- segment placement is part of the
    integrand definition (feature-aligned panels), the node count is
    the numerical knob;
  * lnM: refined GL (16 -> 48).

Integrand definition mirrors sp_detail::ShearPrjCore via its verified
1e-8 Python port (shear_prj_gl.py, formerly shear_prj_fast_mass.py):

  rnd(R) = int dtheta 2 pi sin(theta) w_th
           int dz w_z(z;zob) dV(z) int dlnM n(M,z)
           DSigma_mis(R, theta d_A_o, M)
  cl(R)  = same, x b_sel(theta) xi_NL(dchi_3d(theta, z), zob)
           [theta > theta_excl(z)] b(M,z)

with b_sel(theta) the two-plateau sigmoid closure from the production
``b_sel_marginalised`` output, z in [zt_low, zt_high] = [0.10, 0.75]
(production knobs -- part of the definition), and the production
miscentred-NFW DSigma table (externally validated against
cluster_toolkit by test/nfw_dsigma_mis.test.cu).

All inputs come from the fiducial dump, so a disagreement with the C++
isolates OPERATOR NUMERICS, not input physics. (The input-side
distances-grid defect is tracked separately:
docs/known_issues/distances_grid_resolution_defect.md.)

Usage (prints the pin block for test/shear_prj_external.test.py):
    source ~/cosmosis_y3/cosmosis_init_macos.sh
    python -B test/make_shear_prj_quad_truth_pins.py [--check-convergence]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))
sys.path.insert(0, str(REPO / "src" / "pipelines" / "des_y3"
                       / "shear_projection" / "python" / "0d"))

from shared import datablock_models as dm            # noqa: E402
from shared import lensing_profiles as lp            # noqa: E402
from shared import z_kernel                          # noqa: E402
from shear_prj_gl import (                    # noqa: E402
    build_theta_grid, theta_excl_at_z)

DUMP = REPO / "docs" / "figs" / "real_pipeline_extract_prj2h_output"

# Production knobs (docs/figs/real_pipeline_extract_prj2h.ini [dsigma_prj])
ZT_LO, ZT_HI = 0.10, 0.75
LNM_LO, LNM_HI = 29.9336, 35.6814
R_MAX_CMPCH = 35.0
# Refined numerics
N_PER_SEG, N_M = 30, 48        # production: 10, 16(frozen)/24(exact)
EPSREL = 1.0e-8

# Wall subset to pin: (lambda_bin, zob, [radii]) -- one low-z low-richness,
# one high-z high-richness slice; small / 2h-transition / large R from the
# extract wall's 15-radius grid (docs/figs/real_pipeline_extract_prj2h.ini).
SUBSET = [(0, 0.275, [0.2607, 1.6181, 10.0107]),
          (3, 0.575, [0.2607, 1.6181, 10.0107])]


def quad_truth_point(src, lb, zob, radii, n_per_seg=N_PER_SEG, n_m=N_M):
    dist_z = src.array("distances", "z")
    d_c = src.array("distances", "d_c")
    h0 = src.scalar("cosmological_parameters", "h0")
    omega_m = src.scalar("cosmological_parameters", "omega_m")
    chi = lambda z: np.interp(np.clip(z, dist_z[0], dist_z[-1]),
                              dist_z, d_c) * h0
    dv = dm.DVDoDz(src)
    hmf = dm.HMF(src)
    bias = dm.Bilinear2D(src, "halomodel", "lnm", "z", "bias")
    xi_nl = dm.Bilinear2D(src, "xi_nl", "r", "z", "xi_nl")
    dsmis = lp.NfwDsigmaMisProduction(kernel="single")
    bsel = dm.BSelBins.from_source(src)

    lobc = float(dm.DEFAULT_LOB_CENTERS[lb])
    lob_row, zob_row, bs, bl = bsel.find_exact_row(lb, zob=zob)
    assert np.isclose(zob_row, zob)

    chi_o = float(chi(zob))
    d_a_o = chi_o / (1.0 + zob)
    r_excl = float(lp.r_lambda(lobc)) * (1.0 + zob)
    theta_lam = float(lp.r_lambda(lobc)) * (1.0 + zob) / chi_o
    k_sig, th0 = 2.5 / theta_lam, 0.5 * theta_lam

    lnms, w_m = dm.gl_nodes(LNM_LO, LNM_HI, n_m)

    theta, w_th = build_theta_grid(lobc, zob, radii, chi_o, d_a_o,
                                   r_excl, n_per_seg, R_MAX_CMPCH)
    geom = w_th * 2.0 * np.pi * np.sin(theta)
    bsel_th = bs + (bl - bs) / (1.0 + np.exp(-k_sig * (theta - th0)))

    # z-integrands, vectorised over lnM via quad_vec.
    def w_common(z):
        sz = float(z_kernel.sigma_z(np.array([z]))[0])
        u = (z - zob) / sz
        return 0.0 if abs(u) >= 1.0 else (1.0 - u * u) * float(dv(z))

    def f_rnd(z):
        c = w_common(z)
        if c == 0.0:
            return np.zeros(n_m)
        return c * hmf(lnms, z)

    warnings.filterwarnings("ignore")
    wrnd_M, _ = quad_vec(f_rnd, ZT_LO, ZT_HI, epsrel=EPSREL,
                         points=[zob])

    # One array-valued adaptive z integral for the whole (theta, M)
    # tensor.  Breakpoints: zob plus the outermost exclusion-ring
    # boundaries (per-theta boundaries vary; the adaptive subdivision
    # resolves the interior steps).
    cos_t = np.cos(theta)

    def f_cl(z):
        c = w_common(z)
        if c == 0.0:
            return np.zeros((theta.size, n_m))
        chi_z = float(chi(z))
        th_e = float(theta_excl_at_z(np.array([chi_z]), chi_o, r_excl)[0])
        gate = theta > th_e
        if not gate.any():
            return np.zeros((theta.size, n_m))
        dchi = np.sqrt(np.maximum(
            chi_z**2 + chi_o**2 - 2.0 * chi_z * chi_o * cos_t, 0.0))
        xi = xi_nl(dchi, np.full(theta.size, zob)) * gate
        return np.outer(c * xi, hmf(lnms, z) * bias(lnms, z))

    disc = r_excl**2 - (chi_o * np.sin(theta.min()))**2
    pts = [zob]
    if disc > 0.0:
        root = np.sqrt(disc)
        chi_grid = chi(dist_z)
        for chi_b in (chi_o * np.cos(theta.min()) - root,
                      chi_o * np.cos(theta.min()) + root):
            z_b = float(np.interp(chi_b, chi_grid, dist_z))
            if ZT_LO < z_b < ZT_HI:
                pts.append(z_b)
    kcl, _ = quad_vec(f_cl, ZT_LO, ZT_HI, epsrel=EPSREL,
                      points=sorted(pts))               # (n_th, n_m)

    out = {}
    for r_perp in radii:
        ds = dsmis(r_perp, theta[:, None] * d_a_o, lnms[None, :],
                   rho_mult=omega_m)                    # (n_th, n_m)
        rnd = geom @ (ds @ (w_m * wrnd_M))
        cl = (geom * bsel_th) @ ((ds * kcl) @ w_m)
        out[r_perp] = (float(rnd), float(cl))
    return out


def _read_wall_from_ini():
    """dsigma_prj publishes only vals/rnd/cl (the #10-class metadata gap
    again) -- read the wall vectors from the extract ini instead."""
    import re
    text = (REPO / "docs" / "figs"
            / "real_pipeline_extract_prj2h.ini").read_text()
    sec = re.search(r"^\[dsigma_prj\]\s*$(.*?)(?:^\[|\Z)", text,
                    re.S | re.M).group(1)
    def vec(key):
        m = re.search(rf"^{key}\s*=\s*(.+)$", sec, re.M)
        return np.array([float(v) for v in m.group(1).split()])
    return (vec("lambda_bin"), vec("zo_low"), vec("zo_high"), vec("radii"))


def main():
    src = dm.DumpSource(str(DUMP))
    # production wall for comparison
    lbw, zlw, zhw, rw = _read_wall_from_ini()
    zobw = 0.5 * (zlw + zhw)
    rnd_w = np.loadtxt(DUMP / "dsigma_prj" / "rnd.txt")
    cl_w = np.loadtxt(DUMP / "dsigma_prj" / "cl.txt")

    check = "--check-convergence" in sys.argv
    pins = []
    for lb, zob, radii in SUBSET:
        truth = quad_truth_point(src, lb, zob, radii)
        if check:
            t2 = quad_truth_point(src, lb, zob, radii,
                                  n_per_seg=2 * N_PER_SEG, n_m=2 * N_M)
            for r in radii:
                print(f"# convergence lb={lb} zob={zob} R={r}: "
                      f"rnd {t2[r][0]/truth[r][0]-1:+.2e} "
                      f"cl {t2[r][1]/truth[r][1]-1:+.2e}")
        for r in radii:
            m = (lbw == lb) & np.isclose(zobw, zob) & np.isclose(rw, r)
            (i,) = np.nonzero(m)
            i = int(i[0])
            t_rnd, t_cl = truth[r]
            print(f"# lb={lb} zob={zob} R={r:7.4f}  "
                  f"rnd: quad={t_rnd:.6e} cpp/quad-1={rnd_w[i]/t_rnd-1:+.2e}  "
                  f"cl: quad={t_cl:.6e} cpp/quad-1={cl_w[i]/t_cl-1:+.2e}  "
                  f"vals dev={((rnd_w[i]+cl_w[i])/(t_rnd+t_cl))-1:+.2e}")
            pins.append((lb, zob, r, t_rnd, t_cl))

    print("\n# ---- pins for test/shear_prj_external.test.py ----")
    print("QUAD_PINS = [  # (lambda_bin, zob, R_cMpch, rnd, cl)")
    for lb, zob, r, t_rnd, t_cl in pins:
        print(f"    ({lb}, {zob}, {r:.5f}, {t_rnd:.10e}, {t_cl:.10e}),")
    print("]")


if __name__ == "__main__":
    main()
