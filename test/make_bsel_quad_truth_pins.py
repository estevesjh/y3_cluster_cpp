#!/usr/bin/env python
"""Dump-fed adaptive quad-truth for the b_sel_marg wall (issue #11).

Recomputes the three Costanzi-2026 P[X] operators (P1, I1, J = I2 - I1)
for every row of the production wall, from the SAME fiducial dump the
C++ ``BSelMargIntegrand.so`` consumed -- but with an independent
numerical method:

  * outer z integral: scipy.integrate.quad, adaptive, epsrel 1e-8,
    split at zob (production: fixed ring + fg/bg log-|Delta chi| GL);
  * theta: per-z GL on [theta_excl(z), 2 theta_lob] split at theta_lob
    (production: ONE global theta grid + per-z exclusion masking);
  * lt, lnM: refined GL grids (default 3-5x the production node counts).

The integrand definition mirrors src/models/p_operator_t.hh (which in
turn mirrors richness_selection/sel_bias.py::_P_operator and the
adaptive reference in RichnessSelection
validations/frozen_bsel_validation.py::quad_truth):

  P[X](bin, zob) = int dz w_z(z; zob) dV/dOm/dz(z)
                   int dtheta 2 pi sin(theta) [theta > theta_excl(z)]
                   int_0^lob dlt lt P_mor(lt | M, z) f_A(theta; ...)
                   int dlnM n(M, z) X

with X = 1 (P1), b xi sigma(theta) (I1), b xi (1 - sigma(theta)) (J);
xi = xi_NL(dchi_3d(theta, z), zob) evaluated at the zob slice
(production convention); the continuous shifted-Poisson MOR
(Costanzi-2026 form); w_z the (1 - u^2) photo-z parabola with
sigma_z(z) from the z-kernel table; z endpoints solving
z_fg + sigma_z(z_fg) = zob and z_bg - sigma_z(z_bg) = zob.

All inputs (chi, dV, HMF, bias, xi_NL, HOD parameters, sigma_z) come
from the fiducial dump / shared replicas -- so a disagreement with the
C++ wall isolates OPERATOR NUMERICS OR CONVENTIONS, not input physics.

Usage (writes the pin block for test/bsel_external.test.py to stdout):
    source ~/cosmosis_y3/cosmosis_init_macos.sh
    python -B test/make_bsel_quad_truth_pins.py [--check-convergence]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from shared import datablock_models as dm          # noqa: E402
from shared import z_kernel                        # noqa: E402
from shared.lensing_profiles import r_lambda       # noqa: E402

DUMP = REPO / "docs" / "figs" / "real_pipeline_extract_prj2h_output"

# Production wall config (docs/figs/real_pipeline_extract_prj2h.ini
# [b_sel_marg]): lnm bounds; refined inner-grid node counts below.
LNM_LO, LNM_HI = 29.9336, 35.6814
N_TH, N_LT, N_M = 40, 120, 64      # production: 10, 60, 24
EPSREL = 1.0e-8


def gl_nodes(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def area_overlap(theta, theta_lob, theta_lt):
    """Two-disk fractional overlap, closed form (p_operator_cuhre_t.hh)."""
    theta = np.asarray(theta, dtype=float)
    theta_lt = np.asarray(theta_lt, dtype=float)
    t, l, o = np.broadcast_arrays(theta, theta_lt, theta_lob)
    out = np.zeros(t.shape)
    inside = t <= np.abs(o - l)
    small = l <= o
    out[inside & small] = 1.0
    big = inside & ~small
    out[big] = (o[big] / l[big]) ** 2
    mid = (~inside) & (t < o + l)
    tt, ll, lo = t[mid], l[mid], o[mid]
    c1 = np.clip((tt**2 + ll**2 - lo**2) / (2.0 * tt * ll), -1.0, 1.0)
    c2 = np.clip((tt**2 + lo**2 - ll**2) / (2.0 * tt * lo), -1.0, 1.0)
    sq = np.maximum((-tt + ll + lo) * (tt + ll - lo)
                    * (tt - ll + lo) * (tt + ll + lo), 0.0)
    out[mid] = (ll**2 * np.arccos(c1) + lo**2 * np.arccos(c2)
                - 0.5 * np.sqrt(sq)) / (np.pi * ll**2)
    return out


class MorShiftedPoisson:
    """Continuous Costanzi-2026 shifted-Poisson pdf (p_operator_t.hh)."""

    def __init__(self, source):
        p = dm.HODParameters.from_source(source)
        self.Mmin = 10.0 ** p.log10_Mmin
        self.dM1 = 10.0 ** p.log10_M1 - self.Mmin
        self.alpha, self.eps = p.alpha, p.epsilon
        self.zp, self.sig = p.z_pivot, p.sigma_lambda

    def pdf(self, lt, M, z):
        from scipy.special import gammaln
        zfac = ((1.0 + z) / (1.0 + self.zp)) ** self.eps
        dM = np.maximum(M - self.Mmin, 0.0)
        l_sat = np.where(dM > 0.0, (dM / self.dM1) ** self.alpha * zfac, 0.0)
        mi = (l_sat * self.sig) ** 2
        lam = np.maximum(l_sat + mi, 1e-300)
        x = lt + mi
        return np.exp(-lam + (x - 1.0) * np.log(lam) - gammaln(x))


def quad_truth_row(src, lob, zob, n_th=N_TH, n_lt=N_LT, n_m=N_M):
    dist_z = src.array("distances", "z")
    d_c = src.array("distances", "d_c")
    h0 = src.scalar("cosmological_parameters", "h0")
    chi = lambda z: np.interp(z, dist_z, d_c) * h0            # cMpc/h

    dv = dm.DVDoDz(src)
    hmf = dm.HMF(src)
    bias = dm.Bilinear2D(src, "halomodel", "lnm", "z", "bias")
    mor = MorShiftedPoisson(src)

    # xi at the zob slice (production convention), log-r interp.
    r_xi = src.array("xi_nl", "r")
    z_xi = src.array("xi_nl", "z")
    xi_t = src.array("xi_nl", "xi_nl").reshape(z_xi.size, r_xi.size)
    iz = np.clip(np.searchsorted(z_xi, zob) - 1, 0, z_xi.size - 2)
    fz = (zob - z_xi[iz]) / (z_xi[iz + 1] - z_xi[iz])
    xi_row = xi_t[iz] + fz * (xi_t[iz + 1] - xi_t[iz])
    log_r = np.log(r_xi)
    xi_at = lambda r: np.interp(np.log(np.maximum(r, r_xi[0])),
                                log_r, xi_row)

    chi_o = float(chi(zob))
    theta_lob = float(r_lambda(lob)) * (1.0 + zob) / chi_o
    R_excl = float(r_lambda(lob)) * (1.0 + zob)
    k_sig, th0 = 2.5 / theta_lob, 0.5 * theta_lob
    th_hi = 2.0 * theta_lob

    lnms, w_m = gl_nodes(LNM_LO, LNM_HI, n_m)
    Ms = np.exp(lnms)
    lts, w_lt = gl_nodes(1.0e-6, lob, n_lt)
    lt_wlt = lts * w_lt

    def f_inner(z, which):
        sig_z = float(z_kernel.sigma_z(np.array([z]))[0])
        u = (z - zob) / sig_z
        if abs(u) >= 1.0:
            return 0.0
        wz = 1.0 - u * u
        chi_z = float(chi(z))
        cos_e = np.clip((chi_z**2 + chi_o**2 - R_excl**2)
                        / (2.0 * chi_z * chi_o), -1.0, 1.0)
        th_lo = 1e-6 if cos_e >= 1.0 - 1e-12 else float(np.arccos(cos_e))
        th_lo = max(th_lo, 1e-6)
        if th_lo >= th_hi:
            return 0.0
        # per-z theta grid split at theta_lob (sigmoid transition)
        if th_lo < theta_lob < th_hi:
            t1, w1 = gl_nodes(th_lo, theta_lob, n_th // 2)
            t2, w2 = gl_nodes(theta_lob, th_hi, n_th - n_th // 2)
            ths, w_th = np.concatenate([t1, t2]), np.concatenate([w1, w2])
        else:
            ths, w_th = gl_nodes(th_lo, th_hi, n_th)
        omega = w_th * 2.0 * np.pi * np.sin(ths)
        sig_t = 1.0 / (1.0 + np.exp(-k_sig * (ths - th0)))
        dchi = np.sqrt(np.maximum(
            chi_z**2 + chi_o**2 - 2.0 * chi_z * chi_o * np.cos(ths), 0.0))
        xi = xi_at(dchi)

        theta_lt = r_lambda(lts) * (1.0 + z) / chi_z          # (n_lt,)
        fA = area_overlap(ths[:, None], theta_lob, theta_lt)  # (n_th, n_lt)

        if which == "P1":
            ang = omega @ fA                                  # (n_lt,)
        elif which == "I1":
            ang = (omega * sig_t * xi) @ fA
        else:                                                 # J
            ang = (omega * (1.0 - sig_t) * xi) @ fA

        p_mor = mor.pdf(lts[:, None], Ms[None, :], z)         # (n_lt, n_m)
        lam_int = (lt_wlt * ang) @ p_mor                      # (n_m,)
        n_m_v = hmf(lnms, z)
        if which == "P1":
            return float(dv(z)) * wz * np.sum(w_m * n_m_v * lam_int)
        b_v = bias(lnms, z)
        return float(dv(z)) * wz * np.sum(w_m * n_m_v * b_v * lam_int)

    z_fg = float(bisect(lambda z: z + float(z_kernel.sigma_z(
        np.array([z]))[0]) - zob, 0.01, 2.0))
    z_bg = float(bisect(lambda z: z - float(z_kernel.sigma_z(
        np.array([z]))[0]) - zob, 0.01, 2.0))
    out = {}
    for which in ("P1", "I1", "J"):
        lo, _ = quad(f_inner, z_fg, zob, args=(which,),
                     epsrel=EPSREL, limit=300)
        hi, _ = quad(f_inner, zob, z_bg, args=(which,),
                     epsrel=EPSREL, limit=300)
        out[which] = lo + hi
    return out


def main():
    src = dm.DumpSource(str(DUMP))
    lam_edges = src.array("sel_function", "lambda_edges")
    wall = dm.BSelWallVector.from_source(src, lam_edges)
    check = "--check-convergence" in sys.argv

    rows, dev = [], {"P1": [], "I1": [], "J": []}
    for i in range(wall.p1.size):
        lob, zob = float(wall.lob[i]), float(wall.zob[i])
        truth = quad_truth_row(src, lob, zob)
        cpp = {"P1": wall.p1[i], "I1": wall.i1[i], "J": wall.j[i]}
        for k in dev:
            dev[k].append(abs(cpp[k] / truth[k] - 1.0))
        rows.append((int(wall.lambda_bin[i]), zob, truth))
        print(f"# bin={wall.lambda_bin[i]} zob={zob:.3f} lob={lob:6.1f}  "
              + "  ".join(f"{k}: quad={truth[k]:.6e} cpp/quad-1="
                          f"{cpp[k]/truth[k]-1.0:+.2e}" for k in
                          ("P1", "I1", "J")))
        if check and i == 0:
            t2 = quad_truth_row(src, lob, zob, n_th=2 * N_TH,
                                n_lt=2 * N_LT, n_m=2 * N_M)
            print("# convergence (row 0, 2x inner grids): "
                  + "  ".join(f"{k}: {t2[k]/truth[k]-1.0:+.2e}"
                              for k in ("P1", "I1", "J")))

    for k in dev:
        d = np.array(dev[k])
        print(f"# {k}: max |cpp/quad-1| = {d.max():.3e}  "
              f"median = {np.median(d):.3e}")

    def fmt(vals):
        return ",\n    ".join(", ".join(f"{v:.10e}" for v in vals[j:j + 4])
                              for j in range(0, len(vals), 4))
    print("\n# ---- pins for test/bsel_external.test.py ----")
    for k in ("P1", "I1", "J"):
        vals = [r[2][k] for r in rows]
        print(f"QUAD_{k} = np.array([\n    {fmt(vals)}\n])")


if __name__ == "__main__":
    main()
