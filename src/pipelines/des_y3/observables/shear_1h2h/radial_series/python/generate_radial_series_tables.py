#!/usr/bin/env python
"""Offline generator for the radial_series unit-profile tables U_ell.

Implements the "Offline unit-profile table" section of
docs/module_reorganization_plan.md for the fixed one-halo profile family
(centred NFW + gamma-kernel miscentred NFW, conventions of
nfw_profile_family.py / src/models/nfw_dsigma_mis.hh):

    U_ell(x, x_mis) = (1 / ell! A0(y)) d^ell/dy^ell [ A0(y) u(R e^-y, r_mis e^-y) ]

for 0 <= ell <= 3, with A0 ~ e^y so U_ell needs only the dimensionless
(ln x, ln x_mis) axes. At runtime the series evaluator interpolates U_0,
U_2, U_3 (U_1 is kept for validation; its population coefficient mu_1
vanishes) and restores the analytic amplitude — no derivative is ever
recomputed inside an MCMC sample.

Why regenerate instead of differentiating the production look-up table:
the committed ``data/nfw_off_center`` gamma table carries ~1e-5..3e-3
point-to-point noise in ln u (measured 2026-08-11 over the physically
relevant window), which third-order spline differentiation amplifies
into O(1) garbage. This generator therefore rebuilds the *same* profile
family from first principles at ~1e-9 accuracy — every step an integral
of the analytic Wright & Brainerd NFW shapes, no differentiation of
tabulated data — and takes the y-derivatives with high-order finite
differences on an exact common-spacing log grid, where all quadrature
error is smooth and differentiates away. U_0 is then cross-checked
against the production table (agreement at the table's own noise level),
and the table checksums are recorded in the metadata.

Pipeline (all in shape units where Sigma_cen = f(x), DSigma_cen = g(x),
u = DSigma / 2 so that DSigma = A0(y) u exactly like the source table):

  1. Sigma_off(x, v)   single-offset azimuthal average of f, split GL
                       quadrature (rel. err <= ~6e-11 incl. the x = v
                       log singularity);
  2. Sbar_off(x, v)    cumulative integral of Sigma_off t dt via a
                       cubic-spline antiderivative (smooth in t) +
                       analytic small-x head;
  3. u_single          (Sbar_off - Sigma_off) / 2;
  4. D_ell(x, v)       9-point central stencils along the exact
                       diagonal (ln x - s, ln v - s) of e^s u_single —
                       the (1 - L)^ell derivatives, L = d/dlnx + d/dlnv;
  5. U_ell^mis(x, xm)  gamma-kernel average  int dw w e^-w D_ell(x, w xm)
                       (Gamma(2, xm) offset distribution, exactly the
                       production 'gamma' kernel), composite Simpson
                       over the ln v grid as a matrix product;
  6. U_ell^cen(x)      same diagonal stencils on mpmath-evaluated
                       u_cen = g/2 (1-D), self-checked against mpmath
                       Taylor derivatives.

Writes data/radial_series/radial_series_nfw_mis_gamma_v1.npz (+ .json
sidecar). Run once; the result is committed as versioned data.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import math

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfw_profile_family as pf  # noqa: E402

FORMAT_VERSION = 1
TABLE_NAME = "radial_series_nfw_mis_gamma_v1"

# Computation grid: uniform spacing H in both ln x and ln v so the
# diagonal derivative direction is an exact index shift.
H = 0.0125
T_LO, T_HI = -10.05, 8.60          # ln x
V_LO, V_HI = -10.05, 6.50          # ln v (single-offset distance / r_s)
STENCIL_HALF = 4                   # 9-point stencils

# Committed output axes (subsets/multiples of the computation grid).
OUT_LNX_START_IDX = 250            # T_LO + 250 H = -6.925
OUT_LNX_STEP = 2                   # output spacing 0.025
OUT_LNX_STOP_IDX = 1486            # T_LO + 1486 H = 8.525  (inclusive)
OUT_LNXM = np.arange(-4.625, 3.0 + 1e-12, 0.025)

# Gamma-kernel average: done as composite Simpson directly on the
# uniform ln v grid the D_ell rows live on (see gamma_average_matrix).
# Gauss panels were tried first and hit a hard floor at ~2.6e-3: the
# single-offset profile has a genuine |x - v| crease at offset = radius
# (the azimuthal average of the central log divergence), and the
# w-integration path crosses it, breaking Gauss convergence. Grid-based
# Simpson passes through the kink with an O(h^3) ~ 2e-6 local error and
# the exponentially vanishing kernel needs no endpoint treatment.

# Sigma_off azimuthal quadrature (validated to ~6e-11 rel.).
PHI0, N_PHI_IN, N_PHI_OUT = 0.1, 64, 64


def fd_weights(order, offsets, h):
    """Finite-difference weights for d^order/ds^order on given offsets."""
    k = np.asarray(offsets, dtype=float)
    v = np.vander(k, increasing=True).T          # v[m, j] = k_j^m
    rhs = np.zeros(k.size)
    rhs[order] = math.factorial(order)
    return np.linalg.solve(v, rhs) / h**order


def phi_nodes():
    ta, wa = np.polynomial.legendre.leggauss(N_PHI_IN)
    t01, w01 = 0.5 * (ta + 1.0), 0.5 * wa
    phi_in = PHI0 * t01**3                      # cubic squeeze at phi = 0
    wgt_in = PHI0 * 3.0 * t01**2 * w01
    tb, wb = np.polynomial.legendre.leggauss(N_PHI_OUT)
    phi_out = 0.5 * (np.pi / 2 - PHI0) * (tb + 1.0) + PHI0
    wgt_out = 0.5 * (np.pi / 2 - PHI0) * wb
    return (np.concatenate([phi_in, phi_out]),
            np.concatenate([wgt_in, wgt_out]))


def gamma_average_matrix(lnv_d, lnxm, h):
    """Quadrature matrix Q so that U_ell = D_ell @ Q is the gamma average.

    U(x, xm) = int_0^inf w e^-w D(x, ln(w xm)) dw
             = int dsig K(sig - ln xm) D(x, sig),
    K(tau) = e^{2 tau} exp(-e^tau)  (tau = ln w),

    evaluated with composite Simpson over the uniform sigma = ln v grid.
    The kernel vanishes double-exponentially at the top of the grid and
    as e^{2 tau} at the bottom; the small analytic below-grid tail
    (D ~ its centred limit there) is folded into the first column.
    """
    if lnv_d.size % 2 == 0:
        raise ValueError("Simpson weights need an odd number of grid nodes")
    simp = np.ones(lnv_d.size)
    simp[1:-1:2], simp[2:-1:2] = 4.0, 2.0
    simp *= h / 3.0
    tau = lnv_d[:, None] - lnxm[None, :]
    q = simp[:, None] * np.exp(2.0 * tau) * np.exp(-np.exp(tau))
    # below-grid tail: int_{-inf}^{sig_0} e^{2(sig-lnxm)} D dsig with
    # D frozen at its first-grid-node (centred-limit) value.
    q[0] += 0.5 * np.exp(2.0 * (lnv_d[0] - lnxm))
    return q


def sigma_off_grid(t, lnv, chunk=32):
    """Sigma_off(x, v) on the (t, lnv) grid via the split GL quadrature."""
    phi, wgt = phi_nodes()
    s2 = np.sin(phi)**2
    x = np.exp(t)
    v = np.exp(lnv)
    out = np.empty((t.size, lnv.size))
    for i0 in range(0, t.size, chunk):
        xs = x[i0:i0 + chunk, None, None]
        vs = v[None, :, None]
        arg = np.sqrt((xs - vs)**2 + 4.0 * xs * vs * s2[None, None, :])
        out[i0:i0 + chunk] = (2.0 / np.pi) * (pf.sigma_shape(arg) @ wgt)
    return out


def sbar_off_grid(t, sigma):
    """Sbar_off(x, v) = (2/x^2) int_0^x Sigma_off(t, v) t dt.

    Cumulative integral in t of Sigma e^{2t} via a cubic-spline
    antiderivative, plus an analytic head for (0, x_min] with Sigma_off
    modelled as a + b t from the first two rows (exact for the log-slope
    the profile actually has there).

    The antiderivative must be *smooth* in t, not merely accurate:
    downstream 9-point stencils divide point-to-point roughness by h^3.
    scipy's cumulative_simpson was tried first and its odd/even-interval
    parity error (~1e-7 here) turned the ell >= 2 derivatives into
    ~1e-2 garbage; the spline antiderivative's error is a smooth
    function of t and differentiates away.
    """
    integrand = sigma * np.exp(2.0 * t)[:, None]
    spl = CubicSpline(t, integrand, axis=0)
    cum = spl.antiderivative()(t)
    b = (sigma[1] - sigma[0]) / H
    a = sigma[0] - b * t[0]
    head = np.exp(2.0 * t[0]) * ((a + b * t[0]) / 2.0 - b / 4.0)
    return 2.0 * np.exp(-2.0 * t)[:, None] * (head + cum)


def diagonal_derivs(u, t, lnv, rows, h):
    """(1/ell!) d^ell/ds^ell [e^s u(t - s, lnv - s)] at s=0, ell = 0..3.

    ``rows`` are t-grid indices at which to evaluate; the lnv axis loses
    STENCIL_HALF nodes on each edge. Returns (lnv_out, [D0, D1, D2, D3]).
    """
    ks = np.arange(-STENCIL_HALF, STENCIL_HALF + 1)
    j = np.arange(STENCIL_HALF, lnv.size - STENCIL_HALF)
    d_out = []
    for order in range(4):
        w = fd_weights(order, ks, h)
        acc = np.zeros((rows.size, j.size))
        for k, wk in zip(ks, w):
            acc += wk * np.exp(k * h) * u[np.ix_(rows - k, j - k)]
        d_out.append(acc / math.factorial(order))
    return lnv[j], d_out


def diagonal_derivs_1d(u, rows, h):
    """1-D version of diagonal_derivs for the centred profile."""
    ks = np.arange(-STENCIL_HALF, STENCIL_HALF + 1)
    d_out = []
    for order in range(4):
        w = fd_weights(order, ks, h)
        acc = np.zeros(rows.size)
        for k, wk in zip(ks, w):
            acc += wk * np.exp(k * h) * u[rows - k]
        d_out.append(acc / math.factorial(order))
    return d_out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=None,
                    help="output directory (default data/radial_series/)")
    args = ap.parse_args()

    t0 = time.time()
    repo = pf.repo_root()
    outdir = Path(args.outdir) if args.outdir else repo / "data" / "radial_series"
    outdir.mkdir(parents=True, exist_ok=True)

    t = np.arange(T_LO, T_HI + 1e-9, H)
    lnv = np.arange(V_LO, V_HI + 1e-9, H)
    rows = np.arange(OUT_LNX_START_IDX, OUT_LNX_STOP_IDX + 1, OUT_LNX_STEP)
    out_lnx = t[rows]

    print(f"[1/6] Sigma_off on ({t.size} x {lnv.size}) grid ...", flush=True)
    sigma = sigma_off_grid(t, lnv)

    print("[2/6] Sbar_off (cumulative Simpson) ...", flush=True)
    sbar = sbar_off_grid(t, sigma)
    u_single = 0.5 * (sbar - sigma)
    del sigma, sbar

    print("[3/6] diagonal derivatives D_ell(x, v) ...", flush=True)
    lnv_d, d_ell = diagonal_derivs(u_single, t, lnv, rows, H)

    print("[4/6] gamma-kernel average -> U_ell^mis ...", flush=True)
    qmat = gamma_average_matrix(lnv_d, OUT_LNXM, H)
    u_mis = [d @ qmat for d in d_ell]

    print("[5/6] centred profile (mpmath) -> U_ell^cen ...", flush=True)
    import mpmath as mp
    mp.mp.dps = 30
    lo = rows[0] - STENCIL_HALF
    hi = rows[-1] + STENCIL_HALF
    u_cen_grid = np.zeros(t.size)
    u_cen_grid[lo:hi + 1] = [float(pf.u_cen_mp(mp.exp(ti), mp))
                             for ti in t[lo:hi + 1]]
    u_cen_ell = diagonal_derivs_1d(u_cen_grid, rows, H)

    print("[6/6] self-checks and output ...", flush=True)
    checks = {}

    # (a) centred U_ell vs mpmath Taylor of e^s u_cen(x e^-s) at s = 0.
    idx = np.linspace(10, rows.size - 11, 25).astype(int)
    worst = np.zeros(4)
    for i in idx:
        x0 = float(np.exp(out_lnx[i]))
        f = lambda s: mp.exp(s) * pf.u_cen_mp(x0 * mp.exp(-s), mp)
        tay = mp.taylor(f, 0, 3)
        for ell in range(4):
            ref = float(tay[ell])
            err = abs(u_cen_ell[ell][i] - ref) / max(abs(ref), 1e-30)
            worst[ell] = max(worst[ell], err)
    checks["cen_vs_mpmath_taylor_max_rel"] = worst.tolist()

    # (b) U_0^mis vs the production gamma look-up table over the
    # physically relevant window. Two metrics: ln-space (meaningful only
    # where the profile is not suppressed to ~0 by a large offset) and
    # the mixture-relevant |du| / u_cen(x), which is what an error in
    # u_mis does to the f_mis-weighted observable. Table noise sets the
    # floor of both.
    src = pf.MisTable()
    wx = (out_lnx >= np.log(0.05)) & (out_lnx <= np.log(60.0))
    wm = (OUT_LNXM >= np.log(0.03)) & (OUT_LNXM <= np.log(3.0))
    qx, qm = np.meshgrid(out_lnx[wx], OUT_LNXM[wm], indexing="ij")
    table_u = src.u(qx.ravel(), qm.ravel()).reshape(qx.shape)
    gen_u = u_mis[0][np.ix_(wx, wm)]
    u_cen_ref = pf.u_cen(np.exp(qx))
    signal = gen_u > 0.05 * u_cen_ref
    lndev = np.abs(np.log(gen_u[signal]) - np.log(table_u[signal]))
    lindev = np.abs(gen_u - table_u) / u_cen_ref
    checks["u0_vs_source_table"] = {
        "ln_dev_where_unsuppressed": {
            "median": float(np.median(lndev)),
            "p99": float(np.percentile(lndev, 99)),
            "max": float(lndev.max())},
        "abs_dev_over_u_cen": {
            "median": float(np.median(lindev)),
            "p99": float(np.percentile(lindev, 99)),
            "max": float(lindev.max())},
    }

    # (c) gamma-average consistency: U_0^mis vs adaptive quadrature of
    # the same D_0 rows, split at the |x - v| crease (offset = radius).
    from scipy.integrate import quad
    qerr = 0.0
    for i, m in [(len(rows)//3, len(OUT_LNXM)//3),
                 (len(rows)//2, 2*len(OUT_LNXM)//3),
                 (2*len(rows)//3, len(OUT_LNXM)//2)]:
        row = CubicSpline(lnv_d, d_ell[0][i])
        lnxm_m = OUT_LNXM[m]
        crease = float(np.clip(out_lnx[i], lnv_d[0], lnv_d[-1]))
        ref = quad(lambda s: np.exp(2.0*(s - lnxm_m) - np.exp(s - lnxm_m))
                   * row(s), lnv_d[0], lnv_d[-1],
                   points=[crease], limit=800)[0]
        ref += 0.5 * np.exp(2.0 * (lnv_d[0] - lnxm_m)) * row(lnv_d[0])
        qerr = max(qerr, abs(u_mis[0][i, m] - ref) / abs(ref))
    checks["gamma_average_max_rel"] = float(qerr)

    meta = {
        "name": TABLE_NAME,
        "format_version": FORMAT_VERSION,
        "generated": datetime.date.today().isoformat(),
        "generator": "src/pipelines/des_y3/observables/shear_1h2h/"
                     "radial_series/python/generate_radial_series_tables.py",
        "profile": {
            "family": "NFW centred (Wright & Brainerd 2000) + gamma-kernel "
                      "miscentred, exactly the src/models/nfw_dsigma_mis.hh "
                      "conventions",
            "concentration": pf.CONC,
            "rho_crit_msun_mpc3": pf.RHOC,
            "delta_c": pf.DELTA_C,
            "boundary": "200c on rho_crit; r_s = r_200 / c",
            "miscentering_kernel":
                "gamma: P(R_mis | s) = (R_mis / s^2) exp(-R_mis / s), "
                "s = x_mis r_s",
            "amplitude": "DSigma = A_sample * A0(y) * u; "
                         "A0(y) = 2 e^y delta_c rho_crit * 1e-12 "
                         "[Msun/(h pc^2)]; A_sample = 1 (centred) or "
                         "rho_mult = Omega_m (miscentred), per sample",
            "coordinates": "x = R e^-y, x_mis = r_mis e^-y, y = ln r_s(M)",
        },
        "contents": {
            "U{ell}_mis": "(1/ell!) (1-L)^ell u_mis on (lnx, lnxm) axes, "
                          "L = d/dlnx + d/dlnxm; array layout [ix, ixm]",
            "U{ell}_cen": "(1/ell!) (1-L)^ell u_cen on the lnx axis",
            "U1": "retained for validation only; mu_1 vanishes by "
                  "construction in the population series",
        },
        "axes": {
            "lnx": [float(out_lnx[0]), float(out_lnx[-1]), int(out_lnx.size)],
            "lnxm": [float(OUT_LNXM[0]), float(OUT_LNXM[-1]),
                     int(OUT_LNXM.size)],
            "note": "lnxm domain is reduced vs the source table "
                    "(x_mis <= 20; production usage is x_mis <~ 2); "
                    "evaluators clamp at the edges like Interp2D::clamp",
        },
        "numerics": {
            "computation_grid_spacing": H,
            "sigma_off_quadrature": f"split GL {N_PHI_IN}+{N_PHI_OUT}, "
                                    f"phi0={PHI0}, cubic squeeze; "
                                    "rel err <= ~6e-11",
            "sbar": "cumulative Simpson + analytic linear head",
            "derivatives": "9-point central stencils along the exact "
                           "(ln x, ln v) diagonal of e^s u",
            "gamma_average": "composite Simpson over the uniform ln v "
                             "grid (matrix product); O(h^3) local error "
                             "at the offset = radius crease, analytic "
                             "below-grid tail",
        },
        "source_tables": {
            "role": "cross-check only (U_0 fidelity); the noisy source "
                    "table is NOT differentiated",
            "sha256": src.sha256(),
        },
        "self_checks": checks,
        "interpolation": "cubic splines on the stored axes are adequate; "
                         "see validate_radial_series.py for measured "
                         "interpolation + truncation tolerances",
    }

    npz_path = outdir / f"{TABLE_NAME}.npz"
    np.savez_compressed(
        npz_path,
        lnx=out_lnx, lnxm=OUT_LNXM,
        U0_mis=u_mis[0], U1_mis=u_mis[1], U2_mis=u_mis[2], U3_mis=u_mis[3],
        U0_cen=u_cen_ell[0], U1_cen=u_cen_ell[1],
        U2_cen=u_cen_ell[2], U3_cen=u_cen_ell[3],
        meta_json=np.array(json.dumps(meta)))
    (outdir / f"{TABLE_NAME}.json").write_text(json.dumps(meta, indent=2)
                                               + "\n")

    # Text export for the C++ backend (read_vector + GSL Interp2D):
    # same values as the npz, in the data/nfw_off_center layout — 2-D
    # tables stored with rows = lnxm and columns = lnx, so the row-major
    # flattening is exactly the column-major (x-fastest) storage GSL's
    # interp2d expects through the Interp2D vector constructor.
    fmt2, fmt1 = "%.12e", "%.16e"
    np.savetxt(outdir / f"{TABLE_NAME}_lnx.txt", out_lnx, fmt=fmt1)
    np.savetxt(outdir / f"{TABLE_NAME}_lnxm.txt", OUT_LNXM, fmt=fmt1)
    for ell in range(4):
        np.savetxt(outdir / f"{TABLE_NAME}_u{ell}_mis.txt", u_mis[ell].T,
                   fmt=fmt2)
        np.savetxt(outdir / f"{TABLE_NAME}_u{ell}_cen.txt", u_cen_ell[ell],
                   fmt=fmt1)

    print(json.dumps(checks, indent=2))
    print(f"wrote {npz_path} ({npz_path.stat().st_size/1e6:.1f} MB) "
          f"+ text export in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
