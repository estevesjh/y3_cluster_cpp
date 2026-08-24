"""The explicit-3d selection contraction, shared across observables.

Computes the fully explicit (lambda_true, z) contraction of the DES Y3
selection at fixed-GL mass nodes — the quantity every explicit-3d
observable integrates against its own radial operator:

    W_ij(lnM) = int dz int dlt  n(M,z) dV/dOmega/dz(z) Omega(z)
                [Sigma_crit_inv(z)]  K_j(z) K_i(lt, z) P_HOD(lt | M, z)

with the lt integral on the per-(M,z) HOD bracket exactly as the
maintained sel_function machinery defines it (its kernels are imported,
not copied). Number counts contract W against 1; the 1-halo shear
against its z-free radial profile Phi_i(R, lnM).

This is the *reference* counterpart of the fast path's tabulated
version: production tabulates S_ij once on a fixed grid and interpolates
(sel_function.py -> SelGLCore); here every kernel is evaluated at the
quadrature nodes directly.
"""
from __future__ import annotations

import numpy as np

from . import datablock_models as dm
from systematics.selection_richness.python import sel_kernels


def explicit_mass_integral_adaptive(bins, mor, plob_splines, hmf, dv, *,
                                     sci=None, profile=None, r_perp=None,
                                     zt_low, zt_high, lnm_low, lnm_high,
                                     n_z=64, n_q=32, l_lam=6.0,
                                     epsrel=1e-6, max_panels=256):
    """Vectorised adaptive mass integral — the production-speed fiducial.

    Same contract as explicit_mass_integral_quad (adaptive outer mass
    integral, per-bin scaling-relation limits, reported error bounds)
    but ~two orders of magnitude faster: per panel it evaluates the
    embedded GL-10/GL-20 pair with ONE vectorised (lambda, z)
    contraction over all 30 mass nodes, reuses the weight row for every
    radius (W does not depend on R), and bisects the worst panel until
    every output component's accumulated |I20 - I10| error estimate is
    below epsrel. All output components share the same feature (the
    HOD support in mass), so shared subdivision does not over-refine —
    the failure mode that makes generic quad_vec-style vectorisation
    slow does not apply here. Cross-validated against
    scipy.integrate.quad (see the explicit 3d READMEs).

    Returns (vals, errs): (n_bins,) for counts, (n_bins, n_r) for shear.
    """
    sf = sel_kernels.load()
    n_bins = len(np.asarray(bins["lam_min"]))
    r_arr = None if profile is None else np.asarray(r_perp, dtype=float)

    def limits(b):
        grid = sf._choose_lnM_grid(
            float(np.asarray(bins["lam_min"])[b]),
            float(np.asarray(bins["lam_max"])[b]),
            float(np.asarray(bins["zob_min"])[b]),
            float(np.asarray(bins["zob_max"])[b]),
            mor, lnm_low, lnm_high, 2)
        return lnm_low, float(grid[-1])

    def w_rows(b, m_nodes):
        sub = {k: np.asarray(bins[k])[b:b + 1] for k in
               ("lam_min", "lam_max", "zob_min", "zob_max", "sigma_z")}
        _, _, W = explicit_mass_weights(
            sub, mor, plob_splines, hmf, dv, sci=sci,
            zt_low=zt_low, zt_high=zt_high, lnm_low=lnm_low,
            lnm_high=lnm_high, n_z=n_z, n_q=n_q, l_lam=l_lam,
            lnm_nodes=m_nodes)
        return W[0]

    n_out = 1 if r_arr is None else r_arr.size
    vals = np.empty((n_bins, n_out))
    errs = np.empty((n_bins, n_out))
    for b in range(n_bins):
        lo, hi = limits(b)

        def panel(a, c):
            x10, w10 = dm.gl_nodes(a, c, 10)
            x20, w20 = dm.gl_nodes(a, c, 20)
            m = np.concatenate([x10, x20])
            W = w_rows(b, m)
            if r_arr is None:
                i10 = np.array([w10 @ W[:10]])
                i20 = np.array([w20 @ W[10:]])
            else:
                F = profile(b, r_arr[:, None], m[None, :])
                i10 = F[:, :10] @ (w10 * W[:10])
                i20 = F[:, 10:] @ (w20 * W[10:])
            return i20, np.abs(i20 - i10)

        i0, e0 = panel(lo, hi)
        panels = [(lo, hi, i0, e0)]
        for _ in range(max_panels):
            total = np.sum([p[2] for p in panels], axis=0)
            tot_err = np.sum([p[3] for p in panels], axis=0)
            if np.all(tot_err <= epsrel * np.abs(total)):
                break
            worst = max(range(len(panels)),
                        key=lambda i: np.max(panels[i][3]))
            a, c, _, _ = panels.pop(worst)
            mid = 0.5 * (a + c)
            panels.append((a, mid) + panel(a, mid))
            panels.append((mid, c) + panel(mid, c))
        vals[b] = np.sum([p[2] for p in panels], axis=0)
        errs[b] = np.sum([p[3] for p in panels], axis=0)
    if r_arr is None:
        return vals[:, 0], errs[:, 0]
    return vals, errs


def explicit_mass_integral_quad(bins, mor, plob_splines, hmf, dv, *,
                                 sci=None, profile=None, r_perp=None,
                                 zt_low, zt_high, lnm_low, lnm_high,
                                 n_z=64, n_q=32, l_lam=6.0, epsrel=1e-6):
    """Adaptive-quadrature mass integral — the guaranteed-precision
    fiducial (reviewer requirement: fixed GL cannot certify its own
    error; the outer mass integral therefore uses scipy.integrate.quad,
    scalar per bin, which reports a rigorous error estimate).

    Counts when ``profile`` is None; otherwise the shear observable at
    the radii ``r_perp`` (profile(b, R, lnM) -> Phi). Returns
    (vals, errs): for counts shape (n_bins,); for shear
    (n_bins, n_r). The inner (lambda_true, z) contraction stays on the
    fixed nodes shared with every backend.
    """
    from scipy.integrate import quad
    n_bins = len(np.asarray(bins["lam_min"]))

    # Mass limits (plan owner, 2026-08-12): the UPPER limit comes from
    # the bin's richness support through the inverted scaling relation
    # (mu_sat -> M, the maintained sel_function chooser, 4 lam_max
    # bracket); the LOWER limit corresponds to richness -> 0, i.e. the
    # envelope floor. A naive inverse relation is invalid at low
    # richness where the scatter dominates: the mu >= lam_min/8 floor
    # was measured to cut a real up-to-0.9% projection-tail
    # contribution (low-mass central-only halos up-scattered into
    # lam_ob >= 20 by the EMG kernel) that production includes.
    sf = sel_kernels.load()
    def mass_limits(b):
        grid = sf._choose_lnM_grid(
            float(np.asarray(bins["lam_min"])[b]),
            float(np.asarray(bins["lam_max"])[b]),
            float(np.asarray(bins["zob_min"])[b]),
            float(np.asarray(bins["zob_max"])[b]),
            mor, lnm_low, lnm_high, 2)
        return lnm_low, float(grid[-1])

    def w_row(lnM, b):
        sub = {k: np.asarray(bins[k])[b:b + 1] for k in bins
               if k in ("lam_min", "lam_max", "zob_min", "zob_max",
                        "sigma_z")}
        _, _, W = explicit_mass_weights(
            sub, mor, plob_splines, hmf, dv, sci=sci,
            zt_low=zt_low, zt_high=zt_high, lnm_low=lnm_low,
            lnm_high=lnm_high, n_z=n_z, n_q=n_q, l_lam=l_lam,
            lnm_nodes=np.array([lnM]))
        return float(W[0, 0])

    if profile is None:
        vals, errs = np.empty(n_bins), np.empty(n_bins)
        for b in range(n_bins):
            lo, hi = mass_limits(b)
            vals[b], errs[b] = quad(w_row, lo, hi, args=(b,),
                                    epsabs=0.0, epsrel=epsrel, limit=200)
        return vals, errs

    r_perp = np.asarray(r_perp, dtype=float)
    vals = np.empty((n_bins, r_perp.size))
    errs = np.empty((n_bins, r_perp.size))
    for b in range(n_bins):
        lo, hi = mass_limits(b)
        for i, rp in enumerate(r_perp):
            f = lambda m: w_row(m, b) * float(profile(b, rp, m))
            vals[b, i], errs[b, i] = quad(f, lo, hi,
                                          epsabs=0.0, epsrel=epsrel,
                                          limit=200)
    return vals, errs


def explicit_mass_weights(bins, mor, plob_splines, hmf, dv, *,
                           sci=None, zt_low, zt_high, lnm_low, lnm_high,
                           n_lnm=96, n_z=64, n_q=32, l_lam=6.0,
                           lnm_nodes=None):
    """W_ij on GL mass nodes for every configured bin.

    ``bins``: dict of equal-length arrays (lam_min, lam_max, zob_min,
    zob_max, sigma_z). ``sci``: optional Sigma_crit_inv(z) callable
    (shear observables fold it into the z contraction; counts pass
    None). Returns (lnm_x, lnm_w, W) with W of shape (n_bins, n_lnm).
    """
    sf = sel_kernels.load()

    z_x, z_w = dm.gl_nodes(zt_low, zt_high, n_z)
    if lnm_nodes is None:
        lnm_x, lnm_w = dm.gl_nodes(lnm_low, lnm_high, n_lnm)
    else:
        lnm_x = np.asarray(lnm_nodes, dtype=float)
        lnm_w = np.full(lnm_x.size, np.nan)    # caller integrates itself
    gl_t, gl_w = np.polynomial.legendre.leggauss(n_q)

    lam_k, w_k, p_mz, degenerate = sf._compute_lam_nodes_and_P_HOD(
        lnm_x, z_x, mor, gl_t, gl_w, L=l_lam)

    mu_p, sig_p, tau_p, fprj_p = sf._plob_params(lam_k, z_x, plob_splines)
    lam_min = np.asarray(bins["lam_min"], dtype=float)
    lam_max = np.asarray(bins["lam_max"], dtype=float)
    edges = np.unique(np.concatenate([lam_min, lam_max]))
    cdfs = sf._cdf_lob_stacked(edges, mu_p, sig_p, tau_p, fprj_p)

    zfac = z_w * dv(z_x) * dm.omega_z_des(z_x)
    if sci is not None:
        zfac = zfac * sci(z_x)
    base_kq = hmf(lnm_x[:, None], z_x[None, :]) * zfac[None, :]

    n_bins = lam_min.size
    weights = np.empty((n_bins, lnm_x.size))
    for b in range(n_bins):
        lo = int(np.searchsorted(edges, lam_min[b]))
        hi = int(np.searchsorted(edges, lam_max[b]))
        k_i = cdfs[hi] - cdfs[lo]
        s_kq = np.sum(w_k * k_i * p_mz, axis=-1)
        s_kq = np.where(degenerate, 0.0, s_kq)
        k_j = sf._S_j(z_x, float(bins["zob_min"][b]),
                      float(bins["zob_max"][b]), float(bins["sigma_z"][b]))
        weights[b] = (base_kq * s_kq) @ k_j
    return lnm_x, lnm_w, weights


def explicit_mass_z_weights(bins, mor, plob_splines, hmf, dv, *,
                             sci=None, zt_low, zt_high, lnm_low, lnm_high,
                             n_lnm=96, n_z=64, n_q=32, l_lam=6.0,
                             lnm_nodes=None):
    """z-RESOLVED explicit-3d weights: W2d[b, k, q] before the z sum.

    For observables whose radial operator depends on redshift (e.g. the
    traditional 1h+2h max model, whose two-halo term is b(M,z)
    dSigma_hh(R,z)), the z integral cannot be contracted past the
    profile; this variant returns the (lambda_true)-contracted weight
    on the (lnM, z) node grid together with the z nodes, so the caller
    contracts sum_kq W2d[b,k,q] Phi(R, lnM_k, z_q). Summing W2d over q
    reproduces explicit_mass_weights exactly.
    """
    sf = sel_kernels.load()
    z_x, z_w = dm.gl_nodes(zt_low, zt_high, n_z)
    if lnm_nodes is None:
        lnm_x, lnm_w = dm.gl_nodes(lnm_low, lnm_high, n_lnm)
    else:
        lnm_x = np.asarray(lnm_nodes, dtype=float)
        lnm_w = np.full(lnm_x.size, np.nan)
    gl_t, gl_w = np.polynomial.legendre.leggauss(n_q)

    lam_k, w_k, p_mz, degenerate = sf._compute_lam_nodes_and_P_HOD(
        lnm_x, z_x, mor, gl_t, gl_w, L=l_lam)
    mu_p, sig_p, tau_p, fprj_p = sf._plob_params(lam_k, z_x, plob_splines)
    lam_min = np.asarray(bins["lam_min"], dtype=float)
    lam_max = np.asarray(bins["lam_max"], dtype=float)
    edges = np.unique(np.concatenate([lam_min, lam_max]))
    cdfs = sf._cdf_lob_stacked(edges, mu_p, sig_p, tau_p, fprj_p)

    zfac = z_w * dv(z_x) * dm.omega_z_des(z_x)
    if sci is not None:
        zfac = zfac * sci(z_x)
    base_kq = hmf(lnm_x[:, None], z_x[None, :]) * zfac[None, :]

    n_bins = lam_min.size
    w2d = np.empty((n_bins, lnm_x.size, z_x.size))
    for b in range(n_bins):
        lo = int(np.searchsorted(edges, lam_min[b]))
        hi = int(np.searchsorted(edges, lam_max[b]))
        k_i = cdfs[hi] - cdfs[lo]
        s_kq = np.sum(w_k * k_i * p_mz, axis=-1)
        s_kq = np.where(degenerate, 0.0, s_kq)
        k_j = sf._S_j(z_x, float(bins["zob_min"][b]),
                      float(bins["zob_max"][b]), float(bins["sigma_z"][b]))
        w2d[b] = base_kq * s_kq * k_j[None, :]
    return lnm_x, lnm_w, z_x, w2d
