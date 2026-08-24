#!/usr/bin/env python
"""Validate the max-model (traditional 1h+2h) shear implementation.

Per the accuracy policy, the reference is adaptive: the outer mass
integral on embedded GL-10/20 panels with reported error, with the
z-RESOLVED full_ltmz weights inside (the two-halo term is z-dependent,
so z stays inside the mass integrand). Three comparisons:

1. fast path (S_ij-tabulated W2d, production GL nodes) vs the adaptive
   z-resolved full_ltmz reference — expect the usual S-tabulation
   class (~1e-3);
2. full_ltmz GL (direct kernels, same nodes) vs the adaptive reference
   — certifies the GL variant;
3. sanity: with the 2h term forced to zero the fast path must
   reproduce the validated 1h fast_mass backend exactly.

Usage: python validate_shear1h2h_max.py <dump_dir>   (needs a dump with
halomodel/dsigma_hh, i.e. halo_model run with compute_lensing_2h = T)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

for _p in Path(__file__).resolve().parents:
    if (_p / "shared" / "datablock_models.py").is_file():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from shared import datablock_models as dm
from shared import full_ltmz_core as flc
from shared import lensing_profiles as lp
from systematics.selection_richness.python import sel_kernels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shear1h2h_max import compute_shear_max, z_resolved_weights  # noqa: E402
from shear1h_fast_mass import compute_shear as shear_1h_only     # noqa: E402

R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
BINS = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03))
ENV = dict(zt_low=0.05, zt_high=0.80, lnm_low=29.9336, lnm_high=36.7300)
TOL_FAST = 5e-3
TOL_GL = 5e-4


def adaptive_reference(bins, mor, plob, hmf, dv, sci, phi_z, r_perp, *,
                       epsrel=1e-6, max_panels=256, **env):
    """Adaptive mass integral with a z-dependent profile.

    phi_z(b, R[:, None], m[None, :], z_x) -> (n_r, n_m, n_z); the
    z-resolved weights come from full_ltmz_mass_z_weights at the panel's
    mass nodes. Same panel-bisection scheme as
    full_ltmz_mass_integral_adaptive.
    """
    sf = sel_kernels.load()
    n_bins = len(np.asarray(bins["lam_min"]))
    r_arr = np.asarray(r_perp, dtype=float)

    vals = np.empty((n_bins, r_arr.size))
    errs = np.empty((n_bins, r_arr.size))
    for b in range(n_bins):
        grid = sf._choose_lnM_grid(
            float(bins["lam_min"][b]), float(bins["lam_max"][b]),
            float(bins["zob_min"][b]), float(bins["zob_max"][b]),
            mor, env["lnm_low"], env["lnm_high"], 2)
        lo, hi = env["lnm_low"], float(grid[-1])
        sub = {k: np.asarray(bins[k])[b:b + 1] for k in
               ("lam_min", "lam_max", "zob_min", "zob_max", "sigma_z")}

        def panel(a, c):
            x10, w10 = dm.gl_nodes(a, c, 10)
            x20, w20 = dm.gl_nodes(a, c, 20)
            m = np.concatenate([x10, x20])
            _, _, z_x, w2d = flc.full_ltmz_mass_z_weights(
                sub, mor, plob, hmf, dv, sci=sci, lnm_nodes=m, **env)
            phi = phi_z(b, r_arr[:, None, None], m[None, :, None],
                        z_x[None, None, :])                # (r, 30, q)
            g = np.einsum("rmq,mq->rm", phi, w2d[0])       # (r, 30)
            return g[:, 10:] @ w20, np.abs(g[:, 10:] @ w20 - g[:, :10] @ w10)

        i0, e0 = panel(lo, hi)
        panels = [(lo, hi, i0, e0)]
        for _ in range(max_panels):
            tot = np.sum([p[2] for p in panels], axis=0)
            tot_e = np.sum([p[3] for p in panels], axis=0)
            if np.all(tot_e <= epsrel * np.abs(tot)):
                break
            worst = max(range(len(panels)),
                        key=lambda i: np.max(panels[i][3]))
            a, c, _, _ = panels.pop(worst)
            mid = 0.5 * (a + c)
            panels.append((a, mid) + panel(a, mid))
            panels.append((mid, c) + panel(mid, c))
        vals[b] = np.sum([p[2] for p in panels], axis=0)
        errs[b] = np.sum([p[3] for p in panels], axis=0)
    return vals, errs


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: validate_shear1h2h_max.py <dump_dir> "
                 "(halo_model must have run with compute_lensing_2h=T)")
    source = dm.DumpSource(sys.argv[1])
    mor = sel_kernels.mor_from_source(source)
    plob = sel_kernels.plob_splines_default()
    hmf, dv, sci = dm.HMF(source), dm.DVDoDz(source), dm.SigmaCritInv(source)
    omega_m = source.scalar("cosmological_parameters", "omega_m")
    profile = lp.MaxMixtureProfile(
        source, lob_centers=dm.DEFAULT_LOB_CENTERS,
        f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT, omega_m=omega_m)

    def phi_z(b, R, m, z):
        one = profile._one(b, R[..., 0], m[..., 0])[..., None]
        two = profile._bias(m, z) * profile._hh(R, z)
        return np.maximum(one, two)

    # fast path (S_ij tabulated)
    lnm_x, lnm_w, z_x, w2d = z_resolved_weights(
        source, n_lnm=96, n_z=64, zt_lo=ENV["zt_low"], zt_hi=ENV["zt_high"],
        lnm_lo=ENV["lnm_low"], lnm_hi=ENV["lnm_high"])
    fast = compute_shear_max(profile, lnm_x, lnm_w, z_x, w2d,
                             np.arange(12), R_PERP).reshape(12, -1)

    # full_ltmz GL (direct kernels, same nodes)
    xg, wg, zg, w2dg = flc.full_ltmz_mass_z_weights(
        BINS, mor, plob, hmf, dv, sci=sci, **ENV)
    gl = np.empty((12, R_PERP.size))
    for b in range(12):
        one = profile._one(b, R_PERP[:, None], xg[None, :])
        two = (profile._bias(xg[:, None], zg[None, :])[None, :, :]
               * profile._hh(R_PERP[:, None], zg[None, :])[:, None, :])
        phi = np.maximum(one[:, :, None], two)
        gl[b] = np.einsum("rkq,kq->r", phi, w2dg[b] * wg[:, None])

    ref, ref_err = adaptive_reference(BINS, mor, plob, hmf, dv, sci,
                                      phi_z, R_PERP, **ENV)

    d_gl = float(np.max(np.abs(gl / ref - 1)))
    d_fast = float(np.max(np.abs(fast / ref - 1)))
    print("shear1h2h_max (traditional pipeline), 12 bins x 10 radii:")
    print(f"  adaptive reference reported err <= {np.max(ref_err/np.abs(ref)):.1e}")
    print(f"  full_ltmz GL vs adaptive reference: {d_gl:.2e} (tol {TOL_GL:.0e})")
    print(f"  fast path   vs adaptive reference: {d_fast:.2e} (tol {TOL_FAST:.0e})")

    # sanity: 2h -> 0 must reproduce the validated 1h fast_mass backend
    class Zero:
        def __call__(self, *a):
            return np.zeros(np.broadcast_shapes(*[np.shape(x) for x in a]))
    profile0 = lp.MaxMixtureProfile(
        source, lob_centers=dm.DEFAULT_LOB_CENTERS,
        f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT, omega_m=omega_m)
    profile0._hh = Zero()
    fast0 = compute_shear_max(profile0, lnm_x, lnm_w, z_x, w2d,
                              np.arange(12), R_PERP)
    mzw = dm.MassZWeights(source, n_lnm=96, n_z=64, zt_lo=ENV["zt_low"],
                          zt_hi=ENV["zt_high"], lnm_lo=ENV["lnm_low"],
                          lnm_hi=ENV["lnm_high"], include_sci=True)
    oneh = shear_1h_only(mzw, profile0._one, np.arange(12), R_PERP)
    d_1h = float(np.max(np.abs(fast0 / oneh - 1)))
    print(f"  2h->0 limit vs validated 1h fast_mass: {d_1h:.2e}")

    ok = (d_gl <= TOL_GL) and (d_fast <= TOL_FAST) and (d_1h <= 1e-12)
    if not ok:   # NaN must fail, not slip through a '>' compare
        sys.exit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
