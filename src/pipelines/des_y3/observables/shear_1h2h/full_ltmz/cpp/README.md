# One-halo miscentred shear — `full_ltmz` reference (C++ / Cuhre)

**Status: reference backend** (validated 2026-08-12). Production
remains `Shear1hMisSel.so`. Built as `Shear1hFullLtmz.so`.

One adaptive-Cuhre triple integral per (bin, R) wall point over the
same integrand as the [Python fiducial](../python/README.md), with the
production miscentred mixture profile — deliberately sharing no
quadrature structure with the fixed-GL references it validates.

Configuration: the counts full_ltmz bin options + a zipped wall of
(bin_index, r_perp) with per-row (lt, zt, lnm) volumes; eps_rel = 1e-4
(the lambda_true ~ 1 HOD ridge; see the counts cpp README) and per-bin
lt_high ~ 4 lam_max.

Validation (real pipeline, fiducial point, 12 bins x 10 radii):
max |ratio - 1| vs the Python full_ltmz fiducial = **3.3e-4** — inside
the fiducial's own convergence band (<= 3.1e-4); all 120 Cuhre
statuses converged. Cost: 51 s/sample (reference; exempt from
production timing).

Output: shear1hfullltmz/{vals, errors, probs, status, nregions},
bin slow / R fast.
