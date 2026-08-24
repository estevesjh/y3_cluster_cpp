#!/usr/bin/env python
"""Backend-equivalence check for the radial_series implementations.

The Python backend interpolates the npz tables with cubic splines; the
C++ backend interpolates the text export with GSL bilinear/linear
(the pipeline's production convention). Both read the same committed
values, so the only backend difference is the interpolation scheme;
this script measures it on the real 12-bin x 10-radius production grid
by evaluating the series both ways from a test-sampler dump — a
faithful stand-in for the built .so that runs anywhere Python does.

After building Shear1hRadialSeries.so on a GPU node, the same numbers
can be confirmed end-to-end by running the smoke pipeline with the C++
module and comparing shear1h_radial_series/vals against this script's
bilinear column (they should agree to text-roundtrip precision, ~1e-12).

Usage:  python compare_backends.py [dump_dir]
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
from systematics.selection_richness.python import sel_kernels

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfw_profile_family as pf                        # noqa: E402
from shear1h_radial_series import (RadialSeriesTable,   # noqa: E402
                                   evaluate_series)

R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])
ZT_LO, ZT_HI = 0.05, 0.80
LNM_LO, LNM_HI = 29.9336, 36.7300
TOL = 1.0e-3


class BilinearTextTable:
    """The C++ backend's interpolation semantics, from the text export."""

    def __init__(self, stem=None):
        if stem is None:
            stem = (pf.repo_root() / "data" / "radial_series"
                    / "radial_series_nfw_mis_gamma_v1")
        stem = str(stem)
        self.lnx = np.loadtxt(stem + "_lnx.txt")
        self.lnxm = np.loadtxt(stem + "_lnxm.txt")
        self._mis = {ell: np.loadtxt(stem + f"_u{ell}_mis.txt")
                     for ell in (0, 2, 3)}              # rows = lnxm
        self._cen = {ell: np.loadtxt(stem + f"_u{ell}_cen.txt")
                     for ell in (0, 2, 3)}

    def u_cen(self, ell, lnx):
        return np.interp(np.clip(lnx, self.lnx[0], self.lnx[-1]),
                         self.lnx, self._cen[ell])

    def u_mis(self, ell, lnx, lnxm):
        qx = np.clip(lnx, self.lnx[0], self.lnx[-1])
        qm = np.clip(lnxm, self.lnxm[0], self.lnxm[-1])
        tab = self._mis[ell]
        i = np.clip(np.searchsorted(self.lnx, qx) - 1, 0, self.lnx.size - 2)
        j = np.clip(np.searchsorted(self.lnxm, qm) - 1, 0,
                    self.lnxm.size - 2)
        tx = (qx - self.lnx[i]) / (self.lnx[i + 1] - self.lnx[i])
        tm = (qm - self.lnxm[j]) / (self.lnxm[j + 1] - self.lnxm[j])
        return ((1 - tx) * (1 - tm) * tab[j, i]
                + tx * (1 - tm) * tab[j, i + 1]
                + (1 - tx) * tm * tab[j + 1, i]
                + tx * tm * tab[j + 1, i + 1])

    def u_mix(self, ell, lnx, lnxm, f_mis):
        return ((1.0 - f_mis) * self.u_cen(ell, lnx)
                + f_mis * self.u_mis(ell, lnx, lnxm))


def main():
    if len(sys.argv) > 1:
        dump = Path(sys.argv[1])
    else:
        dump = (sel_kernels.repo_root() / "cosmosis-models"
                / "real_pipeline_extract_output")
    if not dump.is_dir():
        sys.exit(f"dump directory not found: {dump}")

    source = dm.DumpSource(str(dump))
    weights = dm.MassZWeights(source, n_lnm=96, n_z=64,
                              zt_lo=ZT_LO, zt_hi=ZT_HI,
                              lnm_lo=LNM_LO, lnm_hi=LNM_HI,
                              include_sci=True)
    rho_ref = source.scalar("halomodel", "rho_m_ref")
    norm, ybar, mu = weights.moments_of(
        lambda lnm: pf.y_of_lnM(lnm, rho_ref), ell_max=3)

    cubic = RadialSeriesTable()
    bilin = BilinearTextTable()
    f_mis, tau_mis = dm.F_MIS_DEFAULT, dm.TAU_MIS_DEFAULT
    lob = np.asarray(dm.DEFAULT_LOB_CENTERS)

    print("radial_series backend equivalence, cubic-npz (Python) vs "
          "bilinear-text (C++ semantics):")
    worst = 0.0
    for ell_max in (2, 3):
        w = 0.0
        for b in range(12):
            r_mis = tau_mis * float(dm.R_lambda(lob[b % lob.size]))
            py = evaluate_series(cubic, R_PERP, r_mis, norm[b], ybar[b],
                                 mu[b], f_mis=f_mis, rho_ref=rho_ref,
                                 ell_max=ell_max)
            cpp = evaluate_series(bilin, R_PERP, r_mis, norm[b], ybar[b],
                                  mu[b], f_mis=f_mis, rho_ref=rho_ref,
                                  ell_max=ell_max)
            w = max(w, float(np.max(np.abs(cpp / py - 1.0))))
        print(f"  ell_max={ell_max}: max rel diff over 12 bins x "
              f"{R_PERP.size} radii = {w:.2e}")
        worst = max(worst, w)
    print(f"  (tolerance {TOL:.0e}; truncation itself is ~4.5e-3)")
    if worst > TOL:
        sys.exit("FAIL: backends differ more than the interpolation "
                 "schemes should allow")
    print("PASS")


if __name__ == "__main__":
    main()
