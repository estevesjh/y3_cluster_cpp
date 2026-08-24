#!/usr/bin/env python3
"""Contract tests for the SIGNED miscentered-NFW DeltaSigma "single" table.

The table (data/nfw_off_center/table_1000_1e-03_5e+03_deltasigma_signed_
single.txt, regenerated bit-identically by make_signed_deltasigma_table.py)
stores the dimensionless DeltaSigma_mis(x, x_mis) LINEARLY, keeping the
physical negative lobe at x_mis > x that the legacy log table structurally
zeroed. That lobe is what makes the mean-field ("rnd") term of the shear
projection cancel for a uniformly distributed halo population.

Tested here, dump-free (table + analytic NFW only):

1. lobe location   — negative entries exist, live at x_mis > x, and the
                     x >> x_mis core is entirely positive;
2. no-exp reading  — NfwDsigmaMisProduction(kernel="single") returns
                     norm * table (linear); an accidental exp() cannot
                     reproduce the negative lobe, and at a grid node the
                     read must match the stored value exactly;
3. rho_mult        — output scales exactly linearly in rho_mult;
4. clamping        — queries beyond the tabulated axes equal the edge value;
5. cancellation    — int DSig(x|xm) 2 pi xm dxm over the tabulated range
                     cancels to a few % of the positive part (uniform-sheet
                     identity; the residual is range truncation);
6. centered limit  — the x_mis -> 0 row matches the analytic Wright &
                     Brainerd (2000) centered DeltaSigma (measured max
                     4.2e-4 on x in [0.5, 50]; tol 1e-3).

All tolerances are measured envelopes (2026-08-24 regeneration), not
guesses; see the merge-campaign notes for the measurement provenance.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from shared import lensing_profiles as lp  # noqa: E402

DATA = REPO / "data" / "nfw_off_center"
TABLE = np.loadtxt(DATA / "table_1000_1e-03_5e+03_deltasigma_signed_single.txt")
LNX = np.loadtxt(DATA / "table_1000_1e-03_5e+03_single_logx.txt")
LNXM = np.loadtxt(DATA / "table_1000_1e-03_5e+03_single_logxmis.txt")
X, XM = np.exp(LNX), np.exp(LNXM)


def _wb_sigma(x):
    out = np.empty_like(x)
    lt, gt = x < 1, x > 1
    xl, xg = x[lt], x[gt]
    out[lt] = (1 - 2 / np.sqrt(1 - xl**2)
               * np.arctanh(np.sqrt((1 - xl) / (1 + xl)))) / (xl**2 - 1)
    out[gt] = (1 - 2 / np.sqrt(xg**2 - 1)
               * np.arctan(np.sqrt((xg - 1) / (xg + 1)))) / (xg**2 - 1)
    out[~lt & ~gt] = 1.0 / 3.0
    return out


def _wb_sigmabar(x):
    out = np.empty_like(x)
    lt, gt = x < 1, x > 1
    xl, xg = x[lt], x[gt]
    out[lt] = (2 / xl**2) * (2 / np.sqrt(1 - xl**2)
                             * np.arctanh(np.sqrt((1 - xl) / (1 + xl)))
                             + np.log(xl / 2))
    out[gt] = (2 / xg**2) * (2 / np.sqrt(xg**2 - 1)
                             * np.arctan(np.sqrt((xg - 1) / (xg + 1)))
                             + np.log(xg / 2))
    out[~lt & ~gt] = 2 * (1 + np.log(0.5))
    return out


class TestSignedTableStructure(unittest.TestCase):
    def test_shape_axes_and_finiteness(self):
        self.assertEqual(TABLE.shape, (XM.size, LNX.size))
        self.assertTrue(np.isfinite(TABLE).all())

    def test_negative_lobe_located_at_xmis_greater_than_x(self):
        Xg, XMg = np.meshgrid(X, XM)
        lobe = TABLE[XMg > 3 * Xg]
        core = TABLE[Xg > 3 * XMg]
        self.assertGreater((lobe < 0).mean(), 0.9)
        self.assertEqual((core < 0).sum(), 0)

    def test_centered_limit_matches_wright_brainerd(self):
        sel = (X > 0.5) & (X < 50.0)
        analytic = _wb_sigmabar(X[sel]) - _wb_sigma(X[sel])
        rel = np.abs(TABLE[0][sel] / analytic - 1.0)
        self.assertLess(rel.max(), 1e-3)  # measured 4.2e-4

    def test_uniform_sheet_cancellation(self):
        # int_0^inf DSig(x|xm) 2 pi xm dxm = 0; on the truncated table
        # range the residual is a few % of the positive part (measured
        # <= 2.2e-2 for x in [0.5, 10]).
        for xi in (0.5, 1.0, 3.0, 10.0):
            col = np.array([np.interp(np.log(xi), LNX, TABLE[j])
                            for j in range(XM.size)])
            w = 2 * np.pi * XM
            net = np.trapz(col * w, XM)
            pos = np.trapz(np.clip(col, 0, None) * w, XM)
            self.assertLess(abs(net) / pos, 3e-2,
                            msg=f"cancellation broken at x={xi}")


class TestSignedReader(unittest.TestCase):
    def test_single_kernel_reads_linearly_at_grid_nodes(self):
        # At an exact (x, x_mis) grid node the bilinear interpolation is
        # the stored value; the reader must return norm * value with NO
        # exp(). Pick a node inside the negative lobe so an exp() slip
        # cannot fake it.
        ds = lp.NfwDsigmaMisProduction(kernel="single")
        j = np.searchsorted(XM, 2.0)        # x_mis ~ 2
        i = np.searchsorted(X, 0.3)         # x ~ 0.3  -> lobe
        stored = TABLE[j, i]
        self.assertLess(stored, 0.0)
        lnM = 33.0
        r200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * lp.RHOC))
        r_s = r200 / lp.CONC
        got = ds(np.array([X[i] * r_s]), np.array([XM[j] * r_s]), lnM)
        norm = 2.0 * r_s * lp.DELTA_C * lp.RHOC * 1e-12
        np.testing.assert_allclose(got, stored * norm, rtol=1e-12)

    def test_rho_mult_scales_linearly(self):
        ds = lp.NfwDsigmaMisProduction(kernel="single")
        a = ds(np.array([1.0]), np.array([0.2]), 33.0, rho_mult=1.0)
        b = ds(np.array([1.0]), np.array([0.2]), 33.0, rho_mult=0.31)
        np.testing.assert_allclose(b, 0.31 * a, rtol=1e-13)

    def test_clamp_beyond_axes_equals_edge(self):
        ds = lp.NfwDsigmaMisProduction(kernel="single")
        lnM = 33.0
        r200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * lp.RHOC))
        r_s = r200 / lp.CONC
        edge = ds(np.array([X[-1] * r_s]), np.array([XM[0] * r_s]), lnM)
        beyond = ds(np.array([10.0 * X[-1] * r_s]),
                    np.array([0.1 * XM[0] * r_s]), lnM)
        np.testing.assert_allclose(beyond, edge, rtol=1e-12)

    def test_gamma_kernel_stays_legacy_log_positive(self):
        g = lp.NfwDsigmaMisProduction(kernel="gamma")
        v = g(np.array([0.3]), np.array([2.0]), 33.0)
        self.assertGreater(float(v[0]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
