#!/usr/bin/env python3
"""Contract tests for the SIGNED miscentered-NFW DeltaSigma "single" table.

The table (data/nfw_off_center/table_ratio_deltasigma_signed_single.txt,
imported from CLensPy -- see
data/nfw_off_center/import_clenspy_single_table.py and GitHub issue #6)
stores the dimensionless DeltaSigma_mis(x, x_mis) LINEARLY, keeping the
physical negative lobe at x_mis > x that the legacy log table structurally
zeroed. That lobe is what makes the mean-field ("rnd") term of the shear
projection cancel for a uniformly distributed halo population.

The table is gridded on the CUSP-SAFE (ln x_mis, ln q) axes, q = x/x_mis:
DeltaSigma_mis's sign-changing ridge at x = x_mis sits on the exact node
ln q = 0, so bilinear interpolation never straddles it (unlike the legacy
natural (ln x, ln x_mis) axes -- see CLensPy's docs/miscentering_math.md
Sec.~9.3 and review/08272026/issue6_rationale.md).

Tested here, dump-free (table + analytic NFW only):

1. lobe location   — negative entries exist, live at q < 1/3 (x_mis >> x),
                     and the q > 3 (x >> x_mis) core is entirely positive;
2. no-exp reading  — NfwDsigmaMisProduction(kernel="single") returns
                     norm * table (linear); an accidental exp() cannot
                     reproduce the negative lobe, and at a grid node the
                     read must match the stored value exactly;
3. rho_mult        — output scales exactly linearly in rho_mult;
4. clamping        — queries beyond the tabulated axes equal the edge value;
5. cancellation    — int DSig(x|xm) 2 pi xm dxm over the tabulated range
                     cancels to a few % of the positive part (uniform-sheet
                     identity; the residual is range truncation);
6. centered limit  — the x_mis -> x_mis_min limit matches the analytic
                     Wright & Brainerd (2000) centered DeltaSigma.

Tolerances are measured envelopes (2026-08-27 CLensPy import), not
guesses; see review/08272026/issue6_rationale.md for the measurement
provenance.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))

from shared import lensing_profiles as lp  # noqa: E402

DATA = REPO / "data" / "nfw_off_center"
TABLE = np.loadtxt(DATA / "table_ratio_deltasigma_signed_single.txt")
LNXM = np.loadtxt(DATA / "table_ratio_logxmis.txt")
LNQ = np.loadtxt(DATA / "table_ratio_logq.txt")
XM, Q = np.exp(LNXM), np.exp(LNQ)

_RAW_INTERP = RegularGridInterpolator((LNXM, LNQ), TABLE, method="linear",
                                       bounds_error=False, fill_value=None)


def _dsig_hat(x, xmis):
    """Dimensionless (Sigma0=1) DeltaSigma_mis(x, xmis), clamped -- the raw
    table lookup with no physical normalization, for the cancellation and
    centered-limit identities below (which are Sigma0-independent)."""
    x = np.asarray(x, dtype=float)
    xmis = np.asarray(xmis, dtype=float)
    lnxm = np.clip(np.log(xmis), LNXM[0], LNXM[-1])
    lnq = np.clip(np.log(x / xmis), LNQ[0], LNQ[-1])
    lnxm_b, lnq_b = np.broadcast_arrays(lnxm, lnq)
    return _RAW_INTERP(np.stack([lnxm_b, lnq_b], axis=-1))


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
        self.assertEqual(TABLE.shape, (XM.size, Q.size))
        self.assertTrue(np.isfinite(TABLE).all())

    def test_q_zero_is_an_exact_node(self):
        # The whole point of the (ln xmis, ln q) grid: the sign-changing
        # ridge x = xmis (q = 1) never straddles a bilinear cell.
        self.assertTrue(np.any(LNQ == 0.0))

    def test_negative_lobe_located_at_xmis_greater_than_x(self):
        Qg, _XMg = np.meshgrid(Q, XM)
        lobe = TABLE[Qg < 1.0 / 3.0]   # x_mis >> x
        core = TABLE[Qg > 3.0]         # x >> x_mis
        self.assertGreater((lobe < 0).mean(), 0.9)
        self.assertEqual((core < 0).sum(), 0)

    def test_centered_limit_matches_wright_brainerd(self):
        # x_mis -> 0 limit: fix x_mis at the grid's smallest value and vary
        # x continuously via the raw interpolator (q = x/x_mis).
        xmis0 = XM[0]
        x = np.geomspace(0.5, 50.0, 200)
        analytic = _wb_sigmabar(x) - _wb_sigma(x)
        tab = _dsig_hat(x, np.full_like(x, xmis0))
        rel = np.abs(tab / analytic - 1.0)
        self.assertLess(rel.max(), 1e-2)

    def test_uniform_sheet_cancellation(self):
        # int_0^inf DSig(x|xm) 2 pi xm dxm = 0; on the truncated table
        # range the residual is a few % of the positive part (range
        # truncation, not a bug).
        for xi in (0.5, 1.0, 3.0, 10.0):
            col = _dsig_hat(np.full_like(XM, xi), XM)
            w = 2 * np.pi * XM
            net = np.trapz(col * w, XM)
            pos = np.trapz(np.clip(col, 0, None) * w, XM)
            self.assertLess(abs(net) / pos, 3e-2,
                            msg=f"cancellation broken at x={xi}")


class TestSignedReader(unittest.TestCase):
    def test_single_kernel_reads_linearly_at_grid_nodes(self):
        # At an exact (ln xmis, ln q) grid node the bilinear interpolation
        # is the stored value; the reader must return norm * value with NO
        # exp(). Pick a node inside the negative lobe (q < 1/3) so an exp()
        # slip cannot fake it.
        ds = lp.NfwDsigmaMisProduction(kernel="single")
        j = np.searchsorted(XM, 2.0)          # x_mis ~ 2
        i = np.searchsorted(Q, 0.15)          # q ~ 0.15 -> lobe (x << x_mis)
        stored = TABLE[j, i]
        self.assertLess(stored, 0.0)
        lnM = 33.0
        r200 = np.cbrt(3.0 * np.exp(lnM) / (800.0 * np.pi * lp.RHOC))
        r_s = r200 / lp.CONC
        x_i = Q[i] * XM[j]
        got = ds(np.array([x_i * r_s]), np.array([XM[j] * r_s]), lnM)
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
        xmis_edge = XM[0]
        x_edge = xmis_edge * Q[-1]
        edge = ds(np.array([x_edge * r_s]), np.array([xmis_edge * r_s]), lnM)
        beyond = ds(np.array([10.0 * x_edge * r_s]),
                    np.array([0.1 * xmis_edge * r_s]), lnM)
        np.testing.assert_allclose(beyond, edge, rtol=1e-12)

    def test_gamma_kernel_stays_legacy_log_positive(self):
        g = lp.NfwDsigmaMisProduction(kernel="gamma")
        v = g(np.array([0.3]), np.array([2.0]), 33.0)
        self.assertGreater(float(v[0]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
