#!/usr/bin/env python3
"""Unit tests for
``src/pipelines/des_y3/shear_1h2h/python/0d/nfw_profile_family.py``.

Pure-function module (no CosmoSIS ``setup``/``execute``). Existing
coverage before this change: ``test/des_y3_pipeline.test.py`` already
pins ``y_of_lnM``/``lnM_of_y`` round-tripping, ``u_cen`` vs ``u_cen_mp``
near ``x = 1``, and the ``u_mix`` endpoints; the module's own
``validate_radial_series.py`` exercises ``g_shape``/``g_shape_mp``
through the offline table's own self-checks.

Left genuinely uncovered before this file: ``sigma_shape`` (the Sigma,
not DeltaSigma, shape -- used by ``generate_radial_series_tables.py``'s
line-of-sight convolution), ``sigma_shape_mp``, ``dsigma_cen``,
``make_dsigma_mis``, and ``MisTable.sha256``/``partial_w``. Each is
checked here against an INDEPENDENT derivation, not against another
copy of the same formula:

* ``sigma_shape`` against a direct numerical Abel projection of the
  untruncated NFW density profile (``scipy.integrate.quad``, not copied
  from the module) -- the same kind of independent-projection check the
  module's docstring says the radial-factorization study already did for
  ``g_shape`` (to 1.6e-9);
* ``sigma_shape_mp``/``g_shape_mp`` (mpmath) against their float
  counterparts at ordinary x (where float is already accurate) --
  isolating "is the mpmath port doing the same math" from precision;
* ``dsigma_cen``/``make_dsigma_mis`` against a from-scratch expansion of
  their one-line formulas (``A0_of_y(y) * u_cen/u(...)``), which is a
  meaningful regression pin because these two helpers are otherwise
  never exercised anywhere in the tree.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = (REPO / "src" / "pipelines" / "des_y3" / "shear_1h2h"
             / "python" / "0d")
sys.path.insert(0, str(MODULE_DIR))

import nfw_profile_family as pf   # noqa: E402

mp.mp.dps = 40


def _direct_sigma_shape(x):
    """2 * int_0^inf du / [sqrt(x^2+u^2) (1+sqrt(x^2+u^2))^2].

    The Abel (line-of-sight) projection of the untruncated dimensionless
    NFW density 1/(r(1+r)^2), independent of ``pf.sigma_shape``'s
    closed-form branches.
    """
    def integrand(u):
        r = np.sqrt(x * x + u * u)
        return 1.0 / (r * (1.0 + r) ** 2)
    val, _err = quad(integrand, 0.0, np.inf, limit=200)
    return 2.0 * val


class TestSigmaShape(unittest.TestCase):
    X_GRID = np.array([0.05, 0.3, 0.7, 0.95, 0.999, 1.0, 1.001, 1.05,
                       1.5, 4.0, 20.0])

    def test_matches_the_direct_abel_projection(self):
        for x in self.X_GRID:
            mine = float(pf.sigma_shape(np.array([x]))[0])
            direct = _direct_sigma_shape(float(x))
            self.assertAlmostEqual(mine / direct, 1.0, places=8,
                                   msg=f"x={x}")

    def test_at_x_equals_one_matches_the_taylor_constant_term(self):
        # The near-branch's own hardcoded constant (F_TAYLOR_1[0] = 2/3)
        # must agree with the far branches evaluated in the limit.
        self.assertAlmostEqual(float(pf.sigma_shape(np.array([1.0]))[0]),
                               2.0 / 3.0, places=12)

    def test_continuous_across_the_near_branch_boundary(self):
        # |x-1| = 0.01 is exactly where sigma_shape switches from the
        # closed-form branches to the degree-10 Taylor expansion.
        for edge in (0.99, 1.01):
            lo = float(pf.sigma_shape(np.array([edge - 1e-6]))[0])
            hi = float(pf.sigma_shape(np.array([edge + 1e-6]))[0])
            self.assertAlmostEqual(lo, hi, delta=5e-6, msg=f"edge={edge}")

    def test_positive_finite_and_monotonically_falling(self):
        vals = pf.sigma_shape(self.X_GRID)
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertTrue(np.all(vals > 0.0))
        self.assertTrue(np.all(np.diff(vals) < 0.0))


class TestMpmathShapeFunctions(unittest.TestCase):
    """mpmath ports vs their already-accurate float siblings."""

    def test_sigma_shape_mp_matches_sigma_shape_away_from_x_equal_one(self):
        for x in (0.2, 0.6, 2.0, 10.0):
            got = float(pf.sigma_shape_mp(x, mp))
            want = float(pf.sigma_shape(np.array([x]))[0])
            self.assertAlmostEqual(got / want, 1.0, places=10, msg=f"x={x}")

    def test_sigma_shape_mp_hits_its_own_x_equal_one_branch(self):
        self.assertAlmostEqual(float(pf.sigma_shape_mp(1.0, mp)), 2.0 / 3.0,
                               places=12)

    def test_g_shape_float_hits_its_own_x_equal_one_mid_branch(self):
        # |x - 1| <= 1e-6 is the float g_shape's separate constant branch
        # (distinct from sigma_shape's own near-1 Taylor window).
        val = float(pf.g_shape(np.array([1.0]))[0])
        expected = 10.0 / 3.0 + 4.0 * np.log(0.5)
        self.assertAlmostEqual(val, expected, places=12)

    def test_g_shape_mp_matches_g_shape_away_from_x_equal_one(self):
        for x in (0.2, 0.6, 2.0, 10.0):
            got = float(pf.g_shape_mp(mp.mpf(x), mp))
            want = float(pf.g_shape(np.array([x]))[0])
            self.assertAlmostEqual(got / want, 1.0, places=10, msg=f"x={x}")

    def test_g_shape_mp_hits_its_own_x_equal_one_branch(self):
        # abs(x - 1) < 1e-20 -- only reachable with an EXACT mpf(1).
        val = pf.g_shape_mp(mp.mpf(1), mp)
        expected = mp.mpf(10) / 3 + 4 * mp.log(mp.mpf(1) / 2)
        self.assertAlmostEqual(float(val), float(expected), places=12)

    def test_u_cen_mp_is_half_g_shape_mp(self):
        for x in (0.5, 1.0, 3.0):
            self.assertAlmostEqual(
                float(pf.u_cen_mp(x, mp)),
                float(pf.g_shape_mp(mp.mpf(x), mp)) / 2.0, places=12)


class TestDsigmaHelpers(unittest.TestCase):
    """``dsigma_cen``/``make_dsigma_mis`` -- never exercised elsewhere."""

    RHO_REF = 0.31 * pf.RHOC
    LNM = np.array([31.0, 32.5, 34.0])
    R = np.array([0.3, 1.0, 3.0])

    def test_dsigma_cen_matches_its_own_one_line_definition(self):
        y = pf.y_of_lnM(self.LNM, self.RHO_REF)
        expected = pf.A0_of_y(y, self.RHO_REF) * pf.u_cen(
            self.R[:, None] * np.exp(-y)[None, :])
        got = pf.dsigma_cen(self.R[:, None], self.LNM[None, :], self.RHO_REF)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)

    def test_dsigma_cen_is_positive_finite_and_falls_with_radius(self):
        vals = pf.dsigma_cen(self.R, self.LNM[1], self.RHO_REF)
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertTrue(np.all(vals > 0.0))
        self.assertTrue(np.all(np.diff(vals) < 0.0))

    def test_make_dsigma_mis_default_table_matches_a_manual_mistable(self):
        manual_table = pf.MisTable()
        dsigma_mis = pf.make_dsigma_mis()   # table=None -> builds its own
        r_mis = 0.15
        y = pf.y_of_lnM(self.LNM, self.RHO_REF)
        lnx = np.log(self.R[:, None]) - y[None, :]
        lnxm = np.log(r_mis) - y[None, :]
        expected = pf.A0_of_y(y, self.RHO_REF)[None, :] * manual_table.u(
            lnx, lnxm)
        got = dsigma_mis(self.R[:, None], r_mis, self.LNM[None, :],
                         self.RHO_REF)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=0.0)

    def test_make_dsigma_mis_accepts_an_injected_table(self):
        table = pf.MisTable()
        dsigma_mis = pf.make_dsigma_mis(table)
        vals = dsigma_mis(self.R, 0.2, self.LNM[0], self.RHO_REF)
        self.assertTrue(np.all(np.isfinite(vals)))


class TestMisTableMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = pf.MisTable()

    def test_sha256_covers_every_source_file_with_valid_hex_digests(self):
        digests = self.table.sha256()
        self.assertEqual(set(digests), set(pf.GAMMA_TABLE_FILES))
        for name, digest in digests.items():
            self.assertEqual(len(digest), 64, name)
            int(digest, 16)   # raises ValueError if not valid hex

    def test_partial_w_of_order_zero_returns_the_raw_table(self):
        raw = self.table.partial_w(0, 0)
        np.testing.assert_array_equal(raw, self.table.w)
        # Must be a copy, not a view, so callers cannot corrupt the cache.
        raw[0, 0] += 1.0
        self.assertNotEqual(raw[0, 0], self.table.w[0, 0])

    def test_partial_w_first_order_is_finite_on_the_full_grid(self):
        d = self.table.partial_w(1, 0)
        self.assertEqual(d.shape, self.table.w.shape)
        self.assertTrue(np.all(np.isfinite(d)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
