#!/usr/bin/env python3
"""Unit tests for the pure-Python DES Y3 pipeline and radial decomposition."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PIPELINES = REPO / "src" / "pipelines"
RADIAL_PY = (PIPELINES / "des_y3" / "shear_1h2h" / "python" / "0d")
NUMCOUNTS_FULL_LTMZ_PY = (PIPELINES / "des_y3"
                          / "number_counts" / "python" / "0d")
SHEAR_FAST_MASS_PY = (PIPELINES / "des_y3" / "shear_1h2h"
                      / "python" / "0d")
SHEAR_MAX_PY = (PIPELINES / "des_y3" / "shear_1h2h" / "python" / "0d")
SHEAR_FULL_LTMZ_PY = (PIPELINES / "des_y3" / "shear_1h2h"
                      / "3d" / "python")
sys.path.insert(0, str(PIPELINES))
for _p in (RADIAL_PY, NUMCOUNTS_FULL_LTMZ_PY, SHEAR_FAST_MASS_PY,
          SHEAR_MAX_PY, SHEAR_FULL_LTMZ_PY):
    sys.path.insert(0, str(_p))

from shared import datablock_models as dm  # noqa: E402
from shared import lensing_profiles as lp  # noqa: E402
from systematics.selection_richness.python import sel_kernels  # noqa: E402
import generate_radial_series_tables as generator  # noqa: E402
import nfw_profile_family as pf  # noqa: E402
from shear1h_radial_series import (  # noqa: E402
    RadialSeriesTable,
    evaluate_series,
)
from numcounts_explicit_gl import compute_counts as full_ltmz_counts  # noqa: E402
from shear1h_gl import compute_shear as fast_mass_shear  # noqa: E402
from shear1h2h_max import compute_shear_max, z_resolved_weights  # noqa: E402
from shear1h_explicit_gl import compute_shear as full_ltmz_shear  # noqa: E402

SHEAR_PRJ_FAST_MASS_PY = (PIPELINES / "des_y3"
                          / "shear_projection" / "python" / "0d")
sys.path.insert(0, str(SHEAR_PRJ_FAST_MASS_PY))
from shear_prj_gl import (  # noqa: E402
    ShearPrjGl,
    build_theta_grid,
    theta_excl_at_z,
)

REL_TOL = 1.0e-3

# The 12-bin/10-radius production wall (mock_mcmc_buzzard.ini sel_function
# section) and a real test-sampler dump replayed through it. The dump is
# produced by `cosmosis cosmosis-models/real_pipeline_extract.ini` (CLAUDE.md,
# "Testing precision and cost") and is gitignored, so the backend-vs-
# -production tests below skip gracefully when it hasn't been generated.
DUMP_DIR = REPO / "cosmosis-models" / "real_pipeline_extract_output"
HAS_DUMP = DUMP_DIR.is_dir()
_SKIP_MSG = (f"requires a real-pipeline dump at {DUMP_DIR} -- run "
            "`cosmosis cosmosis-models/real_pipeline_extract.ini` first")

# Sibling dump with the projection chain (b_sel_marg/bsel/dsigma_prj/
# shear_prj_frozen_physics) and halo_model run with compute_lensing_2h = T,
# so haloModel/dSigma_hh is populated. real_pipeline_extract_output has
# neither. Produced by `cosmosis cosmosis-models/real_pipeline_extract_prj2h.ini`.
DUMP_PRJ2H_DIR = REPO / "cosmosis-models" / "real_pipeline_extract_prj2h_output"
HAS_DUMP_PRJ2H = DUMP_PRJ2H_DIR.is_dir()
_SKIP_MSG_PRJ2H = (
    f"requires a real-pipeline dump at {DUMP_PRJ2H_DIR} -- run "
    "`cosmosis cosmosis-models/real_pipeline_extract_prj2h.ini` first")

# The 180-point zipped wall (mock_mcmc_buzzard.ini shear_prj_frozen_physics
# section): 4 lambda bins x 3 zob bins x 15 radii.
_PRJ_RADII = [0.0426, 0.0669, 0.1045, 0.1652, 0.2607, 0.4117, 0.6505, 1.0257,
             1.6181, 2.5537, 4.0265, 6.3490, 10.0107, 15.7832, 24.8771]


def _pinned_prj_wall():
    lb, zol, zoh, rr = [], [], [], []
    for zlo, zhi in ((0.20, 0.35), (0.35, 0.50), (0.50, 0.65)):
        for b in range(4):
            lb += [b] * 15
            zol += [zlo] * 15
            zoh += [zhi] * 15
            rr += _PRJ_RADII
    return dict(lambda_bin=np.array(lb), zo_low=np.array(zol),
               zo_high=np.array(zoh), radii=np.array(rr))

ZT_LO, ZT_HI = 0.05, 0.80
LNM_LO, LNM_HI = 29.9336, 36.7300
BINS_12 = dict(
    lam_min=np.array([20., 30., 45., 60.] * 3),
    lam_max=np.array([30., 45., 60., 200.] * 3),
    zob_min=np.array([0.20] * 4 + [0.35] * 4 + [0.50] * 4),
    zob_max=np.array([0.35] * 4 + [0.50] * 4 + [0.65] * 4),
    sigma_z=np.full(12, 0.03),
)
R_PERP = np.array([0.20000, 0.28599, 0.40896, 0.58480, 0.83625,
                   1.19581, 1.70998, 2.44521, 3.49658, 5.00000])


class _SyntheticBselSource:
    """Minimal DataBlockSource stand-in: the exact b_sel_marginalised
    production contract for two wall rows, no dump required."""

    _DATA = {
        ("b_sel_marginalised", "lambda_bin"): np.array([0, 1]),
        ("b_sel_marginalised", "zo_low"): np.array([0.20, 0.20]),
        ("b_sel_marginalised", "zo_high"): np.array([0.35, 0.35]),
        ("b_sel_marginalised", "zob"): np.array([0.275, 0.275]),
        ("b_sel_marginalised", "lob"): np.array([25.0, 37.5]),
        ("b_sel_marginalised", "b_small"): np.array([1.1, 1.3]),
        ("b_sel_marginalised", "b_large"): np.array([2.2, 2.6]),
    }

    def has(self, section, key):
        return (section, key) in self._DATA

    def array(self, section, key):
        return self._DATA[(section, key)]


class TestBSelBinsContract(unittest.TestCase):
    """Dump-free regression guards for the refactored bsel consumers.

    Both defects below were invisible until the wall-metadata contract
    gap (issue #10) was closed and the chain first ran end-to-end; the
    dump-gated TestShearPrjFastMass end-to-end test also exercises them,
    but only on machines that have regenerated the fiducial dump."""

    def test_from_source_returns_the_bins_object_not_none(self):
        # BSelBins.validate() used to return None, and from_source
        # returned output.validate() -- every caller got None.
        bins = dm.BSelBins.from_source(_SyntheticBselSource())
        self.assertIsInstance(bins, dm.BSelBins)

    def test_find_exact_row_usable_across_repeated_lookups(self):
        # ShearPrjFastMass.set_sample() used to clobber its `bsel`
        # BSelBins object with a per-theta ndarray inside the slice
        # loop, so any second lookup raised AttributeError.
        bins = dm.BSelBins.from_source(_SyntheticBselSource())
        for lam_bin, lob_want, bs_want in ((0, 25.0, 1.1), (1, 37.5, 1.3)):
            lob, zob, b_small, b_large = bins.find_exact_row(
                lam_bin, zob=0.275)
            self.assertEqual(lob, lob_want)
            self.assertEqual(zob, 0.275)
            self.assertEqual(b_small, bs_want)
            self.assertEqual(b_large, 2.0 * bs_want)


class TestSharedQuadratureAndMoments(unittest.TestCase):
    def test_gl_nodes_integrate_polynomials(self):
        x, w = dm.gl_nodes(-0.7, 1.4, 8)
        for power in range(0, 16):
            got = np.dot(w, x**power)
            expected = (1.4**(power + 1) - (-0.7)**(power + 1)) \
                / (power + 1)
            self.assertAlmostEqual(got, expected, places=12)

    def test_population_moments_are_normalized_and_central(self):
        # Exercise MassZWeights.moments_of without a DataBlock: W and GL
        # weights are the complete state consumed by the method.
        weights = object.__new__(dm.MassZWeights)
        weights.lnm_x = np.array([0.0, 1.0, 2.0])
        weights.lnm_w = np.ones(3)
        weights.n_bins = 2
        weights.W = np.array([
            [0.25, 0.50, 0.25],
            [0.60, 0.30, 0.10],
        ])

        norm, ybar, mu = weights.moments_of(lambda x: x, ell_max=3)
        np.testing.assert_allclose(norm, 1.0, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(ybar, [1.0, 0.5], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(mu[:, 0], 1.0, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(mu[:, 1], 0.0, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(mu[0, 2:], [0.5, 0.0],
                                   rtol=0.0, atol=1e-15)


class TestNfwDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = RadialSeriesTable()

    def test_mass_scale_radius_round_trip(self):
        lnm = np.linspace(29.9336, 36.7300, 21)
        np.testing.assert_allclose(
            pf.lnM_of_y(pf.y_of_lnM(lnm)), lnm, rtol=0.0, atol=2e-14)

    def test_nfw_shape_is_stable_through_x_equal_one(self):
        import mpmath as mp

        mp.mp.dps = 40
        x = np.array([0.9901, 0.999999, 1.0, 1.000001, 1.0099])
        got = pf.u_cen(x)
        expected = np.array([float(pf.u_cen_mp(v, mp)) for v in x])
        np.testing.assert_allclose(got, expected, rtol=REL_TOL, atol=0.0)

    def test_mixture_endpoints_and_density_factor(self):
        lnx = np.array([-1.2, 0.1, 1.7])
        lnxm = np.array([-0.8, -0.4, 0.2])
        rho_mult = 0.3096
        for ell in range(4):
            cen = self.table.u_cen(ell, lnx)
            mis = self.table.u_mis(ell, lnx, lnxm)
            np.testing.assert_allclose(
                self.table.u_mix(ell, lnx, lnxm, 0.0, rho_mult), cen,
                rtol=0.0, atol=1e-14)
            np.testing.assert_allclose(
                self.table.u_mix(ell, lnx, lnxm, 1.0, rho_mult),
                rho_mult * mis, rtol=0.0, atol=1e-14)

    def test_series_matches_direct_skewed_mass_population(self):
        r = np.array([0.20, 0.84, 3.0, 10.0])
        r_mis = 0.15
        norm = 137.0
        ybar = math.log(0.30)
        f_mis = 0.22
        rho_mult = 0.3096

        # Mean displacement is zero and the third central moment is nonzero.
        dy = np.array([-0.03, 0.06])
        probability = np.array([2.0 / 3.0, 1.0 / 3.0])
        mu = np.array([1.0, np.dot(probability, dy),
                       np.dot(probability, dy**2),
                       np.dot(probability, dy**3)])
        self.assertAlmostEqual(mu[1], 0.0, places=15)
        self.assertNotEqual(mu[3], 0.0)

        direct = np.zeros_like(r)
        for displacement, weight in zip(dy, probability):
            y = ybar + displacement
            lnx = np.log(r) - y
            lnxm = np.full_like(r, math.log(r_mis) - y)
            direct += weight * pf.A0_of_y(y) * self.table.u_mix(
                0, lnx, lnxm, f_mis, rho_mult)
        direct *= norm

        series = evaluate_series(
            self.table, r, r_mis, norm, ybar, mu,
            f_mis=f_mis, rho_mult=rho_mult, ell_max=3)
        np.testing.assert_allclose(series, direct, rtol=REL_TOL, atol=0.0)


class TestOfflineRadialGenerator(unittest.TestCase):
    def test_finite_difference_weights_recover_polynomial_derivatives(self):
        offsets = np.arange(-4, 5)
        h = 0.0125
        # p(s) = sum c_k s^k; the order-n derivative at zero is n! c_n.
        coefficients = np.array([1.1, -0.7, 0.4, 0.2, -0.1,
                                 0.05, 0.01, -0.02, 0.005])
        samples = np.polynomial.polynomial.polyval(offsets * h,
                                                   coefficients)
        for order in range(4):
            got = np.dot(generator.fd_weights(order, offsets, h), samples)
            expected = math.factorial(order) * coefficients[order]
            self.assertAlmostEqual(got, expected, places=8)

    def test_gamma_average_matrix_preserves_a_constant_profile(self):
        h = 0.0125
        lnv = np.arange(-10.0, 7.0 + 0.5 * h, h)
        self.assertEqual(lnv.size % 2, 1)
        lnxm = np.array([-3.0, 0.0, 2.0])
        q = generator.gamma_average_matrix(lnv, lnxm, h)
        # Integral w exp(-w) dw = 1. The analytic below-grid tail is
        # included by gamma_average_matrix; the upper tail is negligible.
        np.testing.assert_allclose(q.sum(axis=0), 1.0,
                                   rtol=REL_TOL, atol=0.0)


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestNumCountsSijGl(unittest.TestCase):
    """des_y3/number_counts/python/0d/numcounts_sij_gl.py"""

    def test_matches_production_numcountssel(self):
        source = dm.DumpSource(str(DUMP_DIR))
        norm = dm.MassZWeights(
            source, n_lnm=96, n_z=64, zt_lo=ZT_LO, zt_hi=ZT_HI,
            lnm_lo=LNM_LO, lnm_hi=LNM_HI, include_sci=False).norm()
        prod = source.array("numcountssel", "vals")
        # Same algorithm, same GL nodes, same S_stack interpolation as
        # NumCountsSel.so -- measured 2.4e-15.
        np.testing.assert_allclose(norm, prod, rtol=REL_TOL, atol=0.0)


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestNumCounts3d(unittest.TestCase):
    """des_y3/number_counts/python/0d/numcounts_explicit_gl.py"""

    def test_matches_production_within_tabulation_error(self):
        source = dm.DumpSource(str(DUMP_DIR))
        mor = sel_kernels.mor_from_source(source)
        plob = sel_kernels.plob_splines_default()
        vals = full_ltmz_counts(
            BINS_12, mor, plob, dm.HMF(source), dm.DVDoDz(source),
            zt_low=ZT_LO, zt_high=ZT_HI, lnm_low=LNM_LO, lnm_high=LNM_HI)
        prod = source.array("numcountssel", "vals")
        # The explicit (lambda_tr, lnM, z) integral disagrees with the
        # tabulated fast path only through the S_ij tabulation error
        # (measured 7.6e-4; the module's own documented ceiling is 5e-3).
        np.testing.assert_allclose(vals, prod, rtol=REL_TOL, atol=0.0)


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestShear1hGl(unittest.TestCase):
    """des_y3/shear_1h2h/python/0d/shear1h_gl.py"""

    def test_matches_production_shear1hmissel(self):
        source = dm.DumpSource(str(DUMP_DIR))
        weights = dm.MassZWeights(
            source, n_lnm=96, n_z=64, zt_lo=ZT_LO, zt_hi=ZT_HI,
            lnm_lo=LNM_LO, lnm_hi=LNM_HI, include_sci=True)
        profile = lp.MisMixtureProfile(
            source, lob_centers=dm.DEFAULT_LOB_CENTERS,
            f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
            omega_m=source.scalar("cosmological_parameters", "omega_m"))
        vals = fast_mass_shear(weights, profile, np.arange(12), R_PERP)
        prod = source.array("shear1hmissel", "vals")
        # measured 3.1e-15 vs Shear1hMisSel.so (method = exact).
        np.testing.assert_allclose(vals, prod, rtol=REL_TOL, atol=0.0)


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestShear1h3d(unittest.TestCase):
    """des_y3/shear_1h2h/python/0d/shear1h_explicit_gl.py"""

    def test_matches_production_within_tabulation_error(self):
        source = dm.DumpSource(str(DUMP_DIR))
        mor = sel_kernels.mor_from_source(source)
        plob = sel_kernels.plob_splines_default()
        profile = lp.MisMixtureProfile(
            source, lob_centers=dm.DEFAULT_LOB_CENTERS,
            f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
            omega_m=source.scalar("cosmological_parameters", "omega_m"))
        sci = dm.SigmaCritInv(source)
        vals = full_ltmz_shear(
            BINS_12, mor, plob, dm.HMF(source), dm.DVDoDz(source), sci,
            profile, np.arange(12), R_PERP,
            zt_low=ZT_LO, zt_high=ZT_HI, lnm_low=LNM_LO, lnm_high=LNM_HI)
        prod = source.array("shear1hmissel", "vals")
        # measured 8.4e-4 (S_ij tabulation error; module's own bound 5e-3).
        np.testing.assert_allclose(vals, prod, rtol=REL_TOL, atol=0.0)


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestShear1h2hMax(unittest.TestCase):
    """des_y3/shear_1h2h/python/0d/shear1h2h_max.py

    The available dump has halo_model run with compute_lensing_2h = F --
    haloModel/dSigma_hh has 3 open defects (see
    docs/known_issues/dsigma_hh_debug_flag.md), so any traditional 1h+2h *sum* is
    provisional until they are fixed. This test therefore exercises only
    the defect-free limit the module's own validate_shear1h2h_max.py
    calls its "2h -> 0 sanity" check: with the two-halo term forced to
    zero, max(one, 0) == one for a physical (non-negative) profile, so
    the max-model composition must reproduce the validated 1h fast_mass
    backend exactly.
    """

    def test_two_halo_zero_limit_matches_1h_fast_mass(self):
        source = dm.DumpSource(str(DUMP_DIR))
        omega_m = source.scalar("cosmological_parameters", "omega_m")
        one_profile = lp.MisMixtureProfile(
            source, lob_centers=dm.DEFAULT_LOB_CENTERS,
            f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
            omega_m=omega_m)

        # Bypass MaxMixtureProfile.__init__, which eagerly reads
        # haloModel/dSigma_hh -- irrelevant here since _hh is zeroed.
        profile0 = object.__new__(lp.MaxMixtureProfile)
        profile0._one = one_profile
        profile0._bias = dm.Bilinear2D(source, "halomodel", "lnm", "z",
                                       "bias")
        profile0._hh = lambda *a: np.zeros(
            np.broadcast_shapes(*[np.shape(x) for x in a]))

        lnm_x, lnm_w, z_x, w2d = z_resolved_weights(
            source, n_lnm=96, n_z=64, zt_lo=ZT_LO, zt_hi=ZT_HI,
            lnm_lo=LNM_LO, lnm_hi=LNM_HI)
        fast0 = compute_shear_max(profile0, lnm_x, lnm_w, z_x, w2d,
                                  np.arange(12), R_PERP)

        weights = dm.MassZWeights(
            source, n_lnm=96, n_z=64, zt_lo=ZT_LO, zt_hi=ZT_HI,
            lnm_lo=LNM_LO, lnm_hi=LNM_HI, include_sci=True)
        oneh = fast_mass_shear(weights, one_profile, np.arange(12), R_PERP)
        np.testing.assert_allclose(fast0, oneh, rtol=1e-12, atol=0.0)

    @unittest.skipUnless(HAS_DUMP_PRJ2H, _SKIP_MSG_PRJ2H)
    def test_full_model_is_finite_and_two_halo_contributes(self):
        # With a real (NaN-heavy, see docs/known_issues/dsigma_hh_debug_flag.md)
        # haloModel/dSigma_hh table: the max-model composition must still
        # produce finite output (the NaN-sanitize-before-interpolate
        # convention keeps NaN out of the profile), and it must genuinely
        # differ from the 1h-only limit (the 2-halo branch actually gets
        # selected somewhere on the wall) -- no golden numeric value is
        # pinned, since that would bake in the known table defects.
        source = dm.DumpSource(str(DUMP_PRJ2H_DIR))
        omega_m = source.scalar("cosmological_parameters", "omega_m")
        profile = lp.MaxMixtureProfile(
            source, lob_centers=dm.DEFAULT_LOB_CENTERS,
            f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
            omega_m=omega_m)
        lnm_x, lnm_w, z_x, w2d = z_resolved_weights(
            source, n_lnm=96, n_z=64, zt_lo=ZT_LO, zt_hi=ZT_HI,
            lnm_lo=LNM_LO, lnm_hi=LNM_HI)
        full = compute_shear_max(profile, lnm_x, lnm_w, z_x, w2d,
                                 np.arange(12), R_PERP)
        self.assertTrue(np.all(np.isfinite(full)))

        one_profile = lp.MisMixtureProfile(
            source, lob_centers=dm.DEFAULT_LOB_CENTERS,
            f_mis=dm.F_MIS_DEFAULT, tau_mis=dm.TAU_MIS_DEFAULT,
            omega_m=omega_m)
        weights = dm.MassZWeights(
            source, n_lnm=96, n_z=64, zt_lo=ZT_LO, zt_hi=ZT_HI,
            lnm_lo=LNM_LO, lnm_hi=LNM_HI, include_sci=True)
        oneh = fast_mass_shear(weights, one_profile, np.arange(12), R_PERP)
        # max(1h, b*2h) >= 1h everywhere (up to the two paths' independent
        # floating-point summation order, demonstrated at 1e-12 relative
        # by the zero-limit test above), with strict inequality somewhere.
        slack = 1e-8 * np.abs(oneh) + 1e-300
        self.assertTrue(np.all(full >= oneh - slack))
        self.assertGreater(np.max(full - oneh), 0.0)


class TestShearPrjGl(unittest.TestCase):
    """des_y3/shear_projection/python/0d/shear_prj_gl.py

    ShearPrjGl.set_sample() needs a full real-pipeline sample (HMF,
    halo bias, xi_nl, distances, b_sel_marginalised) that no dump checked
    into (or regenerable from) this repo currently provides -- see the C++
    sibling test/shear_prj_gl.test.cc for the same conclusion, which
    pins the DataBlock-free building blocks instead. This test does the
    same for the Python port: build_theta_grid / theta_excl_at_z, checked
    both against independently-derived closed-form bounds (non-circular)
    and the golden regression values also pinned on the C++ side.
    """

    LOBC = 37.5
    ZOB = 0.42
    CHI_O = 1085.0  # Mpc/h
    R_VEC = [0.4117, 1.0257, 4.0265]
    N_PER_SEG = 6
    R_MAX_CMPCH = 35.0

    def test_theta_grid_is_bounded_monotone_and_matches_golden(self):
        r_excl = lp.r_lambda(self.LOBC) * (1.0 + self.ZOB)
        d_a_o = self.CHI_O / (1.0 + self.ZOB)
        theta, weight = build_theta_grid(
            self.LOBC, self.ZOB, self.R_VEC, self.CHI_O, d_a_o, r_excl,
            self.N_PER_SEG, self.R_MAX_CMPCH)

        # Bounds from the same closed-form breakpoint recipe
        # build_theta_grid documents -- independent of the golden numbers.
        theta_lam = lp.r_lambda(self.LOBC) * (1.0 + self.ZOB) / self.CHI_O
        theta_excl_o = r_excl / self.CHI_O
        theta_r = np.asarray(self.R_VEC) / d_a_o
        theta_max = max(self.R_MAX_CMPCH / d_a_o, 3.0 * theta_r.max())
        lower = max(1.0e-8, 0.1 * min(theta_excl_o, theta_r.min(), theta_lam))

        self.assertEqual(theta.size, 36)  # 6 dedup'd log-segments x 6
        self.assertGreater(theta[0], lower)
        self.assertLess(theta[-1], theta_max)
        self.assertTrue(np.all(np.diff(theta) > 0))

        # dtheta Jacobian folded into weight -> summing weights is a fixed-GL
        # quadrature of int_lower^theta_max dtheta = theta_max - lower.
        sum_w = float(weight.sum())
        self.assertAlmostEqual(sum_w, theta_max - lower, delta=1e-6 * sum_w)

        # Golden values, also pinned in test/shear_prj_gl.test.cc.
        np.testing.assert_allclose(theta[0], 5.8237767207010004e-05,
                                   rtol=REL_TOL)
        np.testing.assert_allclose(theta[theta.size // 2],
                                   0.0013639379714246158, rtol=REL_TOL)
        np.testing.assert_allclose(theta[-1], 0.04258104827263785,
                                   rtol=REL_TOL)
        np.testing.assert_allclose(sum_w, 0.0457525701381797, rtol=REL_TOL)

    def test_theta_exclusion_angle_matches_golden(self):
        r_excl = lp.r_lambda(self.LOBC) * (1.0 + self.ZOB)

        # Far outside the exclusion ring: the law-of-cosines argument clips
        # to >= 1, so the exclusion angle is exactly zero for any
        # chi_o >> r_excl, independent of the golden values below.
        self.assertAlmostEqual(
            theta_excl_at_z(self.CHI_O - 2.0 * r_excl, self.CHI_O, r_excl),
            0.0, delta=1e-12)
        self.assertAlmostEqual(
            theta_excl_at_z(self.CHI_O + 2.0 * r_excl, self.CHI_O, r_excl),
            0.0, delta=1e-12)

        np.testing.assert_allclose(
            theta_excl_at_z(self.CHI_O - 0.5 * r_excl, self.CHI_O, r_excl),
            0.0009317777252103878, rtol=REL_TOL)
        np.testing.assert_allclose(
            theta_excl_at_z(self.CHI_O, self.CHI_O, r_excl),
            0.0010756348895233233, rtol=REL_TOL)
        np.testing.assert_allclose(
            theta_excl_at_z(self.CHI_O + 0.5 * r_excl, self.CHI_O, r_excl),
            0.0009312767333775487, rtol=REL_TOL)

    @unittest.skipUnless(HAS_DUMP_PRJ2H, _SKIP_MSG_PRJ2H)
    def test_matches_exact_evaluator_and_frozen_physics(self):
        # Full end-to-end ShearPrjGl.set_sample()/wall_outputs()
        # against a real sample with the projection chain populated
        # (cosmosis-models/real_pipeline_extract_prj2h.ini): machine precision
        # vs the exact DSigmaPrjEvaluator.so (same core), and within the
        # documented frozen-physics approximation vs production.
        source = dm.DumpSource(str(DUMP_PRJ2H_DIR))
        # n_lnm=24 matches cosmosis-models/real_pipeline_extract_prj2h.ini's
        # [dsigma_prj] section (the class default of 16 is a different,
        # coarser resolution and would only agree at the ~0.2% level).
        core = ShearPrjGl(_pinned_prj_wall(), n_lnm=24,
                                lob_centers=dm.DEFAULT_LOB_CENTERS)
        core.set_sample(source)
        rnd, cl, _sci = core.wall_outputs()
        vals = rnd + cl

        ex_r = source.array("dsigma_prj", "rnd")
        ex_c = source.array("dsigma_prj", "cl")
        ex_v = source.array("dsigma_prj", "vals")
        np.testing.assert_allclose(rnd, ex_r, rtol=1e-8, atol=0.0)
        np.testing.assert_allclose(cl, ex_c, rtol=1e-8, atol=0.0)
        np.testing.assert_allclose(vals, ex_v, rtol=1e-8, atol=0.0)

        fr_v = source.array("dsigma_prj_frozen_physics", "vals")
        # Documented frozen-physics approximation bound is < 0.2%; use a
        # looser default so this doesn't flake on the fiducial sample.
        np.testing.assert_allclose(vals, fr_v, rtol=3e-3, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
