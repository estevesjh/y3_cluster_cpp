#!/usr/bin/env python3
"""Unit tests for the survey-agnostic physics package ``src/pipelines/cosmology/``.

This package (halo_model, concentration, nfw_model, bsel, sigma_crit_inv,
hydro_mc, prj_params) had NO test of its own. ``halo_model.test.py`` and
``conc_mass_relation.test.py`` import the ``y3_buzzard`` ORIGINALS, which
have drifted from these copies by hundreds of lines -- and only one of the
two homes survives the pipelines/des_y3 merge, so the untested one is the
one that ships.

Two independent legs:

1. EXTERNAL validation (never against the y3_buzzard copy, which is a
   sibling of the code under test, not a reference):
     * Tinker et al. 2010 Eq. 6 halo bias vs ``cluster_toolkit.bias``;
     * Wright & Brainerd 2000 NFW Sigma / DeltaSigma vs
       ``cluster_toolkit.deltasigma`` (Sigma_nfw_at_R / DeltaSigma_at_R),
       in the Msun/h, Mpc/h, 200x-mean convention the code itself uses;
     * Child et al. 2018 (ApJ 859, 55) Eq. 19 + Table 2 coefficients and
       the M*(z) fit, re-typed from the PAPER;
     * Duffy et al. 2008 (MNRAS 390, L64) Table 1 full sample;
     * the ratified rho_m convention (comoving rho_m0 = Omega_m rho_crit,0,
       with (1+z)^3 entering only where the code says it does);
     * the frozen Costanzi-2026 EMG projection coefficient grid.

2. EQUIVALENCE with the y3_buzzard originals. Measured today: every shared
   physics entry point is numerically IDENTICAL (0.0 relative deviation)
   despite the textual drift, and ``hydro_mc.py`` is byte-identical. That
   is the fact worth pinning: this test goes red the day the two homes
   start disagreeing, which is exactly when the "which copy is production"
   question becomes urgent.

KNOWN DEFECT recorded, not fixed (see TestBiasAtMKnownDefect): in BOTH
copies ``biasModel.bias_at_M`` calls ``self.compute_nu`` -- a method that
does not exist -- and then passes ``odelta=`` to a one-argument
``bias_at_nu``. The method cannot ever have run. The working entry points
are ``nu_at_M`` + ``bias_at_nu``, which is what production uses, so this is
a dead-code bug rather than a live one; the test asserts the CURRENT broken
behaviour so it flips to red when someone repairs it.
"""
from __future__ import annotations

import hashlib
import sys
import types
import unittest
import warnings
from pathlib import Path

import numpy as np

import cluster_toolkit.bias as ctb
import cluster_toolkit.deltasigma as ctd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "pipelines"))
sys.path.insert(0, str(REPO / "y3_buzzard"))

from cosmology import concentration as conc           # noqa: E402
from cosmology import halo_model as hm                # noqa: E402
from cosmology import nfw_model as nfw                # noqa: E402
from cosmology.prj_params import COEFF_NAMES, PrjParams  # noqa: E402

# The analytic NFW helpers warn on the x -> 1 branch of arctanh; that
# branch is handled by the f_nfw/g_nfw piecewise split and is not what
# these tests are about.
warnings.filterwarnings("ignore", category=RuntimeWarning)

REL_TOL = 1.0e-3          # the project default
RHOC0 = 2.77533742639e+11  # Msun/Mpc^3/h^2
OMEGA_M = 0.3
RHOM0 = OMEGA_M * RHOC0

# Child et al. 2018 Table 2 (M200c), re-typed from the paper:
#   sample:              (m,     A,     b,       c0)
CHILD18_TABLE2 = {
    "individual_all": (-0.10, 3.44, 430.49, 3.19),
    "individual_relaxed": (-0.09, 2.88, 1644.53, 3.54),
    "stacked_nfw": (-0.07, 4.61, 638.65, 3.59),
    "stacked_einasto": (-0.01, 63.2, 431.48, 3.36),
}

# Duffy et al. 2008 Table 1, full sample, M200 critical.
DUFFY_A, DUFFY_B, DUFFY_C = 7.85, -0.081, -0.71


def _linear_pk(k):
    """A smooth, monotone stand-in P(k) with a turnover.

    Not a fiducial spectrum -- peak height and Tinker bias only need a
    well-behaved P(k) on a wide k range, and using a synthetic one keeps
    this test free of any CAMB/dump dependency.
    """
    return 2.0e4 * (k / 0.05) ** (-1.0) / (1.0 + (k / 0.1) ** 2)


K_GRID = np.logspace(-4.0, 2.0, 500)
P_GRID = _linear_pk(K_GRID)
M_GRID = np.array([1.0e13, 3.0e13, 1.0e14, 3.0e14, 1.0e15])
R_GRID = np.logspace(-1.0, 0.7, 8)


class TestNfwProfileVsClusterToolkit(unittest.TestCase):
    """Wright & Brainerd (2000) Sigma/DeltaSigma vs an independent code."""

    def test_sigma_matches_cluster_toolkit(self):
        # cluster_toolkit's Sigma_nfw_at_R uses r200 = [3M/(800 pi
        # Omega_m rho_crit,0)]^(1/3) -- the same 200x-comoving-mean
        # boundary this package builds when it is handed rho_c = rho_m0.
        # nfw_model returns Msun/(h Mpc^2); the /1e12 to Msun h/pc^2 is the
        # conversion halo_model.first_halo_term applies explicitly.
        for mass in (1.0e13, 1.0e14, 1.0e15):
            for c in (3.0, 4.0, 5.0):
                mine = nfw.sigmaNFW_Analytical(R_GRID, mass, c,
                                               rho_c=RHOM0) / 1.0e12
                ref = ctd.Sigma_nfw_at_R(R_GRID, mass, c, OMEGA_M)
                dev = np.max(np.abs(mine / ref - 1.0))
                self.assertLess(dev, REL_TOL, f"M={mass:.1e} c={c}: {dev:.2e}")

    def test_delta_sigma_matches_cluster_toolkit(self):
        for mass in (1.0e13, 1.0e14, 1.0e15):
            for c in (3.0, 4.0, 5.0):
                mine = nfw.deltaSigmaNFW_Analytical(R_GRID, mass, c,
                                                    rho_c=RHOM0) / 1.0e12
                r_dense = np.logspace(-3.0, 3.0, 2000)
                sig = ctd.Sigma_nfw_at_R(r_dense, mass, c, OMEGA_M)
                ref = ctd.DeltaSigma_at_R(R_GRID, r_dense, sig, mass, c,
                                          OMEGA_M)
                dev = np.max(np.abs(mine / ref - 1.0))
                self.assertLess(dev, REL_TOL, f"M={mass:.1e} c={c}: {dev:.2e}")

    def test_r200_is_the_200x_mean_boundary(self):
        # The single most convention-sensitive line in the package
        # (CLAUDE.md, "UNIFIED rho_m convention"): r_200 = [3M/(800 pi
        # rho)]^(1/3) with ONE reference density for boundary and
        # amplitude.
        for mass in (1.0e13, 1.0e15):
            expected = (3.0 * mass / (800.0 * np.pi * RHOM0)) ** (1.0 / 3.0)
            self.assertAlmostEqual(
                nfw.convert_m200_to_r200(mass, RHOM0) / expected, 1.0,
                places=12)

    def test_profiles_are_finite_positive_and_falling(self):
        sig = nfw.sigmaNFW_Analytical(R_GRID, 2.0e14, 4.0, rho_c=RHOM0)
        dsig = nfw.deltaSigmaNFW_Analytical(R_GRID, 2.0e14, 4.0, rho_c=RHOM0)
        for arr in (sig, dsig):
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertTrue(np.all(arr > 0.0))
            self.assertTrue(np.all(np.diff(arr) < 0.0))

    def test_amplitude_is_linear_in_the_reference_density_at_fixed_x(self):
        # Sigma = 2 r_s delta_c rho f(R/r_s): doubling rho halves r_s^3 at
        # fixed M, so the profile is NOT simply proportional to rho -- but
        # evaluated at the SAME x = R/r_s it is exactly linear. This
        # separates the boundary role of rho from its amplitude role, the
        # exact thing the 2026-08-24 unification collapsed onto one value.
        mass, c = 2.0e14, 4.0
        for rho in (RHOM0, 2.0 * RHOM0):
            r_s = nfw.convert_m200_to_r200(mass, rho) / c
            x = np.array([0.3, 1.0, 3.0])
            sig = nfw.sigmaNFW_Analytical(x * r_s, mass, c, rho_c=rho)
            scaled = sig / (r_s * rho)
            if rho == RHOM0:
                reference = scaled
            else:
                self.assertTrue(np.allclose(scaled, reference, rtol=1e-10))


class TestConcentrationRelations(unittest.TestCase):
    """Child+18 and Duffy+08 against the published coefficients."""

    def test_child18_reproduces_paper_equation_19(self):
        # c = c0 + A [ (M/(b M*))^m (1 + M/(b M*))^{-m} - 1 ]
        for sample, (m, a_coeff, b_coeff, c0) in CHILD18_TABLE2.items():
            for z in (0.0, 0.35, 0.7):
                m_star = conc.peakHeight_nonLinearMass(z)
                ratio = M_GRID / (b_coeff * m_star)
                expected = c0 + a_coeff * (
                    ratio ** m * (1.0 + ratio) ** (-m) - 1.0)
                got = conc.child18_mass_concentration(M_GRID, z,
                                                      halo_sample=sample)
                dev = np.max(np.abs(np.asarray(got) / expected - 1.0))
                self.assertLess(dev, 1e-12, f"{sample} z={z}: {dev:.2e}")

    def test_child18_default_sample_is_stacked_nfw(self):
        # CLAUDE.md's ratified convention: the 1-halo term uses Child18
        # `stacked_nfw`. A silent default change here would move every
        # published concentration.
        got = conc.child18_mass_concentration(M_GRID, 0.35)
        want = conc.child18_mass_concentration(M_GRID, 0.35,
                                               halo_sample="stacked_nfw")
        self.assertTrue(np.allclose(got, want, rtol=0.0, atol=0.0))

    def test_child18_coefficients_match_colossus(self):
        # colossus is a separate package's transcription of the SAME
        # published table.  A direct numeric comparison of the two
        # FUNCTIONS is meaningless -- colossus substitutes a generic
        # cosmology-dependent nonlinear mass (peaks.nonLinearMass) for M*
        # whereas this package uses Child et al.'s own fixed M*(z) fit, so
        # the two are different quantities by design (see
        # halo_model.test.py's note).  What IS comparable is the
        # coefficient set: feed our re-typed (m, A, b, c0) colossus's OWN
        # M*, and it must reproduce colossus's output exactly.
        from colossus.cosmology import cosmology as colossus_cosmology
        from colossus.halo import concentration as colossus_conc
        colossus_cosmology.setCosmology("WMAP7")
        from colossus.lss import peaks

        for z in (0.0, 0.5):
            m_star = peaks.nonLinearMass(z)
            for sample, (m, a_coeff, b_coeff, c0) in CHILD18_TABLE2.items():
                ratio = M_GRID / (m_star * b_coeff)
                ours = c0 + a_coeff * (
                    ratio ** m * (1.0 + ratio) ** (-m) - 1.0)
                theirs, _mask = colossus_conc.modelChild18(
                    M_GRID, z, halo_sample=sample)
                dev = np.max(np.abs(ours / theirs - 1.0))
                self.assertLess(dev, 1e-12,
                                f"{sample} z={z}: coefficients differ from "
                                f"colossus by {dev:.2e}")

    def test_duffy_reproduces_paper_table_1(self):
        for z in (0.0, 0.4, 0.8):
            expected = DUFFY_A * (M_GRID / 2.0e12) ** DUFFY_B \
                * (1.0 + z) ** DUFFY_C
            got = conc.duffy_concentration_relation(M_GRID, z)
            dev = np.max(np.abs(np.asarray(got) / expected - 1.0))
            self.assertLess(dev, 1e-12, f"z={z}: {dev:.2e}")

    def test_duffy_default_redshift_is_the_y3_pivot(self):
        self.assertTrue(np.allclose(
            conc.duffy_concentration_relation(M_GRID),
            conc.duffy_concentration_relation(M_GRID, 0.4)))

    def test_nonlinear_mass_follows_child18_fit(self):
        # log10 M*(z) = 12.5 - 1.5 z (Child+18's own colossus/WMAP7 fit).
        for z in (0.0, 0.2, 0.5, 1.0):
            self.assertAlmostEqual(
                np.log10(conc.peakHeight_nonLinearMass(z)),
                12.5 - 1.5 * z, places=10)

    def test_concentration_decreases_with_mass_and_redshift(self):
        for z in (0.0, 0.4):
            c = np.asarray(conc.child18_mass_concentration(M_GRID, z))
            self.assertTrue(np.all(np.diff(c) < 0.0))
            self.assertTrue(np.all(c > 0.0))
        hi_z = np.asarray(conc.child18_mass_concentration(M_GRID, 0.8))
        lo_z = np.asarray(conc.child18_mass_concentration(M_GRID, 0.0))
        self.assertTrue(np.all(hi_z < lo_z))


class TestHaloBias(unittest.TestCase):
    """Tinker et al. 2010 Eq. 6 vs cluster_toolkit."""

    def setUp(self):
        self.model = hm.biasModel(K_GRID, P_GRID, omega_m=OMEGA_M)

    def test_peak_height_matches_cluster_toolkit(self):
        mine = np.asarray(self.model.nu_at_M(M_GRID))
        ref = ctb.nu_at_M(M_GRID, K_GRID, P_GRID, OMEGA_M)
        self.assertLess(np.max(np.abs(mine / ref - 1.0)), REL_TOL)

    def test_bias_at_nu_matches_cluster_toolkit(self):
        nu = np.asarray(self.model.nu_at_M(M_GRID))
        mine = np.asarray(self.model.bias_at_nu(nu))
        ref = ctb.bias_at_nu(nu, 200)
        self.assertLess(np.max(np.abs(mine / ref - 1.0)), REL_TOL)

    def test_tinker_parameters_match_the_published_delta_200_form(self):
        # Tinker+2010 Table 2's y = log10(Delta) parameterisation, re-typed.
        y = np.log10(200.0)
        expected = [
            1.0 + 0.24 * y * np.exp(-((4.0 / y) ** 4)),
            0.44 * y - 0.88,
            0.183,
            1.5,
            0.019 + 0.107 * y + 0.19 * np.exp(-((4.0 / y) ** 4)),
            2.4,
        ]
        got = self.model.get_tinker_pars()
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=12)

    def test_bias_grows_with_mass(self):
        nu = np.asarray(self.model.nu_at_M(M_GRID))
        b = np.asarray(self.model.bias_at_nu(nu))
        self.assertTrue(np.all(np.diff(b) > 0.0))
        self.assertTrue(np.all(b > 0.0))


class TestBiasAtMKnownDefect(unittest.TestCase):
    """Characterization pin: ``bias_at_M`` is dead code in BOTH copies.

    It calls ``self.compute_nu(M)`` (no such method) and then
    ``self.bias_at_nu(nu, odelta=...)`` against a one-argument signature.
    Production reaches halo bias through ``nu_at_M`` + ``bias_at_nu``, so
    nothing is currently miscomputed -- but the method is a trap for the
    next caller. Asserting the current broken behaviour makes this test go
    RED the moment someone repairs it, which is the signal to delete this
    class.
    """

    def test_bias_at_M_raises_in_the_pipelines_copy(self):
        model = hm.biasModel(K_GRID, P_GRID, omega_m=OMEGA_M)
        with self.assertRaises(AttributeError):
            model.bias_at_M(M_GRID)

    def test_bias_at_M_raises_identically_in_the_y3_buzzard_copy(self):
        import haloModel as buzzard
        model = buzzard.biasModel(K_GRID, P_GRID, omega_m=OMEGA_M)
        with self.assertRaises(AttributeError):
            model.bias_at_M(M_GRID)


class TestFirstHaloTerm(unittest.TestCase):
    """``lensingModel.first_halo_term`` and its rho_m convention."""

    def test_matches_the_analytic_profile_at_the_models_own_concentration(self):
        model = hm.lensingModel(R_GRID, omega_m=OMEGA_M)
        z = 0.35
        model.first_halo_term(M_GRID, z=z)
        c = np.asarray(model.c)
        rho = RHOM0 * (1.0 + z) ** 3
        mm, rr = np.meshgrid(M_GRID, R_GRID, indexing="ij")
        expected = nfw.sigmaNFW_Analytical(rr, mm, c[:, None],
                                           rho_c=rho) / 1.0e12
        self.assertTrue(np.allclose(model.Sigma["1h"], expected, rtol=1e-12))

    def test_uses_the_comoving_rho_m0_with_the_documented_1pz_cubed(self):
        # CLAUDE.md's ratified convention: the amplitude reference is
        # rho_m0 = Omega_m rho_crit,0, and the only z dependence in the
        # 1-halo normalisation is the explicit (1+z)^3 factor (plus
        # concentration). Checked by re-deriving the z = 0.5 call from the
        # z = 0 one at a FIXED concentration.
        z = 0.5
        model_z = hm.lensingModel(R_GRID, omega_m=OMEGA_M)
        model_z.c = 4.0
        model_z.first_halo_term(M_GRID, z=z)
        mm, rr = np.meshgrid(M_GRID, R_GRID, indexing="ij")
        expected = nfw.sigmaNFW_Analytical(
            rr, mm, 4.0, rho_c=RHOM0 * (1.0 + z) ** 3) / 1.0e12
        self.assertTrue(np.allclose(model_z.Sigma["1h"], expected, rtol=1e-12))
        self.assertAlmostEqual(model_z.rhom0 / RHOM0, 1.0, places=12)
        self.assertAlmostEqual(model_z.rhoc0 / RHOC0, 1.0, places=12)

    def test_units_are_msun_h_per_pc2_and_the_profile_falls(self):
        model = hm.lensingModel(R_GRID, omega_m=OMEGA_M)
        model.first_halo_term(np.array([2.0e14]), z=0.35)
        sig = np.asarray(model.Sigma["1h"])[0]
        dsig = np.asarray(model.dSigma["1h"])[0]
        self.assertTrue(np.all(np.isfinite(sig)))
        self.assertTrue(np.all(sig > 0.0))
        self.assertTrue(np.all(np.diff(sig) < 0.0))
        self.assertTrue(np.all(np.diff(dsig) < 0.0))
        # Msun h/pc^2 at ~1 Mpc/h for a 2e14 cluster is O(1-100), not O(1e12):
        # a dropped /1e12 would blow this by twelve orders of magnitude.
        self.assertTrue(np.all(sig < 1.0e4))
        self.assertTrue(np.all(sig > 1.0e-3))


class TestPipelinesVsBuzzardEquivalence(unittest.TestCase):
    """The two homes must agree numerically until one of them is retired.

    Measured 2026-08-26: exact (0.0 relative deviation) on every shared
    entry point, despite 455/258/184/474/114 diff lines across
    halo_model / concentration / nfw_model / bsel / sigma_crit_inv. This
    test turns "the copies drifted textually" into "the copies still agree
    numerically", and fails loudly the day that stops being true.
    """

    def test_concentration_relations_agree(self):
        import haloModel as buzzard
        for z in (0.0, 0.35, 0.8):
            for sample in CHILD18_TABLE2:
                a = np.asarray(conc.child18_mass_concentration(
                    M_GRID, z, halo_sample=sample))
                b = np.asarray(buzzard.child18_mass_concentration(
                    M_GRID, z, halo_sample=sample))
                self.assertTrue(np.allclose(a, b, rtol=1e-14),
                                f"child18 {sample} z={z}")
            self.assertTrue(np.allclose(
                conc.duffy_concentration_relation(M_GRID, z),
                buzzard.duffy_concentration_relation(M_GRID, z), rtol=1e-14))
            self.assertAlmostEqual(
                conc.peakHeight_nonLinearMass(z)
                / buzzard.peakHeight_nonLinearMass(z), 1.0, places=14)

    def test_halo_bias_agrees(self):
        import haloModel as buzzard
        a = hm.biasModel(K_GRID, P_GRID, omega_m=OMEGA_M)
        b = buzzard.biasModel(K_GRID, P_GRID, omega_m=OMEGA_M)
        nu_a = np.asarray(a.nu_at_M(M_GRID))
        nu_b = np.asarray(b.nu_at_M(M_GRID))
        self.assertTrue(np.allclose(nu_a, nu_b, rtol=1e-14))
        self.assertTrue(np.allclose(np.asarray(a.bias_at_nu(nu_a)),
                                    np.asarray(b.bias_at_nu(nu_b)),
                                    rtol=1e-14))
        self.assertTrue(np.allclose(a.get_tinker_pars(), b.get_tinker_pars(),
                                    rtol=1e-14))

    def test_analytic_nfw_agrees(self):
        import nfwModel as buzzard
        for c in (3.0, 4.0, 5.0):
            for fn in ("sigmaNFW_Analytical", "deltaSigmaNFW_Analytical"):
                a = getattr(nfw, fn)(R_GRID[:, None], M_GRID[None, :], c,
                                     rho_c=RHOM0)
                b = getattr(buzzard, fn)(R_GRID[:, None], M_GRID[None, :], c,
                                         rho_c=RHOM0)
                self.assertTrue(np.allclose(a, b, rtol=1e-14), f"{fn} c={c}")

    def test_first_halo_term_agrees(self):
        import haloModel as buzzard
        z = 0.35
        a = hm.lensingModel(R_GRID, omega_m=OMEGA_M)
        b = buzzard.lensingModel(R_GRID, omega_m=OMEGA_M)
        a.first_halo_term(M_GRID, z=z)
        b.first_halo_term(M_GRID, z=z)
        for key in ("1h",):
            self.assertTrue(np.allclose(a.Sigma[key], b.Sigma[key],
                                        rtol=1e-14))
            self.assertTrue(np.allclose(a.dSigma[key], b.dSigma[key],
                                        rtol=1e-14))

    def test_hydro_mc_is_byte_identical(self):
        a = (REPO / "src" / "pipelines" / "cosmology" / "hydro_mc.py")
        b = (REPO / "y3_buzzard" / "hydro_mc.py")
        self.assertEqual(hashlib.sha256(a.read_bytes()).hexdigest(),
                         hashlib.sha256(b.read_bytes()).hexdigest())


class TestPrjParams(unittest.TestCase):
    """The frozen Costanzi-2026 EMG projection-kernel coefficients.

    Previously reached only through imports; the grid values and the
    z-interpolation clamp were never pinned directly.
    """

    def setUp(self):
        self.params = PrjParams.default()

    def test_coefficient_grid_shape_and_redshift_axis(self):
        arrays = self.params.as_dict()
        self.assertEqual(set(arrays) - {"z"}, set(COEFF_NAMES))
        z = np.asarray(arrays["z"])
        self.assertTrue(np.all(np.diff(z) > 0.0), "z axis must be increasing")
        for name in COEFF_NAMES:
            values = np.asarray(arrays[name])
            self.assertEqual(values.shape, z.shape,
                             f"{name} must be one value per z node")
            self.assertTrue(np.all(np.isfinite(values)))

    def test_interpolation_is_linear_between_nodes_and_clamped_outside(self):
        z = np.asarray(self.params.as_dict()["z"])
        for name in COEFF_NAMES:
            values = np.asarray(self.params.as_dict()[name])
            # Exact at every node.
            got = self.params.interp_linear(name, z)
            self.assertTrue(np.allclose(got, values, rtol=1e-12), name)
            # Linear at a midpoint.
            mid = 0.5 * (z[0] + z[1])
            self.assertAlmostEqual(
                float(self.params.interp_linear(name, mid)),
                0.5 * (values[0] + values[1]), places=10, msg=name)
            # Clamped, not extrapolated, outside the tabulated range.
            self.assertAlmostEqual(
                float(self.params.interp_linear(name, z[0] - 5.0)),
                float(values[0]), places=12, msg=name)
            self.assertAlmostEqual(
                float(self.params.interp_linear(name, z[-1] + 5.0)),
                float(values[-1]), places=12, msg=name)

    def test_cdf_is_a_proper_distribution_in_observed_richness(self):
        # The EMG kernel's CDF must be monotone in lob and span [0, 1]:
        # the selection kernels take differences of it across bin edges, so
        # a non-monotone CDF would produce negative bin probabilities.
        z, ltr = 0.4, 30.0
        lob = np.linspace(0.5, 400.0, 400)
        cdf = np.asarray(self.params.cdf_lob(lob, ltr, z))
        self.assertTrue(np.all(np.diff(cdf) >= -1e-12))
        self.assertGreaterEqual(cdf[0], -1e-12)
        self.assertLessEqual(cdf[-1], 1.0 + 1e-9)
        self.assertGreater(cdf[-1] - cdf[0], 0.5,
                           "kernel puts almost no mass on the sampled range")

    def test_splines_expose_every_coefficient(self):
        splines = self.params.splines()
        for name in COEFF_NAMES:
            self.assertIn(name, splines, f"{name} missing from splines()")


class TestSigmaCritInvModule(unittest.TestCase):
    """``cosmology/sigma_crit_inv.py`` is pure CosmoSIS-module code.

    It has no importable pure functions (everything lives in
    setup/execute), so what can be pinned without a live pipeline is the
    module contract: the three CosmoSIS entry points exist, and the beta
    lookup table it hardcodes is actually present in this tree. A missing
    LUT is the failure mode that otherwise only shows up mid-MCMC.
    """

    def test_exports_the_cosmosis_entry_points(self):
        # cosmosis is not importable in the test env; stub the one symbol
        # the module imports at module scope.
        stub = types.ModuleType("cosmosis")
        datablock = types.ModuleType("cosmosis.datablock")
        datablock.option_section = "module_options"
        datablock.names = types.SimpleNamespace(
            cosmological_parameters="cosmological_parameters",
            distances="distances")
        stub.datablock = datablock
        saved = {k: sys.modules.get(k) for k in ("cosmosis",
                                                 "cosmosis.datablock")}
        sys.modules["cosmosis"] = stub
        sys.modules["cosmosis.datablock"] = datablock
        try:
            import importlib
            module = importlib.import_module("cosmology.sigma_crit_inv")
            for name in ("setup", "execute", "cleanup"):
                self.assertTrue(callable(getattr(module, name, None)),
                                f"sigma_crit_inv.{name} is not callable")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_beta_lookup_table_is_present(self):
        lut = REPO / "data" / "beta_sigCrit_lut_mock_buzzard_zl.npz"
        self.assertTrue(lut.is_file(), f"missing beta LUT at {lut}")
        with np.load(lut) as table:
            for key in ("zlens", "Radii", "zEff", "betaEff"):
                self.assertIn(key, table.files, f"{key} absent from the LUT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
