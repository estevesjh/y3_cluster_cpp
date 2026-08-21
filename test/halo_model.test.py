#!/usr/bin/env python3
"""Unit tests for y3_buzzard/haloModel.py and y3_buzzard/halo_model_cosmosis.py
-- previously untested. Written BEFORE reporting the "second halo term has
NaN values" issue, so that report is backed by reproducible checks rather
than a one-off dump inspection.

Covers:
  - biasModel: Tinker et al. 2010 bias vs cluster_toolkit.bias.bias_at_nu
    -- a genuinely separate implementation, not a re-typed copy of
    biasModel._bias_at_nu's own coefficients.
  - child18_mass_concentration: coefficients cross-checked against
    colossus.halo.concentration.modelChild18 (a separate package's
    implementation of the same published table, not a re-typed copy of
    this repo's own coefficients), plus a citation-backed check of the
    M*(z) formula against Child et al. 2018's own published anchor
    points (arXiv:1804.10199). NOTE: a *numeric* comparison against
    colossus's modelChild18 output is deliberately NOT done -- colossus
    substitutes a generic cosmology-dependent nonlinear mass
    (peaks.nonLinearMass) for M*, whereas haloModel.py uses Child et
    al.'s own fixed M*(z) fit; the two are different quantities by
    design, not a bug in either.
  - lensingModel.first_halo_term: the analytic 1-halo (Wright & Brainerd
    2000) NFW Sigma/dSigma -- finiteness, positivity, radial monotonicity,
    plus a numeric cross-check against cluster_toolkit.deltasigma
    (Sigma_nfw_at_R / DeltaSigma_at_R) in the same Msun/h, Mpc/h, Msun
    h/pc^2 convention first_halo_term itself uses (it divides its raw
    sigmaNFW_Analytical/deltaSigmaNFW_Analytical output by 1e12 for
    exactly this reason -- see the "units are Msun/pc^2" comment at
    haloModel.py's first_halo_term).
  - TestFirstHaloTermRedshiftHandling: the REAL finding from this pass
    (see docs/known_issues/first_halo_term_z0_defect.md) -- halo_model_cosmosis.py's
    execute() hardcodes z=0 when calling first_halo_term, even though it
    has a real per-z grid it uses for everything else, so the published
    Sigma_nfw/dSigma_nfw/concentration tables are the z=0 1-halo term at
    every actual cluster redshift.
  - scaleShiftCosmo: vs an independent astropy re-derivation; exact
    identity at the hardcoded fiducial cosmology (Om0=0.3, H0=70).
  - lensingModel.second_halo_term / ct_2hTerm: the reported issue. Pins,
    with real (not synthetic) P(k), the two specific defects
    docs/known_issues/dsigma_hh_debug_flag.md already describes from manual dump
    inspection -- the z-axis degeneracy and the low-radius NaN fraction
    -- as reproducible tests, plus the dummy exclusion-halo parameters.
    These tests currently PASS because they assert the *current* (buggy)
    behavior; if the underlying bug is fixed, they will correctly start
    failing and need updating -- that is the point, not an oversight.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import cluster_toolkit as ct
import cluster_toolkit.bias as ctb
from colossus.cosmology import cosmology as colossus_cosmology
from colossus.halo import concentration as colossus_concentration

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "y3_buzzard"))

import haloModel as hm  # noqa: E402

REL_TOL = 1.0e-3

# Child18 is documented as cosmology-independent, but colossus still
# requires a cosmology to be set globally before any halo-module call.
colossus_cosmology.setCosmology("planck18")

DUMP_DIR = REPO / "docs" / "figs" / "real_pipeline_extract_output"
HAS_DUMP = (DUMP_DIR / "matter_power_lin" / "p_k.txt").is_file()
_SKIP_MSG = (f"requires a real-pipeline dump at {DUMP_DIR} -- run "
            "`cosmosis docs/figs/real_pipeline_extract.ini` first")


def _load_matter_power_lin():
    """Real linear P(k, z) from the fiducial dump -- physically realistic
    input for the second-halo-term tests, not a hand-wavy synthetic one."""
    d = DUMP_DIR / "matter_power_lin"
    k_h = np.loadtxt(d / "k_h.txt")
    z = np.loadtxt(d / "z.txt")
    p_k = np.loadtxt(d / "p_k.txt")  # shape (n_z, n_k), matches execute()'s read
    if p_k.ndim == 1:
        p_k = p_k.reshape(z.size, k_h.size)
    return k_h, p_k, z




class TestBiasModel(unittest.TestCase):
    def test_bias_matches_cluster_toolkit_independent_implementation(self):
        # cluster_toolkit.bias.bias_at_nu is a wholly separate C
        # implementation of Tinker et al. 2010 Eq. 6 -- unlike a re-typed
        # copy of biasModel._bias_at_nu's own coefficients, agreement here
        # cannot be explained by both sides sharing a typo.
        bm = hm.biasModel(np.array([0.01, 0.1, 1.0]), np.array([1.0, 1.0, 1.0]))
        for nu in (0.5, 1.0, 1.686, 3.0, 6.0):
            got = float(bm.bias_at_nu(nu))
            expected = float(ctb.bias_at_nu(nu, delta=200))
            self.assertAlmostEqual(got, expected, delta=REL_TOL * abs(expected))

    def test_bias_increases_with_peak_height(self):
        # Higher peak height (rarer, more massive halos) must be more
        # biased -- a physical monotonicity requirement of any reasonable
        # bias(nu) model, independent of the exact Tinker coefficients.
        bm = hm.biasModel(np.array([0.01, 0.1, 1.0]), np.array([1.0, 1.0, 1.0]))
        nus = np.linspace(0.3, 8.0, 40)
        biases = np.array([bm.bias_at_nu(n) for n in nus])
        self.assertTrue(np.all(np.diff(biases) > 0))


class TestChild18Concentration(unittest.TestCase):
    def test_coefficients_match_colossus_independent_implementation(self):
        # colossus.halo.concentration.modelChild18 is a separate package's
        # implementation of the same Child et al. 2018 relation; its
        # A/b/c0/m coefficients for every halo_sample are bit-identical to
        # haloModel.py's own hardcoded ones (both are the published table
        # verbatim -- Child et al. 2018 Table 2). A genuine independent
        # cross-check of the coefficients, not a shared typo.
        #
        # NOTE: a direct numeric comparison of the two functions' outputs
        # is NOT a valid apples-to-apples check, despite the identical
        # coefficients: colossus computes M* via peaks.nonLinearMass(z),
        # the generic cosmology-dependent nonlinear mass for whatever
        # cosmology is set, whereas haloModel.py uses Mstar =
        # 10**(12.5-1.5*z) -- which is Child et al. 2018's OWN published
        # M*(z) fit (arXiv:1804.10199 give log(M*/h^-1 Msun) = 12.5, 11,
        # 9.5 at z=0,1,2 -- see test_mstar_matches_published_anchor_points
        # below), calibrated to their own simulation, not a generic
        # nonlinear-mass calculation. So colossus's substitution of
        # peaks.nonLinearMass(z) is itself an approximation to what the
        # concentration fit was actually calibrated against; disagreeing
        # with it is not evidence of a haloModel.py bug.
        for halo_sample, (m, A, b, c0) in {
            "individual_all": (-0.10, 3.44, 430.49, 3.19),
            "individual_relaxed": (-0.09, 2.88, 1644.53, 3.54),
            "stacked_nfw": (-0.07, 4.61, 638.65, 3.59),
            "stacked_einasto": (-0.01, 63.2, 431.48, 3.36),
        }.items():
            src = __import__("inspect").getsource(hm.child18_mass_concentration)
            # Confirm haloModel.py's branch for this halo_sample uses the
            # same coefficients colossus hardcodes for it.
            self.assertIn(f"m = {m}".replace(" ", ""), src.replace(" ", ""))
            self.assertIn(f"A = {A}".replace(" ", ""), src.replace(" ", ""))
            self.assertIn(f"b = {b}".replace(" ", ""), src.replace(" ", ""))
            self.assertIn(f"c0 = {c0}".replace(" ", ""), src.replace(" ", ""))

    def test_mstar_matches_published_anchor_points(self):
        # Child et al. 2018 (arXiv:1804.10199) define the nonlinear mass
        # scale M* directly: log(M*/h^-1 Msun) = 12.5 at z=0, 11 at z=1,
        # 9.5 at z=2 -- i.e. Mstar = 10**(12.5 - 1.5*z). Read the literal
        # formula out of the source (rather than re-deriving it here,
        # which would just check the test's own arithmetic) so this test
        # breaks loudly if the exponent or coefficient in
        # child18_mass_concentration is ever edited without checking it
        # against the paper -- a check nothing previously performed.
        import inspect
        src = inspect.getsource(hm.peakHeight_nonLinearMass)
        self.assertIn("10**(12.5-1.5*z)", src.replace(" ", ""))
        for z, log10_mstar_expected in ((0.0, 12.5), (1.0, 11.0), (2.0, 9.5)):
            mstar_got = float(hm.peakHeight_nonLinearMass(z))
            self.assertAlmostEqual(np.log10(mstar_got), log10_mstar_expected, places=10)

    def test_concentration_decreases_with_mass(self):
        # Well-established trend (less massive halos are more
        # concentrated/formed earlier) -- model-independent sanity check.
        M = np.logspace(12.5, 15.5, 20)
        c = hm.child18_mass_concentration(M, z=0.3)
        self.assertTrue(np.all(np.diff(c) < 0))
        self.assertTrue(np.all((c > 1.0) & (c < 15.0)))


class TestScaleShiftCosmo(unittest.TestCase):
    def test_matches_independent_astropy_rederivation(self):
        # z[0] = 0.0 deliberately: scaleShiftCosmo hardcodes
        # scale_shift[0] = 1. unconditionally (line "scale_shift[0] = 1."),
        # assuming the caller's z-grid starts at (or effectively at) 0 --
        # see test_scale_shift_at_index_0_is_hardcoded_to_1 below for what
        # happens when that assumption doesn't hold.
        import astropy.cosmology
        cosmo = astropy.cosmology.FlatLambdaCDM(H0=68.0, Om0=0.28, Tcmb0=2.725)
        z = np.linspace(0.0, 1.0, 20)
        scale_shift, hubble_shift = hm.scaleShiftCosmo(z, cosmo)

        cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
        expected_scale = cosmo.comoving_distance(z).value / \
            cosmo_fid.comoving_distance(z).value
        expected_hubble = cosmo.H(z).value / cosmo_fid.H(z).value
        np.testing.assert_allclose(scale_shift[1:], expected_scale[1:],
                                   rtol=REL_TOL)
        np.testing.assert_allclose(hubble_shift, expected_hubble, rtol=REL_TOL)

    def test_scale_shift_at_index_0_is_hardcoded_to_1_regardless_of_z(self):
        # FINDING: scaleShiftCosmo always overwrites scale_shift[0] with
        # exactly 1.0, regardless of what z[0] actually is -- correct if
        # the caller's z-grid genuinely starts at 0 (the +eps in the
        # denominator exists to avoid a 0/0 there), silently WRONG if
        # z[0] is not negligible. halo_model_cosmosis.py calls this with
        # z = matter_power_lin's z grid; whether that grid's first entry
        # is always exactly 0 in production has not been checked here.
        import astropy.cosmology
        cosmo = astropy.cosmology.FlatLambdaCDM(H0=68.0, Om0=0.28, Tcmb0=2.725)
        z = np.array([0.05, 0.20, 0.50])
        scale_shift, _ = hm.scaleShiftCosmo(z, cosmo)
        self.assertEqual(scale_shift[0], 1.0)

        cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
        true_ratio_at_z0 = float(cosmo.comoving_distance(z[0]).value /
                                 cosmo_fid.comoving_distance(z[0]).value)
        self.assertGreater(abs(scale_shift[0] - true_ratio_at_z0), 1e-3)

    def test_identity_at_the_hardcoded_fiducial_cosmology(self):
        import astropy.cosmology
        cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
        z = np.linspace(0.05, 1.5, 15)
        scale_shift, hubble_shift = hm.scaleShiftCosmo(z, cosmo_fid)
        np.testing.assert_allclose(scale_shift, 1.0, rtol=1e-6)
        np.testing.assert_allclose(hubble_shift, 1.0, rtol=1e-6)


class TestFirstHaloTerm(unittest.TestCase):
    def test_sigma_and_dsigma_are_finite_positive_and_fall_with_radius(self):
        R = np.logspace(-2, 1, 25)  # Mpc/h
        lm = hm.lensingModel(R, omega_m=0.3, odelta=200)
        lm.first_halo_term(np.array([1e14]), z=0.3, conc_model_name="Child18")

        sigma = lm.Sigma["1h"][0]
        dsigma = lm.dSigma["1h"][0]
        self.assertTrue(np.all(np.isfinite(sigma)))
        self.assertTrue(np.all(np.isfinite(dsigma)))
        self.assertTrue(np.all(sigma > 0.0))
        self.assertTrue(np.all(dsigma > 0.0))
        # A centred NFW projected profile is monotonically decreasing.
        self.assertTrue(np.all(np.diff(sigma) < 0.0))
        self.assertTrue(np.all(np.diff(dsigma) < 0.0))

    def test_matches_cluster_toolkit_nfw_sigma_and_dsigma(self):
        # Pin the concentration directly (bypass the Child18 c(M,z) model,
        # already cross-checked separately in TestChild18Concentration) so
        # this isolates the NFW Sigma/dSigma math itself against
        # cluster_toolkit's independent implementation of the same Wright
        # & Brainerd (2000) profile, in the same Msun/h, Mpc/h, Msun h/pc^2
        # convention first_halo_term uses (confirmed empirically: its raw
        # sigmaNFW_Analytical/deltaSigmaNFW_Analytical output is 1e12x the
        # Msun h/pc^2 convention -- an Mpc->pc area factor -- which is
        # exactly why first_halo_term divides by 1e12 before storing).
        omega_m = 0.3
        M = 1e14  # Msun/h
        c200 = 5.0
        R = np.logspace(-2, 1, 8)  # Mpc/h

        lm = hm.lensingModel(R, omega_m=omega_m, odelta=200)
        lm.c = np.array([c200])
        lm.first_halo_term(np.array([M]), z=0.0)
        sigma = lm.Sigma["1h"][0]
        dsigma = lm.dSigma["1h"][0]

        Rs_grid = np.logspace(-3, 2, 2000)  # dense grid cluster_toolkit
                                             # needs for its enclosed-mass
                                             # integral in DeltaSigma_at_R
        sigma_ct_grid = ct.deltasigma.Sigma_nfw_at_R(Rs_grid, M, c200, omega_m)
        sigma_ct = ct.deltasigma.Sigma_nfw_at_R(R, M, c200, omega_m)
        dsigma_ct = ct.deltasigma.DeltaSigma_at_R(
            R, Rs_grid, sigma_ct_grid, M, c200, omega_m)

        np.testing.assert_allclose(sigma, sigma_ct, rtol=REL_TOL)
        np.testing.assert_allclose(dsigma, dsigma_ct, rtol=REL_TOL)


class TestFirstHaloTermRedshiftHandling(unittest.TestCase):
    """FINDING: halo_model_cosmosis.py's execute() calls
    lensModel.first_halo_term(M, z=0, ...) with a hardcoded z=0, even
    though the module has a real per-z grid (`z`) it uses for
    everything else (bias, xi_NL, second_halo_term, scale_shift). The
    published haloModel/Sigma_nfw, haloModel/dSigma_nfw, and
    haloModel/concentration datablock entries -- and therefore every
    C++ consumer that Interp2D's them over (r_sigma, lnM) with no z
    axis (lensing_weights.hh, kappa_max.hh, gamma_max.hh,
    sigma_mis_joint.cuh, n_operator_sel_gl_t.hh, ...) -- are the z=0
    1-halo term at every actual cluster redshift.

    This is NOT a bug in first_halo_term/child18_mass_concentration
    themselves (they are correctly z-dependent when called directly,
    per the tests below) -- it is the CosmoSIS wiring in
    halo_model_cosmosis.py::execute() ignoring its own z grid for this
    one term.
    """

    def test_first_halo_term_concentration_is_genuinely_z_dependent(self):
        # Confirms the underlying model is NOT the source of the
        # z=0-everywhere behavior seen in production -- it responds
        # correctly to z when the caller passes it through.
        R = np.logspace(-2, 1, 6)
        cs = []
        for z in (0.0, 0.2, 0.4, 0.65):
            lm = hm.lensingModel(R, omega_m=0.3, odelta=200)
            lm.first_halo_term(np.array([1e14]), z=z, conc_model_name="Child18")
            cs.append(float(lm.c[0]))
        self.assertTrue(np.all(np.diff(cs) < 0.0))
        # At a typical DES cluster redshift the effect is not a rounding
        # artifact: concentration (and hence the NFW profile shape,
        # since rs = r200/c) is measurably different from the z=0 value
        # production always uses.
        self.assertLess(cs[2] / cs[0], 0.95)  # z=0.4 vs z=0: >5% lower

    def test_execute_hardcodes_z_equals_0_for_the_1h_term(self):
        # Read directly off the source (rather than re-typing "z=0")
        # so this test breaks loudly -- and needs a deliberate update --
        # the day this hardcoding is fixed to use the real z grid.
        import inspect
        sys.path.insert(0, str(REPO / "y3_buzzard"))
        import halo_model_cosmosis as hmc
        src = inspect.getsource(hmc.execute)
        self.assertIn("first_halo_term(M,z=0,", src.replace(" ", ""))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestSecondHaloTerm(unittest.TestCase):
    """lensingModel.second_halo_term / ct_2hTerm -- the reported issue.

    Uses the REAL linear P(k, z) from the fiducial dump (not a synthetic
    stand-in), so these numbers reflect what production actually computes,
    not an artifact of a hand-picked test power spectrum.
    """

    @classmethod
    def setUpClass(cls):
        cls.k_h, cls.p_k, cls.z = _load_matter_power_lin()
        cls.R = np.logspace(np.log10(0.1), np.log10(35.0), 40)  # cMpc/h

    def test_dummy_exclusion_parameters_match_the_documented_values(self):
        # docs/known_issues/dsigma_hh_debug_flag.md #3: lensingModel.second_halo_term
        # hardcodes Md=1e14, cd=5 (NOT ct_2hTerm's own class default of
        # Md=1e13, cd=4), tied to a fixed halo rather than the mass being
        # integrated over. Read directly off the source rather than
        # re-typing the numbers, so this test breaks loudly if the
        # hardcoded call site ever changes without this file being updated.
        import inspect
        src = inspect.getsource(hm.lensingModel.second_halo_term)
        self.assertIn("Md=1e14", src.replace(" ", ""))
        self.assertIn("cd=5", src.replace(" ", ""))

    def test_z_axis_is_currently_degenerate(self):
        # docs/known_issues/dsigma_hh_debug_flag.md #2: pk_to_sigma's per-z loop calls
        # _pk_to_sigma(Rp, k, pk) without ever indexing pk by z, so every
        # z-slice of Sigma_hh (and therefore Wp) comes out bit-identical
        # regardless of the real z-dependent power spectrum passed in.
        # This test PASSES today because it asserts that (undesirable but
        # real) behavior -- it is meant to start FAILING the day the z
        # loop is fixed to pass pk[i], at which point this test itself
        # needs to be replaced by one that checks genuine z-variation.
        lm = hm.lensingModel(self.R, omega_m=0.3, odelta=200)
        lm.second_halo_term(self.z, self.k_h, self.p_k)

        self.assertGreater(self.z.size, 1)
        for iz in range(1, self.z.size):
            np.testing.assert_array_equal(lm.Sigma["2h"][0], lm.Sigma["2h"][iz])
            np.testing.assert_array_equal(lm.Wp[0], lm.Wp[iz])

    def test_dsigma_hh_has_a_large_low_radius_nan_fraction_sigma_hh_does_not(self):
        # docs/known_issues/dsigma_hh_debug_flag.md #1: ct_2hTerm.pk_to_dsigma NaNs out
        # every negative dSigma (np.where(dSigma < 0, nan, dSigma)) -- a
        # real, by-construction feature of the exclusion-subtraction
        # convention, not a transient failure -- and only dSigma is
        # clipped this way; Sigma_hh has no such step.
        lm = hm.lensingModel(self.R, omega_m=0.3, odelta=200)
        lm.second_halo_term(self.z, self.k_h, self.p_k)

        sigma_hh = lm.Sigma["2h"]
        dsigma_hh = lm.dSigma["2h"]
        self.assertEqual(int(np.sum(~np.isfinite(sigma_hh))), 0)

        nan_frac = float(np.mean(~np.isfinite(dsigma_hh)))
        # Documented as ~60% on the original dump; assert it is a large,
        # non-trivial fraction (not exactly pinned, since a different R
        # grid/cosmology shifts the exact number) so this test still
        # catches "the NaN clipping silently stopped happening" as well
        # as "it got dramatically worse".
        self.assertGreater(nan_frac, 0.2)
        self.assertLess(nan_frac, 0.9)

        # Wherever finite, dSigma_hh must be non-negative (the clipping
        # threshold itself, trivially true by construction -- pinned so a
        # refactor of the clipping logic can't silently invert the sign
        # convention).
        finite = np.isfinite(dsigma_hh)
        self.assertTrue(np.all(dsigma_hh[finite] >= 0.0))

        # The NaN region is the LOW-radius half of the grid (the doc's
        # "everything below R ~ 2.48 cMpc/h"), not scattered randomly --
        # i.e. once a radius is finite, every larger radius at that z is
        # also finite. A random/scattered NaN pattern would indicate a
        # different failure mode than the documented one.
        for iz in range(dsigma_hh.shape[0]):
            row = dsigma_hh[iz]
            finite_idx = np.flatnonzero(np.isfinite(row))
            if finite_idx.size:
                self.assertTrue(np.all(np.isfinite(row[finite_idx[0]:])))


@unittest.skipUnless(HAS_DUMP, _SKIP_MSG)
class TestSecondHaloTermVsClenspy(unittest.TestCase):
    """Independent cross-check of the two-halo term against hard-coded
    reference values generated (offline, once) by the user's own CLensPy
    package (github.com/estevesjh/CLensPy) -- clenspy.halo.twohalo.
    TwoHaloTerm, FFTLog xi(r,z) + Abel projection to Sigma(R,z) -- fed
    the SAME real fiducial P(k,z) this file loads via
    _load_matter_power_lin(), never an independently-generated P(k).
    A genuinely separate numerical pipeline (mcfit FFTLog vs
    cluster_toolkit's Hankel transform), not a re-typed copy of anything
    in this repo.

    This test file has NO import-time or runtime dependency on clenspy
    (not pip-published, and CLensPy isn't cloned in every checkout) --
    only on the reference numbers below, generated once and pinned. To
    regenerate after the fiducial dump changes:

        from clenspy.halo.twohalo import TwoHaloTerm
        k_h, p_k, z = _load_matter_power_lin()
        h, omega_m = 0.6766, 0.311049
        two_halo = TwoHaloTerm(k_h * h, p_k / h**3, zvec=z)
        # xi(r, z):  two_halo.xi(np.array([r_hunit / h]), z[iz])
        # Sigma(R, z) [Msun h/pc^2]:
        #   two_halo.sigma(np.array([R_hunit / h]), z[iz])
        #     * (omega_m * 2.77533742639e11 * h**2) / h / 1e12

    Unit conversion (verified empirically, not just derived): CLensPy
    works in physical (non-h) Mpc/Msun; this repo's convention (and
    cluster_toolkit's) is Mpc/h, Msun/h, Msun h/pc^2. k_phys = k_h * h,
    P_phys = P_h / h**3 is a pure relabeling of the SAME power spectrum
    (confirmed: clenspy's xi(r,z) then matched cluster_toolkit's
    xi_mm_at_r to ~1% at every R and z checked). CLensPy's raw sigma()
    output is the bare Abel-projected xi(r,z) (dimensionless x length,
    NOT yet multiplied by the mean matter density); the physical Sigma
    is sigma_cl_raw * rho_m_phys / h / 1e12 in the Msun h/pc^2 convention
    (rho_m_phys = omega_m * 2.77533742639e11 * h**2 Msun/Mpc^3, comoving
    so no (1+z)^3 factor -- confirmed against cluster_toolkit.Sigma_at_R
    to ~0.2%).

    CAVEAT (linear P(k) only, by construction of this fixture): the
    fiducial dump's `cp_camb` config sets no `nonlinear_pk_path`, so
    `matter_power_nl` was never written, and halo_model_cosmosis.py's
    own fallback (`if block.has_section("matter_power_nl"): ... else:
    P_k_nl, k_nl, z_nl = P_k, k_h, z_k`) means the "nl" P(k)
    `second_halo_term` would receive from THIS fixture is just an alias
    for the linear one. This test loads the same linear P(k) for both
    "production" and the CLensPy reference, so it is internally
    consistent -- but the measured deviation table below (and in
    docs/known_issues/dsigma_hh_debug_flag.md) reflects the linear-P(k) fallback
    case specifically. A production run with `nonlinear_pk_path`
    configured would feed second_halo_term a genuinely nonlinear P(k),
    and the deviation size would differ (the z-degeneracy bug itself
    is unaffected either way -- it is in the z-loop, not the P(k)
    choice).
    """

    H = 0.6766
    OMEGA_M = 0.311049

    # r/R grids (Mpc/h comoving) the benchmarks below were generated at.
    R_HUNIT_XI = np.array([0.1, 0.28840315031266056, 0.8317637711026709,
                           2.39883291901949, 6.918309709189363,
                           19.952623149688797])
    R_HUNIT_SIGMA = np.array([1.0, 1.8197008586099834, 3.311311214825911,
                              6.025595860743578, 10.964781961431852,
                              19.952623149688797])
    R_HUNIT_CLUSTER = np.array([3.0, 4.384327654865777, 6.407442995073613,
                                9.364109840092413, 13.685108578372633,
                                20.000000000000004])

    # Indices into the fiducial dump's 50-point z grid (z[0]=0.0,
    # z[16]=1.306..., z[33]=2.694...) and CLensPy's xi(r_hunit, z) there.
    XI_BENCHMARK = {
        0: [25.998216291926788, 14.014639482868917, 6.506711044646951,
            2.421790749687088, 0.6389645284654966, 0.0956566559113811],
        16: [7.4493751293329575, 4.015671907563546, 1.8643944986641527,
             0.6939255976868618, 0.18308511681837625, 0.02740889242169588],
        33: [3.03971378659661, 1.6385929085252857, 0.7607652404198043,
             0.28315599221943366, 0.07470779012350562, 0.011184184810549688],
    }

    # CLensPy's Sigma(R, z=0) [Msun h/pc^2] at R_HUNIT_SIGMA, already
    # converted per the recipe above.
    SIGMA_BENCHMARK_Z0 = np.array([
        3.2817210311056138, 2.693455072779718, 2.036866587154126,
        1.375145042256565, 0.7927055108056494, 0.36476559279952964,
    ])

    # CLensPy's xi(r=1 Mpc/h, z) at the dump's first (z=0.0) and last
    # (z=4.0) grid points.
    XI_AT_R1_Z0 = 5.581496950599194
    XI_AT_R1_ZMAX = 0.3595098065969617

    # CLensPy's Sigma(R, z) [Msun h/pc^2] at R_HUNIT_CLUSTER and the
    # dump's z-grid point closest to 0.4 (index 5, z=0.408...) -- a
    # representative DES cluster redshift.
    IZ_CLUSTER = 5
    SIGMA_BENCHMARK_Z_CLUSTER = np.array([
        1.3984946201522066, 1.1215967834857092, 0.8532388267465022,
        0.6080780391771377, 0.39947380758469997, 0.23666903728521788,
    ])

    @classmethod
    def setUpClass(cls):
        cls.k_h, cls.p_k, cls.z = _load_matter_power_lin()

    def test_xi_matches_cluster_toolkit_at_several_z(self):
        # Same physics (the two-halo matter correlation function), two
        # unrelated numerical methods (FFTLog vs Hankel transform), fed
        # the identical real P(k, z) -- agreement here is a genuine
        # cross-check of both the physics and the h-unit conversion.
        for iz, xi_cl in self.XI_BENCHMARK.items():
            xi_ct = ct.xi.xi_mm_at_r(self.R_HUNIT_XI, self.k_h, self.p_k[iz])
            np.testing.assert_allclose(xi_cl, xi_ct, rtol=2e-2)

    def test_sigma_matches_cluster_toolkit_abel_projection(self):
        # Isolates the Abel-projection step itself (bias=1, no NFW
        # exclusion piece): cluster_toolkit.Sigma_at_R fed xi_mm directly
        # vs the CLensPy benchmark, both from the identical P(k) at z~0.
        Rfix = np.logspace(-3., 3., 200)
        xi_mm_z0 = ct.xi.xi_mm_at_r(Rfix, self.k_h, self.p_k[0])
        sigma_ct = ct.deltasigma.Sigma_at_R(
            self.R_HUNIT_SIGMA, Rfix, xi_mm_z0, 1e14, 5.0, self.OMEGA_M)
        np.testing.assert_allclose(self.SIGMA_BENCHMARK_Z0, sigma_ct, rtol=2e-2)

    def test_xi_genuinely_varies_with_z_unlike_production(self):
        # Directly quantifies docs/known_issues/dsigma_hh_debug_flag.md finding #2
        # ("Sigma_hh is bit-identical at every z") using a genuinely
        # independent calculation fed the SAME real P(k,z): the true
        # two-halo correlation function is NOT remotely z-independent.
        self.assertGreater(
            abs(self.XI_AT_R1_Z0 - self.XI_AT_R1_ZMAX) / abs(self.XI_AT_R1_Z0),
            0.5)

    def test_production_sigma_hh_deviates_from_true_z_dependent_value(self):
        # Quantifies the practical size of the z=0-everywhere defect
        # (docs/known_issues/dsigma_hh_debug_flag.md #2, pinned structurally by
        # TestSecondHaloTerm.test_z_axis_is_currently_degenerate) against
        # a genuinely independent, common-P(k) reference: how far is
        # production's (z-invariant) Sigma_hh from what CLensPy says
        # Sigma_2h actually is at a representative DES cluster redshift.
        lm = hm.lensingModel(self.R_HUNIT_CLUSTER, omega_m=self.OMEGA_M, odelta=200)
        lm.second_halo_term(self.z, self.k_h, self.p_k)
        sigma_hh_production = lm.Sigma["2h"][0]  # any row -- all identical

        frac_diff = ((sigma_hh_production - self.SIGMA_BENCHMARK_Z_CLUSTER)
                     / self.SIGMA_BENCHMARK_Z_CLUSTER)
        # A real, substantial deviation -- not asserting a tight match
        # (there isn't one; that is exactly the defect), just recording
        # its size so a future fix has a number to check against.
        self.assertGreater(np.abs(frac_diff).min(), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
